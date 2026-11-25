"""ニューラルネット（PyTorch + skorch）。

skorchでPyTorchモデルをsklearn互換estimatorにし、他モデル（木・線形）と同じ `ModelSpec` として扱う。
torch・skorchは任意依存（`uv sync --extra nn`）で、未インストールの環境では
このモジュールのimportに失敗し、`modeling.models` はNNモデルを登録せずに動作する。

- `mlp`: 全結合ネットワーク（ベースライン）
- `tabm`: TabM（Gorishniy et al., ICLR 2025）。k 個の MLP を BatchEnsemble で重み共有して
  同時に学習し、予測は k 個の平均にする。小さい表データ（数千行）で分散を抑えやすく、
  CPU でも MLP の数倍の計算で済む。
  数値特徴量の periodic 埋め込み（PLR）を付けられる。
- `cnn1d`: 1次元畳み込みネットワーク。特徴量の**並び順を系列**とみなして畳み込むため、
  ラグ特徴量のように順序に意味がある特徴量（例: `x_lag_1, x_lag_2, ...`）で使う。

いずれも「欠損値補完（中央値）→ スケーリング → ネットワーク」の `NetPipeline` として構築する。
木のモデルとそろえるため、次の点を共通に扱う。

- early stopping: 学習ループ（`modeling.trainer.fit_pipeline`）が渡す検証データ
  （CV の検証 fold）で行う（LightGBM などと同じ）。
  検証データが無いときだけ、学習データの 10% を切り出して使う。
- `sample_weight`: 行ごとの重みを損失に掛ける（検証の損失には重みを付けない）。
- `n_estimators`: エポック数の上限（`max_epochs`）として読み替える。
- 目的関数: `objective_params` で l2・l1・huber・quantile・poisson に対応する。
- 目的変数の標準化: 回帰では学習側の平均・標準偏差で標準化して学習し、予測で元の尺度に戻す
  （huber の閾値は元の尺度で指定し、内部で標準偏差で割る）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, Self

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.dataset import Dataset as SkorchDataset
from skorch.dataset import ValidSplit
from skorch.helper import predefined_split
from skorch.utils import to_tensor
from torch import nn

from evaluation.curves import TrainingHistory
from modeling.models.base import (
    CategoricalEncoding,
    ModelSpec,
    check_objective,
    merge_params,
    objective_alpha,
    register_model,
)
from modeling.tasks import Task, is_classification

_ALL_TASKS = frozenset(Task)

ScalerKind = Literal["standard", "quantile"]


# ---------------------------------------------------------------------------
# ネットワーク（torch.nn.Module）
# ---------------------------------------------------------------------------


class MLPModule(nn.Module):
    """全結合ネットワーク。

    Args:
        n_features: 入力次元（fit時にデータから設定される）。
        n_outputs: 出力次元（回帰は1、分類はクラス数。fit時に設定される）。
        hidden_sizes: 隠れ層のユニット数の並び。
        dropout: ドロップアウト率。
    """

    def __init__(
        self,
        n_features: int = 1,
        n_outputs: int = 1,
        hidden_sizes: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = n_features
        for size in hidden_sizes:
            layers += [nn.Linear(in_size, size), nn.ReLU(), nn.Dropout(dropout)]
            in_size = size
        layers.append(nn.Linear(in_size, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """順伝播。"""
        out: torch.Tensor = self.net(X)
        return out


class Conv1dModule(nn.Module):
    """特徴量の並びを長さ `n_features` の系列とみなす1次元CNN。

    Args:
        n_features: 入力次元（系列長。fit時にデータから設定される）。
        n_outputs: 出力次元（fit時に設定される）。
        channels: 各畳み込み層のチャネル数。
        kernel_size: 畳み込みのカーネルサイズ（系列長を保つようpaddingする）。
        hidden_size: 畳み込み後の全結合層のユニット数。
        dropout: ドロップアウト率。
    """

    def __init__(
        self,
        n_features: int = 1,
        n_outputs: int = 1,
        channels: Sequence[int] = (16, 32),
        kernel_size: int = 3,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        convs: list[nn.Module] = []
        in_channels = 1
        for c in channels:
            convs += [nn.Conv1d(in_channels, c, kernel_size, padding=kernel_size // 2), nn.ReLU()]
            in_channels = c
        self.conv = nn.Sequential(*convs)
        # paddingで系列長を保つため、平坦化後の次元は 最終チャネル数 × 系列長
        conv_len = n_features + len(channels) * (2 * (kernel_size // 2) - kernel_size + 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * conv_len, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_outputs),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """順伝播（(batch, n_features) → (batch, 1, n_features) にしてから畳み込む）。"""
        out: torch.Tensor = self.head(self.conv(X.unsqueeze(1)))
        return out


class PeriodicEmbeddings(nn.Module):
    """数値特徴量の periodic 埋め込み（PLR: Periodic → Linear → ReLU）。

    特徴量 `x_j` ごとに `[cos(2π c_j x_j), sin(2π c_j x_j)]`（`c_j` は学習する周波数）を作り、
    特徴量ごとの全結合層で `d_embedding` 次元にする（Gorishniy et al., 2022）。

    Args:
        n_features: 特徴量の数。
        d_embedding: 1特徴量あたりの埋め込みの次元。
        n_frequencies: 1特徴量あたりの周波数の数。
        frequency_init_scale: 周波数の初期値の標準偏差（大きいほど細かい変化を表しやすいが、
            過学習しやすい）。
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 16,
        n_frequencies: int = 16,
        frequency_init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.randn(n_features, n_frequencies) * frequency_init_scale
        )
        bound = 1 / math.sqrt(2 * n_frequencies)
        self.weight = nn.Parameter(
            torch.empty(n_features, 2 * n_frequencies, d_embedding).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding).uniform_(-bound, bound))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(batch, n_features) → (batch, n_features × d_embedding)。"""
        z = 2 * math.pi * self.frequencies[None] * X[..., None]
        z = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
        # 特徴量ごとに別の全結合層をかける
        out = torch.relu(torch.einsum("bfi,fid->bfd", z, self.weight) + self.bias)
        return out.flatten(1)


class EnsembleLinear(nn.Module):
    """BatchEnsemble の全結合層。

    重み `W` を k 個のメンバーで共有し、メンバーごとの `r`・`s`・bias を持つ。

    出力は `((x * r_i) @ W) * s_i + b_i`（i はメンバー）。`r`・`s` は 1 で初期化し、
    `random_signs=True` なら `r` を ±1 の乱数で初期化する
    （最初の層でメンバーを違うものにするため）。

    Args:
        in_features: 入力次元。
        out_features: 出力次元。
        k: メンバーの数。
        random_signs: `r` を ±1 の乱数で初期化するか。
    """

    def __init__(self, in_features: int, out_features: int, k: int, random_signs: bool) -> None:
        super().__init__()
        bound = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(in_features, out_features).uniform_(-bound, bound))
        r = torch.ones(k, in_features)
        if random_signs:
            r = torch.where(torch.rand(k, in_features) < 0.5, -1.0, 1.0)
        self.r = nn.Parameter(r)
        self.s = nn.Parameter(torch.ones(k, out_features))
        # bias は全メンバーで同じ値から始める
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound).repeat(k, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(batch, k, in_features) → (batch, k, out_features)。"""
        return (X * self.r) @ self.weight * self.s + self.bias


class TabMModule(nn.Module):
    """TabM（Gorishniy et al., ICLR 2025）。k 個の MLP を重み共有して同時に学習するアンサンブル。

    出力は常に `(batch, k, n_outputs)`（k 個のメンバーそれぞれの予測）。学習ではメンバーごとの損失の
    平均を最小化し、評価・予測では `TabularNet*` がメンバーの予測を平均する。

    Args:
        n_features: 入力次元（fit時にデータから設定される）。
        n_outputs: 出力次元（fit時に設定される）。
        k: アンサンブルのメンバーの数。
        n_blocks: 隠れ層の数。
        d_block: 隠れ層のユニット数。
        dropout: ドロップアウト率。
        arch_type: `"tabm"`（全ての隠れ層が BatchEnsemble）/ `"tabm-mini"`（入力に ±1 の
            係数を掛けるだけで、隠れ層は全メンバー共通。出力層だけメンバーごと）。
        num_embeddings: `"periodic"`（数値特徴量の periodic 埋め込み）/ `None`（埋め込みなし）。
        d_embedding: periodic 埋め込みの1特徴量あたりの次元。
        n_frequencies: periodic 埋め込みの周波数の数。
        frequency_init_scale: periodic 埋め込みの周波数の初期値の標準偏差。
    """

    def __init__(
        self,
        n_features: int = 1,
        n_outputs: int = 1,
        k: int = 16,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        arch_type: Literal["tabm", "tabm-mini"] = "tabm",
        num_embeddings: Literal["periodic"] | None = "periodic",
        d_embedding: int = 16,
        n_frequencies: int = 16,
        frequency_init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if arch_type not in ("tabm", "tabm-mini"):
            raise ValueError(f"arch_type は tabm か tabm-mini です: {arch_type}")
        if num_embeddings not in (None, "periodic"):
            raise ValueError(f"num_embeddings は None か periodic です: {num_embeddings}")
        self.k = k
        self.arch_type = arch_type
        self.embeddings: nn.Module | None = None
        d_in = n_features
        if num_embeddings == "periodic":
            self.embeddings = PeriodicEmbeddings(
                n_features, d_embedding, n_frequencies, frequency_init_scale
            )
            d_in = n_features * d_embedding
        layers: list[nn.Module] = []
        if arch_type == "tabm-mini":
            # 入力にメンバーごとの ±1 の係数を掛ける（以降の隠れ層は全メンバー共通）
            self.adapter = nn.Parameter(torch.where(torch.rand(k, d_in) < 0.5, -1.0, 1.0))
        for i in range(n_blocks):
            size_in = d_in if i == 0 else d_block
            if arch_type == "tabm":
                layers.append(EnsembleLinear(size_in, d_block, k, random_signs=i == 0))
            else:
                layers.append(nn.Linear(size_in, d_block))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.blocks = nn.Sequential(*layers)
        d_last = d_block if n_blocks > 0 else d_in
        bound = 1 / math.sqrt(d_last)
        # 出力層はメンバーごとに別の重み
        self.head_weight = nn.Parameter(torch.empty(k, d_last, n_outputs).uniform_(-bound, bound))
        self.head_bias = nn.Parameter(torch.empty(k, n_outputs).uniform_(-bound, bound))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(batch, n_features) → (batch, k, n_outputs)。"""
        if self.embeddings is not None:
            X = self.embeddings(X)
        z = X.unsqueeze(1).expand(-1, self.k, -1)
        if self.arch_type == "tabm-mini":
            z = z * self.adapter
        z = self.blocks(z)
        return torch.einsum("bkd,kdo->bko", z, self.head_weight) + self.head_bias


# ---------------------------------------------------------------------------
# 損失
# ---------------------------------------------------------------------------


class RegressionLoss(nn.Module):
    """回帰の損失（行・要素ごとの値を返す。平均は `TabularNetRegressor.get_loss` で取る）。

    Args:
        objective: 目的関数の共通の名前（`modeling.models.base.OBJECTIVES`）。
        alpha: huber の閾値（モデルの尺度）・quantile の分位。それ以外では使わない。
    """

    def __init__(self, objective: str = "l2", alpha: float | None = None) -> None:
        super().__init__()
        check_objective(objective)
        self.objective = objective
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """要素ごとの損失。poisson では `y_pred` を予測の対数とみなす。"""
        diff = y_true - y_pred
        if self.objective == "l2":
            return diff**2
        if self.objective == "l1":
            return diff.abs()
        if self.objective == "huber":
            delta = 1.0 if self.alpha is None else self.alpha
            return nn.functional.huber_loss(y_pred, y_true, reduction="none", delta=delta)
        if self.objective == "quantile":
            q = 0.5 if self.alpha is None else self.alpha
            return torch.maximum(q * diff, (q - 1) * diff)
        # poisson: 予測の対数 y_pred に対する負の対数尤度（定数項を除く）
        return torch.exp(y_pred) - y_true * y_pred


# ---------------------------------------------------------------------------
# 前処理と Pipeline
# ---------------------------------------------------------------------------


class NetInputScaler(BaseEstimator, TransformerMixin):
    """NN の入力のスケーリング。

    値が2種類以下の列（0/1 のフラグ・ワンホット）は標準化し、それ以外の列を `method` で変換する。
    `"quantile"` は分位点で正規分布に写す（`QuantileTransformer`。外れ値・裾の重い分布に強い。
    TabM 論文の前処理）。フラグに分位点の変換をかけると ±5 付近の極端な値になるため、
    標準化にしている。

    Args:
        method: `"standard"`（標準化）/ `"quantile"`（分位点で正規分布に写す）。
        n_quantiles: 分位点の数の上限（学習の行数を超える場合は行数にする）。
        random_state: `QuantileTransformer` の乱数シード。
    """

    def __init__(
        self, method: ScalerKind = "standard", n_quantiles: int = 1000, random_state: int = 0
    ) -> None:
        self.method = method
        self.n_quantiles = n_quantiles
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> Self:
        """列ごとの変換を学習する。

        Raises:
            ValueError: 未対応の `method` の場合。
        """
        if self.method not in ("standard", "quantile"):
            raise ValueError(f"scaler は standard か quantile です: {self.method}")
        values = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = values.shape[1]
        if self.method == "standard":
            self.binary_ = np.ones(values.shape[1], dtype=bool)
        else:
            self.binary_ = np.array(
                [np.unique(values[:, j]).size <= 2 for j in range(values.shape[1])], dtype=bool
            )
        self.standard_ = (
            StandardScaler().fit(values[:, self.binary_]) if self.binary_.any() else None
        )
        self.quantile_ = None
        if (~self.binary_).any():
            self.quantile_ = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, len(values)),
                output_distribution="normal",
                subsample=10**9,
                random_state=self.random_state,
            ).fit(values[:, ~self.binary_])
        return self

    def transform(self, X: Any) -> np.ndarray:
        """列ごとに変換した配列を返す。"""
        values = np.asarray(X, dtype=np.float64)
        out = np.empty_like(values)
        if self.standard_ is not None:
            out[:, self.binary_] = self.standard_.transform(values[:, self.binary_])
        if self.quantile_ is not None:
            out[:, ~self.binary_] = self.quantile_.transform(values[:, ~self.binary_])
        return out


class NetPipeline(Pipeline):
    """前処理 + NN の Pipeline。`sample_weight` と early stopping 用の検証データを NN に渡す。

    `fit(X, y, sample_weight=..., X_valid=..., y_valid=...)` で、前処理を学習データだけで学習し、
    検証データも同じ前処理で変換してから最後のステップ（NN）に渡す。
    """

    def fit(
        self,
        X: Any,
        y: Any = None,
        sample_weight: Any = None,
        X_valid: Any = None,
        y_valid: Any = None,
    ) -> Self:
        """前処理を学習し、NN を学習する。"""
        preprocess = self[:-1]
        Xt = preprocess.fit_transform(X, y)
        Xv = None if X_valid is None else preprocess.transform(X_valid)
        self.steps[-1][1].fit(Xt, y, sample_weight=sample_weight, X_valid=Xv, y_valid=y_valid)
        return self


# ---------------------------------------------------------------------------
# skorch の estimator
# ---------------------------------------------------------------------------


def _as_float32(X: Any) -> Any:
    """入力をPyTorchが扱えるfloat32のnumpy配列にする。

    `Dataset`・dict（重み付きの学習データ）は変換せずそのまま返す。
    """
    if isinstance(X, torch.utils.data.Dataset | Mapping):
        return X
    return np.asarray(X, dtype=np.float32)


class _TabularNetMixin:
    """回帰・分類の skorch estimator に共通の処理（重み・検証データ・アンサンブルの出力）。

    - 重みがあれば、学習データを `{"X": X, "sample_weight": w}` の dict にして渡し、
      `get_loss` で使う。
    - 検証データがあれば、`train_split` を一時的に `predefined_split` にして early stopping に使う。
    - 出力が `(batch, k, n_outputs)`（TabM）の場合、学習ではメンバーごとの損失の平均を取り、
      評価・予測ではメンバーの予測を平均する（`_reduce_members`）。
    """

    # skorch の NeuralNet の属性（型チェック用の宣言）
    module_: nn.Module
    criterion_: nn.Module
    callbacks: Any
    train_split: Any
    history: Any
    device: Any

    def _fit_with_valid(
        self, X32: np.ndarray, y: np.ndarray, sample_weight: Any, X_valid: Any, y_valid: Any
    ) -> None:
        """重み・検証データを skorch に渡して学習する（`train_split` は学習の後に元に戻す）。"""
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=np.float32).ravel()
            if len(weight) != len(X32):
                raise ValueError(f"sample_weight の長さ {len(weight)} が行数 {len(X32)} と違います")
            X_fit: Any = {"X": X32, "sample_weight": weight}
        else:
            X_fit = X32
        callbacks = self.callbacks or []
        self.early_stopping_ = any(
            isinstance(cb[1] if isinstance(cb, tuple) else cb, EarlyStopping) for cb in callbacks
        )
        original = self.train_split
        if X_valid is not None:
            valid = SkorchDataset(np.asarray(X_valid, dtype=np.float32), y_valid)
            self.train_split = predefined_split(valid)
        self.has_valid_ = self.train_split is not None
        try:
            super().fit(X_fit, y)  # type: ignore[misc]
        finally:
            # 検証データへの参照を残さない（モデルの保存・メモリのため）
            self.train_split = original

    def infer(self, x: Any, **fit_params: Any) -> torch.Tensor:
        """dict の入力から特徴量を取り出して順伝播し、評価時はアンサンブルの予測を平均する。"""
        if isinstance(x, Mapping):
            x = x["X"]
        x = to_tensor(x, device=self.device)
        out: torch.Tensor = self.module_(x, **fit_params)
        if out.dim() == 3 and not self.module_.training:
            out = self._reduce_members(out)
        return out

    def _reduce_members(self, out: torch.Tensor) -> torch.Tensor:
        """(batch, k, n_outputs) のメンバーの予測を (batch, n_outputs) にまとめる。"""
        raise NotImplementedError

    def _row_losses(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """行ごとの損失（メンバーがあればメンバーの平均）。"""
        raise NotImplementedError

    def get_loss(
        self, y_pred: torch.Tensor, y_true: Any, X: Any = None, training: bool = False
    ) -> torch.Tensor:
        """行ごとの損失の平均（学習で重みがあれば重み付き平均）。"""
        y_true = to_tensor(y_true, device=self.device)
        losses = self._row_losses(y_pred, y_true)
        if training and isinstance(X, Mapping) and "sample_weight" in X:
            weight = to_tensor(X["sample_weight"], device=self.device).reshape(-1)
            return (losses * weight).sum() / weight.sum().clamp_min(1e-12)
        return losses.mean()


class TabularNetRegressor(_TabularNetMixin, NeuralNetRegressor):
    """表形式データ用のskorch回帰器。

    入力次元の自動設定・乱数シード固定・1次元出力のほか、目的関数・目的変数の標準化・
    重み・検証データ（early stopping）に対応する。

    Args:
        *args: `NeuralNetRegressor` の位置引数。
        objective: 目的関数の共通の名前（l2・l1・huber・quantile・poisson）。
        objective_alpha: huber の閾値（元の尺度）・quantile の分位。
        standardize_target: 目的変数を学習側の平均・標準偏差で標準化して学習するか
            （poisson では行わない）。
        random_state: fit時に設定するPyTorchの乱数シード。
        **kwargs: `NeuralNetRegressor` のキーワード引数。
    """

    def __init__(
        self,
        *args: Any,
        objective: str = "l2",
        objective_alpha: float | None = None,
        standardize_target: bool = True,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("criterion", RegressionLoss)
        super().__init__(*args, **kwargs)
        self.objective = objective
        self.objective_alpha = objective_alpha
        self.standardize_target = standardize_target
        self.random_state = random_state

    def fit(
        self,
        X: Any,
        y: Any = None,
        sample_weight: Any = None,
        X_valid: Any = None,
        y_valid: Any = None,
    ) -> Self:
        """入力次元・目的関数を設定して学習する。

        Raises:
            ValueError: 未知の目的関数、または poisson で負の目的変数がある場合。
        """
        check_objective(self.objective)
        X32 = _as_float32(X)
        y64 = np.asarray(y, dtype=np.float64).ravel()
        if self.objective == "poisson" and (y64 < 0).any():
            raise ValueError("poisson は負の目的変数に使えません")
        # 目的変数の標準化（poisson は予測の対数を学習するので行わない）
        self.y_mean_, self.y_scale_ = 0.0, 1.0
        if self.standardize_target and self.objective != "poisson":
            self.y_mean_ = float(np.mean(y64))
            std = float(np.std(y64))
            self.y_scale_ = std if std > 0 else 1.0
        alpha = None
        if self.objective in ("huber", "quantile"):
            alpha = objective_alpha(self.objective, self.objective_alpha)
            if self.objective == "huber":
                # 閾値は元の尺度で指定されるので、標準化した尺度に直す
                alpha = alpha / self.y_scale_
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.set_params(
            module__n_features=X32.shape[1],
            module__n_outputs=1,
            criterion__objective=self.objective,
            criterion__alpha=alpha,
        )
        y_fit = self._to_model_scale(y64)
        y_valid_fit = None if y_valid is None else self._to_model_scale(y_valid)
        self._fit_with_valid(X32, y_fit, sample_weight, X_valid, y_valid_fit)
        return self

    def _to_model_scale(self, y: Any) -> np.ndarray:
        """目的変数を学習する尺度（標準化後）の (n, 1) の float32 にする。"""
        y64 = np.asarray(y, dtype=np.float64).ravel()
        return ((y64 - self.y_mean_) / self.y_scale_).astype(np.float32).reshape(-1, 1)

    def _reduce_members(self, out: torch.Tensor) -> torch.Tensor:
        """メンバーの予測の平均（poisson では予測の対数の平均）。"""
        return out.mean(dim=1)

    def _row_losses(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """行ごとの損失。"""
        y_true = y_true.reshape(len(y_true), -1)
        if y_pred.dim() == 3:
            # (batch, k, 1) と比べられるよう、目的変数をメンバーの数だけ並べる
            y_true = y_true.unsqueeze(1).expand_as(y_pred)
        losses = self.criterion_(y_pred, y_true)
        return losses.reshape(len(losses), -1).mean(dim=1)

    def predict(self, X: Any) -> np.ndarray:
        """予測値を元の尺度の1次元配列で返す。"""
        z = np.asarray(super().predict(_as_float32(X)), dtype=np.float64).ravel()
        if self.objective == "poisson":
            return np.exp(z)
        return z * self.y_scale_ + self.y_mean_


class TabularNetClassifier(_TabularNetMixin, NeuralNetClassifier):
    """表形式データ用のskorch分類器（入力次元・クラス数の自動設定・乱数シード固定・重み・検証データ）。

    Args:
        *args: `NeuralNetClassifier` の位置引数。
        random_state: fit時に設定するPyTorchの乱数シード。
        **kwargs: `NeuralNetClassifier` のキーワード引数。
    """

    def __init__(self, *args: Any, random_state: int | None = None, **kwargs: Any) -> None:
        # モジュールはlogitsを出力する。CrossEntropyLossを指定すると、skorchは
        # predict_probaでsoftmaxを自動適用する（既定のNLLLossは確率出力が前提）
        kwargs.setdefault("criterion", nn.CrossEntropyLoss)
        kwargs.setdefault("criterion__reduction", "none")
        super().__init__(*args, **kwargs)
        self.random_state = random_state

    def fit(
        self,
        X: Any,
        y: Any = None,
        sample_weight: Any = None,
        X_valid: Any = None,
        y_valid: Any = None,
    ) -> Self:
        """入力次元・クラス数を設定して学習する。"""
        X32 = _as_float32(X)
        y64 = np.asarray(y, dtype=np.int64)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        # 出力数は最大ラベル+1とし、classes も 0..最大ラベル に固定する
        # （学習foldに無いクラスがあってもラベル番号と出力列の位置を一致させるため）
        n_outputs = int(y64.max()) + 1
        self.set_params(
            module__n_features=X32.shape[1],
            module__n_outputs=n_outputs,
            classes=np.arange(n_outputs),
        )
        y_valid64 = None if y_valid is None else np.asarray(y_valid, dtype=np.int64)
        self._fit_with_valid(X32, y64, sample_weight, X_valid, y_valid64)
        return self

    def _reduce_members(self, out: torch.Tensor) -> torch.Tensor:
        """メンバーの確率の平均の対数（softmax をかけると平均の確率に戻る）。"""
        return torch.log(torch.softmax(out, dim=-1).mean(dim=1).clamp_min(1e-12))

    def _row_losses(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """行ごとの交差エントロピー（メンバーがあればメンバーの平均）。"""
        y_true = y_true.reshape(-1).long()
        if y_pred.dim() == 3:
            n, k, c = y_pred.shape
            losses = self.criterion_(y_pred.reshape(n * k, c), y_true.repeat_interleave(k))
            return losses.reshape(n, k).mean(dim=1)
        return self.criterion_(y_pred, y_true)

    def predict_proba(self, X: Any) -> np.ndarray:
        """クラス確率を返す。"""
        return np.asarray(super().predict_proba(_as_float32(X)))

    def predict(self, X: Any) -> np.ndarray:
        """予測クラスを返す。"""
        return np.asarray(super().predict(_as_float32(X)))


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


class _NeuralNetSpec(ModelSpec):
    """skorchモデル共通の `ModelSpec`。

    Attributes:
        module_cls: ネットワークのクラス。
        default_params: skorch の既定のパラメータ（共通の既定を上書きする）。
        default_scaler: 入力のスケーリングの既定（`NetInputScaler` の `method`）。
    """

    supported_tasks = _ALL_TASKS
    requires_float32 = True
    # permutation 法の SHAP は1行あたり 5万行の予測が要るので、CPU では行数を抑える
    shap_max_samples = 100
    # 整数化したカテゴリの大小に意味を持たせないよう、ワンホットにする
    categorical_encoding: ClassVar[CategoricalEncoding] = "onehot"
    module_cls: type[nn.Module]
    default_params: ClassVar[dict[str, Any]] = {}
    default_scaler: ClassVar[ScalerKind] = "standard"

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """欠損補完 + スケーリング + skorchネットワークの `NetPipeline` を作る。

        `params` の `scaler`（`"standard"` / `"quantile"`）は前処理に使い、`n_estimators` は
        `max_epochs` に読み替える（木のモデルと同じ設定でエポック数の上限を渡せるように）。
        """
        params = dict(params)
        scaler = params.pop("scaler", self.default_scaler)
        n_estimators = params.pop("n_estimators", None)
        if n_estimators is not None:
            params.setdefault("max_epochs", n_estimators)
        defaults: dict[str, Any] = {
            "max_epochs": 100,
            "lr": 1e-3,
            "batch_size": 256,
            # 予測（検証・SHAP）は大きいバッチでまとめて行う（学習には影響しない）
            "iterator_valid__batch_size": 4096,
            "optimizer": torch.optim.AdamW,
            "verbose": 0,
            "random_state": seed,
            "train_split": None,
            **self.default_params,
        }
        if early_stopping_rounds is not None:
            # 検証データ（CV の検証 fold）が渡されればそれを使う。
            # 渡されなければ学習データの 10% を使う
            defaults["train_split"] = ValidSplit(
                0.1, stratified=is_classification(task), random_state=seed
            )
            defaults["callbacks"] = [EarlyStopping(patience=early_stopping_rounds, load_best=True)]
        merged = merge_params(defaults, params)
        if is_classification(task):
            # 分類には回帰の目的関数を渡さない
            for key in ("objective", "objective_alpha", "standardize_target"):
                merged.pop(key, None)
            net: BaseEstimator = TabularNetClassifier(module=self.module_cls, **merged)
        else:
            net = TabularNetRegressor(module=self.module_cls, **merged)
        return NetPipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", NetInputScaler(method=scaler, random_state=seed)),
                ("net", net),
            ]
        )

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """学習率・バッチサイズ・ドロップアウトの探索空間（ネットワーク形状はサブクラス）。"""
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "module__dropout": trial.suggest_float("module__dropout", 0.0, 0.5),
        }

    def fit_kwargs(
        self,
        X_valid: Any,
        y_valid: Any,
        early_stopping_rounds: int | None,
        *,
        X_train: Any = None,
        y_train: Any = None,
        record_history: bool = False,
    ) -> dict[str, Any]:
        """early stopping・学習の推移の記録用に、検証データを渡す（木のモデルと同じ）。"""
        if X_valid is None or (early_stopping_rounds is None and not record_history):
            return {}
        return {"X_valid": X_valid, "y_valid": y_valid}

    def best_iteration(self, estimator: Any) -> int | None:
        """early stoppingで最良だったエポック数（early stoppingなしならNone）。"""
        net = estimator[-1]
        if not getattr(net, "early_stopping_", False) or not getattr(net, "has_valid_", False):
            return None
        valid_losses = net.history[:, "valid_loss"]
        return int(np.argmin(valid_losses)) + 1

    def with_n_iterations(self, params: dict[str, Any], n_iterations: int) -> dict[str, Any]:
        """エポック数を固定したパラメータを返す（`n_estimators` は `max_epochs` に読み替える）。"""
        rest = {k: v for k, v in params.items() if k != "n_estimators"}
        return {**rest, "max_epochs": n_iterations}

    def objective_params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        """目的関数を `TabularNetRegressor` の `objective` に変換する。

        huber・quantile は `objective_alpha` も返す。

        huber の閾値は元の尺度で指定する（学習では目的変数の標準偏差で割る）。
        poisson はネットワークの出力を予測の対数とみなす（目的変数は標準化しない）。
        """
        check_objective(objective)
        if objective in ("huber", "quantile"):
            return {"objective": objective, "objective_alpha": objective_alpha(objective, alpha)}
        return {"objective": objective}

    def training_history(self, estimator: Any) -> TrainingHistory | None:
        """skorch の `history` から、エポックごとの学習・検証の損失を取り出す。

        損失はモデルの尺度（回帰は標準化した目的変数）。検証の損失は、検証データ（CV の検証 fold。
        無ければ学習データから切り出した 10%）のもので、重みは付けない。
        """
        net = estimator[-1]
        if not len(net.history):
            return None
        train = [float(v) for v in net.history[:, "train_loss"]]
        valid = None
        if getattr(net, "has_valid_", False):
            valid = [float(v) for v in net.history[:, "valid_loss"]]
        return TrainingHistory("loss", train, valid, self.best_iteration(estimator))


@register_model
class MLPSpec(_NeuralNetSpec):
    """全結合ネットワーク（`MLPModule`）。"""

    name = "mlp"
    module_cls = MLPModule

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """共通の探索空間に、層数・ユニット数を加える。"""
        width = trial.suggest_int("width", 32, 512, log=True)
        n_layers = trial.suggest_int("n_layers", 1, 4)
        return {**super().search_space(trial, task), "module__hidden_sizes": [width] * n_layers}


@register_model
class TabMSpec(_NeuralNetSpec):
    """TabM（`TabMModule`）。既定は k=16、3層×128、periodic 埋め込み 8次元。

    TabM 論文の既定（3層×256・埋め込み16次元）は CPU では1エポックが重いため、幅と埋め込みの
    次元を半分にしている（約800行・51特徴量で 1 fit 33秒 → 15秒。ダミーデータの検証 MAE は
    ほぼ同じ）。
    1,000〜数千行の表データ向けに、バッチサイズを小さめ（64）、重み減衰を入れ、
    入力は分位点の変換（`scaler="quantile"`）にする。
    """

    name = "tabm"
    module_cls = TabMModule
    default_params: ClassVar[dict[str, Any]] = {
        "max_epochs": 500,
        "lr": 2e-3,
        "optimizer__weight_decay": 3e-4,
        "batch_size": 64,
        "module__d_block": 128,
        "module__d_embedding": 8,
    }
    default_scaler: ClassVar[ScalerKind] = "quantile"

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """学習率・重み減衰・層の数と幅・ドロップアウト・periodic 埋め込みの探索空間（CPU 向け）。

        1試行で Y_cat ごとに CV の fold 数 + 1 回学習するので、CPU で1試行が数分に収まるよう、
        計算量を決める値（層の数・幅・埋め込みの次元・バッチサイズ）の上限を抑えている。
        一番重い組み合わせ（3層×256・埋め込み16次元・バッチ64）は、TabM 論文の既定とほぼ同じ
        （約800行・51特徴量で 1 fit 30〜60秒）。以前の上限（4層×512・32次元・バッチ32）では
        1試行に20分以上かかった。
        """
        use_embeddings = trial.suggest_categorical("num_embeddings", ["periodic", "none"])
        params: dict[str, Any] = {
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "optimizer__weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
            # 小さいバッチは1エポックのステップ数が増えて遅いので、64 以上にする
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "module__n_blocks": trial.suggest_int("n_blocks", 1, 3),
            "module__d_block": trial.suggest_int("d_block", 64, 256, step=64),
            "module__dropout": trial.suggest_float("module__dropout", 0.0, 0.5),
            "module__num_embeddings": None if use_embeddings == "none" else "periodic",
        }
        if use_embeddings == "periodic":
            params["module__d_embedding"] = trial.suggest_int("d_embedding", 8, 16, step=8)
            params["module__frequency_init_scale"] = trial.suggest_float(
                "frequency_init_scale", 0.01, 10.0, log=True
            )
        return params


@register_model
class Conv1dSpec(_NeuralNetSpec):
    """1次元CNN（`Conv1dModule`）。"""

    name = "cnn1d"
    module_cls = Conv1dModule

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """共通の探索空間に、チャネル数・カーネルサイズを加える。"""
        base = trial.suggest_int("base_channels", 8, 64, log=True)
        return {
            **super().search_space(trial, task),
            "module__channels": [base, base * 2],
            "module__kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
            "module__hidden_size": trial.suggest_int("hidden_size", 32, 256, log=True),
        }
