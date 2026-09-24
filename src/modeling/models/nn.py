"""ニューラルネット（PyTorch + skorch）。

skorchでPyTorchモデルをsklearn互換estimatorにし、他モデルと同じ `ModelSpec` として扱う。
torch・skorchは任意依存（`uv sync --extra nn`）で、未インストールの環境では
このモジュールのimportに失敗し、`modeling.models` はNNモデルを登録せずに動作する。

- `mlp`: 全結合ネットワーク
- `cnn1d`: 1次元畳み込みネットワーク。特徴量の**並び順を系列**とみなして畳み込むため、
  ラグ特徴量のように順序に意味がある特徴量（例: `x_lag_1, x_lag_2, ...`）で使う。

いずれも「欠損値補完（中央値）→ 標準化 → ネットワーク」のPipelineとして構築する。
early stoppingは学習foldの中から切り出した検証用データ（10%）で行い、
OOF評価用の検証foldは使わない。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from torch import nn

from modeling.models.base import ModelSpec, merge_params, register_model
from modeling.tasks import Task, is_classification

_ALL_TASKS = frozenset(Task)


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


def _as_float32(X: Any) -> Any:
    """入力をPyTorchが扱えるfloat32のnumpy配列にする。

    skorchは学習中の検証スコア計算で `Dataset` を直接 `predict` に渡すため、
    `Dataset` は変換せずそのまま返す。
    """
    if isinstance(X, torch.utils.data.Dataset):
        return X
    return np.asarray(X, dtype=np.float32)


class TabularNetRegressor(NeuralNetRegressor):
    """表形式データ用のskorch回帰器（入力次元の自動設定・乱数シード固定・1次元出力）。

    Args:
        *args: `NeuralNetRegressor` の位置引数。
        random_state: fit時に設定するPyTorchの乱数シード。
        **kwargs: `NeuralNetRegressor` のキーワード引数。
    """

    def __init__(self, *args: Any, random_state: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> TabularNetRegressor:
        """入力次元を設定して学習する。"""
        X32 = _as_float32(X)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.set_params(module__n_features=X32.shape[1], module__n_outputs=1)
        # skorchの回帰は目的変数を (n, 1) のfloat32で受け取る
        super().fit(X32, np.asarray(y, dtype=np.float32).reshape(-1, 1), **fit_params)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """予測値を1次元配列で返す。"""
        return np.asarray(super().predict(_as_float32(X))).ravel()


class TabularNetClassifier(NeuralNetClassifier):
    """表形式データ用のskorch分類器（入力次元・クラス数の自動設定・乱数シード固定）。

    Args:
        *args: `NeuralNetClassifier` の位置引数。
        random_state: fit時に設定するPyTorchの乱数シード。
        **kwargs: `NeuralNetClassifier` のキーワード引数。
    """

    def __init__(self, *args: Any, random_state: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> TabularNetClassifier:
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
        super().fit(X32, y64, **fit_params)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """クラス確率を返す。"""
        return np.asarray(super().predict_proba(_as_float32(X)))

    def predict(self, X: Any) -> np.ndarray:
        """予測クラスを返す。"""
        return np.asarray(super().predict(_as_float32(X)))


class _NeuralNetSpec(ModelSpec):
    """skorchモデル共通の `ModelSpec`。"""

    supported_tasks = _ALL_TASKS
    module_cls: type[nn.Module]

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """欠損補完 + 標準化 + skorchネットワークのPipelineを作る。"""
        defaults: dict[str, Any] = {
            "max_epochs": 100,
            "lr": 1e-3,
            "batch_size": 256,
            "optimizer": torch.optim.AdamW,
            "verbose": 0,
            "random_state": seed,
            "train_split": None,
        }
        if early_stopping_rounds is not None:
            # 学習foldの一部を検証用に切り出してearly stoppingする（OOF用の検証foldは使わない）
            defaults["train_split"] = ValidSplit(
                0.1, stratified=is_classification(task), random_state=seed
            )
            defaults["callbacks"] = [EarlyStopping(patience=early_stopping_rounds, load_best=True)]
        if is_classification(task):
            # モジュールはlogitsを出力する。CrossEntropyLossを指定すると、skorchは
            # predict_probaでsoftmaxを自動適用する（既定のNLLLossは確率出力が前提）
            defaults["criterion"] = nn.CrossEntropyLoss
        merged = merge_params(defaults, params)
        net_cls = TabularNetClassifier if is_classification(task) else TabularNetRegressor
        net = net_cls(module=self.module_cls, **merged)
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), net
        )

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """学習率・バッチサイズ・ドロップアウトの探索空間（ネットワーク形状はサブクラス）。"""
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "module__dropout": trial.suggest_float("module__dropout", 0.0, 0.5),
        }

    def best_iteration(self, estimator: Any) -> int | None:
        """early stoppingで最良だったエポック数（early stoppingなしならNone）。"""
        net = estimator[-1]
        if net.train_split is None:
            return None
        valid_losses = net.history[:, "valid_loss"]
        return int(np.argmin(valid_losses)) + 1

    def with_n_iterations(self, params: dict[str, Any], n_iterations: int) -> dict[str, Any]:
        """エポック数を固定したパラメータを返す。"""
        return {**params, "max_epochs": n_iterations}


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
