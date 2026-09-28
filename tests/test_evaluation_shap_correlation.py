"""evaluation.shap_correlation のテスト。"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from evaluation.shap_correlation import (
    ShapCorrelationBarDisplay,
    ShapDependenceDisplay,
    ShapScatterMatrixDisplay,
    ShapValueCorrelationDisplay,
    shap_feature_correlation,
    shap_value_correlation,
)

_NAMES = ["up", "down", "const", "missing"]


def _synthetic(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """正の効果・負の効果・定数・欠損を含む特徴量と、そのSHAP値を作る。"""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 4))
    x[:, 2] = 1.0  # 一定の特徴量
    x[::5, 3] = np.nan  # 欠損を含む特徴量
    shap_values = np.column_stack(
        [
            2.0 * x[:, 0],  # 値が大きいほど予測を上げる
            -1.0 * x[:, 1],  # 値が大きいほど予測を下げる
            np.zeros(n),  # 効果なし
            0.5 * np.nan_to_num(x[:, 3]) + rng.normal(scale=0.01, size=n),
        ]
    )
    return shap_values, x


def test_feature_correlation_direction() -> None:
    values, x = _synthetic()
    table = shap_feature_correlation(values, x, _NAMES)
    rows = {r["feature"]: r for r in table.iter_rows(named=True)}
    assert rows["up"]["direction"] == "正"
    assert rows["up"]["pearson"] == pytest.approx(1.0)
    assert rows["down"]["direction"] == "負"
    assert rows["down"]["spearman"] == pytest.approx(-1.0)
    # 一定の特徴量は相関が定義できない
    assert rows["const"]["direction"] == "なし"
    assert rows["const"]["pearson"] is None
    # 欠損を含む特徴量も有効な組だけで計算する
    assert rows["missing"]["pearson"] > 0.9
    # 重要度の降順
    assert table["feature"].to_list()[0] == "up"


def test_value_correlation_matrix() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    values = np.column_stack([a, -a, rng.normal(size=200), np.zeros(200)])
    matrix = shap_value_correlation(values, ["a", "neg_a", "noise", "zero"])
    assert matrix.columns == ["feature", "a", "neg_a", "noise", "zero"]
    m = matrix.drop("feature").to_numpy()
    finite = m[:3, :3]
    assert np.allclose(finite, finite.T)
    assert np.allclose(np.diag(finite), 1.0)
    assert m[0, 1] == pytest.approx(-1.0)
    # 一定のSHAP値の列は欠損
    assert np.isnan(m[3]).all()


def test_correlation_bar_display() -> None:
    values, x = _synthetic()
    disp = ShapCorrelationBarDisplay.from_shap(values, x, _NAMES, max_display=3)
    assert len(disp.ax_.patches) == 3
    assert disp.figure_ is disp.ax_.figure
    plt.close(disp.figure_)


def test_dependence_display_panels() -> None:
    values, x = _synthetic()
    disp = ShapDependenceDisplay.from_shap(values, x, _NAMES, top_k=3, ncols=2, max_points=100)
    assert disp.ax_.shape == (2, 2)
    assert len(disp.features) == 3
    assert disp.features[0] == "up"
    # 余ったパネルは非表示
    assert not disp.ax_[1, 1].axison
    plt.close(disp.figure_)


def test_value_correlation_display_top_k() -> None:
    values, _ = _synthetic()
    disp = ShapValueCorrelationDisplay.from_shap(values, _NAMES, top_k=2)
    assert disp.matrix["feature"].to_list() == ["up", "down"]
    plt.close(disp.figure_)


def test_scatter_matrix_display_shape() -> None:
    values, _ = _synthetic()
    disp = ShapScatterMatrixDisplay.from_shap(values, _NAMES, top_k=3, max_points=50)
    assert disp.ax_.shape == (3, 3)
    assert disp.features == ["up", "down", "missing"]
    # 上三角は非表示、下三角は散布図
    assert not disp.ax_[0, 1].axison
    assert len(disp.ax_[1, 0].collections) == 1
    plt.close(disp.figure_)
