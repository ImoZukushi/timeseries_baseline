"""feature_engineering.scaling のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas_helpers import as_list

from feature_engineering.scaling import (
    GroupScaler,
    MeanNormalizationScaler,
    PandasMaxAbsScaler,
    PandasMinMaxScaler,
    PandasRobustScaler,
)

# --- PandasMinMaxScaler -----------------------------------------------------------


def test_min_max_scaler_scales_to_unit_interval() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = PandasMinMaxScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].pipe(as_list) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_min_max_scaler_test_values_can_exceed_unit_interval() -> None:
    train = pd.DataFrame({"x": [1.0, 5.0]})
    test = pd.DataFrame({"x": [0.0, 10.0]})
    scaler = PandasMinMaxScaler("x").fit(train)
    out = scaler.transform(test)
    assert out["x"].pipe(as_list) == pytest.approx([-0.25, 2.25])


def test_min_max_scaler_constant_column_no_division_by_zero() -> None:
    train = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
    scaler = PandasMinMaxScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].pipe(as_list) == pytest.approx([0.0, 0.0, 0.0])


def test_min_max_scaler_round_trip() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = PandasMinMaxScaler("x").fit(train)
    back = scaler.inverse_transform(scaler.transform(train))
    assert back["x"].pipe(as_list) == pytest.approx(train["x"].pipe(as_list))


def test_min_max_scaler_leak_invariance() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [10.0, -10.0]})
    scaler = PandasMinMaxScaler("x").fit(train)
    alone = scaler.transform(test)["x"].pipe(as_list)
    combined = scaler.transform(pd.concat([train, test], ignore_index=True))["x"].pipe(as_list)[-2:]
    assert alone == pytest.approx(combined)


# --- PandasRobustScaler ------------------------------------------------------------


def test_robust_scaler_uses_median_and_iqr() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = PandasRobustScaler("x").fit(train)
    out = scaler.transform(train)
    # median=3.0, IQR(25-75%)=2.0 -> (x-3)/2
    assert out["x"].pipe(as_list) == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])


def test_robust_scaler_is_not_affected_by_outlier() -> None:
    train_with_outlier = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
    scaler = PandasRobustScaler("x").fit(train_with_outlier)
    out = scaler.transform(pd.DataFrame({"x": [3.0]}))
    # 外れ値1000があっても中央値(3.5)付近はスケール後もほぼ0に近い
    assert abs(out["x"][0]) < 1.0


def test_robust_scaler_round_trip() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = PandasRobustScaler("x").fit(train)
    back = scaler.inverse_transform(scaler.transform(train))
    assert back["x"].pipe(as_list) == pytest.approx(train["x"].pipe(as_list))


def test_robust_scaler_constant_column_no_division_by_zero() -> None:
    train = pd.DataFrame({"x": [7.0, 7.0, 7.0]})
    scaler = PandasRobustScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].pipe(as_list) == pytest.approx([0.0, 0.0, 0.0])


# --- MeanNormalizationScaler --------------------------------------------------------


def test_mean_normalization_formula() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = MeanNormalizationScaler("x").fit(train)
    out = scaler.transform(train)
    # mean=3.0, range=4.0 -> (x-3)/4
    assert out["x"].pipe(as_list) == pytest.approx([-0.5, -0.25, 0.0, 0.25, 0.5])


def test_mean_normalization_constant_column_no_division_by_zero() -> None:
    train = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
    scaler = MeanNormalizationScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].pipe(as_list) == pytest.approx([0.0, 0.0, 0.0])


def test_mean_normalization_round_trip() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = MeanNormalizationScaler("x").fit(train)
    back = scaler.inverse_transform(scaler.transform(train))
    assert back["x"].pipe(as_list) == pytest.approx(train["x"].pipe(as_list))


def test_mean_normalization_leak_invariance() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [100.0]})
    scaler = MeanNormalizationScaler("x").fit(train)
    alone = scaler.transform(test)["x"].pipe(as_list)
    combined = scaler.transform(pd.concat([train, test], ignore_index=True))["x"].pipe(as_list)[-1:]
    assert alone == pytest.approx(combined)


# --- PandasMaxAbsScaler -------------------------------------------------------------


def test_max_abs_scaler_scales_by_max_absolute_value() -> None:
    train = pd.DataFrame({"x": [-4.0, 2.0, 4.0]})
    scaler = PandasMaxAbsScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].pipe(as_list) == pytest.approx([-1.0, 0.5, 1.0])


def test_max_abs_scaler_all_zero_column_no_division_by_zero() -> None:
    train = pd.DataFrame({"x": [0.0, 0.0, 0.0]})
    scaler = PandasMaxAbsScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].pipe(as_list) == pytest.approx([0.0, 0.0, 0.0])


def test_max_abs_scaler_round_trip() -> None:
    train = pd.DataFrame({"x": [-4.0, 2.0, 4.0, -1.0]})
    scaler = PandasMaxAbsScaler("x").fit(train)
    back = scaler.inverse_transform(scaler.transform(train))
    assert back["x"].pipe(as_list) == pytest.approx(train["x"].pipe(as_list))


# --- GroupScaler -------------------------------------------------------------------


def _grouped_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "z": [5.0, 5.0, 5.0, 0.0, 1.0, 2.0],
        }
    )


def test_group_scaler_standardizes_each_group() -> None:
    train = _grouped_frame()
    out = GroupScaler(["x", "z"], group_col="g").fit_transform(train)
    std = np.sqrt(2.0 / 3.0)  # ddof=0 の標準偏差（[1, 2, 3] と [0, 1, 2]）
    # a は [1, 2, 3]、b は [10, 20, 30]（標準偏差は a の10倍）。どちらも同じ値になる
    expected = [-1.0 / std, 0.0, 1.0 / std]
    assert out["x"].pipe(as_list) == pytest.approx(expected * 2)
    # 定数のグループ（a の z）は分母を1にする（0除算にしない）
    assert out["z"].pipe(as_list)[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert out["z"].pipe(as_list)[3:] == pytest.approx([-1.0 / std, 0.0, 1.0 / std])
    # グループの列は変えない
    assert out["g"].pipe(as_list) == train["g"].pipe(as_list)


def test_group_scaler_normalizes_each_group() -> None:
    train = _grouped_frame()
    out = GroupScaler("x", group_col="g", method="normalize").fit_transform(train)
    assert out["x"].pipe(as_list) == pytest.approx([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])


def test_group_scaler_round_trip() -> None:
    train = _grouped_frame()
    for method in ("standardize", "normalize"):
        scaler = GroupScaler(["x", "z"], group_col="g", method=method).fit(train)
        back = scaler.inverse_transform(scaler.transform(train))
        assert back["x"].pipe(as_list) == pytest.approx(train["x"].pipe(as_list))
        assert back["z"].pipe(as_list) == pytest.approx(train["z"].pipe(as_list))


def test_group_scaler_unseen_group_uses_all_rows() -> None:
    train = _grouped_frame()
    scaler = GroupScaler("x", group_col="g", method="normalize").fit(train)
    out = scaler.transform(pd.DataFrame({"g": ["c"], "x": [15.5]}))
    # 全行の最小値 1・最大値 30 で正規化する
    assert out["x"].iloc[0] == pytest.approx((15.5 - 1.0) / 29.0)


def test_group_scaler_keeps_missing_and_inf() -> None:
    train = pd.DataFrame({"g": ["a", "a", "a", "a"], "x": [1.0, 3.0, np.nan, np.inf]})
    scaler = GroupScaler("x", group_col="g", method="normalize").fit(train)
    out = scaler.transform(train)["x"].to_numpy()
    # 統計量は欠損・inf を除いて求める（最小値 1・最大値 3）
    assert out[:2] == pytest.approx([0.0, 1.0])
    assert np.isnan(out[2])
    assert np.isposinf(out[3])


def test_group_scaler_leak_invariance() -> None:
    train = _grouped_frame()
    test = pd.DataFrame({"g": ["b", "a"], "x": [100.0, -100.0], "z": [7.0, 8.0]})
    scaler = GroupScaler(["x", "z"], group_col="g").fit(train)
    alone = scaler.transform(test)["x"].pipe(as_list)
    combined = scaler.transform(pd.concat([train, test], ignore_index=True))["x"].pipe(as_list)[-2:]
    assert alone == pytest.approx(combined)


def test_group_scaler_rejects_invalid_settings() -> None:
    train = _grouped_frame()
    with pytest.raises(ValueError, match="method"):
        GroupScaler("x", group_col="g", method="robust").fit(train)
    with pytest.raises(ValueError, match="グループの列"):
        GroupScaler(["x", "g"], group_col="g").fit(train)
    with pytest.raises(KeyError):
        GroupScaler("missing", group_col="g").fit(train)
