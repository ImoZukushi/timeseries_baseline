"""feature_engineering.categorical のテスト。"""

from __future__ import annotations

import polars as pl
import pytest

from feature_engineering.categorical import (
    PolarsOneHotEncoder,
    PolarsOrdinalEncoder,
    PolarsTargetEncoder,
    RareLabelGrouper,
    TargetGuidedOrdinalEncoder,
)

# --- PolarsOneHotEncoder ------------------------------------------------------


def test_one_hot_encoder_all_categories() -> None:
    train = pl.DataFrame({"c": ["a", "b", "c", "a"]})
    enc = PolarsOneHotEncoder("c").fit(train)
    out = enc.transform(train)
    assert set(out.columns) == {"c", "c_a", "c_b", "c_c"}
    assert out.row(0, named=True) == {"c": "a", "c_a": 1.0, "c_b": 0.0, "c_c": 0.0}


def test_one_hot_encoder_frequent_categories_only() -> None:
    train = pl.DataFrame({"c": ["a", "a", "a", "b", "c"]})
    enc = PolarsOneHotEncoder("c", min_frequency=2).fit(train)
    out = enc.transform(train)
    # b, c は頻度1のため infrequent にまとめられ、専用の列は作られない
    assert "c_b" not in out.columns
    assert "c_c" not in out.columns
    assert "c_a" in out.columns


def test_one_hot_encoder_unseen_category_is_all_zero() -> None:
    train = pl.DataFrame({"c": ["a", "b"]})
    test = pl.DataFrame({"c": ["z"]})
    enc = PolarsOneHotEncoder("c").fit(train)
    out = enc.transform(test)
    assert out.row(0, named=True)["c_a"] == 0.0
    assert out.row(0, named=True)["c_b"] == 0.0


def test_one_hot_encoder_leak_invariance() -> None:
    # fit済みインスタンスでは、testを単独でtransformしてもtrainと結合してtransformしても
    # test側の結果は変わらない
    train = pl.DataFrame({"c": ["a", "a", "b", "b"]})
    test = pl.DataFrame({"c": ["a", "c"]})
    enc = PolarsOneHotEncoder("c").fit(train)
    alone = enc.transform(test)
    combined = enc.transform(pl.concat([train, test])).tail(2)
    assert alone.select(["c_a", "c_b"]).to_dicts() == combined.select(["c_a", "c_b"]).to_dicts()


# --- PolarsOrdinalEncoder ------------------------------------------------------


def test_ordinal_encoder_assigns_sorted_integers() -> None:
    train = pl.DataFrame({"c": ["banana", "apple", "cherry", "apple"]})
    enc = PolarsOrdinalEncoder("c").fit(train)
    out = enc.transform(train)
    # ソート順: apple=0, banana=1, cherry=2
    assert out["c"].to_list() == [1.0, 0.0, 2.0, 0.0]


def test_ordinal_encoder_unseen_category_is_null() -> None:
    train = pl.DataFrame({"c": ["a", "b"]})
    test = pl.DataFrame({"c": ["a", "z"]})
    enc = PolarsOrdinalEncoder("c").fit(train)
    out = enc.transform(test)
    assert out["c"].to_list() == [0.0, None]


# --- TargetGuidedOrdinalEncoder ------------------------------------------------


def test_target_guided_ordinal_ranks_by_target_mean() -> None:
    train = pl.DataFrame({"c": ["low", "low", "high", "mid"], "y": [1.0, 2.0, 10.0, 5.0]})
    enc = TargetGuidedOrdinalEncoder("c").fit(train, train["y"])
    out = enc.transform(train)
    # low(平均1.5) < mid(平均5.0) < high(平均10.0)
    assert enc.mappings_["c"] == {"low": 0, "mid": 1, "high": 2}
    assert out["c"].to_list() == [0, 0, 2, 1]


def test_target_guided_ordinal_unseen_category_is_null() -> None:
    train = pl.DataFrame({"c": ["a", "b"], "y": [1.0, 2.0]})
    test = pl.DataFrame({"c": ["a", "z"]})
    enc = TargetGuidedOrdinalEncoder("c").fit(train, train["y"])
    out = enc.transform(test)
    assert out["c"].to_list()[1] is None


# --- PolarsTargetEncoder --------------------------------------------------------


def test_target_encoder_transform_uses_category_mean() -> None:
    train = pl.DataFrame({"c": ["a", "a", "b", "b"], "y": [1.0, 3.0, 10.0, 20.0]})
    test = pl.DataFrame({"c": ["a", "b"]})
    enc = PolarsTargetEncoder("c", smoothing=0.0, cv=2).fit(train, train["y"])
    out = enc.transform(test)
    assert out["c"].to_list() == pytest.approx([2.0, 15.0])


def test_target_encoder_unseen_category_falls_back_to_global_mean() -> None:
    train = pl.DataFrame({"c": ["a", "a", "b", "b"], "y": [1.0, 3.0, 10.0, 20.0]})
    test = pl.DataFrame({"c": ["unknown"]})
    enc = PolarsTargetEncoder("c", smoothing=0.0, cv=2).fit(train, train["y"])
    out = enc.transform(test)
    assert out["c"].to_list() == pytest.approx([8.5])  # (1+3+10+20)/4


def test_target_encoder_leak_invariance() -> None:
    train = pl.DataFrame({"c": ["a", "a", "b", "b", "c"], "y": [1.0, 3.0, 10.0, 20.0, 5.0]})
    test = pl.DataFrame({"c": ["a", "b"]})
    enc = PolarsTargetEncoder("c", cv=2).fit(train, train["y"])
    alone = enc.transform(test)["c"].to_list()
    combined = enc.transform(pl.concat([train.drop("y"), test]))["c"].to_list()[-2:]
    assert alone == pytest.approx(combined)


def test_target_encoder_fit_transform_is_cross_fitted_not_plain_mean() -> None:
    # fit_transform（訓練データ自身への適用）は、fit().transform()による
    # 「全データの平均をそのまま使う」方式とは異なる値になるはず
    # （cross-fittingにより自分自身のyの寄与が除かれるため）。
    train = pl.DataFrame({"c": ["a", "a", "a", "a"], "y": [1.0, 2.0, 3.0, 4.0]})
    enc = PolarsTargetEncoder("c", smoothing=0.0, cv=2)
    oof = enc.fit_transform(train, train["y"])["c"].to_list()

    enc2 = PolarsTargetEncoder("c", smoothing=0.0, cv=2).fit(train, train["y"])
    plain = enc2.transform(train)["c"].to_list()

    assert oof != pytest.approx(plain)
    # 全て同一カテゴリ"a"の全体平均(2.5)がplain transformの値になっているはず
    assert plain == pytest.approx([2.5, 2.5, 2.5, 2.5])


def test_target_encoder_fit_transform_requires_y() -> None:
    train = pl.DataFrame({"c": ["a", "b"]})
    enc = PolarsTargetEncoder("c")
    with pytest.raises(ValueError):
        enc.fit_transform(train)


def test_target_encoder_fit_transform_is_reproducible_across_calls() -> None:
    # cvに単なる整数を渡すscikit-learn側の既定動作はfold分割がシードされておらず
    # 実行のたびに結果が変わりうる（実際に発生を確認したバグ）。random_stateを
    # 明示的に使うことで、同じデータ・同じパラメータなら常に同じ結果になることを保証する。
    train = pl.DataFrame({"c": ["a", "a", "a", "a"], "y": [1.0, 2.0, 3.0, 4.0]})
    results = [
        PolarsTargetEncoder("c", smoothing=0.0, cv=2)
        .fit_transform(train, train["y"])["c"]
        .to_list()
        for _ in range(5)
    ]
    assert all(r == results[0] for r in results)


# --- RareLabelGrouper -----------------------------------------------------------


def test_rare_label_grouper_by_min_frequency() -> None:
    train = pl.DataFrame({"c": ["a", "a", "a", "b", "c"]})
    grouper = RareLabelGrouper("c", min_frequency=2).fit(train)
    out = grouper.transform(train)
    assert out["c"].to_list() == ["a", "a", "a", "Other", "Other"]


def test_rare_label_grouper_by_min_ratio() -> None:
    train = pl.DataFrame({"c": ["a"] * 8 + ["b"] * 2})
    grouper = RareLabelGrouper("c", min_ratio=0.5).fit(train)
    out = grouper.transform(train)
    assert set(out["c"].to_list()) == {"a", "Other"}


def test_rare_label_grouper_unseen_category_becomes_other() -> None:
    train = pl.DataFrame({"c": ["a", "a", "a", "a"]})
    test = pl.DataFrame({"c": ["z"]})
    grouper = RareLabelGrouper("c", min_frequency=2).fit(train)
    assert grouper.transform(test)["c"].to_list() == ["Other"]


def test_rare_label_grouper_custom_other_label() -> None:
    train = pl.DataFrame({"c": ["a", "a", "b"]})
    grouper = RareLabelGrouper("c", min_frequency=2, other_label="RARE").fit(train)
    assert grouper.transform(train)["c"].to_list() == ["a", "a", "RARE"]


def test_rare_label_grouper_requires_threshold() -> None:
    train = pl.DataFrame({"c": ["a", "b"]})
    with pytest.raises(ValueError):
        RareLabelGrouper("c").fit(train)
