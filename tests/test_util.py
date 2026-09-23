"""util パッケージ（paths / plotting）のテスト。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from util import paths, plotting

# --- paths ---------------------------------------------------------------


def test_get_repo_root_contains_pyproject_toml() -> None:
    root = paths.get_repo_root()
    assert (root / "pyproject.toml").exists()


def test_data_dir_is_under_repo_root() -> None:
    assert paths.data_dir() == paths.get_repo_root() / "data"


def test_outputs_dir_is_under_repo_root() -> None:
    assert paths.outputs_dir() == paths.get_repo_root() / "outputs"


def test_ensure_parent_dir_creates_missing_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"
    assert not target.parent.exists()

    result = paths.ensure_parent_dir(target)

    assert result == target
    assert target.parent.exists()


def test_sanitize_filename_component_replaces_unsafe_chars() -> None:
    assert paths.sanitize_filename_component('a:b/c\\d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"


def test_sanitize_filename_component_leaves_safe_text_unchanged() -> None:
    assert (
        paths.sanitize_filename_component("wind_0_下条川__風速(瞬時)")
        == "wind_0_下条川__風速(瞬時)"
    )


# --- plotting --------------------------------------------------------------


def test_ensure_japanese_font_sets_meiryo() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plotting.ensure_japanese_font()
    assert plt.rcParams["font.family"] == ["Meiryo"]
    assert plt.rcParams["axes.unicode_minus"] is False


def test_add_caption_adds_text_to_figure() -> None:
    fig, _ax = plt.subplots()
    try:
        before = len(fig.texts)
        plotting.add_caption(fig, "対象: sample — テストキャプション。")
        assert len(fig.texts) == before + 1
        assert fig.texts[-1].get_text() == "対象: sample — テストキャプション。"
    finally:
        plt.close(fig)
