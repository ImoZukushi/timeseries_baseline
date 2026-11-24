"""modeling.models のレジストリ（登録できなかったモデルのエラー）のテスト。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from modeling.models.base import _UNAVAILABLE, get_model_spec, register_unavailable

_SRC = Path(__file__).resolve().parents[1] / "src"


def test_unknown_model_lists_available_models() -> None:
    with pytest.raises(KeyError, match="未登録のモデルです: no_such_model"):
        get_model_spec("no_such_model")


def test_unavailable_model_reports_reason() -> None:
    try:
        register_unavailable(["_dummy_unavailable"], "依存パッケージがありません")
        with pytest.raises(KeyError, match="この環境では使えません: 依存パッケージがありません"):
            get_model_spec("_dummy_unavailable")
    finally:
        _UNAVAILABLE.pop("_dummy_unavailable", None)


@pytest.mark.parametrize("blocked", ["torch", "skorch"])
def test_nn_models_without_torch_explain_how_to_install(blocked: str) -> None:
    # torch・skorch が無い環境（本番で `uv sync` だけした場合）を、import を失敗させて再現する
    code = (
        # import の段階で失敗させる（sys.modules に None を入れると、読み込み済みとみなす
        # ライブラリ（scipy）が壊れるため）
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        f"        if name.split('.')[0] == {blocked!r}:\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}')\n"
        "sys.meta_path.insert(0, Block())\n"
        "from modeling.models import available_models, get_model_spec\n"
        "assert 'tabm' not in available_models()\n"
        "assert 'lightgbm' in available_models()\n"
        "try:\n"
        "    get_model_spec('tabm')\n"
        "except KeyError as e:\n"
        "    print(e)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": str(_SRC), "PYTHONIOENCODING": "utf-8"},
        check=True,
    )
    assert "この環境では使えません" in result.stdout
    assert "uv sync --extra nn" in result.stdout
    assert blocked in result.stdout
