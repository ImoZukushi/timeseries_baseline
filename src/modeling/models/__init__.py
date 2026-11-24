"""モデル定義（`ModelSpec`）のレジストリ。

このパッケージをimportすると、同梱のモデルがすべてレジストリに登録される。
NN（`mlp` / `tabm` / `cnn1d`）は任意依存のtorch・skorchがインストールされている場合のみ登録する
（`uv sync --extra nn`）。import できなかった場合は、その理由を記録し、
`get_model_spec("tabm")` などのエラーに出す。
"""

from modeling.models import forest, gbdt, linear
from modeling.models.base import (
    ModelSpec,
    available_models,
    get_model_spec,
    register_model,
    register_unavailable,
)

# 任意依存（torch・skorch）が要る NN のモデル名
NN_MODELS = ("mlp", "tabm", "cnn1d")

try:
    from modeling.models import nn
except ImportError as error:
    # torch / skorch 未インストール（または import に失敗した）環境ではNNモデルを登録しない。
    # モデル名の書き間違いと区別できるよう、理由を残す
    register_unavailable(
        NN_MODELS,
        f"torch・skorch を import できませんでした（{type(error).__name__}: {error}）。"
        "`uv sync --extra nn` でインストールしてください"
        "（`--extra nn` を付けずに `uv sync` すると、torch・skorch は削除されます）",
    )
    nn = None  # type: ignore[assignment]

__all__ = [
    "NN_MODELS",
    "ModelSpec",
    "available_models",
    "forest",
    "gbdt",
    "get_model_spec",
    "linear",
    "nn",
    "register_model",
]
