"""モデル定義（`ModelSpec`）のレジストリ。

このパッケージをimportすると、同梱のモデルがすべてレジストリに登録される。
NN（`mlp` / `tabm` / `cnn1d`）は任意依存のtorch・skorchがインストールされている場合のみ登録する
（`uv sync --extra nn`）。
"""

from modeling.models import forest, gbdt, linear
from modeling.models.base import ModelSpec, available_models, get_model_spec, register_model

try:
    from modeling.models import nn
except ImportError:
    # torch / skorch 未インストールの環境ではNNモデルを登録しない
    nn = None  # type: ignore[assignment]

__all__ = [
    "ModelSpec",
    "available_models",
    "forest",
    "gbdt",
    "get_model_spec",
    "linear",
    "nn",
    "register_model",
]
