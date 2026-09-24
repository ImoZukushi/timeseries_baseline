"""モデル定義（`ModelSpec`）のレジストリ。

このパッケージをimportすると、同梱のモデルがすべてレジストリに登録される。
"""

from modeling.models import gbdt, linear
from modeling.models.base import ModelSpec, available_models, get_model_spec, register_model

__all__ = [
    "ModelSpec",
    "available_models",
    "gbdt",
    "get_model_spec",
    "linear",
    "register_model",
]
