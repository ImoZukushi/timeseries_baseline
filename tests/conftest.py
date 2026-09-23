"""pytest共通設定。"""

import matplotlib

matplotlib.use("Agg")  # ディスプレイのないテスト環境でも図の保存だけ行う
