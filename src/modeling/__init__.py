"""modeling パッケージ（モデル学習・評価・チューニング・解釈・アンサンブルの汎用基盤）。

sklearn互換estimatorを共通インターフェースとし、次の小さな部品を組み合わせて使う。

- `config`: YAML設定の検証（pydantic）
- `cv`: CV分割の生成
- `models`: モデルごとの差異を吸収する `ModelSpec` とそのレジストリ
- `pipeline`: 特徴量エンジニアリング + モデルのsklearn Pipeline組み立て
- `trainer`: CV学習ループ（OOF予測・テスト予測）
- `tracking`: 実験ログ（MLflow）
- `experiment`: 上記を繋いだ1実験の実行
"""
