"""evaluation パッケージ（モデルの誤差評価の可視化）。

予測値と実測値（必要に応じて特徴量行列）を受け取るだけの汎用部品で、`modeling` に依存しない。
各部品は scikit-learn の Display API に合わせ、`XxxDisplay.from_predictions(...)` で計算して描画し、
`plot(ax=...)` で描き直せる。描画後は `figure_` / `ax_` と計算結果を属性に持つ。

- `residuals`: 残差分布・残差プロット・正規Q-Qプロット・残差のACF/PACF
- `influence`: Leverage と Cook の距離
- `classification`: 混同行列・ROC曲線・PR曲線（二値・多クラス）
- `curves`: 学習曲線・検証曲線（sklearnの関数を利用）・学習の推移・予測ステップ別の誤差
"""
