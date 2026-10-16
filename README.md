# ac2026

熱交換器10基の日次運転データ（ダミーデータ）から、各機器の汚れ係数を予測するモデルを作るプロジェクトです。
目的・前提は [プロジェクト概要](docs/agent/project-overview.md)、データの仕様は [データカタログ](docs/agent/data-catalog.md) を参照してください。

## 実行の流れ

```bash
uv sync

# 0. ダミーデータ（A部門の構成、9ファイル）の生成（data/synthetic/）
uv run python scripts/generate_synthetic_heat_exchanger.py
# 以下の 1〜5 は、現時点では旧形式のダミーデータ（data/raw/dummy/）を前提にしている
#    （A部門の構成への対応は今後行う）
# 1. EDA（outputs/figures・outputs/tables）
uv run python scripts/eda_heat_exchanger_dummy.py
# 2. 前処理（data/processed/dummy/{train,test}_dataset.parquet）
uv run python scripts/preprocess_heat_exchanger_dummy.py
# 3. 特徴量の探索（グループの組み合わせ・アブレーションを LightGBM で比べる）
uv run python scripts/run_feature_search.py --config configs/feature_search/hx_dummy.yaml
# 4. 学習（CV・予測・SHAP・誤差評価・MLflow記録）
uv run python scripts/run_experiment.py --config configs/experiments/hx_lgbm.yaml
# 5. 各モデルの探索 → 学習 → アンサンブルの一括実行
uv run python scripts/run_pipeline.py --config configs/pipelines/hx_ensemble.yaml
```

1〜3 は、同じ内容をセルごとに実行できるノートブック（`notebook/001`〜`003`）もあります。

## ドキュメント

- [src/ モジュールリファレンス](docs/modules/README.md): `util` / `eda` / `feature_engineering` / `modeling` / `evaluation` の仕様と使い方
- [特徴量の探索（設定の継承・特徴量ブロック）](docs/guides/feature_exploration.md): YAML を差分だけで書いて特徴量のパターンを比べる方法と、その仕様
- [特徴量の探索（グループ・アブレーション）](docs/guides/feature_search.md): 特徴量をグループに分け、組み合わせとアブレーションを `scripts/run_feature_search.py` で一括比較する方法と、その仕様
- [ハイパーパラメータの探索とアンサンブル](docs/guides/hyperparameter_tuning.md): 各モデルを Optuna で探索し、アンサンブルまでを `scripts/run_pipeline.py` で自動化する方法と、その仕様
- [リポジトリ構成](docs/agent/repository-structure.md)

## 時系列ビューア

`data/`・`outputs/` 以下の任意の時系列データ（CSV・Parquet・Excel）をブラウザで選び、
最大5系列を縦に並べて見比べるアプリ（Streamlit）。

```bash
uv run streamlit run app/timeseries_viewer/main.py
```

- サイドバーで系列ごとにファイル・時刻列・値の列を選ぶ（地点などが混在するファイルはグループ列で1つに絞る）
- 横軸（表示期間・描画点数）と縦軸（範囲・対数軸）はスライダなどで変更できる
- グラフの点をクリックすると、全系列をまたぐ縦線（カーソル）が引かれ、その時刻の各系列の値が表に出る
- 大きなCSVは初回に `data/interim/timeseries_viewer_cache/` へ Parquet のキャッシュを作る（元ファイルは変更しない）

## 画像ギャラリー

リポジトリ内の画像（PNG・JPEG・GIF・BMP・WebP・SVG）をディレクトリごとに一覧表示するアプリ（Streamlit）。

```bash
uv run streamlit run app/image_gallery/main.py
```

- サイドバーで、画像を含むディレクトリ（リポジトリ内を自動で探索。`.venv`・`.git` などは除く）を選ぶ（既定は `outputs/figures`）
- ファイル名（部分一致、`*` `?` のワイルドカード）・拡張子で絞り込み、名前・更新日時・サイズで並べ替える
- サムネイルの格子（1行の枚数・1ページの枚数を変更可）から「拡大」で元の画像・画素数・更新日時を表示し、ダウンロードできる
