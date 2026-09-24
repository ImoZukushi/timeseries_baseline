"""実験ログの記録。

学習コードは `Tracker` Protocol にのみ依存する。既定はローカルのMLflow
（サーバー不要、`mlruns/mlflow.db` + `mlruns/artifacts/`）で、次のコマンドでブラウザから比較できる:

    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

テストや記録不要な試行では `NullTracker` を使う。
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from util.paths import get_repo_root

# MLflow import時の案内メッセージを抑制する
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


class Tracker(Protocol):
    """実験ログの記録先のインターフェース。"""

    def start_run(
        self, run_name: str, tags: Mapping[str, str] | None = None, nested: bool = False
    ) -> Any:
        """runを開始するコンテキストマネージャを返す。"""
        ...

    def log_params(self, params: Mapping[str, Any]) -> None:
        """パラメータを記録する。"""
        ...

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """指標を記録する。"""
        ...

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """ファイルを記録する。"""
        ...

    def active_run_id(self) -> str | None:
        """実行中のrun IDを返す（無ければNone）。"""
        ...


class NullTracker:
    """何も記録しないTracker（テストや記録不要な試行用）。"""

    @contextmanager
    def start_run(
        self, run_name: str, tags: Mapping[str, str] | None = None, nested: bool = False
    ) -> Iterator[None]:
        """何もしないコンテキストマネージャ。"""
        yield None

    def log_params(self, params: Mapping[str, Any]) -> None:
        """何もしない。"""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """何もしない。"""

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """何もしない。"""

    def active_run_id(self) -> str | None:
        """常にNone。"""
        return None


def default_tracking_uri() -> str:
    """リポジトリ直下 `mlruns/mlflow.db` を指すSQLiteのtracking URI。"""
    db_path = get_repo_root() / "mlruns" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


class MLflowTracker:
    """MLflowに記録するTracker。

    Args:
        experiment_name: MLflowの実験名。
        tracking_uri: tracking URI（Noneなら `default_tracking_uri()`）。
        artifact_root: 実験を新規作成する場合のartifact保存先（Noneなら `mlruns/artifacts`）。
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        artifact_root: Path | None = None,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri or default_tracking_uri())
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            root = artifact_root or (get_repo_root() / "mlruns" / "artifacts")
            self.experiment_id = mlflow.create_experiment(
                experiment_name, artifact_location=(root / experiment_name).as_uri()
            )
        else:
            self.experiment_id = experiment.experiment_id

    @contextmanager
    def start_run(
        self, run_name: str, tags: Mapping[str, str] | None = None, nested: bool = False
    ) -> Iterator[Any]:
        """MLflowのrunを開始する。git commit hashを自動でタグに付ける。"""
        all_tags = {"git_commit": _git_commit_hash(), **(tags or {})}
        with self._mlflow.start_run(
            experiment_id=self.experiment_id, run_name=run_name, tags=all_tags, nested=nested
        ) as run:
            yield run

    def log_params(self, params: Mapping[str, Any]) -> None:
        """パラメータを記録する（ネストしたdictは `a.b.c` 形式に平坦化する）。"""
        flat = flatten_dict(params)
        # MLflowのparam値は文字列長に上限があるため切り詰める
        self._mlflow.log_params({k: str(v)[:6000] for k, v in flat.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """指標を記録する。"""
        self._mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """ファイルまたはディレクトリを記録する。"""
        if path.is_dir():
            self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def active_run_id(self) -> str | None:
        """実行中のrun ID。"""
        run = self._mlflow.active_run()
        return None if run is None else str(run.info.run_id)


def flatten_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """ネストしたdictを `親.子` 形式のキーで平坦化する。

    Examples:
        >>> flatten_dict({"model": {"name": "lgbm", "params": {"lr": 0.1}}})
        {'model.name': 'lgbm', 'model.params.lr': 0.1}
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_dict(value, prefix=f"{full_key}."))
        else:
            flat[full_key] = value
    return flat


def _git_commit_hash() -> str:
    """現在のgit commit hash（取得できなければ "unknown"）。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=get_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()
