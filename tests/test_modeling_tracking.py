"""modeling.tracking（MLflow への記録）のテスト。"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pytest

from modeling.tracking import MLflowTracker, artifact_location_is_writable


def _uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_artifact_location_is_writable(tmp_path: Path) -> None:
    assert artifact_location_is_writable((tmp_path / "artifacts").as_uri())
    assert artifact_location_is_writable(str(tmp_path / "plain"))
    # ファイルの下にはフォルダを作れない（別の環境で作られた保存先を想定）
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    assert not artifact_location_is_writable((blocker / "artifacts").as_uri())
    # リモートの保存先は確かめない
    assert artifact_location_is_writable("s3://bucket/path")


def test_moved_experiment_logs_artifacts_to_local_experiment(tmp_path: Path) -> None:
    uri = _uri(tmp_path)
    mlflow.set_tracking_uri(uri)
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    # 別の環境で作られ、成果物の保存先がこの環境に無い実験
    mlflow.create_experiment("moved", artifact_location=(blocker / "artifacts").as_uri())
    with pytest.warns(UserWarning, match="書き込めない"):
        tracker = MLflowTracker("moved", tracking_uri=uri, artifact_root=tmp_path / "artifacts")
    assert tracker.experiment_name.startswith("moved__")
    artifact = tmp_path / "result.txt"
    artifact.write_text("ok")
    with tracker.start_run("run"):
        tracker.log_artifact(artifact, artifact_path="eval")
        run_id = tracker.active_run_id()
    assert run_id is not None
    client = mlflow.MlflowClient(uri)
    stored = client.list_artifacts(run_id, "eval")
    assert [Path(a.path).name for a in stored] == ["result.txt"]
    # run の終了後に MlflowClient で追加する記録（要約の保存など）も、同じ保存先に書ける
    client.log_artifact(run_id, str(artifact), artifact_path="summary")
    assert [Path(a.path).name for a in client.list_artifacts(run_id, "summary")] == ["result.txt"]
    # 2回目は同じ実験（この環境用）を使う
    with pytest.warns(UserWarning):
        again = MLflowTracker("moved", tracking_uri=uri, artifact_root=tmp_path / "artifacts")
    assert again.experiment_id == tracker.experiment_id


def test_log_artifact_failure_does_not_stop_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    uri = _uri(tmp_path)
    tracker = MLflowTracker("ok", tracking_uri=uri, artifact_root=tmp_path / "artifacts")
    assert tracker.experiment_name == "ok"

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("書き込めません")

    monkeypatch.setattr(mlflow, "log_artifact", fail)
    artifact = tmp_path / "result.txt"
    artifact.write_text("ok")
    with caplog.at_level(logging.WARNING), tracker.start_run("run"):
        tracker.log_artifact(artifact)
        tracker.log_metrics({"mae": 1.0})
    assert "記録できませんでした" in caplog.text
