"""modeling.tracking.make_tracker（設定から実験ログの記録先を作る）のテスト。"""

from __future__ import annotations

from pathlib import Path

from modeling.config import TrackingConfig
from modeling.tracking import MLflowTracker, NullTracker, make_tracker


def test_make_tracker_disabled_returns_null_tracker() -> None:
    # 設定で無効、または CLI の --no-tracking（enabled=False）なら記録しない
    assert isinstance(make_tracker(TrackingConfig(enabled=False), "exp"), NullTracker)
    assert isinstance(make_tracker(TrackingConfig(), "exp", enabled=False), NullTracker)


def test_make_tracker_uses_experiment_name_or_default(tmp_path: Path) -> None:
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    # 実験名の指定が無ければ default_name（設定の name）を使う
    tracker = make_tracker(TrackingConfig(tracking_uri=uri), "default_exp")
    assert isinstance(tracker, MLflowTracker)
    assert mlflow.get_experiment(tracker.experiment_id).name == "default_exp"
    named = make_tracker(TrackingConfig(tracking_uri=uri, experiment_name="named"), "default_exp")
    assert isinstance(named, MLflowTracker)
    assert mlflow.get_experiment(named.experiment_id).name == "named"
