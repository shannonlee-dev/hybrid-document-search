"""Checkpoint orchestration uses tiny file workers, never model downloads."""

import json

import pytest


def test_first_run_and_completed_rerun(tmp_path):
    exp, steps, calls = _pipeline(tmp_path)
    exp.run(steps)
    assert calls == ["data", "e5", "bge", "evaluate"]
    calls.clear()
    exp.run(steps)
    assert calls == []
    assert all(row["status"] == "completed" for row in exp.state["steps"].values())


def test_failure_resume_preserves_completed_indexes(tmp_path):
    exp, steps, calls = _pipeline(tmp_path, fail="bge")
    with pytest.raises(RuntimeError, match="worker failed"):
        exp.run(steps)
    assert calls == ["data", "e5", "bge"]
    assert exp.state["steps"]["bge"]["status"] == "failed"
    assert "evaluate" not in exp.state["steps"]
    exp, steps, calls = _pipeline(tmp_path)
    exp.run(steps)
    assert calls == ["bge", "evaluate"]


@pytest.mark.parametrize("mutation", ["file", "revision", "missing", "extra"])
def test_invalidation_is_dependency_scoped(tmp_path, mutation):
    exp, steps, _ = _pipeline(tmp_path)
    exp.run(steps)
    path = exp.root / "e5/value.json"
    if mutation == "file":
        path.write_text("tampered")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        path.with_name("extra").write_text("untracked output")
    exp, steps, calls = _pipeline(
        tmp_path, changed="c" if mutation == "revision" else None
    )
    exp.run(steps)
    assert calls == ["e5", "evaluate"]


def test_fresh_removes_only_owned_workspace(tmp_path):
    exp, steps, _ = _pipeline(tmp_path)
    exp.run(steps)
    other = tmp_path / "other-index"
    other.write_text("protected")
    exp.fresh()
    exp, steps, calls = _pipeline(tmp_path)
    exp.run(steps)
    assert calls == ["data", "e5", "bge", "evaluate"]
    assert other.read_text() == "protected"


def test_unowned_directory_is_rejected(tmp_path):
    from evaluation.experiment_runner import Experiment

    root = tmp_path / "run"
    root.mkdir()
    (root / "personal").write_text("keep")
    with pytest.raises(ValueError, match="owned"):
        Experiment(root)


def test_symlink_cleanup_rejected(tmp_path):
    exp, steps, _ = _pipeline(tmp_path)
    exp.run(steps)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("safe")
    (exp.root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        exp.fresh()
    assert (outside / "keep").read_text() == "safe"


def test_parent_symlink_rejected(tmp_path):
    from evaluation.experiment_runner import Experiment

    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        Experiment(tmp_path / "link/run")


def _pipeline(tmp_path, fail=None, changed=None):
    from evaluation.experiment_runner import Experiment, Step

    calls = []
    experiment = Experiment(tmp_path / "run")

    def _action(name):
        def _run(directory):
            calls.append(name)
            if name == fail:
                raise RuntimeError("worker failed")
            (directory / "value.json").write_text(json.dumps({"name": name}))

        return _run

    steps = [
        Step("data", (), {"seed": 42}, _action("data")),
        Step("e5", ("data",), {"revision": changed or "a"}, _action("e5")),
        Step("bge", ("data",), {"revision": "b"}, _action("bge")),
        Step("evaluate", ("e5", "bge"), {}, _action("evaluate")),
    ]
    return experiment, steps, calls


def test_completed_steps_are_reused_after_removing_final_stage(tmp_path):
    from evaluation.experiment_runner import Step

    exp, steps, _ = _pipeline(tmp_path)
    exp.run(
        [
            *steps,
            Step(
                "package",
                ("evaluate",),
                {},
                lambda directory: (directory / "old.json").write_text("{}"),
            ),
        ]
    )
    resumed, steps, calls = _pipeline(tmp_path)
    resumed.run(steps)
    assert calls == []
    assert all(
        resumed.state["steps"][step.name]["status"] == "completed" for step in steps
    )
