"""Experiment preflight accepts different environments while requiring CUDA."""

import runpy
import sys

import pytest

from scripts import run_experiment


def test_module_loads_without_fcntl(monkeypatch):
    monkeypatch.setitem(sys.modules, "fcntl", None)

    module = runpy.run_path(run_experiment.__file__)

    assert callable(module["main"])


@pytest.mark.parametrize(
    "python_version, kernel, gpu",
    [
        ((3, 13), "6.6.87-microsoft-standard-WSL2", "NVIDIA GeForce RTX 4060"),
        ((3, 12), "6.8.0-generic", "NVIDIA GeForce RTX 4060"),
        ((3, 12), "6.6.87-microsoft-standard-WSL2", "NVIDIA GeForce RTX 4090"),
    ],
)
def test_environment_accepts_cuda_without_hardware_or_os_restrictions(
    monkeypatch, python_version, kernel, gpu
):
    monkeypatch.setattr(run_experiment.sys, "version_info", python_version)
    monkeypatch.setattr(run_experiment.platform, "release", lambda: kernel)
    monkeypatch.setattr(run_experiment.platform, "platform", lambda: kernel)
    monkeypatch.setattr(
        run_experiment.platform,
        "python_version",
        lambda: f"{python_version[0]}.{python_version[1]}.0",
    )
    monkeypatch.setattr(
        run_experiment.importlib.metadata, "version", lambda name: "fixture-version"
    )

    def configure_runtime(threads, *, cuda):
        if not cuda:
            raise RuntimeError("CUDA must be required for the experiment")
        return {"gpu": gpu, "cuda": "12.8", "device": "cuda:0"}

    monkeypatch.setattr(run_experiment, "configure_runtime", configure_runtime)

    environment = run_experiment._environment()

    assert environment["python"] == f"{python_version[0]}.{python_version[1]}.0"
    assert environment["platform"] == kernel
    assert environment["gpu"] == gpu
    assert environment["cuda"] == "12.8"


def test_environment_does_not_fall_back_when_cuda_is_unavailable(monkeypatch):
    def configure_runtime(threads, *, cuda):
        if cuda:
            raise RuntimeError("cuda:0 is required; CPU fallback is forbidden")
        return {"device": "cpu"}

    monkeypatch.setattr(run_experiment, "configure_runtime", configure_runtime)

    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        run_experiment._environment()
