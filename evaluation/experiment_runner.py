"""Small file-checksummed stage runner for the owned Ko-MIRACL workspace."""

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from evaluation.data import file_sha256
from evaluation.json_io import write_json

MARKER = ".ko-miracl-experiment"
_STAGE_NAME_PATTERN = r"[a-z0-9_-]+"


@dataclass
class Step:
    """Stage action with dependencies and inputs used to decide whether to rerun."""

    name: str
    dependencies: tuple[str, ...]
    inputs: dict
    action: Callable[[Path], None]
    can_reuse: Callable[[Path], bool] | None = None


class Experiment:
    """Resume verified stages within a marked experiment directory."""

    def __init__(self, root: Path):
        reject_symlinks(root)
        self.root = root.absolute()
        marker = self.root / MARKER
        if self.root.exists() and any(self.root.iterdir()) and not marker.is_file():
            raise ValueError(f"not an owned experiment directory: {root}")
        self.root.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(str(uuid4()) + "\n")
        self.identifier = marker.read_text().strip()
        self.checkpoint = self.root / "checkpoint.json"
        if self.checkpoint.exists():
            self.state = json.loads(self.checkpoint.read_text())
        else:
            self.state = {"experiment_id": self.identifier, "steps": {}}
        if self.state["experiment_id"] != self.identifier:
            raise ValueError("checkpoint experiment identity mismatch")

    def fresh(self):
        """Remove the owned workspace and create a new experiment identity."""
        reject_symlinks(self.root)
        if (self.root / MARKER).read_text().strip() != self.identifier:
            raise ValueError("not an owned experiment directory")
        shutil.rmtree(self.root)
        self.__init__(self.root)

    def run(self, steps: list[Step]):
        """Run stages in dependency order, reusing outputs whose hashes still match.

        Persist failed stages before propagating errors so later runs can resume.
        """
        seen = set()
        for step in steps:
            if not re.fullmatch(_STAGE_NAME_PATTERN, step.name) or step.name in seen:
                raise ValueError("invalid or duplicate stage name")
            if not set(step.dependencies).issubset(seen):
                raise ValueError("stages must follow dependency order")
            seen.add(step.name)
        for step in steps:
            dependencies = {}
            for name in step.dependencies:
                dependency = self.state["steps"][name]
                # 출력이 같아도 선행 단계를 다시 실행했다면 후속 단계도 갱신한다.
                dependencies[name] = {
                    "generation": dependency["generation"],
                    "outputs": dependency["outputs"],
                }
            inputs = {"configuration": step.inputs, "dependencies": dependencies}
            destination = self.root / step.name
            previous = self.state["steps"].get(step.name, {})
            # 다운로드 단계는 기록된 출력뿐 아니라 외부 모델 캐시도 남아 있어야 한다.
            if (
                previous.get("status") == "completed"
                and previous.get("inputs") == inputs
                and previous.get("outputs")
                and hashes(destination) == previous["outputs"]
                and (step.can_reuse is None or step.can_reuse(destination))
            ):
                print(f"verified: {step.name}", flush=True)
                continue
            row = {
                "stage": step.name,
                "status": "running",
                "inputs": inputs,
                "generation": str(uuid4()),
                "started_at": datetime.now(UTC).isoformat(),
                "output_directory": str(destination),
            }
            self.state["steps"][step.name] = row
            write_json(self.checkpoint, self.state)
            print(f"running: {step.name}", flush=True)
            try:
                reject_symlinks(destination)
                if destination.exists():
                    shutil.rmtree(destination)
                destination.mkdir()
                step.action(destination)
                outputs = hashes(destination)
                if not outputs:
                    raise ValueError(f"empty stage output: {step.name}")
                row.update(
                    status="completed",
                    outputs=outputs,
                    completed_at=datetime.now(UTC).isoformat(),
                    error=None,
                )
            except BaseException as exc:
                # 사용자 중단도 실패로 기록한 뒤 다시 전파해 재개 지점을 보존한다.
                row.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    failed_at=datetime.now(UTC).isoformat(),
                )
                write_json(self.checkpoint, self.state)
                raise
            write_json(self.checkpoint, self.state)


def hashes(directory: Path):
    """Return file hashes keyed by relative path, rejecting symlinks."""
    reject_symlinks(directory)
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def reject_symlinks(path: Path):
    """Reject symlinks in the path, its ancestors and its existing directory tree."""
    path = path.absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"symlink is forbidden: {parent}")
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"symlink is forbidden: {child}")
