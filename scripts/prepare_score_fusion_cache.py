"""Prepare pinned benchmark inputs using only the Python standard library."""

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "docs/experiments/score-fusion/cache-release.json"
_DEFAULT_CACHE = _ROOT / "artifacts/score-fusion-cache"


def _prepare(directory, manifest, archive=None):
    """Download, verify and install a cache; preserve any existing directory."""
    if directory.exists() or directory.is_symlink():
        _verify_files(directory, manifest)
        print(f"Cache already verified: {directory}")
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".score-cache-", dir=directory.parent
    ) as tmp:
        staging = Path(tmp)
        if archive is None:
            archive = staging / manifest["asset"]
            repository = manifest["repository"]
            tag = quote(manifest["tag"], safe="")
            asset = quote(manifest["asset"], safe="")
            url = f"https://github.com/{repository}/releases/download/{tag}/{asset}"
            print(f"Downloading {manifest['tag']}/{manifest['asset']} ...")
            try:
                with urlopen(url, timeout=60) as source, archive.open("xb") as output:
                    shutil.copyfileobj(source, output)
            except HTTPError as exc:
                if exc.code in (401, 403, 404):
                    raise ValueError(
                        "Release asset is unavailable. Publish the pinned asset or "
                        "download it with repository access and use --archive FILE."
                    ) from exc
                raise
        _check_hash(archive, manifest["archive_sha256"])
        extracted = staging / "inputs"
        extracted.mkdir()
        _unpack(archive, extracted, manifest)
        if directory.exists() or directory.is_symlink():
            raise ValueError(f"Destination appeared during preparation: {directory}")
        extracted.rename(directory)
    print(f"Cache verified and ready: {directory}")


def _pack_archive(directory, output, manifest):
    """Create an exclusive, deterministic archive of the verified input files."""
    _verify_files(directory, manifest)
    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in sorted(manifest["files"]):
                    path = directory / name
                    member = tarfile.TarInfo(name)
                    member.size = path.stat().st_size
                    member.mode = 0o644
                    with path.open("rb") as source:
                        archive.addfile(member, source)


def _verify_files(directory, manifest):
    """Verify every pinned input, including the encoding records."""
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid manifest path: {name}")
        path = directory / relative
        if any(parent.is_symlink() for parent in [path, *path.parents]):
            raise ValueError(f"Symlink is not a cache input: {path}")
        _check_hash(path, expected)


def _unpack(archive_path, directory, manifest):
    # 허용된 일반 파일만 복사하며 아카이브의 링크와 경로 속성은 적용하지 않는다.
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if (
                member.name not in manifest["files"]
                or member.name in seen
                or not member.isfile()
            ):
                raise ValueError(f"Unexpected archive member: {member.name}")
            seen.add(member.name)
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid archive path: {member.name}")
            output = directory / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, output.open("xb") as target:
                shutil.copyfileobj(source, target)
    if seen != set(manifest["files"]):
        raise ValueError("Archive is missing required input files")
    _verify_files(directory, manifest)


def _check_hash(path, expected):
    if _sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _sha256(path):
    """Hash a file without loading the vectors into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main(argv=None):
    """Prepare inputs, or package the local cache for the pinned Release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--archive", type=Path, help="Use a downloaded local archive")
    action.add_argument("--pack", type=Path, help="Create the pinned Release archive")
    action.add_argument(
        "--verify-only", action="store_true", help="Verify local inputs"
    )
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(_MANIFEST.read_text())
        if args.pack:
            _pack_archive(args.cache_dir, args.pack, manifest)
            _check_hash(args.pack, manifest["archive_sha256"])
            print(f"Release archive verified: {args.pack}")
        elif args.verify_only:
            _verify_files(args.cache_dir, manifest)
            print(f"Cache verified: {args.cache_dir}")
        else:
            _prepare(args.cache_dir, manifest, args.archive)
    except (OSError, ValueError, tarfile.TarError, URLError) as exc:
        parser.exit(1, f"Cache preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
