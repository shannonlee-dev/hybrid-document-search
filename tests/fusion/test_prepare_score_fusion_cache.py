"""Cache preparation must verify inputs before installing any files."""

import hashlib
import io
import tarfile
from urllib.error import HTTPError

import pytest

from scripts import prepare_score_fusion_cache as cache


def _digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "miracl10k").mkdir()
    (source / "miracl10k/data.json").write_bytes(b'{"name":"fixture"}')
    spec = {
        "files": {"miracl10k/data.json": _digest(b'{"name":"fixture"}')},
        "repository": "example/repo",
        "tag": "inputs-v1",
        "asset": "inputs.tar.gz",
    }
    archive = tmp_path / spec["asset"]
    cache._pack_archive(source, archive, spec)
    spec["archive_sha256"] = cache._sha256(archive)
    return source, archive, spec


def test_local_install_and_idempotent_reuse(bundle, tmp_path, monkeypatch):
    source, archive, spec = bundle
    target = tmp_path / "cache"
    cache._prepare(target, spec, archive)
    assert (target / "miracl10k/data.json").read_bytes() == (
        source / "miracl10k/data.json"
    ).read_bytes()
    monkeypatch.setattr(
        cache, "urlopen", lambda *a, **kw: pytest.fail("downloaded existing cache")
    )
    cache._prepare(target, spec)


def test_download_install(bundle, tmp_path, monkeypatch):
    _, archive, spec = bundle
    requests = []

    def _download(url, **kwargs):
        requests.append(url)
        return io.BytesIO(archive.read_bytes())

    monkeypatch.setattr(cache, "urlopen", _download)
    target = tmp_path / "downloaded"
    cache._prepare(target, spec)
    assert requests == [
        "https://github.com/example/repo/releases/download/inputs-v1/inputs.tar.gz"
    ]
    cache._verify_files(target, spec)


def test_corrupt_archive_leaves_destination_absent(bundle, tmp_path):
    _, archive, spec = bundle
    archive.write_bytes(b"corrupt")
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="SHA-256"):
        cache._prepare(target, spec, archive)
    assert not target.exists()


def test_wrong_file_hash_leaves_destination_absent(bundle, tmp_path):
    _, archive, spec = bundle
    spec["files"]["miracl10k/data.json"] = "0" * 64
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="SHA-256"):
        cache._prepare(target, spec, archive)
    assert not target.exists()


def test_existing_invalid_cache_is_preserved(bundle, tmp_path):
    _, archive, spec = bundle
    target = tmp_path / "cache"
    target.mkdir()
    (target / "my-file").write_text("keep")
    with pytest.raises((ValueError, FileNotFoundError)):
        cache._prepare(target, spec, archive)
    assert (target / "my-file").read_text() == "keep"


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate"])
def test_unsafe_archive_is_rejected(bundle, tmp_path, kind):
    _, archive, spec = bundle
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo(
            "../escape" if kind == "traversal" else "miracl10k/data.json"
        )
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
        output.addfile(member)
        if kind == "duplicate":
            output.addfile(member)
    spec["archive_sha256"] = cache._sha256(archive)
    target = tmp_path / "cache"
    with pytest.raises(ValueError):
        cache._prepare(target, spec, archive)
    assert not target.exists()
    assert not (tmp_path / "escape").exists()


def test_missing_release_has_actionable_error(bundle, tmp_path, monkeypatch):
    _, _, spec = bundle

    def _missing(url, **kwargs):
        raise HTTPError(url, 404, "Not found", {}, None)

    monkeypatch.setattr(cache, "urlopen", _missing)
    with pytest.raises(ValueError, match="Release"):
        cache._prepare(tmp_path / "cache", spec)


def test_pack_is_reproducible(bundle, tmp_path):
    source, archive, spec = bundle
    other = tmp_path / "other.tar.gz"
    cache._pack_archive(source, other, spec)
    assert other.read_bytes() == archive.read_bytes()
