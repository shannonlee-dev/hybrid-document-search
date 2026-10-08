"""Verify offline Dense build/search commands and restoration after restart."""

import builtins
import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from retrievers.model_config import MODELS


@pytest.mark.parametrize("model_name,pinned_revision", MODELS.items())
@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_build_persists_registered_model_revision(
    model_name, pinned_revision, revision, dense_corpus_path, model_stub, tmp_path
):
    pytest.importorskip("faiss")
    from scripts.build_index import main

    index = tmp_path / "index"
    arguments = [
        "--corpus",
        str(dense_corpus_path),
        "--index",
        str(index),
        "--model",
        model_name,
    ]
    if revision is not None:
        arguments.extend(["--revision", revision])
    main(arguments)
    metadata = json.loads((index / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["embedding_config"]["model_name"] == model_name
    assert metadata["embedding_config"]["revision"] == (
        pinned_revision if revision is None else revision
    )


def test_build_persists_requested_revision(
    dense_corpus_path, model_stub, tmp_path, capsys
):
    pytest.importorskip("faiss")
    from scripts.build_index import main

    index = tmp_path / "index"
    main(
        [
            "--corpus",
            str(dense_corpus_path),
            "--index",
            str(index),
            "--model",
            "BAAI/bge-m3",
            "--revision",
            "a" * 40,
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["documents"] == 3
    metadata = json.loads((index / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["embedding_config"]["revision"] == "a" * 40


def test_backend_dependencies_are_independent(
    monkeypatch, dense_documents, model_stub, tmp_path, capsys
):
    pytest.importorskip("faiss")
    from retrievers.dense import DenseRetriever

    retriever = DenseRetriever.build(dense_documents)
    retriever.save(tmp_path / "index")
    original_import = builtins.__import__

    def _isolated_import(name, *args, **kwargs):
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in ("retrievers.tfidf", "sklearn")
        ):
            raise AssertionError(f"Unexpected backend import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _isolated_import)
    import scripts.search as search

    importlib.reload(search)
    assert (
        search.main(["dense", "--index", str(tmp_path / "index"), "--query", "고양이"])
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)
    assert captured.err == ""


def test_internal_non_string_query_is_not_hidden(
    dense_documents, model_stub, tmp_path, capsys
):
    pytest.importorskip("faiss")
    from retrievers.dense import DenseRetriever
    from scripts.search import _build_parser

    retriever = DenseRetriever.build(dense_documents)
    retriever.save(tmp_path / "index")
    args = _build_parser().parse_args(
        ["dense", "--index", str(tmp_path / "index"), "--query", "고양이"]
    )
    args.query = None
    with pytest.raises(TypeError, match="query"):
        args.run(args)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("failure", ["metadata", "index", "model", "permission"])
def test_dense_failures_keep_stage_specific_guidance(
    failure, dense_documents, model_stub, tmp_path, monkeypatch, capsys
):
    pytest.importorskip("faiss")
    from retrievers.dense import DenseEmbedder, DenseRetriever
    from scripts.search import main

    index = tmp_path / "index with spaces"
    retriever = DenseRetriever.build(dense_documents)
    retriever.save(index)
    if failure == "metadata":
        (index / "metadata.json").write_text("{broken", encoding="utf-8")
    elif failure == "index":
        (index / "index.faiss").write_bytes(b"broken index")
    elif failure == "model":

        def _fail_encode(*args, **kwargs):
            raise RuntimeError("offline model failure")

        monkeypatch.setattr(DenseEmbedder, "_encode", _fail_encode)
    else:

        def _fail_load(*args, **kwargs):
            raise PermissionError("index access denied")

        monkeypatch.setattr(DenseRetriever, "load", _fail_load)

    with pytest.raises(SystemExit) as exc:
        main(["dense", "--index", str(index), "--query", "고양이"])
    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "python -m scripts.search dense --help" in captured.err
    assert captured.out == ""
    assert "Traceback" not in captured.err
    if failure in {"metadata", "index"}:
        assert "인덱스를 읽을 수 없습니다" in captured.err
        assert str(index) in captured.err
        assert "올바른 --index 경로" in captured.err
        assert "python -m scripts.build_index" in captured.err
        assert "상세:" in captured.err
    elif failure == "model":
        assert "모델 경로·캐시" in captured.err
        assert "다운로드 연결" in captured.err
        assert "--device cpu" in captured.err
        assert "offline model failure" in captured.err
        assert "python -m scripts.build_index" not in captured.err
    else:
        assert "권한" in captured.err
        assert "index access denied" in captured.err


def test_build_prints_new_shell_safe_search_command(
    dense_corpus_path, model_stub, tmp_path, capsys
):
    pytest.importorskip("faiss")
    from scripts.build_index import main

    index = tmp_path / "index with spaces '$(echo unsafe)'"
    main(["--corpus", str(dense_corpus_path), "--index", str(index)])
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"index": str(index), "documents": 3}
    command = next(
        line.strip()
        for line in captured.err.splitlines()
        if line.strip().startswith("python -m scripts.search ")
    )
    assert shlex.split(command) == [
        "python",
        "-m",
        "scripts.search",
        "dense",
        "--index",
        str(index),
        "--query",
        "검색어",
    ]


def test_dense_cli_returns_shared_search_result(
    dense_documents, model_stub, tmp_path, capsys
):
    pytest.importorskip("faiss")
    from retrievers.base import SearchResult
    from retrievers.dense import DenseRetriever
    from scripts.search import main

    retriever = DenseRetriever.build(dense_documents)
    retriever.save(tmp_path / "index")

    main(
        [
            "dense",
            "--index",
            str(tmp_path / "index"),
            "--query",
            "고양이",
            "--top-k",
            "1",
        ]
    )

    captured = capsys.readouterr()
    results = json.loads(captured.out)
    assert results == [
        {
            "document_id": "cat",
            "rank": 1,
            "score": 1.0,
            "title": "고양이",
            "snippet": "고양이 울음소리는 야옹 입니다.",
        }
    ]
    assert SearchResult(**results[0]).document_id == "cat"
    assert captured.err == ""


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_cli_returns_json_list(
    query, dense_documents, model_stub, tmp_path, capsys
):
    pytest.importorskip("faiss")
    from retrievers.dense import DenseConfig, DenseRetriever
    from scripts.search import main

    # 빈 질의의 검색은 모델 파일 없이도 동작해야 한다.
    retriever = DenseRetriever.build(
        dense_documents, DenseConfig(model_name=str(tmp_path / "missing-model"))
    )
    retriever.save(tmp_path / "index")

    main(["dense", "--index", str(tmp_path / "index"), "--query", query])

    captured = capsys.readouterr()
    assert json.loads(captured.out) == []
    assert captured.err == ""


def test_dense_cli_build_and_restart(dense_corpus_path, tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    st = pytest.importorskip("sentence_transformers")
    from sentence_transformers.sentence_transformer.modules import BoW

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    model_path = tmp_path / "model"
    # 로컬 BoW 모델로 다운로드 없이 실제 임베딩 API를 검증한다.
    model = st.SentenceTransformer(
        modules=[BoW(["고양이", "야옹", "강아지", "멍멍", "문서", "검색"])],
        device="cpu",
    )
    model.save(str(model_path))
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    build_command = [sys.executable, "-m", "scripts.build_index"]
    search_command = [sys.executable, "-m", "scripts.search", "dense"]
    index = tmp_path / "index"
    common_args = ["--index", str(index), "--device", "cpu"]
    model_args = ["--model", str(model_path)]

    query_args = ["--query", "고양이", "--top-k", "2"]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(dense_corpus_path.read_bytes())
    build = subprocess.run(
        [
            *build_command,
            *common_args,
            *model_args,
            "--corpus",
            str(corpus),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    assert json.loads(build.stdout) == {"index": str(index), "documents": 3}
    # corpus 없이 재시작해 저장된 문서 매핑만 사용하는지 확인한다.
    corpus.unlink()
    search = subprocess.run(
        [*search_command, *common_args, *query_args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    after = json.loads(search.stdout)
    assert len(after) == 2
    assert after[0]["document_id"] == "cat"
    assert after[0]["rank"] == 1
    assert after[0]["title"] == "고양이"
    assert set(after[0]) == {"document_id", "rank", "score", "title", "snippet"}
    assert after[0]["snippet"] == "고양이 울음소리는 야옹 입니다."
    # 제목의 고양이 토큰도 반영되므로 cosine([2, 1], [1, 0]) = 2 / sqrt(5)이다.
    assert after[0]["score"] == pytest.approx(2 / 5**0.5)

    missing_dependency = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "scripts.search",
            "dense",
            *common_args,
            *query_args,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert missing_dependency.returncode == 2
    assert "uv sync --locked --extra dense" in missing_dependency.stderr
    assert "Traceback" not in missing_dependency.stderr
    assert missing_dependency.stdout == ""

    metadata_path = index / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["embedding_config"]["model_name"] = str(tmp_path / "missing-model")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    missing_model = subprocess.run(
        [*search_command, *common_args, *query_args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert missing_model.returncode == 2
    assert "모델 경로·캐시" in missing_model.stderr
    assert "python -m scripts.build_index" not in missing_model.stderr
    assert "Traceback" not in missing_model.stderr
    assert missing_model.stdout == ""

    (index / "metadata.json").write_text("{broken", encoding="utf-8")
    corrupt = subprocess.run(
        [*search_command, *common_args, *query_args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert corrupt.returncode == 2
    assert "인덱스를 읽을 수 없습니다" in corrupt.stderr
    assert "python -m scripts.build_index" in corrupt.stderr
    assert "Traceback" not in corrupt.stderr
    assert corrupt.stdout == ""


@pytest.mark.parametrize("contents", [b"{broken", b"", b"\xff"])
def test_invalid_corpus_content_shows_input_guidance(contents, tmp_path):
    root = Path(__file__).resolve().parents[2]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(contents)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_index",
            "--corpus",
            str(corpus),
            "--index",
            str(tmp_path / "index"),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "UTF-8 JSONL" in result.stderr
    assert "document_id" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "index").exists()
