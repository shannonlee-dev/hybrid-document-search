"""Opt-in local model smoke test; never download or require weights in CI."""

import json
import math
import os
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import pytest


@pytest.mark.skipif(
    not os.environ.get("DAT_JUDGE_MODEL"), reason="set DAT_JUDGE_MODEL for local smoke"
)
def test_local_qwen_loads_and_returns_two_yes_probabilities(tmp_path):
    import torch

    from evaluation.contextual_execution import snapshot_identity, source_identity
    from evaluation.data import file_sha256
    from fusion.contextual_judge import LocalRelevanceJudge

    model = Path(os.environ["DAT_JUDGE_MODEL"])
    device = os.environ.get("DAT_JUDGE_DEVICE", "cpu")
    query = "신용카드 분실 시 어떻게 신고하나요?"
    bodies = [
        "신용카드를 분실하면 카드사 고객센터 또는 모바일 앱에서 즉시 분실 신고하세요.",
        "오늘 점심 메뉴는 비빔밥과 된장국입니다.",
    ]
    started = perf_counter()
    judge = LocalRelevanceJudge(str(model), device=device)
    loaded_seconds = perf_counter() - started
    observed = {}

    def observe_forward(module, args, kwargs, output):
        logits = output.logits
        observed.update(
            input_shape=list(kwargs["input_ids"].shape),
            logits_shape=list(logits.shape),
            logits_dtype=str(logits.dtype),
            logits_to_keep=kwargs.get("logits_to_keep"),
            use_cache=kwargs["use_cache"],
            logits_no_yes=logits[:, -1, [judge.no, judge.yes]].float().cpu().tolist(),
        )

    handle = judge.model.register_forward_hook(observe_forward, with_kwargs=True)
    try:
        started = perf_counter()
        probabilities = judge(query, bodies)
        inference_seconds = perf_counter() - started
    finally:
        handle.remove()
    assert len(probabilities) == 2
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    # A smoke fixture, not a BC Card quality threshold or tuning signal.
    assert probabilities[0] > probabilities[1]
    assert judge.pairs == 2 and judge.truncated_pairs == 0
    assert observed["use_cache"] is False
    assert len(observed["logits_no_yes"]) == 2
    report = {
        "status": "passed",
        "scope": "local Qwen load, tokenization, forward, last-token yes/no softmax",
        "executed_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "model_path": str(model),
        "model_identity": snapshot_identity(model),
        "source_sha256": source_identity(),
        "test_sha256": file_sha256(Path(__file__)),
        "environment": {
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "device": device,
            "gpu": torch.cuda.get_device_name(device)
            if device.startswith("cuda")
            else None,
            "model_dtype": str(judge.model.dtype),
            "local_files_only": True,
        },
        "load_seconds": loaded_seconds,
        "inference_seconds": inference_seconds,
        "query": query,
        "bodies_dense_sparse": bodies,
        "token_ids": {"no": judge.no, "yes": judge.yes},
        "forward": observed,
        "probabilities_dense_sparse": probabilities,
        "pairs": judge.pairs,
        "truncated_pairs": judge.truncated_pairs,
    }
    output = Path(
        os.environ.get("DAT_JUDGE_SMOKE_OUTPUT", tmp_path / "judge-smoke.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
