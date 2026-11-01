"""Train and encode query models against fixed, validated document vectors."""

import hashlib
import json
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

from data.bccard import parent_and_index
from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.json_io import read_json, write_json
from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from retrievers.base import SearchResult

PRIMARY_PROVENANCE = {"base", "faq_direct"}
CHECKPOINT_STEPS = (64, 128, 256, 512, 1024, 2048, 4096)


LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.05
TETHER_WEIGHT = 1.0
HARD_NEGATIVES = 8


def configure_trainable_blocks(model, blocks, last_n):
    """Freeze the model and enable only its last N supplied transformer blocks.

    Return the selected module slice, preserving archived ModuleList state keys.
    """
    if (
        isinstance(last_n, bool)
        or not isinstance(last_n, int)
        or not 1 <= last_n <= len(blocks)
    ):
        raise ValueError("last_n must select between one and all transformer blocks")
    selected = blocks[-last_n:] if last_n > 1 else blocks[-1]
    model.float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in selected.parameters():
        parameter.requires_grad_(True)
    return selected


def contrastive_candidates(
    baseline,
    documents,
    positive_ids,
    query_parents,
    document_parents,
    *,
    hard_negatives=HARD_NEGATIVES,
):
    """Mine frozen-baseline hard negatives and mask same-parent candidates.

    Every batch positive is a candidate for every query. Each query retains its
    designated positive even when other batch positives share the same parent.
    """
    import torch

    if (
        isinstance(hard_negatives, bool)
        or not isinstance(hard_negatives, int)
        or hard_negatives < 1
    ):
        raise ValueError("hard_negatives must be a positive integer")
    if (
        baseline.ndim != 2
        or documents.ndim != 2
        or baseline.shape[1] != documents.shape[1]
    ):
        raise ValueError("baseline and documents must have matching vector dimensions")
    batch, count = len(baseline), len(documents)
    if (
        positive_ids.shape != (batch,)
        or query_parents.shape != (batch,)
        or document_parents.shape != (count,)
    ):
        raise ValueError("positive and parent arrays must match query/document rows")
    if ((positive_ids < 0) | (positive_ids >= count)).any():
        raise ValueError("positive IDs must be corpus row indices")
    if not torch.equal(document_parents[positive_ids], query_parents):
        raise ValueError("positive documents must have the query parent")
    with torch.no_grad():
        same_parent = query_parents[:, None] == document_parents[None, :]
        if ((~same_parent).sum(1) < hard_negatives).any():
            raise ValueError("not enough different-parent documents for hard negatives")
        scores = baseline.detach() @ documents.detach().T
        scores.masked_fill_(same_parent, -torch.inf)
        hard = scores.topk(hard_negatives, dim=1).indices
        candidates = torch.cat((positive_ids, hard.flatten()))
        forbidden = query_parents[:, None] == document_parents[candidates][None, :]
        indices = torch.arange(batch, device=baseline.device)
        forbidden[indices, indices] = False
    return candidates, forbidden


def query_training_step(
    model,
    features,
    optimizer,
    documents,
    baseline,
    positive_ids,
    query_parents,
    document_parents,
    *,
    hard_negatives=HARD_NEGATIVES,
    temperature=TEMPERATURE,
    tether_weight=TETHER_WEIGHT,
):
    """Perform one AdamW-compatible update with contrastive and tether losses.

    Parameters remain FP32. CUDA forwards use BF16 autocast; CPU fixtures use
    FP32. Document and baseline vectors never receive gradients.
    """
    import torch
    import torch.nn.functional as functional

    if temperature <= 0 or tether_weight < 0:
        raise ValueError("temperature must be positive and tether weight nonnegative")
    candidates, forbidden = contrastive_candidates(
        baseline,
        documents,
        positive_ids,
        query_parents,
        document_parents,
        hard_negatives=hard_negatives,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if documents.device.type == "cuda"
        else nullcontext()
    )
    with context:
        vector = functional.normalize(
            model(features)["sentence_embedding"].float(), dim=-1
        )
    logits = vector @ documents.detach()[candidates].T / temperature
    logits = logits.masked_fill(forbidden, -torch.inf)
    target = torch.arange(len(vector), device=vector.device)
    loss = functional.cross_entropy(logits, target)
    loss = loss + tether_weight * (1 - (vector * baseline.detach()).sum(-1)).mean()
    if not torch.isfinite(loss):
        raise ValueError("non-finite training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        1.0,
    )
    optimizer.step()
    return float(loss.detach())


def load_query_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def eligible_training_indices(train, heldout):
    """Exclude held-out text duplicates and require disjoint parent documents.

    Held-out query texts/parent metadata are used only for contamination checks;
    held-out relevance labels are not accepted by this function.
    """
    import numpy as np

    def normalize(text):
        return " ".join(text.casefold().split())

    texts = {normalize(query["text"]) for query in heldout}
    eligible = [
        i for i, query in enumerate(train) if normalize(query["text"]) not in texts
    ]
    if not eligible:
        raise ValueError("no eligible training queries after duplicate exclusion")
    parents = {query["parent_doc_id"] for query in heldout}
    if parents & {train[i]["parent_doc_id"] for i in eligible}:
        raise ValueError("training and held-out parent documents overlap")
    return np.asarray(eligible, dtype=np.int64)


def training_data_audit(data_dir):
    """Recheck duplicate exclusion with held-out texts/parents, never test qrels."""
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    queries = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"queries_{split}.jsonl"
        if file_sha256(path) != manifest["output_sha256"][path.name]:
            raise ValueError("query checksum mismatch in training exclusion audit")
        queries[split] = load_query_rows(path)
    eligible = eligible_training_indices(
        queries["train"], queries["validation"] + queries["test"]
    )
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if {q["query_id"] for q in queries[left]} & {
            q["query_id"] for q in queries[right]
        }:
            raise ValueError("query IDs overlap between official splits")
    return {
        "train_examples": len(queries["train"]),
        "eligible_train_examples": len(eligible),
        "excluded_exact_duplicate_examples": len(queries["train"]) - len(eligible),
        "normalization": "Unicode casefold; collapse whitespace",
        "heldout_scope": "all validation/test query texts; qrels not used",
        "parent_documents_disjoint": True,
        "query_ids_disjoint": True,
    }


def checkpoint_selection(history):
    """Choose strictly better validation nDCG; step-zero baseline wins ties."""
    from math import isfinite

    if not history or history[0].get("step") != 0:
        raise ValueError("checkpoint history must start with the baseline at step zero")
    steps = [entry["step"] for entry in history]
    if steps != sorted(set(steps)) or any(step < 0 for step in steps):
        raise ValueError("checkpoint steps must be unique and increasing")
    if any(not isfinite(entry["validation_ndcg"]) for entry in history):
        raise ValueError("validation scores must be finite")
    return max(history, key=lambda entry: entry["validation_ndcg"])


def ranking_quality(ranking, documents, judgments):
    results = [
        SearchResult(document_id=documents[index].document_id, rank=rank, score=0.0)
        for rank, index in enumerate(ranking, 1)
    ]
    return [
        ndcg_at_k(results, judgments, 10),
        precision_at_k(results, judgments, 1),
        recall_at_k(results, judgments, 10),
    ]


def load_encoder_inputs(data_dir, cache_dir, splits):
    """Verify corpus/query row identity before using cached vectors."""
    import hashlib

    import numpy as np

    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    encoding = read_json(cache_dir / "encoding.json")
    if encoding["dataset_manifest_sha256"] != file_sha256(data_dir / "manifest.json"):
        raise ValueError("cache dataset manifest mismatch")
    datasets = {
        split: load_evaluation_dataset(data_dir, split, verify_all_splits=False)
        for split in splits
    }
    documents = next(iter(datasets.values())).documents
    digest = hashlib.sha256(
        "\n".join(doc.document_id for doc in documents).encode()
    ).hexdigest()
    if encoding["document_ids_sha256"] != digest:
        raise ValueError("cache document order mismatch")
    if (
        file_sha256(cache_dir / "document-vectors.npy")
        != encoding["vector_sha256"]["document-vectors.npy"]
    ):
        raise ValueError("document vector checksum mismatch")
    vectors = np.load(
        cache_dir / "document-vectors.npy", mmap_mode="r", allow_pickle=False
    )
    if (
        vectors.ndim != 2
        or len(vectors) != len(documents)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("document vectors do not match corpus")
    queries, qrels = {}, {}
    for split in splits:
        path = data_dir / f"queries_{split}.jsonl"
        query_objects = datasets[split].queries
        queries[split] = load_query_rows(path)
        if [query.query_id for query in query_objects] != encoding["splits"][split][
            "query_ids"
        ]:
            raise ValueError("cache query order mismatch")
        qrels[split] = datasets[split].qrels
    return documents, vectors, queries, qrels


def encode_queries(model, texts, *, device="cpu", batch_size=16, max_tokens=512):
    """Encode normalized FP32 outputs, using BF16 forward autocast on CUDA."""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    if not texts or batch_size < 1 or max_tokens < 1:
        raise ValueError("texts must be nonempty and encoding limits positive")
    model.eval()
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else nullcontext()
    )
    chunks = []
    with torch.inference_mode(), context:
        for start in range(0, len(texts), batch_size):
            tokens = model.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            features = {key: value.to(device) for key, value in tokens.items()}
            chunks.append(
                functional.normalize(
                    model(features)["sentence_embedding"].float(), dim=-1
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(chunks)


def _dense_quality(vectors, documents, document_vectors, queries, qrels):
    import numpy as np

    quality = []
    for start in range(0, len(vectors), 64):
        scores = vectors[start : start + 64] @ document_vectors.T
        for row, score in enumerate(scores, start):
            ranking = np.argsort(-score, kind="stable")[:10]
            quality.append(
                ranking_quality(ranking, documents, qrels[queries[row]["query_id"]])
            )
    return np.asarray(quality)


def train_query_encoder(
    data_dir,
    cache_dir,
    model_path,
    output_dir,
    *,
    seed=42,
    steps=256,
    layers=4,
    batch_size=16,
    device="cuda",
):
    """Train the query-side final blocks; freeze validation choice before test IO.

    No test qrels or test document rankings are read during training/selection.
    The test query texts are consulted only to exclude exact training duplicates.
    Test encoding/evaluation is a separate CLI action after frozen.json exists.
    """
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    from evaluation.query_encoder_report import model_identity, training_run_provenance

    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    output_dir, data_dir, cache_dir = Path(output_dir), Path(data_dir), Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    documents, dvec, queries, qrels = load_encoder_inputs(
        data_dir, cache_dir, ("train", "validation")
    )
    heldout = queries["validation"] + load_query_rows(data_dir / "queries_test.jsonl")
    audit = training_data_audit(data_dir)
    eligible = eligible_training_indices(queries["train"], heldout)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = np.concatenate(
        [
            rng.permutation(eligible)
            for _ in range((steps * batch_size + len(eligible) - 1) // len(eligible))
        ]
    )[: steps * batch_size]
    baseline = np.load(cache_dir / "query-vectors-train.npy", allow_pickle=False)
    encoding = read_json(cache_dir / "encoding.json")
    if (
        file_sha256(cache_dir / "query-vectors-train.npy")
        != encoding["vector_sha256"]["query-vectors-train.npy"]
    ):
        raise ValueError("training query vector checksum mismatch")
    if (
        baseline.shape != (len(queries["train"]), dvec.shape[1])
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("baseline training query vectors do not match queries")
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    selected = configure_trainable_blocks(
        model, model[0].auto_model.encoder.layer, layers
    )
    optimizer = torch.optim.AdamW(
        selected.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    docs_tensor = torch.tensor(dvec, device=device)
    parents = [parent_and_index(doc.document_id)[0] for doc in documents]
    codes = {parent: index for index, parent in enumerate(sorted(set(parents)))}
    doc_parents = torch.tensor([codes[parent] for parent in parents], device=device)
    ids = {doc.document_id: index for index, doc in enumerate(documents)}
    primary = np.array(
        [query["provenance"] in PRIMARY_PROVENANCE for query in queries["validation"]]
    )
    if not primary.any():
        raise ValueError("validation requires base or faq_direct queries")
    write_json(
        output_dir / "protocol.json",
        {
            **training_run_provenance(output_dir.name, device),
            "seed": seed,
            "steps": steps,
            "trainable_layers": layers,
            "batch": batch_size,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "tether": TETHER_WEIGHT,
            "hard_negatives": HARD_NEGATIVES,
            "max_train_tokens": 128,
            "max_eval_tokens": 512,
            "precision": "FP32 parameters; CUDA BF16 autocast; CPU FP32",
            "model": str(model_path),
            "model_files_sha256": model_identity(model_path),
            "revision": read_json(cache_dir / "encoding.json")["model"]["revision"],
            "excluded_exact_duplicate_examples": len(queries["train"]) - len(eligible),
            "training_data_audit": audit,
            "train_examples": steps * batch_size,
            "selection": "validation base+faq_direct binary nDCG@10, baseline eligible, earliest tie",
            "test": "separate evaluate-checkpoint action after frozen selection",
            "input_sha256": {
                str(path): file_sha256(path)
                for path in (
                    data_dir / "manifest.json",
                    cache_dir / "encoding.json",
                    cache_dir / "document-vectors.npy",
                    cache_dir / "query-vectors-train.npy",
                    data_dir / "corpus.jsonl",
                    data_dir / "queries_train.jsonl",
                    data_dir / "qrels_train.jsonl",
                    data_dir / "queries_validation.jsonl",
                    data_dir / "qrels_validation.jsonl",
                    data_dir / "queries_test.jsonl",
                )
            },
        },
    )
    history = []
    checkpoints = {0, steps, *(step for step in CHECKPOINT_STEPS if step <= steps)}
    started = time.perf_counter()
    for step in range(steps + 1):
        if step in checkpoints:
            vectors = encode_queries(
                model, [query["text"] for query in queries["validation"]], device=device
            )
            quality = _dense_quality(
                vectors, documents, dvec, queries["validation"], qrels["validation"]
            )
            entry = {
                "step": step,
                "validation_ndcg": float(quality[primary, 0].mean()),
                "validation_all_ndcg": float(quality[:, 0].mean()),
                "seconds": time.perf_counter() - started,
            }
            history.append(entry)
            if checkpoint_selection(history) is entry:
                torch.save(selected.state_dict(), output_dir / "selected-last-block.pt")
                np.save(
                    output_dir / "selected-validation-vectors.npy",
                    vectors,
                    allow_pickle=False,
                )
            write_json(output_dir / "history.json", history)
        if step == steps:
            break
        indices = order[step * batch_size : (step + 1) * batch_size]
        batch = [queries["train"][index] for index in indices]
        positives = []
        for query in batch:
            positive = [
                ids[doc]
                for doc, grade in qrels["train"][query["query_id"]].items()
                if grade == 3
            ]
            if len(positive) != 1:
                raise ValueError(
                    "training requires exactly one relevance-three positive per query"
                )
            positives.append(positive[0])
        features = model.tokenizer(
            [query["text"] for query in batch],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        loss = query_training_step(
            model,
            {key: value.to(device) for key, value in features.items()},
            optimizer,
            docs_tensor,
            torch.tensor(baseline[indices], device=device),
            torch.tensor(positives, device=device),
            torch.tensor(
                [codes[query["parent_doc_id"]] for query in batch], device=device
            ),
            doc_parents,
        )
        if (step + 1) % 64 == 0 or step + 1 == steps:
            write_json(
                output_dir / "progress.json",
                {
                    "step": step + 1,
                    "loss": loss,
                    "seconds": time.perf_counter() - started,
                },
            )
    chosen = checkpoint_selection(history)
    frozen = {
        "run_id": output_dir.name,
        "protocol_sha256": file_sha256(output_dir / "protocol.json"),
        "selected_step": chosen["step"],
        "validation_primary_ndcg": chosen["validation_ndcg"],
        "checkpoint_sha256": file_sha256(output_dir / "selected-last-block.pt"),
        "trainable_parameters": sum(
            parameter.numel() for parameter in selected.parameters()
        ),
        "history": history,
        "validation_vectors_sha256": file_sha256(
            output_dir / "selected-validation-vectors.npy"
        ),
    }
    write_json(output_dir / "frozen.json", frozen)
    return frozen


def evaluate_checkpoint(
    data_dir,
    cache_dir,
    model_path,
    checkpoint_dir,
    output_dir,
    *,
    layers=4,
    device="cpu",
):
    """Evaluate a checksum-verified frozen checkpoint without reselecting it."""
    import shutil

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    from evaluation.query_encoder_report import model_identity

    checkpoint_dir, output_dir = Path(checkpoint_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError("checkpoint evaluation output already exists")
    frozen = read_json(checkpoint_dir / "frozen.json")
    protocol = read_json(checkpoint_dir / "protocol.json")
    if (
        frozen.get("protocol_sha256") is not None
        and file_sha256(checkpoint_dir / "protocol.json") != frozen["protocol_sha256"]
    ):
        raise ValueError("frozen training protocol checksum mismatch")
    if protocol["trainable_layers"] != layers:
        raise ValueError("layer count differs from frozen checkpoint training")
    checkpoint = checkpoint_dir / "selected-last-block.pt"
    if file_sha256(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint checksum mismatch")
    documents, dvec, queries, qrels = load_encoder_inputs(
        data_dir, cache_dir, ("test",)
    )
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    selected = configure_trainable_blocks(
        model, model[0].auto_model.encoder.layer, layers
    )
    initial = {
        key: value.detach().cpu().clone()
        for key, value in selected.state_dict().items()
    }
    selected.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    texts = [query["text"] for query in queries["test"]]
    adapted = encode_queries(model, texts, device=device)
    selected.load_state_dict(initial)
    baseline = encode_queries(model, texts, device=device)
    mask = np.array(
        [query["provenance"] in PRIMARY_PROVENANCE for query in queries["test"]]
    )
    if not mask.any():
        raise ValueError("test requires base or faq_direct queries")
    output_dir.mkdir(parents=True, exist_ok=False)
    quality = {}
    for name, vector in (("baseline", baseline), ("adapted", adapted)):
        quality[name] = _dense_quality(
            vector, documents, dvec, queries["test"], qrels["test"]
        )
        np.save(output_dir / f"{name}-test-vectors.npy", vector, allow_pickle=False)
    np.savez_compressed(output_dir / "test-quality.npz", **quality)
    report = {
        "test_status": "previously exposed official test; explicitly reused; no test selection",
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "selected_step": frozen["selected_step"],
        "precision": "matched BF16 CUDA forwards / FP32 parameters and storage; CPU uses FP32",
        "non_synthetic": {
            name: dict(
                zip(
                    ("ndcg@10", "p@1", "recall@10"),
                    values[mask].mean(0).tolist(),
                    strict=True,
                )
            )
            for name, values in quality.items()
        },
    }
    original_vectors = checkpoint_dir / "selected-test-vectors.npy"
    if original_vectors.is_file():
        reference = np.load(original_vectors, allow_pickle=False)
        if reference.shape != adapted.shape:
            raise ValueError("saved checkpoint query vector shape differs")
        cosine = (adapted * reference).sum(1) / (
            np.linalg.norm(adapted, axis=1) * np.linalg.norm(reference, axis=1)
        )
        report["saved_vector_parity"] = {
            "queries": len(reference),
            "reference_sha256": file_sha256(original_vectors),
            "minimum_cosine": float(cosine.min()),
            "maximum_absolute_difference": float(np.abs(adapted - reference).max()),
        }
    report["model_files_sha256"] = model_identity(model_path)
    if (
        protocol.get("model_files_sha256") is not None
        and protocol["model_files_sha256"] != report["model_files_sha256"]
    ):
        raise ValueError("base model snapshot differs from training")
    report["document_vectors_sha256"] = file_sha256(
        Path(cache_dir) / "document-vectors.npy"
    )
    report["device"] = str(device)
    report["torch"] = torch.__version__
    report["gpu"] = (
        torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None
    )
    for name in (
        "selected-last-block.pt",
        "frozen.json",
        "protocol.json",
        "selected-validation-vectors.npy",
    ):
        shutil.copyfile(checkpoint_dir / name, output_dir / name)
    np.save(output_dir / "selected-test-vectors.npy", adapted, allow_pickle=False)
    write_json(output_dir / "summary.json", report)
    return report


def prepare_baseline_vectors(
    data_dir, cache_dir, model_path, output_dir, *, test_source=None, device="cuda"
):
    """Encode original Validation with BF16 and reuse verified checkpoint Test output.

    No training or test qrels are consumed. Without test_source, encode original
    Test queries with the same batch/token/precision settings as Validation.
    """
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    from evaluation.query_encoder_report import model_identity

    if not str(device).startswith("cuda"):
        raise ValueError(
            "matched BF16 encoding requires CUDA; CPU fallback is forbidden"
        )
    data_dir, cache_dir, output_dir = map(Path, (data_dir, cache_dir, output_dir))
    if output_dir.exists():
        raise ValueError("baseline vector output already exists")
    encoding = read_json(cache_dir / "encoding.json")
    manifest = read_json(data_dir / "manifest.json")
    if file_sha256(data_dir / "manifest.json") != encoding["dataset_manifest_sha256"]:
        raise ValueError("baseline dataset identity mismatch")
    model_hashes = model_identity(model_path)
    document_hash = file_sha256(cache_dir / "document-vectors.npy")
    if document_hash != encoding["vector_sha256"]["document-vectors.npy"]:
        raise ValueError("document vector checksum mismatch")
    model = SentenceTransformer(
        str(model_path), device=device, local_files_only=True
    ).float()
    metadata = {
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "document_vectors_sha256": document_hash,
        "model": encoding["model"],
        "model_files_sha256": model_hashes,
        "query_encoding": {
            "forward_dtype": "bfloat16",
            "storage_dtype": "float32",
            "parameter_dtype": "float32",
            "max_tokens": 512,
            "batch_size": 16,
        },
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "threads": torch.get_num_threads(),
        },
        "splits": {},
    }
    output_dir.mkdir(parents=True)
    for split in ("validation", "test"):
        query_path = data_dir / f"queries_{split}.jsonl"
        if file_sha256(query_path) != manifest["output_sha256"][query_path.name]:
            raise ValueError("query checksum mismatch")
        queries = [json.loads(line) for line in query_path.read_text().splitlines()]
        ids = [q["query_id"] for q in queries]
        if ids != encoding["splits"][split]["query_ids"]:
            raise ValueError("baseline query order mismatch")
        path = output_dir / f"baseline-{split}-vectors.npy"
        if split == "test" and test_source is not None:
            source = Path(test_source)
            summary = read_json(source / "summary.json")
            training = read_json(source / "protocol.json")
            training_manifest = {
                Path(name).name: digest
                for name, digest in training["input_sha256"].items()
            }["manifest.json"]
            if (
                summary["model_files_sha256"] != model_hashes
                or summary["document_vectors_sha256"] != document_hash
                or not summary["device"].startswith("cuda")
                or "BF16 CUDA" not in summary["precision"]
                or training_manifest != encoding["dataset_manifest_sha256"]
                or training["max_eval_tokens"] != 512
            ):
                raise ValueError(
                    "checkpoint Test vectors do not match BF16 baseline inputs"
                )
            shutil.copyfile(source / path.name, path)
            metadata["test_source_summary_sha256"] = file_sha256(
                source / "summary.json"
            )
        else:
            vectors = encode_queries(
                model,
                [q["text"] for q in queries],
                device=device,
                batch_size=16,
                max_tokens=512,
            )
            np.save(path, vectors, allow_pickle=False)
        metadata["splits"][split] = {
            "query_ids": ids,
            "sha256": file_sha256(path),
            "query_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        }
    write_json(output_dir / "baseline-vectors.json", metadata)
    return {
        "splits": {
            key: len(value["query_ids"]) for key, value in metadata["splits"].items()
        },
        "query_encoding": metadata["query_encoding"],
    }
