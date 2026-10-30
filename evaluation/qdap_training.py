"""Full official Train fitting with checkpoint/readout selection on Validation."""

import math
import time

import torch

from evaluation.json_io import write_json
from evaluation.qdap import probability_alphas, rank_pools, ranking_metrics
from fusion.qdap import distribution_targets, qdap_loss


def audit_official_partitions(train, validation, test):
    """Keep official rows intact, reject ID/parent leakage, disclose text overlap."""
    parts = (train, validation, test)
    for key in ("query_id", "parent_doc_id"):
        sets = [{row[key] for row in rows} for rows in parts]
        if key == "query_id" and any(len(s) != len(p) for s, p in zip(sets, parts)):
            raise ValueError("duplicate query_id")
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"official partition {key} overlap")
    texts = [{" ".join(row["text"].casefold().split()) for row in p} for p in parts]
    return {
        "train_queries": len(train),
        "validation_queries": len(validation),
        "test_queries": len(test),
        "train_validation_text_overlap": len(texts[0] & texts[1]),
        "train_test_text_overlap": len(texts[0] & texts[2]),
        "query_and_parent_disjoint": True,
        "policy": "preserve all official rows; existing text overlaps disclosed",
    }


def select_validation_checkpoint(records, *, split):
    """Maximize Validation nDCG only; input order resolves exact ties."""
    if split != "validation":
        raise ValueError("checkpoint selection requires validation")
    if not records or any(not math.isfinite(r["metrics"]["ndcg@10"]) for r in records):
        raise ValueError("finite nonempty Validation records required")
    return max(records, key=lambda r: r["metrics"]["ndcg@10"])


def fit_official(
    model,
    utilities,
    forward_train,
    predict_validation,
    validation_pools,
    validation_qrels,
    output,
    *,
    ce_mode="literal",
    seed=42,
    epochs=5,
    microbatch=64,
    effective_batch=128,
    learning_rate=1e-4,
):
    """Fit all supplied Train rows; never accepts Test scores or labels.

    Callbacks receive row indices, return differentiable Train probabilities or
    NumPy Validation probabilities. L's backbone and optimizer can use BF16;
    the prediction head/loss stay FP32. Last short groups are weighted by size.
    """
    if epochs <= 0 or microbatch <= 0 or effective_batch < microbatch:
        raise ValueError("positive epochs and valid batch sizes required")
    output.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    target = distribution_targets(utilities).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=epochs * math.ceil(len(target) / effective_batch),
        pct_start=0.05,
    )
    rng = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    records, history, selected, best = [], [], None, -float("inf")
    # Record an actual update in both ends of L's backbone, not just requires_grad.
    encoder_parameters = [
        (name, value)
        for name, value in model.named_parameters()
        if name.startswith("encoder.") and value.ndim > 1
    ]
    probes = {}
    if encoder_parameters:
        layers = [
            (name, value) for name, value in encoder_parameters if "layer." in name
        ]
        chosen = [encoder_parameters[0], *(layers[:1] + layers[-1:])]
        probes = {name: value.detach().cpu().clone() for name, value in chosen}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(target), generator=rng).numpy()
        total = 0.0
        for start in range(0, len(order), effective_batch):
            group = order[start : start + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(group), microbatch):
                batch = group[offset : offset + microbatch]
                loss = qdap_loss(forward_train(batch), target[batch], ce_mode=ce_mode)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite Train loss")
                (loss * (len(batch) / len(group))).backward()
                total += float(loss.detach()) * len(batch)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=False
            )
            if not torch.isfinite(norm):
                raise ValueError("nonfinite gradient norm")
            optimizer.step()
            scheduler.step()
            if start % (effective_batch * 20) == 0:
                print(
                    {
                        "epoch": epoch,
                        "train_rows": start,
                        "seconds": round(time.perf_counter() - started, 1),
                    },
                    flush=True,
                )
        model.eval()
        with torch.inference_mode():
            probabilities = predict_validation()
        current = []
        for readout in ("mean", "argmax"):
            alphas = probability_alphas(probabilities, readout)
            metrics, _ = ranking_metrics(
                rank_pools(validation_pools, alphas), validation_qrels
            )
            record = {
                "epoch": epoch,
                "readout": readout,
                "ce_mode": ce_mode,
                "seed": seed,
                "metrics": metrics,
            }
            current.append(record)
            records.append(record)
        choice = select_validation_checkpoint(current, split="validation")
        if choice["metrics"]["ndcg@10"] > best:
            best = choice["metrics"]["ndcg@10"]
            selected = choice
            torch.save(model.state_dict(), output / "selected.pt")
            selected_encoder_updates = {
                name: not torch.equal(value.detach().cpu(), probes[name])
                for name, value in model.named_parameters()
                if name in probes
            }
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(target),
                "seconds": time.perf_counter() - started,
            }
        )
        report = {
            "selection_split": "validation",
            "train_queries": len(target),
            "selected": selected,
            "validation_candidates": records,
            "history": history,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "selected_encoder_updates": selected_encoder_updates,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
            if device.type == "cuda"
            else None,
        }
        write_json(output / "training.json", report)
        print(
            {
                "epoch_finished": epoch,
                "selected": selected,
                "seconds": history[-1]["seconds"],
            },
            flush=True,
        )
    return report
