"""Exploratory downstream RAG evaluation on a fixed Ko-StrategyQA dev subset.

Stages share one output directory and run in order:
retrieve -> generate (generator server) -> judge (judge server) -> summarize.
generate/judge append to JSONL and skip finished rows, so they can resume.
Retrievers, α=0.05 and the pinned cache are reused unchanged from main.
"""

import asyncio
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean

from evaluation.json_io import read_json, write_json
from evaluation.rag_downstream import (
    MAX_QUERIES,
    PAIRED_COMPARISONS,
    PILOT_QUERIES,
    PROMPT_TEMPLATE,
    RAG_CONTEXT_TOP_K,
    RAG_METHODS,
    RAG_SEED,
    build_prompt,
    context_records,
    context_text,
    finite_mean,
    id_context_metrics,
    is_abstention,
    paired_bootstrap,
    parent_id,
    shuffled_query_ids,
)
from scripts._cli import CliArgumentParser, _positive_int

DATASET = "strategyqa"
# 2026-10-20 검증 기록(docs/validation/2026-10-20/evaluation.txt)의 dev 592 nDCG@10.
FROZEN_NDCG = {
    "tfidf": 0.5597942020076703,
    "bm25": 0.562242200473678,
    "dense": 0.7940549183290013,
    "rrf": 0.7331519380656257,
    "score_fusion": 0.8000796704836824,
}
GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "seed": 0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}
RAGAS_METRICS = ("context_precision", "faithfulness", "response_relevancy")
IR_METRICS = ("recall@5", "rr@10", "ndcg@10")
ID_METRICS = ("id_context_precision", "id_context_recall")


def _read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        output.flush()


def _key(row):
    return row["method"], row["query_id"]


def _retrieve(args):
    from evaluation.benchmark import check_output_directory
    from evaluation.constants import EVALUATION_TOP_K
    from evaluation.final import _build_final_retrievers, load_final_dataset
    from evaluation.metrics import evaluate_query

    check_output_directory(args.output_dir)
    dataset, data, vectors, query_vectors = load_final_dataset(args.cache_dir / DATASET)
    retrievers, _, configs = _build_final_retrievers(
        dataset, data, vectors, query_vectors, RAG_METHODS
    )
    texts = {query.query_id: query.text for query in dataset.queries}
    documents = {doc.document_id: doc for doc in dataset.documents}
    # 기존 검색기가 바뀌지 않았는지 dev 전체 592질의의 고정 nDCG@10으로 먼저 확인한다.
    full_dev = {}
    for method, retriever in retrievers.items():
        per_query = [
            evaluate_query(retriever.search(texts[q], EVALUATION_TOP_K), qrels)
            for q, qrels in dataset.qrels.items()
        ]
        full_dev[method] = {k: fmean(p[k] for p in per_query) for k in per_query[0]}
        if abs(full_dev[method]["ndcg@10"] - FROZEN_NDCG[method]) > 1e-9:
            raise ValueError(f"{method} nDCG@10이 고정 결과와 다릅니다.")
    order = shuffled_query_ids(texts, args.seed)[: args.queries]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "queries.jsonl").open("w", encoding="utf-8") as output:
        for position, query_id in enumerate(order):
            qrels = dataset.qrels[query_id]
            row = {
                "position": position,
                "query_id": query_id,
                "parent_id": parent_id(query_id),
                "question": texts[query_id],
                "relevant": sorted(d for d, label in qrels.items() if label > 0),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "retrieval.jsonl").open("w", encoding="utf-8") as output:
        for position, query_id in enumerate(order):
            qrels = dataset.qrels[query_id]
            for method in RAG_METHODS:
                # RRF baseline은 각 검색기의 Top-10 후보를 쓰므로 10개를 받은 뒤 자른다.
                hits = retrievers[method].search(texts[query_id], EVALUATION_TOP_K)
                contexts = context_records(hits, documents)
                row = {
                    "position": position,
                    "query_id": query_id,
                    "parent_id": parent_id(query_id),
                    "method": method,
                    "question": texts[query_id],
                    "top10": [
                        {"document_id": h.document_id, "rank": h.rank, "score": h.score}
                        for h in hits
                    ],
                    "contexts": contexts,
                    "ir": evaluate_query(hits, qrels),
                    **id_context_metrics(contexts, qrels),
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        args.output_dir / "config.json",
        {
            "schema_version": "1",
            "purpose": "exploratory downstream RAG validation; not a final benchmark",
            "dataset": {
                "id": "taeminlee/Ko-StrategyQA",
                "split": "dev",
                "queries": "decomposed sub-questions; no reference answers",
                **{
                    k: dataset.provenance[k]
                    for k in ("dataset_revision", "input_sha256", "corpus_size")
                },
                "dev_query_count": len(texts),
            },
            "subset": {
                "seed": args.seed,
                "selection": "random.Random(seed).shuffle(sorted dev IDs); prefix",
                "selected": len(order),
                "pilot": PILOT_QUERIES,
                "expansion_rule": (
                    "extend the pilot prefix to 100 only if generation/judge "
                    "failures are under 5% of pilot rows; decided before scores"
                ),
            },
            "retrieval": {
                "methods": list(RAG_METHODS),
                "configs": configs,
                "search_top_k": EVALUATION_TOP_K,
                "context_top_k": RAG_CONTEXT_TOP_K,
                "full_dev_reproduction": full_dev,
                "frozen_ndcg@10": FROZEN_NDCG,
            },
            "prompt_template": PROMPT_TEMPLATE,
        },
    )
    print(json.dumps(full_dev, indent=2), file=sys.stderr)


def _select(rows, limit):
    return [row for row in rows if row["position"] < limit]


def _server_models(client):
    return [model.model_dump() for model in client.models.list().data]


def _check_or_write(path, record):
    if path.exists():
        previous = read_json(path)
        if {k: previous[k] for k in record if k != "server"} != {
            k: v for k, v in record.items() if k != "server"
        }:
            raise ValueError(f"{path.name} 설정이 이전 실행과 다릅니다.")
        return
    write_json(path, record)


def _generate(args):
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}, "top_k": -1}
    _check_or_write(
        args.output_dir / "generation-config.json",
        {
            "model": args.model,
            "model_revision": args.model_revision,
            "params": GENERATION_PARAMS,
            "extra_body": extra_body,
            "messages": "single user message from prompt_template",
            "server": _server_models(client),
        },
    )
    output = args.output_dir / "generations.jsonl"
    done = {_key(row) for row in _read_jsonl(output)}
    pending = [
        row
        for row in _select(
            _read_jsonl(args.output_dir / "retrieval.jsonl"), args.queries
        )
        if _key(row) not in done
    ]

    def call(row):
        prompt = build_prompt(row["question"], row["contexts"])
        result = {"method": row["method"], "query_id": row["query_id"]}
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra_body,
                **GENERATION_PARAMS,
            )
            choice = completion.choices[0]
            result |= {
                "response": (choice.message.content or "").strip(),
                "finish_reason": choice.finish_reason,
                "usage": completion.usage.model_dump() if completion.usage else None,
                "error": None,
            }
        except Exception as exc:  # 실패도 제거하지 않고 기록한다.
            result |= {"response": None, "error": f"{type(exc).__name__}: {exc}"}
        return result

    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(call, row) for row in pending]
        for count, future in enumerate(as_completed(futures), 1):
            _append_jsonl(output, future.result())
            if count % 50 == 0:
                print(f"generated {count}/{len(pending)}", file=sys.stderr)


async def _judge_rows(args, rows, contexts, questions):
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    from openai import AsyncOpenAI
    from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        Faithfulness,
    )

    llm = llm_factory(
        args.model,
        provider="openai",
        client=AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=900),
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )
    embeddings = HuggingFaceEmbeddings(
        model=args.embedding_model,
        device=args.embedding_device,
        revision=args.embedding_revision,
    )
    metrics = {
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "faithfulness": Faithfulness(llm=llm),
        "response_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    }
    gate = asyncio.Semaphore(args.concurrency)
    output = args.output_dir / "ragas-results.jsonl"

    async def score(row):
        key = _key(row)
        inputs = {
            "user_input": questions[row["query_id"]],
            "response": row["response"],
            "retrieved_contexts": [context_text(c) for c in contexts[key]],
        }
        result = {"method": row["method"], "query_id": row["query_id"]}
        for name, metric in metrics.items():
            if row["error"] or not row["response"]:
                result[name], result[f"{name}_error"] = None, "generation_failed"
                continue
            needed = {
                k: inputs[k]
                for k in ("user_input", "response", "retrieved_contexts")
                if name != "response_relevancy" or k != "retrieved_contexts"
            }
            async with gate:
                try:
                    value = (await metric.ascore(**needed)).value
                    result[name], result[f"{name}_error"] = value, None
                except Exception as exc:
                    result[name] = None
                    result[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:500]
        _append_jsonl(output, result)

    await asyncio.gather(*(score(row) for row in rows))


def _judge(args):
    from openai import OpenAI

    _check_or_write(
        args.output_dir / "judge-config.json",
        {
            "model": args.model,
            "model_revision": args.model_revision,
            "params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": args.max_tokens},
            "ragas_version": __import__("importlib.metadata").metadata.version("ragas"),
            "metrics": {
                "context_precision": "ContextPrecisionWithoutReference",
                "faithfulness": "Faithfulness",
                "response_relevancy": "AnswerRelevancy (strictness=3)",
            },
            "excluded": {
                "answer_correctness": "requires a reference answer; none exists",
                "context_recall": "LLM variant requires a reference answer",
            },
            "embedding": {
                "model": args.embedding_model,
                "revision": args.embedding_revision,
                "used_by": "response_relevancy only",
            },
            "server": _server_models(OpenAI(base_url=args.base_url, api_key="EMPTY")),
        },
    )
    retrieval = _select(_read_jsonl(args.output_dir / "retrieval.jsonl"), args.queries)
    contexts = {_key(row): row["contexts"] for row in retrieval}
    questions = {row["query_id"]: row["question"] for row in retrieval}
    done = {_key(row) for row in _read_jsonl(args.output_dir / "ragas-results.jsonl")}
    generations = {
        _key(row): row for row in _read_jsonl(args.output_dir / "generations.jsonl")
    }
    missing = set(contexts) - set(generations)
    if missing:
        raise ValueError(f"생성 결과가 없는 행이 {len(missing)}개 있습니다.")
    rows = [generations[key] for key in contexts if key not in done]
    asyncio.run(_judge_rows(args, rows, contexts, questions))


def _joined(output_dir, limit):
    retrieval = _select(_read_jsonl(output_dir / "retrieval.jsonl"), limit)
    generations = {_key(r): r for r in _read_jsonl(output_dir / "generations.jsonl")}
    judged = {_key(r): r for r in _read_jsonl(output_dir / "ragas-results.jsonl")}
    rows = []
    for row in retrieval:
        key = _key(row)
        if key not in generations or key not in judged:
            raise ValueError(f"생성/평가가 끝나지 않은 행이 있습니다: {key}")
        generation, judge = generations[key], judged[key]
        rows.append(
            {
                "position": row["position"],
                "query_id": row["query_id"],
                "parent_id": row["parent_id"],
                "method": row["method"],
                "question": row["question"],
                "context_ids": [c["document_id"] for c in row["contexts"]],
                **row["ir"],
                **{k: row[k] for k in ID_METRICS},
                "response": generation["response"],
                "generation_error": generation["error"],
                "finish_reason": generation.get("finish_reason"),
                "abstained": is_abstention(generation["response"]),
                **{k: judge[k] for k in RAGAS_METRICS},
                **{f"{k}_error": judge[f"{k}_error"] for k in RAGAS_METRICS},
            }
        )
    return rows


def _value(row, metric):
    value = row[metric]
    return None if value is None else float(value)


def _compare(index, query_ids, a, b, seed):
    """Paired a-b deltas, bootstrapped by original StrategyQA question ID."""
    same = [
        q for q in query_ids if index[a, q]["context_ids"] == index[b, q]["context_ids"]
    ]
    metrics = {}
    for metric in (*IR_METRICS, *ID_METRICS, *RAGAS_METRICS, "abstained"):
        pairs = [
            (
                index[a, q]["parent_id"],
                _value(index[a, q], metric),
                _value(index[b, q], metric),
            )
            for q in query_ids
        ]
        entry = paired_bootstrap(pairs, seed=seed)
        # 같은 context에서 생긴 차이는 생성·judge 비결정성의 하한 추정치다.
        noise = [
            abs(x - y)
            for (_, x, y), q in zip(pairs, query_ids, strict=True)
            if q in same and x is not None and y is not None
        ]
        entry["identical_context_mean_abs_delta"] = fmean(noise) if noise else None
        metrics[metric] = entry
    differing = [
        {
            "query_id": q,
            "position": index[a, q]["position"],
            "question": index[a, q]["question"],
            "context_ids": {
                a: index[a, q]["context_ids"],
                b: index[b, q]["context_ids"],
            },
            "responses": {a: index[a, q]["response"], b: index[b, q]["response"]},
            "deltas": {
                m: None
                if _value(index[a, q], m) is None or _value(index[b, q], m) is None
                else _value(index[a, q], m) - _value(index[b, q], m)
                for m in (*IR_METRICS, *ID_METRICS, *RAGAS_METRICS)
            },
        }
        for q in query_ids
        if set(index[a, q]["context_ids"]) != set(index[b, q]["context_ids"])
    ]
    return {
        "identical_top5_queries": len(same),
        "same_set_different_order_queries": len(query_ids) - len(same) - len(differing),
        "different_set_queries": len(differing),
        "metrics": metrics,
        "differing_queries": differing,
    }


def _summarize(args):
    rows = _joined(args.output_dir, args.queries)
    by_method = {m: [r for r in rows if r["method"] == m] for m in RAG_METHODS}
    counts = {len(v) for v in by_method.values()}
    if len(counts) != 1:
        raise ValueError("method마다 질의 수가 다릅니다.")
    values = (*IR_METRICS, *ID_METRICS, *RAGAS_METRICS)
    table = {}
    for method, method_rows in by_method.items():
        table[method] = {k: finite_mean([r[k] for r in method_rows]) for k in values}
        table[method]["abstention_rate"] = fmean(r["abstained"] for r in method_rows)
        table[method]["judge_errors"] = {
            k: sum(
                bool(r[f"{k}_error"]) and r[f"{k}_error"] != "generation_failed"
                for r in method_rows
            )
            for k in RAGAS_METRICS
        }
        table[method]["generation_errors"] = sum(
            bool(r["generation_error"]) for r in method_rows
        )
    index = {(r["method"], r["query_id"]): r for r in rows}
    query_ids = [r["query_id"] for r in by_method[RAG_METHODS[0]]]
    truncated = [
        {"method": r["method"], "query_id": r["query_id"], "position": r["position"]}
        for r in rows
        if r["finish_reason"] == "length"
    ]
    paired = {}
    for a, b in PAIRED_COMPARISONS:
        comparison = _compare(index, query_ids, a, b, args.seed)
        # 본 결과는 잘린 생성을 포함한다. 아래는 영향 확인용 sensitivity note다.
        cut = {t["query_id"] for t in truncated if t["method"] in (a, b)}
        if cut:
            kept = [q for q in query_ids if q not in cut]
            comparison["sensitivity_without_truncated"] = {
                "excluded_query_ids": sorted(cut),
                "metrics": _compare(index, kept, a, b, args.seed)["metrics"],
            }
        paired[f"{a}-{b}"] = comparison
    summary = {
        "queries": len(query_ids),
        "config": read_json(args.output_dir / "config.json"),
        "generation": read_json(args.output_dir / "generation-config.json"),
        "judge": read_json(args.output_dir / "judge-config.json"),
        "methods": table,
        "truncated_generations": truncated,
        "paired": paired,
    }
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "ragas-results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "context_ids": " | ".join(row["context_ids"])})
    print(
        json.dumps({"methods": table, "paired": paired}, ensure_ascii=False, indent=2)
    )


def main(argv=None):
    parser = CliArgumentParser(description=__doc__, allow_abbrev=False)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("retrieve", "generate", "judge", "summarize"):
        stage = stages.add_parser(name)
        stage.add_argument(
            "--output-dir", type=Path, default=Path("artifacts/ragas-eval")
        )
        stage.add_argument(
            "--queries",
            type=_positive_int,
            default=MAX_QUERIES if name == "retrieve" else PILOT_QUERIES,
        )
        stage.add_argument("--seed", type=int, default=RAG_SEED)
        if name == "retrieve":
            stage.add_argument(
                "--cache-dir", type=Path, default=Path("artifacts/score-fusion-cache")
            )
        if name in ("generate", "judge"):
            stage.add_argument("--base-url", required=True)
            stage.add_argument("--model", required=True)
            stage.add_argument("--model-revision", required=True)
            stage.add_argument("--concurrency", type=_positive_int, default=16)
        if name == "judge":
            stage.add_argument("--max-tokens", type=_positive_int, default=4096)
            stage.add_argument("--embedding-model", default="BAAI/bge-m3")
            stage.add_argument(
                "--embedding-revision",
                default="5617a9f61b028005a4858fdac845db406aefb181",
            )
            stage.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args(argv)
    if args.queries > MAX_QUERIES:
        parser.error(f"--queries는 {MAX_QUERIES} 이하여야 합니다.")
    handler = {
        "retrieve": _retrieve,
        "generate": _generate,
        "judge": _judge,
        "summarize": _summarize,
    }[args.stage]
    try:
        handler(args)
    except ImportError as exc:
        parser.error(f"sparse/dense/rag extras가 필요합니다: {exc}")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"downstream 평가 실패: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
