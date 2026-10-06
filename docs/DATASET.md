# Dataset Draft: Ko-miracl

Status: Draft — pending team alignment
Related issue: #4
Owner: @bangahee

## Dataset selection

Selected dataset:
https://huggingface.co/datasets/taeminlee/Ko-miracl

Original MIRACL:
https://github.com/project-miracl/miracl

Ko-miracl converts the Korean portion of MIRACL into BEIR format.
It provides Korean Wikipedia passages, queries, and relevance
judgments for retrieval experiments.

Selection rationale:
- Korean content supports a Korean-language search demonstration.
- Existing queries and relevance judgments support later evaluation.
- The same corpus can be used for Sparse, Dense, and Hybrid retrieval.
- Source fields map naturally to the proposed document schema.

## Source structure

| Component | Source fields |
| --- | --- |
| Corpus | _id, title, text |
| Queries | _id, text |
| Qrels | query-id, corpus-id, score |

The retrieval unit is a passage, not a complete Wikipedia article.
Preserve original passage IDs, including their # suffixes.

## Proposed MVP preparation

Start with a reproducible subset rather than indexing the full
corpus immediately.

Proposed starting size: approximately 10,000–20,000 passages,
subject to resource checks and team agreement.

For selected queries:
- Include all their judged passages.
- Add background passages sampled from the remaining corpus.
- Record the selection rule, random seed, and selected IDs.
- Preserve the original train/dev query membership.
- Use the same prepared corpus for every retrieval method.

Reduced-corpus results will be identified as subset results,
not full MIRACL benchmark results.

## Proposed preprocessing

- Preserve document and query IDs.
- Validate required fields and duplicate IDs.
- Preserve original passage text for display.
- Apply agreed Unicode and whitespace normalization for indexing.
- Keep retrieval-specific tokenization separate from shared text.
- Report invalid records and unresolved references.

## Reproducibility and storage

- Dataset revision: pending selection and verification.
- Exact loaded counts: pending validation.
- Subset size and seed: pending team agreement.
- Source and usage terms: pending documented review.
- Raw downloads: data/raw/ or datasets/.
- Prepared data: data/processed/.
- Large dataset files remain outside Git.
- Commit preparation code and documentation so others can reproduce it.

## Decisions requiring team alignment

- MVP corpus size and query selection.
- Shared document and query/qrels schemas.
- Whether both retrievers index title + text.
- Shared normalization policy.
- Queries reserved for later held-out evaluation.
