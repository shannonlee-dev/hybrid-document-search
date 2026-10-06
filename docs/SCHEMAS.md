# Shared Data Schema Draft

Status: Draft — pending team alignment
Related issue: #4
Owner: @bangahee

## Document

One record represents one source passage.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| document_id | string | Yes | Stable original passage ID |
| text | string | Yes | Original passage text |
| title | string or null | No | Source article title |

Source mapping:
- _id → document_id
- text → text
- title → title

Rules:
- document_id must be nonempty and unique within the corpus.
- Preserve the complete source ID, including its # suffix.
- IDs are strings; do not replace them with array positions.
- Preserve the passage-level retrieval unit.
- Missing titles must be supported.
- Preserve text for display; do not overwrite it with TF-IDF tokens.

Synthetic example:

```json
{
  "document_id": "fixture-doc-001",
  "text": "서울은 대한민국의 수도입니다.",
  "title": "서울"
}
```

## Query

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| query_id | string | Yes | Stable source query ID |
| text | string | Yes | Query text |

Source mapping:
- _id → query_id
- text → text

Synthetic example:

```json
{
  "query_id": "fixture-query-001",
  "text": "대한민국의 수도는 어디인가요?"
}
```

Query IDs and document IDs belong to separate namespaces.
Do not infer relevant documents from the appearance of a query ID.

## Relevance judgment / qrel

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| query_id | string | Yes | Referenced query |
| document_id | string | Yes | Judged passage |
| relevance | number | Yes | Original relevance label |

Source mapping:
- query-id → query_id
- corpus-id → document_id
- score → relevance

Rules:
- Preserve source label values and train/dev membership.
- Verify observed label values before freezing evaluation rules.
- Validate that referenced queries and documents exist.
- Report conflicting duplicate judgments.
- Relevance labels are not retriever similarity scores.
- A missing judgment means unjudged, not an explicit negative.

Synthetic example:

```json
{
  "query_id": "fixture-query-001",
  "document_id": "fixture-doc-001",
  "relevance": 1
}
```

## Proposed indexed content

Both Sparse and Dense retrieval should use the same source content:
title + newline + text when a title is available, otherwise text.

This is a proposal requiring team agreement.
Method-specific tokenization or model-required formatting may differ.

## Existing search contract

Preserve the existing Retriever and SearchResult definitions.

search(query: str, top_k: int) -> list[SearchResult]

Results use stable document IDs, descending retrieval order,
and contiguous ranks starting at 1.

Scores are specific to each retrieval method.
RRF combines ranks rather than comparing raw scores directly.

## Pending decisions

- Approve field names and missing-title handling.
- Approve title + text as shared indexed content.
- Agree on normalization and invalid-record handling.
- Agree on JSONL file layout and split representation.
