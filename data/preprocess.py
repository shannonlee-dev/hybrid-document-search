"""원본 문서를 변경하지 않고 공통 인덱싱 텍스트를 만듭니다."""

from data.loader import Document


def build_index_text(document: Document) -> str:
    """제목이 있으면 제목과 본문을 줄바꿈으로 연결합니다."""
    if document.title is None or not document.title.strip():
        return document.text
    return f"{document.title}\n{document.text}"
