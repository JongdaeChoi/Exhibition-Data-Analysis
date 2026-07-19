from __future__ import annotations

import json
from typing import Any

import pandas as pd

from data.preprocessing import comparison_summary
from insight.models import InsightAttachment


MAX_CONTEXT_CHARS = 90_000
MAX_SECTION_CHARS = 30_000


def _clip(text: str, limit: int = MAX_SECTION_CHARS) -> str:
    value = str(text)
    return value if len(value) <= limit else value[:limit] + "\n[길이 제한으로 일부 생략]"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def build_evidence_context(
    original: pd.DataFrame,
    clean: pd.DataFrame,
    source_filename: str | None,
    preprocessing_history: list[dict[str, Any]],
    visualization_sources: list[dict[str, Any]],
    attachments: list[InsightAttachment] | None = None,
) -> str:
    """Build a bounded, source-first prompt context from the current application state."""
    profile = pd.DataFrame(
        {
            "변수명": [str(column) for column in clean.columns],
            "데이터 타입": [str(clean[column].dtype) for column in clean.columns],
            "유효값": [int(clean[column].notna().sum()) for column in clean.columns],
            "결측값": [int(clean[column].isna().sum()) for column in clean.columns],
            "고유값": [int(clean[column].nunique(dropna=False)) for column in clean.columns],
        }
    )
    try:
        descriptive = clean.describe(include="all").transpose().to_string()
    except (TypeError, ValueError):
        descriptive = clean.describe().transpose().to_string()
    references = []
    for item in attachments or []:
        if item.kind == "document":
            references.append(
                f"파일명: {item.filename}\n형식: {item.mime_type}\n내용:\n{_clip(item.extracted_text, 15_000)}"
            )
        else:
            references.append(
                f"이미지: {item.filename} ({item.mime_type}) - 이미지 원본은 API 요청에 함께 전달됨"
            )
    sections = [
        f"""[데이터 기본 구조]
파일명: {source_filename or '확인되지 않음'}
원본 크기: {original.shape}
현재 분석 데이터 크기: {clean.shape}
{_clip(profile.to_string(index=False))}""",
        f"""[전처리 이전·이후 비교]
{_clip(comparison_summary(original, clean).to_string(index=False))}""",
        f"""[전처리 이력]
{_clip(_json_text(preprocessing_history or [{'message': '기록된 전처리 작업이 없습니다.'}]))}""",
        f"""[주요 기술통계]
{_clip(descriptive)}""",
        f"""[현재 데이터 상위 5행]
{_clip(clean.head(5).to_string(index=False))}""",
        f"""[Visualization 저장 통계자료]
{_clip(_json_text(visualization_sources[-10:] or [{'message': '저장된 시각화 통계자료가 없습니다.'}]))}""",
        f"""[사용자 업로드 참고자료]
{_clip(chr(10).join(references) if references else '등록된 참고자료가 없습니다.')}""",
    ]
    context = "\n\n".join(sections)
    return context if len(context) <= MAX_CONTEXT_CHARS else context[:MAX_CONTEXT_CHARS] + "\n[전체 컨텍스트 길이 제한]"


def bounded_history_text(history: list[dict[str, Any]], limit: int = 25_000) -> str:
    rows = []
    for message in history[-20:]:
        role = "사용자" if message.get("role") == "user" else "분석가"
        text = str(message.get("text", "")).strip()
        if text:
            rows.append(f"{role}: {text}")
        for chart in message.get("charts", []):
            source = chart.get("source", {}) if isinstance(chart, dict) else {}
            rows.append("분석가가 생성한 차트 통계: " + _clip(_json_text(source), 8_000))
        if message.get("code"):
            code = message["code"]
            rows.append(
                "분석가가 실행한 Python 코드와 결과: "
                + _clip(_json_text(code), 8_000)
            )
        for attachment in message.get("attachments", []):
            rows.append(
                f"사용자 첨부자료: {attachment.get('filename', '')} ({attachment.get('mime_type', '')})"
            )
    return _clip("\n\n".join(rows), limit)
