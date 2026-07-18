from __future__ import annotations

import base64
import datetime as dt
import json
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import ValidationError

from insight.context import bounded_history_text
from insight.models import (
    InsightChartRecord,
    InsightDecision,
    InsightHistoryPayload,
    InsightMessage,
)
from visualization.models import FigureSpec
from visualization.service import (
    build_visualization,
    figure_to_bytes,
    source_payload,
)
from visualization.statistics import VisualizationDataError


class InsightAPIError(RuntimeError):
    """Raised when Gemini cannot complete an Insight request."""


@dataclass
class InsightExecution:
    message: InsightMessage
    visualization_source: dict[str, Any] | None = None


ROUTER_SYSTEM_PROMPT = """
당신은 전시·행사 데이터 분석을 지원하는 수석 비즈니스 분석가입니다.
반드시 제공된 데이터 근거와 기존 대화만 사용하고, 근거가 없는 원인은 가설이라고 표시하세요.

사용자 요청을 다음 중 하나로 분류하세요.
- text: 설명, 분석, 요약, 질의응답 또는 전체 보고서
- chart: 차트만 요청
- chart_with_insight: 차트와 해석을 함께 요청

차트는 Python 코드를 작성하지 말고 chart_spec으로만 정의하세요.
사용 가능한 차트는 bar, line, multi_variable, pie, histogram, scatter_plot,
grouped_bar, stacked_bar, scatter_bubble, heatmap, correlation_heatmap입니다.
실제 컬럼명만 사용하세요. 합계·평균이면 수치형 value_column을 지정하세요.
grouped_bar, stacked_bar, scatter_bubble, heatmap은 x와 y를 지정하세요.
correlation_heatmap과 multi_variable은 variables에 2개 이상의 컬럼을 지정하세요.

text 답변은 사용자가 요구한 범위만 한국어 Markdown으로 작성하세요.
전체 비즈니스 인사이트 또는 보고서 요청이면 다음 근거와 역할을 포함하세요.
- Executive Summary
- 데이터 기본 구조와 주요 기술통계
- 전처리 이력의 영향
- 저장된 시각화 통계자료 분석
- 핵심 결과, 원인 해석, 업무적 의미
- 권장 실행방안
- 추가 확인 질문
- 분석 한계와 가정
수치 주장은 제공된 근거에서 확인되는 값만 사용하세요.
""".strip()


CHART_INSIGHT_SYSTEM_PROMPT = """
당신은 전시·행사 데이터 분석 수석 컨설턴트입니다.
제공된 차트 통계자료를 우선 근거로 사용하여 한국어 Markdown으로 답하세요.
핵심 결과, 원인 해석, 업무적 의미, 권장 실행방안, 분석 한계를 구분하세요.
원인은 데이터로 검증되지 않았다면 반드시 '가능한 가설'이라고 표시하세요.
차트에 없는 숫자나 사실을 만들지 마세요.
""".strip()


def _client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:
        raise InsightAPIError("google-genai가 설치되어 있지 않습니다. requirements.txt를 다시 설치하세요.") from exc
    return genai.Client(api_key=api_key)


def _config(*, system_instruction: str, response_schema=None, temperature: float = 0.2):
    try:
        from google.genai import types
    except ImportError as exc:
        raise InsightAPIError("google-genai가 설치되어 있지 않습니다.") from exc
    kwargs: dict[str, Any] = {
        "system_instruction": system_instruction,
        "temperature": temperature,
        "max_output_tokens": 8192,
    }
    if response_schema is not None:
        kwargs.update(response_mime_type="application/json", response_schema=response_schema)
    return types.GenerateContentConfig(**kwargs)


def _parse_decision(response) -> InsightDecision:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, InsightDecision):
        return parsed
    if parsed is not None:
        return InsightDecision.model_validate(parsed)
    text = getattr(response, "text", "") or ""
    try:
        return InsightDecision.model_validate_json(text)
    except (ValidationError, ValueError, TypeError) as exc:
        raise InsightAPIError("Gemini의 구조화 응답을 해석할 수 없습니다.") from exc


def _decision_prompt(
    question: str,
    evidence_context: str,
    history: list[dict[str, Any]],
    correction: str | None = None,
) -> str:
    correction_text = f"\n\n[이전 ChartSpec 검증 오류]\n{correction}\n오류를 수정하세요." if correction else ""
    return f"""[데이터 및 분석 근거]
{evidence_context}

[기존 대화]
{bounded_history_text(history) or '기존 대화 없음'}

[현재 사용자 요청]
{question}{correction_text}
""".strip()


def plan_request(
    client,
    model: str,
    question: str,
    evidence_context: str,
    history: list[dict[str, Any]],
    correction: str | None = None,
) -> InsightDecision:
    try:
        response = client.models.generate_content(
            model=model,
            contents=_decision_prompt(question, evidence_context, history, correction),
            config=_config(system_instruction=ROUTER_SYSTEM_PROMPT, response_schema=InsightDecision),
        )
        return _parse_decision(response)
    except InsightAPIError:
        raise
    except Exception as exc:
        raise InsightAPIError(f"Gemini 요청 목적 분석에 실패했습니다: {exc}") from exc


def _chart_interpretation(
    client,
    model: str,
    question: str,
    evidence_context: str,
    history: list[dict[str, Any]],
    chart_source: dict[str, Any],
) -> str:
    prompt = f"""[사용자 요청]
{question}

[차트와 통계자료]
{json.dumps(chart_source, ensure_ascii=False, indent=2, default=str)}

[전체 데이터 근거]
{evidence_context}

[기존 대화]
{bounded_history_text(history)}
"""
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_config(system_instruction=CHART_INSIGHT_SYSTEM_PROMPT, temperature=0.3),
        )
        return (getattr(response, "text", "") or "").strip() or "차트는 생성되었지만 해석 텍스트가 없습니다."
    except Exception as exc:
        raise InsightAPIError(f"차트 해석 생성에 실패했습니다: {exc}") from exc


def execute_request(
    *,
    api_key: str,
    model: str,
    question: str,
    frame: pd.DataFrame,
    source_filename: str,
    evidence_context: str,
    history: list[dict[str, Any]],
) -> InsightExecution:
    client = _client(api_key)
    correction = None
    for attempt in range(2):
        decision = plan_request(
            client, model, question, evidence_context, history, correction=correction
        )
        if decision.action == "text":
            return InsightExecution(message=InsightMessage(role="model", text=decision.answer.strip()))
        try:
            spec = decision.chart_spec.to_chart_spec()
            result = build_visualization(frame, [spec], FigureSpec(rows=1, columns=1, width=10, height=6))
            chart_source = source_payload(result, source_filename)
            image = figure_to_bytes(result, "png")
            plt.close(result.figure)
            chart_record = InsightChartRecord(
                image_base64=base64.b64encode(image).decode("ascii"),
                source=chart_source,
            )
            if decision.action == "chart_with_insight":
                answer = _chart_interpretation(
                    client, model, question, evidence_context, history, chart_source
                )
            else:
                answer = f"요청한 `{spec.title}` 차트를 생성했습니다."
            return InsightExecution(
                message=InsightMessage(role="model", text=answer, charts=[chart_record]),
                visualization_source=chart_source,
            )
        except (ValidationError, VisualizationDataError, ValueError, TypeError) as exc:
            correction = str(exc)
            if attempt == 1:
                raise InsightAPIError(f"ChartSpec 검증에 실패했습니다: {exc}") from exc
    raise InsightAPIError("Insight 요청을 완료하지 못했습니다.")


def history_payload_bytes(
    history: list[dict[str, Any]],
    model: str,
    source_filename: str | None,
    frame: pd.DataFrame,
) -> bytes:
    payload = InsightHistoryPayload(
        saved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        model=model,
        source_filename=source_filename,
        data_signature={"shape": list(frame.shape), "columns": [str(column) for column in frame.columns]},
        history=[InsightMessage.model_validate(message) for message in history],
    )
    return payload.model_dump_json(indent=2).encode("utf-8")


def history_markdown_bytes(history: list[dict[str, Any]], source_filename: str | None) -> bytes:
    lines = ["# 비즈니스 인사이트", "", f"분석 파일: {source_filename or '확인되지 않음'}", ""]
    for message in history:
        if message.get("hidden"):
            continue
        heading = "## 사용자" if message.get("role") == "user" else "## 분석가"
        lines.extend([heading, "", str(message.get("text", "")), ""])
        for chart in message.get("charts", []):
            lines.extend(
                [
                    "### 생성 차트 통계자료",
                    "",
                    "```json",
                    json.dumps(chart.get("source", {}), ensure_ascii=False, indent=2, default=str),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).encode("utf-8-sig")


def restore_history(raw: bytes, filename: str) -> tuple[list[dict[str, Any]], str | None]:
    text = None
    for encoding in ("utf-8-sig", "cp949", "utf-16"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("업로드한 인사이트 파일을 텍스트로 읽을 수 없습니다.")
    if filename.lower().endswith(".json"):
        try:
            payload = InsightHistoryPayload.model_validate_json(text)
        except (ValidationError, ValueError) as exc:
            raise ValueError("Business Insight JSON 형식이 올바르지 않습니다.") from exc
        return [message.model_dump(mode="json") for message in payload.history], payload.model
    content = text.strip()
    if not content:
        raise ValueError("업로드한 인사이트 파일이 비어 있습니다.")
    return [InsightMessage(role="model", text=content).model_dump(mode="json")], None
