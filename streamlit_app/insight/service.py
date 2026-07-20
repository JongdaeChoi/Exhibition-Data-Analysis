from __future__ import annotations

import base64
import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import ValidationError

from insight.code_runner import InsightCodeError, execute_generated_code
from insight.context import bounded_history_text
from insight.models import (
    InsightAttachment,
    InsightChartRecord,
    InsightCodeRecord,
    InsightDecision,
    InsightHistoryPayload,
    InsightMessage,
)
from visualization.models import ChartSpec, FigureSpec
from visualization.service import build_visualization, figure_to_bytes, source_payload
from visualization.statistics import VisualizationDataError


class InsightAPIError(RuntimeError):
    """Raised when an Insight provider cannot complete a request."""


@dataclass
class InsightExecution:
    message: InsightMessage
    visualization_source: dict[str, Any] | None = None
    updated_frame: pd.DataFrame | None = None


def rebuild_chart_record(
    frame: pd.DataFrame,
    spec_value: ChartSpec | dict[str, Any],
    source_filename: str,
) -> tuple[InsightChartRecord, dict[str, Any]]:
    """Validate a ChartSpec and rebuild an Insight chart through the shared pipeline."""
    spec = spec_value if isinstance(spec_value, ChartSpec) else ChartSpec.model_validate(spec_value)
    result = build_visualization(
        frame, [spec], FigureSpec(rows=1, columns=1, width=10, height=6)
    )
    try:
        chart_source = source_payload(result, source_filename)
        image = figure_to_bytes(result, "png")
    finally:
        plt.close(result.figure)
    return (
        InsightChartRecord(
            image_base64=base64.b64encode(image).decode("ascii"),
            source=chart_source,
        ),
        chart_source,
    )


INSIGHT_SYSTEM_PROMPT = """
당신은 전시회 전문 데이터 분석 수석 컨설턴트입니다.
제공된 데이터와 분석 컨텍스트를 근거로 정확하고 실행 가능한 결과를 제공하세요.
현재 사용자 요청에 답변·번역·해석 언어가 명시되어 있으면 그 언어를 최우선으로 따르세요.
사용자가 언어를 명시하지 않은 경우에만 프롬프트의 [기본 응답 언어]를 따르세요.
데이터에서 확인되지 않은 내용은 사실처럼 단정하지 말고 추정 또는 가정임을 명시하세요.

[기본 응답 규칙]
- 사용자가 요청한 결과만 제공하고 요청하지 않은 서론, 개요, 결측치, 기초통계, 해석, 제언 또는 보고서를 추가하지 마세요.
- 표를 요청하면 표와 직접 관련된 결과만, 설명을 요청하면 설명만 제공하세요.
- 사용자가 분석이나 해석을 명시적으로 요청한 경우에만 결과를 해석하세요.
- 저장된 요약 정보만으로 실제 계산값을 추정하거나 임의의 숫자를 생성하지 마세요.
- 지원되지 않는 요청을 다른 기능으로 임의 변경하지 말고 지원되지 않는 항목을 안내하세요.

[요청 분류]
- text: 설명, 요약, 해석 또는 일반 질의응답
- data: df_clean 조회, 필터링, 집계, 생성, 수정 또는 삭제
- chart: 차트만 생성
- chart_with_insight: 차트 생성과 실제 통계표 해석
- business_insight: 사용자가 명시적으로 요청한 종합 비즈니스 인사이트

[데이터프레임 규칙]
- df와 df_clean은 현재 메모리에 이미 존재합니다. 샘플 데이터를 만들거나 파일을 다시 읽지 마세요.
- df는 업로드 원본이므로 수정하거나 재할당하지 마세요.
- df_clean은 분석 및 전처리 작업본이며 사용자가 요청한 경우에만 수정하세요.
- 조회 및 집계 결과는 result_df, summary_df 또는 df_result 같은 별도 변수에 저장하고 display()로 표시하세요.

[일반 Python 요청 규칙]
- 실제 조회, 필터링, 집계, 컬럼 생성·삭제, 값 수정, 결측치 처리, 형변환 또는 중복 제거 요청만 data로 분류하세요.
- data 응답의 python_code에는 실행 가능한 Python 코드만 넣으세요. 코드 펜스, 설명, import, 파일 입출력, 네트워크 접근을 넣지 마세요.
- 사용자가 df_clean 변경을 명시한 경우에만 modifies_df_clean을 true로 설정하세요.
- 한글 폰트 설치 또는 설정 코드를 작성하지 마세요.
- data 응답에는 하나의 실행 단위만 작성하고, 조회·집계 결과를 반드시 display()로 출력하세요.

[차트 요청 규칙]
- 차트 요청에는 Python, Matplotlib 또는 Seaborn 코드를 작성하지 말고 지원되는 chart_spec만 생성하세요.
- 기존 대화의 차트에 대한 축 순서, 차트 유형, 집계 또는 표시 변경 요청도 chart로 분류하고 기존 ChartSpec을 요청대로 갱신하세요.
- 지원 차트: bar, line, multi_variable, pie, histogram, scatter_plot, grouped_bar, stacked_bar, scatter_bubble, heatmap, correlation_heatmap.
- 지원 집계: count, valid_count, sum, mean, ratio.
- 실제 컬럼명만 사용하세요. sum과 mean에는 수치형 value_column을 지정하세요.
- grouped_bar, stacked_bar, scatter_bubble, heatmap은 x와 y가 필요합니다.
- correlation_heatmap과 multi_variable은 variables에 두 개 이상의 컬럼이 필요합니다.
- X축 범주 표시 순서를 요청하면 x_category_order에 실제 범주값을 문자열 목록으로 입력하세요. 예: ["1", "2", "3", "4", "5"].
- Y축 범주 표시 순서를 요청하면 y_category_order에 실제 범주값을 문자열 목록으로 입력하세요.
- 범주 순서 기능을 지원하지 않는다고 답하지 말고, 기존 차트의 나머지 설정을 유지하면서 요청된 축 순서만 반영하세요.
- 지원되지 않는 차트·집계·변수 조합을 다른 기능으로 바꾸지 마세요.
- 차트 계산과 렌더링은 프로그램의 build_statistics 및 render_chart 함수가 수행합니다.

[이미지 사용 규칙]
- 업로드 이미지는 현재 질문의 참고자료로 사용하고 직접 확인되는 제목, 축, 범례, 수치와 패턴만 사실로 사용하세요.
- 식별하기 어렵거나 표시되지 않은 정보는 추측하지 마세요.
- 이미지와 df_clean 또는 저장 통계자료가 다르면 차이를 명시하고, 확인 사실과 해석을 구분하세요.
- 개인정보와 민감정보를 불필요하게 재출력하지 마세요.

[비즈니스 인사이트 규칙]
- 사용자가 명시적으로 요청한 경우에만 business_insight로 분류하세요.
- 데이터 기본 구조, 전처리 이력, 주요 기술통계, 저장된 시각화 통계자료와 현재 대화의 분석 결과를 근거로 사용하세요.
- 핵심 결과, 원인 해석, 업무적 의미, 권장 실행방안, 분석 한계로 구성하세요.
- 데이터로 확인되지 않은 원인은 추정 또는 가정이라고 명시하세요.

[최종 응답 점검]
- 현재 사용자 요청에 명시된 답변 언어를 따랐는지 확인하세요.
- 요청하지 않은 내용, 실제 계산하지 않은 숫자 또는 불필요한 코드를 포함하지 않았는지 확인하세요.
- data 요청에서만 python_code를 작성하고 df를 수정하지 않았는지 확인하세요.
- 사용자가 변경을 요청한 경우에만 modifies_df_clean을 true로 설정했는지 확인하세요.
- 차트 요청에는 Python 코드가 없고 지원되는 chart_spec만 있는지 확인하세요.
- 지원되지 않는 기능을 임의 생성하거나 다른 기능으로 변경하지 않았는지 확인하세요.
""".strip()


CHART_INSIGHT_SYSTEM_PROMPT = """
당신은 전시회 전문 데이터 분석 수석 컨설턴트입니다.
제공된 실제 차트 통계자료를 최우선 근거로 사용하세요.
현재 사용자 요청에 답변·번역·해석 언어가 명시되어 있으면 그 언어를 최우선으로 따르고,
언어 지시가 없을 때만 프롬프트의 [기본 응답 언어]로 간결하게 답하세요.
핵심 결과, 원인 해석, 업무적 의미, 권장 실행방안, 분석 한계를 구분하세요.
검증되지 않은 원인은 반드시 '가능한 가설'이라고 표시하고, 자료에 없는 숫자를 만들지 마세요.
""".strip()


_GEMINI_JSON_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "type",
    "format",
    "title",
    "description",
    "enum",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "anyOf",
    "oneOf",
    "properties",
    "required",
}


def _gemini_json_schema(model: type) -> dict[str, Any]:
    """Convert Pydantic JSON Schema to Gemini's supported structured-output subset."""

    def clean(value: Any, parent: str | None = None) -> Any:
        if isinstance(value, list):
            return [clean(item) for item in value]
        if not isinstance(value, dict):
            return value
        if parent in {"properties", "$defs"}:
            return {key: clean(item) for key, item in value.items()}
        return {
            key: clean(item, key)
            for key, item in value.items()
            if key in _GEMINI_JSON_SCHEMA_KEYS
        }

    return clean(model.model_json_schema())


def _client(api_key: str, provider: str = "Gemini"):
    if provider == "OpenAI":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise InsightAPIError("openai 패키지가 설치되어 있지 않습니다. requirements.txt를 다시 설치하세요.") from exc
        return OpenAI(api_key=api_key)
    try:
        from google import genai
    except ImportError as exc:
        raise InsightAPIError("google-genai가 설치되어 있지 않습니다. requirements.txt를 다시 설치하세요.") from exc
    return genai.Client(api_key=api_key)


def _gemini_config(*, response_schema=None, temperature: float = 0.2):
    try:
        from google.genai import types
    except ImportError as exc:
        raise InsightAPIError("google-genai가 설치되어 있지 않습니다.") from exc
    kwargs: dict[str, Any] = {
        "system_instruction": INSIGHT_SYSTEM_PROMPT,
        "temperature": temperature,
        "max_output_tokens": 8192,
    }
    if response_schema is not None:
        kwargs.update(
            response_mime_type="application/json",
            response_json_schema=_gemini_json_schema(response_schema),
        )
    return types.GenerateContentConfig(**kwargs)


def _image_attachments(attachments: list[InsightAttachment]) -> list[InsightAttachment]:
    return [item for item in attachments if item.kind == "image" and item.content_base64]


def _gemini_contents(prompt: str, attachments: list[InsightAttachment]):
    try:
        from google.genai import types
    except ImportError as exc:
        raise InsightAPIError("google-genai가 설치되어 있지 않습니다.") from exc
    parts = [types.Part.from_text(text=prompt)]
    for item in _image_attachments(attachments):
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(item.content_base64),
                mime_type=item.mime_type,
            )
        )
    return parts


def _openai_input(prompt: str, attachments: list[InsightAttachment]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for item in _image_attachments(attachments):
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{item.mime_type};base64,{item.content_base64}",
            }
        )
    return [{"role": "user", "content": content}]


def _parse_decision(response) -> InsightDecision:
    parsed = getattr(response, "parsed", None) or getattr(response, "output_parsed", None)
    if isinstance(parsed, InsightDecision):
        return parsed
    if parsed is not None:
        return InsightDecision.model_validate(parsed)
    text = getattr(response, "text", None) or getattr(response, "output_text", "") or ""
    try:
        return InsightDecision.model_validate_json(text)
    except (ValidationError, ValueError, TypeError) as exc:
        raise InsightAPIError("API 모델의 구조화 응답을 해석할 수 없습니다.") from exc


def _decision_prompt(
    question: str,
    evidence_context: str,
    history: list[dict[str, Any]],
    correction: str | None = None,
    response_language: str = "한국어",
) -> str:
    correction_text = (
        f"\n\n[이전 생성 결과의 검증 또는 실행 오류]\n{correction}\n요청을 바꾸지 말고 오류만 수정하세요."
        if correction
        else ""
    )
    return f"""[데이터 및 분석 근거]
{evidence_context}

[기존 대화]
{bounded_history_text(history) or '기존 대화 없음'}

[현재 사용자 요청]
{question}{correction_text}

[응답 언어 우선순위]
1. 현재 사용자 요청에 답변, 번역 또는 해석 언어가 명시되어 있으면 그 언어로 작성하세요.
2. 언어가 명시되지 않은 경우에만 아래 기본 응답 언어를 사용하세요.

[기본 응답 언어]
{response_language}
""".strip()


def _resolve_response_language(question: str, default_language: str) -> str:
    """Resolve explicit Korean/English instructions; the model handles other named languages."""
    normalized = " ".join((question or "").casefold().split())
    korean_patterns = (
        r"(?:한국어|한글)로(?:\s+(?:답|응답|설명|해석|번역|작성))?",
        r"(?:answer|respond|write|explain|interpret|translate)\s+(?:it\s+)?(?:in|into|to)\s+korean\b",
        r"(?:in|into)\s+korean\b",
    )
    english_patterns = (
        r"영어로(?:\s+(?:답|응답|설명|해석|번역|작성))?",
        r"(?:answer|respond|write|explain|interpret|translate)\s+(?:it\s+)?(?:in|into|to)\s+english\b",
        r"(?:in|into)\s+english\b",
    )
    matches: list[tuple[int, str]] = []
    for language, patterns in (("한국어", korean_patterns), ("English", english_patterns)):
        for pattern in patterns:
            for match in re.finditer(pattern, normalized):
                matches.append((match.start(), language))
    return max(matches, default=(-1, default_language))[1]


def _explicit_chart_aggregation(question: str):
    normalized = question.casefold()
    keyword_map = (
        (("valid count", "non-null count", "유효값", "결측 제외 개수"), "valid_count"),
        (("mean", "average", "평균"), "mean"),
        (("sum", "total", "합계"), "sum"),
        (("ratio", "percentage", "percent", "비율", "백분율"), "ratio"),
        (("count", "개수", "건수"), "count"),
    )
    for keywords, aggregation in keyword_map:
        if any(keyword in normalized for keyword in keywords):
            return aggregation
    return None


def plan_request(
    client,
    model: str,
    question: str,
    evidence_context: str,
    history: list[dict[str, Any]],
    correction: str | None = None,
    provider: str = "Gemini",
    attachments: list[InsightAttachment] | None = None,
    response_language: str = "한국어",
) -> InsightDecision:
    prompt = _decision_prompt(
        question, evidence_context, history, correction, response_language
    )
    attachments = attachments or []
    try:
        if provider == "OpenAI":
            response = client.responses.parse(
                model=model,
                instructions=INSIGHT_SYSTEM_PROMPT,
                input=_openai_input(prompt, attachments),
                text_format=InsightDecision,
            )
        else:
            response = client.models.generate_content(
                model=model,
                contents=_gemini_contents(prompt, attachments),
                config=_gemini_config(response_schema=InsightDecision),
            )
        return _parse_decision(response)
    except InsightAPIError:
        raise
    except Exception as exc:
        raise InsightAPIError(f"{provider} 요청 목적 분석에 실패했습니다: {exc}") from exc


def _generate_text(
    client,
    provider: str,
    model: str,
    system_prompt: str,
    prompt: str,
    attachments: list[InsightAttachment],
) -> str:
    try:
        if provider == "OpenAI":
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=_openai_input(prompt, attachments),
            )
            return (getattr(response, "output_text", "") or "").strip()
        response = client.models.generate_content(
            model=model,
            contents=_gemini_contents(prompt, attachments),
            config=_gemini_config(temperature=0.3),
        )
        return (getattr(response, "text", "") or "").strip()
    except Exception as exc:
        raise InsightAPIError(f"{provider} 후속 해석 생성에 실패했습니다: {exc}") from exc


def _chart_interpretation(
    client,
    provider: str,
    model: str,
    question: str,
    evidence_context: str,
    history: list[dict[str, Any]],
    chart_source: dict[str, Any],
    attachments: list[InsightAttachment],
    response_language: str = "한국어",
) -> str:
    prompt = f"""[사용자 요청]
{question}

[차트와 실제 통계자료]
{json.dumps(chart_source, ensure_ascii=False, indent=2, default=str)}

[전체 데이터 근거]
{evidence_context}

[기존 대화]
{bounded_history_text(history)}

[응답 언어 우선순위]
1. 현재 사용자 요청에 답변, 번역 또는 해석 언어가 명시되어 있으면 그 언어로 작성하세요.
2. 언어가 명시되지 않은 경우에만 아래 기본 응답 언어를 사용하세요.

[기본 응답 언어]
{response_language}
"""
    return _generate_text(
        client, provider, model, CHART_INSIGHT_SYSTEM_PROMPT, prompt, attachments
    ) or "차트는 생성되었지만 해석 텍스트가 없습니다."


def execute_request(
    *,
    api_key: str,
    model: str,
    question: str,
    frame: pd.DataFrame,
    source_filename: str,
    evidence_context: str,
    history: list[dict[str, Any]],
    provider: str = "Gemini",
    original_frame: pd.DataFrame | None = None,
    attachments: list[InsightAttachment] | None = None,
    response_language: str = "한국어",
) -> InsightExecution:
    attachments = attachments or []
    response_language = _resolve_response_language(question, response_language)
    client = _client(api_key) if provider == "Gemini" else _client(api_key, provider)
    correction = None
    for attempt in range(2):
        decision = plan_request(
            client,
            model,
            question,
            evidence_context,
            history,
            correction=correction,
            provider=provider,
            attachments=attachments,
            response_language=response_language,
        )
        if decision.action in {"chart", "chart_with_insight"}:
            requested_aggregation = _explicit_chart_aggregation(question)
            actual_aggregation = decision.chart_spec.aggregation.value
            if requested_aggregation and actual_aggregation != requested_aggregation:
                correction = (
                    f"사용자가 명시한 집계 방식은 {requested_aggregation}이지만 "
                    f"ChartSpec aggregation은 {actual_aggregation}입니다. "
                    "사용자 요청과 동일하게 수정하세요."
                )
                if attempt == 1:
                    raise InsightAPIError(correction)
                continue
        if decision.action in {"text", "business_insight"}:
            return InsightExecution(
                message=InsightMessage(role="model", text=decision.answer.strip())
            )
        try:
            if decision.action == "data":
                before_shape = tuple(frame.shape)
                code_result = execute_generated_code(
                    decision.python_code,
                    original_frame if original_frame is not None else frame,
                    frame,
                    allow_mutation=decision.modifies_df_clean,
                )
                code_record = InsightCodeRecord(
                    code=decision.python_code.strip(),
                    modifies_df_clean=decision.modifies_df_clean,
                    outputs=code_result.outputs,
                    before_shape=before_shape,
                    after_shape=tuple(code_result.frame.shape),
                )
                return InsightExecution(
                    message=InsightMessage(
                        role="model",
                        text="요청한 데이터 작업을 실행했습니다.",
                        code=code_record,
                    ),
                    updated_frame=code_result.frame if decision.modifies_df_clean else None,
                )

            spec = decision.chart_spec.to_chart_spec()
            chart_record, chart_source = rebuild_chart_record(
                frame, spec, source_filename
            )
            if decision.action == "chart_with_insight":
                answer = _chart_interpretation(
                    client,
                    provider,
                    model,
                    question,
                    evidence_context,
                    history,
                    chart_source,
                    attachments,
                    response_language,
                )
            else:
                answer = f"요청한 `{spec.title}` 차트를 생성했습니다."
            return InsightExecution(
                message=InsightMessage(role="model", text=answer, charts=[chart_record]),
                visualization_source=chart_source,
            )
        except (
            InsightCodeError,
            ValidationError,
            VisualizationDataError,
            ValueError,
            TypeError,
        ) as exc:
            correction = str(exc)
            if attempt == 1:
                raise InsightAPIError(f"생성 결과 검증 또는 실행에 실패했습니다: {exc}") from exc
    raise InsightAPIError("Insight 요청을 완료하지 못했습니다.")


def history_payload_bytes(
    history: list[dict[str, Any]],
    model: str,
    source_filename: str | None,
    frame: pd.DataFrame,
    provider: str = "Gemini",
) -> bytes:
    payload = InsightHistoryPayload(
        saved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        provider=provider,
        model=model,
        source_filename=source_filename,
        data_signature={
            "shape": list(frame.shape),
            "columns": [str(column) for column in frame.columns],
        },
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
        if message.get("code"):
            code = message["code"]
            lines.extend(["### 실행 코드", "", "```python", code.get("code", ""), "```", ""])
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


def restore_history(
    raw: bytes, filename: str
) -> tuple[list[dict[str, Any]], str | None]:
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
