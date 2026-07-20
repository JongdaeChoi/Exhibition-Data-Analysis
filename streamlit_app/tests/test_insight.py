from __future__ import annotations

import json
import sys
import types

import pandas as pd
import pytest
from pydantic import ValidationError

from insight.context import build_evidence_context
from insight.models import InsightChartInput, InsightDecision, InsightMessage
from insight.service import (
    _decision_prompt,
    _gemini_json_schema,
    execute_request,
    history_markdown_bytes,
    history_payload_bytes,
    plan_request,
    rebuild_chart_record,
    restore_history,
)
from ui.insight_view import _configured_api_key


@pytest.fixture
def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    original = pd.DataFrame(
        {"국가": ["한국", "미국", None], "매출": [10.0, 20.0, None], "연도": [2024, 2025, 2025]}
    )
    clean = original.copy(deep=True)
    clean["국가"] = clean["국가"].fillna("미입력")
    clean["매출"] = clean["매출"].fillna(clean["매출"].mean())
    return original, clean


def test_context_contains_all_required_evidence_sections(sample_frames) -> None:
    original, clean = sample_frames
    context = build_evidence_context(
        original,
        clean,
        "sample.csv",
        [{"message": "결측값 2개를 처리했습니다."}],
        [{"charts": [{"statistics": [{"국가": "한국", "값": 1}]}]}],
    )
    for heading in (
        "[데이터 기본 구조]",
        "[전처리 이전·이후 비교]",
        "[전처리 이력]",
        "[주요 기술통계]",
        "[Visualization 저장 통계자료]",
    ):
        assert heading in context


def test_insight_decision_requires_text_or_valid_chart() -> None:
    with pytest.raises(ValidationError):
        InsightDecision(action="text", answer="")
    with pytest.raises(ValidationError):
        InsightDecision(action="chart")

    chart = InsightChartInput(chart_type="bar", x="국가", aggregation="sum", value_column="매출")
    spec = chart.to_chart_spec()
    assert spec.x == "국가"
    assert spec.value_column == "매출"
    assert spec.title == "국가 · 매출"


def test_gemini_schema_uses_supported_json_schema_subset() -> None:
    schema = _gemini_json_schema(InsightDecision)
    encoded = json.dumps(schema, ensure_ascii=False)
    assert schema["type"] == "object"
    assert "action" in schema["properties"]
    assert "additionalProperties" not in encoded
    assert '"default"' not in encoded
    chart_schema = schema["$defs"]["InsightChartInput"]
    assert "aggregation" in chart_schema["required"]


def test_decision_prompt_requires_selected_response_language() -> None:
    prompt = _decision_prompt(
        "Summarize the data", "evidence", [], response_language="English"
    )
    assert "[응답 언어]" in prompt
    assert "English" in prompt


def test_history_json_and_markdown_round_trip(sample_frames) -> None:
    _, clean = sample_frames
    history = [
        InsightMessage(role="user", text="요약해 줘").model_dump(mode="json"),
        InsightMessage(role="model", text="핵심 요약입니다.").model_dump(mode="json"),
    ]
    payload = history_payload_bytes(history, "gemini-2.5-flash", "sample.csv", clean)
    restored, model = restore_history(payload, "insight.json")
    assert model == "gemini-2.5-flash"
    assert [item["text"] for item in restored] == ["요약해 줘", "핵심 요약입니다."]
    markdown = history_markdown_bytes(restored, "sample.csv").decode("utf-8-sig")
    assert "# 비즈니스 인사이트" in markdown
    assert "핵심 요약입니다." in markdown
    assert json.loads(payload)["data_signature"]["shape"] == [3, 3]


def test_chart_request_uses_existing_visualization_pipeline(monkeypatch, sample_frames) -> None:
    _, clean = sample_frames
    monkeypatch.setattr("insight.service._client", lambda api_key: object())
    monkeypatch.setattr(
        "insight.service.plan_request",
        lambda *args, **kwargs: InsightDecision(
            action="chart",
            chart_spec=InsightChartInput(chart_type="bar", x="국가", aggregation="count"),
        ),
    )
    execution = execute_request(
        api_key="test-key",
        model="gemini-2.5-flash",
        question="국가별 차트",
        frame=clean,
        source_filename="sample.csv",
        evidence_context="근거",
        history=[],
    )
    assert execution.visualization_source is not None
    assert execution.message.charts[0].image_base64.startswith("iVBOR")
    assert execution.visualization_source["charts"][0]["spec"]["chart_type"] == "bar"


def test_edited_pydantic_spec_rebuilds_chart_with_shared_pipeline(sample_frames) -> None:
    _, clean = sample_frames
    record, source = rebuild_chart_record(
        clean,
        {
            "chart_type": "multi_variable",
            "variables": ["매출", "연도"],
            "aggregation": "mean",
            "comparison_chart": "bar",
            "title": "수치형 평균 비교",
        },
        "sample.csv",
    )
    assert record.image_base64.startswith("iVBOR")
    assert source["charts"][0]["spec"]["aggregation"] == "mean"
    values = {row["변수"]: row["값"] for row in source["charts"][0]["statistics"]}
    assert values["매출"] == 15.0
    assert values["연도"] == pytest.approx(2024.6666666667)


def test_chart_history_downloads_as_json_and_markdown(sample_frames) -> None:
    _, clean = sample_frames
    record, _ = rebuild_chart_record(
        clean,
        {
            "chart_type": "correlation_heatmap",
            "variables": ["매출", "연도"],
            "aggregation": "count",
            "title": "수치형 변수 상관관계",
        },
        "sample.csv",
    )
    history = [
        InsightMessage(
            role="model", text="상관관계 차트입니다.", charts=[record]
        ).model_dump(mode="json")
    ]

    json_download = history_payload_bytes(
        history, "gemini-2.5-flash", "sample.csv", clean
    )
    markdown_download = history_markdown_bytes(history, "sample.csv")

    payload = json.loads(json_download)
    assert payload["history"][0]["charts"][0]["image_base64"].startswith("iVBOR")
    markdown = markdown_download.decode("utf-8-sig")
    assert "상관관계 차트입니다." in markdown
    assert '"chart_type": "correlation_heatmap"' in markdown


def test_invalid_chart_is_replanned_once(monkeypatch, sample_frames) -> None:
    _, clean = sample_frames
    corrections = []

    def fake_plan(*args, **kwargs):
        corrections.append(kwargs.get("correction"))
        column = "없는 변수" if len(corrections) == 1 else "국가"
        return InsightDecision(
            action="chart",
            chart_spec=InsightChartInput(chart_type="bar", x=column, aggregation="count"),
        )

    monkeypatch.setattr("insight.service._client", lambda api_key: object())
    monkeypatch.setattr("insight.service.plan_request", fake_plan)
    execution = execute_request(
        api_key="test-key",
        model="gemini-2.5-flash",
        question="차트를 만들어 줘",
        frame=clean,
        source_filename="sample.csv",
        evidence_context="근거",
        history=[],
    )
    assert corrections[0] is None
    assert "찾을 수 없습니다" in corrections[1]
    assert execution.message.charts


def test_explicit_mean_chart_request_rejects_count_default(monkeypatch, sample_frames) -> None:
    _, clean = sample_frames
    corrections = []

    def fake_plan(*args, **kwargs):
        corrections.append(kwargs.get("correction"))
        aggregation = "count" if len(corrections) == 1 else "mean"
        return InsightDecision(
            action="chart",
            chart_spec=InsightChartInput(
                chart_type="multi_variable",
                variables=["매출", "연도"],
                aggregation=aggregation,
            ),
        )

    monkeypatch.setattr("insight.service._client", lambda api_key: object())
    monkeypatch.setattr("insight.service.plan_request", fake_plan)
    execution = execute_request(
        api_key="test-key",
        model="gemini-2.5-flash",
        question="수치형 변수의 mean 값으로 multivariable bar chart를 그려줘",
        frame=clean,
        source_filename="sample.csv",
        evidence_context="근거",
        history=[],
    )
    assert corrections[0] is None
    assert "aggregation은 count" in corrections[1]
    chart = execution.visualization_source["charts"][0]
    assert chart["spec"]["aggregation"] == "mean"
    assert {row["변수"]: row["값"] for row in chart["statistics"]} == {
        "매출": 15.0,
        "연도": pytest.approx(2024.6666666667),
    }


def test_api_key_lookup_does_not_access_colab_kernel(monkeypatch) -> None:
    class UnavailableUserData:
        @staticmethod
        def get(name: str):
            raise AssertionError("Streamlit 프로세스에서 Colab 커널을 조회하면 안 됩니다.")

    fake_colab = types.ModuleType("google.colab")
    fake_colab.userdata = UnavailableUserData()
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    api_key, source = _configured_api_key("")

    assert api_key is None
    assert source == "미설정"


def test_openai_provider_uses_structured_decision_api(sample_frames) -> None:
    class Response:
        output_parsed = InsightDecision(action="text", answer="요청한 설명입니다.")

    class Responses:
        def __init__(self):
            self.kwargs = None

        def parse(self, **kwargs):
            self.kwargs = kwargs
            return Response()

    class Client:
        def __init__(self):
            self.responses = Responses()

    client = Client()
    decision = plan_request(
        client,
        "gpt-5.6-terra",
        "설명해 줘",
        "데이터 근거",
        [],
        provider="OpenAI",
    )
    assert decision.answer == "요청한 설명입니다."
    assert client.responses.kwargs["model"] == "gpt-5.6-terra"
    assert client.responses.kwargs["text_format"] is InsightDecision
