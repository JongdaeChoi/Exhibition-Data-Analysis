from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from insight.context import build_evidence_context
from insight.models import InsightChartInput, InsightDecision, InsightMessage
from insight.service import (
    execute_request,
    history_markdown_bytes,
    history_payload_bytes,
    restore_history,
)


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
