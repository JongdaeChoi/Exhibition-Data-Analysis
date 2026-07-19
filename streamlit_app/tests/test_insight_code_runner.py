from __future__ import annotations

import pandas as pd
import pytest

from insight.code_runner import InsightCodeError, execute_generated_code, validate_generated_code
from insight.context import build_evidence_context
from insight.models import InsightAttachment, InsightDecision, PROVIDER_MODELS


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    original = pd.DataFrame({"국가": ["한국", "미국", "한국"], "매출": [10.0, 20.0, None]})
    return original, original.copy(deep=True)


def test_data_decision_requires_code_and_supports_provider_models() -> None:
    with pytest.raises(ValueError):
        InsightDecision(action="data")
    decision = InsightDecision(
        action="data",
        python_code='result_df = df_clean.groupby("국가", as_index=False).size()\ndisplay(result_df)',
    )
    assert decision.action == "data"
    assert "gpt-5.6-sol" in PROVIDER_MODELS["OpenAI"]


def test_query_code_returns_table_without_mutating_frames() -> None:
    original, clean = _frames()
    result = execute_generated_code(
        'result_df = df_clean.groupby("국가", as_index=False).size()\ndisplay(result_df)',
        original,
        clean,
        allow_mutation=False,
    )
    assert result.outputs[0]["type"] == "dataframe"
    assert result.outputs[0]["row_count"] == 2
    pd.testing.assert_frame_equal(original, clean)
    pd.testing.assert_frame_equal(result.frame, clean)


def test_explicit_mutation_updates_only_clean_copy() -> None:
    original, clean = _frames()
    result = execute_generated_code(
        'df_clean["매출"] = df_clean["매출"].fillna(0)\ndisplay(df_clean)',
        original,
        clean,
        allow_mutation=True,
    )
    assert result.frame["매출"].isna().sum() == 0
    assert clean["매출"].isna().sum() == 1
    assert original["매출"].isna().sum() == 1


def test_query_cannot_mutate_clean_and_unsafe_io_is_blocked() -> None:
    original, clean = _frames()
    with pytest.raises(InsightCodeError, match="조회 요청"):
        execute_generated_code(
            'df_clean["신규"] = 1\ndisplay(df_clean)', original, clean, allow_mutation=False
        )
    with pytest.raises(InsightCodeError, match="지원하지 않는 구문"):
        validate_generated_code("import os")
    with pytest.raises(InsightCodeError, match="사용할 수 없는 함수 또는 속성"):
        validate_generated_code('df_clean.to_csv("result.csv")')


def test_uploaded_reference_is_added_to_evidence_context() -> None:
    original, clean = _frames()
    attachment = InsightAttachment(
        id="doc1",
        filename="goal.txt",
        mime_type="text/plain",
        kind="document",
        extracted_text="올해 목표 매출은 100입니다.",
    )
    context = build_evidence_context(original, clean, "sample.csv", [], [], [attachment])
    assert "[사용자 업로드 참고자료]" in context
    assert "올해 목표 매출은 100입니다." in context
