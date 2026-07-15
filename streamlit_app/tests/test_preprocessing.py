import pandas as pd
import pytest

from data.preprocessing import (
    PreprocessingError,
    comparison_summary,
    date_column_candidates,
    fill_missing_values,
    missing_value_summary,
    noise_candidates,
    paginate,
    replace_selected_value,
    split_date_components,
    unique_value_counts,
)


def test_missing_summary_and_specific_fill_preserve_input():
    frame = pd.DataFrame({"category": ["A", None], "value": [1.0, None]})
    summary = missing_value_summary(frame)
    result = fill_missing_values(frame, "category", "특정값", "미입력")

    assert summary["결측 개수"].tolist() == [1, 1]
    assert result.frame["category"].tolist() == ["A", "미입력"]
    assert pd.isna(frame.loc[1, "category"])


@pytest.mark.parametrize(("method", "expected"), [("평균값", 2.0), ("중앙값", 2.0)])
def test_numeric_missing_fill(method, expected):
    frame = pd.DataFrame({"value": [1.0, None, 3.0]})
    result = fill_missing_values(frame, "value", method)
    assert result.frame.loc[1, "value"] == expected


def test_missing_row_delete():
    frame = pd.DataFrame({"value": [1, None, 3]})
    result = fill_missing_values(frame, "value", "해당 행 삭제")
    assert len(result.frame) == 2
    assert result.affected_rows == 1


def test_mean_rejects_text_column():
    with pytest.raises(PreprocessingError):
        fill_missing_values(pd.DataFrame({"x": ["A", None]}), "x", "평균값")


def test_noise_detection_and_replacement():
    frame = pd.DataFrame({"category": ["Seoul", " Seoul ", "seoul", "Busan"] * 30 + ["Rare"]})
    candidates = noise_candidates(frame, "category")
    assert "앞뒤·중복 공백" in " ".join(candidates["탐지 근거"])
    result = replace_selected_value(frame, "category", " Seoul ", "Seoul")
    assert result.affected_rows == 30
    assert " Seoul " not in result.frame["category"].tolist()


def test_unique_values_include_missing_and_paginate_by_twenty():
    frame = pd.DataFrame({"x": list(range(25)) + [None]})
    values = unique_value_counts(frame, "x")
    page, pages = paginate(values, 2)
    assert len(page) == 6
    assert pages == 2
    assert "<결측값>" in values["표시값"].tolist()


def test_date_detection_and_selected_component_split():
    frame = pd.DataFrame({"created": ["2026-07-15 09:30:45", "2026-08-16 17:00:00"]})
    candidates = date_column_candidates(frame)
    result = split_date_components(frame, "created", ["year_month", "hour"])

    assert candidates["변수명"].tolist() == ["created"]
    assert result.frame["created_년월"].tolist() == ["2026년-07월", "2026년-08월"]
    assert result.frame["created_시간"].tolist() == ["09시", "17시"]
    assert "created_월" not in result.frame


def test_comparison_summary_reports_before_and_after():
    before = pd.DataFrame({"x": [1, None]})
    after = pd.DataFrame({"x": [1]})
    summary = comparison_summary(before, after).set_index("항목")
    assert summary.loc["행 수", "변화"] == -1
    assert summary.loc["전체 결측 개수", "변화"] == -1
