import pandas as pd
import pytest

from data.preprocessing import (
    PreprocessingError,
    apply_missing_plan,
    comparison_summary,
    date_column_candidates,
    drop_columns,
    fill_missing_values,
    missing_value_summary,
    paginate,
    replace_multiple_values,
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


def test_unique_values_sort_by_unique_value_descending_and_paginate_by_twenty():
    frame = pd.DataFrame({"x": list(range(25)) + [20, 20, None]})
    values = unique_value_counts(frame, "x")
    page, pages = paginate(values, 2)
    assert len(page) == 6
    assert pages == 2
    assert values.iloc[0]["값"] == 24
    assert values.iloc[-1]["표시값"] == "<결측값>"
    assert "<결측값>" in values["표시값"].tolist()


def test_date_detection_and_selected_component_split():
    frame = pd.DataFrame({"created": ["2026-07-15 09:30:45", "2026-08-16 17:00:00"]})
    candidates = date_column_candidates(frame)
    result = split_date_components(frame, "created", ["year_month_day", "month_day", "day", "hour"])

    assert candidates["변수명"].tolist() == ["created"]
    assert result.frame["created_년월일"].tolist() == ["2026년 07월 15일", "2026년 08월 16일"]
    assert result.frame["created_월일"].tolist() == ["07월15일", "08월16일"]
    assert result.frame["created_일"].tolist() == ["15일", "16일"]
    assert result.frame["created_시간"].tolist() == ["09시", "17시"]


def test_batch_missing_plan_and_multiple_replacements_preserve_input():
    frame = pd.DataFrame({"category": ["A", None, "B"], "value": [1.0, None, 3.0]})
    missing = apply_missing_plan(
        frame,
        [
            {"변수명": "category", "처리방법": "특정값", "처리값": "미입력"},
            {"변수명": "value", "처리방법": "평균값", "처리값": ""},
        ],
    )
    replaced = replace_multiple_values(missing.frame, "category", [("A", "통합"), ("B", "통합")])
    assert replaced.frame["category"].tolist() == ["통합", "미입력", "통합"]
    assert replaced.frame["value"].tolist() == [1.0, 2.0, 3.0]
    assert pd.isna(frame.loc[1, "category"])


def test_multiple_replacements_do_not_cascade():
    frame = pd.DataFrame({"x": ["A", "B", "C"]})
    result = replace_multiple_values(frame, "x", [("A", "B"), ("B", "C")])
    assert result.frame["x"].tolist() == ["B", "C", "C"]


def test_display_equivalent_unique_values_are_combined_after_processing():
    frame = pd.DataFrame({"x": pd.Series([20, "20", 10, None], dtype="object")})
    before = unique_value_counts(frame, "x").set_index("표시값")
    assert before.loc["20", "개수"] == 2
    assert before.loc["20", "데이터 타입"] == "int, str"

    replaced = replace_multiple_values(frame, "x", [(10, "20")])
    after = unique_value_counts(replaced.frame, "x").set_index("표시값")
    assert after.loc["20", "개수"] == 3
    assert after.index.tolist().count("20") == 1
    assert all(isinstance(value, int) for value in replaced.frame["x"].dropna())

    filled = fill_missing_values(replaced.frame, "x", "특정값", "20")
    final = unique_value_counts(filled.frame, "x").set_index("표시값")
    assert final.loc["20", "개수"] == 4
    assert all(isinstance(value, int) for value in filled.frame["x"])


def test_drop_selected_columns_and_reject_all_columns():
    frame = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = drop_columns(frame, ["b", "c"])
    assert result.frame.columns.tolist() == ["a"]
    assert frame.columns.tolist() == ["a", "b", "c"]
    with pytest.raises(PreprocessingError):
        drop_columns(frame, ["a", "b", "c"])


def test_comparison_summary_reports_before_and_after():
    before = pd.DataFrame({"x": [1, None]})
    after = pd.DataFrame({"x": [1]})
    summary = comparison_summary(before, after).set_index("항목")
    assert summary.loc["행 수", "변화"] == -1
    assert summary.loc["전체 결측 개수", "변화"] == -1
