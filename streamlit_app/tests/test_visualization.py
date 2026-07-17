from __future__ import annotations

import matplotlib
import pandas as pd
import pytest
from pydantic import ValidationError

matplotlib.use("Agg")

from visualization.models import ChartSpec, FigureSpec
from visualization.service import automatic_chart_title, build_visualization, figure_to_bytes, parse_text_request, source_payload
from visualization.statistics import build_statistics, variable_type_table


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "국가": ["한국", "한국", "미국", "일본", "미국", None],
            "연도": [2024, 2025, 2024, 2025, 2025, 2025],
            "매출": [10.0, 20.0, 30.0, 15.0, 25.0, 5.0],
            "만족도": [4.1, 4.3, 3.8, 4.0, 4.2, 3.9],
        }
    )


def test_pydantic_rejects_invalid_chart_settings() -> None:
    with pytest.raises(ValidationError):
        ChartSpec(chart_type="scatter_bubble", x="연도")
    with pytest.raises(ValidationError):
        ChartSpec(
            chart_type="bar",
            x="국가",
            advanced={"bar_mode": "stacked"},
        )


def test_variable_types_and_bar_statistics(sample_frame: pd.DataFrame) -> None:
    types = variable_type_table(sample_frame).set_index("변수명")
    assert types.loc["국가", "분석 타입"] == "범주형"
    assert types.loc["매출", "분석 타입"] == "수치형"
    spec = ChartSpec(chart_type="bar", x="국가", aggregation="sum", value_column="매출")
    table = build_statistics(sample_frame, spec)
    assert table.set_index("국가").loc["한국", "값"] == 30.0


def test_categorical_histogram_exposes_index_mapping(sample_frame: pd.DataFrame) -> None:
    spec = ChartSpec(chart_type="histogram", x="국가", advanced={"histogram_bins": 3})
    table = build_statistics(sample_frame, spec)
    mapping = table.attrs["category_mapping"]
    assert set(mapping.columns) == {"범주", "수치 인덱스"}
    assert int(table["값"].sum()) == 5


def test_all_chart_builders_and_exports(sample_frame: pd.DataFrame) -> None:
    specs = [
        ChartSpec(chart_type="bar", x="국가"),
        ChartSpec(chart_type="line", x="연도", aggregation="sum", value_column="매출"),
        ChartSpec(chart_type="scatter_bubble", x="매출", y="만족도", aggregation="count"),
        ChartSpec(chart_type="heatmap", x="국가", y="연도", aggregation="ratio"),
    ]
    result = build_visualization(sample_frame, specs, FigureSpec(grid_size=2))
    assert len(result.artifacts) == 4
    assert figure_to_bytes(result, "png").startswith(b"\x89PNG")
    assert figure_to_bytes(result, "pdf").startswith(b"%PDF")
    payload = source_payload(result, "sample.csv")
    assert len(payload["charts"]) == 4
    assert all(len(chart["insight"].splitlines()) <= 3 for chart in payload["charts"])


def test_text_request_uses_variables_in_sentence_order(sample_frame: pd.DataFrame) -> None:
    spec = parse_text_request("국가별 매출 합계 막대그래프", sample_frame, 1)[0]
    assert spec.x == "국가"
    assert spec.value_column == "매출"
    assert spec.aggregation.value == "sum"


def test_automatic_chart_title_uses_selected_variable_names_once() -> None:
    assert automatic_chart_title("국가", None, "지역", "매출", "국가") == "국가 · 지역 · 매출"


def test_axis_value_sorting_and_category_selection(sample_frame: pd.DataFrame) -> None:
    sorted_spec = ChartSpec(
        chart_type="bar",
        x="국가",
        advanced={"x_sort": "ascending", "y_sort": "descending", "top_n": None},
    )
    sorted_table = build_statistics(sample_frame, sorted_spec)
    assert sorted_table["국가"].tolist() == ["미국", "일본", "한국"]

    selected_spec = ChartSpec(
        chart_type="bar",
        x="국가",
        deep={"x_axis_mode": "category_select", "x_selected_categories": ["한국"]},
    )
    selected_table = build_statistics(sample_frame, selected_spec)
    assert selected_table["국가"].tolist() == ["한국"]


def test_grouped_bar_value_labels_line_curvature_and_numeric_axis(sample_frame: pd.DataFrame) -> None:
    grouped = ChartSpec(
        chart_type="bar",
        x="국가",
        group="연도",
        show_values=True,
        advanced={"bar_mode": "grouped", "top_n": None},
    )
    grouped_result = build_visualization(sample_frame, [grouped], FigureSpec(grid_size=1))
    assert any(text.get_text() for text in grouped_result.figure.axes[0].texts)

    curved = ChartSpec(
        chart_type="line",
        x="매출",
        aggregation="count",
        advanced={"line_curvature": 0.8, "x_sort": "ascending", "top_n": None},
        deep={
            "y_axis_mode": "numeric_range",
            "y_min": 0,
            "y_max": 100,
            "y_tick_interval": 20,
        },
    )
    curved_result = build_visualization(sample_frame, [curved], FigureSpec(grid_size=1))
    axis = curved_result.figure.axes[0]
    assert max(len(line.get_xdata()) for line in axis.lines) > sample_frame["매출"].nunique()
    assert axis.get_ylim() == pytest.approx((0, 100))


def test_donut_orders_slices_and_legend_by_label_or_value() -> None:
    frame = pd.DataFrame({"만족도": ["4", "5", "1", "3", "2", "4", "4", "5"]})
    by_label = ChartSpec(
        chart_type="pie",
        x="만족도",
        advanced={"donut": True, "pie_sort_by": "label", "pie_sort_direction": "ascending", "top_n": None},
    )
    label_table = build_statistics(frame, by_label)
    assert label_table["만족도"].tolist() == ["1", "2", "3", "4", "5"]

    by_value = ChartSpec(
        chart_type="pie",
        x="만족도",
        advanced={"donut": True, "pie_sort_by": "value", "pie_sort_direction": "descending", "top_n": None},
    )
    value_table = build_statistics(frame, by_value)
    assert value_table.iloc[0]["만족도"] == "4"
    result = build_visualization(frame, [by_label], FigureSpec(grid_size=1))
    legend_labels = [text.get_text() for text in result.figure.axes[0].get_legend().texts]
    assert legend_labels == ["1", "2", "3", "4", "5"]
