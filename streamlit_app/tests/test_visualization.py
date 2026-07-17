from __future__ import annotations

from datetime import date

import matplotlib
import pandas as pd
import pytest
from matplotlib.patches import Circle, Shadow, Wedge
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
    result = build_visualization(sample_frame, specs, FigureSpec(rows=2, columns=2))
    assert len(result.artifacts) == 4
    assert figure_to_bytes(result, "png").startswith(b"\x89PNG")
    assert figure_to_bytes(result, "pdf").startswith(b"%PDF")
    payload = source_payload(result, "sample.csv")
    assert len(payload["charts"]) == 4
    assert all(len(chart["insight"].splitlines()) <= 3 for chart in payload["charts"])


def test_rectangular_subplot_layout_and_chart_count(sample_frame: pd.DataFrame) -> None:
    specs = [
        ChartSpec(chart_type="bar", x="국가"),
        ChartSpec(chart_type="line", x="연도", aggregation="sum", value_column="매출"),
    ]
    figure_spec = FigureSpec(rows=1, columns=2)
    result = build_visualization(sample_frame, specs, figure_spec)
    grid = result.figure.axes[0].get_subplotspec().get_gridspec()
    assert (grid.nrows, grid.ncols) == (1, 2)
    assert len(result.artifacts) == 2

    with pytest.raises(ValueError, match="1×2 subplot에는 2개"):
        build_visualization(sample_frame, specs[:1], figure_spec)
    with pytest.raises(ValidationError):
        FigureSpec(rows=0, columns=1)


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
    grouped_result = build_visualization(sample_frame, [grouped], FigureSpec(rows=1, columns=1))
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
    curved_result = build_visualization(sample_frame, [curved], FigureSpec(rows=1, columns=1))
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
    result = build_visualization(frame, [by_label], FigureSpec(rows=1, columns=1))
    legend_labels = [text.get_text() for text in result.figure.axes[0].get_legend().texts]
    assert legend_labels == ["1", "2", "3", "4", "5"]


def test_data_label_style_position_and_visibility(sample_frame: pd.DataFrame) -> None:
    styled = ChartSpec(
        chart_type="bar",
        x="국가",
        show_values=True,
        advanced={
            "label_position_mode": "manual",
            "label_offset_x": 7,
            "label_offset_y": 11,
            "label_font_size": 14,
            "label_color": "#FF0000",
            "top_n": None,
        },
    )
    result = build_visualization(sample_frame, [styled], FigureSpec(rows=1, columns=1))
    labels = result.figure.axes[0].texts
    assert labels
    assert labels[0].get_position() == (7, 11)
    assert labels[0].get_fontsize() == 14
    assert labels[0].get_color().lower() == "#ff0000"

    hidden = styled.model_copy(update={"show_values": False})
    hidden_result = build_visualization(sample_frame, [hidden], FigureSpec(rows=1, columns=1))
    assert not hidden_result.figure.axes[0].texts


def test_donut_geometry_center_style_and_pie_label_modes() -> None:
    frame = pd.DataFrame({"만족도": ["1", "2", "2", "3"]})
    spec = ChartSpec(
        chart_type="pie",
        x="만족도",
        advanced={
            "donut": True,
            "donut_hole_size": 0.45,
            "donut_ring_width": 0.35,
            "donut_center_color": "#FFF7ED",
            "donut_center_border": True,
            "donut_center_border_color": "#EA580C",
            "donut_center_border_width": 2,
            "donut_center_border_alpha": 0.55,
            "pie_label_mode": "label_ratio",
            "pie_ratio_format": ".0f",
            "pie_explode_labels": ["2"],
            "pie_explode_width": 0.2,
            "pie_shadow": True,
            "pie_shadow_width": 0.06,
            "pie_shadow_color": "#111827",
            "pie_shadow_alpha": 0.4,
            "pie_edge_alpha": 0.45,
            "top_n": None,
        },
    )
    result = build_visualization(frame, [spec], FigureSpec(rows=1, columns=1))
    axis = result.figure.axes[0]
    centers = [patch for patch in axis.patches if isinstance(patch, Circle)]
    wedges = [patch for patch in axis.patches if isinstance(patch, Wedge) and not isinstance(patch, Shadow)]
    shadows = [patch for patch in axis.patches if isinstance(patch, Shadow)]
    assert centers and centers[0].get_radius() == pytest.approx(0.45)
    assert centers[0].get_linewidth() == pytest.approx(2)
    assert centers[0].get_edgecolor()[3] == pytest.approx(0.55)
    assert shadows and all(shadow.get_alpha() == pytest.approx(0.4) for shadow in shadows)
    assert centers[0].get_zorder() > max(shadow.get_zorder() for shadow in shadows)
    assert any(wedge.center != (0.0, 0.0) for wedge in wedges)
    assert all(wedge.get_edgecolor()[3] == pytest.approx(0.45) for wedge in wedges)
    texts = [text.get_text() for text in axis.texts]
    assert {"1", "2", "3"}.issubset(texts)
    assert "25%" in texts and "50%" in texts
    with pytest.raises(ValidationError):
        ChartSpec(
            chart_type="pie",
            x="만족도",
            advanced={"donut": True, "donut_hole_size": 0.8, "donut_ring_width": 0.4},
        )


def test_reference_lines_support_category_and_numeric_axes(sample_frame: pd.DataFrame) -> None:
    spec = ChartSpec(
        chart_type="bar",
        x="국가",
        deep={
            "reference_enabled": True,
            "reference_targets": ["x", "y"],
            "reference_line_style": ":",
            "reference_line_width": 2.5,
            "reference_line_alpha": 0.6,
            "reference_label": "기준",
            "reference_label_size": 12,
            "reference_label_alpha": 0.7,
            "x_reference_kind": "category",
            "x_reference_value": "한국",
            "y_reference_kind": "numeric",
            "y_reference_value": 1.5,
        },
    )
    result = build_visualization(sample_frame, [spec], FigureSpec(rows=1, columns=1))
    axis = result.figure.axes[0]
    reference_lines = [line for line in axis.lines if line.get_linestyle() == ":"]
    assert len(reference_lines) == 2
    assert all(line.get_linewidth() == pytest.approx(2.5) for line in reference_lines)
    assert all(line.get_alpha() == pytest.approx(0.6) for line in reference_lines)
    assert [text.get_text() for text in axis.texts].count("기준") == 2


def test_reference_line_supports_date_axis() -> None:
    frame = pd.DataFrame(
        {
            "조사일자": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "매출": [10, 20, 15],
        }
    )
    spec = ChartSpec(
        chart_type="line",
        x="조사일자",
        value_column="매출",
        aggregation="sum",
        deep={
            "reference_enabled": True,
            "reference_targets": ["x"],
            "reference_line_style": "--",
            "reference_label": "기준일",
            "x_reference_kind": "date",
            "x_reference_value": date(2025, 2, 1),
        },
    )
    result = build_visualization(frame, [spec], FigureSpec(rows=1, columns=1))
    axis = result.figure.axes[0]
    reference_lines = [line for line in axis.lines if line.get_linestyle() == "--"]
    assert len(reference_lines) == 1
    assert "기준일" in [text.get_text() for text in axis.texts]
