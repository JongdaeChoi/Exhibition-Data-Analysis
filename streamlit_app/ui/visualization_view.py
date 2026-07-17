from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from visualization.models import AdvancedSettings, ChartSpec, DeepSettings, FigureSpec
from visualization.service import (
    automatic_chart_title,
    build_visualization,
    figure_to_bytes,
    parse_text_request,
    source_payload,
    source_payload_bytes,
)
from visualization.statistics import VisualizationDataError, variable_type_table


CHART_LABELS = {
    "bar": "Bar chart",
    "line": "Line chart",
    "pie": "Pie chart",
    "histogram": "Histogram",
    "scatter_bubble": "Scatter bubble",
    "heatmap": "Heatmap",
}
AGGREGATION_LABELS = {"count": "개수", "sum": "합계", "mean": "평균", "ratio": "비율"}
NONE_OPTION = "(없음)"


def _optional_column(label: str, columns: list[str], key: str, disabled: bool = False) -> str | None:
    value = st.selectbox(label, [NONE_OPTION] + columns, key=key, disabled=disabled)
    return None if value == NONE_OPTION else value


def _number_or_none(label: str, key: str, disabled: bool = False) -> float | None:
    enabled = st.checkbox(f"{label} 사용", key=f"{key}_enabled", disabled=disabled)
    if not enabled:
        return None
    return st.number_input(label, value=0.0, key=key, disabled=disabled)


def _figure_controls(grid_size: int) -> FigureSpec:
    st.markdown("### Figure 공통 설정")
    c1, c2, c3 = st.columns(3)
    width = c1.number_input("Figure 가로", 4.0, 30.0, float(max(10, grid_size * 6)), 0.5, key="viz_fig_width")
    height = c2.number_input("Figure 세로", 3.0, 30.0, float(max(7, grid_size * 5)), 0.5, key="viz_fig_height")
    dpi = c3.number_input("DPI", 72, 600, 120, 10, key="viz_dpi")
    c4, c5, c6 = st.columns(3)
    figure_background = c4.color_picker("Figure 배경색", "#FFFFFF", key="viz_figure_bg")
    axes_background = c5.color_picker("차트 영역 배경색", "#FFFFFF", key="viz_axes_bg")
    font_color = c6.color_picker("폰트 색상", "#172033", key="viz_font_color")
    c7, c8, c9 = st.columns(3)
    horizontal_space = c7.slider("subplot 가로 간격", 0.0, 1.0, 0.28, 0.02, key="viz_wspace")
    vertical_space = c8.slider("subplot 세로 간격", 0.0, 1.0, 0.35, 0.02, key="viz_hspace")
    filename = c9.text_input("출력 파일명", "visualization", key="viz_filename")
    c10, c11, c12, c13 = st.columns(4)
    tight_layout = c10.checkbox("tight_layout", True, key="viz_tight")
    constrained_layout = c11.checkbox("constrained_layout", False, key="viz_constrained")
    transparent = c12.checkbox("투명 배경", False, key="viz_transparent")
    font_family = c13.selectbox(
        "폰트 패밀리",
        ["NanumGothic", "DejaVu Sans", "sans-serif"],
        key="viz_font_family",
    )
    return FigureSpec(
        grid_size=grid_size,
        width=width,
        height=height,
        dpi=dpi,
        figure_background=figure_background,
        axes_background=axes_background,
        horizontal_space=horizontal_space,
        vertical_space=vertical_space,
        tight_layout=tight_layout,
        constrained_layout=constrained_layout,
        font_family=font_family,
        font_color=font_color,
        transparent=transparent,
        filename=filename or "visualization",
    )


def _basic_controls(index: int, columns: list[str], numeric_columns: list[str]) -> dict:
    prefix = f"viz_{index}"
    chart_type = st.selectbox(
        "차트 유형",
        list(CHART_LABELS),
        format_func=lambda item: CHART_LABELS[item],
        key=f"{prefix}_type",
    )
    requires_y = chart_type in {"scatter_bubble", "heatmap"}
    supports_group = chart_type in {"bar", "line", "scatter_bubble"}
    c1, c2 = st.columns(2)
    x = c1.selectbox("X축 변수", columns, key=f"{prefix}_x")
    y = c2.selectbox("Y축 변수", columns, index=min(1, len(columns) - 1), key=f"{prefix}_y", disabled=not requires_y)
    c3, c4 = st.columns(2)
    aggregation_options = ["count", "sum", "mean", "ratio"]
    if chart_type == "histogram":
        aggregation_options = ["count", "sum"]
    elif chart_type == "scatter_bubble":
        aggregation_options = ["count", "sum", "ratio"]
    aggregation = c3.selectbox(
        "집계 방식",
        aggregation_options,
        format_func=lambda item: AGGREGATION_LABELS[item],
        key=f"{prefix}_{chart_type}_aggregation",
    )
    value_column = c4.selectbox(
        "집계 대상",
        [NONE_OPTION] + numeric_columns,
        key=f"{prefix}_{chart_type}_value",
        disabled=aggregation not in {"sum", "mean"},
    )
    c5, c6 = st.columns(2)
    group = c5.selectbox(
        "그룹/색상 변수",
        [NONE_OPTION] + columns,
        key=f"{prefix}_group",
        disabled=not supports_group,
    )
    selected_y = y if requires_y else None
    selected_group = None if group == NONE_OPTION or not supports_group else group
    selected_value = None if value_column == NONE_OPTION or aggregation not in {"sum", "mean"} else value_column
    title_key = f"{prefix}_title"
    title_signature_key = f"{prefix}_title_variable_signature"
    title_signature = (chart_type, x, selected_y, selected_group, selected_value)
    if st.session_state.get(title_signature_key) != title_signature:
        st.session_state[title_key] = automatic_chart_title(x, selected_y, selected_group, selected_value)
        st.session_state[title_signature_key] = title_signature
    title = c6.text_input("차트 제목", key=title_key)
    c7, c8 = st.columns(2)
    x_label = c7.text_input("X축 제목", x, key=f"{prefix}_xlabel")
    y_label = c8.text_input("Y축 제목", "값", key=f"{prefix}_ylabel")
    show_values = st.checkbox("값 표시", True, key=f"{prefix}_show_values", disabled=chart_type in {"scatter_bubble", "heatmap"})
    if chart_type in {"scatter_bubble", "heatmap"}:
        show_values = False
    return {
        "chart_type": chart_type,
        "x": x,
        "y": selected_y,
        "group": selected_group,
        "value_column": selected_value,
        "aggregation": aggregation,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "show_values": show_values,
    }


def _advanced_controls(index: int, basic: dict) -> AdvancedSettings:
    prefix = f"viz_{index}_adv"
    chart_type = basic["chart_type"]
    show_values = basic["show_values"]
    c1, c2, c3, c4 = st.columns(4)
    sort_options = ["none", "ascending", "descending"]
    sort_labels = {"none": "정렬 안 함", "ascending": "오름차순", "descending": "내림차순"}
    x_sort = c1.selectbox(
        "X축 값 정렬",
        sort_options,
        format_func=lambda item: sort_labels[item],
        key=f"{prefix}_x_sort",
        disabled=chart_type == "pie",
    )
    y_sort = c2.selectbox(
        "Y축 값 정렬",
        sort_options,
        format_func=lambda item: sort_labels[item],
        key=f"{prefix}_y_sort",
        disabled=chart_type == "pie",
    )
    if chart_type == "pie":
        x_sort = y_sort = "none"
    top_n_enabled = c3.checkbox("상위 N개 제한", True, key=f"{prefix}_top_enabled")
    top_n = c4.number_input("상위 N", 1, 500, 20, key=f"{prefix}_top", disabled=not top_n_enabled)
    c4, c5, c6 = st.columns(3)
    include_missing = c4.checkbox("결측값 포함", False, key=f"{prefix}_missing")
    palette = c5.selectbox("색상 팔레트", ["Blues", "viridis", "magma", "Set2", "tab10", "coolwarm"], key=f"{prefix}_palette")
    base_color = c6.color_picker("기본 색상", "#2563EB", key=f"{prefix}_color")
    c7, c8, c9 = st.columns(3)
    alpha = c7.slider("투명도", 0.05, 1.0, 0.85, 0.05, key=f"{prefix}_alpha")
    edge_color = c8.color_picker("테두리 색상", "#334155", key=f"{prefix}_edge_color")
    edge_width = c9.slider("테두리 두께", 0.0, 5.0, 0.6, 0.1, key=f"{prefix}_edge_width")
    c10, c11, c12 = st.columns(3)
    grid = c10.checkbox("격자 표시", True, key=f"{prefix}_grid")
    grid_axis = c11.selectbox("격자 방향", ["x", "y", "both"], index=1, key=f"{prefix}_grid_axis", disabled=not grid)
    tick_rotation = c12.slider("축 눈금 회전", -90, 90, 0, 5, key=f"{prefix}_rotation")
    c13, c14, c15 = st.columns(3)
    legend = c13.checkbox("범례 표시", True, key=f"{prefix}_legend")
    legend_location = c14.selectbox("범례 위치", ["best", "upper right", "upper left", "lower right", "lower left", "center left", "center right"], key=f"{prefix}_legend_location", disabled=not legend)
    number_format = c15.selectbox("숫자 형식", [",.0f", ",.1f", ",.2f", ".1%"], index=1, key=f"{prefix}_format")
    c16, c17, c18 = st.columns(3)
    title_size = c16.slider("제목 글자 크기", 6, 40, 13, key=f"{prefix}_title_size")
    axis_size = c17.slider("축 글자 크기", 6, 30, 10, key=f"{prefix}_axis_size")
    unit = c18.text_input("표시 단위", "", key=f"{prefix}_unit")

    st.markdown("##### 값(Data Label) 설정")
    label_a, label_b, label_c = st.columns(3)
    label_position_mode = label_a.selectbox(
        "표시 위치",
        ["auto", "manual"],
        format_func=lambda item: {"auto": "자동", "manual": "직접 입력"}[item],
        key=f"{prefix}_label_position",
        disabled=not show_values,
    )
    manual_position = show_values and label_position_mode == "manual"
    label_offset_x = label_b.number_input(
        "X 조정값(points)", -100.0, 100.0, 0.0, 1.0,
        key=f"{prefix}_label_offset_x", disabled=not manual_position,
    )
    label_offset_y = label_c.number_input(
        "Y 조정값(points)", -100.0, 100.0, 5.0, 1.0,
        key=f"{prefix}_label_offset_y", disabled=not manual_position,
    )
    label_d, label_e = st.columns(2)
    label_font_size = label_d.slider(
        "라벨 폰트 크기", 4, 40, 8, key=f"{prefix}_label_font_size", disabled=not show_values
    )
    label_color = label_e.color_picker(
        "라벨 색상", "#172033", key=f"{prefix}_label_color", disabled=not show_values
    )
    pie_label_mode = "ratio"
    if chart_type == "pie":
        pie_label_mode = st.selectbox(
            "Pie/Donut 표시 방식",
            ["ratio", "label", "label_ratio"],
            format_func=lambda item: {
                "ratio": "비율만",
                "label": "라벨만",
                "label_ratio": "라벨 + 비율",
            }[item],
            key=f"{prefix}_pie_label_mode",
            disabled=not show_values,
        )

    orientation = "vertical"
    bar_mode = "basic"
    histogram_bins = 10
    histogram_density = False
    line_style, line_width, marker, marker_size, area_fill, line_curvature = "-", 2.0, "o", 5.0, False, 0.0
    pie_start_angle, donut, pie_shadow, pie_min_ratio = 90, False, False, 0.0
    pie_sort_by, pie_sort_direction = "none", "ascending"
    donut_hole_size, donut_ring_width = 0.5, 0.4
    donut_center_color, donut_center_border = "#FFFFFF", False
    donut_center_border_color, donut_center_border_width = "#334155", 1.0
    scatter_size, trendline = 80.0, False
    heatmap_cmap, heatmap_annotate, heatmap_colorbar, heatmap_linewidth = "Blues", True, True, 0.5
    st.markdown("##### 차트별 설정")
    if chart_type == "bar":
        a, b = st.columns(2)
        orientation = a.selectbox("막대 방향", ["vertical", "horizontal"], key=f"{prefix}_orientation")
        bar_mode = b.selectbox("막대 유형", ["basic", "grouped", "stacked", "stacked_100"], key=f"{prefix}_bar_mode")
    elif chart_type == "line":
        a, b, c = st.columns(3)
        line_style = a.selectbox("선 스타일", ["-", "--", "-.", ":"], key=f"{prefix}_line_style")
        line_width = b.slider("선 두께", 0.2, 10.0, 2.0, 0.2, key=f"{prefix}_line_width")
        marker = c.selectbox("마커", ["o", "s", "^", "D", "x", "+"], key=f"{prefix}_marker")
        a2, b2, c2 = st.columns(3)
        marker_size = a2.slider("마커 크기", 1.0, 30.0, 5.0, 1.0, key=f"{prefix}_marker_size")
        area_fill = b2.checkbox("영역 채우기", False, key=f"{prefix}_area")
        line_curvature = c2.slider("선의 곡선률", 0.0, 1.0, 0.0, 0.05, key=f"{prefix}_curvature")
    elif chart_type == "pie":
        a, b, c = st.columns(3)
        pie_start_angle = a.slider("시작 각도", 0, 360, 90, key=f"{prefix}_pie_angle")
        donut = b.checkbox("도넛 사용", False, key=f"{prefix}_donut")
        pie_shadow = c.checkbox("그림자", False, key=f"{prefix}_shadow")
        pie_min_ratio = st.slider("최소 비율 미만을 기타로 통합(%)", 0.0, 30.0, 0.0, 0.5, key=f"{prefix}_pie_min")
        donut_a, donut_b = st.columns(2)
        donut_hole_size = donut_a.slider(
            "도넛 구멍 크기", 0.05, 0.9, 0.5, 0.05,
            key=f"{prefix}_donut_hole", disabled=not donut,
        )
        ring_key = f"{prefix}_donut_ring"
        maximum_ring = round(max(0.05, 1.0 - donut_hole_size), 2)
        if ring_key in st.session_state and st.session_state[ring_key] > maximum_ring:
            st.session_state[ring_key] = maximum_ring
        donut_ring_width = donut_b.slider(
            "도넛 링 두께", 0.05, maximum_ring, min(0.4, maximum_ring), 0.05,
            key=ring_key, disabled=not donut,
        )
        center_a, center_b = st.columns(2)
        donut_center_color = center_a.color_picker(
            "중앙 배경색", "#FFFFFF", key=f"{prefix}_donut_center_color", disabled=not donut
        )
        donut_center_border = center_b.checkbox(
            "중앙 테두리", False, key=f"{prefix}_donut_center_border", disabled=not donut
        )
        border_a, border_b = st.columns(2)
        donut_center_border_color = border_a.color_picker(
            "중앙 테두리 색상", "#334155", key=f"{prefix}_donut_border_color",
            disabled=not donut or not donut_center_border,
        )
        donut_center_border_width = border_b.slider(
            "중앙 테두리 두께", 0.0, 10.0, 1.0, 0.5, key=f"{prefix}_donut_border_width",
            disabled=not donut or not donut_center_border,
        )
        sort_col, direction_col = st.columns(2)
        pie_sort_by = sort_col.selectbox(
            "조각·범례 정렬 기준",
            ["none", "label", "value"],
            format_func=lambda item: {"none": "원본 순서", "label": "컬럼 값", "value": "집계 값"}[item],
            key=f"{prefix}_pie_sort_by",
        )
        pie_sort_direction = direction_col.selectbox(
            "정렬 방향",
            ["ascending", "descending"],
            format_func=lambda item: {"ascending": "오름차순", "descending": "내림차순"}[item],
            key=f"{prefix}_pie_sort_direction",
            disabled=pie_sort_by == "none",
        )
    elif chart_type == "histogram":
        a, b = st.columns(2)
        histogram_bins = a.slider("구간 수", 2, 100, 10, key=f"{prefix}_bins")
        histogram_density = b.checkbox("밀도 표시", False, key=f"{prefix}_density")
    elif chart_type == "scatter_bubble":
        a, b = st.columns(2)
        scatter_size = a.slider("점 최대 크기", 5.0, 1000.0, 80.0, 5.0, key=f"{prefix}_scatter_size")
        trendline = b.checkbox("추세선", False, key=f"{prefix}_trend")
    elif chart_type == "heatmap":
        a, b, c = st.columns(3)
        heatmap_cmap = a.selectbox("컬러맵", ["Blues", "viridis", "magma", "coolwarm", "RdBu_r"], key=f"{prefix}_heatmap_cmap")
        heatmap_annotate = b.checkbox("셀 값 표시", True, key=f"{prefix}_heatmap_annot")
        heatmap_colorbar = c.checkbox("컬러바 표시", True, key=f"{prefix}_heatmap_cbar")
        heatmap_linewidth = st.slider("셀 경계선", 0.0, 5.0, 0.5, 0.1, key=f"{prefix}_heatmap_line")
    return AdvancedSettings(
        x_sort=x_sort,
        y_sort=y_sort,
        top_n=top_n if top_n_enabled else None,
        include_missing=include_missing,
        orientation=orientation,
        bar_mode=bar_mode,
        palette=palette,
        base_color=base_color,
        alpha=alpha,
        edge_color=edge_color,
        edge_width=edge_width,
        grid=grid,
        grid_axis=grid_axis,
        legend=legend,
        legend_location=legend_location,
        tick_rotation=tick_rotation,
        title_size=title_size,
        axis_size=axis_size,
        number_format=number_format,
        unit=unit,
        label_position_mode=label_position_mode if show_values else "auto",
        label_offset_x=label_offset_x,
        label_offset_y=label_offset_y,
        label_font_size=label_font_size,
        label_color=label_color,
        pie_label_mode=pie_label_mode,
        histogram_bins=histogram_bins,
        histogram_density=histogram_density,
        line_style=line_style,
        line_width=line_width,
        marker=marker,
        marker_size=marker_size,
        area_fill=area_fill,
        line_curvature=line_curvature,
        pie_start_angle=pie_start_angle,
        donut=donut,
        pie_shadow=pie_shadow,
        pie_min_ratio=pie_min_ratio,
        pie_sort_by=pie_sort_by,
        pie_sort_direction=pie_sort_direction,
        donut_hole_size=donut_hole_size,
        donut_ring_width=donut_ring_width,
        donut_center_color=donut_center_color,
        donut_center_border=donut_center_border if donut else False,
        donut_center_border_color=donut_center_border_color,
        donut_center_border_width=donut_center_border_width,
        scatter_size=scatter_size,
        trendline=trendline,
        heatmap_cmap=heatmap_cmap,
        heatmap_annotate=heatmap_annotate,
        heatmap_colorbar=heatmap_colorbar,
        heatmap_linewidth=heatmap_linewidth,
    )


def _axis_kind(frame: pd.DataFrame, basic: dict, axis: str) -> tuple[str, str | None]:
    chart_type = basic["chart_type"]
    if chart_type == "pie":
        return "unavailable", None
    if axis == "x":
        column = basic["x"]
        if chart_type in {"bar", "heatmap"}:
            return "category", column
        if chart_type == "histogram":
            return "numeric", column
    else:
        column = basic.get("y")
        if chart_type == "heatmap":
            return "category", column
        if chart_type != "scatter_bubble":
            return "numeric", None
    if column and pd.api.types.is_numeric_dtype(frame[column]):
        return "numeric", column
    return "category", column


def _axis_controls(index: int, axis: str, kind: str, column: str | None, frame: pd.DataFrame) -> dict:
    prefix = f"viz_{index}_deep_{axis}axis"
    axis_name = axis.upper()
    result = {
        f"{axis}_axis_mode": "all",
        f"{axis}_min": None,
        f"{axis}_max": None,
        f"{axis}_tick_interval": None,
        f"{axis}_category_start": None,
        f"{axis}_category_end": None,
        f"{axis}_selected_categories": [],
    }
    with st.container(border=True):
        st.markdown(f"##### {axis_name}축 제어")
        if kind == "unavailable":
            st.selectbox("축 범위", ["사용할 수 없음"], key=f"{prefix}_disabled", disabled=True)
            st.caption("이 차트 유형에서는 해당 축을 직접 제어할 수 없습니다.")
            return result
        if kind == "numeric":
            mode = st.selectbox(
                "숫자축 설정",
                ["전체", "최소값~최대값"],
                key=f"{prefix}_numeric_mode",
            )
            result[f"{axis}_axis_mode"] = "numeric_range" if mode == "최소값~최대값" else "all"
            if column and pd.api.types.is_numeric_dtype(frame[column]):
                numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
                default_min = float(numeric.min()) if not numeric.empty else 0.0
                default_max = float(numeric.max()) if not numeric.empty else 1.0
            else:
                default_min, default_max = 0.0, float(max(len(frame), 1))
            if default_min >= default_max:
                default_max = default_min + 1.0
            minimum, maximum = st.columns(2)
            result[f"{axis}_min"] = minimum.number_input(
                "최소값",
                value=default_min,
                key=f"{prefix}_{column}_min",
                disabled=mode == "전체",
            ) if mode != "전체" else None
            result[f"{axis}_max"] = maximum.number_input(
                "최대값",
                value=default_max,
                key=f"{prefix}_{column}_max",
                disabled=mode == "전체",
            ) if mode != "전체" else None
            interval_enabled = st.checkbox("눈금 간격 입력", key=f"{prefix}_interval_enabled")
            result[f"{axis}_tick_interval"] = st.number_input(
                "눈금 간격",
                min_value=0.000001,
                value=max((default_max - default_min) / 5, 0.000001),
                key=f"{prefix}_{column}_interval",
                disabled=not interval_enabled,
            ) if interval_enabled else None
            return result
        values = list(dict.fromkeys(frame[column].dropna().astype(str).tolist())) if column else []
        if not values:
            st.selectbox("범주축 설정", ["사용할 수 없음"], key=f"{prefix}_empty", disabled=True)
            return result
        mode = st.selectbox(
            "범주축 설정",
            ["전체", "시작값~종료값", "항목 직접 선택"],
            key=f"{prefix}_{column}_category_mode",
        )
        if mode == "시작값~종료값":
            result[f"{axis}_axis_mode"] = "category_range"
            start_col, end_col = st.columns(2)
            result[f"{axis}_category_start"] = start_col.selectbox(
                "시작값", values, key=f"{prefix}_{column}_start"
            )
            result[f"{axis}_category_end"] = end_col.selectbox(
                "종료값", values, index=len(values) - 1, key=f"{prefix}_{column}_end"
            )
        elif mode == "항목 직접 선택":
            result[f"{axis}_axis_mode"] = "category_select"
            result[f"{axis}_selected_categories"] = st.multiselect(
                "표시할 항목", values, default=values, key=f"{prefix}_{column}_selected"
            )
        return result


def _reference_kind(frame: pd.DataFrame, column: str | None, axis_kind: str) -> str:
    if axis_kind == "unavailable":
        return "unavailable"
    if column:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return "date"
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_null = series.dropna()
            if not non_null.empty:
                parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
                if float(parsed.notna().mean()) >= 0.8:
                    return "date"
    return "numeric" if axis_kind == "numeric" else "category"


def _reference_controls(
    index: int,
    frame: pd.DataFrame,
    x_kind: str,
    x_column: str | None,
    y_kind: str,
    y_column: str | None,
) -> dict:
    prefix = f"viz_{index}_deep_reference"
    x_reference_kind = _reference_kind(frame, x_column, x_kind)
    y_reference_kind = _reference_kind(frame, y_column, y_kind)
    available = [axis for axis, kind in (("x", x_reference_kind), ("y", y_reference_kind)) if kind != "unavailable"]
    st.markdown("##### 기준선")
    enabled = st.checkbox("기준선 사용", False, key=f"{prefix}_enabled", disabled=not available)
    default_target = ["y"] if "y" in available else available[:1]
    targets = st.multiselect(
        "적용 대상",
        available,
        default=default_target,
        format_func=lambda axis: f"{axis.upper()}축",
        key=f"{prefix}_targets",
        disabled=not enabled,
    )
    style_a, style_b, style_c = st.columns(3)
    line_style = style_a.selectbox(
        "선 스타일", ["-", "--", "-.", ":"], index=1, key=f"{prefix}_style", disabled=not enabled
    )
    line_width = style_b.slider(
        "선 두께", 0.1, 10.0, 1.2, 0.1, key=f"{prefix}_width", disabled=not enabled
    )
    line_alpha = style_c.slider(
        "선 투명도", 0.0, 1.0, 0.8, 0.05, key=f"{prefix}_alpha", disabled=not enabled
    )
    label_a, label_b, label_c = st.columns(3)
    label = label_a.text_input("기준선 라벨", key=f"{prefix}_label", disabled=not enabled)
    label_size = label_b.slider(
        "라벨 크기", 4, 40, 9, key=f"{prefix}_label_size", disabled=not enabled
    )
    label_alpha = label_c.slider(
        "라벨 투명도", 0.0, 1.0, 0.9, 0.05, key=f"{prefix}_label_alpha", disabled=not enabled
    )
    values: dict[str, object | None] = {"x": None, "y": None}
    kinds = {"x": x_reference_kind, "y": y_reference_kind}
    columns = {"x": x_column, "y": y_column}
    for axis in available:
        kind = kinds[axis]
        column = columns[axis]
        disabled = not enabled or axis not in targets
        if kind == "numeric":
            if column and pd.api.types.is_numeric_dtype(frame[column]):
                numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
                default = float(numeric.median()) if not numeric.empty else 0.0
            else:
                default = 0.0
            values[axis] = st.number_input(
                f"{axis.upper()}축 기준 수치", value=default,
                key=f"{prefix}_{axis}_{column}_numeric", disabled=disabled,
            )
        elif kind == "date":
            parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed").dropna()
            default_date = parsed.min().date() if not parsed.empty else pd.Timestamp.today().date()
            values[axis] = st.date_input(
                f"{axis.upper()}축 기준 날짜", value=default_date,
                key=f"{prefix}_{axis}_{column}_date", disabled=disabled,
            )
        else:
            categories = list(dict.fromkeys(frame[column].dropna().astype(str).tolist())) if column else []
            values[axis] = st.selectbox(
                f"{axis.upper()}축 기준 범주", categories or ["사용할 수 없음"],
                key=f"{prefix}_{axis}_{column}_category", disabled=disabled or not categories,
            ) if categories else None
    if not enabled:
        targets = []
        values = {"x": None, "y": None}
    else:
        for axis in ("x", "y"):
            if axis not in targets:
                values[axis] = None
    return {
        "reference_enabled": enabled,
        "reference_targets": targets,
        "reference_line_style": line_style,
        "reference_line_width": line_width,
        "reference_line_alpha": line_alpha,
        "reference_label": label,
        "reference_label_size": label_size,
        "reference_label_alpha": label_alpha,
        "x_reference_kind": x_reference_kind if "x" in targets else None,
        "y_reference_kind": y_reference_kind if "y" in targets else None,
        "x_reference_value": values["x"],
        "y_reference_value": values["y"],
    }


def _deep_controls(index: int, basic: dict, frame: pd.DataFrame) -> DeepSettings:
    prefix = f"viz_{index}_deep"
    chart_type = basic["chart_type"]
    x_kind, x_column = _axis_kind(frame, basic, "x")
    y_kind, y_column = _axis_kind(frame, basic, "y")
    x_col, y_col = st.columns(2)
    with x_col:
        x_axis = _axis_controls(index, "x", x_kind, x_column, frame)
    with y_col:
        y_axis = _axis_controls(index, "y", y_kind, y_column, frame)
    c5, c6, c7, c8 = st.columns(4)
    x_log = c5.checkbox("X 로그 스케일", False, key=f"{prefix}_xlog", disabled=x_kind != "numeric")
    y_log = c6.checkbox("Y 로그 스케일", False, key=f"{prefix}_ylog", disabled=y_kind != "numeric")
    invert_x = c7.checkbox("X축 반전", False, key=f"{prefix}_invertx", disabled=x_kind == "unavailable")
    invert_y = c8.checkbox("Y축 반전", False, key=f"{prefix}_inverty", disabled=y_kind == "unavailable")
    if x_kind != "numeric":
        x_log = False
    if y_kind != "numeric":
        y_log = False
    if x_kind == "unavailable":
        invert_x = False
    if y_kind == "unavailable":
        invert_y = False
    reference = _reference_controls(index, frame, x_kind, x_column, y_kind, y_column)
    c10, c11 = st.columns(2)
    normalize = c10.checkbox("정규화", False, key=f"{prefix}_normalize")
    cumulative = c11.checkbox("누적합", False, key=f"{prefix}_cumulative", disabled=chart_type != "line")
    moving_average = None
    show_mean = show_median = False
    jitter = 0.0
    show_correlation = highlight_outliers = False
    heatmap_center = None
    if chart_type == "line":
        enabled = st.checkbox("이동평균", False, key=f"{prefix}_ma_enabled")
        moving_average = st.number_input("이동평균 기간", 2, 100, 3, key=f"{prefix}_ma", disabled=not enabled) if enabled else None
    elif chart_type == "histogram":
        a, b = st.columns(2)
        show_mean = a.checkbox("평균 기준선", False, key=f"{prefix}_mean")
        show_median = b.checkbox("중앙값 기준선", False, key=f"{prefix}_median")
    elif chart_type == "scatter_bubble":
        a, b, c = st.columns(3)
        jitter = a.slider("지터", 0.0, 2.0, 0.0, 0.05, key=f"{prefix}_jitter")
        show_correlation = b.checkbox("상관계수 표시", False, key=f"{prefix}_corr")
        highlight_outliers = c.checkbox("이상치 강조", False, key=f"{prefix}_outlier")
    elif chart_type == "heatmap":
        heatmap_center = _number_or_none("색상 중심점", f"{prefix}_center")
    return DeepSettings(
        **x_axis,
        **y_axis,
        x_log=x_log,
        y_log=y_log,
        invert_x=invert_x,
        invert_y=invert_y,
        moving_average=moving_average,
        cumulative=cumulative,
        normalize=normalize,
        **reference,
        show_mean=show_mean,
        show_median=show_median,
        jitter=jitter,
        show_correlation=show_correlation,
        highlight_outliers=highlight_outliers,
        heatmap_center=heatmap_center,
    )


def _structured_specs(frame: pd.DataFrame, chart_count: int) -> list[ChartSpec]:
    columns = [str(c) for c in frame.columns]
    numeric_columns = [str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    specs = []
    for index in range(1, chart_count + 1):
        st.markdown(f"## Subplot {index}")
        with st.expander("Basic · 기본 설정", expanded=True):
            basic = _basic_controls(index, columns, numeric_columns)
        with st.expander("Advanced1 · 고급 설정", expanded=False):
            advanced = _advanced_controls(index, basic)
        with st.expander("Advanced2 · 심화 설정", expanded=False):
            deep = _deep_controls(index, basic, frame)
        specs.append(ChartSpec.model_validate({**basic, "advanced": advanced, "deep": deep}))
    return specs


def _render_result(result) -> None:
    st.markdown("### 시각화 결과")
    st.pyplot(result.figure, clear_figure=False, width="stretch")
    for index, artifact in enumerate(result.artifacts, 1):
        with st.expander(f"Chart {index} · 통계자료와 요약 인사이트", expanded=index == 1):
            st.markdown(artifact.insight.replace("\n", "  \n"))
            st.dataframe(artifact.statistics, width="stretch", hide_index=True)
            mapping = artifact.statistics.attrs.get("category_mapping")
            if mapping is not None:
                st.caption("범주형 변수의 수치 인덱스 매핑")
                st.dataframe(mapping, width="stretch", hide_index=True)
            mappings = artifact.statistics.attrs.get("category_mappings", {})
            for variable, variable_mapping in mappings.items():
                st.caption(f"{variable} 범주형 변수의 수치 인덱스 매핑")
                st.dataframe(variable_mapping, width="stretch", hide_index=True)
    st.markdown("### 결과 다운로드")
    cols = st.columns(5)
    mime_types = {"png": "image/png", "jpg": "image/jpeg", "svg": "image/svg+xml", "pdf": "application/pdf"}
    for column, fmt in zip(cols[:4], ["png", "jpg", "svg", "pdf"]):
        column.download_button(
            fmt.upper(),
            data=figure_to_bytes(result, fmt),
            file_name=f"{result.figure_spec.filename}.{fmt}",
            mime=mime_types[fmt],
            width="stretch",
            key=f"viz_download_{fmt}",
        )
    source_filename = st.session_state.source_filename or "data"
    cols[4].download_button(
        "Source JSON",
        data=source_payload_bytes(result, source_filename),
        file_name=f"{result.figure_spec.filename}_source.json",
        mime="application/json",
        width="stretch",
        key="viz_download_source",
    )


def render_visualization() -> None:
    st.header("데이터 시각화")
    frame = st.session_state.df_clean
    st.caption("시각화와 통계자료는 현재 전처리 데이터 `df_clean`을 기준으로 생성됩니다.")
    with st.expander("변수별 데이터 타입", expanded=False):
        st.dataframe(variable_type_table(frame), width="stretch", hide_index=True)
    method = st.radio(
        "시각화 요청방법",
        ["구조화된 메뉴", "텍스트 요청"],
        horizontal=True,
        key="visualization_request_method",
    )
    grid_size = st.selectbox(
        "subplot 구성",
        [1, 2, 3],
        format_func=lambda size: f"{size} × {size}",
        key="visualization_grid_size",
    )
    chart_count = int(grid_size) ** 2
    figure_spec = _figure_controls(int(grid_size))
    try:
        if method == "텍스트 요청":
            request = st.text_area(
                "요청 내용",
                placeholder="예: 국가별 참가업체수 합계 막대그래프; 조사일자_년월별 개수 추세선",
                key="visualization_text_request",
            )
            specs = parse_text_request(request, frame, chart_count) if request.strip() else []
            if specs:
                st.caption("텍스트 요청에서 해석된 Pydantic 차트 설정")
                st.json([spec.model_dump(mode="json") for spec in specs], expanded=False)
        else:
            specs = _structured_specs(frame, chart_count)
        if st.button("시각화 실행", type="primary", key="run_visualization", width="stretch"):
            if not specs:
                raise ValueError("시각화 설정을 입력하세요.")
            with st.spinner("통계자료를 구성하고 차트를 생성하고 있습니다..."):
                result = build_visualization(frame, specs, figure_spec)
                payload = source_payload(result, st.session_state.source_filename or "data")
                st.session_state.visualization_result = result
                saved_sources = list(st.session_state.get("visualization_sources", []))
                saved_sources.append(payload)
                st.session_state.visualization_sources = saved_sources[-20:]
                st.session_state.visualization_notice = "시각화와 통계 source 저장을 완료했습니다."
                st.rerun()
    except (ValidationError, VisualizationDataError, ValueError, TypeError) as exc:
        st.error(str(exc))
    notice = st.session_state.pop("visualization_notice", None)
    if notice:
        st.success(notice)
    result = st.session_state.get("visualization_result")
    if result is not None:
        _render_result(result)
