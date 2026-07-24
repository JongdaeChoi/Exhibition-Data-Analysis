from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from core.i18n import current_language, localized_table, translate
from visualization.models import AdvancedSettings, ChartSpec, DeepSettings, FigureSpec
from visualization.service import (
    automatic_chart_title,
    build_visualization,
    figure_to_bytes,
    source_payload,
    source_payload_bytes,
    summarize_artifact,
)
from visualization.statistics import VisualizationDataError, semantic_type, variable_type_table


CHART_LABELS = {
    "bar": "Bar Chart",
    "line": "Line Chart",
    "multi_variable": "Multi-variable Comparison",
    "pie": "Pie Chart",
    "histogram": "Histogram",
    "scatter_plot": "Scatter Plot",
    "grouped_bar": "Grouped Bar Chart",
    "stacked_bar": "Stacked Bar Chart",
    "scatter_bubble": "Scatter Bubble",
    "heatmap": "Heatmap",
    "correlation_heatmap": "Correlation Heatmap",
}
STRUCTURED_CHART_TYPES = [
    "bar",
    "line",
    "multi_variable",
    "pie",
    "histogram",
    "scatter_plot",
    "scatter_bubble",
    "heatmap",
    "correlation_heatmap",
]
AGGREGATION_LABELS = {
    "count": "행 개수", "valid_count": "유효값 개수(결측 제외)",
    "sum": "합계", "mean": "평균", "ratio": "비율",
}
NONE_OPTION = "(없음)"
BASIC_COLORS = {
    "흰색": "#FFFFFF", "밝은 회색": "#F8FAFC", "회색": "#CBD5E1",
    "검정": "#111827", "파랑": "#2563EB", "주황": "#F59E0B",
}


def _optional_column(label: str, columns: list[str], key: str, disabled: bool = False) -> str | None:
    value = st.selectbox(label, [NONE_OPTION] + columns, key=key, disabled=disabled)
    return None if value == NONE_OPTION else value


def _number_or_none(label: str, key: str, disabled: bool = False) -> float | None:
    enabled = st.checkbox(f"{label} 사용", key=f"{key}_enabled", disabled=disabled)
    if not enabled:
        return None
    return st.number_input(label, value=0.0, key=key, disabled=disabled)


def _color_control(label: str, key: str, default: str, disabled: bool = False) -> str:
    mode_column, value_column = st.columns([1, 2], gap="small")
    mode = mode_column.selectbox(
        f"{label} 방식", ["basic", "hex"],
        format_func=lambda value: "기본 색상" if value == "basic" else "HEX 직접 입력",
        key=f"{key}_mode", disabled=disabled,
    )
    if mode == "basic":
        names = list(BASIC_COLORS)
        default_name = next((name for name, value in BASIC_COLORS.items() if value == default), names[0])
        selected = value_column.selectbox(
            label, names, index=names.index(default_name), key=f"{key}_basic", disabled=disabled
        )
        return BASIC_COLORS[selected]
    return value_column.color_picker(label, default, key=f"{key}_hex", disabled=disabled)


def _figure_controls_body() -> tuple[FigureSpec, int, int]:
    with st.container(border=True):
        st.markdown("##### 차트 개수(N * N)")
        layout_a, layout_b = st.columns(2, gap="small")
        rows = int(layout_a.number_input(
            "행 개수(n1)", min_value=1, max_value=3, value=1, step=1,
            key="visualization_subplot_rows",
        ))
        columns = int(layout_b.number_input(
            "열 개수(n2)", min_value=1, max_value=3, value=1, step=1,
            key="visualization_subplot_columns",
        ))
        chart_count = rows * columns
        st.caption(
            f"{rows} × {columns} layout · {chart_count} chart setting(s)"
            if current_language() == "English"
            else f"{rows} × {columns} 구성, 차트 {chart_count}개"
        )

    with st.container(border=True):
        st.markdown("##### 캔버스(Figure)")
        # 1행 : 가로 크기, 세로 크기 ,DPI
        c1, c2, c3 = st.columns(3, gap="small")
        width = c1.number_input("가로 크기", 4.0, 30.0, float(max(10, columns * 6)), 0.5, key="viz_fig_width")
        height = c2.number_input("세로 크기", 3.0, 30.0, float(max(7, rows * 5)), 0.5, key="viz_fig_height")
        dpi = c3.number_input("해상도(DPI)", 72, 600, 120, 10, key="viz_dpi")

        figure_background = _color_control("배경색", "viz_figure_bg", "#FFFFFF")
        figure_alpha = st.slider("배경 투명도", 0.0, 1.0, 1.0, 0.05, key="viz_figure_alpha")

        border_a, border_b, border_c, border_d = st.columns(4, gap="small")
        figure_border_visible = border_a.checkbox("테두리 표시", False, key="viz_border_visible")
        figure_border_width = border_b.slider(
            "테두리 두께", 0.0, 20.0, 1.0, 0.5, key="viz_border_width", disabled=not figure_border_visible
        )
        figure_border_alpha = border_c.slider(
            "테두리 투명도", 0.0, 1.0, 1.0, 0.05, key="viz_border_alpha", disabled=not figure_border_visible
        )
        figure_border_color = border_d.color_picker(
            "테두리 색상", "#CBD5E1", key="viz_border_color", disabled=not figure_border_visible
        )

    with st.container(border=True):
        st.markdown("##### 축 공유")
        share_a, share_b, share_c, share_d = st.columns(4, gap="small")
        share_x = share_a.checkbox("X축 공유", False, key="viz_share_x")
        share_y = share_b.checkbox("Y축 공유", False, key="viz_share_y")

    with st.container(border=True):
        st.markdown("##### 여백(Padding)")
        layout_mode = st.radio(
            "배치 방식", ["tight", "constrained", "basic", "custom"], horizontal=True,
            label_visibility = "collapsed",
            format_func=lambda value: {"tight": "Tight", "constrained": "Constrained", "basic": "Basic", "custom": "사용자 설정"}[value],
            key="viz_layout_mode",
        )
        margin_left, margin_right, margin_bottom, margin_top = 0.08, 0.98, 0.08, 0.92
        horizontal_space, vertical_space = 0.28, 0.35
        if layout_mode == "custom":
            margin_a, margin_b, margin_c, margin_d = st.columns(4, gap="small")
            margin_left = margin_a.slider("내부 왼쪽", 0.0, 0.9, 0.08, 0.01, key="viz_margin_left")
            margin_right = margin_b.slider("내부 오른쪽", 0.1, 1.0, 0.98, 0.01, key="viz_margin_right")
            margin_bottom = margin_c.slider("내부 아래", 0.0, 0.9, 0.08, 0.01, key="viz_margin_bottom")
            margin_top = margin_d.slider("내부 위", 0.1, 1.0, 0.92, 0.01, key="viz_margin_top")
            space_a, space_b = st.columns(2, gap="small")
            horizontal_space = space_a.slider("차트간 가로 간격", 0.0, 1.5, 0.28, 0.02, key="viz_wspace")
            vertical_space = space_b.slider("차트간 세로 간격", 0.0, 1.5, 0.35, 0.02, key="viz_hspace")

    with st.container(border=True):
        st.markdown("##### 차트(Axes)")
        axes_background = _color_control("배경색", "viz_axes_bg", "#FFFFFF")
        axes_background_alpha = st.slider("배경 투명도", 0.0, 1.0, 1.0, 0.05, key="viz_axes_bg_alpha")
        scope_a, scope_b = st.columns(2, gap="small")
        axes_style_scope = scope_a.radio(
            "영역별 설정", ["all", "selected"], horizontal=True,
            format_func=lambda value: "전체 차트" if value == "all" else "선택한 차트",
            key="viz_axes_scope",
        )
        axes_target_index = scope_b.number_input(
            "대상 Subplot", 1, rows * columns, 1, key="viz_axes_target",
            disabled=axes_style_scope == "all",
        )
        axes_border_visible = st.checkbox("테두리 표시", True, key="viz_axes_border_visible")
        axes_border_positions = st.multiselect(
            "테두리 위치", ["top", "bottom", "left", "right"],
            default=["top", "bottom", "left", "right"], key="viz_axes_border_positions",
            disabled=not axes_border_visible,
        )
        axes_a, axes_b, axes_c, axes_d = st.columns(4, gap="small")
        axes_border_width = axes_a.slider("두께", 0.0, 10.0, 0.8, 0.1, key="viz_axes_border_width", disabled=not axes_border_visible)
        axes_border_color = axes_b.color_picker("색상", "#334155", key="viz_axes_border_color", disabled=not axes_border_visible)
        axes_border_style = axes_c.selectbox("선 스타일", ["-", "--", "-.", ":"], key="viz_axes_border_style", disabled=not axes_border_visible)
        axes_border_alpha = axes_d.slider("투명도", 0.0, 1.0, 1.0, 0.05, key="viz_axes_border_alpha", disabled=not axes_border_visible)

    with st.container(border=True):
        st.markdown("##### 저장 설정")
        save_a, save_b, save_c = st.columns(3, gap="small")
        output_formats = save_a.multiselect(
            "저장 형식", ["png", "jpg", "pdf", "svg"], default=["png"],
            format_func=str.upper, key="viz_output_formats",
        )
        filename = save_b.text_input("출력 파일명", "visualization", key="viz_filename")
        include_metadata = save_c.checkbox("메타 데이터 포함", True, key="viz_include_metadata")
        transparent = save_a.checkbox("저장 배경 투명", False, key="viz_transparent")
        font_family = save_b.selectbox("폰트", ["NanumGothic", "DejaVu Sans", "sans-serif"], key="viz_font_family")
        font_color = save_c.color_picker("폰트 색상", "#172033", key="viz_font_color")
    return FigureSpec(
        rows=rows,
        columns=columns,
        width=width,
        height=height,
        dpi=dpi,
        figure_background=figure_background,
        figure_alpha=figure_alpha,
        figure_border_width=figure_border_width if figure_border_visible else 0.0,
        figure_border_color=figure_border_color,
        figure_border_alpha=figure_border_alpha,
        axes_background=axes_background,
        axes_background_alpha=axes_background_alpha,
        axes_border_visible=axes_border_visible,
        axes_border_positions=axes_border_positions,
        axes_border_width=axes_border_width,
        axes_border_color=axes_border_color,
        axes_border_style=axes_border_style,
        axes_border_alpha=axes_border_alpha,
        axes_style_scope=axes_style_scope,
        axes_target_index=axes_target_index,
        share_x=share_x,
        share_y=share_y,
        horizontal_space=horizontal_space,
        vertical_space=vertical_space,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
        margin_top=margin_top,
        tight_layout=layout_mode == "tight",
        constrained_layout=layout_mode == "constrained",
        layout_mode=layout_mode,
        font_family=font_family,
        font_color=font_color,
        transparent=transparent,
        filename=filename or "visualization",
        output_formats=output_formats,
        include_metadata=include_metadata,
    ), rows, columns


def _figure_controls() -> tuple[FigureSpec, int, int]:
    with st.container(key="visualization_figure_settings"):
        with st.expander("레이아웃 설정", expanded=False):
            return _figure_controls_body()


def _basic_controls(index: int, columns: list[str], numeric_columns: list[str]) -> dict:
    prefix = f"viz_{index}"
    chart_type_key = f"{prefix}_type"
    legacy_chart_type = st.session_state.get(chart_type_key)
    if legacy_chart_type in {"grouped_bar", "stacked_bar"}:
        st.session_state[chart_type_key] = "bar"
        legacy_group = st.session_state.get(f"{prefix}_y")
        if legacy_group in columns:
            st.session_state[f"{prefix}_group"] = legacy_group
        st.session_state[f"{prefix}_bar_mode"] = (
            "stacked" if legacy_chart_type == "stacked_bar" else "grouped"
        )
    chart_type = st.selectbox(
        "차트 유형",
        STRUCTURED_CHART_TYPES,
        format_func=lambda item: CHART_LABELS[item],
        key=chart_type_key,
    )
    type_signature_key = f"{prefix}_active_chart_type"
    previous_chart_type = st.session_state.get(type_signature_key)
    if previous_chart_type and previous_chart_type != chart_type:
        st.info("차트 유형이 변경되어 지원하지 않는 이전 설정은 현재 ChartSpec 기본값으로 초기화됩니다.")
    st.session_state[type_signature_key] = chart_type
    requires_y = chart_type in {"scatter_plot", "grouped_bar", "stacked_bar", "scatter_bubble", "heatmap"}
    has_axes = chart_type != "pie"

    data_group = st.container(border=True)
    data_group.markdown("##### 데이터 설정")
    variables = []
    comparison_chart = "bar"
    selected_group = None
    bar_mode = "basic"
    preselected_value = NONE_OPTION
    bubble_size = 80.0
    chart_color_mode = "hex"
    chart_colormap = "viridis"
    chart_hex = "#2563EB"
    chart_alpha = 0.85
    aggregation_options = ["count", "valid_count", "sum", "mean", "ratio"]
    if chart_type == "histogram":
        aggregation_options = ["count", "sum", "ratio"]
    elif chart_type == "pie":
        aggregation_options = ["count", "valid_count", "ratio"]

    def aggregation_label(item: str) -> str:
        if chart_type == "pie" and item == "valid_count":
            return "행 개수(결측치 제외)"
        if chart_type in {"scatter_bubble", "heatmap"} and item == "valid_count":
            return "행 개수(결측 제외)"
        return AGGREGATION_LABELS[item]

    aggregation = None
    if chart_type == "multi_variable":
        available = numeric_columns if chart_type == "correlation_heatmap" else columns
        data_a, data_b, data_c = data_group.columns([2, 1, 1], gap="small")
        variables = data_a.multiselect(
            "비교 변수(X1~Xn)", available, default=available[: min(2, len(available))],
            key=f"{prefix}_variables",
        )
        x, y = "", None
        aggregation = data_b.selectbox(
            "집계 방식", aggregation_options, format_func=aggregation_label,
            key=f"{prefix}_{chart_type}_aggregation",
        )
        comparison_chart = data_c.radio(
            "표현 방식", ["bar", "line"], horizontal=True,
            format_func=lambda value: "Bar Chart" if value == "bar" else "Line Chart",
            key=f"{prefix}_comparison_chart",
        )
    elif chart_type == "correlation_heatmap":
        variables = data_group.multiselect(
            "비교 변수(X1~Xn)", numeric_columns,
            default=numeric_columns[: min(2, len(numeric_columns))], key=f"{prefix}_variables",
        )
        x, y = "", None
        aggregation = "count"
    elif chart_type in {"bar", "line"}:
        data_a, data_b, data_c, data_d = data_group.columns(4, gap="small")
        x = data_a.selectbox("컬럼 선택", columns, key=f"{prefix}_x")
        y = None
        group = data_b.selectbox("세부그룹 선택", [NONE_OPTION] + columns, key=f"{prefix}_group")
        selected_group = None if group == NONE_OPTION else group
        aggregation = data_c.selectbox(
            "집계 방식", aggregation_options, format_func=aggregation_label,
            key=f"{prefix}_{chart_type}_aggregation",
        )
        if aggregation in {"valid_count", "sum", "mean"}:
            candidates = numeric_columns if aggregation in {"sum", "mean"} else columns
            preselected_value = data_d.selectbox(
                "집계대상", [NONE_OPTION] + candidates, key=f"{prefix}_{chart_type}_value"
            )
        if chart_type == "bar" and selected_group:
            bar_mode = data_group.radio(
                "막대 표시 방식",
                ["grouped", "stacked"],
                horizontal=True,
                format_func=lambda value: "그룹막대" if value == "grouped" else "누적막대",
                key=f"{prefix}_bar_mode",
            )
    elif chart_type == "pie":
        data_a, data_b = data_group.columns(2, gap="small")
        x = data_a.selectbox("컬럼 선택", columns, key=f"{prefix}_x")
        y = None
        aggregation = data_b.selectbox(
            "집계 방식", aggregation_options, format_func=aggregation_label,
            key=f"{prefix}_{chart_type}_aggregation",
        )
    elif chart_type in {"scatter_bubble", "heatmap"}:
        c1, c2, c3, c4 = data_group.columns(4, gap="small")
        x = c1.selectbox("컬럼1", columns, key=f"{prefix}_x")
        y = c2.selectbox(
            "컬럼2", columns, index=min(1, len(columns) - 1), key=f"{prefix}_y"
        )
        preselected_value = c3.selectbox(
            "집계대상", [NONE_OPTION] + columns, key=f"{prefix}_{chart_type}_value"
        )
        aggregation = c4.selectbox(
            "집계 방식",
            aggregation_options,
            format_func=aggregation_label,
            key=f"{prefix}_{chart_type}_aggregation",
        )
    elif chart_type == "scatter_plot":
        c1, c2, c3 = data_group.columns(3, gap="small")
        x = c1.selectbox("컬럼1", columns, key=f"{prefix}_x")
        y = c2.selectbox("컬럼2", columns, index=min(1, len(columns) - 1), key=f"{prefix}_y")
        group = c3.selectbox("색상 컬럼", [NONE_OPTION] + columns, key=f"{prefix}_group")
        selected_group = None if group == NONE_OPTION else group
        aggregation = "count"
    elif chart_type == "histogram":
        c1, c2, c3 = data_group.columns(3, gap="small")
        x = c1.selectbox("컬럼 선택", columns, key=f"{prefix}_x")
        y = None
        aggregation = c2.selectbox(
            "집계 방식", aggregation_options, format_func=aggregation_label,
            key=f"{prefix}_{chart_type}_aggregation",
        )
        if aggregation == "sum":
            preselected_value = c3.selectbox(
                "집계대상", [NONE_OPTION] + numeric_columns, key=f"{prefix}_{chart_type}_value"
            )
    elif requires_y:
        c1, c2, c3, c4 = data_group.columns(4, gap="small")
        x_label = "X1 변수"
        y_label = "X2 변수"
        x = c1.selectbox(x_label, columns, key=f"{prefix}_x")
        y = c2.selectbox(y_label, columns, index=min(1, len(columns) - 1), key=f"{prefix}_y")
        preselected_value = c3.selectbox(
            "집계대상", [NONE_OPTION] + columns, key=f"{prefix}_{chart_type}_value"
        )
        aggregation = c4.selectbox(
            "집계 방식", aggregation_options, format_func=aggregation_label,
            key=f"{prefix}_{chart_type}_aggregation",
        )
    else:
        x = data_group.selectbox("X1 변수" if has_axes else "범주 변수(X1)", columns, key=f"{prefix}_x")
        y = None

    if chart_type in {"scatter_plot", "correlation_heatmap"}:
        aggregation = "count"
    elif aggregation is None:
        aggregation = data_group.selectbox(
            "집계 방식", aggregation_options,
            format_func=aggregation_label,
            key=f"{prefix}_{chart_type}_aggregation",
        )
    if chart_type == "scatter_bubble":
        data_group.caption("Bubble")
        bubble_size = data_group.slider(
            "크기", 5.0, 1000.0, 80.0, 5.0, key=f"{prefix}_bubble_size"
        )
    color_group = st.container(border=True)
    color_group.markdown("##### 차트 색상")
    palette_default = (
        chart_type in {
            "multi_variable", "pie", "grouped_bar", "stacked_bar",
            "heatmap", "correlation_heatmap",
        }
        or selected_group is not None
    )
    color_a, color_b, color_c, color_d = color_group.columns(4, gap="small")
    chart_color_mode = color_a.radio(
        "색상 방식",
        ["colormap", "hex"],
        index=0 if palette_default else 1,
        horizontal=True,
        format_func=lambda value: "HEX" if value == "hex" else "컬러맵",
        key=f"{prefix}_{chart_type}_chart_color_mode",
    )
    chart_colormap = color_b.selectbox(
        "컬러맵",
        ["viridis", "Blues", "magma", "plasma", "coolwarm", "RdBu_r", "Set2", "tab10"],
        key=f"{prefix}_{chart_type}_chart_colormap",
        disabled=chart_color_mode != "colormap",
    )
    chart_hex = color_c.color_picker(
        "HEX",
        "#2563EB",
        key=f"{prefix}_{chart_type}_chart_hex",
        disabled=chart_color_mode != "hex",
    )
    chart_alpha = color_d.slider(
        "투명도",
        0.05,
        1.0,
        0.85,
        0.05,
        key=f"{prefix}_{chart_type}_chart_alpha",
    )
    value_column = preselected_value
    selected_y = y if requires_y else None
    selected_value = (
        x if chart_type == "pie" and aggregation == "valid_count"
        else x
        if chart_type in {"scatter_bubble", "heatmap"}
        and aggregation == "valid_count"
        and value_column == NONE_OPTION
        else None if value_column == NONE_OPTION
        else value_column
    )

    swap_supported = chart_type not in {"pie", "histogram", "correlation_heatmap"}
    display_group = st.container(border=True)
    display_group.markdown("##### 표시 옵션")
    display_a, display_b = display_group.columns(2, gap="small")
    x_y_swap = display_a.checkbox(
        "X-Y 축 전환", False, key=f"{prefix}_swap", disabled=not swap_supported
    )
    show_values = False
    if chart_type not in {"scatter_plot", "scatter_bubble", "heatmap", "correlation_heatmap"}:
        show_values = display_b.checkbox("값 표시", True, key=f"{prefix}_show_values")
    title_visible = True
    title_size = 13
    title_weight = "bold"
    x_label_visible = has_axes
    x_label_size = 10
    x_label_rotation = 0
    x_label_weight = "normal"
    x_tick_visible = has_axes
    x_tick_size = 10
    x_tick_rotation = 0
    x_tick_weight = "normal"
    y_label_visible = has_axes
    y_label_size = 10
    y_label_rotation = 90
    y_label_weight = "normal"
    y_tick_visible = has_axes
    y_tick_size = 10
    y_tick_rotation = 0
    y_tick_weight = "normal"
    title_key = f"{prefix}_title"
    title_signature_key = f"{prefix}_title_variable_signature"
    title_signature = (chart_type, x, selected_y, selected_group, selected_value, tuple(variables), x_y_swap)
    if st.session_state.get(title_signature_key) != title_signature:
        st.session_state[title_key] = automatic_chart_title(
            selected_y if x_y_swap and selected_y else x,
            x if x_y_swap and selected_y else selected_y,
            selected_group, selected_value, *variables,
        )
        st.session_state[title_signature_key] = title_signature
    title = st.session_state[title_key]
    x_label = ""
    y_label = ""
    if has_axes:
        label_signature_key = f"{prefix}_axis_label_signature"
        label_signature = (chart_type, x, selected_y, aggregation, tuple(variables), x_y_swap)
        if st.session_state.get(label_signature_key) != label_signature:
            if chart_type == "correlation_heatmap":
                axis_label = translate("변수")
                st.session_state[f"{prefix}_xlabel"] = axis_label
                st.session_state[f"{prefix}_ylabel"] = axis_label
            else:
                st.session_state[f"{prefix}_xlabel"] = (
                    selected_y or translate("값") if x_y_swap else x
                )
                st.session_state[f"{prefix}_ylabel"] = (
                    x if x_y_swap else (
                        selected_y or translate("비율(%)" if aggregation == "ratio" else "값")
                    )
                )
            st.session_state[label_signature_key] = label_signature
        x_label = st.session_state[f"{prefix}_xlabel"]
        y_label = st.session_state[f"{prefix}_ylabel"]
    title_group = st.container(border=True)
    title_group.markdown("##### 제목")
    title_control_a, title_control_b, title_control_c, title_control_empty, title_control_d = title_group.columns(
        [1, 2, 1, 1, 1], gap="small"
    )
    title_visible = title_control_a.checkbox(
        "표시", True, key=f"{prefix}_title_visible"
    )
    title = title_control_b.text_input(
        "입력값",
        key=title_key,
        disabled=not title_visible,
        help="변수 설정이 변경되면 선택된 변수명으로 자동 갱신됩니다.",
    )
    title_size = title_control_c.slider(
        "크기", 6, 40, 13, key=f"{prefix}_title_size", disabled=not title_visible
    )
    title_weight = title_control_d.selectbox(
        "굵기", ["normal", "bold"], index=1,
        key=f"{prefix}_title_weight", disabled=not title_visible,
    )
    if has_axes:
        x_axis_group = st.container(border=True)
        x_axis_group.markdown("##### X축")
        x_axis_group.caption("라벨")
        x_label_control_a, x_label_control_b, x_label_control_c, x_label_control_d, x_label_control_e = x_axis_group.columns(
            [1, 2, 1, 1, 1], gap="small"
        )
        x_label_visible = x_label_control_a.checkbox(
            "표시", True, key=f"{prefix}_x_label_visible"
        )
        x_label = x_label_control_b.text_input(
            "입력값",
            key=f"{prefix}_xlabel",
            disabled=not x_label_visible,
            help="선택된 변수명을 기본값으로 사용합니다.",
        )
        x_label_size = x_label_control_c.slider(
            "크기", 6, 40, 10, key=f"{prefix}_x_label_size",
            disabled=not x_label_visible,
        )
        x_label_rotation = x_label_control_d.slider(
            "회전", -180, 180, 0, 5, key=f"{prefix}_x_label_rotation",
            disabled=not x_label_visible,
        )
        x_label_weight = x_label_control_e.selectbox(
            "굵기", ["normal", "bold"], key=f"{prefix}_x_label_weight",
            disabled=not x_label_visible,
        )
        x_axis_group.caption("눈금")
        x_tick_a, x_tick_empty, x_tick_b, x_tick_c, x_tick_d = x_axis_group.columns(
            [1, 2, 1, 1, 1], gap="small"
        )
        x_tick_visible = x_tick_a.checkbox("표시", True, key=f"{prefix}_x_tick_visible")
        x_tick_size = x_tick_b.slider(
            "크기", 6, 40, 10, key=f"{prefix}_x_tick_size", disabled=not x_tick_visible
        )
        x_tick_rotation = x_tick_c.slider(
            "회전", -180, 180, 0, 5, key=f"{prefix}_x_tick_rotation",
            disabled=not x_tick_visible,
        )
        x_tick_weight = x_tick_d.selectbox(
            "굵기", ["normal", "bold"], key=f"{prefix}_x_tick_weight",
            disabled=not x_tick_visible,
        )

        y_axis_group = st.container(border=True)
        y_axis_group.markdown("##### Y축")
        y_axis_group.caption("라벨")
        y_label_control_a, y_label_control_b, y_label_control_c, y_label_control_d, y_label_control_e = y_axis_group.columns(
            [1, 2, 1, 1, 1], gap="small"
        )
        y_label_visible = y_label_control_a.checkbox(
            "표시", True, key=f"{prefix}_y_label_visible"
        )
        y_label = y_label_control_b.text_input(
            "입력값",
            key=f"{prefix}_ylabel",
            disabled=not y_label_visible,
            help="선택된 변수명을 기본값으로 사용합니다.",
        )
        y_label_size = y_label_control_c.slider(
            "크기", 6, 40, 10, key=f"{prefix}_y_label_size",
            disabled=not y_label_visible,
        )
        y_label_rotation = y_label_control_d.slider(
            "회전", -180, 180, 90, 5, key=f"{prefix}_y_label_rotation",
            disabled=not y_label_visible,
        )
        y_label_weight = y_label_control_e.selectbox(
            "굵기", ["normal", "bold"], key=f"{prefix}_y_label_weight",
            disabled=not y_label_visible,
        )
        y_axis_group.caption("눈금")
        y_tick_a, y_tick_empty, y_tick_b, y_tick_c, y_tick_d = y_axis_group.columns([1, 2, 1, 1, 1], gap="small")
        y_tick_visible = y_tick_a.checkbox("표시", True, key=f"{prefix}_y_tick_visible")
        y_tick_size = y_tick_b.slider(
            "크기", 6, 40, 10, key=f"{prefix}_y_tick_size", disabled=not y_tick_visible
        )
        y_tick_rotation = y_tick_c.slider(
            "회전", -180, 180, 0, 5, key=f"{prefix}_y_tick_rotation",
            disabled=not y_tick_visible,
        )
        y_tick_weight = y_tick_d.selectbox(
            "굵기", ["normal", "bold"], key=f"{prefix}_y_tick_weight",
            disabled=not y_tick_visible,
        )
    ratio_basis = "total"
    if aggregation == "ratio" and (
        chart_type in {"grouped_bar", "stacked_bar", "scatter_bubble", "heatmap"}
        or (chart_type == "bar" and selected_group)
    ):
        ratio_basis = st.selectbox(
            "비율 기준", ["total", "within_x", "within_y"],
            format_func=lambda value: {
                "total": "전체 기준",
                "within_x": "컬럼 내부 기준" if chart_type == "bar" else "X1 내부 기준",
                "within_y": "세부그룹 내부 기준" if chart_type == "bar" else "X2 내부 기준",
            }[value],
            key=f"{prefix}_ratio_basis",
        )
    category_orders = {}
    category_variables = [(x, "컬럼" if chart_type == "bar" else "X1"), (selected_y, "X2")]
    if chart_type == "bar":
        category_variables.append((selected_group, "세부그룹"))
    rendered_category_variables = set()
    for variable, label in category_variables:
        if variable in rendered_category_variables:
            continue
        rendered_category_variables.add(variable)
        if variable and variable in columns and variable not in numeric_columns:
            observed = list(dict.fromkeys(st.session_state.df_clean[variable].dropna().astype(str).tolist()))
            selected_order = st.multiselect(
                f"{label} 범주 표시 순서", observed, default=observed,
                key=f"{prefix}_{variable}_category_order",
            )
            if selected_order != observed:
                category_orders[variable] = selected_order
    return {
        "chart_type": chart_type,
        "x": x,
        "y": selected_y,
        "group": selected_group,
        "value_column": selected_value,
        "aggregation": aggregation,
        "ratio_basis": ratio_basis,
        "variables": variables,
        "comparison_chart": comparison_chart,
        "category_orders": category_orders,
        "x_y_swap": x_y_swap,
        "title": title,
        "title_visible": title_visible,
        "title_size": title_size,
        "title_weight": title_weight,
        "x_label": x_label,
        "x_label_visible": x_label_visible,
        "x_label_size": x_label_size,
        "x_label_rotation": x_label_rotation,
        "x_label_weight": x_label_weight,
        "x_tick_visible": x_tick_visible,
        "x_tick_size": x_tick_size,
        "x_tick_rotation": x_tick_rotation,
        "x_tick_weight": x_tick_weight,
        "y_label": y_label,
        "y_label_visible": y_label_visible,
        "y_label_size": y_label_size,
        "y_label_rotation": y_label_rotation,
        "y_label_weight": y_label_weight,
        "y_tick_visible": y_tick_visible,
        "y_tick_size": y_tick_size,
        "y_tick_rotation": y_tick_rotation,
        "y_tick_weight": y_tick_weight,
        "show_values": show_values,
        "bar_mode": bar_mode,
        "bubble_size": bubble_size,
        "chart_color_mode": chart_color_mode,
        "chart_colormap": chart_colormap,
        "chart_hex": chart_hex,
        "chart_alpha": chart_alpha,
    }


def _advanced_controls(index: int, basic: dict, frame: pd.DataFrame) -> AdvancedSettings:
    prefix = f"viz_{index}_adv"
    chart_type = basic["chart_type"]
    show_values = basic["show_values"]
    values = AdvancedSettings().model_dump()
    values["chart_color_mode"] = basic["chart_color_mode"]
    values["palette"] = basic["chart_colormap"]
    values["base_color"] = basic["chart_hex"]
    values["heatmap_cmap"] = basic["chart_colormap"]
    values["alpha"] = basic["chart_alpha"]
    values["title_visible"] = basic["title_visible"]
    values["x_label_visible"] = basic["x_label_visible"]
    values["y_label_visible"] = basic["y_label_visible"]
    for setting_name in (
        "title_size", "title_weight",
        "x_label_size", "x_label_rotation", "x_label_weight",
        "x_tick_visible", "x_tick_size", "x_tick_rotation", "x_tick_weight",
        "y_label_size", "y_label_rotation", "y_label_weight",
        "y_tick_visible", "y_tick_size", "y_tick_rotation", "y_tick_weight",
    ):
        values[setting_name] = basic[setting_name]
    sort_options = ["none", "ascending", "descending"]
    sort_labels = {"none": "정렬 안 함", "ascending": "오름차순", "descending": "내림차순"}

    st.markdown("##### 데이터 표시 범위(Chart Element Range)")
    if chart_type == "pie":
        pie_sort_a, pie_sort_b = st.columns(2, gap="small")
        pie_sort_options = ["ascending", "descending", "none"]
        values["pie_value_sort"] = pie_sort_a.selectbox(
            "값 기준 정렬", pie_sort_options, index=2,
            format_func=lambda item: sort_labels[item], key=f"{prefix}_pie_value_sort",
        )
        values["pie_category_sort"] = pie_sort_b.selectbox(
            "범주 기준 정렬", pie_sort_options, index=2,
            format_func=lambda item: sort_labels[item], key=f"{prefix}_pie_category_sort",
        )
    elif chart_type != "correlation_heatmap":
        sort_a, sort_b = st.columns(2, gap="small")
        values["x_sort"] = sort_a.selectbox(
            "X축 값 정렬", sort_options, index=1,
            format_func=lambda item: sort_labels[item], key=f"{prefix}_x_sort"
        )
        if chart_type == "histogram":
            values["y_sort"] = sort_b.selectbox(
                "Y축 값 정렬", ["none"], disabled=True,
                format_func=lambda item: sort_labels[item], key=f"{prefix}_histogram_y_sort",
            )
        else:
            y_sort_label = (
                "세부그룹(X2) 값 정렬"
                if chart_type == "bar" and basic.get("group")
                else "Y축 값 정렬"
            )
            values["y_sort"] = sort_b.selectbox(
                y_sort_label, sort_options, index=1,
                format_func=lambda item: sort_labels[item], key=f"{prefix}_y_sort"
            )
    range_a, range_b = st.columns(2, gap="small")
    if chart_type == "histogram":
        values["element_range"] = range_a.selectbox(
            "범위 기준", ["all"], disabled=True,
            format_func=lambda item: "전체", key=f"{prefix}_histogram_element_range",
        )
        range_b.number_input(
            "N 값", 1, 500, 20, disabled=True, key=f"{prefix}_histogram_top"
        )
        values["top_n"] = None
    else:
        values["element_range"] = range_a.selectbox(
            "범위 기준", ["all", "top", "bottom"],
            format_func=lambda item: {"all": "전체", "top": "상위 N개", "bottom": "하위 N개"}[item],
            index=0, key=f"{prefix}_element_range",
        )
        values["top_n"] = (
            range_b.number_input("N 값", 1, 500, 20, key=f"{prefix}_top")
            if values["element_range"] != "all" else None
        )
    values["rank_basis"] = "value"
    range_d, range_e = st.columns(2, gap="small")
    values["remaining_items"] = range_d.selectbox(
        "나머지 항목", ["exclude", "other"],
        format_func=lambda item: "제외" if item == "exclude" else "기타 항목으로 통합",
        key=f"{prefix}_remaining",
        disabled=chart_type == "histogram" or values["element_range"] == "all",
    )
    values["include_missing"] = range_e.checkbox("결측값 포함", False, key=f"{prefix}_missing")

    st.markdown("##### 차트 제목(Title)")
    title_a, title_b = st.columns(2, gap="small")
    values["title_color"] = title_a.color_picker("색상", "#172033", key=f"{prefix}_title_color", disabled=not values["title_visible"])
    values["title_location"] = title_b.selectbox("위치", ["left", "center", "right"], index=1, key=f"{prefix}_title_location", disabled=not values["title_visible"])
    title_e, title_f = st.columns(2, gap="small")
    values["title_alpha"] = title_e.slider("투명도", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_title_alpha", disabled=not values["title_visible"])
    values["title_pad"] = title_f.slider("차트 영역과 간격", 0.0, 100.0, 6.0, 1.0, key=f"{prefix}_title_pad", disabled=not values["title_visible"])

    if chart_type != "pie":
        st.markdown("##### X축 및 Y축(Axis)")
        for axis_name, label_location_options in (("x", ["left", "center", "right"]), ("y", ["bottom", "center", "top"])):
            st.caption(translate(f"{axis_name.upper()}축 라벨과 눈금 스타일"))
            label_disabled = not values[f"{axis_name}_label_visible"]
            label_a, label_b, label_c, label_d = st.columns(4, gap="small")
            values[f"{axis_name}_label_color"] = label_a.color_picker("라벨 색상", "#172033", key=f"{prefix}_{axis_name}_label_color", disabled=label_disabled)
            values[f"{axis_name}_label_alpha"] = label_b.slider("라벨 투명도", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_{axis_name}_label_alpha", disabled=label_disabled)
            values[f"{axis_name}_label_location"] = label_c.selectbox("라벨 위치", label_location_options, index=1, key=f"{prefix}_{axis_name}_label_location", disabled=label_disabled)
            values[f"{axis_name}_label_pad"] = label_d.slider("축과 간격", 0.0, 100.0, 4.0, 1.0, key=f"{prefix}_{axis_name}_label_pad", disabled=label_disabled)
            tick_disabled = not values[f"{axis_name}_tick_visible"]
            tick_e, tick_f, tick_g = st.columns(3, gap="small")
            values[f"{axis_name}_tick_color"] = tick_e.color_picker("눈금 색상", "#334155", key=f"{prefix}_{axis_name}_tick_color", disabled=tick_disabled)
            values[f"{axis_name}_tick_alpha"] = tick_f.slider("눈금 투명도", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_{axis_name}_tick_alpha", disabled=tick_disabled)
            values[f"{axis_name}_tick_pad"] = tick_g.slider("축선과 간격", 0.0, 100.0, 3.5, 0.5, key=f"{prefix}_{axis_name}_tick_pad", disabled=tick_disabled)
            axis_kind, _ = _axis_kind(frame, basic, axis_name)
            if axis_kind == "numeric":
                values[f"{axis_name}_tick_number_format"] = st.selectbox(
                    "숫자 표시 형식", ["auto", "integer", "decimal1", "decimal2", "thousands", "percent"],
                    format_func=lambda value: {
                        "auto": "자동", "integer": "정수", "decimal1": "소수점 1자리",
                        "decimal2": "소수점 2자리", "thousands": "천 단위 구분", "percent": "백분율",
                    }[value], key=f"{prefix}_{axis_name}_tick_number_format",
                )
        st.markdown("##### 격자(Grid)")
        grid_toggle_a, grid_toggle_b = st.columns(2, gap="small")
        values["grid_x"] = grid_toggle_a.checkbox("X축 격자", False, key=f"{prefix}_grid_x")
        values["grid_y"] = grid_toggle_b.checkbox("Y축 격자", True, key=f"{prefix}_grid_y")
        values["grid"] = values["grid_x"] or values["grid_y"]
        if values["grid"]:
            grid_a, grid_b, grid_c, grid_d, grid_e = st.columns(5, gap="small")
            values["grid_which"] = grid_a.selectbox("표시 위치", ["major", "minor", "both"], key=f"{prefix}_grid_which")
            values["grid_style"] = grid_b.selectbox("격자 스타일", ["-", "--", "-.", ":"], key=f"{prefix}_grid_style")
            values["grid_width"] = grid_c.slider("격자 굵기", 0.1, 10.0, 0.7, 0.1, key=f"{prefix}_grid_width")
            values["grid_color"] = grid_d.color_picker("격자 색상", "#CBD5E1", key=f"{prefix}_grid_color")
            values["grid_alpha"] = grid_e.slider("격자 투명도", 0.0, 1.0, 0.22, 0.02, key=f"{prefix}_grid_alpha")

    supports_legend = chart_type in {"pie", "histogram", "grouped_bar", "stacked_bar"} or bool(basic.get("group"))
    if supports_legend:
        st.markdown("##### 범례")
        legend_a, legend_b = st.columns(2, gap="small")
        values["legend"] = legend_a.checkbox("범례 표시", True, key=f"{prefix}_legend")
        if values["legend"]:
            values["legend_location"] = legend_b.selectbox(
                "범례 위치",
                ["best", "upper center", "lower center", "center left", "center right", "center", "outside_right", "outside_bottom"],
                format_func=lambda value: {
                    "best": "자동", "upper center": "위", "lower center": "아래",
                    "center left": "왼쪽", "center right": "오른쪽", "center": "차트 내부",
                    "outside_right": "차트 외부(오른쪽)", "outside_bottom": "차트 외부(아래)",
                }[value],
                key=f"{prefix}_legend_location",
            )
            legend_c, legend_d, legend_e, legend_f = st.columns(4, gap="small")
            values["legend_title"] = legend_c.text_input("범례 제목", key=f"{prefix}_legend_title")
            values["legend_font_size"] = legend_d.slider("글자 크기", 4, 40, 9, key=f"{prefix}_legend_size")
            values["legend_font_weight"] = legend_e.selectbox("글자 굵기", ["normal", "bold"], key=f"{prefix}_legend_weight")
            values["legend_color"] = legend_f.color_picker("글자색", "#172033", key=f"{prefix}_legend_color")
            legend_g, legend_h, legend_i = st.columns(3, gap="small")
            values["legend_alpha"] = legend_g.slider("글자 투명도", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_legend_alpha")
            values["legend_direction"] = legend_h.radio("항목 방향", ["vertical", "horizontal"], horizontal=True, key=f"{prefix}_legend_direction")
            values["legend_background_alpha"] = legend_i.slider("배경 투명도", 0.0, 1.0, 0.8, 0.05, key=f"{prefix}_legend_bg_alpha")
            values["legend_background"] = st.color_picker("범례 배경색", "#FFFFFF", key=f"{prefix}_legend_bg")
            values["legend_border_visible"] = st.checkbox("범례 테두리 표시", False, key=f"{prefix}_legend_border")
            if values["legend_border_visible"]:
                border_a, border_b = st.columns(2, gap="small")
                values["legend_border_color"] = border_a.color_picker("범례 테두리 색상", "#CBD5E1", key=f"{prefix}_legend_border_color")
                values["legend_border_width"] = border_b.slider("범례 테두리 두께", 0.0, 10.0, 0.8, 0.1, key=f"{prefix}_legend_border_width")
    else:
        values["legend"] = False

    if show_values:
        st.markdown("##### 값(Data Label) 표시")
        if chart_type == "pie":
            label_a, label_b = st.columns(2, gap="small")
            values["pie_label_mode"] = label_a.selectbox(
                "표시 방식", ["ratio", "label", "label_ratio"],
                format_func=lambda item: {"ratio": "비율", "label": "라벨", "label_ratio": "비율 + 라벨"}[item],
                key=f"{prefix}_pie_label_mode",
            )
            if values["pie_label_mode"] in {"ratio", "label_ratio"}:
                values["pie_ratio_format"] = label_b.selectbox(
                    "비율 표시 형식", [".0f", ".1f", ".2f"],
                    format_func=lambda item: {".0f": "0%", ".1f": "0.0%", ".2f": "0.00%"}[item],
                    index=1, key=f"{prefix}_pie_ratio_format",
                )
        else:
            format_a, format_b = st.columns(2, gap="small")
            values["number_format"] = format_a.selectbox(
                "숫자 형식", [",.0f", ",.1f", ",.2f", ".1%"], index=1, key=f"{prefix}_format"
            )
            values["unit"] = format_b.text_input("표시 단위", "", key=f"{prefix}_unit")
        position_a, font_a, font_b = st.columns(3, gap="small")
        values["label_position_mode"] = position_a.selectbox(
            "표시 위치", ["auto", "inside", "center", "edge", "outside", "manual"],
            format_func=lambda item: {"auto": "자동", "inside": "요소 내부", "center": "요소 중앙", "edge": "요소 끝점", "outside": "요소 바깥쪽", "manual": "직접 입력"}[item],
            key=f"{prefix}_label_position",
        )
        values["label_font_size"] = font_a.slider(
            "라벨 폰트 크기", 4, 40, 8, key=f"{prefix}_label_font_size"
        )
        values["label_color"] = font_b.color_picker(
            "라벨 색상", "#172033", key=f"{prefix}_label_color"
        )
        font_c, font_d, font_e = st.columns(3, gap="small")
        values["label_font_weight"] = font_c.selectbox("라벨 굵기", ["normal", "bold"], key=f"{prefix}_label_weight")
        values["label_alpha"] = font_d.slider("라벨 투명도", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_label_alpha")
        values["label_rotation"] = font_e.slider("라벨 회전", -180, 180, 0, 5, key=f"{prefix}_label_rotation")
        sign_a, sign_b = st.columns(2, gap="small")
        values["label_positive_sign"] = sign_a.checkbox("양수에 + 표시", False, key=f"{prefix}_label_positive")
        values["label_negative_format"] = sign_b.selectbox(
            "음수 표시", ["minus", "parentheses"],
            format_func=lambda value: "- 기호" if value == "minus" else "괄호",
            key=f"{prefix}_label_negative",
        )
        if values["label_position_mode"] == "manual":
            offset_a, offset_b = st.columns(2, gap="small")
            values["label_offset_x"] = offset_a.number_input(
                "X 방향 이동(points)", -100.0, 100.0, 0.0, 1.0, key=f"{prefix}_label_offset_x"
            )
            values["label_offset_y"] = offset_b.number_input(
                "Y 방향 이동(points)", -100.0, 100.0, 5.0, 1.0, key=f"{prefix}_label_offset_y"
            )

    st.markdown("##### 차트별 설정")
    if chart_type in {"bar", "grouped_bar", "stacked_bar", "multi_variable"} and basic.get("comparison_chart", "bar") == "bar":
        chart_a, chart_b = st.columns(2, gap="small")
        values["orientation"] = chart_a.selectbox(
            "막대 방향", ["vertical", "horizontal"], key=f"{prefix}_orientation"
        )
        values["bar_width"] = chart_b.slider("막대 너비", 0.05, 1.0, 0.8, 0.05, key=f"{prefix}_bar_width")
        gap_a, gap_b = st.columns(2, gap="small")
        values["bar_gap"] = gap_a.slider("막대 간격", 0.0, 0.9, 0.0, 0.05, key=f"{prefix}_bar_gap")
        values["group_gap"] = gap_b.slider("그룹 간 간격", 0.0, 0.9, 0.0, 0.05, key=f"{prefix}_group_gap")
        values["bar_corner_style"] = st.selectbox(
            "막대 모서리", ["square", "rounded"],
            format_func=lambda value: "각진 모서리" if value == "square" else "둥근 모서리",
            key=f"{prefix}_bar_corner_style",
        )
        values["bar_mode"] = (
            "stacked" if chart_type == "stacked_bar"
            else "grouped" if chart_type == "grouped_bar"
            else basic.get("bar_mode", "basic") if chart_type == "bar"
            else "basic"
        )
    elif chart_type == "line" or (chart_type == "multi_variable" and basic.get("comparison_chart") == "line"):
        line_a, line_b, line_c = st.columns(3, gap="small")
        values["line_style"] = line_a.selectbox("선 스타일", ["-", "--", "-.", ":"], key=f"{prefix}_line_style")
        values["line_width"] = line_b.slider("선 두께", 0.2, 10.0, 2.0, 0.2, key=f"{prefix}_line_width")
        values["marker"] = line_c.selectbox("마커", ["o", "s", "^", "D", "x", "+"], key=f"{prefix}_marker")
        line_d, line_e, line_f = st.columns(3, gap="small")
        values["marker_size"] = line_d.slider("마커 크기", 1.0, 30.0, 5.0, 1.0, key=f"{prefix}_marker_size")
        values["area_fill"] = line_e.checkbox("영역 채우기", False, key=f"{prefix}_area")
        values["line_curvature"] = line_f.slider("선의 곡선률", 0.0, 1.0, 0.0, 0.05, key=f"{prefix}_curvature")
        event_enabled = st.checkbox("이벤트 구간 음영", False, key=f"{prefix}_event_enabled")
        if event_enabled:
            event_a, event_b, event_c, event_d = st.columns(4, gap="small")
            values["event_start"] = event_a.text_input("시작값", key=f"{prefix}_event_start")
            values["event_end"] = event_b.text_input("종료값", key=f"{prefix}_event_end")
            values["event_color"] = event_c.color_picker("음영 색상", "#F59E0B", key=f"{prefix}_event_color")
            values["event_alpha"] = event_d.slider("음영 투명도", 0.0, 1.0, 0.15, 0.05, key=f"{prefix}_event_alpha")
    elif chart_type == "pie":
        geometry_a, geometry_b = st.columns(2, gap="small")
        values["pie_start_angle"] = geometry_a.slider("시작 각도", 0, 360, 90, key=f"{prefix}_pie_angle")
        values["pie_min_ratio"] = geometry_b.slider(
            "최소 비율 미만을 기타로 통합(%)", 0.0, 30.0, 0.0, 0.5, key=f"{prefix}_pie_min"
        )
        st.markdown("###### Pie 분리")
        categories = list(dict.fromkeys(frame[basic["x"]].dropna().astype(str).tolist()))
        if "기타" not in categories:
            categories.append("기타")
        values["pie_explode_labels"] = st.multiselect(
            "분리할 파이 선택", categories, key=f"{prefix}_pie_explode_labels"
        )
        if values["pie_explode_labels"]:
            values["pie_explode_width"] = st.slider(
                "분리 Width", 0.0, 0.5, 0.08, 0.01, key=f"{prefix}_pie_explode_width"
            )

        st.markdown("###### 그림자")
        values["pie_shadow"] = st.checkbox("그림자 사용", False, key=f"{prefix}_shadow")
        if values["pie_shadow"]:
            shadow_a, shadow_b, shadow_c = st.columns(3, gap="small")
            values["pie_shadow_width"] = shadow_a.slider(
                "그림자 두께", 0.0, 0.2, 0.04, 0.01, key=f"{prefix}_shadow_width"
            )
            values["pie_shadow_color"] = shadow_b.color_picker(
                "그림자 색상", "#475569", key=f"{prefix}_shadow_color"
            )
            values["pie_shadow_alpha"] = shadow_c.slider(
                "그림자 Alpha", 0.0, 1.0, 0.35, 0.05, key=f"{prefix}_shadow_alpha"
            )

        st.markdown("###### Chart 테두리")
        edge_a, edge_b, edge_c = st.columns(3, gap="small")
        values["edge_width"] = edge_a.slider("테두리 두께", 0.0, 5.0, 0.6, 0.1, key=f"{prefix}_edge_width")
        values["edge_color"] = edge_b.color_picker("테두리 색상", "#334155", key=f"{prefix}_edge_color")
        values["pie_edge_alpha"] = edge_c.slider("테두리 Alpha", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_edge_alpha")

        st.markdown("###### 도넛과 중앙 테두리")
        values["donut"] = st.checkbox("도넛 사용", False, key=f"{prefix}_donut")
        if values["donut"]:
            donut_a, donut_b, donut_c = st.columns(3, gap="small")
            values["donut_hole_size"] = donut_a.slider(
                "구멍 크기", 0.05, 0.9, 0.5, 0.05, key=f"{prefix}_donut_hole"
            )
            ring_key = f"{prefix}_donut_ring"
            maximum_ring = round(max(0.05, 1.0 - values["donut_hole_size"]), 2)
            if ring_key in st.session_state and st.session_state[ring_key] > maximum_ring:
                st.session_state[ring_key] = maximum_ring
            values["donut_ring_width"] = donut_b.slider(
                "링 두께", 0.05, maximum_ring, min(0.4, maximum_ring), 0.05, key=ring_key
            )
            values["donut_center_color"] = donut_c.color_picker(
                "중앙 배경색", "#FFFFFF", key=f"{prefix}_donut_center_color"
            )
            values["donut_center_border"] = st.checkbox(
                "중앙 테두리 사용", False, key=f"{prefix}_donut_center_border"
            )
            if values["donut_center_border"]:
                center_a, center_b, center_c = st.columns(3, gap="small")
                values["donut_center_border_width"] = center_a.slider(
                    "중앙 테두리 두께", 0.0, 10.0, 1.0, 0.5, key=f"{prefix}_donut_border_width"
                )
                values["donut_center_border_color"] = center_b.color_picker(
                    "중앙 테두리 색상", "#334155", key=f"{prefix}_donut_border_color"
                )
                values["donut_center_border_alpha"] = center_c.slider(
                    "중앙 테두리 Alpha", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_donut_border_alpha"
                )
    elif chart_type == "histogram":
        histogram_a, histogram_b, histogram_c = st.columns(3, gap="small")
        values["histogram_bins"] = histogram_a.slider("구간 수", 2, 100, 10, key=f"{prefix}_bins")
        values["histogram_density"] = histogram_b.checkbox("밀도 표시", False, key=f"{prefix}_density")
        use_width = histogram_c.checkbox("구간 폭 사용", False, key=f"{prefix}_bin_width_enabled")
        if use_width:
            values["histogram_bin_width"] = st.number_input("구간 폭", min_value=0.000001, value=1.0, key=f"{prefix}_bin_width")
    elif chart_type == "scatter_plot":
        scatter_a, scatter_b, scatter_c = st.columns(3, gap="small")
        values["scatter_size"] = scatter_a.slider("점 최대 크기", 5.0, 1000.0, 80.0, 5.0, key=f"{prefix}_scatter_size")
        values["trendline"] = scatter_b.checkbox("추세선", False, key=f"{prefix}_trend")
        values["scatter_marker"] = scatter_c.selectbox("점 모양", ["o", "s", "^", "D", "x", "+"], key=f"{prefix}_scatter_marker")
    elif chart_type == "scatter_bubble":
        scatter_a, scatter_b = st.columns(2, gap="small")
        values["trendline"] = scatter_a.checkbox("추세선", False, key=f"{prefix}_trend")
        values["scatter_marker"] = scatter_b.selectbox(
            "점 모양", ["o", "s", "^", "D", "x", "+"], key=f"{prefix}_scatter_marker"
        )
        values["scatter_size"] = basic["bubble_size"]
    elif chart_type in {"heatmap", "correlation_heatmap"}:
        heatmap_a, heatmap_b = st.columns(2, gap="small")
        values["heatmap_annotate"] = heatmap_a.checkbox("셀 값 표시", True, key=f"{prefix}_heatmap_annot")
        values["heatmap_colorbar"] = heatmap_b.checkbox("컬러바 표시", True, key=f"{prefix}_heatmap_cbar")
        values["heatmap_linewidth"] = st.slider("셀 경계선", 0.0, 5.0, 0.5, 0.1, key=f"{prefix}_heatmap_line")
        heat_edge_a, heat_edge_b = st.columns(2, gap="small")
        values["heatmap_linecolor"] = heat_edge_a.color_picker("셀 경계 색상", "#FFFFFF", key=f"{prefix}_heatmap_linecolor")
        values["heatmap_linealpha"] = heat_edge_b.slider("셀 경계 투명도", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_heatmap_linealpha")
    if chart_type in {"bar", "grouped_bar", "stacked_bar", "histogram", "scatter_plot", "scatter_bubble"}:
        st.markdown("##### 요소 테두리")
        element_a, element_b = st.columns(2, gap="small")
        values["edge_color"] = element_a.color_picker("테두리 색상", "#334155", key=f"{prefix}_edge_color")
        values["edge_width"] = element_b.slider("테두리 두께", 0.0, 5.0, 0.6, 0.1, key=f"{prefix}_edge_width")
    return AdvancedSettings.model_validate(values)


def _axis_kind(frame: pd.DataFrame, basic: dict, axis: str) -> tuple[str, str | None]:
    chart_type = basic["chart_type"]
    logical_axis = axis
    if basic.get("x_y_swap"):
        logical_axis = "y" if axis == "x" else "x"
    if chart_type in {"pie", "correlation_heatmap"}:
        return "unavailable", None
    if chart_type == "multi_variable":
        return ("category", None) if logical_axis == "x" else ("numeric", None)
    if logical_axis == "x":
        column = basic["x"]
        if chart_type in {"bar", "grouped_bar", "stacked_bar", "heatmap"}:
            return ("date" if column and semantic_type(frame[column]) == "날짜형" else "category"), column
        if chart_type == "histogram":
            return "numeric", column
    else:
        column = basic.get("y")
        if chart_type in {"grouped_bar", "stacked_bar", "heatmap"}:
            return ("date" if column and semantic_type(frame[column]) == "날짜형" else "category"), column
        if chart_type not in {"scatter_plot", "scatter_bubble"}:
            return "numeric", None
    if column and pd.api.types.is_numeric_dtype(frame[column]):
        return "numeric", column
    if column and semantic_type(frame[column]) == "날짜형":
        return "date", column
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
        f"{axis}_date_start": None,
        f"{axis}_date_end": None,
        f"{axis}_date_tick_frequency": "auto",
        f"{axis}_date_format": "%Y-%m-%d",
    }
    with st.container(border=True):
        st.markdown(translate(f"##### {axis_name}축 제어"))
        if kind == "unavailable":
            st.selectbox("축 범위", ["사용할 수 없음"], key=f"{prefix}_disabled", disabled=True)
            st.caption("이 차트 유형에서는 해당 축을 직접 제어할 수 없습니다.")
            return result
        if kind == "date":
            parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed").dropna() if column else pd.Series(dtype="datetime64[ns]")
            if parsed.empty:
                st.selectbox("날짜축 설정", ["사용할 수 없음"], key=f"{prefix}_date_empty", disabled=True)
                return result
            mode = st.selectbox("날짜 표시 범위", ["전체 기간", "시작일~종료일"], key=f"{prefix}_{column}_date_mode")
            if mode == "시작일~종료일":
                result[f"{axis}_axis_mode"] = "date_range"
                start_col, end_col = st.columns(2, gap="small")
                result[f"{axis}_date_start"] = start_col.date_input("시작일", parsed.min().date(), key=f"{prefix}_{column}_date_start")
                result[f"{axis}_date_end"] = end_col.date_input("종료일", parsed.max().date(), key=f"{prefix}_{column}_date_end")
            date_a, date_b = st.columns(2, gap="small")
            result[f"{axis}_date_tick_frequency"] = date_a.selectbox(
                "날짜 눈금 간격", ["auto", "day", "week", "month", "quarter", "year"],
                format_func=lambda value: {"auto": "자동", "day": "일", "week": "주", "month": "월", "quarter": "분기", "year": "연도"}[value],
                key=f"{prefix}_{column}_date_frequency",
            )
            result[f"{axis}_date_format"] = date_b.selectbox(
                "날짜 표시 형식", ["%Y-%m-%d", "%Y-%m", "%m-%d", "%Y년 %m월", "%Y"],
                key=f"{prefix}_{column}_date_format",
            )
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


def _repeatable_overlay_controls(
    index: int,
    frame: pd.DataFrame,
    x_kind: str,
    x_column: str | None,
    y_kind: str,
    y_column: str | None,
) -> tuple[list[dict], list[dict]]:
    prefix = f"viz_{index}_overlay"
    references: list[dict] = []
    available = [axis for axis, kind in (("x", x_kind), ("y", y_kind)) if kind != "unavailable"]
    reference_count = int(st.number_input("추가 기준선 개수", 0, 10, 0, key=f"{prefix}_reference_count"))
    for number in range(reference_count):
        with st.container(border=True):
            st.caption(f"기준선 {number + 1}")
            targets = st.multiselect(
                "기준선 방향", available, default=(["y"] if "y" in available else available[:1]),
                format_func=lambda axis: "수직 기준선(X축)" if axis == "x" else "수평 기준선(Y축)",
                key=f"{prefix}_ref_{number}_targets",
            )
            style_a, style_b, style_c, style_d = st.columns(4, gap="small")
            style = style_a.selectbox("스타일", ["-", "--", "-.", ":"], index=1, key=f"{prefix}_ref_{number}_style")
            width = style_b.slider("두께", 0.1, 10.0, 1.2, 0.1, key=f"{prefix}_ref_{number}_width")
            color = style_c.color_picker("색상", "#475569", key=f"{prefix}_ref_{number}_color")
            alpha = style_d.slider("투명도", 0.0, 1.0, 0.8, 0.05, key=f"{prefix}_ref_{number}_alpha")
            label_visible = st.checkbox("기준선 라벨 표시", True, key=f"{prefix}_ref_{number}_label_visible")
            label_a, label_b, label_c, label_d = st.columns(4, gap="small")
            label = label_a.text_input("라벨", key=f"{prefix}_ref_{number}_label", disabled=not label_visible)
            label_position = label_b.selectbox("라벨 위치", ["start", "center", "end"], index=2, key=f"{prefix}_ref_{number}_label_position", disabled=not label_visible)
            label_size = label_c.slider("라벨 크기", 4, 40, 9, key=f"{prefix}_ref_{number}_label_size", disabled=not label_visible)
            label_alpha = label_d.slider("라벨 투명도", 0.0, 1.0, 0.9, 0.05, key=f"{prefix}_ref_{number}_label_alpha", disabled=not label_visible)
            label_e, label_f, label_g = st.columns(3, gap="small")
            label_weight = label_e.selectbox("라벨 굵기", ["normal", "bold"], key=f"{prefix}_ref_{number}_label_weight", disabled=not label_visible)
            label_color = label_f.color_picker("라벨 색상", "#475569", key=f"{prefix}_ref_{number}_label_color", disabled=not label_visible)
            label_pad = label_g.slider("기준선과 간격", -100.0, 100.0, 2.0, 1.0, key=f"{prefix}_ref_{number}_label_pad", disabled=not label_visible)
            entry = dict(targets=targets, style=style, width=width, color=color, alpha=alpha,
                         label=label, label_visible=label_visible, label_position=label_position,
                         label_size=label_size, label_weight=label_weight, label_color=label_color,
                         label_alpha=label_alpha, label_pad=label_pad)
            for axis, kind, column in (("x", x_kind, x_column), ("y", y_kind, y_column)):
                if axis not in targets:
                    continue
                ref_kind = _reference_kind(frame, column, kind)
                entry[f"{axis}_kind"] = ref_kind
                if ref_kind == "numeric":
                    entry[f"{axis}_value"] = st.number_input(
                        f"{axis.upper()}축 수치", value=0.0, key=f"{prefix}_ref_{number}_{axis}_numeric"
                    )
                elif ref_kind == "date":
                    entry[f"{axis}_value"] = st.date_input(
                        f"{axis.upper()}축 날짜", key=f"{prefix}_ref_{number}_{axis}_date"
                    )
                else:
                    options = list(dict.fromkeys(frame[column].dropna().astype(str).tolist())) if column else []
                    if options:
                        entry[f"{axis}_value"] = st.selectbox(
                            f"{axis.upper()}축 범주", options, key=f"{prefix}_ref_{number}_{axis}_category"
                        )
            references.append(entry)

    annotations: list[dict] = []
    annotation_count = int(st.number_input("주석 개수", 0, 10, 0, key=f"{prefix}_annotation_count"))
    for number in range(annotation_count):
        with st.container(border=True):
            st.caption(f"주석 {number + 1}")
            text = st.text_input("내용", key=f"{prefix}_note_{number}_text")
            position_a, position_b, position_c = st.columns(3, gap="small")
            x = position_a.number_input("X 위치", value=0.5, key=f"{prefix}_note_{number}_x")
            y = position_b.number_input("Y 위치", value=0.5, key=f"{prefix}_note_{number}_y")
            coordinate = position_c.selectbox(
                "위치 기준", ["axes", "data"], format_func=lambda value: "차트 비율" if value == "axes" else "데이터 좌표",
                key=f"{prefix}_note_{number}_coordinate",
            )
            style_a, style_b, style_c, style_d = st.columns(4, gap="small")
            size = style_a.slider("크기", 4, 60, 10, key=f"{prefix}_note_{number}_size")
            weight = style_b.selectbox("굵기", ["normal", "bold"], key=f"{prefix}_note_{number}_weight")
            color = style_c.color_picker("색상", "#172033", key=f"{prefix}_note_{number}_color")
            alpha = style_d.slider("투명도", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_note_{number}_alpha")
            extra_a, extra_b, extra_c = st.columns(3, gap="small")
            rotation = extra_a.slider("회전", -180.0, 180.0, 0.0, 5.0, key=f"{prefix}_note_{number}_rotation")
            horizontal_alignment = extra_b.selectbox("가로 정렬", ["left", "center", "right"], index=1, key=f"{prefix}_note_{number}_ha")
            vertical_alignment = extra_c.selectbox("세로 정렬", ["top", "center", "bottom"], index=1, key=f"{prefix}_note_{number}_va")
            box_visible = st.checkbox("배경 상자 표시", False, key=f"{prefix}_note_{number}_box")
            box_a, box_b, box_c, box_d = st.columns(4, gap="small")
            box_color = box_a.color_picker("배경 색상", "#FFFFFF", key=f"{prefix}_note_{number}_box_color", disabled=not box_visible)
            box_alpha = box_b.slider("배경 투명도", 0.0, 1.0, 0.8, 0.05, key=f"{prefix}_note_{number}_box_alpha", disabled=not box_visible)
            box_edge_color = box_c.color_picker("테두리 색상", "#CBD5E1", key=f"{prefix}_note_{number}_box_edge", disabled=not box_visible)
            box_line_style = box_d.selectbox("테두리 스타일", ["-", "--", "-.", ":"], key=f"{prefix}_note_{number}_box_style", disabled=not box_visible)
            arrow_visible = st.checkbox("화살표 표시", False, key=f"{prefix}_note_{number}_arrow")
            arrow_x = arrow_y = None
            if arrow_visible:
                arrow_a, arrow_b = st.columns(2, gap="small")
                arrow_x = arrow_a.number_input("화살표 대상 X", value=x, key=f"{prefix}_note_{number}_arrow_x")
                arrow_y = arrow_b.number_input("화살표 대상 Y", value=y, key=f"{prefix}_note_{number}_arrow_y")
            if text.strip():
                annotations.append(dict(text=text, x=x, y=y, coordinate=coordinate, size=size,
                                        weight=weight, color=color, alpha=alpha, rotation=rotation,
                                        horizontal_alignment=horizontal_alignment,
                                        vertical_alignment=vertical_alignment, box_visible=box_visible,
                                        box_color=box_color, box_alpha=box_alpha,
                                        box_edge_color=box_edge_color, box_line_style=box_line_style,
                                        arrow_visible=arrow_visible,
                                        arrow_x=arrow_x, arrow_y=arrow_y))
    return references, annotations


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
    st.markdown("##### 축 방향과 스케일")
    control_names = []
    if x_kind == "numeric":
        control_names.append("x_log")
    if y_kind == "numeric":
        control_names.append("y_log")
    if x_kind != "unavailable":
        control_names.append("invert_x")
    if y_kind != "unavailable":
        control_names.append("invert_y")
    control_columns = st.columns(len(control_names), gap="small") if control_names else []
    controls = dict(zip(control_names, control_columns))
    x_log = controls["x_log"].checkbox("X 로그 스케일", False, key=f"{prefix}_xlog") if "x_log" in controls else False
    y_log = controls["y_log"].checkbox("Y 로그 스케일", False, key=f"{prefix}_ylog") if "y_log" in controls else False
    invert_x = controls["invert_x"].checkbox("X축 반전", False, key=f"{prefix}_invertx") if "invert_x" in controls else False
    invert_y = controls["invert_y"].checkbox("Y축 반전", False, key=f"{prefix}_inverty") if "invert_y" in controls else False
    st.markdown("##### 기준선과 주석")
    reference_lines, annotations = _repeatable_overlay_controls(
        index, frame, x_kind, x_column, y_kind, y_column
    )
    normalize = False
    cumulative = False
    if chart_type == "line":
        cumulative = st.checkbox("누적합", False, key=f"{prefix}_cumulative")
    elif chart_type in {"scatter_bubble", "heatmap"}:
        normalize = st.checkbox("정규화", False, key=f"{prefix}_normalize")
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
    elif chart_type in {"scatter_plot", "scatter_bubble"}:
        a, b, c = st.columns(3)
        jitter_enabled = a.checkbox("지터링 사용", False, key=f"{prefix}_jitter_enabled")
        show_correlation = b.checkbox("상관계수 표시", False, key=f"{prefix}_corr")
        highlight_outliers = c.checkbox("이상치 강조", False, key=f"{prefix}_outlier")
        if jitter_enabled:
            jitter = st.slider(
                "지터링 너비", 0.01, 2.0, 0.1, 0.01, key=f"{prefix}_jitter"
            )
    elif chart_type in {"heatmap", "correlation_heatmap"}:
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
        reference_lines=reference_lines,
        annotations=annotations,
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
        with st.container(key=f"visualization_layout_{index}"):
            st.markdown(f"#### {translate('차트')} {index}")
            with st.expander("차트·데이터 설정", expanded=True):
                basic = _basic_controls(index, columns, numeric_columns)
            with st.expander("Subplot별 Axes 설정", expanded=False):
                advanced = _advanced_controls(index, basic, frame)
                deep = _deep_controls(index, basic, frame)
        ui_only_keys = {
            "bar_mode", "bubble_size", "chart_color_mode", "chart_colormap", "chart_hex",
            "chart_alpha", "title_visible", "x_label_visible", "y_label_visible",
            "title_size", "title_weight",
            "x_label_size", "x_label_rotation", "x_label_weight",
            "x_tick_visible", "x_tick_size", "x_tick_rotation", "x_tick_weight",
            "y_label_size", "y_label_rotation", "y_label_weight",
            "y_tick_visible", "y_tick_size", "y_tick_rotation", "y_tick_weight",
        }
        chart_values = {key: value for key, value in basic.items() if key not in ui_only_keys}
        specs.append(ChartSpec.model_validate({**chart_values, "advanced": advanced, "deep": deep}))
    return specs


def _editable_pydantic_specs(specs: list[ChartSpec], method: str) -> list[ChartSpec]:
    if not specs:
        return []
    generated = json.dumps([spec.model_dump(mode="json") for spec in specs], ensure_ascii=False, indent=2)
    signature_key = f"visualization_pydantic_signature_{method}"
    editor_key = f"visualization_pydantic_editor_{method}"
    if st.session_state.get(signature_key) != generated:
        st.session_state[signature_key] = generated
        st.session_state[editor_key] = generated
    with st.container(key="visualization_pydantic_section"):
        st.divider()
        st.markdown(f"#### {translate('직접 설정·조회')}")
        with st.expander("ChartSpec JSON", expanded=False):
            st.caption("아래 JSON을 직접 수정하면 Pydantic 값으로 검증하고 시각화합니다.")
            edited = st.text_area("ChartSpec JSON", height=360, key=editor_key)
    payload = json.loads(edited)
    if not isinstance(payload, list):
        raise ValueError("ChartSpec JSON의 최상위 값은 배열이어야 합니다.")
    return [ChartSpec.model_validate(item) for item in payload]


def _render_result(result) -> None:
    st.markdown("#### 시각화 결과")
    st.pyplot(result.figure, clear_figure=False, width="stretch")
    for index, artifact in enumerate(result.artifacts, 1):
        with st.expander(f"Chart {index} · 통계자료와 요약 인사이트", expanded=index == 1):
            insight = summarize_artifact(
                artifact.statistics, artifact.spec, language=current_language()
            )
            st.markdown(insight.replace("\n", "  \n"))
            st.dataframe(
                localized_table(artifact.statistics, value_columns=("분석 타입",)),
                width="stretch",
                hide_index=True,
            )
            mapping = artifact.statistics.attrs.get("category_mapping")
            if mapping is not None:
                st.caption("범주형 변수의 수치 인덱스 매핑")
                st.dataframe(localized_table(mapping), width="stretch", hide_index=True)
            mappings = artifact.statistics.attrs.get("category_mappings", {})
            for variable, variable_mapping in mappings.items():
                st.caption(f"{variable} 범주형 변수의 수치 인덱스 매핑")
                st.dataframe(localized_table(variable_mapping), width="stretch", hide_index=True)
    st.markdown("### 결과 다운로드")
    formats = result.figure_spec.output_formats
    cols = st.columns(max(len(formats) + 1, 1))
    mime_types = {"png": "image/png", "jpg": "image/jpeg", "svg": "image/svg+xml", "pdf": "application/pdf"}
    for column, fmt in zip(cols[: len(formats)], formats):
        column.download_button(
            fmt.upper(),
            data=figure_to_bytes(result, fmt),
            file_name=f"{result.figure_spec.filename}.{fmt}",
            mime=mime_types[fmt],
            width="stretch",
            key=f"viz_download_{fmt}",
        )
    source_filename = st.session_state.source_filename or "data"
    cols[-1].download_button(
        "Source JSON",
        data=source_payload_bytes(result, source_filename),
        file_name=f"{result.figure_spec.filename}_source.json",
        mime="application/json",
        width="stretch",
        key="viz_download_source",
    )


def render_visualization(*, preserve_existing_result: bool = False) -> None:
    st.header("데이터 시각화")
    st.markdown(
        """
        <style>
        div[class*="st-key-visualization_figure_settings"] div[data-testid="stVerticalBlock"],
        div[class*="st-key-visualization_layout_"] div[data-testid="stVerticalBlock"] {
            gap: 0.22rem;
        }
        div[class*="st-key-visualization_figure_settings"] div[data-testid="stHorizontalBlock"],
        div[class*="st-key-visualization_layout_"] div[data-testid="stHorizontalBlock"] {
            gap: 0.45rem;
        }
        div[class*="st-key-visualization_figure_settings"] div[data-testid="stExpanderDetails"],
        div[class*="st-key-visualization_layout_"] div[data-testid="stExpanderDetails"] {
            padding: 0.4rem 0.55rem 0.5rem;
        }
        div[class*="st-key-visualization_figure_settings"] div[data-testid="stVerticalBlockBorderWrapper"],
        div[class*="st-key-visualization_layout_"] div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.4rem 0.55rem;
        }
        div[class*="st-key-visualization_figure_settings"] div[data-testid="stWidgetLabel"],
        div[class*="st-key-visualization_layout_"] div[data-testid="stWidgetLabel"] {
            margin-bottom: -0.15rem;
        }
        div[class*="st-key-visualization_figure_settings"] h4,
        div[class*="st-key-visualization_figure_settings"] h5,
        div[class*="st-key-visualization_figure_settings"] h6,
        div[class*="st-key-visualization_layout_"] h4,
        div[class*="st-key-visualization_layout_"] h5,
        div[class*="st-key-visualization_layout_"] h6 {
            margin: 0.15rem 0 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    frame = st.session_state.df_clean
    st.caption("시각화와 통계자료는 현재 전처리 데이터 `df_clean`을 기준으로 생성됩니다.")
    with st.expander("변수별 데이터 타입", expanded=False):
        st.dataframe(
            localized_table(variable_type_table(frame), value_columns=("분석 타입",)),
            width="stretch",
            hide_index=True,
    )
    figure_spec, rows, columns = _figure_controls()
    chart_count = rows * columns
    try:
        specs = _structured_specs(frame, chart_count)
        specs = _editable_pydantic_specs(specs, "structured")
        render_signature = json.dumps(
            {
                "figure": figure_spec.model_dump(mode="json"),
                "charts": [spec.model_dump(mode="json") for spec in specs],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        action_a, action_b = st.columns([1, 3], gap="small")
        auto_update = action_a.checkbox(
            "설정 변경 즉시 반영", False, key="visualization_auto_update",
            help="첫 시각화 실행 이후 설정 변경 시 현재 차트를 자동 갱신합니다.",
        )
        run_requested = action_b.button(
            "시각화 실행", type="primary", key="run_visualization", width="stretch"
        )
        previous_signature = st.session_state.get("visualization_last_render_signature")
        should_build = run_requested or (
            auto_update
            and st.session_state.get("visualization_result") is not None
            and previous_signature is not None
            and render_signature != previous_signature
            and not preserve_existing_result
        )
        if should_build:
            if not specs:
                raise ValueError("시각화 설정을 입력하세요.")
            with st.spinner("통계자료를 구성하고 차트를 생성하고 있습니다..."):
                result = build_visualization(frame, specs, figure_spec)
                st.session_state.visualization_result = result
                st.session_state.visualization_last_render_signature = render_signature
                if run_requested:
                    payload = source_payload(result, st.session_state.source_filename or "data")
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
