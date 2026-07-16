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


def _advanced_controls(index: int, chart_type: str) -> AdvancedSettings:
    prefix = f"viz_{index}_adv"
    c1, c2, c3 = st.columns(3)
    sort = c1.selectbox("정렬", ["none", "ascending", "descending"], key=f"{prefix}_sort")
    top_n_enabled = c2.checkbox("상위 N개 제한", True, key=f"{prefix}_top_enabled")
    top_n = c3.number_input("상위 N", 1, 500, 20, key=f"{prefix}_top", disabled=not top_n_enabled)
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

    orientation = "vertical"
    bar_mode = "basic"
    histogram_bins = 10
    histogram_density = False
    line_style, line_width, marker, marker_size, area_fill = "-", 2.0, "o", 5.0, False
    pie_start_angle, donut, pie_shadow, pie_min_ratio = 90, False, False, 0.0
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
        a2, b2 = st.columns(2)
        marker_size = a2.slider("마커 크기", 1.0, 30.0, 5.0, 1.0, key=f"{prefix}_marker_size")
        area_fill = b2.checkbox("영역 채우기", False, key=f"{prefix}_area")
    elif chart_type == "pie":
        a, b, c = st.columns(3)
        pie_start_angle = a.slider("시작 각도", 0, 360, 90, key=f"{prefix}_pie_angle")
        donut = b.checkbox("도넛 형태", False, key=f"{prefix}_donut")
        pie_shadow = c.checkbox("그림자", False, key=f"{prefix}_shadow")
        pie_min_ratio = st.slider("최소 비율 미만을 기타로 통합(%)", 0.0, 30.0, 0.0, 0.5, key=f"{prefix}_pie_min")
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
        sort=sort,
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
        histogram_bins=histogram_bins,
        histogram_density=histogram_density,
        line_style=line_style,
        line_width=line_width,
        marker=marker,
        marker_size=marker_size,
        area_fill=area_fill,
        pie_start_angle=pie_start_angle,
        donut=donut,
        pie_shadow=pie_shadow,
        pie_min_ratio=pie_min_ratio,
        scatter_size=scatter_size,
        trendline=trendline,
        heatmap_cmap=heatmap_cmap,
        heatmap_annotate=heatmap_annotate,
        heatmap_colorbar=heatmap_colorbar,
        heatmap_linewidth=heatmap_linewidth,
    )


def _deep_controls(index: int, chart_type: str) -> DeepSettings:
    prefix = f"viz_{index}_deep"
    x_min = _number_or_none("X축 최소", f"{prefix}_xmin")
    x_max = _number_or_none("X축 최대", f"{prefix}_xmax")
    y_min = _number_or_none("Y축 최소", f"{prefix}_ymin")
    y_max = _number_or_none("Y축 최대", f"{prefix}_ymax")
    c5, c6, c7, c8 = st.columns(4)
    x_log = c5.checkbox("X 로그 스케일", False, key=f"{prefix}_xlog")
    y_log = c6.checkbox("Y 로그 스케일", False, key=f"{prefix}_ylog")
    invert_x = c7.checkbox("X축 반전", False, key=f"{prefix}_invertx")
    invert_y = c8.checkbox("Y축 반전", False, key=f"{prefix}_inverty")
    c9, c10, c11 = st.columns(3)
    reference_line = _number_or_none("기준선", f"{prefix}_reference")
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
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_log=x_log,
        y_log=y_log,
        invert_x=invert_x,
        invert_y=invert_y,
        moving_average=moving_average,
        cumulative=cumulative,
        normalize=normalize,
        reference_line=reference_line,
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
            advanced = _advanced_controls(index, basic["chart_type"])
        with st.expander("Advanced2 · 심화 설정", expanded=False):
            deep = _deep_controls(index, basic["chart_type"])
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
