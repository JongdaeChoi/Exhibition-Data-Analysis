from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from insight.models import InsightMessage
from insight.service import rebuild_chart_record
from visualization.models import ChartSpec, FigureSpec
from visualization.service import build_visualization


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _loaded_app() -> AppTest:
    frame = pd.DataFrame(
        {
            "조사일자": ["2025-01-01", "2025-02-01"],
            "참가 경로": ["검색", "추천"],
            "값": [10, 20],
        }
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["df"] = frame
    app.session_state["df_clean"] = frame.copy(deep=True)
    app.session_state["source_filename"] = "sample.csv"
    return app.run()


def test_data_load_defaults_to_basic_profile_only() -> None:
    app = _loaded_app()

    assert not app.exception
    assert app.segmented_control[0].value == "기본 현황"
    assert "전처리" not in [header.value for header in app.header]
    assert "데이터 시각화" not in [header.value for header in app.header]
    assert not app.download_button
    assert "2. 데이터 기본 현황" in {expander.label for expander in app.expander}


def test_heavy_sections_render_only_when_selected() -> None:
    app = _loaded_app()

    app.segmented_control[0].set_value("전처리")
    app.run()
    assert not app.exception
    assert "전처리" in [header.value for header in app.header]
    assert "데이터 시각화" not in [header.value for header in app.header]
    assert {button.label for button in app.download_button} == {"CSV 다운로드", "Excel 다운로드"}

    app.segmented_control[0].set_value("시각화")
    app.run()
    assert not app.exception
    assert "전처리" not in [header.value for header in app.header]
    assert "데이터 시각화" in [header.value for header in app.header]
    markdown_values = {str(item.value) for item in app.markdown}
    assert "#### 차트 1" in markdown_values
    assert "#### 직접 설정·조회" in markdown_values
    assert "## Subplot 1" not in markdown_values
    assert "#### Subplot 구성" not in markdown_values
    assert not any(
        "구조화 메뉴에서 ChartSpec을 생성" in str(item.value)
        for item in app.caption
    )
    number_labels = {control.label for control in app.number_input}
    assert {"행 개수(n1)", "열 개수(n2)"}.issubset(number_labels)
    expander_labels = {expander.label for expander in app.expander}
    assert {
        "레이아웃 설정", "차트·데이터 설정", "Subplot별 Axes 설정", "ChartSpec JSON"
    }.issubset(expander_labels)
    chart_type = next(control for control in app.selectbox if control.label == "차트 유형")
    assert "Grouped Bar Chart" not in chart_type.options
    assert "Stacked Bar Chart" not in chart_type.options
    labels = [control.label for control in app.selectbox]
    assert labels.index("컬럼 선택") < labels.index("세부그룹 선택") < labels.index("집계 방식")
    assert "순위 기준" not in labels
    assert next(control for control in app.selectbox if control.label == "X축 값 정렬").value == "ascending"
    assert next(control for control in app.selectbox if control.label == "Y축 값 정렬").value == "ascending"
    assert next(control for control in app.selectbox if control.label == "범위 기준").value == "all"
    chart_alpha = next(control for control in app.slider if control.key == "viz_1_bar_chart_alpha")
    assert chart_alpha.label == "투명도"
    assert chart_alpha.value == 0.85
    color_mode = next(control for control in app.radio if control.key == "viz_1_bar_chart_color_mode")
    assert color_mode.label == "색상 방식"
    assert color_mode.options == ["컬러맵", "HEX"]
    checkbox_labels = [control.label for control in app.checkbox]
    assert checkbox_labels.index("X-Y 축 전환") < checkbox_labels.index("값 표시")
    assert not any("제목 및 표시" in str(item.value) for item in app.markdown)
    assert {item.value for item in app.caption}.issuperset({"라벨", "눈금"})
    title_visible = next(control for control in app.checkbox if control.key == "viz_1_title_visible")
    assert title_visible.label == "표시"
    title_visible.set_value(False)
    app.run()
    assert next(control for control in app.text_input if control.key == "viz_1_title").disabled
    assert next(control for control in app.slider if control.key == "viz_1_title_size").disabled
    assert next(control for control in app.selectbox if control.key == "viz_1_title_weight").disabled
    x_label_visible = next(control for control in app.checkbox if control.key == "viz_1_x_label_visible")
    x_label_visible.set_value(False)
    app.run()
    assert next(control for control in app.text_input if control.key == "viz_1_xlabel").disabled
    assert next(control for control in app.slider if control.key == "viz_1_x_label_size").disabled
    x_tick_visible = next(control for control in app.checkbox if control.key == "viz_1_x_tick_visible")
    x_tick_visible.set_value(False)
    app.run()
    assert next(control for control in app.slider if control.key == "viz_1_x_tick_size").disabled

    subgroup = next(control for control in app.selectbox if control.label == "세부그룹 선택")
    subgroup.select("조사일자")
    app.run()
    bar_mode = next(control for control in app.radio if control.label == "막대 표시 방식")
    assert bar_mode.options == ["그룹막대", "누적막대"]
    subgroup_sort = next(
        control for control in app.selectbox if control.label == "세부그룹(X2) 값 정렬"
    )
    assert subgroup_sort.value == "ascending"

    app.segmented_control[0].set_value("인사이트")
    app.run()
    assert not app.exception
    assert "Business Insight" in [header.value for header in app.header]
    assert app.chat_input and not app.chat_input[0].disabled


def test_line_and_pie_use_simplified_data_controls() -> None:
    app = _loaded_app()
    app.segmented_control[0].set_value("시각화")
    app.run()

    chart_type = next(control for control in app.selectbox if control.label == "차트 유형")
    chart_type.select("Line Chart")
    app.run()
    labels = [control.label for control in app.selectbox]
    assert labels.index("컬럼 선택") < labels.index("세부그룹 선택") < labels.index("집계 방식")

    chart_type = next(control for control in app.selectbox if control.label == "차트 유형")
    chart_type.select("Pie Chart")
    app.run()
    labels = [control.label for control in app.selectbox]
    assert labels.index("컬럼 선택") < labels.index("집계 방식")
    aggregation = next(control for control in app.selectbox if control.label == "집계 방식")
    assert aggregation.options == ["행 개수", "행 개수(결측치 제외)", "비율"]
    assert "X2 집계 대상" not in labels
    assert "값 기준 정렬" in labels
    assert "범주 기준 정렬" in labels


def test_histogram_and_scatter_use_chart_specific_controls() -> None:
    app = _loaded_app()
    app.segmented_control[0].set_value("시각화")
    app.run()

    chart_type = next(control for control in app.selectbox if control.label == "차트 유형")
    chart_type.select("Histogram")
    app.run()
    x_sort = next(control for control in app.selectbox if control.label == "X축 값 정렬")
    y_sort = next(control for control in app.selectbox if control.label == "Y축 값 정렬")
    element_range = next(control for control in app.selectbox if control.label == "범위 기준")
    top_n = next(control for control in app.number_input if control.label == "N 값")
    assert not x_sort.disabled
    assert y_sort.disabled and y_sort.value == "none"
    assert element_range.disabled and element_range.value == "all"
    assert top_n.disabled

    chart_type = next(control for control in app.selectbox if control.label == "차트 유형")
    chart_type.select("Scatter Plot")
    app.run()
    labels = {control.label for control in app.selectbox}
    assert {"컬럼1", "컬럼2", "색상 컬럼"}.issubset(labels)
    assert "X1 변수" not in labels
    assert "X2 변수" not in labels
    assert "그룹/색상 변수" not in labels
    jitter_enabled = next(control for control in app.checkbox if control.label == "지터링 사용")
    assert not jitter_enabled.value
    assert not any(control.label == "지터링 너비" for control in app.slider)
    jitter_enabled.set_value(True)
    app.run()
    assert any(control.label == "지터링 너비" for control in app.slider)

    chart_type = next(control for control in app.selectbox if control.label == "차트 유형")
    chart_type.select("Scatter Bubble")
    app.run()
    labels = [control.label for control in app.selectbox]
    assert labels.index("컬럼1") < labels.index("컬럼2") < labels.index("집계대상") < labels.index("집계 방식")
    assert "그룹/색상 변수" not in labels
    assert "색상 컬럼" not in labels
    assert "X3 집계 대상" not in labels
    aggregation = next(control for control in app.selectbox if control.label == "집계 방식")
    assert "행 개수(결측 제외)" in aggregation.options
    assert "유효값 개수(결측 제외)" not in aggregation.options
    bubble_size = next(control for control in app.slider if control.key == "viz_1_bubble_size")
    color_mode = next(control for control in app.radio if control.key == "viz_1_scatter_bubble_chart_color_mode")
    bubble_colormap = next(control for control in app.selectbox if control.key == "viz_1_scatter_bubble_chart_colormap")
    bubble_hex = next(control for control in app.color_picker if control.key == "viz_1_scatter_bubble_chart_hex")
    assert bubble_size.value == 80.0
    assert color_mode.value == "hex"
    assert bubble_colormap.disabled
    assert not bubble_hex.disabled

    color_mode.set_value("colormap")
    app.run()
    bubble_colormap = next(control for control in app.selectbox if control.key == "viz_1_scatter_bubble_chart_colormap")
    bubble_hex = next(control for control in app.color_picker if control.key == "viz_1_scatter_bubble_chart_hex")
    assert not bubble_colormap.disabled
    assert bubble_hex.disabled

    chart_type = next(control for control in app.selectbox if control.label == "차트 유형")
    chart_type.select("Heatmap")
    app.run()
    labels = [control.label for control in app.selectbox]
    assert labels.index("컬럼1") < labels.index("컬럼2") < labels.index("집계대상") < labels.index("집계 방식")
    target = next(control for control in app.selectbox if control.label == "집계대상")
    assert target.value == "(없음)"
    aggregation = next(control for control in app.selectbox if control.label == "집계 방식")
    assert "행 개수(결측 제외)" in aggregation.options
    assert "색상 팔레트" not in labels
    assert not any(control.label == "기본 색상" for control in app.color_picker)
    assert any(control.key == "viz_1_heatmap_chart_colormap" and control.label == "컬러맵" for control in app.selectbox)
    assert any(control.key == "viz_1_heatmap_chart_hex" and control.label == "HEX" for control in app.color_picker)


def test_stage_switch_preserves_visualization_and_insight_history() -> None:
    app = _loaded_app()
    chart_record, _ = rebuild_chart_record(
        app.session_state.df_clean,
        {"chart_type": "bar", "x": "참가 경로", "aggregation": "count"},
        "sample.csv",
    )
    app.session_state["insight_history"] = [
        InsightMessage(role="model", text="차트", charts=[chart_record]).model_dump(mode="json")
    ]

    app.segmented_control[0].set_value("인사이트")
    app.run()
    assert not any("API Key" in item.label for item in app.text_input)
    assert any(button.label == "수정한 Pydantic 설정으로 차트 재실행" for button in app.button)
    editor = next(item for item in app.text_area if item.label == "ChartSpec JSON 직접 수정")
    edited_spec = json.loads(editor.value)
    edited_spec["title"] = "사용자 수정 제목"
    editor.input(json.dumps(edited_spec, ensure_ascii=False, indent=2))
    app.run()
    next(
        button for button in app.button
        if button.label == "수정한 Pydantic 설정으로 차트 재실행"
    ).click()
    app.run()
    updated_spec = app.session_state.insight_history[0]["charts"][0]["source"]["charts"][0]["spec"]
    assert updated_spec["title"] == "사용자 수정 제목"

    app.segmented_control[0].set_value("전처리")
    app.run()
    app.segmented_control[0].set_value("인사이트")
    app.run()
    assert app.session_state.insight_history
    assert not any("API Key" in item.label for item in app.text_input)

    visualization_result = build_visualization(
        app.session_state.df_clean,
        [ChartSpec(chart_type="bar", x="참가 경로", aggregation="count")],
        FigureSpec(rows=1, columns=1),
    )
    app.session_state["visualization_result"] = visualization_result
    app.segmented_control[0].set_value("전처리")
    app.run()
    app.segmented_control[0].set_value("시각화")
    app.run()
    assert not app.exception
    assert app.session_state.visualization_result is not None
    assert any("시각화 결과" in item.value for item in app.markdown)


def test_local_upload_opens_fast_basic_stage() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.file_uploader[0].upload(
        "sample.csv",
        b"name,value\nA,1\nB,2\n",
        "text/csv",
    )
    app.run()
    next(button for button in app.button if button.label == "새 업로드 파일 등록").click()
    app.run()

    next(radio for radio in app.radio if radio.label == "분석할 데이터").set_value("uploaded")
    app.run()
    next(button for button in app.button if button.label == "선택 데이터 분석").click()
    app.run()

    assert not app.exception
    assert app.segmented_control[0].value == "기본 현황"
    assert len(app.session_state.df) == 2
    assert app.session_state.df is not app.session_state.df_clean
    assert app.session_state.current_file_name == "sample.csv"


def test_language_selection_localizes_all_workflow_stages() -> None:
    app = _loaded_app()
    language = next(item for item in app.selectbox if item.label == "Language / 언어")
    language.set_value("English")
    app.run()

    assert app.session_state.ui_language == "English"
    assert "Data Analysis" in [item.value for item in app.title]
    assert "2. Data Overview" in {item.label for item in app.expander}
    assert app.segmented_control[0].label == "Stage to Display"

    app.segmented_control[0].set_value("Preprocessing")
    app.run()
    assert "Preprocessing" in [item.value for item in app.header]
    assert {button.label for button in app.download_button} == {
        "Download CSV",
        "Download Excel",
    }
    summary = next(table.value for table in app.dataframe if "Item" in table.value.columns)
    assert list(summary.columns) == [
        "Item", "Before Preprocessing", "After Preprocessing", "Change"
    ]
    assert summary["Item"].tolist()[:2] == ["Rows", "Variables"]

    # Use a fresh AppTest tree for each conditionally-rendered stage. AppTest
    # serializes formatted option labels rather than their underlying values,
    # unlike a browser session, when a widget disappears between runs.
    app = _loaded_app()
    next(item for item in app.selectbox if item.label == "Language / 언어").set_value("English")
    app.run()
    x_column = str(app.session_state.df_clean.columns[1])
    app.session_state["visualization_result"] = build_visualization(
        app.session_state.df_clean,
        [ChartSpec(chart_type="bar", x=x_column, aggregation="count")],
        FigureSpec(rows=1, columns=1),
    )
    app.segmented_control[0].set_value("Visualization")
    app.run()
    assert "Data Visualization" in [item.value for item in app.header]
    assert "Layout Settings" in {item.label for item in app.expander}
    assert "Generate Visualization" in {item.label for item in app.button}
    assert any("The statistical table contains" in item.value for item in app.markdown)
    assert any("Value" in table.value.columns for table in app.dataframe)

    app = _loaded_app()
    next(item for item in app.selectbox if item.label == "Language / 언어").set_value("English")
    app.run()
    app.segmented_control[0].set_value("Insight")
    app.run()
    assert app.session_state.ui_language == "English"
    assert "Generate Business Insight" in {item.label for item in app.button}
    assert {item.label for item in app.file_uploader} >= {
        "Select File",
        "Upload Reference Datasets",
        "Upload Existing Conversations",
    }
    insight_metrics = {item.label: str(item.value) for item in app.metric}
    assert insight_metrics["Current Data"].endswith("rows")
    assert insight_metrics["Preprocessing History"] == "0"


def test_visualization_download_buttons_survive_rerun() -> None:
    app = _loaded_app()
    x_column = str(app.session_state.df_clean.columns[1])
    app.session_state["visualization_result"] = build_visualization(
        app.session_state.df_clean,
        [ChartSpec(chart_type="bar", x=x_column, aggregation="count")],
        FigureSpec(rows=1, columns=1),
    )
    app.segmented_control[0].set_value("시각화")
    app.run()
    assert not app.exception
    assert {"PNG", "JPG", "PDF", "SVG", "Source JSON"}.issubset(
        {item.label for item in app.download_button}
    )

    app.run()
    assert not app.exception


def test_english_dynamic_preprocessing_and_axis_labels() -> None:
    app = _loaded_app()
    next(item for item in app.selectbox if item.label == "Language / 언어").set_value("English")
    app.run()
    app.session_state["preprocessing_section"] = "특정값 변경"
    app.segmented_control[0].set_value("Preprocessing")
    app.run()

    replacement_button = next(
        item for item in app.button if item.label.startswith("Apply Entered Replacements")
    )
    assert replacement_button.label == "Apply Entered Replacements (0 items)"
    unique_caption = next(item.value for item in app.caption if "Unique Value" in item.value)
    assert "descending" in unique_caption
    assert "Rows without a replacement are unchanged" in unique_caption
    assert not any("\uac00" <= character <= "\ud7a3" for character in unique_caption)

    axis_app = AppTest.from_string(
        """
import streamlit as st
from core.i18n import install_streamlit_i18n

st.session_state["ui_language"] = "English"
install_streamlit_i18n()
st.number_input("최소값", value=0.0)
st.number_input("최대값", value=1.0)
""",
        default_timeout=30,
    ).run()

    numeric_labels = {item.label for item in axis_app.number_input}
    assert {"Minimum", "Maximum"}.issubset(numeric_labels)
    assert not axis_app.exception


def test_english_visualization_controls_have_no_korean_ui_labels() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-02-01"],
            "category": ["A", "B"],
            "value": [1, 2],
        }
    )
    element_groups = (
        "caption", "markdown", "info", "warning", "success", "error",
        "button", "text_input", "checkbox", "number_input", "slider",
        "expander", "toggle", "color_picker", "date_input", "selectbox",
        "multiselect", "radio",
    )

    for chart_type in ("bar", "line", "pie", "histogram", "scatter_plot", "heatmap"):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30)
        app.session_state["df"] = frame
        app.session_state["df_clean"] = frame.copy(deep=True)
        app.session_state["source_filename"] = "sample.csv"
        app.session_state["ui_language"] = "English"
        app.session_state["analysis_stage"] = "시각화"
        app.session_state["viz_0_type"] = chart_type
        app.run()

        assert not app.exception
        for group_name in element_groups:
            for element in getattr(app, group_name):
                fields = ("label",) if group_name in {"selectbox", "multiselect", "radio"} else ("label", "value")
                for field_name in fields:
                    value = getattr(element, field_name, None)
                    if isinstance(value, str):
                        assert not any("가" <= character <= "힣" for character in value), (
                            chart_type, group_name, field_name, value
                        )
                    if group_name in {"selectbox", "multiselect", "radio"}:
                        for option in getattr(element, "options", ()):
                            if option == "한국어":
                                continue
                            assert not any("가" <= character <= "힣" for character in option), (
                            chart_type, group_name, "option", option
                        )
