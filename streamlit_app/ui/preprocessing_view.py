from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from data.preprocessing import (
    DATE_COMPONENT_LABELS,
    PreprocessingError,
    comparison_summary,
    date_column_candidates,
    fill_missing_values,
    missing_value_summary,
    noise_candidates,
    paginate,
    replace_selected_value,
    split_date_components,
    to_csv_bytes,
    to_excel_bytes,
    unique_value_counts,
)


def _record_result(result, operation: str) -> None:
    st.session_state.df_clean = result.frame
    st.session_state.preprocessing_history.append(
        {
            "시간": dt.datetime.now().strftime("%H:%M:%S"),
            "작업": operation,
            "영향 행": result.affected_rows,
            "결과": result.message,
        }
    )
    st.success(result.message)


def _value_selector(table: pd.DataFrame, label: str, key: str, value_column: str):
    records = table.to_dict(orient="records")
    if not records:
        return None
    options = list(range(len(records)))
    selected_index = st.selectbox(
        label,
        options,
        format_func=lambda i: f"{records[i].get('표시값', records[i][value_column])} ({records[i]['개수']:,}건)",
        key=key,
    )
    return records[selected_index][value_column]


def _render_missing(frame: pd.DataFrame) -> None:
    st.dataframe(missing_value_summary(frame), width="stretch", hide_index=True)
    columns_with_missing = [column for column in frame.columns if frame[column].isna().any()]
    if not columns_with_missing:
        st.info("현재 df_clean에는 결측값이 없습니다.")
        return
    column = st.selectbox("변수", columns_with_missing, key="missing_column")
    methods = ["특정값", "해당 행 삭제"]
    if pd.api.types.is_numeric_dtype(frame[column]):
        methods[1:1] = ["평균값", "중앙값"]
    method = st.selectbox("처리방법", methods, key="missing_method")
    value = st.text_input("기입할 특정값", key="missing_value") if method == "특정값" else None
    if st.button("결측값 처리 실행", type="primary", key="run_missing"):
        try:
            _record_result(fill_missing_values(frame, column, method, value), f"결측값 · {column} · {method}")
        except (PreprocessingError, ValueError) as exc:
            st.error(str(exc))


def _render_noise(frame: pd.DataFrame) -> None:
    column = st.selectbox("변수", frame.columns, key="noise_column")
    candidates = noise_candidates(frame, column)
    if candidates.empty:
        st.info("공백·표기 충돌·저빈도 규칙으로 탐지된 노이즈 후보가 없습니다.")
        return
    page_count = max(1, (len(candidates) + 19) // 20)
    page = st.number_input("조회 페이지", 1, page_count, 1, key="noise_page")
    page_table, total_pages = paginate(candidates, page)
    st.caption(f"총 {len(candidates):,}개 후보 · {int(page)}/{total_pages} 페이지 · 페이지당 20개")
    st.dataframe(page_table, width="stretch", hide_index=True)
    selected = _value_selector(page_table, "처리할 노이즈", "noise_selected", "원본 값")
    replacement = st.text_input("변경할 특정값", key="noise_replacement")
    if st.button("노이즈 처리 실행", type="primary", key="run_noise"):
        try:
            _record_result(
                replace_selected_value(frame, column, selected, replacement),
                f"노이즈 · {column} · {selected!r} 변경",
            )
        except (PreprocessingError, ValueError) as exc:
            st.error(str(exc))


def _render_replace(frame: pd.DataFrame) -> None:
    column = st.selectbox("변수", frame.columns, key="replace_column")
    values = unique_value_counts(frame, column)
    page_count = max(1, (len(values) + 19) // 20)
    page = st.number_input("조회 페이지", 1, page_count, 1, key="replace_page")
    page_table, total_pages = paginate(values, page)
    st.caption(f"Unique Value {len(values):,}개 · {int(page)}/{total_pages} 페이지 · 페이지당 20개")
    st.dataframe(page_table[["표시값", "개수"]], width="stretch", hide_index=True)
    selected = _value_selector(page_table, "변경할 값", "replace_selected", "값")
    replacement = st.text_input("새로운 특정값", key="replace_value")
    if st.button("특정값 변경 실행", type="primary", key="run_replace"):
        try:
            _record_result(
                replace_selected_value(frame, column, selected, replacement),
                f"특정값 · {column} · {selected!r} 변경",
            )
        except (PreprocessingError, ValueError) as exc:
            st.error(str(exc))


def _render_date(frame: pd.DataFrame) -> None:
    candidates = date_column_candidates(frame)
    st.dataframe(candidates, width="stretch", hide_index=True)
    if candidates.empty:
        st.info("값의 80% 이상을 날짜로 변환할 수 있는 변수가 없습니다.")
        return
    column = st.selectbox("날짜 변수", candidates["변수명"].tolist(), key="date_column")
    components = st.multiselect(
        "분리할 요소",
        list(DATE_COMPONENT_LABELS),
        format_func=lambda item: DATE_COMPONENT_LABELS[item],
        key="date_components",
    )
    if st.button("날짜 변수 분리 실행", type="primary", key="run_date"):
        try:
            _record_result(split_date_components(frame, column, components), f"날짜 분리 · {column}")
        except (PreprocessingError, ValueError) as exc:
            st.error(str(exc))


def render_preprocessing() -> None:
    st.header("전처리")
    st.caption("모든 작업은 분석용 `df_clean`에만 적용됩니다. 원본 `df`는 변경되지 않습니다.")
    frame = st.session_state.df_clean
    missing_tab, noise_tab, replace_tab, date_tab = st.tabs(
        ["결측값", "노이즈", "특정값 변경", "날짜 변수 분리"]
    )
    with missing_tab:
        _render_missing(frame)
    with noise_tab:
        _render_noise(frame)
    with replace_tab:
        _render_replace(frame)
    with date_tab:
        _render_date(frame)

    st.subheader("전처리 결과 요약")
    st.dataframe(
        comparison_summary(st.session_state.df, st.session_state.df_clean),
        width="stretch",
        hide_index=True,
    )
    history = st.session_state.preprocessing_history
    if history:
        with st.expander("전처리 작업 이력", expanded=True):
            st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
    if st.button("전처리 전체 초기화", key="reset_preprocessing"):
        st.session_state.df_clean = st.session_state.df.copy(deep=True)
        st.session_state.preprocessing_history = []
        st.success("df_clean을 원본 상태로 초기화했습니다.")
        st.rerun()

    st.subheader("최종 전처리 데이터 다운로드")
    filename_stem = (st.session_state.source_filename or "data").rsplit(".", 1)[0]
    csv_col, excel_col = st.columns(2)
    csv_col.download_button(
        "CSV 다운로드",
        data=to_csv_bytes(st.session_state.df_clean),
        file_name=f"{filename_stem}_clean.csv",
        mime="text/csv",
        width="stretch",
    )
    excel_col.download_button(
        "Excel 다운로드",
        data=to_excel_bytes(st.session_state.df_clean),
        file_name=f"{filename_stem}_clean.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
