from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from core.i18n import current_language, localized_columns, localized_table, option_label, translate
from data.preprocessing import (
    DATE_COMPONENT_LABELS,
    PreprocessingError,
    apply_missing_plan,
    comparison_summary,
    date_column_candidates,
    drop_columns,
    missing_operations_from_editor,
    missing_value_summary,
    paginate,
    replace_multiple_values,
    split_date_components,
    to_csv_bytes,
    to_excel_bytes,
    unique_value_counts,
)


SECTIONS = ["결측값", "특정값 변경", "날짜 변수 분리", "Column 삭제"]
MISSING_METHODS = ["처리 안 함", "특정값", "평균값", "중앙값", "해당 행 삭제"]


def _apply_result(result) -> None:
    before = st.session_state.df_clean
    st.session_state.df_clean = result.frame
    st.session_state.preprocessing_notice = result.message
    st.session_state.preprocessing_revision = int(st.session_state.get("preprocessing_revision", 0)) + 1
    history = list(st.session_state.get("preprocessing_history", []))
    history.append(
        {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "message": result.message,
            "affected_rows": int(result.affected_rows),
            "before_shape": list(before.shape),
            "after_shape": list(result.frame.shape),
        }
    )
    st.session_state.preprocessing_history = history[-100:]
    st.rerun()


def _render_missing(frame: pd.DataFrame) -> None:
    st.markdown("#### 결측값 일괄 처리")
    st.caption("처리할 컬럼의 처리방법을 선택하고, ‘특정값’인 경우 같은 행의 처리값을 입력하세요.")
    table = missing_value_summary(frame)
    table["처리방법"] = option_label("처리 안 함")
    table["처리값"] = ""
    revision = st.session_state.get("preprocessing_revision", 0)
    edited = st.data_editor(
        table,
        width="stretch",
        hide_index=True,
        disabled=["변수명", "결측 개수", "결측률(%)"],
        column_config={
            "변수명": st.column_config.TextColumn(translate("변수명")),
            "결측 개수": st.column_config.NumberColumn(translate("결측 개수"), format="%d"),
            "결측률(%)": st.column_config.NumberColumn(translate("결측률(%)"), format="%.2f%%"),
            "처리방법": st.column_config.SelectboxColumn(
                translate("처리방법"),
                options=[option_label(item) for item in MISSING_METHODS],
                required=True,
            ),
            "처리값": st.column_config.TextColumn(
                translate("처리값"), help=translate("처리방법이 ‘특정값’일 때 입력합니다.")
            ),
        },
        key=f"missing_editor_{revision}",
    )
    if current_language() == "English":
        method_lookup = {option_label(item): item for item in MISSING_METHODS}
        edited["처리방법"] = edited["처리방법"].map(
            lambda value: method_lookup.get(value, value)
        )
    operations = missing_operations_from_editor(edited)
    if st.button(
        f"선택한 결측값 처리 ({len(operations):,}개 컬럼)",
        type="primary",
        key="run_missing_plan",
        disabled=not operations,
        width="stretch",
    ):
        try:
            _apply_result(apply_missing_plan(frame, operations))
        except (PreprocessingError, ValueError, TypeError) as exc:
            st.error(str(exc))


def _render_replace(frame: pd.DataFrame) -> None:
    st.markdown("#### Unique Value 일괄 변경")
    column = st.selectbox("변수", frame.columns, key="replace_column")
    values = unique_value_counts(frame, column)
    page_count = max(1, (len(values) + 19) // 20)
    page = st.selectbox(
        "Unique Value 조회 페이지",
        options=list(range(1, page_count + 1)),
        key="replace_page",
    )
    page_table, total_pages = paginate(values, page)
    page_table = page_table.reset_index(drop=True)
    editor_table = page_table[["표시값", "데이터 타입", "개수"]].copy()
    editor_table["처리값"] = ""
    st.caption(
        f"Unique Value 내림차순 · 총 {len(values):,}개 · {int(page)}/{total_pages} 페이지 · "
        "페이지당 20개 · 입력하지 않은 행은 변경하지 않음"
    )
    revision = st.session_state.get("preprocessing_revision", 0)
    edited = st.data_editor(
        editor_table,
        width="stretch",
        hide_index=True,
        disabled=["표시값", "데이터 타입", "개수"],
        column_config={
            "표시값": st.column_config.TextColumn("Unique Value"),
            "데이터 타입": st.column_config.TextColumn(
                translate("원본 데이터 타입"),
                help=translate("같게 표시되는 값에 여러 타입이 있으면 함께 표시됩니다."),
            ),
            "개수": st.column_config.NumberColumn(translate("개수"), format="%d"),
            "처리값": st.column_config.TextColumn(
                translate("처리값"), help=translate("이 값으로 변경할 행에만 입력하세요.")
            ),
        },
        key=f"replace_editor_{column}_{page}_{revision}",
    )
    replacements = [
        (page_table.iloc[index]["값"], replacement)
        for index, replacement in enumerate(edited["처리값"].tolist())
        if replacement is not None and str(replacement).strip()
    ]
    if st.button(
        f"입력한 특정값 변경 ({len(replacements):,}개)",
        type="primary",
        key="run_replace_plan",
        disabled=not replacements,
        width="stretch",
    ):
        try:
            _apply_result(replace_multiple_values(frame, column, replacements))
        except (PreprocessingError, ValueError, TypeError) as exc:
            st.error(str(exc))


def _render_date(frame: pd.DataFrame) -> None:
    st.markdown("#### 날짜 파생변수 생성")
    candidates = date_column_candidates(frame)
    st.dataframe(localized_columns(candidates), width="stretch", hide_index=True)
    if candidates.empty:
        st.info("값의 80% 이상을 날짜로 변환할 수 있는 변수가 없습니다.")
        return
    column = st.selectbox("날짜 변수", candidates["변수명"].tolist(), key="date_column")
    components = st.multiselect(
        "분리할 요소",
        list(DATE_COMPONENT_LABELS),
        format_func=lambda item: DATE_COMPONENT_LABELS[item],
        key="date_components",
        help="년·월·일, 월·일, 일, 시간을 필요한 만큼 선택할 수 있습니다.",
    )
    if st.button(
        "날짜 변수 분리 실행",
        type="primary",
        key="run_date",
        disabled=not components,
        width="stretch",
    ):
        try:
            _apply_result(split_date_components(frame, column, components))
        except (PreprocessingError, ValueError) as exc:
            st.error(str(exc))


def _render_drop_columns(frame: pd.DataFrame) -> None:
    st.markdown("#### Column 삭제")
    st.warning("선택한 Column은 분석용 `df_clean`에서만 삭제됩니다. 원본 `df`에는 남아 있습니다.")
    selected = st.multiselect("삭제할 Column", list(frame.columns), key="drop_columns")
    if selected:
        remaining = frame.shape[1] - len(selected)
        metric_columns = st.columns(3)
        metric_columns[0].metric("현재 Column", f"{frame.shape[1]:,}")
        metric_columns[1].metric("삭제 예정", f"{len(selected):,}")
        metric_columns[2].metric("삭제 후", f"{remaining:,}")
    if st.button(
        "선택한 Column 삭제",
        type="primary",
        key="run_drop_columns",
        disabled=not selected,
        width="stretch",
    ):
        try:
            _apply_result(drop_columns(frame, selected))
        except PreprocessingError as exc:
            st.error(str(exc))


def render_preprocessing() -> None:
    st.header("전처리")
    st.caption("모든 작업은 분석용 `df_clean`에만 적용됩니다. 원본 `df`는 변경되지 않습니다.")
    notice = st.session_state.pop("preprocessing_notice", None)
    if notice:
        st.success(notice)
    frame = st.session_state.df_clean
    if st.session_state.get("preprocessing_section") not in SECTIONS:
        st.session_state.preprocessing_section = "결측값"
    section = st.radio(
        "전처리 작업 선택",
        SECTIONS,
        horizontal=True,
        key="preprocessing_section",
        help="처리 후에도 현재 작업 화면을 유지하고 조회 테이블을 최신 상태로 갱신합니다.",
    )
    st.divider()
    if section == "결측값":
        _render_missing(frame)
    elif section == "특정값 변경":
        _render_replace(frame)
    elif section == "날짜 변수 분리":
        _render_date(frame)
    else:
        _render_drop_columns(frame)

    st.subheader("전처리 결과 요약")
    st.dataframe(
        localized_table(
            comparison_summary(st.session_state.df, st.session_state.df_clean),
            value_columns=("항목",),
        ),
        width="stretch",
        hide_index=True,
    )
    if st.button("전처리 전체 초기화", key="reset_preprocessing"):
        before_shape = list(st.session_state.df_clean.shape)
        st.session_state.df_clean = st.session_state.df.copy(deep=True)
        st.session_state.preprocessing_revision = int(st.session_state.get("preprocessing_revision", 0)) + 1
        st.session_state.preprocessing_notice = "df_clean을 원본 상태로 초기화했습니다."
        history = list(st.session_state.get("preprocessing_history", []))
        history.append(
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "message": "df_clean을 원본 상태로 초기화했습니다.",
                "affected_rows": 0,
                "before_shape": before_shape,
                "after_shape": list(st.session_state.df_clean.shape),
            }
        )
        st.session_state.preprocessing_history = history[-100:]
        st.rerun()

    st.subheader("최종 전처리 데이터 다운로드")
    filename_stem = (st.session_state.source_filename or "data").rsplit(".", 1)[0]
    csv_col, excel_col = st.columns(2)
    csv_col.download_button(
        "CSV 다운로드",
        data=lambda frame=st.session_state.df_clean: to_csv_bytes(frame),
        file_name=f"{filename_stem}_clean.csv",
        mime="text/csv",
        width="stretch",
        on_click="ignore",
    )
    excel_col.download_button(
        "Excel 다운로드",
        data=lambda frame=st.session_state.df_clean: to_excel_bytes(frame),
        file_name=f"{filename_stem}_clean.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        on_click="ignore",
    )
