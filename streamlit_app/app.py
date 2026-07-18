from __future__ import annotations

import streamlit as st

from core.session import has_dataset, initialize_session, store_dataset
from data.loader import DataLoadError, download_google_drive_file, load_table
from data.profiler import build_basic_profile


st.set_page_config(page_title="데이터 분석", page_icon="📊", layout="wide")
initialize_session()


def load_and_store(raw: bytes, filename: str, source: str) -> None:
    with st.status("데이터를 적재하고 있습니다...", expanded=True) as status:
        st.write("파일 형식과 내용을 확인하고 있습니다.")
        dataset = load_table(raw, filename)
        st.write("원본 데이터셋 `df`를 생성했습니다.")
        st.write("분석용 데이터셋 `df_clean = df.copy()`를 생성했습니다.")
        store_dataset(dataset, source)
        status.update(label="데이터 적재가 완료되었습니다.", state="complete")


st.title("데이터 분석")
st.caption("CSV 또는 Excel 데이터를 불러와 원본과 분석용 데이터셋으로 안전하게 분리합니다.")

with st.container(border=True):
    st.subheader("1. 데이터 파일 선택")
    source_tab, drive_tab = st.tabs(["로컬 파일", "Google Drive"])

    with source_tab:
        uploaded_file = st.file_uploader(
            "파일 선택",
            type=["csv", "xlsx", "xls"],
            help="버튼을 누르면 운영체제의 파일 선택창이 열립니다.",
        )
        if st.button(
            "로컬 파일 적재",
            type="primary",
            disabled=uploaded_file is None,
            width="stretch",
        ):
            try:
                load_and_store(uploaded_file.getvalue(), uploaded_file.name, "local")
                st.success(f"{uploaded_file.name} 파일을 적재했습니다.")
            except DataLoadError as exc:
                st.error(str(exc))

    with drive_tab:
        st.info("현재 단계에서는 '링크가 있는 모든 사용자'에게 공개된 Drive 파일을 지원합니다.")
        drive_url = st.text_input(
            "Google Drive 공유 링크",
            placeholder="https://drive.google.com/file/d/.../view",
        )
        drive_filename = st.text_input(
            "파일명 (선택)",
            placeholder="Drive 링크에서 파일명이 확인되지 않을 때 sample.csv처럼 입력",
        )
        if st.button(
            "Drive 파일 적재",
            type="primary",
            disabled=not drive_url.strip(),
            width="stretch",
        ):
            try:
                with st.spinner("Google Drive에서 파일을 내려받고 있습니다..."):
                    raw, filename = download_google_drive_file(drive_url)
                if drive_filename.strip():
                    filename = drive_filename.strip()
                if not filename.lower().endswith((".csv", ".xlsx", ".xls")):
                    raise DataLoadError(
                        "Drive 응답에서 파일 형식을 확인하지 못했습니다. "
                        "위 파일명 입력란에 .csv, .xlsx 또는 .xls 확장자를 포함해 입력하세요."
                    )
                load_and_store(raw, filename, "google_drive")
                st.success(f"{filename} 파일을 적재했습니다.")
            except DataLoadError as exc:
                st.error(str(exc))

if has_dataset():
    df = st.session_state.df
    df_clean = st.session_state.df_clean
    if st.session_state.basic_profile is None:
        st.session_state.basic_profile = build_basic_profile(df)
    basic_profile = st.session_state.basic_profile
    st.subheader("2. 데이터 기본 현황")
    metric_columns = st.columns(4)
    metric_columns[0].metric("파일명", st.session_state.source_filename)
    metric_columns[1].metric("행", f"{len(df):,}")
    metric_columns[2].metric("변수", f"{df.shape[1]:,}")
    metric_columns[3].metric("전체 결측", f"{int(basic_profile['결측 개수'].sum()):,}")

    st.caption(
        "원본 `df`와 분석용 `df_clean`을 서로 다른 복사본으로 보관하고 있습니다. "
        "이 화면의 현황은 원본 `df` 기준입니다."
    )
    st.dataframe(basic_profile, width="stretch", hide_index=True)

    with st.expander("원본 데이터 미리보기", expanded=False):
        st.dataframe(df.head(20), width="stretch")

    st.subheader("3. 작업 단계")
    stage = st.segmented_control(
        "표시할 작업",
        ["기본 현황", "전처리", "시각화", "인사이트"],
        selection_mode="single",
        key="analysis_stage",
        help="선택한 단계만 실행하여 데이터 적재 후 화면 표시 속도를 높입니다.",
    )
    if stage == "전처리":
        # Lazy import: preprocessing code and widgets are loaded only on demand.
        from ui.preprocessing_view import render_preprocessing

        st.divider()
        render_preprocessing()
    elif stage == "시각화":
        # Lazy import: Matplotlib, Seaborn and font registration are not part of
        # the data-load/basic-profile path.
        from ui.visualization_view import render_visualization

        st.divider()
        render_visualization()
    elif stage == "인사이트":
        # Gemini SDK and report context are loaded only when Insight is selected.
        from ui.insight_view import render_insight

        st.divider()
        render_insight()
else:
    st.info("파일을 선택하고 적재하면 데이터 기본 현황이 여기에 표시됩니다.")
