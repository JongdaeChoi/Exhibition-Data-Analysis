from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import supabase_storage_config
from core.i18n import LANGUAGE_OPTIONS, install_streamlit_i18n, localized_columns, translate
from core.session import (
    activate_data_source,
    clear_saved_dataset,
    has_dataset,
    initialize_session,
    register_saved_dataset,
    register_uploaded_dataset,
)
from data.loader import DataLoadError, download_google_drive_file, load_table
from data.profiler import build_basic_profile
from data.storage import PersistentStorageError, SupabaseDefaultFileStore


st.set_page_config(page_title="Data Analysis | 데이터 분석", page_icon="📊", layout="wide")
initialize_session()
install_streamlit_i18n()


def _storage_backend() -> SupabaseDefaultFileStore | None:
    config = supabase_storage_config()
    return SupabaseDefaultFileStore(config) if config is not None else None


def _dataset_label(frame: pd.DataFrame | None, filename: str | None) -> str:
    if not isinstance(frame, pd.DataFrame) or not filename:
        return translate("없음")
    return f"{filename} · {len(frame):,} × {frame.shape[1]:,}"


def _register_new_file(raw: bytes, filename: str, source: str) -> None:
    with st.status(translate("데이터를 적재하고 있습니다..."), expanded=True) as status:
        st.write(translate("파일 형식과 내용을 확인하고 있습니다."))
        dataset = load_table(raw, filename)
        register_uploaded_dataset(dataset, raw)
        status.update(
            label=translate("새 업로드 파일 등록이 완료되었습니다."),
            state="complete",
        )
    st.session_state.uploaded_file_source = source


def _autoload_saved_file(storage: SupabaseDefaultFileStore | None) -> None:
    if st.session_state.get("saved_test_loaded"):
        return
    st.session_state.saved_test_loaded = True
    if storage is None:
        return
    try:
        stored = storage.load()
        if stored is None:
            return
        dataset = load_table(stored.raw, stored.filename)
        register_saved_dataset(dataset, stored.raw)
        if st.session_state.get("current_dataframe") is None:
            activate_data_source("saved")
    except (PersistentStorageError, DataLoadError) as exc:
        st.session_state.storage_error = str(exc)


language_space, language_control = st.columns([8, 2])
with language_control:
    st.selectbox(
        "Language / 언어",
        LANGUAGE_OPTIONS,
        key="ui_language",
        help="화면에 표시할 언어를 선택합니다. / Select the interface language.",
    )

# Streamlit removes widget state when a conditionally rendered stage disappears.
for state_key in list(st.session_state):
    if state_key.startswith(("viz_", "visualization_")) and not state_key.startswith(
        "viz_download_"
    ):
        st.session_state[state_key] = st.session_state[state_key]

storage = _storage_backend()
_autoload_saved_file(storage)

st.title("데이터 분석")
st.caption("저장된 기본 테스트 파일 또는 새 업로드 파일 하나를 선택하여 분석합니다.")

with st.container(border=True):
    st.subheader("1. 데이터 파일과 기본 테스트 파일")
    source_tab, drive_tab = st.tabs(["로컬 파일", "Google Drive"])

    with source_tab:
        uploaded_file = st.file_uploader(
            "파일 선택",
            type=["csv", "xlsx", "xls"],
            help="새 파일은 현재 세션에만 등록되며 저장 버튼을 눌러야 기본 파일로 교체됩니다.",
            key="analysis_file_upload",
        )
        if st.button(
            "새 업로드 파일 등록",
            type="primary",
            disabled=uploaded_file is None,
            width="stretch",
        ):
            try:
                _register_new_file(uploaded_file.getvalue(), uploaded_file.name, "local")
                st.success(f"{uploaded_file.name} 파일을 새 업로드 데이터로 등록했습니다.")
            except DataLoadError as exc:
                st.error(str(exc))

    with drive_tab:
        st.info("'링크가 있는 모든 사용자'에게 공개된 Google Drive 파일만 지원합니다.")
        drive_url = st.text_input(
            "Google Drive 공유 링크",
            placeholder="https://drive.google.com/file/d/.../view",
        )
        drive_filename = st.text_input(
            "파일명 (선택)",
            placeholder="Drive 링크에서 확인되지 않을 때 sample.csv처럼 입력",
        )
        if st.button(
            "Drive 파일을 새 업로드로 등록",
            type="primary",
            disabled=not drive_url.strip(),
            width="stretch",
        ):
            try:
                with st.spinner("Google Drive에서 파일을 내려받고 있습니다..."):
                    raw, filename = download_google_drive_file(drive_url)
                filename = drive_filename.strip() or filename
                _register_new_file(raw, filename, "google_drive")
                st.success(f"{filename} 파일을 새 업로드 데이터로 등록했습니다.")
            except DataLoadError as exc:
                st.error(str(exc))

    st.markdown("#### 기본 테스트 파일 영구 저장")
    if storage is None:
        st.info(
            "Supabase Storage가 설정되지 않았습니다. 데이터 적재·전처리·시각화는 계속 사용할 수 있습니다."
        )
    else:
        st.caption("기본 테스트 파일은 개발자 소유의 비공개 Supabase Storage에 저장됩니다.")

    saved_frame = st.session_state.get("saved_test_dataframe")
    uploaded_frame = st.session_state.get("uploaded_dataframe")
    status_columns = st.columns(2)
    status_columns[0].metric(
        "저장된 기본 테스트 파일",
        _dataset_label(saved_frame, st.session_state.get("saved_test_file_name")),
    )
    status_columns[1].metric(
        "이번에 새로 업로드한 파일",
        _dataset_label(uploaded_frame, st.session_state.get("uploaded_file_name")),
    )

    save_col, delete_col = st.columns(2)
    if save_col.button(
        "새 업로드 파일을 기본 테스트 파일로 저장/교체",
        disabled=storage is None or not isinstance(uploaded_frame, pd.DataFrame),
        type="primary",
        width="stretch",
    ):
        try:
            stored = storage.save(
                filename=st.session_state.uploaded_file_name,
                raw=st.session_state.uploaded_file_bytes,
                rows=len(uploaded_frame),
                columns=uploaded_frame.shape[1],
            )
            dataset = load_table(stored.raw, stored.filename)
            register_saved_dataset(dataset, stored.raw)
            st.success("기본 테스트 파일을 Supabase Storage에 저장·교체했습니다.")
            st.rerun()
        except (PersistentStorageError, DataLoadError) as exc:
            st.error(str(exc))
    if delete_col.button(
        "저장된 기본 테스트 파일 삭제",
        disabled=storage is None or not isinstance(saved_frame, pd.DataFrame),
        width="stretch",
    ):
        try:
            storage.delete()
            clear_saved_dataset()
            st.success("저장된 기본 테스트 파일을 삭제했습니다.")
            st.rerun()
        except PersistentStorageError as exc:
            st.error(str(exc))

    storage_error = st.session_state.pop("storage_error", None)
    if storage_error:
        st.error(storage_error)

    options = []
    if isinstance(saved_frame, pd.DataFrame):
        options.append("saved")
    if isinstance(uploaded_frame, pd.DataFrame):
        options.append("uploaded")
    if options:
        current_choice = st.session_state.get("selected_data_source")
        if current_choice not in options:
            st.session_state.selected_data_source = "saved" if "saved" in options else options[0]
        labels = {
            "saved": "저장된 기본 테스트 파일",
            "uploaded": "이번에 새로 업로드한 파일",
        }
        st.radio(
            "분석할 데이터",
            options,
            format_func=lambda value: translate(labels[value]),
            key="selected_data_source",
            horizontal=True,
        )
        if st.button("선택 데이터 분석", type="primary", width="stretch"):
            try:
                activate_data_source(st.session_state.selected_data_source)
                st.success("선택한 데이터로 분석 상태를 초기화했습니다.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.caption(
        f"현재 분석 중인 파일: {st.session_state.get('current_file_name') or translate('선택되지 않음')}"
    )

if has_dataset():
    df = st.session_state.df
    if st.session_state.basic_profile is None:
        st.session_state.basic_profile = build_basic_profile(df)
    basic_profile = st.session_state.basic_profile
    with st.expander("2. 데이터 기본 현황", expanded=False):
        metric_columns = st.columns(4)
        metric_columns[0].metric("파일명", st.session_state.current_file_name)
        metric_columns[1].metric("행", f"{len(df):,}")
        metric_columns[2].metric("변수", f"{df.shape[1]:,}")
        metric_columns[3].metric("전체 결측", f"{int(basic_profile['결측 개수'].sum()):,}")
        st.caption(
            "원본 `df`와 분석용 `df_clean`을 서로 다른 복사본으로 보관하고 있습니다. "
            "이 화면의 현황은 원본 `df` 기준입니다."
        )
        st.dataframe(localized_columns(basic_profile), width="stretch", hide_index=True)
        st.markdown("#### 원본 데이터 미리보기")
        st.dataframe(df.head(20), width="stretch")

    st.subheader("3. 작업 단계")
    stage = st.segmented_control(
        "표시할 작업",
        ["기본 현황", "전처리", "시각화", "인사이트"],
        selection_mode="single",
        key="analysis_stage",
        help="선택한 단계만 실행하여 화면 표시 속도를 높입니다.",
    )
    previous_stage = st.session_state.get("_last_analysis_stage")
    stage_changed = previous_stage is not None and previous_stage != stage
    st.session_state._last_analysis_stage = stage
    if stage == "전처리":
        from ui.preprocessing_view import render_preprocessing

        st.divider()
        render_preprocessing()
    elif stage == "시각화":
        from ui.visualization_view import render_visualization

        st.divider()
        render_visualization(preserve_existing_result=stage_changed)
    elif stage == "인사이트":
        from ui.insight_view import render_insight

        st.divider()
        render_insight()
else:
    st.info("파일을 등록하고 분석할 데이터 원본을 선택하세요.")
