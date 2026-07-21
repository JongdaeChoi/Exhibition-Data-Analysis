from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from data.loader import LoadedDataset
from data.profiler import build_basic_profile


SESSION_DEFAULTS = {
    "ui_language": "한국어",
    "df": None,
    "df_clean": None,
    "saved_test_dataframe": None,
    "saved_test_file_name": None,
    "saved_test_file_bytes": None,
    "saved_test_loaded": False,
    "uploaded_dataframe": None,
    "uploaded_file_name": None,
    "uploaded_file_bytes": None,
    "selected_data_source": None,
    "current_dataframe": None,
    "current_file_name": None,
    "source_filename": None,
    "load_source": None,
    "basic_profile": None,
    "analysis_stage": "기본 현황",
    "preprocessing_notice": None,
    "preprocessing_section": "결측값",
    "preprocessing_revision": 0,
    "preprocessing_history": [],
    "visualization_result": None,
    "visualization_last_render_signature": None,
    "visualization_sources": [],
    "visualization_notice": None,
    "insight_history": [],
    "insight_provider": "OpenAI",
    "insight_model": "gpt-5.6-sol",
    "insight_references": [],
    "insight_reference_datasets": {},
    "insight_active_dataset": "main",
    "insight_pending_attachment_ids": [],
    "insight_visible_from": 0,
    "insight_notice": None,
    "insight_error": None,
}


def initialize_session() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def reset_analysis_results() -> None:
    st.session_state.preprocessing_notice = None
    st.session_state.preprocessing_section = "결측값"
    st.session_state.preprocessing_revision = 0
    st.session_state.preprocessing_history = []
    st.session_state.visualization_result = None
    st.session_state.visualization_last_render_signature = None
    st.session_state.visualization_sources = []
    st.session_state.visualization_notice = None
    st.session_state.insight_history = []
    st.session_state.insight_references = []
    st.session_state.insight_reference_datasets = {}
    st.session_state.insight_active_dataset = "main"
    st.session_state.insight_pending_attachment_ids = []
    st.session_state.insight_visible_from = 0
    st.session_state.insight_notice = None
    st.session_state.insight_error = None
    for key in list(st.session_state):
        if key.startswith(("viz_", "visualization_pydantic_", "insight_chart_spec_editor_")):
            del st.session_state[key]


def register_uploaded_dataset(dataset: LoadedDataset, raw: bytes) -> None:
    st.session_state.uploaded_dataframe = dataset.df
    st.session_state.uploaded_file_name = dataset.filename
    st.session_state.uploaded_file_bytes = bytes(raw)
    if st.session_state.get("saved_test_dataframe") is None:
        st.session_state.selected_data_source = "uploaded"


def register_saved_dataset(dataset: LoadedDataset, raw: bytes) -> None:
    st.session_state.saved_test_dataframe = dataset.df
    st.session_state.saved_test_file_name = dataset.filename
    st.session_state.saved_test_file_bytes = bytes(raw)
    if st.session_state.get("selected_data_source") not in {"saved", "uploaded"}:
        st.session_state.selected_data_source = "saved"


def clear_saved_dataset() -> None:
    was_current = st.session_state.get("selected_data_source") == "saved"
    st.session_state.saved_test_dataframe = None
    st.session_state.saved_test_file_name = None
    st.session_state.saved_test_file_bytes = None
    if was_current:
        st.session_state.selected_data_source = (
            "uploaded" if isinstance(st.session_state.get("uploaded_dataframe"), pd.DataFrame) else None
        )
        st.session_state.current_dataframe = None
        st.session_state.current_file_name = None
        st.session_state.df = None
        st.session_state.df_clean = None
        st.session_state.source_filename = None
        st.session_state.basic_profile = None
        reset_analysis_results()


def activate_data_source(source: str) -> None:
    if source == "saved":
        frame = st.session_state.get("saved_test_dataframe")
        filename = st.session_state.get("saved_test_file_name")
    elif source == "uploaded":
        frame = st.session_state.get("uploaded_dataframe")
        filename = st.session_state.get("uploaded_file_name")
    else:
        raise ValueError("분석할 데이터 원본을 선택하세요.")
    if not isinstance(frame, pd.DataFrame) or not filename:
        raise ValueError("선택한 데이터 원본을 사용할 수 없습니다.")
    original = frame.copy(deep=True)
    st.session_state.current_dataframe = original
    st.session_state.current_file_name = str(filename)
    st.session_state.df = original
    st.session_state.df_clean = original.copy(deep=True)
    st.session_state.source_filename = str(filename)
    st.session_state.load_source = source
    st.session_state.basic_profile = build_basic_profile(original)
    st.session_state.analysis_stage = "기본 현황"
    reset_analysis_results()


def store_dataset(dataset: LoadedDataset, source: str) -> None:
    """Backward-compatible helper used by tests and non-interactive callers."""
    register_uploaded_dataset(dataset, b"")
    activate_data_source("uploaded")


def has_dataset() -> bool:
    return isinstance(st.session_state.get("df"), pd.DataFrame) and isinstance(
        st.session_state.get("df_clean"), pd.DataFrame
    )
