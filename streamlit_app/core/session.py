from __future__ import annotations

import pandas as pd
import streamlit as st

from data.loader import LoadedDataset
from data.profiler import build_basic_profile


SESSION_DEFAULTS = {
    "ui_language": "한국어",
    "df": None,
    "df_clean": None,
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
    "insight_api_keys": {},
    "insight_provider": "OpenAI",
    "insight_model": "gpt-5.6-sol",
    "insight_references": [],
    "insight_pending_attachment_ids": [],
    "insight_visible_from": 0,
    "insight_notice": None,
    "insight_error": None,
}


def initialize_session() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def store_dataset(dataset: LoadedDataset, source: str) -> None:
    # `load_table` already created two independent frames. Transfer ownership
    # to session state without repeating expensive deep copies.
    st.session_state.df = dataset.df
    st.session_state.df_clean = dataset.df_clean
    st.session_state.source_filename = dataset.filename
    st.session_state.load_source = source
    st.session_state.basic_profile = build_basic_profile(st.session_state.df)
    st.session_state.analysis_stage = "기본 현황"
    st.session_state.preprocessing_notice = None
    st.session_state.preprocessing_section = "결측값"
    st.session_state.preprocessing_revision = 0
    st.session_state.preprocessing_history = []
    st.session_state.visualization_result = None
    st.session_state.visualization_last_render_signature = None
    st.session_state.visualization_sources = []
    st.session_state.visualization_notice = None
    st.session_state.insight_history = []
    st.session_state.insight_provider = "OpenAI"
    st.session_state.insight_model = "gpt-5.6-sol"
    st.session_state.insight_references = []
    st.session_state.insight_pending_attachment_ids = []
    st.session_state.insight_visible_from = 0
    st.session_state.insight_notice = None
    st.session_state.insight_error = None


def has_dataset() -> bool:
    return isinstance(st.session_state.get("df"), pd.DataFrame) and isinstance(
        st.session_state.get("df_clean"), pd.DataFrame
    )
