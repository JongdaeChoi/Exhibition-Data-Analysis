from __future__ import annotations

import pandas as pd
import streamlit as st

from data.loader import LoadedDataset


SESSION_DEFAULTS = {
    "df": None,
    "df_clean": None,
    "source_filename": None,
    "load_source": None,
    "preprocessing_notice": None,
}


def initialize_session() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def store_dataset(dataset: LoadedDataset, source: str) -> None:
    # Keep two independent objects: df is read-only source, df_clean is the workspace.
    st.session_state.df = dataset.df.copy(deep=True)
    st.session_state.df_clean = dataset.df_clean.copy(deep=True)
    st.session_state.source_filename = dataset.filename
    st.session_state.load_source = source
    st.session_state.preprocessing_notice = None


def has_dataset() -> bool:
    return isinstance(st.session_state.get("df"), pd.DataFrame) and isinstance(
        st.session_state.get("df_clean"), pd.DataFrame
    )
