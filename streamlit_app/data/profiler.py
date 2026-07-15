from __future__ import annotations

import pandas as pd


PROFILE_COLUMNS = ["변수명", "데이터 개수", "데이터 타입", "결측 개수"]


def build_basic_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row per column without mutating the input frame."""
    return pd.DataFrame(
        {
            "변수명": [str(column) for column in frame.columns],
            "데이터 개수": [int(frame[column].count()) for column in frame.columns],
            "데이터 타입": [str(frame[column].dtype) for column in frame.columns],
            "결측 개수": [int(frame[column].isna().sum()) for column in frame.columns],
        },
        columns=PROFILE_COLUMNS,
    )
