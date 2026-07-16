from __future__ import annotations

import pandas as pd


PROFILE_COLUMNS = ["변수명", "데이터 개수", "데이터 타입", "결측 개수"]


def build_basic_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row per column without mutating the input frame."""
    missing = frame.isna().sum()
    row_count = len(frame)
    return pd.DataFrame(
        {
            "변수명": [str(column) for column in frame.columns],
            "데이터 개수": (row_count - missing).astype(int).tolist(),
            "데이터 타입": frame.dtypes.astype(str).tolist(),
            "결측 개수": missing.astype(int).tolist(),
        },
        columns=PROFILE_COLUMNS,
    )
