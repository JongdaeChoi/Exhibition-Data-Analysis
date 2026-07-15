import pandas as pd

from data.profiler import build_basic_profile


def test_basic_profile_reports_required_fields_without_mutating_frame():
    frame = pd.DataFrame({"name": ["A", None], "value": [1, 2]})
    before = frame.copy(deep=True)

    profile = build_basic_profile(frame)

    assert profile.columns.tolist() == ["변수명", "데이터 개수", "데이터 타입", "결측 개수"]
    assert profile.loc[profile["변수명"] == "name", "데이터 개수"].item() == 1
    assert profile.loc[profile["변수명"] == "name", "결측 개수"].item() == 1
    assert frame.equals(before)
