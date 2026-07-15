import io

import pandas as pd
import pytest

from data.loader import DataLoadError, extract_google_drive_file_id, load_table


def test_load_csv_creates_independent_source_and_clean_frames():
    dataset = load_table("name,value\nA,1\nB,\n".encode(), "sample.csv")

    assert dataset.df.equals(dataset.df_clean)
    assert dataset.df is not dataset.df_clean
    dataset.df_clean.loc[0, "value"] = 99
    assert dataset.df.loc[0, "value"] == 1


def test_load_xlsx():
    buffer = io.BytesIO()
    pd.DataFrame({"name": ["A"]}).to_excel(buffer, index=False)

    dataset = load_table(buffer.getvalue(), "sample.xlsx")

    assert dataset.df.to_dict(orient="records") == [{"name": "A"}]


def test_rejects_unsupported_file_type():
    with pytest.raises(DataLoadError):
        load_table(b"text", "sample.txt")


def test_extract_google_drive_file_id():
    assert (
        extract_google_drive_file_id("https://drive.google.com/file/d/abc_123-XYZ/view")
        == "abc_123-XYZ"
    )


def test_rejects_non_google_drive_host():
    with pytest.raises(DataLoadError):
        extract_google_drive_file_id("https://example.com/file/d/abc/view")
