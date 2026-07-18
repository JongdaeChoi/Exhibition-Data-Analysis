from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _loaded_app() -> AppTest:
    frame = pd.DataFrame(
        {
            "조사일자": ["2025-01-01", "2025-02-01"],
            "참가 경로": ["검색", "추천"],
            "값": [10, 20],
        }
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["df"] = frame
    app.session_state["df_clean"] = frame.copy(deep=True)
    app.session_state["source_filename"] = "sample.csv"
    return app.run()


def test_data_load_defaults_to_basic_profile_only() -> None:
    app = _loaded_app()

    assert not app.exception
    assert app.segmented_control[0].value == "기본 현황"
    assert "전처리" not in [header.value for header in app.header]
    assert "데이터 시각화" not in [header.value for header in app.header]
    assert not app.download_button


def test_heavy_sections_render_only_when_selected() -> None:
    app = _loaded_app()

    app.segmented_control[0].set_value("전처리")
    app.run()
    assert not app.exception
    assert "전처리" in [header.value for header in app.header]
    assert "데이터 시각화" not in [header.value for header in app.header]
    assert {button.label for button in app.download_button} == {"CSV 다운로드", "Excel 다운로드"}

    app.segmented_control[0].set_value("시각화")
    app.run()
    assert not app.exception
    assert "전처리" not in [header.value for header in app.header]
    assert "데이터 시각화" in [header.value for header in app.header]

    app.segmented_control[0].set_value("인사이트")
    app.run()
    assert not app.exception
    assert "Business Insight" in [header.value for header in app.header]


def test_local_upload_opens_fast_basic_stage() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.file_uploader[0].upload(
        "sample.csv",
        b"name,value\nA,1\nB,2\n",
        "text/csv",
    )
    app.run()
    next(button for button in app.button if button.label == "로컬 파일 적재").click()
    app.run()

    assert not app.exception
    assert app.segmented_control[0].value == "기본 현황"
    assert len(app.session_state.df) == 2
    assert app.session_state.df is not app.session_state.df_clean
    assert "sample.csv 파일을 적재했습니다." in [message.value for message in app.success]
