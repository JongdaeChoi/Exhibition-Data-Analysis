from __future__ import annotations

import base64
import os

import streamlit as st

from insight.context import build_evidence_context
from insight.models import MODEL_OPTIONS, InsightMessage
from insight.service import (
    InsightAPIError,
    execute_request,
    history_markdown_bytes,
    history_payload_bytes,
    restore_history,
)


def _configured_api_key(entered: str) -> tuple[str | None, str]:
    if entered.strip():
        return entered.strip(), "화면 입력"
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name)
        if value:
            return value, f"환경변수 {name}"
    try:
        for name in ("GEMINI_API_KEY", "exhibition"):
            value = st.secrets.get(name)
            if value:
                return str(value), f"Streamlit secrets {name}"
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return None, "미설정"


def _render_history(history: list[dict]) -> None:
    for message in history:
        if message.get("hidden"):
            continue
        role = "user" if message.get("role") == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message.get("text", ""))
            for chart in message.get("charts", []):
                try:
                    image = base64.b64decode(chart.get("image_base64", ""))
                except (ValueError, TypeError):
                    continue
                if image:
                    st.image(image, use_container_width=True)
                source = chart.get("source", {})
                with st.expander("차트 통계자료와 Pydantic 설정", expanded=False):
                    charts = source.get("charts", []) if isinstance(source, dict) else []
                    for item in charts:
                        st.json(item.get("spec", {}), expanded=False)
                        st.dataframe(item.get("statistics", []), width="stretch", hide_index=True)


def _restore_uploaded_history(uploaded) -> None:
    history, saved_model = restore_history(uploaded.getvalue(), uploaded.name)
    st.session_state.insight_history = history
    if saved_model in MODEL_OPTIONS:
        st.session_state.insight_model = saved_model
    st.session_state.insight_notice = f"기존 인사이트 {len(history):,}개 메시지를 등록했습니다."
    st.session_state.insight_error = None
    st.rerun()


def render_insight() -> None:
    st.header("Business Insight")
    st.caption(
        "Gemini가 현재 df_clean, 전처리 이력, 기술통계, 저장된 시각화 통계자료와 대화를 근거로 답합니다."
    )
    frame = st.session_state.df_clean
    history = list(st.session_state.get("insight_history", []))

    with st.container(border=True):
        model_col, key_col = st.columns([1, 2], gap="small")
        model = model_col.selectbox(
            "Gemini 모델", MODEL_OPTIONS, key="insight_model",
            help="모델을 변경해도 현재 대화 이력은 유지됩니다.",
        )
        entered_key = key_col.text_input(
            "Gemini API Key (선택)", type="password", key="insight_api_key_input",
            help="Colab 테스트 노트북이 exhibition Secret을 GEMINI_API_KEY로 전달하거나, 직접 입력할 수 있습니다.",
        )
        api_key, key_source = _configured_api_key(entered_key)
        if api_key:
            st.success(f"API 인증 준비 완료 · {key_source}")
        else:
            st.warning("`GEMINI_API_KEY`를 설정하거나 Gemini API Key를 입력하세요.")

        upload_col, clear_col = st.columns([3, 1], gap="small")
        uploaded = upload_col.file_uploader(
            "기존 인사이트 업로드", type=["json", "md", "txt"], key="insight_history_upload",
            help="이 앱에서 저장한 JSON은 대화와 차트를 복원하며, MD/TXT는 기존 분석가 답변으로 등록합니다.",
        )
        if upload_col.button(
            "업로드한 인사이트 등록", disabled=uploaded is None,
            key="restore_insight_history", width="stretch",
        ):
            try:
                _restore_uploaded_history(uploaded)
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
        if clear_col.button("새 대화", key="new_insight_chat", width="stretch"):
            st.session_state.insight_history = []
            st.session_state.insight_error = None
            st.session_state.insight_notice = "새로운 Insight 대화를 시작했습니다."
            st.rerun()

    evidence_columns = st.columns(4)
    evidence_columns[0].metric("현재 데이터", f"{len(frame):,}행")
    evidence_columns[1].metric("변수", f"{frame.shape[1]:,}개")
    evidence_columns[2].metric("전처리 이력", f"{len(st.session_state.get('preprocessing_history', [])):,}건")
    evidence_columns[3].metric("시각화 Source", f"{len(st.session_state.get('visualization_sources', [])):,}건")

    notice = st.session_state.pop("insight_notice", None)
    if notice:
        st.success(notice)
    error = st.session_state.pop("insight_error", None)
    if error:
        st.error(error)

    _render_history(history)

    question = st.chat_input(
        "데이터 설명, 전체 보고서 또는 차트를 요청하세요.",
        key="insight_question",
        disabled=api_key is None,
    )
    if question:
        user_message = InsightMessage(role="user", text=question).model_dump(mode="json")
        previous_history = list(st.session_state.get("insight_history", []))
        st.session_state.insight_history = previous_history + [user_message]
        try:
            with st.spinner("Gemini가 요청 목적과 분석 근거를 확인하고 있습니다..."):
                evidence = build_evidence_context(
                    st.session_state.df,
                    frame,
                    st.session_state.source_filename,
                    st.session_state.get("preprocessing_history", []),
                    st.session_state.get("visualization_sources", []),
                )
                execution = execute_request(
                    api_key=api_key,
                    model=model,
                    question=question,
                    frame=frame,
                    source_filename=st.session_state.source_filename or "data",
                    evidence_context=evidence,
                    history=previous_history,
                )
            st.session_state.insight_history.append(execution.message.model_dump(mode="json"))
            if execution.visualization_source is not None:
                sources = list(st.session_state.get("visualization_sources", []))
                sources.append(execution.visualization_source)
                st.session_state.visualization_sources = sources[-20:]
            st.session_state.insight_notice = "답변과 관련 source를 대화 이력에 저장했습니다."
        except InsightAPIError as exc:
            st.session_state.insight_error = str(exc)
        st.rerun()

    if history:
        st.subheader("비즈니스 인사이트 다운로드")
        json_col, markdown_col = st.columns(2)
        stem = (st.session_state.source_filename or "data").rsplit(".", 1)[0]
        json_col.download_button(
            "대화·차트 JSON 다운로드",
            data=lambda: history_payload_bytes(
                st.session_state.insight_history,
                st.session_state.insight_model,
                st.session_state.source_filename,
                st.session_state.df_clean,
            ),
            file_name=f"{stem}_business_insight.json",
            mime="application/json",
            width="stretch",
            on_click="ignore",
        )
        markdown_col.download_button(
            "보고서 Markdown 다운로드",
            data=lambda: history_markdown_bytes(
                st.session_state.insight_history, st.session_state.source_filename
            ),
            file_name=f"{stem}_business_insight.md",
            mime="text/markdown",
            width="stretch",
            on_click="ignore",
        )
