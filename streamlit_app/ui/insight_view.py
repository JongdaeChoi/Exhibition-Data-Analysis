from __future__ import annotations

import base64
import hashlib
import json
import os

import streamlit as st

from insight.context import build_evidence_context
from insight.models import (
    MODEL_OPTIONS,
    MODEL_LABELS,
    PROVIDER_MODELS,
    InsightAttachment,
    InsightMessage,
)
from insight.service import (
    InsightAPIError,
    execute_request,
    history_markdown_bytes,
    history_payload_bytes,
    restore_history,
)


def _configured_api_key(entered: str, provider: str = "Gemini") -> tuple[str | None, str]:
    if entered.strip():
        return entered.strip(), "화면 입력"
    names = ("OPENAI_API_KEY",) if provider == "OpenAI" else ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    for name in names:
        value = os.getenv(name)
        if value:
            return value, f"환경변수 {name}"
    secret_names = names if provider == "OpenAI" else (*names, "exhibition")
    try:
        for name in secret_names:
            value = st.secrets.get(name)
            if value:
                return str(value), f"Streamlit secrets {name}"
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return None, "미설정"


def _decode_document(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("참고자료를 텍스트로 읽을 수 없습니다.")


def _attachment_from_upload(uploaded) -> InsightAttachment:
    raw = uploaded.getvalue()
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError(f"{uploaded.name}: 파일 크기는 5MB 이하여야 합니다.")
    mime_type = uploaded.type or "application/octet-stream"
    is_image = mime_type.startswith("image/")
    if not is_image and not uploaded.name.lower().endswith((".json", ".txt", ".md")):
        raise ValueError(f"{uploaded.name}: JSON, TXT, MD 또는 이미지 파일만 등록할 수 있습니다.")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    extracted = ""
    if not is_image:
        extracted = _decode_document(raw)
        if uploaded.name.lower().endswith(".json"):
            try:
                extracted = json.dumps(json.loads(extracted), ensure_ascii=False, indent=2)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{uploaded.name}: JSON 형식이 올바르지 않습니다.") from exc
        extracted = extracted[:100_000]
    return InsightAttachment(
        id=digest,
        filename=uploaded.name,
        mime_type=mime_type,
        kind="image" if is_image else "document",
        content_base64=base64.b64encode(raw).decode("ascii") if is_image else "",
        extracted_text=extracted,
    )


def _render_code(code: dict) -> None:
    with st.expander("실행한 Python 코드와 결과", expanded=True):
        st.code(code.get("code", ""), language="python")
        if code.get("modifies_df_clean"):
            st.caption(
                f"df_clean 반영 · {code.get('before_shape')} → {code.get('after_shape')}"
            )
        for output in code.get("outputs", []):
            if output.get("type") == "dataframe":
                st.dataframe(output.get("records", []), width="stretch", hide_index=True)
                if output.get("truncated"):
                    st.caption(f"전체 {output.get('row_count', 0):,}행 중 처음 500행을 표시합니다.")
            else:
                st.text(output.get("value", ""))


def _render_history(history: list[dict], visible_from: int = 0) -> None:
    for message in history[visible_from:]:
        if message.get("hidden"):
            continue
        role = "user" if message.get("role") == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message.get("text", ""))
            for attachment in message.get("attachments", []):
                st.caption(f"첨부 · {attachment.get('filename', '')}")
                if attachment.get("kind") == "image" and attachment.get("content_base64"):
                    st.image(base64.b64decode(attachment["content_base64"]), width=360)
            if message.get("code"):
                _render_code(message["code"])
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
        st.session_state.insight_provider = next(
            provider for provider, models in PROVIDER_MODELS.items() if saved_model in models
        )
    st.session_state.insight_visible_from = 0
    st.session_state.insight_notice = f"기존 대화 {len(history):,}개 메시지를 등록했습니다."
    st.session_state.insight_error = None
    st.rerun()


def _register_references(uploaded_files) -> None:
    current = [InsightAttachment.model_validate(item) for item in st.session_state.get("insight_references", [])]
    by_id = {item.id: item for item in current}
    pending = set(st.session_state.get("insight_pending_attachment_ids", []))
    for uploaded in uploaded_files or []:
        item = _attachment_from_upload(uploaded)
        by_id[item.id] = item
        pending.add(item.id)
    st.session_state.insight_references = [item.model_dump(mode="json") for item in by_id.values()]
    st.session_state.insight_pending_attachment_ids = list(pending)
    st.session_state.insight_notice = f"참고자료 {len(uploaded_files or []):,}개를 분석 컨텍스트에 등록했습니다."
    st.rerun()


def render_insight() -> None:
    st.header("Business Insight")
    st.caption(
        "현재 df_clean, 전처리 이력, 기술통계, 저장된 시각화 자료, 첨부자료와 대화를 근거로 답합니다."
    )
    frame = st.session_state.df_clean
    history = list(st.session_state.get("insight_history", []))

    with st.container(border=True):
        provider_col, model_col, key_col = st.columns([1, 1.4, 2], gap="small")
        provider = provider_col.selectbox(
            "API 공급자", tuple(PROVIDER_MODELS), key="insight_provider"
        )
        allowed_models = PROVIDER_MODELS[provider]
        if st.session_state.get("insight_model") not in allowed_models:
            st.session_state.insight_model = allowed_models[0]
        model = model_col.selectbox(
            "모델", allowed_models, key="insight_model",
            format_func=lambda value: MODEL_LABELS.get(value, value),
            help="모델을 변경해도 저장된 대화 이력은 유지됩니다.",
        )
        entered_key = key_col.text_input(
            f"{provider} API Key (선택)", type="password", key=f"insight_api_key_{provider}",
        )
        api_key, key_source = _configured_api_key(entered_key, provider)
        if api_key:
            st.success(f"API 인증 준비 완료 · {key_source}")
        else:
            expected = "OPENAI_API_KEY" if provider == "OpenAI" else "GEMINI_API_KEY"
            st.warning(f"`{expected}`를 설정하거나 API Key를 입력하세요.")

        reference_col, history_col = st.columns(2, gap="small")
        references_upload = reference_col.file_uploader(
            "참고자료·이미지 업로드",
            type=["json", "txt", "md", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="insight_reference_upload",
        )
        if reference_col.button(
            "참고자료 등록", disabled=not references_upload,
            key="register_insight_references", width="stretch",
        ):
            try:
                _register_references(references_upload)
            except (ValueError, TypeError) as exc:
                st.error(str(exc))

        history_upload = history_col.file_uploader(
            "기존 대화 등록", type=["json", "md", "txt"], key="insight_history_upload",
        )
        if history_col.button(
            "기존 대화 불러오기", disabled=history_upload is None,
            key="restore_insight_history", width="stretch",
        ):
            try:
                _restore_uploaded_history(history_upload)
            except (ValueError, TypeError) as exc:
                st.error(str(exc))

        action_col, clear_col = st.columns([3, 1], gap="small")
        insight_requested = action_col.button(
            "비즈니스 인사이트 생성",
            disabled=api_key is None,
            key="generate_business_insight",
            width="stretch",
        )
        if clear_col.button("현재 화면 지우기", key="clear_insight_screen", width="stretch"):
            st.session_state.insight_visible_from = len(history)
            st.session_state.insight_notice = "화면을 지웠습니다. 전체 대화 이력은 계속 저장됩니다."
            st.rerun()

    references = [
        InsightAttachment.model_validate(item)
        for item in st.session_state.get("insight_references", [])
    ]
    evidence_columns = st.columns(5)
    evidence_columns[0].metric("현재 데이터", f"{len(frame):,}행")
    evidence_columns[1].metric("변수", f"{frame.shape[1]:,}개")
    evidence_columns[2].metric("전처리 이력", f"{len(st.session_state.get('preprocessing_history', [])):,}건")
    evidence_columns[3].metric("시각화 Source", f"{len(st.session_state.get('visualization_sources', [])):,}건")
    evidence_columns[4].metric("참고자료", f"{len(references):,}개")

    notice = st.session_state.pop("insight_notice", None)
    if notice:
        st.success(notice)
    error = st.session_state.pop("insight_error", None)
    if error:
        st.error(error)

    visible_from = int(st.session_state.get("insight_visible_from", 0))
    _render_history(history, visible_from)

    typed_question = st.chat_input(
        "데이터 설명·조회·수정, 차트 또는 비즈니스 인사이트를 요청하세요.",
        key="insight_question",
    )
    question = (
        "현재 분석 컨텍스트 전체를 근거로 비즈니스 인사이트를 작성해 주세요."
        if insight_requested
        else typed_question
    )
    if question:
        if api_key is None:
            st.session_state.insight_error = (
                f"{provider} API Key를 먼저 설정하세요. 입력한 질문은 실행되지 않았습니다."
            )
            st.rerun()
        pending_ids = set(st.session_state.get("insight_pending_attachment_ids", []))
        message_attachments = [item for item in references if item.id in pending_ids]
        user_message = InsightMessage(
            role="user", text=question, attachments=message_attachments
        ).model_dump(mode="json")
        previous_history = list(st.session_state.get("insight_history", []))
        st.session_state.insight_history = previous_history + [user_message]
        try:
            with st.spinner(f"{provider} 모델이 요청 목적과 분석 근거를 확인하고 있습니다..."):
                evidence = build_evidence_context(
                    st.session_state.df,
                    frame,
                    st.session_state.source_filename,
                    st.session_state.get("preprocessing_history", []),
                    st.session_state.get("visualization_sources", []),
                    references,
                )
                execution = execute_request(
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    question=question,
                    original_frame=st.session_state.df,
                    frame=frame,
                    source_filename=st.session_state.source_filename or "data",
                    evidence_context=evidence,
                    history=previous_history,
                    attachments=references,
                )
            st.session_state.insight_history.append(execution.message.model_dump(mode="json"))
            st.session_state.insight_pending_attachment_ids = []
            if execution.updated_frame is not None:
                before_shape = tuple(frame.shape)
                st.session_state.df_clean = execution.updated_frame
                preprocessing = list(st.session_state.get("preprocessing_history", []))
                preprocessing.append(
                    {
                        "section": "Insight Chat",
                        "operation": "사용자 요청에 따른 df_clean 수정",
                        "before_shape": before_shape,
                        "after_shape": tuple(execution.updated_frame.shape),
                        "code": execution.message.code.code if execution.message.code else "",
                    }
                )
                st.session_state.preprocessing_history = preprocessing
            if execution.visualization_source is not None:
                sources = list(st.session_state.get("visualization_sources", []))
                sources.append(execution.visualization_source)
                st.session_state.visualization_sources = sources[-20:]
            st.session_state.insight_notice = "답변과 실행 결과를 전체 대화 이력에 저장했습니다."
        except InsightAPIError as exc:
            st.session_state.insight_error = str(exc)
        st.rerun()

    if history:
        st.subheader("대화 내용 다운로드")
        json_col, markdown_col = st.columns(2)
        stem = (st.session_state.source_filename or "data").rsplit(".", 1)[0]
        json_col.download_button(
            "대화·차트·코드 JSON 다운로드",
            data=lambda: history_payload_bytes(
                st.session_state.insight_history,
                st.session_state.insight_model,
                st.session_state.source_filename,
                st.session_state.df_clean,
                st.session_state.insight_provider,
            ),
            file_name=f"{stem}_insight_chat.json",
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
