from __future__ import annotations

import base64
import hashlib
import json

import pandas as pd
import streamlit as st

from core.config import configured_api_keys, default_ai_provider
from core.i18n import current_language, localized_table, translate
from data.loader import DataLoadError, load_table
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
    rebuild_chart_record,
    restore_history,
)


def _configured_api_key(provider: str = "OpenAI") -> tuple[str | None, str]:
    """Return a server-side credential without exposing or persisting its value."""
    value = configured_api_keys().get(provider)
    expected = "OPENAI_API_KEY" if provider == "OpenAI" else "GEMINI_API_KEY"
    return (value, f"Streamlit Secrets / environment · {expected}") if value else (None, "미설정")


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
    suffix = uploaded.name.lower().rsplit(".", 1)[-1]
    tabular = suffix in {"csv", "xlsx", "xls", "json"}
    if not is_image and suffix not in {"csv", "xlsx", "xls", "json", "txt"}:
        raise ValueError(f"{uploaded.name}: Excel, CSV, JSON 또는 TXT 파일만 등록할 수 있습니다.")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    extracted = ""
    if tabular:
        try:
            if suffix == "json":
                value = json.loads(_decode_document(raw))
                frame = pd.DataFrame(value if isinstance(value, list) else value.get("data", value))
            else:
                frame = load_table(raw, uploaded.name).df
        except (DataLoadError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{uploaded.name}: 데이터셋을 읽을 수 없습니다. {exc}") from exc
        if frame.empty:
            raise ValueError(f"{uploaded.name}: 데이터가 비어 있습니다.")
        st.session_state.setdefault("insight_reference_datasets", {})[digest] = {
            "filename": uploaded.name,
            "frame": frame,
        }
        description = frame.describe(include="all").transpose().reset_index().head(100)
        extracted = (
            f"[참고 데이터셋] {uploaded.name}\n행={len(frame)}, 열={frame.shape[1]}\n"
            f"변수={list(map(str, frame.columns))}\n"
            f"기술통계={description.to_json(force_ascii=False, orient='records', default_handler=str)}"
        )[:100_000]
    elif not is_image:
        extracted = _decode_document(raw)[:100_000]
    return InsightAttachment(
        id=digest,
        filename=uploaded.name,
        mime_type=mime_type,
        kind="image" if is_image else "dataset" if tabular else "document",
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


def _reexecute_history_chart(
    message_index: int,
    chart_index: int,
    raw_spec: str,
) -> None:
    try:
        payload = json.loads(raw_spec)
        if not isinstance(payload, dict):
            raise ValueError("ChartSpec JSON은 하나의 객체여야 합니다.")
        chart_record, chart_source = rebuild_chart_record(
            st.session_state.df_clean,
            payload,
            st.session_state.source_filename or "data",
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        st.session_state.insight_error = f"ChartSpec 재실행 실패: {exc}"
        st.rerun()
    history = list(st.session_state.get("insight_history", []))
    history[message_index]["charts"][chart_index] = chart_record.model_dump(mode="json")
    st.session_state.insight_history = history
    sources = list(st.session_state.get("visualization_sources", []))
    sources.append(chart_source)
    st.session_state.visualization_sources = sources[-20:]
    st.session_state.insight_notice = "수정한 Pydantic ChartSpec으로 차트를 다시 실행했습니다."
    st.session_state.insight_error = None
    st.rerun()


def _render_history(history: list[dict], visible_from: int = 0) -> None:
    for message_index in range(visible_from, len(history)):
        message = history[message_index]
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
            for chart_index, chart in enumerate(message.get("charts", [])):
                try:
                    image = base64.b64decode(chart.get("image_base64", ""))
                except (ValueError, TypeError):
                    continue
                if image:
                    st.image(image, width="stretch")
                source = chart.get("source", {})
                with st.expander("차트 통계자료와 Pydantic 설정", expanded=False):
                    charts = source.get("charts", []) if isinstance(source, dict) else []
                    for item_index, item in enumerate(charts):
                        spec_json = json.dumps(
                            item.get("spec", {}), ensure_ascii=False, indent=2
                        )
                        editor_key = (
                            f"insight_chart_spec_editor_{message_index}_{chart_index}_{item_index}"
                        )
                        signature_key = f"{editor_key}_signature"
                        if st.session_state.get(signature_key) != spec_json:
                            st.session_state[signature_key] = spec_json
                            st.session_state[editor_key] = spec_json
                        edited_spec = st.text_area(
                            "ChartSpec JSON 직접 수정",
                            height=320,
                            key=editor_key,
                        )
                        if st.button(
                            "수정한 Pydantic 설정으로 차트 재실행",
                            key=f"insight_chart_rerun_{message_index}_{chart_index}_{item_index}",
                            type="primary",
                            width="stretch",
                        ):
                            _reexecute_history_chart(
                                message_index, chart_index, edited_spec
                            )
                        statistics = item.get("statistics", [])
                        if statistics:
                            statistics = localized_table(
                                pd.DataFrame(statistics), value_columns=("분석 타입",)
                            )
                        st.dataframe(statistics, width="stretch", hide_index=True)


def _restore_uploaded_history(uploaded) -> None:
    history, saved_model = restore_history(uploaded.getvalue(), uploaded.name)
    st.session_state.insight_history = history
    if saved_model in MODEL_OPTIONS:
        st.session_state.insight_restored_model = saved_model
        st.session_state.insight_restored_provider = next(
            provider for provider, models in PROVIDER_MODELS.items() if saved_model in models
        )
    st.session_state.insight_visible_from = 0
    st.session_state.insight_notice = f"기존 대화 {len(history):,}개 메시지를 등록했습니다."
    st.session_state.insight_error = None
    st.rerun()


def _apply_restored_model_before_widgets() -> None:
    """Apply restored widget values before their widgets are instantiated."""
    saved_model = st.session_state.pop("insight_restored_model", None)
    saved_provider = st.session_state.pop("insight_restored_provider", None)
    if saved_model in MODEL_OPTIONS and saved_provider in PROVIDER_MODELS:
        st.session_state["insight_provider"] = saved_provider
        st.session_state["insight_model"] = saved_model


def _register_references(uploaded_files) -> None:
    current = [InsightAttachment.model_validate(item) for item in st.session_state.get("insight_references", [])]
    by_id = {item.id: item for item in current}
    pending = set(st.session_state.get("insight_pending_attachment_ids", []))
    errors = []
    success_count = 0
    for uploaded in uploaded_files or []:
        try:
            item = _attachment_from_upload(uploaded)
            by_id[item.id] = item
            pending.add(item.id)
            success_count += 1
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
    st.session_state.insight_references = [item.model_dump(mode="json") for item in by_id.values()]
    st.session_state.insight_pending_attachment_ids = list(pending)
    st.session_state.insight_notice = f"참고자료 {success_count:,}개를 분석 컨텍스트에 등록했습니다."
    st.session_state.insight_reference_errors = errors
    st.rerun()


def render_insight() -> None:
    _apply_restored_model_before_widgets()
    api_keys = configured_api_keys()
    if st.session_state.get("insight_provider") not in PROVIDER_MODELS:
        st.session_state.insight_provider = default_ai_provider(api_keys)
    st.header("Business Insight")
    st.caption(
        "현재 df_clean, 전처리 이력, 기술통계, 저장된 시각화 자료, 첨부자료와 대화를 근거로 답합니다."
    )
    frame = st.session_state.df_clean
    history = list(st.session_state.get("insight_history", []))

    with st.container(border=True):
        provider_col, model_col = st.columns([1, 2], gap="small")
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
        api_key, key_source = _configured_api_key(provider)
        if api_key:
            st.success(f"API 인증 준비 완료 · {key_source}")
        else:
            expected = "OPENAI_API_KEY" if provider == "OpenAI" else "GEMINI_API_KEY"
            st.warning(f"Streamlit Secrets 또는 환경변수에 `{expected}`를 설정하세요.")

        reference_col, history_col = st.columns(2, gap="small")
        references_upload = reference_col.file_uploader(
            "참고 데이터셋 업로드",
            type=["csv", "xlsx", "xls", "json", "txt"],
            accept_multiple_files=True,
            key="insight_reference_upload",
        )
        if reference_col.button(
            "참고자료 등록", disabled=not references_upload,
            key="register_insight_references", width="stretch",
        ):
            _register_references(references_upload)

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

        reference_datasets = st.session_state.get("insight_reference_datasets", {})
        dataset_options = ["main", *reference_datasets.keys()]
        dataset_labels = {
            value: (
                f"주 분석 데이터 · {st.session_state.current_file_name}"
                if value == "main"
                else f"참고 데이터 · {reference_datasets[value]['filename']}"
            )
            for value in dataset_options
        }
        label_to_id = {label: value for value, label in dataset_labels.items()}
        selected_id = st.session_state.get("insight_active_dataset", "main")
        if selected_id not in dataset_options:
            selected_id = "main"
        selected_dataset_label = st.selectbox(
            "Insight에서 사용할 데이터셋",
            list(label_to_id),
            index=dataset_options.index(selected_id),
            key="insight_active_dataset_label",
            help="데이터셋은 자동 병합되지 않으며 선택한 한 파일만 계산과 차트에 사용합니다.",
        )
        active_dataset_id = label_to_id[selected_dataset_label]
        st.session_state.insight_active_dataset = active_dataset_id

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
    if active_dataset_id == "main":
        analysis_frame = frame
        analysis_original = st.session_state.df
        analysis_filename = st.session_state.source_filename or "data"
    else:
        selected_reference = st.session_state.insight_reference_datasets[active_dataset_id]
        analysis_frame = selected_reference["frame"]
        analysis_original = analysis_frame
        analysis_filename = selected_reference["filename"]
    reference_errors = st.session_state.pop("insight_reference_errors", [])
    for reference_error in reference_errors:
        st.warning(reference_error)
    evidence_columns = st.columns(5)
    english = current_language() == "English"
    evidence_columns[0].metric(
        "현재 데이터", f"{len(frame):,} rows" if english else f"{len(frame):,}행"
    )
    evidence_columns[1].metric(
        "변수", f"{frame.shape[1]:,}" if english else f"{frame.shape[1]:,}개"
    )
    history_count = len(st.session_state.get("preprocessing_history", []))
    source_count = len(st.session_state.get("visualization_sources", []))
    evidence_columns[2].metric(
        "전처리 이력", f"{history_count:,}" if english else f"{history_count:,}건"
    )
    evidence_columns[3].metric(
        "시각화 Source", f"{source_count:,}" if english else f"{source_count:,}건"
    )
    evidence_columns[4].metric(
        "참고자료", f"{len(references):,}" if english else f"{len(references):,}개"
    )

    notice = st.session_state.pop("insight_notice", None)
    if notice:
        st.success(notice)
    error = st.session_state.pop("insight_error", None)
    if error:
        st.error(error)

    visible_from = int(st.session_state.get("insight_visible_from", 0))
    _render_history(history, visible_from)

    typed_question = st.chat_input(
        "데이터 설명·조회, 차트 또는 비즈니스 인사이트를 요청하세요.",
        key="insight_question",
    )
    question = (
        translate("현재 분석 컨텍스트 전체를 근거로 비즈니스 인사이트를 작성해 주세요.")
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
                    analysis_original,
                    analysis_frame,
                    analysis_filename,
                    st.session_state.get("preprocessing_history", []),
                    st.session_state.get("visualization_sources", []),
                    references,
                )
                execution = execute_request(
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    question=question,
                    original_frame=analysis_original,
                    frame=analysis_frame,
                    source_filename=analysis_filename,
                    evidence_context=evidence,
                    history=previous_history,
                    attachments=references,
                    response_language=current_language(),
                )
            st.session_state.insight_history.append(execution.message.model_dump(mode="json"))
            st.session_state.insight_pending_attachment_ids = []
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
        # Streamlit executes callable download data on a separate thread. Build
        # both files while the script still has a valid session context instead
        # of reading st.session_state from a deferred lambda.
        download_history = list(st.session_state.insight_history)
        json_download = history_payload_bytes(
            download_history,
            st.session_state.insight_model,
            st.session_state.source_filename,
            st.session_state.df_clean,
            st.session_state.insight_provider,
        )
        markdown_download = history_markdown_bytes(
            download_history, st.session_state.source_filename
        )
        json_col.download_button(
            "대화·차트·코드 JSON 다운로드",
            data=json_download,
            file_name=f"{stem}_insight_chat.json",
            mime="application/json",
            key="download_insight_json",
            width="stretch",
            on_click="ignore",
        )
        markdown_col.download_button(
            "보고서 Markdown 다운로드",
            data=markdown_download,
            file_name=f"{stem}_business_insight.md",
            mime="text/markdown",
            key="download_insight_markdown",
            width="stretch",
            on_click="ignore",
        )
