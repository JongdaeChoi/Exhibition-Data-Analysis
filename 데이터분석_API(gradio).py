"""Gradio version of the exhibition data analysis assistant.

Run locally:
    python -m pip install -r requirements.txt
    python ".\\데이터분석_API(gradio).py"
"""

from __future__ import annotations

import ast
import contextlib
import datetime as dt
import io
import json
import mimetypes
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google import genai
from google.genai import types


MODEL_OPTIONS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
)
MAX_CONTEXT_CHARS = 60_000
HISTORY_SCHEMA_VERSION = 1
MODE_OPTIONS = ("일반 분석", "파이썬 코드 자동 실행")


def new_session() -> dict[str, Any]:
    return {
        "dataframe": None,
        "source_filename": None,
        "messages": [],
    }


def clone_session(session: dict[str, Any] | None) -> dict[str, Any]:
    return session if isinstance(session, dict) else new_session()


def read_table(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        last_error = None
        for encoding in ("utf-8-sig", "cp949", "utf-8"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error:
            raise last_error
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("CSV, XLSX 또는 XLS 파일만 데이터로 등록할 수 있습니다.")


def load_dataset(file_path: str | None, session: dict[str, Any] | None):
    session = clone_session(session)
    if not file_path:
        return session, None, "데이터 파일을 선택하세요."
    try:
        frame = read_table(file_path)
        session["dataframe"] = frame.copy()
        session["source_filename"] = Path(file_path).name
        return (
            session,
            frame.head(20),
            f"✅ 데이터 등록 완료: {Path(file_path).name} / {frame.shape[0]:,}행 × {frame.shape[1]:,}열",
        )
    except Exception as exc:
        return session, None, f"❌ 데이터 등록 실패: {exc}"


def make_data_context(frame: pd.DataFrame, source_filename: str) -> str:
    describe = frame.describe(include="all").transpose().to_string()
    context = f"""
[분석 파일]
{source_filename}

[데이터 크기]
{frame.shape}

[컬럼 및 데이터 타입]
{frame.dtypes.to_string()}

[결측치]
{frame.isnull().sum().to_string()}

[기초 통계]
{describe}

[상위 5행]
{frame.head(5).to_string()}
""".strip()
    return context[:MAX_CONTEXT_CHARS]


def make_system_prompt(frame: pd.DataFrame, source_filename: str) -> str:
    return f"""
당신은 전시회 전문 데이터 분석 수석 컨설턴트입니다.
아래 데이터 정보를 바탕으로 정확하고 실행 가능한 분석을 제공하세요.
항상 한국어로 답하세요. 데이터에 없는 사실은 추측이라고 명시하세요.

[응답 규칙]
- 사용자가 요청한 결과만 간결하게 답하세요.
- 실제 계산이 필요하면 현재 데이터프레임의 컬럼에 맞는 Python 코드를 제공하세요.
- 이 Gradio 테스트 앱은 모델이 작성한 코드를 자동 실행하지 않습니다.
- 데이터에 없는 숫자를 계산된 사실처럼 만들지 마세요.

{make_data_context(frame, source_filename)}
""".strip()


def content_history(messages: list[dict[str, Any]]) -> list[types.Content]:
    history = []
    for message in messages:
        if message.get("include_in_context") is False:
            continue
        text = str(message.get("content", "")).strip()
        if not text:
            continue
        role = "model" if message.get("role") == "assistant" else "user"
        history.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
    return history


def table_for_gradio(value: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Gradio가 숨기는 의미 있는 인덱스를 일반 열로 변환합니다."""
    table = value.to_frame() if isinstance(value, pd.Series) else value.copy()
    index = table.index
    is_default_index = (
        isinstance(index, pd.RangeIndex)
        and index.name is None
        and index.start == 0
        and index.step == 1
        and index.stop == len(table)
    )
    if is_default_index:
        return table

    existing_names = {str(column) for column in table.columns}
    safe_names = []
    for position, name in enumerate(index.names, 1):
        candidate = str(name) if name is not None else f"index_{position}"
        while candidate in existing_names or candidate in safe_names:
            candidate += "_index"
        safe_names.append(candidate)
    table.index = table.index.set_names(safe_names)
    return table.reset_index()


def execute_python_blocks(answer: str, frame: pd.DataFrame):
    """제한된 분석 환경에서 모델의 Python 블록을 실행합니다."""
    blocks = re.findall(r"```python\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        return frame, "코드 블록이 없어 자동 실행하지 않았습니다.", None, []

    captured: list[Any] = []
    stdout = io.StringIO()
    chart_paths: list[str] = []
    output_dir = Path(tempfile.mkdtemp(prefix="exhibition_gradio_charts_"))

    def capture_display(value: Any):
        captured.append(value)

    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance,
        "len": len, "list": list, "max": max, "min": min, "print": print,
        "range": range, "round": round, "set": set, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    }
    namespace = {
        "__builtins__": safe_builtins,
        "df": frame.copy(deep=True),
        "df_clean": frame,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "display": capture_display,
    }
    blocked_calls = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"}
    blocked_roots = {"os", "sys", "subprocess", "socket", "pathlib", "requests", "httpx", "shutil"}

    def target_root_name(target):
        while isinstance(target, (ast.Subscript, ast.Attribute)):
            target = target.value
        return target.id if isinstance(target, ast.Name) else None

    def uses_dataframe(expression):
        return any(isinstance(item, ast.Name) and item.id in {"df", "df_clean"}
                   for item in ast.walk(expression))

    for block_index, raw_code in enumerate(blocks, 1):
        code = re.sub(
            r"^\s*(?:from\s+(?:pandas|numpy|matplotlib(?:\.pyplot)?|seaborn)\s+import.*|"
            r"import\s+(?:pandas|numpy|matplotlib(?:\.pyplot)?|seaborn)(?:\s+as\s+\w+)?\s*)$",
            "", raw_code, flags=re.MULTILINE,
        )
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise RuntimeError("안전 차단: 생성 코드에서 추가 모듈을 import할 수 없습니다.")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise RuntimeError("안전 차단: 특수 속성 접근은 허용되지 않습니다.")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
                    raise RuntimeError(f"안전 차단: {node.func.id}() 호출은 허용되지 않습니다.")
                if isinstance(node.func, ast.Attribute) and target_root_name(node.func) in blocked_roots:
                    raise RuntimeError("안전 차단: 파일·프로세스·네트워크 작업은 허용되지 않습니다.")
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                roots = {target_root_name(target) for target in targets}
                if "df" in roots:
                    raise RuntimeError("안전 차단: 업로드 원본 df는 수정할 수 없습니다.")
                if "df_clean" in roots and any(
                    isinstance(target, ast.Name) and target.id == "df_clean" for target in targets
                ):
                    value = getattr(node, "value", None)
                    if value is not None and not uses_dataframe(value):
                        raise RuntimeError("안전 차단: df_clean을 샘플 데이터로 교체할 수 없습니다.")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if target_root_name(node.func.value) == "df" and any(
                    kw.arg == "inplace" and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True for kw in node.keywords
                ):
                    raise RuntimeError("안전 차단: 업로드 원본 df의 inplace 수정은 허용되지 않습니다.")

        before = set(plt.get_fignums())
        with contextlib.redirect_stdout(stdout):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                final_expression = ast.Expression(tree.body.pop().value)
                ast.fix_missing_locations(tree)
                ast.fix_missing_locations(final_expression)
                if tree.body:
                    exec(compile(tree, "<gemini-code>", "exec"), namespace)
                result = eval(compile(final_expression, "<gemini-code>", "eval"), namespace)
                if result is not None:
                    capture_display(result)
            else:
                exec(compile(tree, "<gemini-code>", "exec"), namespace)

        for fig_num in sorted(set(plt.get_fignums()) - before):
            chart_path = output_dir / f"generated_chart_{block_index}_{fig_num}.png"
            plt.figure(fig_num).savefig(chart_path, dpi=150, bbox_inches="tight")
            chart_paths.append(str(chart_path))
            plt.close(fig_num)

    updated_frame = namespace.get("df_clean")
    if not isinstance(updated_frame, pd.DataFrame):
        raise RuntimeError("실행 결과의 df_clean이 DataFrame이 아닙니다.")

    result_table = None
    text_outputs = []
    for value in captured:
        if isinstance(value, pd.DataFrame):
            result_table = table_for_gradio(value)
        elif isinstance(value, pd.Series):
            result_table = table_for_gradio(value)
        else:
            text_outputs.append(str(value))
    printed = stdout.getvalue().strip()
    if printed:
        text_outputs.append(printed)
    summary = "✅ Python 코드 자동 실행 완료"
    if text_outputs:
        summary += "\n\n```text\n" + "\n".join(text_outputs)[:10_000] + "\n```"
    if result_table is not None:
        summary += f"\n\n표 결과: {result_table.shape[0]:,}행 × {result_table.shape[1]:,}열"
    if chart_paths:
        summary += f"\n\n차트 결과: {len(chart_paths)}개"
    return updated_frame, summary, result_table, chart_paths


def visible_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history = []
    for message in messages:
        if message.get("hidden") or message.get("role") not in {"user", "assistant"}:
            continue
        kind = message.get("kind", "text")
        if kind == "table":
            content = gr.Dataframe(
                value=message.get("content"),
                label="실행 표 결과",
                interactive=False,
            )
        elif kind == "chart":
            content = gr.Image(
                value=message.get("content"),
                label="실행 차트 결과",
                interactive=False,
            )
        else:
            content = str(message.get("content", ""))
        history.append({"role": message["role"], "content": content})
    return history


def ask_gemini(
    question: str,
    image_path: str | None,
    model_name: str,
    mode_name: str,
    api_key_input: str,
    session: dict[str, Any] | None,
):
    session = clone_session(session)
    question = (question or "").strip()
    frame = session.get("dataframe")
    api_key = (api_key_input or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

    if not question:
        return session, visible_history(session["messages"]), "질문을 입력하세요.", question, image_path
    if not isinstance(frame, pd.DataFrame):
        return session, visible_history(session["messages"]), "먼저 분석 데이터 파일을 등록하세요.", question, image_path
    if not api_key:
        return session, visible_history(session["messages"]), "Gemini API 키를 입력하거나 GEMINI_API_KEY를 설정하세요.", question, image_path

    try:
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=make_system_prompt(
                    frame, session.get("source_filename") or "uploaded data"
                )
            ),
            history=content_history(session["messages"]),
        )

        request_text = question
        if mode_name == "파이썬 코드 자동 실행":
            request_text += """

[이번 응답 형식]
표·차트·집계·변환처럼 실제 계산이 필요한 요청이면 Python 코드 블록 하나만 반환하세요.
요약·설명·해석 요청이면 코드를 작성하지 말고 요청한 설명만 답하세요.
코드는 이미 존재하는 df_clean을 사용하세요. 파일·프로세스·네트워크 작업과 추가 import는 하지 마세요.
결과는 result_df 같은 별도 변수에 저장하고 display()로 표시하세요.
"""
        parts = [types.Part.from_text(text=request_text)]
        visible_question = question
        if image_path:
            image_bytes = Path(image_path).read_bytes()
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            visible_question += f"\n\n[첨부 이미지: {Path(image_path).name}]"

        response = chat.send_message(parts)
        answer = response.text or "(텍스트 응답 없음)"
        session["messages"].extend(
            [
                {"role": "user", "content": visible_question},
                {"role": "assistant", "content": answer},
            ]
        )
        execution_summary, result_table, chart_paths = "", None, []
        if mode_name == "파이썬 코드 자동 실행":
            try:
                updated_frame, execution_summary, result_table, chart_paths = execute_python_blocks(
                    answer, frame
                )
                session["dataframe"] = updated_frame
                if execution_summary.startswith("✅"):
                    session["messages"].append({
                        "role": "assistant",
                        "content": execution_summary,
                        "include_in_context": False,
                    })
                    if result_table is not None:
                        session["messages"].append({
                            "role": "assistant",
                            "content": result_table,
                            "kind": "table",
                            "include_in_context": False,
                        })
                    for chart_path in chart_paths:
                        session["messages"].append({
                            "role": "assistant",
                            "content": chart_path,
                            "kind": "chart",
                            "include_in_context": False,
                        })
                    session["messages"].append({
                        "role": "user",
                        "content": f"[로컬 Python 실행 결과]\n{execution_summary}",
                        "hidden": True,
                    })
            except Exception as exc:
                execution_summary = f"❌ Python 코드 실행 실패: {exc}"
                session["messages"].append({
                    "role": "assistant",
                    "content": execution_summary,
                    "include_in_context": False,
                })
        return session, visible_history(session["messages"]), "✅ 응답 완료", "", None
    except Exception as exc:
        return session, visible_history(session["messages"]), f"❌ Gemini 요청 실패: {exc}", question, image_path


def decode_readable_file(content: bytes) -> str:
    encodings = ["utf-8-sig", "cp949"]
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.insert(1, "utf-16")
    for encoding in encodings:
        try:
            text = content.decode(encoding)
            visible = sum(char.isprintable() or char in "\r\n\t" for char in text)
            if "\x00" not in text and visible / max(len(text), 1) >= 0.9:
                return text
        except UnicodeError:
            continue
    raise ValueError("텍스트로 읽을 수 없는 파일입니다.")


def normalize_saved_messages(data: Any) -> list[dict[str, Any]]:
    history = data if isinstance(data, list) else data.get("history", data.get("messages", []))
    normalized = []
    for message in history:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "model":
            role = "assistant"
        if role not in {"user", "assistant"}:
            continue
        if "content" in message:
            text = str(message["content"])
        else:
            text = "\n".join(
                str(part.get("data", ""))
                for part in message.get("parts", [])
                if part.get("type") == "text"
            )
        if text.strip():
            normalized.append({"role": role, "content": text})
    return normalized


def register_previous_conversation(file_path: str | None, session: dict[str, Any] | None):
    session = clone_session(session)
    if not file_path:
        return session, visible_history(session["messages"]), "기존 대화 파일을 선택하세요."
    try:
        path = Path(file_path)
        text = decode_readable_file(path.read_bytes())
        try:
            data = json.loads(text)
            restored = normalize_saved_messages(data)
        except (json.JSONDecodeError, AttributeError, TypeError):
            restored = []

        if restored:
            session["messages"] = restored
            status = f"✅ 기존 대화 {len(restored)}개 메시지 등록 완료"
        else:
            reference = text.strip()
            if not reference:
                raise ValueError("등록할 내용이 없는 빈 파일입니다.")
            truncated = len(reference) > MAX_CONTEXT_CHARS
            reference = reference[:MAX_CONTEXT_CHARS]
            session["messages"] = [{
                "role": "user",
                "content": (
                    f"[등록된 기존 대화 파일: {path.name}]\n"
                    "다음은 이전 대화입니다. 후속 질문에 답할 때 참고하세요.\n\n"
                    f"{reference}"
                ),
                "hidden": True,
            }]
            suffix = " (길이 제한으로 일부만 등록)" if truncated else ""
            status = f"✅ 기존 대화 파일 등록 완료: {path.name}{suffix}"
        return session, visible_history(session["messages"]), status
    except Exception as exc:
        return session, visible_history(session["messages"]), f"❌ 기존 대화 등록 실패: {exc}"


def save_conversation(session: dict[str, Any] | None):
    session = clone_session(session)
    if not session["messages"]:
        raise gr.Error("저장할 대화가 없습니다.")
    payload = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_filename": session.get("source_filename"),
        "messages": [
            message for message in session["messages"]
            if message.get("include_in_context") is not False
        ],
    }
    output_dir = Path(tempfile.gettempdir()) / "exhibition_gradio_exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"exhibition_chat_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def clear_display(session: dict[str, Any] | None):
    session = clone_session(session)
    return session, [], "출력만 지웠습니다. 대화 문맥은 유지됩니다."


def start_new_conversation(session: dict[str, Any] | None):
    session = clone_session(session)
    session["messages"] = []
    return session, [], "✅ 새 대화를 시작했습니다."


with gr.Blocks(title="전시 데이터 분석") as demo:
    gr.Markdown("# 전시 데이터 분석 · Gradio 테스트")
    gr.Markdown("CSV/Excel 데이터를 등록한 뒤 Gemini에게 질문하세요. API 키는 저장되지 않습니다.")
    session_state = gr.State(new_session)

    with gr.Row():
        model = gr.Dropdown(MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Gemini 모델")
        mode = gr.Dropdown(MODE_OPTIONS, value=MODE_OPTIONS[0], label="모드")
        api_key = gr.Textbox(
            label="Gemini API 키",
            type="password",
            placeholder="입력하지 않으면 GEMINI_API_KEY 환경변수 사용",
        )

    with gr.Row():
        dataset_file = gr.File(label="분석 데이터 등록", file_types=[".csv", ".xlsx", ".xls"], type="filepath")
        previous_file = gr.File(label="기존대화 등록", type="filepath")

    load_button = gr.Button("데이터 불러오기", variant="primary")
    register_button = gr.Button("기존대화 불러오기")
    preview = gr.Dataframe(label="데이터 미리보기", interactive=False)
    chatbot = gr.Chatbot(label="분석 대화", height=500)
    status = gr.Markdown("준비됨")

    with gr.Row():
        question = gr.Textbox(label="질문", placeholder="전시 데이터에 관한 질문을 입력하세요.", lines=3, scale=4)
        image = gr.Image(label="이미지 첨부", type="filepath", scale=1)

    with gr.Row():
        send_button = gr.Button("질문 전송", variant="primary")
        clear_button = gr.Button("출력 지우기")
        new_button = gr.Button("새 대화")
        save_button = gr.DownloadButton("대화 JSON 저장")

    load_button.click(
        load_dataset,
        inputs=[dataset_file, session_state],
        outputs=[session_state, preview, status],
    )
    register_button.click(
        register_previous_conversation,
        inputs=[previous_file, session_state],
        outputs=[session_state, chatbot, status],
    )
    send_inputs = [question, image, model, mode, api_key, session_state]
    send_outputs = [session_state, chatbot, status, question, image]
    send_button.click(ask_gemini, inputs=send_inputs, outputs=send_outputs)
    question.submit(ask_gemini, inputs=send_inputs, outputs=send_outputs)
    clear_button.click(clear_display, inputs=[session_state], outputs=[session_state, chatbot, status])
    new_button.click(start_new_conversation, inputs=[session_state], outputs=[session_state, chatbot, status])
    save_button.click(save_conversation, inputs=[session_state], outputs=[save_button])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
