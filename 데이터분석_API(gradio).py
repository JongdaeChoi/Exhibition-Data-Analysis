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
import logging
import math
import mimetypes
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd
import seaborn as sns
from google import genai
from google.genai import types

try:
    import koreanize_matplotlib
except ImportError:
    koreanize_matplotlib = None


MODEL_OPTIONS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
)
MAX_CONTEXT_CHARS = 60_000
HISTORY_SCHEMA_VERSION = 1
MODE_OPTIONS = ("일반 분석", "파이썬 코드 자동 실행")
APP_CSS = """
#analysis-chatbot .message-row.bot-row:has(.image-container),
#analysis-chatbot .message-row.bot-row:has(.image-container) .flex-wrap,
#analysis-chatbot .message-row.bot-row:has(.image-container) .bot.message,
#analysis-chatbot .message-row.bot-row:has(.image-container) .message.component,
#analysis-chatbot .message-row.bot-row:has(.image-container) .image-container,
#analysis-chatbot .message-row.bot-row:has(.image-container) .image-container > button,
#analysis-chatbot .message-row.bot-row:has(.image-container) .image-frame {
    width: fit-content !important;
    max-width: 100% !important;
    height: auto !important;
}

#analysis-chatbot .message-row.bot-row:has(.image-container) .image-frame img {
    width: auto !important;
    height: auto !important;
    max-width: 100% !important;
    max-height: none !important;
    object-fit: contain !important;
}

#analysis-chatbot [data-testid="bot"],
#analysis-chatbot [data-testid="user"] {
    font-size: var(--text-sm) !important;
    line-height: 1.5 !important;
}
"""


def apply_korean_chart_font(figure=None) -> str:
    """실행 환경과 무관하게 차트의 한글 글꼴을 사용 가능한 폰트로 교정합니다."""
    if koreanize_matplotlib is not None:
        koreanize_matplotlib.koreanize()
        font_family = "NanumGothic"
    else:
        font_candidates = [
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf",
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        ]
        font_family = None
        for font_path in font_candidates:
            if font_path.exists():
                fm.fontManager.addfont(str(font_path))
                font_family = fm.FontProperties(fname=str(font_path)).get_name()
                break
        if font_family is None:
            available = {font.name for font in fm.fontManager.ttflist}
            font_family = next(
                (name for name in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR") if name in available),
                "DejaVu Sans",
            )
        plt.rcParams["font.family"] = font_family
        plt.rcParams["axes.unicode_minus"] = False

    if figure is not None:
        for text_item in figure.findobj(match=mtext.Text):
            text_item.set_fontfamily(font_family)
    return font_family


apply_korean_chart_font()
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


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
            frame.head(5),
            f"✅ 데이터 등록 완료: {Path(file_path).name} / {frame.shape[0]:,}행 × {frame.shape[1]:,}열",
        )
    except Exception as exc:
        return session, None, f"❌ 데이터 등록 실패: {exc}"


def make_data_context(frame: pd.DataFrame, source_filename: str) -> str:
    describe = frame.describe(include="all").transpose().to_string()
    column_profiles = []
    for column in frame.columns:
        series = frame[column]
        samples = series.drop_duplicates().head(8).astype(str).tolist()
        column_profiles.append(
            f"- {column!r}: 고유값 {series.nunique(dropna=False):,}개 / 예시 {samples}"
        )
    context = f"""
[분석 파일]
{source_filename}

[데이터 크기]
{frame.shape}

[컬럼 및 데이터 타입]
{frame.dtypes.to_string()}

[정확한 컬럼명과 고유값 예시]
{chr(10).join(column_profiles)}

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
- 자동 실행 모드에서는 모델이 작성한 Python 코드가 제한된 분석 환경에서 실행될 수 있습니다.
- 사용자가 지정한 컬럼명은 완전히 동일한 이름으로 사용하세요. 접두어가 같거나 의미가 비슷한 다른 컬럼으로 추정·대체하지 마세요.
- 유사한 컬럼명이 여러 개면 제공된 정확한 컬럼명과 고유값 예시를 확인하고, 사용자가 명시한 컬럼만 사용하세요.
- 사용자가 "각 특성별 표"처럼 표를 나누어 요청하면 컬럼마다 별도 DataFrame을 만들고 반복문 안에서 각각 display() 하세요.
- 사용자가 하나의 통합 표를 요청하면 각 컬럼의 결과와 원본 컬럼명을 누적한 뒤 하나의 DataFrame으로 만드세요. 사용자가 요청한 표 분리 방식을 임의로 바꾸지 마세요.
- 차트 글꼴은 실행 환경에서 이미 설정하므로 Malgun Gothic 같은 운영체제 전용 글꼴을 직접 지정하지 마세요.
- seaborn 막대그래프에서 palette를 사용하면 범주 축 변수를 hue에도 지정하고 legend=False로 설정하세요. 단색이면 palette 대신 color를 사용하세요.
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
    """인덱스·컬럼·셀 값을 Gradio/JSON에서 안전한 표로 정규화합니다."""
    table = value.to_frame() if isinstance(value, pd.Series) else value.copy()
    index = table.index
    is_default_index = (
        isinstance(index, pd.RangeIndex)
        and index.name is None
        and index.start == 0
        and index.step == 1
        and index.stop == len(table)
    )
    if not is_default_index:
        existing_names = {str(column) for column in table.columns}
        safe_names = []
        for position, name in enumerate(index.names, 1):
            candidate = str(name) if name is not None else f"index_{position}"
            while candidate in existing_names or candidate in safe_names:
                candidate += "_index"
            safe_names.append(candidate)
        table.index = table.index.set_names(safe_names)
        table = table.reset_index()

    flattened_columns = []
    used_names = set()
    for column in table.columns:
        if isinstance(column, tuple):
            base_name = " | ".join(str(part) for part in column if str(part)) or "column"
        else:
            base_name = str(column)
        candidate = base_name
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        flattened_columns.append(candidate)
    table.columns = flattened_columns

    def json_safe_cell(cell):
        if cell is None or cell is pd.NA or cell is pd.NaT:
            return None
        if isinstance(cell, np.generic):
            return json_safe_cell(cell.item())
        if isinstance(cell, (dt.datetime, dt.date, dt.time, pd.Timestamp, pd.Timedelta)):
            return str(cell)
        if isinstance(cell, float):
            return cell if math.isfinite(cell) else None
        if isinstance(cell, (str, int, bool)):
            return cell
        if isinstance(cell, (list, tuple, set, dict)):
            return json.dumps(cell, ensure_ascii=False, default=str)
        try:
            if bool(pd.isna(cell)):
                return None
        except (TypeError, ValueError):
            pass
        return str(cell)

    for column in table.columns:
        table[column] = table[column].map(json_safe_cell)
    return table


def captured_value_table(value: Any) -> pd.DataFrame | None:
    """모델이 display()한 표 또는 컬럼별 빈도 사전을 DataFrame으로 변환합니다."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_frame() if isinstance(value, pd.Series) else value
    if isinstance(value, dict) and value:
        rows = []
        for feature, counts in value.items():
            if isinstance(counts, pd.Series):
                items = counts.items()
            elif isinstance(counts, dict):
                items = counts.items()
            else:
                return None
            for unique_value, count in items:
                rows.append({"feature": feature, "unique_value": unique_value, "count": count})
        return pd.DataFrame(rows)
    return None


def table_payload(table: pd.DataFrame, title: str | None = None) -> dict[str, Any]:
    """세션 상태에는 DataFrame 대신 JSON 직렬화 가능한 값만 저장합니다."""
    payload = {
        "headers": list(table.columns),
        "data": table.values.tolist(),
    }
    if title:
        payload["title"] = title
    return payload


def table_payload_to_markdown(
    payload: dict[str, Any],
    max_rows: int = 500,
    max_columns: int = 50,
) -> str:
    """Colab에서도 안정적으로 보이도록 실행 표를 Markdown으로 변환합니다."""
    headers = list(payload.get("headers", []))[:max_columns]
    rows = list(payload.get("data", []))

    def escape_cell(value: Any) -> str:
        if value is None:
            return ""
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r\n", "<br>")
            .replace("\n", "<br>")
            .replace("\r", "<br>")
        )

    title = str(payload.get("title") or "실행 표 결과")
    safe_title = escape_cell(title).replace("*", "\\*").replace("_", "\\_")
    if not headers:
        return f"**{safe_title}**\n\n표시할 열이 없습니다."

    displayed_rows = rows[:max_rows]
    lines = [
        f"**{safe_title}**",
        "",
        "| " + " | ".join(escape_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in displayed_rows:
        row_values = list(row)[: len(headers)]
        row_values.extend([""] * (len(headers) - len(row_values)))
        lines.append("| " + " | ".join(escape_cell(value) for value in row_values) + " |")

    notes = []
    if len(rows) > max_rows:
        notes.append(f"전체 {len(rows):,}행 중 처음 {max_rows:,}행만 표시했습니다.")
    total_columns = len(payload.get("headers", []))
    if total_columns > max_columns:
        notes.append(f"전체 {total_columns:,}열 중 처음 {max_columns:,}열만 표시했습니다.")
    if notes:
        lines.extend(["", "⚠️ " + " ".join(notes)])
    return "\n".join(lines)


def displayable_wide_table(table: pd.DataFrame, max_columns: int = 100):
    """초광폭 결과는 원본 의미를 바꾸지 않고 안전한 표시 폭으로 제한합니다."""
    if table.shape[1] <= max_columns:
        return table, ""

    limited = table.iloc[:, :max_columns].copy()
    note = (
        f"⚠️ 원본 결과가 {table.shape[1]:,}열이어서 화면에는 처음 "
        f"{max_columns}열만 표시합니다. 분석 조건을 더 좁혀 주세요."
    )
    return limited, note


def execute_python_blocks(
    answer: str,
    frame: pd.DataFrame,
    required_columns: list[str] | None = None,
):
    """제한된 분석 환경에서 모델의 Python 블록을 실행합니다."""
    blocks = re.findall(r"```python\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        return frame, "코드 블록이 없어 자동 실행하지 않았습니다.", None, []

    required_columns = required_columns or []
    used_column_names = set()
    for raw_code in blocks:
        parsed = ast.parse(raw_code)
        used_column_names.update(
            node.value for node in ast.walk(parsed)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in frame.columns
        )
    missing_columns = [column for column in required_columns if column not in used_column_names]
    if missing_columns:
        raise RuntimeError(
            "컬럼명 검증 실패: 요청한 컬럼을 생성 코드가 정확히 사용하지 않았습니다: "
            + ", ".join(repr(column) for column in missing_columns)
        )

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
        with warnings.catch_warnings(), contextlib.redirect_stdout(stdout):
            warnings.filterwarnings(
                "ignore",
                message=r"(?s).*Passing `palette` without assigning `hue`.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Glyph .* missing from font.*",
                category=UserWarning,
            )
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
            figure = plt.figure(fig_num)
            apply_korean_chart_font(figure)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Glyph .* missing from font.*",
                    category=UserWarning,
                )
                figure.savefig(chart_path, dpi=150, bbox_inches="tight")
            chart_paths.append(str(chart_path))
            plt.close(fig_num)

    updated_frame = namespace.get("df_clean")
    if not isinstance(updated_frame, pd.DataFrame):
        raise RuntimeError("실행 결과의 df_clean이 DataFrame이 아닙니다.")

    result_tables = []
    text_outputs = []
    table_notes = []
    for value in captured:
        captured_table = captured_value_table(value)
        if captured_table is not None:
            display_table, table_note = displayable_wide_table(captured_table)
            result_tables.append(table_for_gradio(display_table))
            if table_note:
                table_notes.append(table_note)
        else:
            text_outputs.append(str(value))
    printed = stdout.getvalue().strip()
    if printed:
        text_outputs.append(printed)
    summary = "✅ Python 코드 자동 실행 완료"
    if text_outputs:
        summary += "\n\n```text\n" + "\n".join(text_outputs)[:10_000] + "\n```"
    if len(result_tables) == 1:
        result_table = result_tables[0]
        summary += f"\n\n표 결과: {result_table.shape[0]:,}행 × {result_table.shape[1]:,}열"
    elif result_tables:
        total_rows = sum(len(table) for table in result_tables)
        summary += f"\n\n표 결과: {len(result_tables):,}개 표 · 총 {total_rows:,}행"
    if table_notes:
        summary += "\n\n" + "\n\n".join(table_notes)
    if chart_paths:
        summary += f"\n\n차트 결과: {len(chart_paths)}개"
    return updated_frame, summary, result_tables, chart_paths


def visible_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history = []
    for message in messages:
        if message.get("hidden") or message.get("role") not in {"user", "assistant"}:
            continue
        kind = message.get("kind", "text")
        if kind == "table":
            payload = message.get("content")
            if isinstance(payload, pd.DataFrame):
                payload = table_payload(table_for_gradio(payload))
            else:
                payload = payload if isinstance(payload, dict) else {}
            content = table_payload_to_markdown(payload)
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


def requested_column_names(question: str, columns) -> list[str]:
    """겹치는 컬럼명은 가장 긴 정확한 일치부터 안전하게 추출합니다."""
    occupied_spans: list[tuple[int, int]] = []
    requested = []
    for column in sorted((str(item) for item in columns), key=len, reverse=True):
        matched = False
        for match in re.finditer(re.escape(column), question):
            span = match.span()
            if any(span[0] >= start and span[1] <= end for start, end in occupied_spans):
                continue
            occupied_spans.append(span)
            matched = True
        if matched:
            requested.append(column)
    return requested


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
사용자가 명시한 컬럼명은 철자까지 완전히 동일하게 사용하세요.
접두어가 같거나 의미가 비슷한 다른 컬럼으로 추정·대체하지 마세요.
결과는 result_df 같은 별도 변수에 저장하고 display()로 표시하세요.
사용자가 "각 특성별 표"처럼 별도 표를 요청하면 컬럼별 DataFrame을 만들고 반복문 안에서 각각 display() 하세요.
사용자가 하나의 통합 표를 요청한 경우에만 각 컬럼 결과를 누적하여 하나의 result_df로 합치세요.
여러 display() 결과도 화면에 순서대로 표시되므로 마지막 표만 남기려고 결과를 덮어쓰지 마세요.
차트 글꼴은 앱에서 설정하므로 Malgun Gothic 같은 운영체제 전용 글꼴을 지정하지 마세요.
seaborn barplot에서 palette를 쓰면 범주 축과 같은 변수를 hue에 지정하고 legend=False로 설정하세요. 단색 차트는 color를 사용하세요.
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
        execution_summary, result_tables, chart_paths = "", [], []
        if mode_name == "파이썬 코드 자동 실행":
            try:
                required_columns = requested_column_names(question, frame.columns)
                updated_frame, execution_summary, result_tables, chart_paths = execute_python_blocks(
                    answer, frame, required_columns=required_columns
                )
                session["dataframe"] = updated_frame
                if execution_summary.startswith("✅"):
                    session["messages"].append({
                        "role": "assistant",
                        "content": execution_summary,
                        "include_in_context": False,
                    })
                    for result_table in result_tables:
                        if len(result_tables) > 1 and len(result_table.columns) > 0:
                            table_title = f"Feature: {result_table.columns[0]}"
                        else:
                            table_title = "실행 표 결과"
                        session["messages"].append({
                            "role": "assistant",
                            "content": table_payload(result_table, title=table_title),
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
        dataset_file = gr.File(
            label="분석 데이터 등록",
            file_types=[".csv", ".xlsx", ".xls"],
            type="filepath",
            height=110,
        )
        previous_file = gr.File(label="기존대화 등록", type="filepath", height=110)

    load_button = gr.Button("데이터 불러오기", variant="primary")
    register_button = gr.Button("기존대화 불러오기")
    with gr.Accordion("데이터 미리보기 표시/숨기기", open=False):
        preview = gr.Dataframe(label="데이터 미리보기 (최대 5행)", interactive=False)
    chatbot = gr.Chatbot(label="분석 대화", height=650, elem_id="analysis-chatbot")
    status = gr.Markdown("준비됨")

    with gr.Row():
        question = gr.Textbox(label="질문", placeholder="전시 데이터에 관한 질문을 입력하세요.", lines=3, scale=4)
        image = gr.Image(label="이미지 첨부", type="filepath", height=110, scale=1)

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
    running_in_colab = bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_GPU"))
    demo.launch(inbrowser=not running_in_colab, debug=running_in_colab, css=APP_CSS)
