# Generated from 데이터분석_API.ipynb
# Edit the notebook as the source of truth; regenerate this file after notebook changes.

# %% [notebook cell 1]
# @title 1. 환경 설정 및 라이브러리 설치
# Colab 기본 라이브러리(pandas, matplotlib, seaborn)는 업데이트하지 않습니다.
# !pip install -q -U google-genai openpyxl ipywidgets
# !DEBIAN_FRONTEND=noninteractive apt-get install -yqq fonts-nanum
# !fc-cache -fv > /dev/null
# !rm -rf ~/.cache/matplotlib

import base64
import ast
import datetime as dt
import io
import json
import re
import traceback
from pathlib import Path

import ipywidgets as widgets
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from google import genai
from google.genai import types
from google.colab import files, userdata
from IPython.display import HTML, Image, Markdown, clear_output, display

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
if Path(FONT_PATH).exists():
    fm.fontManager.addfont(FONT_PATH)
    FONT_NAME = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rc("font", family=FONT_NAME)
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=FONT_NAME,
                  rc={"axes.unicode_minus": False})
else:
    print("[경고] 나눔바른고딕을 찾지 못했습니다.")

sns.set_palette("viridis")
print("환경 설정을 완료했습니다.")

# %% [notebook cell 2]
# @title 2. Gemini API 인증
api_key = userdata.get("exhibition")
if not api_key:
    raise ValueError("Colab Secrets에 'exhibition' 이름으로 Gemini API 키를 등록하세요.")

client = genai.Client(api_key=api_key)
print("Gemini API 인증을 완료했습니다.")

# %% [notebook cell 3]
# @title 3. 분석 데이터 업로드
print("CSV 또는 Excel 파일을 선택하세요.")
uploaded = files.upload()
if not uploaded:
    raise ValueError("업로드된 파일이 없습니다.")

# 여러 파일을 올린 경우 첫 번째 지원 파일을 사용합니다.
df = None
source_filename = None
for filename, raw in uploaded.items():
    lower = filename.lower()
    try:
        if lower.endswith(".csv"):
            last_error = None
            for encoding in ("utf-8-sig", "cp949", "utf-8"):
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
                    break
                except UnicodeDecodeError as exc:
                    last_error = exc
            if df is None:
                raise last_error
        elif lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            continue
        source_filename = filename
        break
    except Exception as exc:
        print(f"[경고] {filename} 로드 실패: {exc}")

if df is None:
    raise ValueError("읽을 수 있는 CSV/Excel 파일이 없습니다.")

df_clean = df.copy()
print(f"분석 파일: {source_filename} / 크기: {df_clean.shape}")
display(df_clean.head(3))

# %% [notebook cell 4]
# @title 4. 전시 데이터 분석 UI (모델 선택 + 기존대화 등록 + 이미지 첨부)
MODEL_OPTIONS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
)
HISTORY_SCHEMA_VERSION = 2
MAX_CONTEXT_CHARS = 60_000


def make_data_context(frame):
    """데이터 전체를 보내지 않고 구조·통계·샘플만 안전한 크기로 구성합니다."""
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


SYSTEM_PROMPT = f"""
당신은 전시회 전문 데이터 분석 수석 컨설턴트입니다.
아래 데이터 정보를 바탕으로 정확하고 실행 가능한 분석을 제공하세요.
항상 한국어로 답하세요. 데이터에 없는 사실은 추측이라고 명시하세요.
파이썬 코드는 현재 Colab의 df_clean을 사용하고, 한글 폰트 설정 코드는 작성하지 마세요.

[응답 범위 절대 규칙]
- 사용자가 요청한 결과만 답하고, 요청하지 않은 데이터 개요·결측치·기초통계·해석·제언·보고서를 추가하지 마세요.
- 표를 요청하면 표만, 차트를 요청하면 차트 생성 코드만, 설명을 요청하면 설명만 제공하세요.
- df_clean의 실제 계산이 필요한 경우, 아래 요약 정보만으로 계산값을 추정하거나 숫자 표를 만들어 내지 마세요.
- 실제 계산 요청에는 실행 가능한 Python 코드 블록 하나만 작성하세요.
- 코드의 마지막 결과는 print가 아니라 display(result_df)로 출력하세요.
- 사용자가 분석이나 해석을 명시적으로 요청한 경우에만 계산 결과를 해석하세요.
- df_clean은 Colab 메모리에 이미 존재합니다. 샘플 데이터를 만들거나 파일을 다시 읽지 마세요.
- df는 업로드 원본이므로 절대로 수정하거나 재할당하지 마세요.
- df_clean은 전처리용 작업본입니다. 사용자가 요청하면 결측치 처리, 형변환,
  필터링, 컬럼 생성·삭제, 중복 제거 등을 실제 df_clean에 적용하세요.
- df_clean을 임의의 예시·샘플 데이터로 새로 만들거나 파일을 다시 읽어 교체하지 마세요.
- 조회·집계 결과는 result_df, summary_df 또는 df_result 같은 별도 변수에 저장하세요.
- "요약해 줘", "설명해 줘" 같은 요청은 아래 데이터 컨텍스트를 이용해 간결한 한국어 설명으로 답하고 코드를 작성하지 마세요.

{make_data_context(df_clean)}
""".strip()

# SDK와 독립적인 영속 기록. 이미지/차트는 base64로 보존합니다.
app_history = []
chat_session = None
current_model = MODEL_OPTIONS[0]
pending_image = None


def text_part(text):
    return {"type": "text", "data": str(text)}


def image_part(data, mime_type="image/png", name=None, origin="upload"):
    return {
        "type": "image",
        "mime_type": mime_type or "image/png",
        "data": base64.b64encode(bytes(data)).decode("ascii"),
        "name": name,
        "origin": origin,
    }


def sdk_parts(saved_parts):
    converted = []
    for part in saved_parts:
        if part.get("type") == "text":
            converted.append(types.Part.from_text(text=part.get("data", "")))
        elif part.get("type") == "image" and part.get("data"):
            converted.append(types.Part.from_bytes(
                data=base64.b64decode(part["data"]),
                mime_type=part.get("mime_type", "image/png"),
            ))
    return converted


def rebuild_chat(model_name=None):
    """저장 기록으로 선택 모델의 채팅 세션을 다시 만듭니다."""
    global chat_session, current_model
    current_model = model_name or current_model
    history = []
    for message in app_history:
        role = "model" if message.get("role") == "model" else "user"
        parts = sdk_parts(message.get("parts", []))
        if parts:
            history.append(types.Content(role=role, parts=parts))
    chat_session = client.chats.create(
        model=current_model,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        history=history,
    )


def upload_item(value):
    """ipywidgets 7/8의 FileUpload 값 차이를 흡수합니다."""
    if not value:
        return None
    item = next(iter(value.values())) if isinstance(value, dict) else value[0]
    if isinstance(item, dict):
        return {
            "name": item.get("name", "upload"),
            "type": item.get("type", "application/octet-stream"),
            "content": bytes(item.get("content", b"")),
        }
    return {
        "name": getattr(item, "name", "upload"),
        "type": getattr(item, "type", "application/octet-stream"),
        "content": bytes(getattr(item, "content", b"")),
    }


def response_text(response):
    try:
        return response.text or "(텍스트 응답 없음)"
    except Exception:
        return "(응답 텍스트를 읽을 수 없습니다.)"


def execute_python_blocks(answer):
    """응답의 Python 블록을 실행하고 새 차트를 PNG 기록으로 반환합니다."""
    charts = []
    blocks = re.findall(r"```python\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    for index, code in enumerate(blocks, 1):
        # 모델이 폰트 전역 설정을 덮어쓰는 것만 제거합니다.
        code = re.sub(r"^.*plt\.rcParams\[.*$", "", code, flags=re.MULTILINE)
        code = re.sub(r"^.*plt\.rc\(.*font.*$", "", code, flags=re.MULTILINE)
        tree = ast.parse(code)
        def target_root_name(target):
            while isinstance(target, (ast.Subscript, ast.Attribute)):
                target = target.value
            return target.id if isinstance(target, ast.Name) else None

        def expression_uses_name(expression, names):
            return any(isinstance(item, ast.Name) and item.id in names
                       for item in ast.walk(expression))

        for node in ast.walk(tree):
            # df는 원본이므로 보호하고, df_clean은 원본/작업본에서 파생된 전처리만 허용합니다.
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                roots = {target_root_name(target) for target in targets}
                if "df" in roots:
                    raise RuntimeError(
                        "안전 차단: 업로드 원본 df를 수정하려고 했습니다. df_clean을 사용하세요."
                    )
                if "df_clean" in roots and any(
                    isinstance(target, ast.Name) and target.id == "df_clean"
                    for target in targets
                ):
                    value = getattr(node, "value", None)
                    if value is not None and not expression_uses_name(value, {"df", "df_clean"}):
                        raise RuntimeError(
                            "안전 차단: df_clean을 원본과 무관한 샘플 데이터로 교체하려고 했습니다."
                        )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if target_root_name(node.func.value) == "df" and any(
                    kw.arg == "inplace" and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True for kw in node.keywords
                ):
                    raise RuntimeError(
                        "안전 차단: 업로드 원본 df를 inplace 방식으로 수정하려고 했습니다."
                    )
        before = set(plt.get_fignums())
        # exec()는 노트북과 달리 마지막 표현식(df.describe() 등)을 자동 출력하지 않습니다.
        # 앞 문장은 실행하고 마지막 표현식은 eval하여 반환값이 있을 때 자동 표시합니다.
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            final_expression = ast.Expression(tree.body.pop().value)
            ast.fix_missing_locations(tree)
            ast.fix_missing_locations(final_expression)
            if tree.body:
                exec(compile(tree, "<gemini-code>", "exec"), globals())
            result = eval(compile(final_expression, "<gemini-code>", "eval"), globals())
            if result is not None:
                display(result)
        else:
            exec(compile(tree, "<gemini-code>", "exec"), globals())
        after = set(plt.get_fignums())
        for fig_num in sorted(after - before):
            buffer = io.BytesIO()
            plt.figure(fig_num).savefig(buffer, format="png", dpi=150,
                                        bbox_inches="tight")
            charts.append(image_part(
                buffer.getvalue(), "image/png",
                name=f"generated_chart_{index}_{fig_num}.png",
                origin="executed_chart",
            ))
    return charts


def render_history():
    output_area.clear_output()
    with output_area:
        for message in app_history:
            if message.get("display") is False:
                continue
            label = "👤 기획자" if message.get("role") == "user" else "🤖 분석가"
            for part in message.get("parts", []):
                if part.get("type") == "text":
                    display(Markdown(f"**{label}:**\n\n{part.get('data', '')}"))
                elif part.get("type") == "image" and part.get("data"):
                    display(Image(data=base64.b64decode(part["data"]), width=650))
            display(Markdown("---"))


model_selector = widgets.Dropdown(
    options=MODEL_OPTIONS, value=current_model, description="Gemini 모델:",
    layout=widgets.Layout(width="390px"),
    style={"description_width": "110px"},
)
mode_selector = widgets.Dropdown(
    options=("💬 일반 분석", "▶️ 파이썬 코드 자동 실행"),
    value="💬 일반 분석", description="모드:",
    layout=widgets.Layout(width="310px"),
)
text_input = widgets.Textarea(
    placeholder="전시 데이터에 관한 질문을 입력하세요...",
    description="👤 기획자:",
    layout=widgets.Layout(width="100%", height="90px"),
    style={"description_width": "90px"},
)
btn_send = widgets.Button(description="질문 전송", button_style="info", icon="paper-plane")
btn_clear = widgets.Button(description="출력 지우기", button_style="warning", icon="eraser")
btn_new = widgets.Button(description="새 대화", icon="plus")
btn_save = widgets.Button(description="JSON 저장", button_style="success", icon="save")
history_upload = widgets.FileUpload(
    accept="", multiple=False, description="기존대화 등록",
    layout=widgets.Layout(width="150px"),
)
image_upload = widgets.FileUpload(accept="image/*", multiple=False, description="이미지 첨부")
status_html = widgets.HTML("<span style='color:gray'>✅ 준비됨</span>")
output_area = widgets.Output(layout=widgets.Layout(
    width="100%", max_height="520px", overflow="auto",
    border="1px solid #ddd", padding="12px",
))
output_area.add_class("gemini-chat-output")


def scroll_to_latest():
    """출력 영역의 스크롤을 가장 최근 질문·응답 위치로 이동합니다."""
    # 이 UI 셀만 교체·실행해도 동작하도록 함수 내부에서 가져옵니다.
    from IPython.display import Javascript as _Javascript
    display(_Javascript("""
    setTimeout(() => {
      const widgets = document.querySelectorAll('.gemini-chat-output');
      if (!widgets.length) return;
      const widget = widgets[widgets.length - 1];
      const candidates = [widget, ...widget.querySelectorAll('*')];
      for (const element of candidates) {
        if (element.scrollHeight > element.clientHeight) {
          element.scrollTop = element.scrollHeight;
        }
      }
    }, 150);
    """))


def on_model_changed(change):
    if change.get("name") != "value" or change.get("new") == change.get("old"):
        return
    try:
        rebuild_chat(change["new"])
        status_html.value = f"<b>✅ 모델 변경 완료: {change['new']} (기존 대화 유지)</b>"
    except Exception as exc:
        status_html.value = f"<span style='color:red'>모델 변경 실패: {exc}</span>"


def on_image_uploaded(change):
    global pending_image
    pending_image = upload_item(change.get("new"))
    if pending_image:
        status_html.value = f"<b>🖼️ 첨부됨: {pending_image['name']}</b>"


def on_send_clicked(_):
    global pending_image
    question = text_input.value.strip()
    if not question:
        status_html.value = "<span style='color:#b36b00'>질문을 입력하세요.</span>"
        return
    btn_send.disabled = True
    status_html.value = "<b style='color:#0078d7'>⏳ Gemini가 분석 중입니다...</b>"
    try:
        user_parts = [text_part(question)]
        if pending_image:
            user_parts.append(image_part(
                pending_image["content"], pending_image["type"],
                pending_image["name"], "upload",
            ))
        execute_mode = mode_selector.value.startswith("▶️")
        request_rule = """

[이번 응답 형식]
요청하지 않은 서론, 데이터 개요, 결측치, 기초통계, 해석, 제언은 절대 추가하지 마세요.
사용자가 요구한 결과만 간결하게 제공하세요.
"""
        if execute_mode:
            request_rule += """
표·차트·집계·변환처럼 실제 계산이 필요한 요청이면 Python 코드 블록 하나만 반환하세요.
단, 요약·설명·해석 요청이면 코드를 작성하지 말고 요청한 설명만 간결하게 답하세요.
코드를 작성할 경우 df와 df_clean은 이미 존재하므로 샘플 데이터 생성이나 파일 재로딩을
하지 마세요. df는 원본이므로 수정하지 말고, 사용자가 전처리를 요청하면 실제 df_clean을
수정하세요. 조회·집계 결과는 result_df 같은 별도 변수에 저장하고 display()로 표시하세요.
"""
        api_parts = [dict(part) for part in user_parts]
        api_parts[0] = text_part(question + request_rule)
        response = chat_session.send_message(sdk_parts(api_parts))
        answer = response_text(response)
        has_python_code = bool(re.search(
            r"```python\s*.*?```", answer, flags=re.DOTALL | re.IGNORECASE
        ))
        run_generated_code = execute_mode and has_python_code
        app_history.append({"role": "user", "parts": user_parts})
        # 코드 블록을 먼저 보여준 다음 실제 실행 결과를 이어서 표시합니다.
        app_history.append({
            "role": "model",
            "parts": [text_part(answer)],
            "display": True,
        })
        render_history()
        scroll_to_latest()

        if run_generated_code:
            with output_area:
                try:
                    charts = execute_python_blocks(answer)
                    if charts:
                        # 차트는 '사용자가 제공한 실행 결과'로 기록하여 다음 질문과 복원 세션에 전달합니다.
                        chart_message = {
                            "role": "user",
                            "parts": [text_part("[이전 분석 코드의 로컬 실행 결과 차트]")] + charts,
                        }
                        app_history.append(chart_message)
                        rebuild_chat(current_model)
                        display(Markdown(f"*생성 차트 {len(charts)}개를 대화 기록에 저장했습니다.*"))
                except Exception:
                    display(Markdown("**코드 실행 오류**\n```\n" + traceback.format_exc() + "\n```"))
            scroll_to_latest()
        text_input.value = ""
        pending_image = None
        status_html.value = "<span style='color:gray'>✅ 답변 완료</span>"
    except Exception as exc:
        status_html.value = f"<span style='color:red'>오류: {exc}</span>"
    finally:
        btn_send.disabled = False


def on_save_clicked(_):
    payload = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": current_model,
        "source_filename": source_filename,
        "data_signature": {
            "shape": list(df_clean.shape),
            "columns": [str(c) for c in df_clean.columns],
        },
        "history": app_history,
    }
    filename = f"exhibition_chat_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    Path(filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    files.download(filename)
    status_html.value = f"<b style='color:green'>✅ 저장 완료: {filename}</b>"


def normalize_old_history(data):
    """v6의 최상위 리스트 형식과 v7 객체 형식을 모두 지원합니다."""
    history = data if isinstance(data, list) else data.get("history", [])
    normalized = []
    for message in history:
        role = "model" if message.get("role") == "model" else "user"
        parts = []
        for part in message.get("parts", []):
            kind = part.get("type")
            if kind in ("text", "image") and part.get("data") is not None:
                parts.append(dict(part))
        if parts:
            restored_message = {"role": role, "parts": parts}
            if message.get("display") is False:
                restored_message["display"] = False
            normalized.append(restored_message)
    return normalized


def decode_readable_file(content):
    """일반 텍스트 파일을 Colab에서 자주 쓰는 인코딩 순서로 읽습니다."""
    encodings = ["utf-8-sig", "cp949"]
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.insert(1, "utf-16")
    for encoding in encodings:
        try:
            text = content.decode(encoding)
            visible = sum(char.isprintable() or char in "\r\n\t" for char in text)
            if "\x00" not in text and visible / max(len(text), 1) >= 0.9:
                return text
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError("텍스트로 읽을 수 없는 파일입니다. JSON, TXT, MD, CSV 등 읽을 수 있는 파일을 등록하세요.")


def on_history_uploaded(change):
    global app_history
    item = upload_item(change.get("new"))
    if not item:
        return
    try:
        text = decode_readable_file(item["content"])
        try:
            data = json.loads(text)
            restored = normalize_old_history(data)
        except (json.JSONDecodeError, AttributeError, TypeError):
            data, restored = None, []

        if restored:
            app_history = restored
            saved_model = data.get("model") if isinstance(data, dict) else None
            if saved_model in MODEL_OPTIONS:
                model_selector.value = saved_model
            result_message = f"✅ 기존 대화 {len(app_history)}개 메시지 등록 완료"
        else:
            reference_text = text.strip()
            if not reference_text:
                raise ValueError("등록할 내용이 없는 빈 파일입니다.")
            truncated = len(reference_text) > MAX_CONTEXT_CHARS
            reference_text = reference_text[:MAX_CONTEXT_CHARS]
            context = (
                f"[등록된 기존 대화 파일: {item['name']}]\n"
                "다음 내용은 사용자가 등록한 이전 대화입니다. 후속 질문에 답할 때 참고하세요.\n\n"
                f"{reference_text}"
            )
            app_history = [{
                "role": "user",
                "parts": [text_part(context)],
                "display": False,
            }]
            suffix = " (길이 제한으로 일부만 등록)" if truncated else ""
            result_message = f"✅ 기존 대화 파일 등록 완료: {item['name']}{suffix}"

        rebuild_chat(model_selector.value)
        render_history()
        status_html.value = f"<b style='color:green'>{result_message}</b>"
    except Exception as exc:
        status_html.value = f"<span style='color:red'>기존대화 등록 실패: {exc}</span>"


def on_clear_clicked(_):
    output_area.clear_output()
    status_html.value = "<span style='color:gray'>출력만 지웠습니다. 대화 문맥은 유지됩니다.</span>"


def on_new_clicked(_):
    global app_history, pending_image
    app_history = []
    pending_image = None
    rebuild_chat(model_selector.value)
    output_area.clear_output()
    status_html.value = "<span style='color:gray'>✅ 새 대화를 시작했습니다.</span>"


model_selector.observe(on_model_changed, names="value")
image_upload.observe(on_image_uploaded, names="value")
history_upload.observe(on_history_uploaded, names="value")
btn_send.on_click(on_send_clicked)
btn_save.on_click(on_save_clicked)
btn_clear.on_click(on_clear_clicked)
btn_new.on_click(on_new_clicked)

rebuild_chat(current_model)
controls_1 = widgets.HBox([model_selector, mode_selector])
controls_2 = widgets.HBox([btn_send, btn_clear, btn_new, btn_save, history_upload, image_upload])
display(widgets.VBox([controls_1, output_area, text_input, controls_2, status_html]))

