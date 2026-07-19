from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class InsightCodeError(ValueError):
    """Raised when generated analysis code is unsafe or cannot be executed."""


@dataclass
class CodeExecutionResult:
    frame: pd.DataFrame
    outputs: list[dict[str, Any]]


_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
)
_FORBIDDEN_NAMES = {
    "__builtins__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "exit",
}
_FORBIDDEN_ATTRIBUTES = {
    "__class__",
    "__dict__",
    "__globals__",
    "__subclasses__",
    "read_csv",
    "read_excel",
    "read_json",
    "read_pickle",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_html",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
}


def _strip_code_fence(code: str) -> str:
    value = code.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```python", "```py"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def validate_generated_code(code: str) -> str:
    value = _strip_code_fence(code)
    if not value:
        raise InsightCodeError("실행할 Python 코드가 없습니다.")
    if len(value) > 12_000:
        raise InsightCodeError("생성된 Python 코드가 허용 길이를 초과했습니다.")
    try:
        tree = ast.parse(value, mode="exec")
    except SyntaxError as exc:
        raise InsightCodeError(f"Python 코드 문법이 올바르지 않습니다: {exc.msg}") from exc
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise InsightCodeError(f"안전한 실행 환경에서 지원하지 않는 구문입니다: {type(node).__name__}")
        if isinstance(node, ast.Name) and (
            node.id in _FORBIDDEN_NAMES or node.id.startswith("__")
        ):
            raise InsightCodeError(f"사용할 수 없는 이름입니다: {node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr in _FORBIDDEN_ATTRIBUTES or node.attr.startswith("__")
        ):
            raise InsightCodeError(f"사용할 수 없는 함수 또는 속성입니다: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in {"abs", "all", "any", "bool", "dict", "display", "enumerate", "float", "int", "len", "list", "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip"}:
                raise InsightCodeError(f"허용되지 않은 함수 호출입니다: {node.func.id}")
    return value


def _serialize_output(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        limited = value.head(500).copy()
        limited.columns = [str(column) for column in limited.columns]
        limited = limited.astype(object).where(pd.notna(limited), None)
        for column in limited.columns:
            limited[column] = limited[column].map(
                lambda item: item.item()
                if isinstance(item, np.generic)
                else item.isoformat()
                if isinstance(item, (pd.Timestamp, pd.Timedelta))
                else item
            )
        return {
            "type": "dataframe",
            "columns": list(limited.columns),
            "records": limited.to_dict(orient="records"),
            "row_count": int(len(value)),
            "truncated": len(value) > len(limited),
        }
    if isinstance(value, pd.Series):
        return _serialize_output(value.rename(value.name or "value").reset_index())
    if isinstance(value, np.generic):
        value = value.item()
    return {"type": "text", "value": str(value)[:20_000]}


def execute_generated_code(
    code: str,
    original: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    allow_mutation: bool,
) -> CodeExecutionResult:
    validated = validate_generated_code(code)
    protected_original = original.copy(deep=True)
    working = clean.copy(deep=True)
    outputs: list[dict[str, Any]] = []

    def display(value: Any) -> None:
        outputs.append(_serialize_output(value))

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "df": protected_original,
        "df_clean": working,
        "pd": pd,
        "np": np,
        "display": display,
    }
    try:
        exec(compile(validated, "<insight-chat>", "exec"), namespace, namespace)
    except Exception as exc:
        raise InsightCodeError(f"생성된 Python 코드 실행에 실패했습니다: {exc}") from exc

    candidate = namespace.get("df_clean")
    if not isinstance(candidate, pd.DataFrame):
        raise InsightCodeError("실행 후 df_clean이 DataFrame 형식이 아닙니다.")
    if not protected_original.equals(original):
        raise InsightCodeError("원본 df를 변경하는 코드는 허용되지 않습니다.")
    if not allow_mutation and not candidate.equals(clean):
        raise InsightCodeError("조회 요청에서 df_clean을 수정할 수 없습니다.")
    if not outputs:
        for name in ("result_df", "summary_df", "df_result"):
            if name in namespace:
                outputs.append(_serialize_output(namespace[name]))
                break
    if not outputs:
        outputs.append({"type": "text", "value": "코드는 실행되었지만 표시할 결과가 없습니다."})
    return CodeExecutionResult(frame=candidate.copy(deep=True), outputs=outputs)
