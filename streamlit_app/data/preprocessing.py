from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


PAGE_SIZE = 20
DATE_COMPONENT_LABELS = {
    "year_month": "년-월",
    "month": "월",
    "day": "일",
    "hour": "시간",
}


class PreprocessingError(ValueError):
    """Raised when a requested preprocessing operation is invalid."""


@dataclass(frozen=True)
class OperationResult:
    frame: pd.DataFrame
    affected_rows: int
    message: str


def missing_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "변수명": [str(column) for column in frame.columns],
            "결측 개수": [int(frame[column].isna().sum()) for column in frame.columns],
            "결측률(%)": [round(float(frame[column].isna().mean() * 100), 2) for column in frame.columns],
        }
    )


def fill_missing_values(
    frame: pd.DataFrame,
    column: str,
    method: str,
    value: Any | None = None,
) -> OperationResult:
    result = frame.copy(deep=True)
    if column not in result.columns:
        raise PreprocessingError(f"{column!r} 변수를 찾을 수 없습니다.")
    mask = result[column].isna()
    affected = int(mask.sum())
    if affected == 0:
        return OperationResult(result, 0, "처리할 결측값이 없습니다.")

    if method == "특정값":
        if value is None or (isinstance(value, str) and not value.strip()):
            raise PreprocessingError("결측값을 대체할 특정값을 입력하세요.")
        replacement = _coerce_for_series(value, result[column])
    elif method in {"평균값", "중앙값"}:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise PreprocessingError("평균값과 중앙값은 수치형 변수에만 사용할 수 있습니다.")
        replacement = result[column].mean() if method == "평균값" else result[column].median()
        if pd.isna(replacement):
            raise PreprocessingError("유효한 수치가 없어 대체값을 계산할 수 없습니다.")
    elif method == "해당 행 삭제":
        result = result.loc[~mask].copy()
        return OperationResult(result, affected, f"{affected:,}개 행을 삭제했습니다.")
    else:
        raise PreprocessingError("지원하지 않는 결측값 처리방법입니다.")

    result.loc[mask, column] = replacement
    return OperationResult(result, affected, f"결측값 {affected:,}개를 {replacement!r}(으)로 대체했습니다.")


def _display_value(value: Any) -> str:
    if _is_missing_scalar(value):
        return "<결측값>"
    return str(value)


def _is_missing_scalar(value: Any) -> bool:
    try:
        missing = pd.isna(value)
        return bool(missing) if np.isscalar(missing) else False
    except (TypeError, ValueError):
        return False


def _string_normal_form(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def noise_candidates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Detect explainable string noise and low-frequency category candidates."""
    if column not in frame.columns:
        raise PreprocessingError(f"{column!r} 변수를 찾을 수 없습니다.")
    series = frame[column].dropna()
    if series.empty:
        return pd.DataFrame(columns=["원본 값", "개수", "탐지 근거"])

    counts = series.value_counts(dropna=False)
    normalized_groups: dict[str, list[Any]] = {}
    for value in counts.index:
        if isinstance(value, str):
            normalized_groups.setdefault(_string_normal_form(value), []).append(value)

    rows = []
    rare_limit = max(1, int(len(series) * 0.01))
    categorical_enough = int(series.nunique()) <= max(100, int(len(series) * 0.5))
    for value, count in counts.items():
        reasons = []
        if isinstance(value, str):
            if not value.strip():
                reasons.append("빈 문자열/공백")
            if value != re.sub(r"\s+", " ", value).strip():
                reasons.append("앞뒤·중복 공백")
            if re.search(r"[\x00-\x1f\x7f]", value):
                reasons.append("제어문자")
            variants = normalized_groups.get(_string_normal_form(value), [])
            if len(variants) > 1:
                reasons.append("공백·대소문자 정규화 시 중복")
        if categorical_enough and count <= rare_limit:
            reasons.append(f"저빈도 값(전체의 1% 이하, {int(count):,}건)")
        if reasons:
            rows.append({"원본 값": value, "개수": int(count), "탐지 근거": ", ".join(reasons)})
    return pd.DataFrame(rows, columns=["원본 값", "개수", "탐지 근거"])


def unique_value_counts(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame.columns:
        raise PreprocessingError(f"{column!r} 변수를 찾을 수 없습니다.")
    counts = frame[column].value_counts(dropna=False)
    return pd.DataFrame(
        {"값": counts.index.tolist(), "표시값": [_display_value(v) for v in counts.index], "개수": counts.values.astype(int)}
    )


def paginate(table: pd.DataFrame, page: int, page_size: int = PAGE_SIZE) -> tuple[pd.DataFrame, int]:
    total_pages = max(1, (len(table) + page_size - 1) // page_size)
    safe_page = min(max(int(page), 1), total_pages)
    start = (safe_page - 1) * page_size
    return table.iloc[start : start + page_size].copy(), total_pages


def replace_selected_value(
    frame: pd.DataFrame,
    column: str,
    old_value: Any,
    new_value: Any,
) -> OperationResult:
    result = frame.copy(deep=True)
    if column not in result.columns:
        raise PreprocessingError(f"{column!r} 변수를 찾을 수 없습니다.")
    if new_value is None or (isinstance(new_value, str) and not new_value.strip()):
        raise PreprocessingError("변경할 특정값을 입력하세요.")
    mask = result[column].isna() if _is_missing_scalar(old_value) else result[column].eq(old_value)
    affected = int(mask.sum())
    replacement = _coerce_for_series(new_value, result[column])
    result.loc[mask, column] = replacement
    return OperationResult(result, affected, f"{affected:,}개 값을 {replacement!r}(으)로 변경했습니다.")


def _coerce_for_series(value: Any, series: pd.Series) -> Any:
    if pd.api.types.is_bool_dtype(series):
        lowered = str(value).strip().casefold()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        raise PreprocessingError("불리언 변수에는 true 또는 false를 입력하세요.")
    if pd.api.types.is_integer_dtype(series):
        numeric = float(value)
        if not numeric.is_integer():
            raise PreprocessingError("정수형 변수에는 정수를 입력하세요.")
        return int(numeric)
    if pd.api.types.is_float_dtype(series):
        return float(value)
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise PreprocessingError("날짜 형식으로 변환할 수 없는 값입니다.")
        return parsed
    return value


def date_column_candidates(frame: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        if non_null.empty:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            success_rate = 100.0
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
            success_rate = float(parsed.notna().mean() * 100)
            if success_rate < threshold * 100:
                continue
        else:
            continue
        rows.append({"변수명": str(column), "날짜 변환 성공률(%)": round(success_rate, 2)})
    return pd.DataFrame(rows, columns=["변수명", "날짜 변환 성공률(%)"])


def split_date_components(
    frame: pd.DataFrame,
    column: str,
    components: Iterable[str],
) -> OperationResult:
    selected = list(dict.fromkeys(components))
    if not selected:
        raise PreprocessingError("분리할 날짜 요소를 하나 이상 선택하세요.")
    unknown = set(selected) - set(DATE_COMPONENT_LABELS)
    if unknown:
        raise PreprocessingError(f"지원하지 않는 날짜 요소입니다: {sorted(unknown)}")

    result = frame.copy(deep=True)
    parsed = pd.to_datetime(result[column], errors="coerce", format="mixed")
    valid = int(parsed.notna().sum())
    if valid == 0:
        raise PreprocessingError("유효한 날짜값을 찾을 수 없습니다.")
    if "year_month" in selected:
        result[f"{column}_년월"] = parsed.dt.strftime("%Y년-%m월")
    if "month" in selected:
        result[f"{column}_월"] = parsed.dt.strftime("%m월")
    if "day" in selected:
        result[f"{column}_일"] = parsed.dt.strftime("%d일")
    if "hour" in selected:
        result[f"{column}_시간"] = parsed.dt.strftime("%H시")
    return OperationResult(result, valid, f"날짜값 {valid:,}건에서 {len(selected)}개 파생변수를 생성했습니다.")


def comparison_summary(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("행 수", len(before), len(after)),
        ("변수 수", before.shape[1], after.shape[1]),
        ("전체 데이터 셀", before.size, after.size),
        ("전체 결측 개수", int(before.isna().sum().sum()), int(after.isna().sum().sum())),
        ("중복 행 수", int(before.duplicated().sum()), int(after.duplicated().sum())),
        ("메모리 사용량(bytes)", int(before.memory_usage(deep=True).sum()), int(after.memory_usage(deep=True).sum())),
    ]
    return pd.DataFrame(
        [{"항목": name, "전처리 이전": old, "전처리 이후": new, "변화": new - old} for name, old, new in metrics]
    )


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="df_clean")
    return buffer.getvalue()
