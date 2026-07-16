from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from visualization.models import Aggregation, ChartSpec, ChartType


class VisualizationDataError(ValueError):
    """Raised when the selected data cannot support a requested chart."""


def variable_type_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            semantic = "날짜형"
        elif pd.api.types.is_numeric_dtype(series):
            semantic = "수치형"
        elif pd.api.types.is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            semantic = "범주형"
        else:
            non_null = series.dropna()
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed") if not non_null.empty else None
            if parsed is not None and float(parsed.notna().mean()) >= 0.8:
                semantic = "날짜형"
            else:
                semantic = "범주형"
        rows.append(
            {
                "변수명": str(column),
                "분석 타입": semantic,
                "pandas 타입": str(series.dtype),
                "유효값": int(series.notna().sum()),
                "Unique Value": int(series.nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows)


def _validate_columns(frame: pd.DataFrame, spec: ChartSpec) -> None:
    selected = [spec.x, spec.y, spec.group, spec.value_column]
    missing = [column for column in selected if column and column not in frame.columns]
    if missing:
        raise VisualizationDataError("선택한 변수를 찾을 수 없습니다: " + ", ".join(missing))
    if spec.value_column and not pd.api.types.is_numeric_dtype(frame[spec.value_column]):
        raise VisualizationDataError("합계·평균의 집계 대상은 수치형 변수여야 합니다.")


def _working_data(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    columns = list(dict.fromkeys(c for c in [spec.x, spec.y, spec.group, spec.value_column] if c))
    data = frame[columns].copy()
    if spec.advanced.include_missing:
        for column in columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                data[column] = data[column].astype("object").where(data[column].notna(), "<결측값>")
    else:
        required = [spec.x]
        if spec.y:
            required.append(spec.y)
        if spec.group:
            required.append(spec.group)
        if spec.value_column and spec.aggregation in {Aggregation.SUM, Aggregation.MEAN}:
            required.append(spec.value_column)
        data = data.dropna(subset=required)
    if data.empty:
        raise VisualizationDataError("선택 조건에 사용할 수 있는 데이터가 없습니다.")
    return data


def _aggregate(data: pd.DataFrame, keys: list[str], spec: ChartSpec) -> pd.DataFrame:
    if spec.aggregation in {Aggregation.COUNT, Aggregation.RATIO}:
        result = data.groupby(keys, dropna=False, observed=False).size().reset_index(name="값")
    elif spec.aggregation == Aggregation.SUM:
        result = data.groupby(keys, dropna=False, observed=False)[spec.value_column].sum().reset_index(name="값")
    else:
        result = data.groupby(keys, dropna=False, observed=False)[spec.value_column].mean().reset_index(name="값")
    if spec.aggregation == Aggregation.RATIO:
        total = float(result["값"].sum())
        result["값"] = result["값"] / total * 100 if total else 0.0
    return result


def _sort_and_limit(table: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    result = table.copy()
    if spec.advanced.sort != "none" and "값" in result:
        result = result.sort_values("값", ascending=spec.advanced.sort == "ascending")
    if spec.advanced.top_n:
        result = result.head(spec.advanced.top_n)
    return result.reset_index(drop=True)


def build_bar_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    data = _working_data(frame, spec)
    keys = [spec.x] + ([spec.group] if spec.group else [])
    table = _aggregate(data, keys, spec)
    if spec.advanced.bar_mode == "stacked_100" and spec.group:
        totals = table.groupby(spec.x)["값"].transform("sum")
        table["값"] = np.where(totals.ne(0), table["값"] / totals * 100, 0.0)
    return _sort_and_limit(table, spec)


def build_line_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    data = _working_data(frame, spec)
    keys = [spec.x] + ([spec.group] if spec.group else [])
    table = _aggregate(data, keys, spec).sort_values(spec.x)
    if spec.deep.cumulative:
        table["값"] = table.groupby(spec.group)["값"].cumsum() if spec.group else table["값"].cumsum()
    if spec.deep.moving_average:
        window = spec.deep.moving_average
        table["값"] = (
            table.groupby(spec.group)["값"].transform(lambda s: s.rolling(window, min_periods=1).mean())
            if spec.group
            else table["값"].rolling(window, min_periods=1).mean()
        )
    return table.reset_index(drop=True)


def build_pie_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    table = _sort_and_limit(_aggregate(_working_data(frame, spec), [spec.x], spec), spec)
    threshold = spec.advanced.pie_min_ratio
    if threshold > 0 and len(table) > 1:
        total = table["값"].sum()
        small = table["값"] / total * 100 < threshold if total else pd.Series(False, index=table.index)
        if small.any():
            other = pd.DataFrame({spec.x: ["기타"], "값": [table.loc[small, "값"].sum()]})
            table = pd.concat([table.loc[~small], other], ignore_index=True)
    return table


def build_histogram_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    data = _working_data(frame, spec)
    series = data[spec.x]
    mapping = None
    if not pd.api.types.is_numeric_dtype(series):
        categories = pd.Categorical(series.astype(str), categories=pd.unique(series.astype(str)), ordered=True)
        numeric = pd.Series(categories.codes, index=series.index, dtype=float)
        mapping = pd.DataFrame({"범주": categories.categories, "수치 인덱스": range(len(categories.categories))})
    else:
        numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()
    numeric = numeric[valid]
    if numeric.empty:
        raise VisualizationDataError("히스토그램으로 변환할 유효한 값이 없습니다.")
    weights = None
    if spec.aggregation == Aggregation.SUM:
        weights = pd.to_numeric(data.loc[valid, spec.value_column], errors="coerce").fillna(0)
    counts, edges = np.histogram(
        numeric,
        bins=spec.advanced.histogram_bins,
        weights=weights,
        density=spec.advanced.histogram_density,
    )
    table = pd.DataFrame(
        {
            "구간 시작": edges[:-1],
            "구간 끝": edges[1:],
            "구간 중심": (edges[:-1] + edges[1:]) / 2,
            "값": counts,
        }
    )
    if mapping is not None:
        table.attrs["category_mapping"] = mapping
    return table


def build_scatter_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    data = _working_data(frame, spec)
    converted = data.copy()
    mappings: dict[str, pd.DataFrame] = {}
    for column in [spec.x, spec.y]:
        if not pd.api.types.is_numeric_dtype(converted[column]):
            categories = pd.Categorical(converted[column].astype(str))
            mappings[column] = pd.DataFrame({"범주": categories.categories, "수치 인덱스": range(len(categories.categories))})
            converted[column] = categories.codes.astype(float)
    keys = [spec.x, spec.y] + ([spec.group] if spec.group else [])
    table = _aggregate(converted, keys, spec)
    if spec.deep.normalize and table["값"].max() != table["값"].min():
        table["값"] = (table["값"] - table["값"].min()) / (table["값"].max() - table["값"].min())
    table.attrs["category_mappings"] = mappings
    return table


def build_heatmap_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    data = _working_data(frame, spec)
    table = _aggregate(data, [spec.x, spec.y], spec)
    if spec.deep.normalize:
        totals = table.groupby(spec.x)["값"].transform("sum")
        table["값"] = np.where(totals.ne(0), table["값"] / totals * 100, 0.0)
    return table


BUILDERS = {
    ChartType.BAR: build_bar_statistics,
    ChartType.LINE: build_line_statistics,
    ChartType.PIE: build_pie_statistics,
    ChartType.HISTOGRAM: build_histogram_statistics,
    ChartType.SCATTER_BUBBLE: build_scatter_statistics,
    ChartType.HEATMAP: build_heatmap_statistics,
}


def build_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    _validate_columns(frame, spec)
    return BUILDERS[spec.chart_type](frame, spec)


def json_safe_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    safe = table.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(
            lambda value: None
            if value is None or (isinstance(value, float) and not np.isfinite(value))
            else value.item()
            if isinstance(value, np.generic)
            else str(value)
            if isinstance(value, (pd.Timestamp, pd.Timedelta))
            else value
        )
    return safe.to_dict(orient="records")
