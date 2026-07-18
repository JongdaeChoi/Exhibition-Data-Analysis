from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from visualization.models import Aggregation, ChartSpec, ChartType


class VisualizationDataError(ValueError):
    """Raised when selected data cannot support the requested chart."""


def semantic_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "날짜형"
    if pd.api.types.is_numeric_dtype(series):
        return "수치형"
    non_null = series.dropna()
    if not non_null.empty:
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if float(parsed.notna().mean()) >= 0.8:
            return "날짜형"
    return "범주형"


def variable_type_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "변수명": str(column),
                "분석 타입": semantic_type(frame[column]),
                "pandas 타입": str(frame[column].dtype),
                "유효값": int(frame[column].notna().sum()),
                "Unique Value": int(frame[column].nunique(dropna=False)),
            }
            for column in frame.columns
        ]
    )


def _validate_columns(frame: pd.DataFrame, spec: ChartSpec) -> None:
    selected = [spec.x, spec.y, spec.group, spec.value_column, *spec.variables]
    missing = [column for column in selected if column and column not in frame.columns]
    if missing:
        raise VisualizationDataError("선택한 변수를 찾을 수 없습니다: " + ", ".join(dict.fromkeys(missing)))
    if spec.value_column and spec.aggregation in {Aggregation.SUM, Aggregation.MEAN}:
        if not pd.api.types.is_numeric_dtype(frame[spec.value_column]):
            raise VisualizationDataError("합계·평균의 집계 대상은 수치형 변수여야 합니다.")
    if spec.aggregation == Aggregation.VALID_COUNT and not spec.value_column and spec.chart_type != ChartType.MULTI_VARIABLE:
        raise VisualizationDataError("유효값 개수에는 결측치를 확인할 X2/X3 변수가 필요합니다.")
    if spec.chart_type == ChartType.CORRELATION_HEATMAP:
        non_numeric = [column for column in spec.variables if not pd.api.types.is_numeric_dtype(frame[column])]
        if non_numeric:
            raise VisualizationDataError("상관 히트맵은 수치형 변수만 사용할 수 있습니다: " + ", ".join(non_numeric))
    if spec.chart_type == ChartType.MULTI_VARIABLE:
        types = {semantic_type(frame[column]) for column in spec.variables}
        if len(types) != 1:
            raise VisualizationDataError("다변수 비교는 동일한 데이터 유형의 변수만 선택할 수 있습니다.")
        if spec.aggregation in {Aggregation.SUM, Aggregation.MEAN} and types != {"수치형"}:
            raise VisualizationDataError("합계·평균 다변수 비교는 수치형 변수만 사용할 수 있습니다.")


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
    for axis, column in (("x", spec.x), ("y", spec.y)):
        if not column:
            continue
        mode = getattr(spec.deep, f"{axis}_axis_mode")
        display = data[column].map(str)
        if mode == "category_select":
            data = data.loc[display.isin(getattr(spec.deep, f"{axis}_selected_categories"))].copy()
        elif mode == "category_range":
            start = getattr(spec.deep, f"{axis}_category_start")
            end = getattr(spec.deep, f"{axis}_category_end")
            ordered = list(dict.fromkeys(display.tolist()))
            if start not in ordered or end not in ordered:
                raise VisualizationDataError(f"{axis.upper()}축 범주 범위를 현재 데이터에서 찾을 수 없습니다.")
            low, high = sorted((ordered.index(start), ordered.index(end)))
            data = data.loc[display.isin(ordered[low : high + 1])].copy()
        elif mode == "date_range":
            parsed = pd.to_datetime(data[column], errors="coerce", format="mixed")
            start = pd.Timestamp(getattr(spec.deep, f"{axis}_date_start"))
            end = pd.Timestamp(getattr(spec.deep, f"{axis}_date_end")) + pd.Timedelta(days=1)
            data = data.loc[parsed.ge(start) & parsed.lt(end)].copy()
    if data.empty:
        raise VisualizationDataError("선택 조건에 사용할 수 있는 데이터가 없습니다.")
    return data


def _aggregate(data: pd.DataFrame, keys: list[str], spec: ChartSpec) -> pd.DataFrame:
    grouped = data.groupby(keys, dropna=False, observed=False, sort=False)
    if spec.aggregation == Aggregation.COUNT:
        result = grouped.size().reset_index(name="값")
    elif spec.aggregation in {Aggregation.VALID_COUNT, Aggregation.RATIO} and spec.value_column:
        result = grouped[spec.value_column].count().reset_index(name="값")
    elif spec.aggregation == Aggregation.RATIO:
        result = grouped.size().reset_index(name="값")
    elif spec.aggregation == Aggregation.SUM:
        result = grouped[spec.value_column].sum().reset_index(name="값")
    else:
        result = grouped[spec.value_column].mean().reset_index(name="값")
    if spec.aggregation == Aggregation.RATIO:
        if spec.ratio_basis == "within_x" and spec.x in result and len(keys) > 1:
            denominator = result.groupby(spec.x, dropna=False)["값"].transform("sum")
        elif spec.ratio_basis == "within_y" and spec.y and spec.y in result and len(keys) > 1:
            denominator = result.groupby(spec.y, dropna=False)["값"].transform("sum")
        else:
            denominator = float(result["값"].sum())
        result["값"] = np.where(np.asarray(denominator) != 0, result["값"] / denominator * 100, 0.0)
    return result


def _apply_category_orders(table: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    result = table.copy()
    sort_columns = []
    for column, requested in spec.category_orders.items():
        if column not in result or not requested:
            continue
        observed = result[column].dropna().astype(str).unique().tolist()
        order = requested + [value for value in observed if value not in requested]
        result[column] = pd.Categorical(result[column].astype(str), categories=order, ordered=True)
        sort_columns.append(column)
    if sort_columns:
        result = result.sort_values(sort_columns, kind="stable")
        for column in sort_columns:
            result[column] = result[column].astype("object")
    return result.reset_index(drop=True)


def _sort_and_limit(table: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    result = table.copy()
    targets = []
    x_target = spec.x if spec.x in result else "구간 중심" if "구간 중심" in result else None
    if spec.advanced.x_sort != "none" and x_target:
        targets.append((x_target, spec.advanced.x_sort == "ascending"))
    y_target = spec.y if spec.y and spec.y in result else "값"
    if spec.advanced.y_sort != "none" and y_target in result and y_target != spec.x:
        targets.append((y_target, spec.advanced.y_sort == "ascending"))
    for column, ascending in reversed(targets):
        try:
            result = result.sort_values(column, ascending=ascending, kind="stable")
        except TypeError:
            order = result[column].astype(str).str.casefold().sort_values(ascending=ascending, kind="stable").index
            result = result.loc[order]
    if spec.advanced.top_n and spec.advanced.element_range != "all":
        if spec.advanced.rank_basis in {"value", "ratio"} and "값" in result:
            result = result.sort_values(
                "값", ascending=spec.advanced.element_range == "bottom", kind="stable"
            )
        selected = result.head(spec.advanced.top_n)
        if spec.advanced.remaining_items == "other" and spec.x in result and len(result) > len(selected):
            remaining = result.loc[~result.index.isin(selected.index)].copy()
            remaining[spec.x] = "기타"
            group_columns = [column for column in result.columns if column != "값"]
            other = remaining.groupby(group_columns, dropna=False, observed=False, sort=False)["값"].sum().reset_index()
            selected = pd.concat([selected, other], ignore_index=True)
        result = selected
    return result.reset_index(drop=True)


def build_bar_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    data = _working_data(frame, spec)
    keys = [spec.x] + ([spec.group] if spec.group else [])
    table = _aggregate(data, keys, spec)
    if spec.advanced.bar_mode == "stacked_100" and spec.group:
        totals = table.groupby(spec.x)["값"].transform("sum")
        table["값"] = np.where(totals.ne(0), table["값"] / totals * 100, 0.0)
    return _sort_and_limit(_apply_category_orders(table, spec), spec)


def build_line_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    data = _working_data(frame, spec)
    keys = [spec.x] + ([spec.group] if spec.group else [])
    table = _sort_and_limit(_apply_category_orders(_aggregate(data, keys, spec), spec), spec)
    if spec.deep.cumulative:
        table["값"] = table.groupby(spec.group)["값"].cumsum() if spec.group else table["값"].cumsum()
    if spec.deep.moving_average:
        window = spec.deep.moving_average
        table["값"] = (
            table.groupby(spec.group)["값"].transform(lambda s: s.rolling(window, min_periods=1).mean())
            if spec.group else table["값"].rolling(window, min_periods=1).mean()
        )
    return table.reset_index(drop=True)


def build_pie_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    table = _apply_category_orders(_aggregate(_working_data(frame, spec), [spec.x], spec), spec)
    ascending = spec.advanced.pie_sort_direction == "ascending"
    if spec.advanced.pie_sort_by == "label":
        order = table[spec.x].astype(str).str.casefold().sort_values(ascending=ascending, kind="stable").index
        table = table.loc[order]
    elif spec.advanced.pie_sort_by == "value":
        table = table.sort_values("값", ascending=ascending, kind="stable")
    if spec.advanced.top_n and spec.advanced.element_range != "all":
        if spec.advanced.rank_basis in {"value", "ratio"}:
            table = table.sort_values(
                "값", ascending=spec.advanced.element_range == "bottom", kind="stable"
            )
        selected = table.head(spec.advanced.top_n)
        if spec.advanced.remaining_items == "other" and len(table) > len(selected):
            selected = pd.concat(
                [selected, pd.DataFrame({spec.x: ["기타"], "값": [table.loc[~table.index.isin(selected.index), "값"].sum()]})],
                ignore_index=True,
            )
        table = selected
    table = table.reset_index(drop=True)
    threshold = spec.advanced.pie_min_ratio
    if threshold > 0 and len(table) > 1:
        total = table["값"].sum()
        small = table["값"] / total * 100 < threshold if total else pd.Series(False, index=table.index)
        if small.any():
            table = pd.concat(
                [table.loc[~small], pd.DataFrame({spec.x: ["기타"], "값": [table.loc[small, "값"].sum()]})],
                ignore_index=True,
            )
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
    bins: int | np.ndarray = spec.advanced.histogram_bins
    if spec.advanced.histogram_bin_width:
        start, end = float(numeric.min()), float(numeric.max())
        bins = np.arange(start, end + spec.advanced.histogram_bin_width, spec.advanced.histogram_bin_width)
        if len(bins) < 2:
            bins = np.asarray([start, start + spec.advanced.histogram_bin_width])
    counts, edges = np.histogram(numeric, bins=bins, weights=weights, density=spec.advanced.histogram_density)
    if spec.aggregation == Aggregation.RATIO and counts.sum():
        counts = counts / counts.sum() * 100
    table = pd.DataFrame(
        {"구간 시작": edges[:-1], "구간 끝": edges[1:], "구간 중심": (edges[:-1] + edges[1:]) / 2, "값": counts}
    )
    if mapping is not None:
        table.attrs["category_mapping"] = mapping
    return _sort_and_limit(table, spec)


def _convert_axes(table: pd.DataFrame, columns: list[str], spec: ChartSpec) -> dict[str, pd.DataFrame]:
    mappings = {}
    for column in columns:
        if pd.api.types.is_datetime64_any_dtype(table[column]):
            table[column] = pd.to_datetime(table[column]).map(pd.Timestamp.toordinal).astype(float)
        elif not pd.api.types.is_numeric_dtype(table[column]):
            requested = spec.category_orders.get(column, [])
            observed = list(dict.fromkeys(table[column].astype(str).tolist()))
            categories = requested + [value for value in observed if value not in requested]
            categorical = pd.Categorical(table[column].astype(str), categories=categories, ordered=True)
            mappings[column] = pd.DataFrame({"범주": categories, "수치 인덱스": range(len(categories))})
            table[column] = categorical.codes.astype(float)
    return mappings


def build_scatter_plot_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    data = _working_data(frame, spec)
    columns = [spec.x, spec.y] + ([spec.group] if spec.group else [])
    table = data[columns].copy()
    table.insert(0, "행 번호", data.index.to_numpy())
    table.attrs["category_mappings"] = _convert_axes(table, [spec.x, spec.y], spec)
    return _sort_and_limit(table, spec)


def build_grouped_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    return _sort_and_limit(
        _apply_category_orders(_aggregate(_working_data(frame, spec), [spec.x, spec.y], spec), spec), spec
    )


def build_scatter_bubble_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    data = _working_data(frame, spec)
    converted = data.copy()
    mappings = _convert_axes(converted, [spec.x, spec.y], spec)
    table = _aggregate(converted, [spec.x, spec.y] + ([spec.group] if spec.group else []), spec)
    if spec.deep.normalize and table["값"].max() != table["값"].min():
        table["값"] = (table["값"] - table["값"].min()) / (table["값"].max() - table["값"].min())
    table.attrs["category_mappings"] = mappings
    return _sort_and_limit(table, spec)


def build_heatmap_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    return _sort_and_limit(
        _apply_category_orders(_aggregate(_working_data(frame, spec), [spec.x, spec.y], spec), spec), spec
    )


def build_multi_variable_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for column in spec.variables:
        series = frame[column]
        if spec.aggregation == Aggregation.COUNT:
            value = total
        elif spec.aggregation == Aggregation.VALID_COUNT:
            value = int(series.notna().sum())
        elif spec.aggregation == Aggregation.RATIO:
            value = float(series.notna().sum() / total * 100) if total else 0.0
        elif spec.aggregation == Aggregation.SUM:
            value = float(pd.to_numeric(series, errors="coerce").sum())
        else:
            value = float(pd.to_numeric(series, errors="coerce").mean())
        rows.append({"변수": column, "값": value, "분석 타입": semantic_type(series)})
    return pd.DataFrame(rows)


def build_correlation_statistics(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    matrix = frame[spec.variables].apply(pd.to_numeric, errors="coerce").corr()
    table = matrix.rename_axis("변수1").reset_index().melt(id_vars="변수1", var_name="변수2", value_name="값")
    table.attrs["correlation_matrix"] = matrix
    return table


BUILDERS = {
    ChartType.BAR: build_bar_statistics,
    ChartType.LINE: build_line_statistics,
    ChartType.MULTI_VARIABLE: build_multi_variable_statistics,
    ChartType.PIE: build_pie_statistics,
    ChartType.HISTOGRAM: build_histogram_statistics,
    ChartType.SCATTER_PLOT: build_scatter_plot_statistics,
    ChartType.GROUPED_BAR: build_grouped_statistics,
    ChartType.STACKED_BAR: build_grouped_statistics,
    ChartType.SCATTER_BUBBLE: build_scatter_bubble_statistics,
    ChartType.HEATMAP: build_heatmap_statistics,
    ChartType.CORRELATION_HEATMAP: build_correlation_statistics,
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
