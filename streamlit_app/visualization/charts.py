from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.ticker import MultipleLocator

from visualization.models import ChartSpec


def _palette(name: str, count: int) -> list:
    try:
        return list(sns.color_palette(name, max(count, 1)))
    except ValueError:
        return list(sns.color_palette("viridis", max(count, 1)))


def _format_value(value: float, spec: ChartSpec) -> str:
    suffix = "%" if spec.aggregation.value == "ratio" or spec.advanced.bar_mode == "stacked_100" else spec.advanced.unit
    try:
        return f"{value:{spec.advanced.number_format}}{suffix}"
    except (ValueError, TypeError):
        return f"{value:,.1f}{suffix}"


def _smooth_coordinates(x: np.ndarray, y: np.ndarray, curvature: float) -> tuple[np.ndarray, np.ndarray]:
    if curvature <= 0 or len(x) < 3:
        return x, y
    dense_x: list[float] = []
    dense_y: list[float] = []
    for index in range(len(x) - 1):
        t = np.linspace(0.0, 1.0, 24, endpoint=index == len(x) - 2)
        y0 = y[max(index - 1, 0)]
        y1 = y[index]
        y2 = y[index + 1]
        y3 = y[min(index + 2, len(y) - 1)]
        catmull = 0.5 * (
            (2 * y1)
            + (-y0 + y2) * t
            + (2 * y0 - 5 * y1 + 4 * y2 - y3) * t**2
            + (-y0 + 3 * y1 - 3 * y2 + y3) * t**3
        )
        linear = y1 + (y2 - y1) * t
        dense_x.extend((x[index] + (x[index + 1] - x[index]) * t).tolist())
        dense_y.extend(((1 - curvature) * linear + curvature * catmull).tolist())
    return np.asarray(dense_x), np.asarray(dense_y)


def _apply_common(ax: Axes, spec: ChartSpec) -> None:
    ax.set_title(spec.title or f"{spec.x} {spec.chart_type.value}", fontsize=spec.advanced.title_size, fontweight="bold")
    ax.set_xlabel(spec.x_label or spec.x, fontsize=spec.advanced.axis_size)
    ax.set_ylabel(spec.y_label or ("비율(%)" if spec.aggregation.value == "ratio" else "값"), fontsize=spec.advanced.axis_size)
    ax.tick_params(axis="x", labelrotation=spec.advanced.tick_rotation, labelsize=spec.advanced.axis_size)
    ax.tick_params(axis="y", labelsize=spec.advanced.axis_size)
    if spec.advanced.grid:
        ax.grid(True, axis=spec.advanced.grid_axis, alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)
    if spec.deep.reference_line is not None:
        ax.axhline(spec.deep.reference_line, color="#475569", linestyle="--", linewidth=1.2)
    if spec.deep.x_min is not None or spec.deep.x_max is not None:
        ax.set_xlim(left=spec.deep.x_min, right=spec.deep.x_max)
    if spec.deep.y_min is not None or spec.deep.y_max is not None:
        ax.set_ylim(bottom=spec.deep.y_min, top=spec.deep.y_max)
    if spec.deep.x_tick_interval is not None:
        ax.xaxis.set_major_locator(MultipleLocator(spec.deep.x_tick_interval))
    if spec.deep.y_tick_interval is not None:
        ax.yaxis.set_major_locator(MultipleLocator(spec.deep.y_tick_interval))
    if spec.deep.x_log:
        ax.set_xscale("log")
    if spec.deep.y_log:
        ax.set_yscale("log")
    if spec.deep.invert_x:
        ax.invert_xaxis()
    if spec.deep.invert_y:
        ax.invert_yaxis()


def render_bar(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    group = spec.group if spec.group and spec.group in table else None
    horizontal = spec.advanced.orientation == "horizontal"
    if group:
        pivot = table.pivot_table(index=spec.x, columns=group, values="값", aggfunc="sum", fill_value=0, sort=False)
        colors = _palette(spec.advanced.palette, pivot.shape[1])
        pivot.plot(
            kind="barh" if horizontal else "bar",
            stacked=spec.advanced.bar_mode in {"stacked", "stacked_100"},
            ax=ax,
            color=colors,
            alpha=spec.advanced.alpha,
            edgecolor=spec.advanced.edge_color,
            linewidth=spec.advanced.edge_width,
        )
        if spec.show_values:
            stacked = spec.advanced.bar_mode in {"stacked", "stacked_100"}
            for container in ax.containers:
                if isinstance(container, BarContainer):
                    values = [bar.get_width() if horizontal else bar.get_height() for bar in container]
                    labels = [_format_value(value, spec) if value else "" for value in values]
                    ax.bar_label(
                        container,
                        labels=labels,
                        label_type="center" if stacked else "edge",
                        padding=0 if stacked else 3,
                        fontsize=8,
                    )
    else:
        labels = table[spec.x].astype(str)
        values = table["값"].to_numpy()
        if horizontal:
            bars = ax.barh(labels, values, color=spec.advanced.base_color, alpha=spec.advanced.alpha, edgecolor=spec.advanced.edge_color, linewidth=spec.advanced.edge_width)
        else:
            bars = ax.bar(labels, values, color=spec.advanced.base_color, alpha=spec.advanced.alpha, edgecolor=spec.advanced.edge_color, linewidth=spec.advanced.edge_width)
        if spec.show_values:
            ax.bar_label(bars, labels=[_format_value(v, spec) for v in values], padding=3, fontsize=8)
    _apply_common(ax, spec)
    if group and spec.advanced.legend:
        ax.legend(loc=spec.advanced.legend_location, fontsize=8)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()


def render_line(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    groups = table.groupby(spec.group, dropna=False) if spec.group else [(None, table)]
    colors = _palette(spec.advanced.palette, table[spec.group].nunique() if spec.group else 1)
    categorical_labels = None
    if spec.advanced.line_curvature > 0 and not pd.api.types.is_numeric_dtype(table[spec.x]):
        categorical_labels = list(dict.fromkeys(table[spec.x].astype(str).tolist()))
        category_positions = {label: position for position, label in enumerate(categorical_labels)}
    for color, (name, part) in zip(colors, groups):
        x_values = part[spec.x]
        y_values = part["값"].to_numpy()
        line_color = color if spec.group else spec.advanced.base_color
        if spec.advanced.line_curvature > 0:
            x_plot = (
                np.asarray([category_positions[str(value)] for value in x_values], dtype=float)
                if categorical_labels is not None
                else pd.to_numeric(x_values, errors="coerce").to_numpy(dtype=float)
            )
            valid = np.isfinite(x_plot) & np.isfinite(y_values)
            x_plot, y_plot = x_plot[valid], y_values[valid]
            smooth_x, smooth_y = _smooth_coordinates(x_plot, y_plot, spec.advanced.line_curvature)
            ax.plot(
                smooth_x,
                smooth_y,
                label=str(name) if name is not None else None,
                color=line_color,
                linestyle=spec.advanced.line_style,
                linewidth=spec.advanced.line_width,
                alpha=spec.advanced.alpha,
            )
            ax.plot(
                x_plot,
                y_plot,
                linestyle="",
                marker=spec.advanced.marker,
                markersize=spec.advanced.marker_size,
                color=line_color,
                alpha=spec.advanced.alpha,
            )
        else:
            x_plot, y_plot = x_values, y_values
            smooth_x, smooth_y = x_values, y_values
            ax.plot(
                x_values,
                y_values,
                label=str(name) if name is not None else None,
                color=line_color,
                linestyle=spec.advanced.line_style,
                linewidth=spec.advanced.line_width,
                marker=spec.advanced.marker,
                markersize=spec.advanced.marker_size,
                alpha=spec.advanced.alpha,
            )
        if spec.advanced.area_fill:
            try:
                ax.fill_between(smooth_x, smooth_y, alpha=0.14, color=line_color)
            except (TypeError, ValueError):
                # Categorical axes are positioned at 0..n-1 by Matplotlib.
                ax.fill_between(range(len(part)), y_values, alpha=0.14, color=line_color)
        if spec.show_values:
            for x, y in zip(x_plot, y_plot):
                ax.annotate(_format_value(y, spec), (x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    if categorical_labels is not None:
        ax.set_xticks(range(len(categorical_labels)), categorical_labels)
    _apply_common(ax, spec)
    if spec.group and spec.advanced.legend:
        ax.legend(loc=spec.advanced.legend_location, fontsize=8)


def render_pie(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    values = table["값"].to_numpy()
    labels = table[spec.x].astype(str).tolist()
    colors = _palette(spec.advanced.palette, len(table))
    wedgeprops = {"width": 0.45, "edgecolor": spec.advanced.edge_color, "linewidth": spec.advanced.edge_width} if spec.advanced.donut else {"edgecolor": spec.advanced.edge_color, "linewidth": spec.advanced.edge_width}
    ax.pie(
        values,
        labels=labels if not spec.advanced.legend else None,
        colors=colors,
        startangle=spec.advanced.pie_start_angle,
        autopct="%1.1f%%" if spec.show_values else None,
        shadow=spec.advanced.pie_shadow,
        wedgeprops=wedgeprops,
        textprops={"fontsize": 8},
    )
    ax.set_title(spec.title or f"{spec.x} 구성", fontsize=spec.advanced.title_size, fontweight="bold")
    if spec.advanced.legend:
        ax.legend(labels, loc=spec.advanced.legend_location, fontsize=8)


def render_histogram(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    widths = table["구간 끝"] - table["구간 시작"]
    bars = ax.bar(
        table["구간 중심"],
        table["값"],
        width=widths * 0.92,
        color=spec.advanced.base_color,
        alpha=spec.advanced.alpha,
        edgecolor=spec.advanced.edge_color,
        linewidth=spec.advanced.edge_width,
    )
    if spec.show_values:
        ax.bar_label(bars, labels=[_format_value(v, spec) for v in table["값"]], padding=2, fontsize=7)
    if spec.deep.show_mean:
        weighted_mean = np.average(table["구간 중심"], weights=np.maximum(table["값"], 0)) if table["값"].sum() else table["구간 중심"].mean()
        ax.axvline(weighted_mean, color="#D97706", linestyle="--", label="평균")
    if spec.deep.show_median:
        total = float(table["값"].sum())
        median = (
            table.loc[table["값"].cumsum().ge(total / 2).idxmax(), "구간 중심"]
            if total > 0
            else table["구간 중심"].median()
        )
        ax.axvline(median, color="#7C3AED", linestyle=":", label="중앙값")
    _apply_common(ax, spec)
    if (spec.deep.show_mean or spec.deep.show_median) and spec.advanced.legend:
        ax.legend(fontsize=8)


def render_scatter_bubble(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    x = pd.to_numeric(table[spec.x], errors="coerce")
    y = pd.to_numeric(table[spec.y], errors="coerce")
    size_values = pd.to_numeric(table["값"], errors="coerce").fillna(0)
    scale = spec.advanced.scatter_size
    if size_values.max() > size_values.min():
        sizes = 20 + (size_values - size_values.min()) / (size_values.max() - size_values.min()) * scale
    else:
        sizes = np.full(len(table), scale)
    if spec.deep.jitter:
        rng = np.random.default_rng(42)
        x = x + rng.normal(0, spec.deep.jitter, len(x))
        y = y + rng.normal(0, spec.deep.jitter, len(y))
    if spec.group and spec.group in table:
        codes, uniques = pd.factorize(table[spec.group].astype(str))
        scatter = ax.scatter(x, y, s=sizes, c=codes, cmap=spec.advanced.palette, alpha=spec.advanced.alpha, edgecolors=spec.advanced.edge_color, linewidths=spec.advanced.edge_width)
        if spec.advanced.legend:
            handles = [plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)), label=name) for i, name in enumerate(uniques)]
            ax.legend(handles=handles, loc=spec.advanced.legend_location, fontsize=8)
    else:
        ax.scatter(x, y, s=sizes, color=spec.advanced.base_color, alpha=spec.advanced.alpha, edgecolors=spec.advanced.edge_color, linewidths=spec.advanced.edge_width)
    valid = x.notna() & y.notna()
    if spec.advanced.trendline and valid.sum() >= 2:
        coefficients = np.polyfit(x[valid], y[valid], 1)
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, coefficients[0] * x_line + coefficients[1], color="#D97706", linestyle="--", linewidth=1.5)
    if spec.deep.show_correlation and valid.sum() >= 2:
        corr = float(np.corrcoef(x[valid], y[valid])[0, 1])
        ax.text(0.02, 0.98, f"r = {corr:.3f}", transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#CBD5E1"})
    _apply_common(ax, spec)
    ax.set_ylabel(spec.y_label or spec.y)


def render_heatmap(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    matrix = table.pivot_table(index=spec.y, columns=spec.x, values="값", aggfunc="sum", fill_value=0, sort=False)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=spec.advanced.heatmap_cmap,
        annot=spec.advanced.heatmap_annotate,
        fmt=".1f",
        cbar=spec.advanced.heatmap_colorbar,
        linewidths=spec.advanced.heatmap_linewidth,
        linecolor=spec.advanced.edge_color,
        center=spec.deep.heatmap_center,
    )
    _apply_common(ax, spec)
    ax.set_ylabel(spec.y_label or spec.y)


RENDERERS = {
    "bar": render_bar,
    "line": render_line,
    "pie": render_pie,
    "histogram": render_histogram,
    "scatter_bubble": render_scatter_bubble,
    "heatmap": render_heatmap,
}


def render_chart(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    RENDERERS[spec.chart_type.value](ax, table, spec)
