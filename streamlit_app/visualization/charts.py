from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.container import BarContainer
from matplotlib.dates import DateConverter, DateFormatter, DayLocator, MonthLocator, WeekdayLocator, YearLocator, date2num
from matplotlib.patches import Circle, Shadow
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.transforms import offset_copy

from visualization.models import ChartSpec


def _palette(name: str, count: int) -> list:
    try:
        return list(sns.color_palette(name, max(count, 1)))
    except ValueError:
        return list(sns.color_palette("viridis", max(count, 1)))


def _legend_kwargs(spec: ChartSpec) -> dict:
    location = spec.advanced.legend_location
    if location == "outside_right":
        return {"loc": "center left", "bbox_to_anchor": (1.02, 0.5)}
    if location == "outside_bottom":
        return {"loc": "upper center", "bbox_to_anchor": (0.5, -0.14)}
    return {"loc": location}


def _format_value(value: float, spec: ChartSpec) -> str:
    suffix = "%" if spec.aggregation.value == "ratio" or spec.advanced.bar_mode == "stacked_100" else spec.advanced.unit
    try:
        display_value = abs(value) if value < 0 and spec.advanced.label_negative_format == "parentheses" else value
        text = f"{display_value:{spec.advanced.number_format}}{suffix}"
        if value < 0 and spec.advanced.label_negative_format == "parentheses":
            return f"({text})"
        if value > 0 and spec.advanced.label_positive_sign:
            return f"+{text}"
        return text
    except (ValueError, TypeError):
        return f"{value:,.1f}{suffix}"


def _label_offset(spec: ChartSpec, default_x: float = 0.0, default_y: float = 5.0) -> tuple[float, float]:
    if spec.advanced.label_position_mode == "manual":
        return spec.advanced.label_offset_x, spec.advanced.label_offset_y
    return default_x, default_y


def _label_bar_containers(
    ax: Axes,
    containers: list[BarContainer],
    spec: ChartSpec,
    horizontal: bool,
    stacked: bool = False,
) -> None:
    offset_x, offset_y = _label_offset(spec, 0.0, 0.0 if stacked else 3.0)
    for container in containers:
        values = [bar.get_width() if horizontal else bar.get_height() for bar in container]
        labels = [_format_value(value, spec) if value else "" for value in values]
        if spec.advanced.label_position_mode != "manual":
            position = spec.advanced.label_position_mode
            centered = stacked or position in {"inside", "center"}
            ax.bar_label(
                container,
                labels=labels,
                label_type="center" if centered else "edge",
                padding=0 if centered else 3,
                fontsize=spec.advanced.label_font_size,
                fontweight=spec.advanced.label_font_weight,
                color=spec.advanced.label_color,
                alpha=spec.advanced.label_alpha,
                rotation=spec.advanced.label_rotation,
            )
            continue
        for bar, label in zip(container, labels):
            if not label:
                continue
            endpoint = (
                (bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2)
                if horizontal
                else (bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height())
            )
            ax.annotate(
                label,
                endpoint,
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha="left" if horizontal else "center",
                va="center" if horizontal else "bottom",
                fontsize=spec.advanced.label_font_size,
                fontweight=spec.advanced.label_font_weight,
                color=spec.advanced.label_color,
                alpha=spec.advanced.label_alpha,
                rotation=spec.advanced.label_rotation,
            )


def _reference_coordinate(ax: Axes, axis: str, value, kind: str | None) -> float | None:
    ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    target = str(value)
    for tick, label in zip(ticks, labels):
        label_text = label.get_text().strip()
        if label_text == target:
            return float(tick)
        if kind == "date":
            try:
                if pd.to_datetime(label_text).date() == pd.to_datetime(value).date():
                    return float(tick)
            except (TypeError, ValueError):
                pass
    if kind == "numeric":
        return float(value)
    if kind == "date":
        try:
            return float(date2num(pd.to_datetime(value).to_pydatetime()))
        except (TypeError, ValueError):
            return None
    return None


def _apply_reference_lines(ax: Axes, spec: ChartSpec) -> None:
    deep = spec.deep
    if not deep.reference_enabled and not deep.reference_lines:
        return
    color = "#D97706"
    for axis in deep.reference_targets if deep.reference_enabled else []:
        value = getattr(deep, f"{axis}_reference_value")
        kind = getattr(deep, f"{axis}_reference_kind")
        coordinate = _reference_coordinate(ax, axis, value, kind)
        if coordinate is None:
            continue
        line_kwargs = {
            "color": color,
            "linestyle": deep.reference_line_style,
            "linewidth": deep.reference_line_width,
            "alpha": deep.reference_line_alpha,
        }
        if axis == "x":
            ax.axvline(coordinate, **line_kwargs)
            if deep.reference_label:
                ax.text(
                    coordinate, 0.98, deep.reference_label,
                    transform=ax.get_xaxis_transform(), ha="left", va="top",
                    fontsize=deep.reference_label_size, color=color, alpha=deep.reference_label_alpha,
                )
        else:
            ax.axhline(coordinate, **line_kwargs)
            if deep.reference_label:
                ax.text(
                    0.98, coordinate, deep.reference_label,
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=deep.reference_label_size, color=color, alpha=deep.reference_label_alpha,
                )
    for reference in deep.reference_lines:
        for axis in reference.targets:
            coordinate = _reference_coordinate(
                ax, axis, getattr(reference, f"{axis}_value"), getattr(reference, f"{axis}_kind")
            )
            if coordinate is None:
                continue
            kwargs = dict(
                color=reference.color, linestyle=reference.style,
                linewidth=reference.width, alpha=reference.alpha,
            )
            if axis == "x":
                ax.axvline(coordinate, **kwargs)
                if reference.label and reference.label_visible:
                    y, va = {"start": (0.02, "bottom"), "center": (0.5, "center"), "end": (0.98, "top")}[reference.label_position]
                    label_transform = offset_copy(
                        ax.get_xaxis_transform(), fig=ax.figure, x=reference.label_pad, units="points"
                    )
                    ax.text(coordinate, y, reference.label, transform=label_transform,
                            ha="left", va=va, fontsize=reference.label_size,
                            fontweight=reference.label_weight,
                            color=reference.label_color or reference.color, alpha=reference.label_alpha)
            else:
                ax.axhline(coordinate, **kwargs)
                if reference.label and reference.label_visible:
                    x, ha = {"start": (0.02, "left"), "center": (0.5, "center"), "end": (0.98, "right")}[reference.label_position]
                    label_transform = offset_copy(
                        ax.get_yaxis_transform(), fig=ax.figure, y=reference.label_pad, units="points"
                    )
                    ax.text(x, coordinate, reference.label, transform=label_transform,
                            ha=ha, va="bottom", fontsize=reference.label_size,
                            fontweight=reference.label_weight,
                            color=reference.label_color or reference.color, alpha=reference.label_alpha)


def _apply_annotations(ax: Axes, spec: ChartSpec) -> None:
    for note in spec.deep.annotations:
        transform = ax.transAxes if note.coordinate == "axes" else ax.transData
        box = (
            dict(facecolor=note.box_color, alpha=note.box_alpha, edgecolor=note.box_edge_color,
                 linestyle=note.box_line_style)
            if note.box_visible else None
        )
        arrowprops = dict(arrowstyle="->", color=note.color) if note.arrow_visible else None
        ax.annotate(
            note.text, xy=(note.arrow_x, note.arrow_y) if note.arrow_visible else (note.x, note.y),
            xytext=(note.x, note.y), xycoords=transform, textcoords=transform,
            fontsize=note.size, fontweight=note.weight, color=note.color, alpha=note.alpha,
            ha=note.horizontal_alignment, va=note.vertical_alignment, rotation=note.rotation,
            bbox=box, arrowprops=arrowprops,
        )


def _apply_date_ticks(ax: Axes, spec: ChartSpec) -> None:
    frequency_map = {
        "day": DayLocator(), "week": WeekdayLocator(), "month": MonthLocator(),
        "quarter": MonthLocator(interval=3), "year": YearLocator(),
    }
    for axis_name in ("x", "y"):
        frequency = getattr(spec.deep, f"{axis_name}_date_tick_frequency")
        if frequency == "auto":
            continue
        axis = ax.xaxis if axis_name == "x" else ax.yaxis
        date_format = getattr(spec.deep, f"{axis_name}_date_format")
        if isinstance(axis.get_converter(), DateConverter):
            axis.set_major_locator(frequency_map[frequency])
            axis.set_major_formatter(DateFormatter(date_format))
            continue
        labels = ax.get_xticklabels() if axis_name == "x" else ax.get_yticklabels()
        formatted = []
        for label in labels:
            try:
                formatted.append(pd.to_datetime(label.get_text()).strftime(date_format))
            except (TypeError, ValueError):
                formatted.append(label.get_text())
        if axis_name == "x":
            ax.set_xticks(ax.get_xticks(), formatted)
        else:
            ax.set_yticks(ax.get_yticks(), formatted)


def _apply_numeric_tick_format(ax: Axes, spec: ChartSpec) -> None:
    formatters = {
        "integer": lambda value, _: f"{value:.0f}",
        "decimal1": lambda value, _: f"{value:.1f}",
        "decimal2": lambda value, _: f"{value:.2f}",
        "thousands": lambda value, _: f"{value:,.0f}",
        "percent": lambda value, _: f"{value:.1f}%",
    }
    for axis_name in ("x", "y"):
        name = getattr(spec.advanced, f"{axis_name}_tick_number_format")
        if name != "auto":
            axis = ax.xaxis if axis_name == "x" else ax.yaxis
            axis.set_major_formatter(FuncFormatter(formatters[name]))


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
    adv = spec.advanced
    if adv.title_visible:
        ax.set_title(
            spec.title or f"{spec.x} {spec.chart_type.value}", fontsize=adv.title_size,
            fontweight=adv.title_weight, color=adv.title_color, loc=adv.title_location,
            alpha=adv.title_alpha, pad=adv.title_pad,
        )
    else:
        ax.set_title("")
    ax.set_xlabel(
        (spec.x_label or spec.x) if adv.x_label_visible else "", fontsize=adv.x_label_size,
        fontweight=adv.x_label_weight, color=adv.x_label_color, alpha=adv.x_label_alpha,
        rotation=adv.x_label_rotation, labelpad=adv.x_label_pad, loc=adv.x_label_location,
    )
    ax.set_ylabel(
        (spec.y_label or ("비율(%)" if spec.aggregation.value == "ratio" else "값"))
        if adv.y_label_visible else "",
        fontsize=adv.y_label_size, fontweight=adv.y_label_weight, color=adv.y_label_color,
        alpha=adv.y_label_alpha, rotation=adv.y_label_rotation, labelpad=adv.y_label_pad,
        loc=adv.y_label_location,
    )
    ax.tick_params(axis="x", labelrotation=adv.x_tick_rotation or adv.tick_rotation,
                   labelsize=adv.x_tick_size, colors=adv.x_tick_color,
                   labelbottom=adv.x_tick_visible, pad=adv.x_tick_pad)
    ax.tick_params(axis="y", labelrotation=adv.y_tick_rotation,
                   labelsize=adv.y_tick_size, colors=adv.y_tick_color,
                   labelleft=adv.y_tick_visible, pad=adv.y_tick_pad)
    for label in ax.get_xticklabels():
        label.set_fontweight(adv.x_tick_weight)
        label.set_alpha(adv.x_tick_alpha)
    for label in ax.get_yticklabels():
        label.set_fontweight(adv.y_tick_weight)
        label.set_alpha(adv.y_tick_alpha)
    if spec.advanced.grid and (adv.grid_x or adv.grid_y):
        grid_axis = "both" if adv.grid_x and adv.grid_y else "x" if adv.grid_x else "y"
        if adv.grid_which in {"minor", "both"}:
            ax.minorticks_on()
        ax.grid(
            True, axis=grid_axis, which=adv.grid_which, linestyle=spec.advanced.grid_style,
            linewidth=spec.advanced.grid_width, color=spec.advanced.grid_color,
            alpha=spec.advanced.grid_alpha,
        )
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
    _apply_date_ticks(ax, spec)
    _apply_numeric_tick_format(ax, spec)
    _apply_reference_lines(ax, spec)
    _apply_annotations(ax, spec)


def render_bar(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    group = spec.group if spec.group and spec.group in table else None
    horizontal = (spec.advanced.orientation == "horizontal") ^ spec.x_y_swap
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
            width=spec.advanced.bar_width * (1.0 - spec.advanced.bar_gap) * (1.0 - spec.advanced.group_gap),
        )
        if spec.show_values:
            stacked = spec.advanced.bar_mode in {"stacked", "stacked_100"}
            containers = [container for container in ax.containers if isinstance(container, BarContainer)]
            _label_bar_containers(ax, containers, spec, horizontal, stacked)
    else:
        labels = table[spec.x].astype(str)
        values = table["값"].to_numpy()
        if horizontal:
            bars = ax.barh(labels, values, height=spec.advanced.bar_width * (1.0 - spec.advanced.bar_gap), color=spec.advanced.base_color, alpha=spec.advanced.alpha, edgecolor=spec.advanced.edge_color, linewidth=spec.advanced.edge_width)
        else:
            bars = ax.bar(labels, values, width=spec.advanced.bar_width * (1.0 - spec.advanced.bar_gap), color=spec.advanced.base_color, alpha=spec.advanced.alpha, edgecolor=spec.advanced.edge_color, linewidth=spec.advanced.edge_width)
        if spec.show_values:
            _label_bar_containers(ax, [bars], spec, horizontal)
    _apply_common(ax, spec)
    if group and spec.advanced.legend:
        ax.legend(fontsize=8, **_legend_kwargs(spec))
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    if spec.advanced.bar_corner_style == "rounded":
        for patch in ax.patches:
            patch.set_joinstyle("round")


def render_line(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    groups = table.groupby(spec.group, dropna=False) if spec.group else [(None, table)]
    colors = _palette(spec.advanced.palette, table[spec.group].nunique() if spec.group else 1)
    categorical_labels = None
    if spec.advanced.line_curvature > 0 and not pd.api.types.is_numeric_dtype(table[spec.x]):
        categorical_labels = list(dict.fromkeys(table[spec.x].astype(str).tolist()))
        category_positions = {label: position for position, label in enumerate(categorical_labels)}
    for color, (name, part) in zip(colors, groups):
        x_values = part[spec.x]
        physical_date_axis = "y" if spec.x_y_swap else "x"
        if (
            getattr(spec.deep, f"{physical_date_axis}_date_tick_frequency") != "auto"
            or getattr(spec.deep, f"{physical_date_axis}_axis_mode") == "date_range"
        ):
            parsed_x = pd.to_datetime(x_values, errors="coerce", format="mixed")
            if parsed_x.notna().all():
                x_values = parsed_x
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
                smooth_y if spec.x_y_swap else smooth_x,
                smooth_x if spec.x_y_swap else smooth_y,
                label=str(name) if name is not None else None,
                color=line_color,
                linestyle=spec.advanced.line_style,
                linewidth=spec.advanced.line_width,
                alpha=spec.advanced.alpha,
            )
            ax.plot(
                y_plot if spec.x_y_swap else x_plot,
                x_plot if spec.x_y_swap else y_plot,
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
                y_values if spec.x_y_swap else x_values,
                x_values if spec.x_y_swap else y_values,
                label=str(name) if name is not None else None,
                color=line_color,
                linestyle=spec.advanced.line_style,
                linewidth=spec.advanced.line_width,
                marker=spec.advanced.marker,
                markersize=spec.advanced.marker_size,
                alpha=spec.advanced.alpha,
            )
        if spec.advanced.area_fill and not spec.x_y_swap:
            try:
                ax.fill_between(smooth_x, smooth_y, alpha=0.14, color=line_color)
            except (TypeError, ValueError):
                # Categorical axes are positioned at 0..n-1 by Matplotlib.
                ax.fill_between(range(len(part)), y_values, alpha=0.14, color=line_color)
        if spec.show_values:
            offset_x, offset_y = _label_offset(spec)
            for x, y in zip(x_plot, y_plot):
                ax.annotate(
                    _format_value(y, spec), (y, x) if spec.x_y_swap else (x, y), xytext=(offset_x, offset_y),
                    textcoords="offset points", ha="center",
                    fontsize=spec.advanced.label_font_size, fontweight=spec.advanced.label_font_weight,
                    color=spec.advanced.label_color, alpha=spec.advanced.label_alpha,
                    rotation=spec.advanced.label_rotation,
                )
    if categorical_labels is not None:
        ax.set_xticks(range(len(categorical_labels)), categorical_labels)
    _apply_common(ax, spec)
    if spec.x_y_swap:
        ax.set_xlabel(spec.y_label or ("비율(%)" if spec.aggregation.value == "ratio" else "값"))
        ax.set_ylabel(spec.x_label or spec.x)
    if spec.advanced.event_start is not None and spec.advanced.event_end is not None:
        try:
            ax.axvspan(
                spec.advanced.event_start, spec.advanced.event_end,
                color=spec.advanced.event_color, alpha=spec.advanced.event_alpha,
            )
        except (TypeError, ValueError):
            pass
    if spec.group and spec.advanced.legend:
        ax.legend(fontsize=8, **_legend_kwargs(spec))


def render_pie(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    values = table["값"].to_numpy()
    labels = table[spec.x].astype(str).tolist()
    colors = [to_rgba(color, spec.advanced.alpha) for color in _palette(spec.advanced.palette, len(table))]
    outer_radius = 1.0
    wedgeprops = {
        "edgecolor": to_rgba(spec.advanced.edge_color, spec.advanced.pie_edge_alpha),
        "linewidth": spec.advanced.edge_width,
    }
    if spec.advanced.donut:
        outer_radius = spec.advanced.donut_hole_size + spec.advanced.donut_ring_width
        wedgeprops["width"] = spec.advanced.donut_ring_width
    pctdistance = (
        (spec.advanced.donut_hole_size + spec.advanced.donut_ring_width * 0.5) / outer_radius
        if spec.advanced.donut
        else 0.6
    )
    show_label = spec.show_values and spec.advanced.pie_label_mode in {"label", "label_ratio"}
    show_ratio = spec.show_values and spec.advanced.pie_label_mode in {"ratio", "label_ratio"}
    explode = [
        spec.advanced.pie_explode_width if label in spec.advanced.pie_explode_labels else 0.0
        for label in labels
    ]
    pie_result = ax.pie(
        values,
        labels=labels if show_label else None,
        colors=colors,
        startangle=spec.advanced.pie_start_angle,
        autopct=f"%1{spec.advanced.pie_ratio_format}%%" if show_ratio else None,
        pctdistance=pctdistance,
        explode=explode,
        shadow=False,
        wedgeprops=wedgeprops,
        radius=outer_radius,
        labeldistance=1.15 if spec.advanced.label_position_mode == "outside" else 1.0,
        textprops={"fontsize": spec.advanced.label_font_size, "fontweight": spec.advanced.label_font_weight,
                   "color": spec.advanced.label_color, "alpha": spec.advanced.label_alpha,
                   "rotation": spec.advanced.label_rotation},
    )
    if hasattr(pie_result, "wedges"):
        wedges = pie_result.wedges
        pie_texts = list(pie_result.texts)
    else:
        wedges = pie_result[0]
        pie_texts = list(pie_result[1]) + (list(pie_result[2]) if len(pie_result) > 2 else [])
    if spec.advanced.pie_shadow:
        for wedge in wedges:
            shadow = Shadow(
                wedge,
                spec.advanced.pie_shadow_width,
                -spec.advanced.pie_shadow_width,
                facecolor=spec.advanced.pie_shadow_color,
                edgecolor="none",
                alpha=spec.advanced.pie_shadow_alpha,
            )
            shadow.set_zorder(wedge.get_zorder() - 0.1)
            ax.add_patch(shadow)
    if spec.show_values and spec.advanced.label_position_mode == "manual":
        for text in pie_texts:
            text.set_transform(
                offset_copy(
                    ax.transData,
                    fig=ax.figure,
                    x=spec.advanced.label_offset_x,
                    y=spec.advanced.label_offset_y,
                    units="points",
                )
            )
    if spec.advanced.donut:
        ax.add_patch(
            Circle(
                (0, 0),
                radius=spec.advanced.donut_hole_size,
                facecolor=spec.advanced.donut_center_color,
                edgecolor=(
                    to_rgba(
                        spec.advanced.donut_center_border_color,
                        spec.advanced.donut_center_border_alpha,
                    )
                    if spec.advanced.donut_center_border
                    else "none"
                ),
                linewidth=spec.advanced.donut_center_border_width if spec.advanced.donut_center_border else 0,
                zorder=1.1,
            )
        )
    if spec.advanced.title_visible:
        ax.set_title(
            spec.title or f"{spec.x} 구성", fontsize=spec.advanced.title_size,
            fontweight=spec.advanced.title_weight, color=spec.advanced.title_color,
            loc=spec.advanced.title_location, alpha=spec.advanced.title_alpha,
            pad=spec.advanced.title_pad,
        )
    if spec.advanced.legend:
        ax.legend(wedges, labels, fontsize=8, **_legend_kwargs(spec))


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
        _label_bar_containers(ax, [bars], spec, horizontal=False)
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
    if spec.x_y_swap:
        x, y = y, x
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
        scatter = ax.scatter(x, y, s=sizes, marker=spec.advanced.scatter_marker, c=codes, cmap=spec.advanced.palette, alpha=spec.advanced.alpha, edgecolors=spec.advanced.edge_color, linewidths=spec.advanced.edge_width)
        if spec.advanced.legend:
            handles = [plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)), label=name) for i, name in enumerate(uniques)]
            ax.legend(handles=handles, fontsize=8, **_legend_kwargs(spec))
    else:
        ax.scatter(x, y, s=sizes, marker=spec.advanced.scatter_marker, color=spec.advanced.base_color, alpha=spec.advanced.alpha, edgecolors=spec.advanced.edge_color, linewidths=spec.advanced.edge_width)
    valid = x.notna() & y.notna()
    if spec.advanced.trendline and valid.sum() >= 2:
        coefficients = np.polyfit(x[valid], y[valid], 1)
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, coefficients[0] * x_line + coefficients[1], color="#D97706", linestyle="--", linewidth=1.5)
    if spec.deep.show_correlation and valid.sum() >= 2:
        corr = float(np.corrcoef(x[valid], y[valid])[0, 1])
        ax.text(0.02, 0.98, f"r = {corr:.3f}", transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#CBD5E1"})
    mappings = table.attrs.get("category_mappings", {})
    display_x, display_y = (spec.y, spec.x) if spec.x_y_swap else (spec.x, spec.y)
    if display_x in mappings:
        mapping = mappings[display_x]
        ax.set_xticks(mapping["수치 인덱스"], mapping["범주"].astype(str))
    if display_y in mappings:
        mapping = mappings[display_y]
        ax.set_yticks(mapping["수치 인덱스"], mapping["범주"].astype(str))
    _apply_common(ax, spec)
    ax.set_ylabel(spec.y_label or spec.y)


def render_heatmap(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    matrix = table.pivot_table(index=spec.y, columns=spec.x, values="값", aggfunc="sum", fill_value=0, sort=False)
    if spec.x_y_swap:
        matrix = matrix.T
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=spec.advanced.heatmap_cmap,
        annot=spec.advanced.heatmap_annotate,
        fmt=spec.advanced.heatmap_value_format,
        cbar=spec.advanced.heatmap_colorbar,
        linewidths=spec.advanced.heatmap_linewidth,
        linecolor=to_rgba(spec.advanced.heatmap_linecolor, spec.advanced.heatmap_linealpha),
        center=spec.deep.heatmap_center,
    )
    _apply_common(ax, spec)
    ax.set_ylabel(spec.x_label or spec.x if spec.x_y_swap else spec.y_label or spec.y)
    if spec.x_y_swap:
        ax.set_xlabel(spec.y_label or spec.y)


def render_scatter_plot(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    x = pd.to_numeric(table[spec.x], errors="coerce")
    y = pd.to_numeric(table[spec.y], errors="coerce")
    if spec.x_y_swap:
        x, y = y, x
    if spec.deep.jitter:
        rng = np.random.default_rng(42)
        x = x + rng.normal(0, spec.deep.jitter, len(x))
        y = y + rng.normal(0, spec.deep.jitter, len(y))
    ax.scatter(
        x, y, s=spec.advanced.scatter_size, marker=spec.advanced.scatter_marker,
        color=spec.advanced.base_color, alpha=spec.advanced.alpha,
        edgecolors=spec.advanced.edge_color, linewidths=spec.advanced.edge_width,
    )
    valid = x.notna() & y.notna()
    if spec.advanced.trendline and valid.sum() >= 2:
        coefficients = np.polyfit(x[valid], y[valid], 1)
        line_x = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(line_x, coefficients[0] * line_x + coefficients[1], color="#D97706", linestyle="--")
    mappings = table.attrs.get("category_mappings", {})
    display_x, display_y = (spec.y, spec.x) if spec.x_y_swap else (spec.x, spec.y)
    if display_x in mappings:
        mapping = mappings[display_x]
        ax.set_xticks(mapping["수치 인덱스"], mapping["범주"].astype(str))
    if display_y in mappings:
        mapping = mappings[display_y]
        ax.set_yticks(mapping["수치 인덱스"], mapping["범주"].astype(str))
    _apply_common(ax, spec)
    if spec.x_y_swap:
        ax.set_xlabel(spec.y_label or spec.y)
        ax.set_ylabel(spec.x_label or spec.x)


def render_grouped_bar(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    mode = "stacked" if spec.chart_type.value == "stacked_bar" else "grouped"
    proxy = spec.model_copy(
        update={"group": spec.y, "advanced": spec.advanced.model_copy(update={"bar_mode": mode})}
    )
    render_bar(ax, table, proxy)


def render_multi_variable(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    proxy = spec.model_copy(update={"x": "변수", "group": None})
    if spec.comparison_chart == "line":
        render_line(ax, table, proxy)
    else:
        render_bar(ax, table, proxy)


def render_correlation_heatmap(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    matrix = table.attrs.get("correlation_matrix")
    if matrix is None:
        matrix = table.pivot(index="변수1", columns="변수2", values="값")
    sns.heatmap(
        matrix, ax=ax, vmin=-1, vmax=1, center=0,
        cmap=spec.advanced.heatmap_cmap, annot=spec.advanced.heatmap_annotate,
        fmt=spec.advanced.heatmap_value_format, cbar=spec.advanced.heatmap_colorbar,
        linewidths=spec.advanced.heatmap_linewidth,
        linecolor=to_rgba(spec.advanced.heatmap_linecolor, spec.advanced.heatmap_linealpha),
    )
    _apply_common(ax, spec.model_copy(update={"x": "변수", "x_label": "", "y_label": ""}))


RENDERERS = {
    "bar": render_bar,
    "line": render_line,
    "multi_variable": render_multi_variable,
    "pie": render_pie,
    "histogram": render_histogram,
    "scatter_plot": render_scatter_plot,
    "grouped_bar": render_grouped_bar,
    "stacked_bar": render_grouped_bar,
    "scatter_bubble": render_scatter_bubble,
    "heatmap": render_heatmap,
    "correlation_heatmap": render_correlation_heatmap,
}


def render_chart(ax: Axes, table: pd.DataFrame, spec: ChartSpec) -> None:
    RENDERERS[spec.chart_type.value](ax, table, spec)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(spec.advanced.legend_title)
        legend.set_ncols(1 if spec.advanced.legend_direction == "vertical" else max(1, len(legend.texts)))
        legend.get_frame().set_visible(spec.advanced.legend_border_visible or spec.advanced.legend_background_alpha > 0)
        legend.get_frame().set_facecolor(to_rgba(spec.advanced.legend_background, spec.advanced.legend_background_alpha))
        legend.get_frame().set_edgecolor(spec.advanced.legend_border_color)
        legend.get_frame().set_linewidth(spec.advanced.legend_border_width if spec.advanced.legend_border_visible else 0)
        for text in legend.get_texts():
            text.set_color(spec.advanced.legend_color)
            text.set_fontsize(spec.advanced.legend_font_size)
            text.set_fontweight(spec.advanced.legend_font_weight)
            text.set_alpha(spec.advanced.legend_alpha)
