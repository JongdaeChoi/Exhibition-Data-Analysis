from __future__ import annotations

import datetime as dt
import io
import json
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import koreanize_matplotlib  # noqa: F401 - registers bundled NanumGothic for Colab/Linux
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from visualization.charts import render_chart
from visualization.models import Aggregation, ChartSpec, ChartType, FigureSpec
from visualization.statistics import build_statistics, json_safe_records


@dataclass
class ChartArtifact:
    spec: ChartSpec
    statistics: pd.DataFrame
    insight: str


@dataclass
class VisualizationResult:
    figure: Figure
    artifacts: list[ChartArtifact]
    figure_spec: FigureSpec


def automatic_chart_title(*columns: str | None) -> str:
    selected = [column for column in columns if column and column not in {"(없음)"}]
    return " · ".join(dict.fromkeys(selected))


def summarize_artifact(
    table: pd.DataFrame, spec: ChartSpec, language: str = "한국어"
) -> str:
    english = language == "English"
    lines = (
        [f"The statistical table contains {len(table):,} rows."]
        if english
        else [f"통계표는 {len(table):,}개 행으로 구성되었습니다."]
    )
    if "값" in table and not table.empty:
        maximum_index = table["값"].idxmax()
        maximum = float(table.loc[maximum_index, "값"])
        label_columns = [c for c in table.columns if c not in {"값", "구간 시작", "구간 끝", "구간 중심"}]
        label_values = [str(table.loc[maximum_index, c]) for c in label_columns[:2]]
        if english:
            type_labels = {"수치형": "Numeric", "범주형": "Categorical", "날짜형": "Date"}
            label_values = [type_labels.get(value, value) for value in label_values]
        label = " · ".join(label_values) if label_values else (
            "maximum bin" if english else "최대 구간"
        )
        lines.append(
            f"The largest value is {maximum:,.2f} for {label}."
            if english
            else f"가장 큰 값은 {label}의 {maximum:,.2f}입니다."
        )
        if spec.chart_type == ChartType.SCATTER_BUBBLE and spec.y and len(table) >= 2:
            x = pd.to_numeric(table[spec.x], errors="coerce")
            y = pd.to_numeric(table[spec.y], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() >= 2:
                correlation = x[valid].corr(y[valid])
                lines.append(
                    f"The correlation between the two axes is {correlation:.3f}."
                    if english
                    else f"두 축의 상관계수는 {correlation:.3f}입니다."
                )
        elif len(table) > 1:
            total = float(table["값"].sum())
            if total:
                share = maximum / total * 100
                lines.append(
                    f"The largest item represents {share:,.1f}% of the displayed total."
                    if english
                    else f"최대 항목은 표시 합계의 {share:,.1f}%를 차지합니다."
                )
    if spec.aggregation == Aggregation.RATIO:
        basis_labels = (
            {"total": "all data", "within_x": "each X1 group", "within_y": "each X2 group"}
            if english
            else {"total": "전체", "within_x": "X1 내부", "within_y": "X2 내부"}
        )
        basis = basis_labels[spec.ratio_basis]
        lines.append(
            f"The ratio denominator is based on {basis}."
            if english
            else f"비율의 분모 기준은 {basis}입니다."
        )
    if spec.advanced.top_n and len(table) >= spec.advanced.top_n:
        lines.append(
            f"Chart elements are limited to the top {spec.advanced.top_n}."
            if english
            else f"차트 요소는 상위 {spec.advanced.top_n}개로 제한되었습니다."
        )
    return "\n".join(lines[:5])


def build_visualization(
    frame: pd.DataFrame,
    chart_specs: list[ChartSpec],
    figure_spec: FigureSpec,
) -> VisualizationResult:
    expected = figure_spec.rows * figure_spec.columns
    if len(chart_specs) != expected:
        raise ValueError(f"{figure_spec.rows}×{figure_spec.columns} subplot에는 {expected}개 차트 설정이 필요합니다.")
    plt.rcParams["font.family"] = figure_spec.font_family
    plt.rcParams["text.color"] = figure_spec.font_color
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(
        figure_spec.rows,
        figure_spec.columns,
        figsize=(figure_spec.width, figure_spec.height),
        dpi=figure_spec.dpi,
        squeeze=False,
        sharex=figure_spec.share_x,
        sharey=figure_spec.share_y,
        constrained_layout=figure_spec.layout_mode == "constrained",
    )
    fig.patch.set_facecolor(to_rgba(figure_spec.figure_background, figure_spec.figure_alpha))
    if figure_spec.figure_border_width > 0:
        fig.add_artist(
            Rectangle(
                (0, 0), 1, 1, transform=fig.transFigure, fill=False,
                edgecolor=to_rgba(figure_spec.figure_border_color, figure_spec.figure_border_alpha),
                linewidth=figure_spec.figure_border_width,
            )
        )
    artifacts = []
    for index, (ax, spec) in enumerate(zip(axes.flat, chart_specs), 1):
        style_axes = figure_spec.axes_style_scope == "all" or figure_spec.axes_target_index == index
        ax.set_facecolor(
            to_rgba(figure_spec.axes_background, figure_spec.axes_background_alpha)
            if style_axes else "#FFFFFF"
        )
        if style_axes:
            for position, spine in ax.spines.items():
                visible = figure_spec.axes_border_visible and position in figure_spec.axes_border_positions
                spine.set_visible(visible)
                if visible:
                    spine.set_linewidth(figure_spec.axes_border_width)
                    spine.set_color(to_rgba(figure_spec.axes_border_color, figure_spec.axes_border_alpha))
                    spine.set_linestyle(figure_spec.axes_border_style)
        table = build_statistics(frame, spec)
        render_chart(ax, table, spec)
        artifacts.append(ChartArtifact(spec=spec, statistics=table, insight=summarize_artifact(table, spec)))
    if figure_spec.layout_mode in {"basic", "custom"}:
        fig.subplots_adjust(
            left=figure_spec.margin_left, right=figure_spec.margin_right,
            bottom=figure_spec.margin_bottom, top=figure_spec.margin_top,
            wspace=figure_spec.horizontal_space, hspace=figure_spec.vertical_space,
        )
    elif figure_spec.layout_mode == "tight":
        fig.tight_layout()
    return VisualizationResult(figure=fig, artifacts=artifacts, figure_spec=figure_spec)


def figure_to_bytes(result: VisualizationResult, output_format: str) -> bytes:
    fmt = output_format.lower()
    if fmt not in {"png", "jpg", "svg", "pdf"}:
        raise ValueError("PNG, JPG, SVG, PDF 형식만 지원합니다.")
    buffer = io.BytesIO()
    result.figure.savefig(
        buffer,
        format="jpeg" if fmt == "jpg" else fmt,
        dpi=result.figure_spec.dpi,
        bbox_inches="tight",
        transparent=result.figure_spec.transparent,
        metadata=(
            {"Title": result.figure_spec.filename, "Creator": "Streamlit Data Visualization"}
            if result.figure_spec.include_metadata and fmt in {"png", "pdf", "svg"}
            else None
        ),
    )
    return buffer.getvalue()


def source_payload(result: VisualizationResult, source_filename: str) -> dict:
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_filename": source_filename,
        "figure": result.figure_spec.model_dump(mode="json"),
        "charts": [
            {
                "chart_number": index,
                "spec": artifact.spec.model_dump(mode="json"),
                "insight": artifact.insight,
                "statistics": json_safe_records(artifact.statistics),
            }
            for index, artifact in enumerate(result.artifacts, 1)
        ],
    }


def source_payload_bytes(result: VisualizationResult, source_filename: str) -> bytes:
    return json.dumps(source_payload(result, source_filename), ensure_ascii=False, indent=2).encode("utf-8")
