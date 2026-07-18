from __future__ import annotations

import datetime as dt
import io
import json
import re
from dataclasses import dataclass

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


def _default_columns(frame: pd.DataFrame) -> tuple[str, str | None]:
    categorical = [str(c) for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    numeric = [str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    x = categorical[0] if categorical else str(frame.columns[0])
    y = numeric[0] if numeric else (str(frame.columns[1]) if frame.shape[1] > 1 else None)
    return x, y


def parse_text_request(text: str, frame: pd.DataFrame, chart_count: int) -> list[ChartSpec]:
    """Convert bounded Korean/English chart requests to validated chart specs."""
    request = (text or "").strip()
    if not request:
        raise ValueError("시각화 요청 내용을 입력하세요.")
    parts = [part.strip() for part in re.split(r"[;\n]+", request) if part.strip()]
    if not parts:
        parts = [request]
    default_x, default_y = _default_columns(frame)
    specs = []
    columns = sorted((str(c) for c in frame.columns), key=len, reverse=True)
    for index in range(chart_count):
        sentence = parts[min(index, len(parts) - 1)]
        lowered = sentence.casefold()
        if any(word in lowered for word in ["correlation", "상관 히트맵", "상관계수 행렬"]):
            chart_type = ChartType.CORRELATION_HEATMAP
        elif any(word in lowered for word in ["heatmap", "히트맵", "열지도"]):
            chart_type = ChartType.HEATMAP
        elif any(word in lowered for word in ["grouped bar", "그룹 막대"]):
            chart_type = ChartType.GROUPED_BAR
        elif any(word in lowered for word in ["stacked bar", "누적 막대"]):
            chart_type = ChartType.STACKED_BAR
        elif any(word in lowered for word in ["scatter plot", "산점도"]):
            chart_type = ChartType.SCATTER_PLOT
        elif any(word in lowered for word in ["bubble", "버블"]):
            chart_type = ChartType.SCATTER_BUBBLE
        elif any(word in lowered for word in ["histogram", "히스토그램", "분포"]):
            chart_type = ChartType.HISTOGRAM
        elif any(word in lowered for word in ["pie", "원형", "파이"]):
            chart_type = ChartType.PIE
        elif any(word in lowered for word in ["line", "선형", "추세"]):
            chart_type = ChartType.LINE
        else:
            chart_type = ChartType.BAR
        mentioned = [column for column in columns if column.casefold() in lowered]
        mentioned.sort(key=lambda column: lowered.find(column.casefold()))
        x = mentioned[0] if mentioned else default_x
        y = mentioned[1] if len(mentioned) > 1 else default_y
        aggregation = (
            Aggregation.RATIO
            if any(word in lowered for word in ["ratio", "비율", "%"])
            else Aggregation.MEAN
            if any(word in lowered for word in ["mean", "average", "평균"])
            else Aggregation.SUM
            if any(word in lowered for word in ["sum", "합계"])
            else Aggregation.VALID_COUNT
            if any(word in lowered for word in ["valid count", "유효값"])
            else Aggregation.COUNT
        )
        value_column = None
        if aggregation in {Aggregation.SUM, Aggregation.MEAN, Aggregation.VALID_COUNT}:
            numeric_mentions = [c for c in mentioned if pd.api.types.is_numeric_dtype(frame[c])]
            value_column = numeric_mentions[-1] if numeric_mentions else default_y
        group = mentioned[2] if len(mentioned) > 2 else None
        if chart_type == ChartType.BAR and any(word in lowered for word in ["grouped", "그룹"]):
            bar_mode = "grouped"
        elif chart_type == ChartType.BAR and any(word in lowered for word in ["100%", "100％"]):
            bar_mode = "stacked_100"
        elif chart_type == ChartType.BAR and any(word in lowered for word in ["stacked", "누적"]):
            bar_mode = "stacked"
        else:
            bar_mode = "basic"
        multi_types = {ChartType.CORRELATION_HEATMAP}
        two_axis_types = {
            ChartType.SCATTER_PLOT, ChartType.GROUPED_BAR, ChartType.STACKED_BAR,
            ChartType.SCATTER_BUBBLE, ChartType.HEATMAP,
        }
        spec_data = {
            "chart_type": chart_type,
            "x": "" if chart_type in multi_types else x,
            "y": y if chart_type in two_axis_types else None,
            "variables": mentioned if chart_type in multi_types else [],
            "group": group,
            "value_column": value_column,
            "aggregation": aggregation,
            "title": sentence[:80],
            "advanced": {"bar_mode": bar_mode},
        }
        specs.append(ChartSpec.model_validate(spec_data))
    return specs


def summarize_artifact(table: pd.DataFrame, spec: ChartSpec) -> str:
    lines = [f"통계표는 {len(table):,}개 행으로 구성되었습니다."]
    if "값" in table and not table.empty:
        maximum_index = table["값"].idxmax()
        maximum = float(table.loc[maximum_index, "값"])
        label_columns = [c for c in table.columns if c not in {"값", "구간 시작", "구간 끝", "구간 중심"}]
        label = " · ".join(str(table.loc[maximum_index, c]) for c in label_columns[:2]) if label_columns else "최대 구간"
        lines.append(f"가장 큰 값은 {label}의 {maximum:,.2f}입니다.")
        if spec.chart_type == ChartType.SCATTER_BUBBLE and spec.y and len(table) >= 2:
            x = pd.to_numeric(table[spec.x], errors="coerce")
            y = pd.to_numeric(table[spec.y], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() >= 2:
                lines.append(f"두 축의 상관계수는 {x[valid].corr(y[valid]):.3f}입니다.")
        elif len(table) > 1:
            total = float(table["값"].sum())
            if total:
                lines.append(f"최대 항목은 표시 합계의 {maximum / total * 100:,.1f}%를 차지합니다.")
    if spec.aggregation == Aggregation.RATIO:
        basis = {"total": "전체", "within_x": "X1 내부", "within_y": "X2 내부"}[spec.ratio_basis]
        lines.append(f"비율의 분모 기준은 {basis}입니다.")
    if spec.advanced.top_n and len(table) >= spec.advanced.top_n:
        lines.append(f"차트 요소는 상위 {spec.advanced.top_n}개로 제한되었습니다.")
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
