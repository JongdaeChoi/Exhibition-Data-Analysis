from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    MULTI_VARIABLE = "multi_variable"
    PIE = "pie"
    HISTOGRAM = "histogram"
    SCATTER_PLOT = "scatter_plot"
    GROUPED_BAR = "grouped_bar"
    STACKED_BAR = "stacked_bar"
    SCATTER_BUBBLE = "scatter_bubble"
    HEATMAP = "heatmap"
    CORRELATION_HEATMAP = "correlation_heatmap"


class Aggregation(str, Enum):
    COUNT = "count"
    VALID_COUNT = "valid_count"
    SUM = "sum"
    MEAN = "mean"
    RATIO = "ratio"


class ReferenceLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targets: list[Literal["x", "y"]] = Field(default_factory=lambda: ["y"])
    x_kind: Literal["numeric", "category", "date"] | None = None
    y_kind: Literal["numeric", "category", "date"] | None = None
    x_value: float | date | str | None = None
    y_value: float | date | str | None = None
    style: Literal["-", "--", "-.", ":"] = "--"
    width: float = Field(default=1.2, ge=0.1, le=10.0)
    color: str = "#475569"
    alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    label: str = ""
    label_visible: bool = True
    label_position: Literal["start", "center", "end"] = "end"
    label_size: int = Field(default=9, ge=4, le=40)
    label_weight: Literal["normal", "bold"] = "normal"
    label_color: str | None = None
    label_alpha: float = Field(default=0.9, ge=0.0, le=1.0)
    label_pad: float = Field(default=2.0, ge=-100.0, le=100.0)

    @model_validator(mode="after")
    def validate_values(self):
        if not self.targets:
            raise ValueError("기준선을 적용할 축을 하나 이상 선택하세요.")
        for axis in self.targets:
            if getattr(self, f"{axis}_value") is None:
                raise ValueError(f"{axis.upper()}축 기준값을 입력하세요.")
        return self


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    x: float = 0.5
    y: float = 0.5
    coordinate: Literal["axes", "data"] = "axes"
    size: int = Field(default=10, ge=4, le=60)
    weight: Literal["normal", "bold"] = "normal"
    color: str = "#172033"
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    horizontal_alignment: Literal["left", "center", "right"] = "center"
    vertical_alignment: Literal["top", "center", "bottom"] = "center"
    rotation: float = Field(default=0.0, ge=-180.0, le=180.0)
    box_visible: bool = False
    box_color: str = "#FFFFFF"
    box_alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    box_edge_color: str = "#CBD5E1"
    box_line_style: Literal["-", "--", "-.", ":"] = "-"
    arrow_visible: bool = False
    arrow_x: float | None = None
    arrow_y: float | None = None


class AdvancedSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_sort: Literal["none", "ascending", "descending"] = "none"
    y_sort: Literal["none", "ascending", "descending"] = "none"
    top_n: int | None = Field(default=20, ge=1, le=500)
    include_missing: bool = False
    orientation: Literal["vertical", "horizontal"] = "vertical"
    bar_mode: Literal["basic", "grouped", "stacked", "stacked_100"] = "basic"
    bar_width: float = Field(default=0.8, ge=0.05, le=1.0)
    bar_gap: float = Field(default=0.0, ge=0.0, le=0.9)
    group_gap: float = Field(default=0.0, ge=0.0, le=0.9)
    bar_corner_style: Literal["square", "rounded"] = "square"
    palette: str = "Blues"
    base_color: str = "#2563EB"
    alpha: float = Field(default=0.85, ge=0.05, le=1.0)
    edge_color: str = "#334155"
    edge_width: float = Field(default=0.6, ge=0.0, le=5.0)
    grid: bool = True
    grid_axis: Literal["x", "y", "both"] = "y"
    grid_style: Literal["-", "--", "-.", ":"] = "-"
    grid_width: float = Field(default=0.7, ge=0.1, le=10.0)
    grid_color: str = "#CBD5E1"
    grid_alpha: float = Field(default=0.22, ge=0.0, le=1.0)
    legend: bool = True
    legend_location: str = "best"
    legend_color: str = "#172033"
    tick_rotation: int = Field(default=0, ge=-90, le=90)
    title_size: int = Field(default=13, ge=6, le=40)
    title_visible: bool = True
    title_weight: Literal["normal", "bold"] = "bold"
    title_color: str = "#172033"
    title_location: Literal["left", "center", "right"] = "center"
    title_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    title_pad: float = Field(default=6.0, ge=0.0, le=100.0)
    axis_size: int = Field(default=10, ge=6, le=30)
    axis_weight: Literal["normal", "bold"] = "normal"
    axis_color: str = "#172033"
    x_label_rotation: float = Field(default=0.0, ge=-180.0, le=180.0)
    y_label_rotation: float = Field(default=90.0, ge=-180.0, le=180.0)
    x_label_pad: float = Field(default=4.0, ge=0.0, le=100.0)
    y_label_pad: float = Field(default=4.0, ge=0.0, le=100.0)
    x_label_visible: bool = True
    y_label_visible: bool = True
    x_label_size: int = Field(default=10, ge=6, le=40)
    y_label_size: int = Field(default=10, ge=6, le=40)
    x_label_weight: Literal["normal", "bold"] = "normal"
    y_label_weight: Literal["normal", "bold"] = "normal"
    x_label_color: str = "#172033"
    y_label_color: str = "#172033"
    x_label_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    y_label_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    x_label_location: Literal["left", "center", "right"] = "center"
    y_label_location: Literal["bottom", "center", "top"] = "center"
    x_tick_rotation: float = Field(default=0.0, ge=-180.0, le=180.0)
    y_tick_rotation: float = Field(default=0.0, ge=-180.0, le=180.0)
    tick_weight: Literal["normal", "bold"] = "normal"
    tick_color: str = "#334155"
    tick_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    x_tick_visible: bool = True
    y_tick_visible: bool = True
    x_tick_size: int = Field(default=10, ge=4, le=40)
    y_tick_size: int = Field(default=10, ge=4, le=40)
    x_tick_weight: Literal["normal", "bold"] = "normal"
    y_tick_weight: Literal["normal", "bold"] = "normal"
    x_tick_color: str = "#334155"
    y_tick_color: str = "#334155"
    x_tick_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    y_tick_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    x_tick_pad: float = Field(default=3.5, ge=0.0, le=100.0)
    y_tick_pad: float = Field(default=3.5, ge=0.0, le=100.0)
    x_tick_number_format: Literal["auto", "integer", "decimal1", "decimal2", "thousands", "percent"] = "auto"
    y_tick_number_format: Literal["auto", "integer", "decimal1", "decimal2", "thousands", "percent"] = "auto"
    number_format: str = ",.1f"
    unit: str = ""
    label_position_mode: Literal["auto", "inside", "center", "edge", "outside", "manual"] = "auto"
    label_offset_x: float = Field(default=0.0, ge=-100.0, le=100.0)
    label_offset_y: float = Field(default=5.0, ge=-100.0, le=100.0)
    label_font_size: int = Field(default=8, ge=4, le=40)
    label_font_weight: Literal["normal", "bold"] = "normal"
    label_color: str = "#172033"
    label_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    label_rotation: float = Field(default=0.0, ge=-180.0, le=180.0)
    label_positive_sign: bool = False
    label_negative_format: Literal["minus", "parentheses"] = "minus"
    pie_label_mode: Literal["ratio", "label", "label_ratio"] = "ratio"
    pie_ratio_format: Literal[".0f", ".1f", ".2f"] = ".1f"
    histogram_bins: int = Field(default=10, ge=2, le=100)
    histogram_bin_width: float | None = Field(default=None, gt=0)
    histogram_density: bool = False
    line_style: Literal["-", "--", "-.", ":"] = "-"
    line_width: float = Field(default=2.0, ge=0.2, le=10.0)
    marker: str = "o"
    marker_size: float = Field(default=5.0, ge=1.0, le=30.0)
    scatter_marker: str = "o"
    area_fill: bool = False
    line_curvature: float = Field(default=0.0, ge=0.0, le=1.0)
    pie_start_angle: int = Field(default=90, ge=0, le=360)
    donut: bool = False
    pie_shadow: bool = False
    pie_shadow_width: float = Field(default=0.04, ge=0.0, le=0.2)
    pie_shadow_color: str = "#475569"
    pie_shadow_alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    pie_explode_labels: list[str] = Field(default_factory=list)
    pie_explode_width: float = Field(default=0.08, ge=0.0, le=0.5)
    pie_edge_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    pie_min_ratio: float = Field(default=0.0, ge=0.0, le=30.0)
    pie_sort_by: Literal["none", "label", "value"] = "none"
    pie_sort_direction: Literal["ascending", "descending"] = "ascending"
    donut_hole_size: float = Field(default=0.5, ge=0.05, le=0.9)
    donut_ring_width: float = Field(default=0.4, ge=0.05, le=0.9)
    donut_center_color: str = "#FFFFFF"
    donut_center_border: bool = False
    donut_center_border_color: str = "#334155"
    donut_center_border_width: float = Field(default=1.0, ge=0.0, le=10.0)
    donut_center_border_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    scatter_size: float = Field(default=80.0, ge=5.0, le=1000.0)
    trendline: bool = False
    heatmap_cmap: str = "Blues"
    heatmap_annotate: bool = True
    heatmap_colorbar: bool = True
    heatmap_linewidth: float = Field(default=0.5, ge=0.0, le=5.0)
    heatmap_linecolor: str = "#FFFFFF"
    heatmap_linealpha: float = Field(default=1.0, ge=0.0, le=1.0)
    heatmap_value_format: str = ".2g"
    event_start: Any | None = None
    event_end: Any | None = None
    event_color: str = "#F59E0B"
    event_alpha: float = Field(default=0.15, ge=0.0, le=1.0)
    legend_title: str = ""
    legend_font_size: int = Field(default=9, ge=4, le=40)
    legend_font_weight: Literal["normal", "bold"] = "normal"
    legend_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    legend_background: str = "#FFFFFF"
    legend_background_alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    legend_border_visible: bool = False
    legend_border_color: str = "#CBD5E1"
    legend_border_width: float = Field(default=0.8, ge=0.0, le=10.0)
    legend_direction: Literal["vertical", "horizontal"] = "vertical"
    grid_x: bool = False
    grid_y: bool = True
    grid_which: Literal["major", "minor", "both"] = "major"
    element_range: Literal["all", "top", "bottom"] = "top"
    rank_basis: Literal["value", "ratio", "original"] = "value"
    remaining_items: Literal["exclude", "other"] = "exclude"

    @model_validator(mode="after")
    def validate_donut_geometry(self):
        if self.donut and self.donut_hole_size + self.donut_ring_width > 1.0:
            raise ValueError("도넛 구멍 크기와 링 두께의 합은 1.0 이하여야 합니다.")
        return self


class DeepSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    x_axis_mode: Literal["all", "numeric_range", "category_range", "category_select", "date_range"] = "all"
    y_axis_mode: Literal["all", "numeric_range", "category_range", "category_select", "date_range"] = "all"
    x_tick_interval: float | None = Field(default=None, gt=0)
    y_tick_interval: float | None = Field(default=None, gt=0)
    x_category_start: str | None = None
    x_category_end: str | None = None
    y_category_start: str | None = None
    y_category_end: str | None = None
    x_selected_categories: list[str] = Field(default_factory=list)
    y_selected_categories: list[str] = Field(default_factory=list)
    x_date_start: date | None = None
    x_date_end: date | None = None
    y_date_start: date | None = None
    y_date_end: date | None = None
    x_date_tick_frequency: Literal["auto", "day", "week", "month", "quarter", "year"] = "auto"
    y_date_tick_frequency: Literal["auto", "day", "week", "month", "quarter", "year"] = "auto"
    x_date_format: str = "%Y-%m-%d"
    y_date_format: str = "%Y-%m-%d"
    x_log: bool = False
    y_log: bool = False
    invert_x: bool = False
    invert_y: bool = False
    moving_average: int | None = Field(default=None, ge=2, le=100)
    cumulative: bool = False
    normalize: bool = False
    reference_line: float | None = None
    reference_enabled: bool = False
    reference_targets: list[Literal["x", "y"]] = Field(default_factory=list)
    reference_line_style: Literal["-", "--", "-.", ":"] = "--"
    reference_line_width: float = Field(default=1.2, ge=0.1, le=10.0)
    reference_line_alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    reference_label: str = ""
    reference_label_size: int = Field(default=9, ge=4, le=40)
    reference_label_alpha: float = Field(default=0.9, ge=0.0, le=1.0)
    x_reference_kind: Literal["numeric", "category", "date"] | None = None
    y_reference_kind: Literal["numeric", "category", "date"] | None = None
    x_reference_value: float | date | str | None = None
    y_reference_value: float | date | str | None = None
    show_mean: bool = False
    show_median: bool = False
    jitter: float = Field(default=0.0, ge=0.0, le=2.0)
    show_correlation: bool = False
    highlight_outliers: bool = False
    heatmap_center: float | None = None
    reference_lines: list[ReferenceLine] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_axis_controls(self):
        for axis in ("x", "y"):
            mode = getattr(self, f"{axis}_axis_mode")
            minimum = getattr(self, f"{axis}_min")
            maximum = getattr(self, f"{axis}_max")
            if mode == "numeric_range":
                if minimum is None or maximum is None:
                    raise ValueError(f"{axis.upper()}축 최소값과 최대값을 모두 입력하세요.")
                if minimum >= maximum:
                    raise ValueError(f"{axis.upper()}축 최소값은 최대값보다 작아야 합니다.")
            if mode == "category_range" and (
                getattr(self, f"{axis}_category_start") is None
                or getattr(self, f"{axis}_category_end") is None
            ):
                raise ValueError(f"{axis.upper()}축 범주의 시작값과 종료값을 선택하세요.")
            if mode == "category_select" and not getattr(self, f"{axis}_selected_categories"):
                raise ValueError(f"{axis.upper()}축에 표시할 항목을 하나 이상 선택하세요.")
            if mode == "date_range":
                start = getattr(self, f"{axis}_date_start")
                end = getattr(self, f"{axis}_date_end")
                if start is None or end is None:
                    raise ValueError(f"{axis.upper()}축 시작일과 종료일을 모두 입력하세요.")
                if start > end:
                    raise ValueError(f"{axis.upper()}축 시작일은 종료일보다 빨라야 합니다.")
        if self.reference_enabled:
            if not self.reference_targets:
                raise ValueError("기준선을 적용할 X축 또는 Y축을 선택하세요.")
            for axis in self.reference_targets:
                if getattr(self, f"{axis}_reference_value") is None:
                    raise ValueError(f"{axis.upper()}축 기준값을 입력하세요.")
        return self


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chart_type: ChartType
    x: str = ""
    y: str | None = None
    group: str | None = None
    value_column: str | None = None
    aggregation: Aggregation = Aggregation.COUNT
    ratio_basis: Literal["total", "within_x", "within_y"] = "total"
    variables: list[str] = Field(default_factory=list)
    comparison_chart: Literal["bar", "line"] = "bar"
    category_orders: dict[str, list[str]] = Field(default_factory=dict)
    x_y_swap: bool = False
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    show_values: bool = True
    advanced: AdvancedSettings = Field(default_factory=AdvancedSettings)
    deep: DeepSettings = Field(default_factory=DeepSettings)

    @model_validator(mode="after")
    def validate_chart_fields(self):
        if self.chart_type == ChartType.MULTI_VARIABLE:
            if len(self.variables) < 2:
                raise ValueError("다변수 비교 차트에는 동일 유형의 변수를 2개 이상 선택하세요.")
        elif self.chart_type == ChartType.CORRELATION_HEATMAP:
            if len(self.variables) < 2:
                raise ValueError("상관 히트맵에는 수치형 변수를 2개 이상 선택하세요.")
        elif not self.x:
            raise ValueError("X1 변수를 선택하세요.")
        if self.chart_type in {
            ChartType.SCATTER_PLOT,
            ChartType.GROUPED_BAR,
            ChartType.STACKED_BAR,
            ChartType.SCATTER_BUBBLE,
            ChartType.HEATMAP,
        }:
            if not self.y:
                raise ValueError(f"{self.chart_type.value} 차트에는 Y축 변수가 필요합니다.")
        if (
            self.aggregation in {Aggregation.SUM, Aggregation.MEAN}
            and self.chart_type != ChartType.MULTI_VARIABLE
            and not self.value_column
        ):
            raise ValueError("합계 또는 평균 집계에는 집계 대상 변수가 필요합니다.")
        if self.chart_type == ChartType.BAR and self.advanced.bar_mode != "basic" and not self.group:
            raise ValueError("그룹/누적 막대에는 그룹 변수가 필요합니다.")
        return self


class FigureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: int = Field(default=1, ge=1, le=3)
    columns: int = Field(default=1, ge=1, le=3)
    width: float = Field(default=12.0, ge=4.0, le=30.0)
    height: float = Field(default=8.0, ge=3.0, le=30.0)
    dpi: int = Field(default=120, ge=72, le=600)
    figure_background: str = "#FFFFFF"
    figure_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    figure_border_width: float = Field(default=0.0, ge=0.0, le=20.0)
    figure_border_color: str = "#CBD5E1"
    figure_border_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    axes_background: str = "#FFFFFF"
    axes_background_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    axes_border_visible: bool = True
    axes_border_positions: list[Literal["top", "bottom", "left", "right"]] = Field(
        default_factory=lambda: ["top", "bottom", "left", "right"]
    )
    axes_border_width: float = Field(default=0.8, ge=0.0, le=10.0)
    axes_border_color: str = "#334155"
    axes_border_style: Literal["-", "--", "-.", ":"] = "-"
    axes_border_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    axes_style_scope: Literal["all", "selected"] = "all"
    axes_target_index: int = Field(default=1, ge=1, le=9)
    share_x: bool = False
    share_y: bool = False
    horizontal_space: float = Field(default=0.28, ge=0.0, le=1.5)
    vertical_space: float = Field(default=0.35, ge=0.0, le=1.5)
    margin_left: float = Field(default=0.08, ge=0.0, le=0.9)
    margin_right: float = Field(default=0.98, ge=0.1, le=1.0)
    margin_bottom: float = Field(default=0.08, ge=0.0, le=0.9)
    margin_top: float = Field(default=0.92, ge=0.1, le=1.0)
    tight_layout: bool = True
    constrained_layout: bool = False
    layout_mode: Literal["tight", "constrained", "basic", "custom"] = "tight"
    font_family: str = "NanumGothic"
    font_color: str = "#172033"
    transparent: bool = False
    filename: str = "visualization"
    output_formats: list[Literal["png", "jpg", "pdf", "svg"]] = Field(
        default_factory=lambda: ["png", "jpg", "pdf", "svg"]
    )
    include_metadata: bool = True

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_layout_mode(cls, value):
        if isinstance(value, dict) and "layout_mode" not in value:
            value = dict(value)
            if value.get("constrained_layout"):
                value["layout_mode"] = "constrained"
            elif value.get("tight_layout") is False:
                value["layout_mode"] = "basic"
        return value

    @model_validator(mode="after")
    def validate_margins(self):
        if self.margin_left >= self.margin_right or self.margin_bottom >= self.margin_top:
            raise ValueError("Figure 여백의 왼쪽/아래 값은 오른쪽/위 값보다 작아야 합니다.")
        if self.axes_style_scope == "selected" and self.axes_target_index > self.rows * self.columns:
            raise ValueError("개별 Axes 대상 번호가 Subplot 개수를 초과했습니다.")
        return self
