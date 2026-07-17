from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    HISTOGRAM = "histogram"
    SCATTER_BUBBLE = "scatter_bubble"
    HEATMAP = "heatmap"


class Aggregation(str, Enum):
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    RATIO = "ratio"


class AdvancedSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_sort: Literal["none", "ascending", "descending"] = "none"
    y_sort: Literal["none", "ascending", "descending"] = "none"
    top_n: int | None = Field(default=20, ge=1, le=500)
    include_missing: bool = False
    orientation: Literal["vertical", "horizontal"] = "vertical"
    bar_mode: Literal["basic", "grouped", "stacked", "stacked_100"] = "basic"
    palette: str = "Blues"
    base_color: str = "#2563EB"
    alpha: float = Field(default=0.85, ge=0.05, le=1.0)
    edge_color: str = "#334155"
    edge_width: float = Field(default=0.6, ge=0.0, le=5.0)
    grid: bool = True
    grid_axis: Literal["x", "y", "both"] = "y"
    legend: bool = True
    legend_location: str = "best"
    tick_rotation: int = Field(default=0, ge=-90, le=90)
    title_size: int = Field(default=13, ge=6, le=40)
    axis_size: int = Field(default=10, ge=6, le=30)
    number_format: str = ",.1f"
    unit: str = ""
    histogram_bins: int = Field(default=10, ge=2, le=100)
    histogram_density: bool = False
    line_style: Literal["-", "--", "-.", ":"] = "-"
    line_width: float = Field(default=2.0, ge=0.2, le=10.0)
    marker: str = "o"
    marker_size: float = Field(default=5.0, ge=1.0, le=30.0)
    area_fill: bool = False
    line_curvature: float = Field(default=0.0, ge=0.0, le=1.0)
    pie_start_angle: int = Field(default=90, ge=0, le=360)
    donut: bool = False
    pie_shadow: bool = False
    pie_min_ratio: float = Field(default=0.0, ge=0.0, le=30.0)
    pie_sort_by: Literal["none", "label", "value"] = "none"
    pie_sort_direction: Literal["ascending", "descending"] = "ascending"
    scatter_size: float = Field(default=80.0, ge=5.0, le=1000.0)
    trendline: bool = False
    heatmap_cmap: str = "Blues"
    heatmap_annotate: bool = True
    heatmap_colorbar: bool = True
    heatmap_linewidth: float = Field(default=0.5, ge=0.0, le=5.0)


class DeepSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    x_axis_mode: Literal["all", "numeric_range", "category_range", "category_select"] = "all"
    y_axis_mode: Literal["all", "numeric_range", "category_range", "category_select"] = "all"
    x_tick_interval: float | None = Field(default=None, gt=0)
    y_tick_interval: float | None = Field(default=None, gt=0)
    x_category_start: str | None = None
    x_category_end: str | None = None
    y_category_start: str | None = None
    y_category_end: str | None = None
    x_selected_categories: list[str] = Field(default_factory=list)
    y_selected_categories: list[str] = Field(default_factory=list)
    x_log: bool = False
    y_log: bool = False
    invert_x: bool = False
    invert_y: bool = False
    moving_average: int | None = Field(default=None, ge=2, le=100)
    cumulative: bool = False
    normalize: bool = False
    reference_line: float | None = None
    show_mean: bool = False
    show_median: bool = False
    jitter: float = Field(default=0.0, ge=0.0, le=2.0)
    show_correlation: bool = False
    highlight_outliers: bool = False
    heatmap_center: float | None = None

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
        return self


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chart_type: ChartType
    x: str
    y: str | None = None
    group: str | None = None
    value_column: str | None = None
    aggregation: Aggregation = Aggregation.COUNT
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    show_values: bool = True
    advanced: AdvancedSettings = Field(default_factory=AdvancedSettings)
    deep: DeepSettings = Field(default_factory=DeepSettings)

    @model_validator(mode="after")
    def validate_chart_fields(self):
        if self.chart_type in {ChartType.SCATTER_BUBBLE, ChartType.HEATMAP}:
            if not self.y:
                raise ValueError(f"{self.chart_type.value} 차트에는 Y축 변수가 필요합니다.")
            if self.x == self.y:
                raise ValueError("X축과 Y축 변수는 서로 달라야 합니다.")
        if self.aggregation in {Aggregation.SUM, Aggregation.MEAN} and not self.value_column:
            raise ValueError("합계 또는 평균 집계에는 집계 대상 변수가 필요합니다.")
        if self.chart_type == ChartType.BAR and self.advanced.bar_mode != "basic" and not self.group:
            raise ValueError("그룹/누적 막대에는 그룹 변수가 필요합니다.")
        return self


class FigureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid_size: int = Field(default=1, ge=1, le=3)
    width: float = Field(default=12.0, ge=4.0, le=30.0)
    height: float = Field(default=8.0, ge=3.0, le=30.0)
    dpi: int = Field(default=120, ge=72, le=600)
    figure_background: str = "#FFFFFF"
    axes_background: str = "#FFFFFF"
    horizontal_space: float = Field(default=0.28, ge=0.0, le=1.5)
    vertical_space: float = Field(default=0.35, ge=0.0, le=1.5)
    tight_layout: bool = True
    constrained_layout: bool = False
    font_family: str = "NanumGothic"
    font_color: str = "#172033"
    transparent: bool = False
    filename: str = "visualization"
