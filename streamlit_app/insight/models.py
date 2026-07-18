from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from visualization.models import Aggregation, ChartSpec, ChartType
from visualization.service import automatic_chart_title


MODEL_OPTIONS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
)


class InsightChartInput(BaseModel):
    """Small model-facing schema converted into the application's full ChartSpec."""

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
    title: str = ""
    show_values: bool = True

    def to_chart_spec(self) -> ChartSpec:
        title = self.title.strip() or automatic_chart_title(
            self.x, self.y, self.group, self.value_column, *self.variables
        )
        return ChartSpec.model_validate(
            {
                **self.model_dump(),
                "title": title,
                "advanced": {"top_n": 20},
            }
        )


class InsightDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["text", "chart", "chart_with_insight"]
    answer: str = ""
    chart_spec: InsightChartInput | None = None

    @model_validator(mode="after")
    def validate_action(self):
        if self.action != "text" and self.chart_spec is None:
            raise ValueError("차트 요청에는 chart_spec이 필요합니다.")
        if self.action == "text" and not self.answer.strip():
            raise ValueError("텍스트 요청에는 answer가 필요합니다.")
        return self


class InsightChartRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_base64: str
    mime_type: str = "image/png"
    source: dict[str, Any]


class InsightMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "model"]
    text: str
    charts: list[InsightChartRecord] = Field(default_factory=list)
    hidden: bool = False


class InsightHistoryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    saved_at: str | None = None
    model: str = MODEL_OPTIONS[0]
    source_filename: str | None = None
    data_signature: dict[str, Any] = Field(default_factory=dict)
    history: list[InsightMessage] = Field(default_factory=list)
