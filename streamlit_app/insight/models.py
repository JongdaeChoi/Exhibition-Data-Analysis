from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from visualization.models import Aggregation, ChartSpec, ChartType
from visualization.service import automatic_chart_title


PROVIDER_MODELS = {
    "OpenAI": (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ),
    "Gemini": (
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3.1-pro-preview",
    ),
}
MODEL_LABELS = {
    "gemini-2.5-flash": "Gemini 2.5 Flash · 빠른 응답, 일반 분석",
    "gemini-2.5-pro": "Gemini 2.5 Pro · 복잡한 분석, 고급 추론",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview · 최신 성능 미리보기",
    "gpt-5.6-sol": "GPT-5.6 Sol · 복잡한 분석, 고급 추론",
    "gpt-5.6-terra": "GPT-5.6 Terra · 일반 분석",
    "gpt-5.6-luna": "GPT-5.6 Luna · 빠른 응답, 대량 처리",
}
MODEL_OPTIONS = tuple(model for models in PROVIDER_MODELS.values() for model in models)


class InsightAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str
    mime_type: str
    kind: Literal["document", "image", "dataset"]
    content_base64: str = ""
    extracted_text: str = ""


class InsightCodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    modifies_df_clean: bool = False
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    before_shape: tuple[int, int] | None = None
    after_shape: tuple[int, int] | None = None


class InsightChartInput(BaseModel):
    """Small model-facing schema converted into the application's full ChartSpec."""

    model_config = ConfigDict(extra="forbid")

    chart_type: ChartType
    x: str = ""
    y: str | None = None
    group: str | None = None
    value_column: str | None = None
    aggregation: Aggregation = Field(
        description="사용자가 요청한 집계 방식: count, valid_count, sum, mean, ratio"
    )
    ratio_basis: Literal["total", "within_x", "within_y"] = "total"
    variables: list[str] = Field(default_factory=list)
    comparison_chart: Literal["bar", "line"] = "bar"
    x_category_order: list[str] = Field(
        default_factory=list,
        description="X축 범주의 사용자 지정 표시 순서. 실제 표시값을 문자열로 입력",
    )
    y_category_order: list[str] = Field(
        default_factory=list,
        description="Y축 범주의 사용자 지정 표시 순서. 실제 표시값을 문자열로 입력",
    )
    title: str = ""
    show_values: bool = True

    def to_chart_spec(self) -> ChartSpec:
        title = self.title.strip() or automatic_chart_title(
            self.x, self.y, self.group, self.value_column, *self.variables
        )
        values = self.model_dump(exclude={"x_category_order", "y_category_order"})
        category_orders = {}
        if self.x and self.x_category_order:
            category_orders[self.x] = self.x_category_order
        if self.y and self.y_category_order:
            category_orders[self.y] = self.y_category_order
        return ChartSpec.model_validate(
            {
                **values,
                "title": title,
                "category_orders": category_orders,
                "advanced": {"top_n": 20},
            }
        )


class InsightDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["text", "chart", "chart_with_insight", "business_insight"]
    answer: str = ""
    chart_spec: InsightChartInput | None = None

    @model_validator(mode="after")
    def validate_action(self):
        if self.action in {"chart", "chart_with_insight"} and self.chart_spec is None:
            raise ValueError("차트 요청에는 chart_spec이 필요합니다.")
        if self.action in {"text", "business_insight"} and not self.answer.strip():
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
    code: InsightCodeRecord | None = None
    attachments: list[InsightAttachment] = Field(default_factory=list)
    hidden: bool = False


class InsightHistoryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = 2
    saved_at: str | None = None
    provider: Literal["Gemini", "OpenAI"] = "OpenAI"
    model: str = MODEL_OPTIONS[0]
    source_filename: str | None = None
    data_signature: dict[str, Any] = Field(default_factory=dict)
    history: list[InsightMessage] = Field(default_factory=list)
