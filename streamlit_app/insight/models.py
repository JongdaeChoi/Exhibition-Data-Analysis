from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from visualization.models import Aggregation, ChartSpec, ChartType
from visualization.service import automatic_chart_title


PROVIDER_MODELS = {
    "Gemini": (
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3.1-pro-preview",
    ),
    "OpenAI": (
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ),
}
MODEL_LABELS = {
    "gemini-2.5-flash": "Gemini 2.5 Flash · 빠른 응답, 일반 분석",
    "gemini-2.5-pro": "Gemini 2.5 Pro · 복잡한 분석, 코드 생성",
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
    kind: Literal["document", "image"]
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

    action: Literal["text", "data", "chart", "chart_with_insight", "business_insight"]
    answer: str = ""
    chart_spec: InsightChartInput | None = None
    python_code: str = ""
    modifies_df_clean: bool = False

    @model_validator(mode="after")
    def validate_action(self):
        if self.action in {"chart", "chart_with_insight"} and self.chart_spec is None:
            raise ValueError("차트 요청에는 chart_spec이 필요합니다.")
        if self.action in {"text", "business_insight"} and not self.answer.strip():
            raise ValueError("텍스트 요청에는 answer가 필요합니다.")
        if self.action == "data" and not self.python_code.strip():
            raise ValueError("데이터 요청에는 python_code가 필요합니다.")
        if self.action != "data" and (self.python_code.strip() or self.modifies_df_clean):
            raise ValueError("Python 코드와 df_clean 수정 여부는 데이터 요청에서만 지정할 수 있습니다.")
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
    provider: Literal["Gemini", "OpenAI"] = "Gemini"
    model: str = MODEL_OPTIONS[0]
    source_filename: str | None = None
    data_signature: dict[str, Any] = Field(default_factory=dict)
    history: list[InsightMessage] = Field(default_factory=list)
