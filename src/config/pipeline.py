"""
configurações relacionadas ao pipeline.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class CleaningSettings(BaseModel):
    """configuração do processador de limpeza."""

    remove_empty_rows: bool = True
    remove_duplicate_rows: bool = True
    normalize_column_names: bool = True
    normalize_string_values: bool = True


class MissingSettings(BaseModel):
    """configuração do processador de valores ausentes."""

    numeric_strategy: str = "median"
    categorical_strategy: str = "mode"
    datetime_strategy: str = "keep"
    constant_value: str | int | float | datetime | None = None

class DatetimeSettings(BaseModel):
    """configuração do processador de datas."""

    create_year: bool = True
    create_month: bool = True
    create_quarter: bool = True
    create_semester: bool = True
    create_week: bool = True
    create_day: bool = True
    create_day_of_week: bool = True
    create_is_weekend: bool = True
    create_is_month_start: bool = True
    create_is_month_end: bool = True

class AggregationSettings(BaseModel):
    """configuração da agregação de dados."""

    group_by: list[str]
    metrics: dict[str, str]

class TemporalFeatureSettings(BaseModel):
    """configuração da engenharia de features temporais."""

    enabled: bool
    date_column: str
    create_business_day: bool
    create_holiday: bool
    create_days_to_month_end: bool
    create_holiday_distance: bool
    create_cyclical_features: bool


class LagFeatureSettings(BaseModel):
    """configuração da engenharia de features de lag."""

    enabled: bool
    date_column: str
    target_columns: list[str]
    group_levels: list[list[str]]
    lags: list[int]


class RollingFeatureSettings(BaseModel):
    """configuração da engenharia de features de janela móvel."""

    enabled: bool = True

    date_column: str = "DATA"

    target_columns: list[str] = Field(
        default_factory=list
    )

    group_levels: list[list[str]] = Field(
        default_factory=list
    )

    windows: list[int] = Field(
        default_factory=lambda: [3, 6, 12]
    )

    functions: list[str] = Field(
        default_factory=lambda: [
            "mean",
            "std",
        ]
    )


class TrendFeatureSettings(BaseModel):
    """configuração da engenharia de features de tendência."""

    enabled: bool = True

    date_column: str = "DATA"

    target_columns: list[str] = []

    group_levels: list[list[str]] = []

    periods: list[int] = [1, 3, 6, 12]

class BusinessFeatureSettings(BaseModel):
    """configuração da engenharia de features de negócio."""

    enabled: bool
    date_column: str
    target_columns: list[str]
    group_levels: list[list[str]]

class FeatureEngineeringSettings(BaseModel):
    """configuração da engenharia de features."""

    lag_features: LagFeatureSettings
    temporal_features: TemporalFeatureSettings
    rolling_features: RollingFeatureSettings
    trend_features: TrendFeatureSettings
    business_features: BusinessFeatureSettings


class PreprocessingSettings(BaseModel):
    """configuração do pré-processamento."""

    cleaning: CleaningSettings
    missing: MissingSettings
    datetime: DatetimeSettings
    aggregation: AggregationSettings
    feature_engineering: FeatureEngineeringSettings

class PipelineSettings(BaseModel):
    """configuração do pipeline."""

    preprocessing: PreprocessingSettings