"""adapta as features ao contrato de entrada de cada família de modelos."""

from __future__ import annotations

import pandas as pd

from src.core.exceptions.validation import DataValidationError
from src.ml.models.base_model import BaseModel

TIME_SERIES_MODELS = {"prophet", "sarima", "baseline"}


def adapt_features(
    model: BaseModel,
    features: pd.DataFrame,
    date_column: str = "DATA",
) -> pd.DataFrame:
    """retorna features compatíveis com o modelo treinado."""

    if model.name in TIME_SERIES_MODELS:
        if model.name == "prophet" and date_column not in features.columns:
            raise DataValidationError(
                f"Date column '{date_column}' was not found."
            )
        return features.copy()

    numeric = features.select_dtypes(include=["number", "bool"]).copy()
    if numeric.empty:
        raise DataValidationError(
            f"No numeric features available for model '{model.name}'."
        )
    return numeric