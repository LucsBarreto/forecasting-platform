"""funções auxiliares para conversão de datas."""

from __future__ import annotations

import pandas as pd


def coerce_excel_datetime(series: pd.Series) -> pd.Series:
    """converte strings de data e números seriais do excel em uma série mista."""

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    converted = pd.to_datetime(
        series,
        errors="coerce",
        format="mixed",
    )

    if numeric_mask.any():
        converted.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    return converted