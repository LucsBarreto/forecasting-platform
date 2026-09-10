"""
resultado da ação de validação
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ActionResult:
    """
    resultado retornado pela ação de validação
    """

    dataframe: pd.DataFrame

    removed_rows: int = 0

    corrected_rows: int = 0

    action: str = ""