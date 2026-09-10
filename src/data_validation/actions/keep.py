"""
mantém os registros invalidos
"""

from __future__ import annotations

import pandas as pd

from .base import BaseValidationAction
from .registry import register_action
from .result import ActionResult
from src.core.enums import ValidationActionType


@register_action(ValidationActionType.KEEP)
class KeepAction(BaseValidationAction):
    """
    mantém os registros invalidos do dataframe
    """

    @property
    def name(self) -> str:
        return "keep"

    def execute(
        self,
        dataframe: pd.DataFrame,
        mask: pd.Series,
    ) -> ActionResult:

        return ActionResult(
            dataframe=dataframe,
            removed_rows=0,
            corrected_rows=0,
            action=self.name,
        )