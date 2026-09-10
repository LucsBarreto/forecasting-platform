"""
avisa sobre registros invalidos sem modificar os dados
"""

from __future__ import annotations

import pandas as pd

from .base import BaseValidationAction
from .registry import register_action
from .result import ActionResult
from src.core.enums import ValidationActionType


@register_action(ValidationActionType.WARN)
class WarnAction(BaseValidationAction):
    """
    avisa sobre registros invalidos sem modificar os dados
    """

    @property
    def name(self) -> str:
        return "warn"

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