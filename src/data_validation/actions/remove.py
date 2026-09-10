"""
remove registros invalidos
"""

from __future__ import annotations

import pandas as pd

from .base import BaseValidationAction
from .registry import register_action
from .result import ActionResult
from src.core.enums import ValidationActionType


@register_action(ValidationActionType.REMOVE)
class RemoveAction(BaseValidationAction):
    """
    remove registros invalidos do dataframe
    """

    @property
    def name(self) -> str:
        return "remove"

    def execute(
        self,
        dataframe: pd.DataFrame,
        mask: pd.Series,
    ) -> ActionResult:

        cleaned = dataframe.loc[
            ~mask
        ].copy()

        return ActionResult(
            dataframe=cleaned,
            removed_rows=int(mask.sum()),
            corrected_rows=0,
            action=self.name,
        )