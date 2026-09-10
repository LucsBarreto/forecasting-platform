"""
ação de validação base
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .result import ActionResult


class BaseValidationAction(ABC):
    """
    classe base para ações de validação
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        action name.
        """

    @abstractmethod
    def execute(
        self,
        dataframe: pd.DataFrame,
        mask: pd.Series,
    ) -> ActionResult:
        """
        executa ação de validação
        """