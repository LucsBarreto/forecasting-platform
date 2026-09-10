"""
interface base dos modelos.

este módulo define o contrato comum para todos os modelos de machine learning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """
    interface abstrata para modelos de previsão.

    toda implementação de modelo deve fornecer:
    - um nome de modelo.
    - um método fit.
    - um método predict.
    - inspeção de parâmetros.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """retorna o nome do modelo."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseModel:
        """
        treina o modelo.

        parameters
        ----------
        x
            features de treinamento.

        y
            alvo de treinamento.

        returns
        -------
        basemodel
            instância do modelo treinado.
        """

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        gera previsões.

        parameters
        ----------
        x
            features para previsão.

        returns
        -------
        pd.series
            previsões do modelo.
        """

    def fit_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """
        treina o modelo e gera previsões.

        a implementação padrão delega as operações aos métodos
        fit() e predict().
        """

        self.fit(X, y)

        return self.predict(X)

    def get_params(self) -> dict[str, Any]:
        """
        retorna os parâmetros do modelo.

        implementações concretas podem sobrescrever este método
        quando expõem parâmetros específicos do modelo.
        """

        return {}