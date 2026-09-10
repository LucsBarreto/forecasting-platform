"""
explicabilidade de modelos.

fornece utilitários independentes de modelo para extrair
a importância das features de modelos treinados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.exceptions.modeling import ModelingError


@dataclass(slots=True)
class ModelExplainer:
    """
    extrai informações de explicabilidade de modelos treinados.

    responsabilidades
    ------------------
    - validar modelos treinados.
    - extrair a importância das features.
    - associar valores de importância aos nomes das features.
    - retornar os resultados em um dataframe padronizado.
    """

    def feature_importance(
        self,
        model: Any,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """
        retorna a importância das features de um modelo treinado.

        parameters
        ----------
        model
            modelo treinado ou wrapper de modelo.

        feature_names
            nomes das features utilizadas pelo modelo.

        returns
        -------
        pd.dataframe
            dataframe contendo nomes e valores de importância das features.

        raises
        ------
        modelingerror
            se o modelo não expuser a importância das features ou a
            quantidade de valores de importância não corresponder às features.
        """

        self._validate_feature_names(
            feature_names,
        )

        estimator = self._get_estimator(
            model,
        )

        importance = self._get_importance(
            estimator,
        )

        if len(importance) != len(feature_names):
            raise ModelingError(
                "The number of feature importance values "
                "does not match the number of features."
            )

        result = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )

        return result.sort_values(
            by="importance",
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)

    @staticmethod
    def _validate_feature_names(
        feature_names: list[str],
    ) -> None:
        """valida os nomes das features."""

        if not feature_names:
            raise ModelingError(
                "No feature names were provided."
            )

        if len(feature_names) != len(set(feature_names)):
            raise ModelingError(
                "Duplicated feature names are not allowed."
            )

        if any(
            not isinstance(feature, str)
            or not feature.strip()
            for feature in feature_names
        ):
            raise ModelingError(
                "Feature names must be non-empty strings."
            )

    @staticmethod
    def _get_estimator(
        model: Any,
    ) -> Any:
        """extrai o estimador subjacente de um wrapper de modelo."""

        if model is None:
            raise ModelingError(
                "A model is required."
            )

        estimator = getattr(
            model,
            "model",
            None,
        )

        if estimator is not None:
            return estimator

        return model

    @staticmethod
    def _get_importance(
        model: Any,
    ) -> Any:
        """extrai a importância das features de um modelo."""

        if not hasattr(
            model,
            "feature_importances_",
        ):
            raise ModelingError(
                "Model does not expose feature importance."
            )

        importance = model.feature_importances_

        if importance is None:
            raise ModelingError(
                "Model feature importance is not available."
            )

        return importance