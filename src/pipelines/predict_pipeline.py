"""
pipeline de predição.

este módulo orquestra a geração de previsões e a exportação dos resultados.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.export.forecast_exporter import ForecastExporter
from src.ml.forecast import Forecaster

from .base_pipeline import BasePipeline


@dataclass(slots=True)
class PredictionPipelineResult:
    """
    resultado produzido pelo pipeline de predição.

    attributes
    ----------
    predictions
        dataframe de previsões ou previsões indexadas pelo nome do modelo.

    exported_files
        arquivos de previsão exportados indexados pelo formato.
    """

    predictions: Any

    exported_files: dict[str, Path] | None = None

    def __post_init__(self) -> None:
        """inicializa os arquivos exportados opcionais."""

        if self.exported_files is None:
            self.exported_files = {}


class PredictPipeline(BasePipeline):
    """
    pipeline responsável por gerar previsões.

    o pipeline suporta três modos de execução:

    1. previsão com um único modelo:

        run(
            features=features,
            model=model,
        )

    2. previsão com múltiplos modelos:

        run(
            models=models,
            features=features,
        )

    3. modo legado:

        run(
            features=features,
        )

    a exportação das previsões está disponível no modo de modelo único
    quando um forecastexporter está configurado.
    """

    def __init__(
        self,
        forecaster: Forecaster | None = None,
        predictor: Any | None = None,
        forecast_exporter: ForecastExporter | None = None,
    ) -> None:
        """
        inicializa o pipeline de predição.

        parameters
        ----------
        forecaster
            estratégia utilizada para gerar previsões.

        predictor
            objeto utilizado pela api de modelo único e exportação.

        forecast_exporter
            exportador opcional para previsões em csv e parquet.
        """

        super().__init__()

        self.forecaster = forecaster
        self.predictor = predictor
        self.forecast_exporter = forecast_exporter

    @property
    def name(self) -> str:
        """retorna o nome do pipeline."""

        return "pipeline de predição"

    def run(
        self,
        models: dict[str, Any] | None = None,
        features: pd.DataFrame | None = None,
        model: Any | None = None,
        export: bool = True,
    ) -> (
        PredictionPipelineResult
        | pd.DataFrame
    ):
        """
        gera previsões.

        suporta execução com um único modelo ou múltiplos modelos.

        parameters
        ----------
        models
            dicionário de modelos treinados indexados pelo nome.

        features
            features utilizadas para gerar as previsões.

        model
            modelo treinado utilizado na previsão.

        export
            indica se as previsões do modelo único devem ser exportadas.

        returns
        -------
        predictionpipelineresult or pandas.dataframe
            resultado da previsão.

        notes
        -----
        quando nenhum predictor ou forecaster está configurado e apenas
        features são fornecidas, o dataframe original é retornado
        para manter compatibilidade com versões anteriores.
        """

        start = self._log_start()

        try:
            # --------------------------------------------------
            # Legacy mode
            # --------------------------------------------------

            if (
                model is None
                and models is None
                and self.predictor is None
                and self.forecaster is None
            ):
                self._validate_features(
                    features,
                )

                self._log_finish(start)

                return features

            # --------------------------------------------------
            # Single-model mode
            # --------------------------------------------------

            if model is not None:
                self._validate_features(
                    features,
                )

                if self.predictor is None:
                    raise ValueError(
                        "predictor must be configured "
                        "for single-model prediction."
                    )

                predictions = self.predictor.predict(
                    model,
                    features,
                )

                self._validate_predictions(
                    predictions,
                )

                exported_files: dict[str, Path] = {}

                if (
                    export
                    and self.forecast_exporter is not None
                ):
                    exported_files[
                        "csv"
                    ] = self.forecast_exporter.export_csv(
                        predictions,
                    )

                    exported_files[
                        "parquet"
                    ] = self.forecast_exporter.export_parquet(
                        predictions,
                    )

                result = PredictionPipelineResult(
                    predictions=predictions,
                    exported_files=exported_files,
                )

                self._log_finish(start)

                return result

            # --------------------------------------------------
            # Multiple-model mode
            # --------------------------------------------------

            if models is not None:
                self._validate_models(
                    models,
                )

                self._validate_features(
                    features,
                )

                if self.forecaster is None:
                    raise ValueError(
                        "forecaster must be configured "
                        "for multiple-model prediction."
                    )

                predictions: dict[str, pd.Series] = {}

                for model_name, fitted_model in models.items():
                    predictions[
                        model_name
                    ] = self.forecaster.predict(
                        fitted_model,
                        features,
                    )

                result = PredictionPipelineResult(
                    predictions=predictions,
                )

                self._log_finish(start)

                return result

            # --------------------------------------------------
            # Invalid invocation
            # --------------------------------------------------

            raise ValueError(
                "model must be provided."
            )

        except Exception as exc:
            self._log_failure(exc)
            raise

    @staticmethod
    def _validate_models(
        models: dict[str, Any],
    ) -> None:
        """valida a entrada com múltiplos modelos."""

        if not isinstance(
            models,
            dict,
        ):
            raise TypeError(
                "models must be a dictionary."
            )

        if not models:
            raise ValueError(
                "models cannot be empty."
            )

        for model_name in models:
            if not isinstance(
                model_name,
                str,
            ) or not model_name.strip():
                raise ValueError(
                    "Model names must be non-empty strings."
                )

    @staticmethod
    def _validate_features(
        features: pd.DataFrame | None,
    ) -> None:
        """valida as features utilizadas na previsão."""

        if features is None:
            raise ValueError(
                "features cannot be None."
            )

        if not isinstance(
            features,
            pd.DataFrame,
        ):
            raise TypeError(
                "features must be a pandas DataFrame."
            )

        if features.empty:
            raise ValueError(
                "features cannot be empty."
            )

    @staticmethod
    def _validate_predictions(
        predictions: Any,
    ) -> None:
        """valida a saída do predictor de modelo único."""

        if not isinstance(
            predictions,
            pd.DataFrame,
        ):
            raise TypeError(
                "predictions must be a pandas DataFrame."
            )