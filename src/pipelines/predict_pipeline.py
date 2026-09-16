"""
pipeline de predição.

este módulo orquestra a geração de previsões e a exportação dos resultados.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions.validation import DataValidationError
from src.export.forecast_exporter import ForecastExporter
from src.ml.forecast import (
    Forecaster,
    FutureFeatureAvailabilityContract,
    FutureFeatureFrameBuilder,
    FutureForecastContract,
    FutureForecastResult,
)

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

    modo futuro:

        run_future_forecast(
            model=model,
            history_frame=history,
            selected_features=[...],
            available_features=[...],
            horizon=3,
        )

    """

    def __init__(
        self,
        forecaster: Forecaster | None = None,
        predictor: Any | None = None,
        forecast_exporter: ForecastExporter | None = None,
        future_feature_builder: FutureFeatureFrameBuilder | None = None,
        future_feature_availability_contract: FutureFeatureAvailabilityContract | None = None,
        future_forecast_contract: FutureForecastContract | None = None,
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

        future_feature_builder
            construtor de frame de features futuras.

        future_feature_availability_contract
            validador de disponibilidade de features futuras.

        future_forecast_contract
            contrato de previsão futura.
        """

        super().__init__()

        self.forecaster = forecaster
        self.predictor = predictor
        self.forecast_exporter = forecast_exporter
        self.future_feature_builder = future_feature_builder or FutureFeatureFrameBuilder()
        self.future_feature_availability_contract = future_feature_availability_contract or FutureFeatureAvailabilityContract()
        self.future_forecast_contract = future_forecast_contract or FutureForecastContract()

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

    def run_future_forecast(
        self,
        model: Any,
        history_frame: pd.DataFrame,
        selected_features: list[str],
        available_features: list[str] | None = None,
        horizon: int = 1,
        target_column: str = "target",
        time_column: str = "date",
        predictions: list[float] | pd.Series | None = None,
    ) -> FutureForecastResult:
        """
        Consome o contrato de forecast futuro em forma isolada.

        Este método mantém o `PredictPipeline` como consumidor da nova
        abstração de forecast futuro, sem abrir o restante do fluxo de
        produção. Ele valida disponibilidade, organiza o frame futuro e
        entrega `FutureForecastResult` pela cadeia formal.
        """
        if not isinstance(history_frame, pd.DataFrame):
            raise TypeError("history_frame must be a pandas DataFrame.")
        if history_frame.empty:
            raise ValueError("history_frame cannot be empty.")
        if not isinstance(selected_features, list) or not selected_features:
            raise ValueError("selected_features must be a non-empty list.")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError("target_column must be a non-empty string.")
        if target_column not in history_frame.columns:
            raise ValueError(f"Target column '{target_column}' was not found in history_frame.")
        if not isinstance(time_column, str) or not time_column.strip():
            raise ValueError("time_column must be a non-empty string.")
        if time_column not in history_frame.columns:
            raise ValueError(f"Time column '{time_column}' was not found in history_frame.")
        if model is None:
            raise DataValidationError("A model is required for future forecasting.")

        available = available_features or selected_features
        if not isinstance(available, list):
            raise TypeError("available_features must be a list of strings.")

        self.future_feature_availability_contract.run(
            selected_features=selected_features,
            available_features=available,
        )

        future_frame = self.future_feature_builder.build_recursive(
            history_frame=history_frame,
            horizon=horizon,
            target_column=target_column,
            predictions=predictions or [],
            lag_values=[1],
            rolling_windows=[2],
        )

        return self.future_forecast_contract.run(
            future_frame=future_frame,
            model=model,
        )

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