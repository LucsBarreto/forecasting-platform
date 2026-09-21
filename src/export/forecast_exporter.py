"""
utilitários para exportação de previsões.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.core.exceptions.data import DataError
from src.ml.forecast.future_forecast import FutureForecastResult
from src.schemas.metadata import MetadataSchema, SingleForecastMetadataSchema


@dataclass(slots=True)
class ForecastExporter:
    """
    exporta resultados de previsão.

    responsabilidades
    ------------------
    - validar os dados de previsão.
    - criar diretórios de saída.
    - exportar os resultados de previsão.
    """

    output_directory: Path

    def export_csv(
        self,
        dataframe: pd.DataFrame,
        filename: str = "forecast.csv",
    ) -> Path:
        """exporta os resultados de previsão para csv."""

        self._validate_dataframe(dataframe)
        filename = self._validate_filename(filename)

        output_path = self._prepare_output_path(
            filename,
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        return output_path

    def export_future_result(
        self,
        result: FutureForecastResult,
        filename: str = "future_forecast.csv",
        metadata: dict | None = None,
    ) -> Path:
        """Converte um FutureForecastResult em DataFrame, persiste o CSV e escreve um sidecar de metadados auditáveis."""
        if not isinstance(result, FutureForecastResult):
            raise TypeError("result must be a FutureForecastResult.")

        predictions = result.predictions.copy()
        if not isinstance(predictions, pd.Series):
            raise TypeError("result.predictions must be a pandas Series.")

        if metadata is not None:
            validated_metadata = SingleForecastMetadataSchema.model_validate(metadata)
            self._validate_metadata_consistency(result, validated_metadata)

        dataframe = pd.DataFrame(
            {
                "prediction": predictions.to_numpy(),
            },
            index=predictions.index,
        )

        exported_path = self.export_csv(dataframe, filename=filename)

        if metadata is not None:
            metadata_path = self.output_directory / "forecast_metadata.json"
            self.output_directory.mkdir(parents=True, exist_ok=True)
            with metadata_path.open(mode="w", encoding="utf-8") as file:
                json.dump(
                    validated_metadata.model_dump(mode="json", exclude_none=True),
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

        return exported_path

    @staticmethod
    def _validate_metadata_consistency(
        result: FutureForecastResult,
        metadata: SingleForecastMetadataSchema,
    ) -> None:
        """Valida que o sidecar de metadados é consistente com o envelope de resultado do forecast artefact."""
        if metadata.horizon != result.horizon:
            raise DataError("Forecast metadata is inconsistent with the forecast artifact horizon.")

        if metadata.forecast_rows is not None:
            if metadata.forecast_rows != len(result.predictions):
                raise DataError("Forecast metadata is inconsistent with the forecast artifact row count.")

        if metadata.target is None:
            raise DataError("Forecast metadata is inconsistent with the forecast artifact target.")

    def export_multimodel_forecast(
        self,
        dataframe: pd.DataFrame,
        metadata: dict,
        filename: str = "future_forecast.csv",
        date_column: str = "date",
    ) -> Path:
        """Exporta um artefato multimodelo e seu sidecar auditável."""
        self._validate_dataframe(dataframe)
        if date_column not in dataframe.columns:
            raise DataError(f"Forecast date column '{date_column}' was not found.")

        validated_metadata = MetadataSchema.model_validate(metadata)
        model_columns = [column for column in dataframe.columns if column != date_column]
        if model_columns != validated_metadata.models:
            raise DataError(
                "Forecast metadata models are inconsistent with the forecast artifact columns."
            )
        if validated_metadata.horizon != len(dataframe):
            raise DataError(
                "Forecast metadata is inconsistent with the forecast artifact horizon."
            )
        if validated_metadata.forecast_rows is not None and validated_metadata.forecast_rows != len(dataframe):
            raise DataError(
                "Forecast metadata is inconsistent with the forecast artifact row count."
            )

        exported_path = self.export_csv(dataframe, filename=filename)
        metadata_path = self.output_directory / "forecast_metadata.json"
        with metadata_path.open(mode="w", encoding="utf-8") as file:
            json.dump(
                validated_metadata.model_dump(mode="json", exclude_none=True),
                file,
                indent=4,
                ensure_ascii=False,
            )
        return exported_path

    def export_parquet(
        self,
        dataframe: pd.DataFrame,
        filename: str = "forecast.parquet",
    ) -> Path:
        """exporta os resultados de previsão para parquet."""

        self._validate_dataframe(dataframe)
        filename = self._validate_filename(filename)

        output_path = self._prepare_output_path(
            filename,
        )

        dataframe.to_parquet(
            output_path,
            index=False,
        )

        return output_path

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:
        """valida o dataframe de previsão."""

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise DataError(
                "Cannot export an empty forecast dataframe."
            )

    @staticmethod
    def _validate_filename(
        filename: str,
    ) -> str:
        """valida o nome do arquivo de saída."""

        if not isinstance(
            filename,
            str,
        ) or not filename.strip():
            raise DataError(
                "Output filename must be a non-empty string."
            )

        path = Path(filename)

        if path.name != filename:
            raise DataError(
                "Output filename must not contain directories."
            )

        return filename

    def _prepare_output_path(
        self,
        filename: str,
    ) -> Path:
        """cria o diretório de saída e retorna o caminho do arquivo."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self.output_directory / filename