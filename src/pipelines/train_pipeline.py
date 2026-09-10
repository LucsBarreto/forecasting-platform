"""
pipeline de treinamento.

este módulo orquestra o carregamento de dados, pré-processamento,
engenharia de features, seleção de features, divisão temporal,
treinamento de modelos e exportação dos modelos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.utils.output_manager import OutputManager
from src.data_sources.factory import DataSourceFactory
from src.export.model_exporter import ModelExporter
from src.feature_selection.selector import FeatureSelector
from src.ml.training.cross_validation import TemporalSplitter
from src.ml.training.training_manager import TrainingManager
from src.preprocessing.preprocessor import PreprocessingPipeline

from .base_pipeline import BasePipeline


@dataclass(slots=True)
class TrainingPipelineResult:
    models: dict[str, Any] | None = None
    training_data: pd.DataFrame | None = None
    exported_models: dict[str, Any] = field(
        default_factory=dict
    )
    validation_data: pd.DataFrame | None = None
    test_data: pd.DataFrame | None = None
    selected_features: list[str] = field(
        default_factory=list
    )


class TrainingPipeline(BasePipeline):
    """
    pipeline responsável por preparar os dados e treinar os modelos.

    responsabilidades
    ------------------
    - carregar os dados de entrada.
    - executar o pré-processamento.
    - executar a engenharia de features quando configurada.
    - selecionar as features dos modelos.
    - dividir os dados cronologicamente.
    - treinar os modelos configurados.
    - exportar os modelos treinados.
    """

    def __init__(
        self,
        output_manager: OutputManager,
        preprocessing: PreprocessingPipeline,
        feature_engineering: Any | None = None,
        feature_selector: FeatureSelector | None = None,
        temporal_splitter: TemporalSplitter | None = None,
        training_manager: TrainingManager | None = None,
        model_exporter: ModelExporter | None = None,
    ) -> None:
        """inicializa o pipeline de treinamento."""

        super().__init__()

        self.output_manager = output_manager
        self.preprocessing = preprocessing
        self.feature_engineering = feature_engineering
        self.feature_selector = feature_selector
        self.temporal_splitter = temporal_splitter
        self.training_manager = training_manager
        self.model_exporter = model_exporter

    @property
    def name(self) -> str:
        """retorna o nome do pipeline."""

        return "treinando pipeline"

    def run(
        self,
        source: Path | list[Path],
        target_column: str | None = None,
        model_params: dict[str, dict[str, Any]] | None = None,
    ) -> pd.DataFrame | TrainingPipelineResult:
        """
        executa o pipeline de treinamento.

        quando os componentes de machine learning não estão configurados,
        o pipeline preserva seu comportamento original e retorna o
        dataframe pré-processado.

        quando todos os componentes de treinamento estão configurados,
        o pipeline também executa a seleção de features, divisão temporal,
        treinamento dos modelos e exportação dos modelos.

        parameters
        ----------
        source
            caminho para o arquivo de entrada.

        target_column
            coluna alvo utilizada no treinamento.

        model_params
            parâmetros opcionais específicos de cada modelo.

        returns
        -------
        pd.dataframe or trainingpipelineresult
            dataframe pré-processado quando os componentes de treinamento
            não estão configurados; caso contrário, modelos treinados,
            dados de treinamento, dados de validação e teste temporais
            e caminhos dos modelos exportados.
        """

        start = self._log_start()

        try:
            dataframe = self._load_data(source)

            dataframe = self.preprocessing.process(
                dataframe,
            )

            if not self._training_is_configured():
                self._log_finish(start)

                return dataframe

            if target_column is None:
                raise ValueError(
                    "target_column must be provided "
                    "when training is configured."
                )

            dataframe = self._apply_feature_engineering(
                dataframe,
            )

            X, y = self._prepare_training_data(
                dataframe,
                target_column,
            )

            split = self.temporal_splitter.split(
                dataframe,
            )

            X_train, y_train = self._prepare_split_data(
                split.train,
                target_column,
                X.columns,
            )

            training_result = self.training_manager.train(
                X_train,
                y_train,
                model_params=model_params,
            )

            exported_models = self._export_models(
                training_result,
            )

            self._log_finish(start)

            return TrainingPipelineResult(
                models=training_result.models,
                training_data=dataframe,
                exported_models=exported_models,
                validation_data=split.validation.copy(),
                test_data=split.test.copy(),
                selected_features=list(X.columns),
            )

        except Exception as exc:
            self._log_failure(exc)
            raise

    def _training_is_configured(self) -> bool:
        """verifica se todos os componentes de treinamento estão disponíveis."""

        return all(
            component is not None
            for component in (
                self.feature_selector,
                self.temporal_splitter,
                self.training_manager,
            )
        )

    def _apply_feature_engineering(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """executa a engenharia de features quando configurada."""

        if self.feature_engineering is None:
            return dataframe

        return self.feature_engineering.process(
            dataframe,
        )

    def _prepare_training_data(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """prepara as features e o alvo para treinamento."""

        if not isinstance(
            target_column,
            str,
        ) or not target_column.strip():
            raise ValueError(
                "target_column must be a non-empty string."
            )

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                "was not found."
            )

        target = dataframe[
            target_column
        ].copy()

        features = dataframe.drop(
            columns=[target_column],
        )

        selected_features = self.feature_selector.select(
            features,
        )

        return selected_features, target

    def _prepare_split_data(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        feature_columns: pd.Index,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """prepara x e y a partir de uma divisão temporal."""

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                "was not found in training split."
            )

        X = dataframe[
            feature_columns
        ].copy()

        y = dataframe[
            target_column
        ].copy()

        return X, y

    def _export_models(
        self,
        training_result: Any,
    ) -> dict[str, Path]:
        """
        exporta todos os modelos treinados.

        parameters
        ----------
        training_result
            resultado retornado pelo trainingmanager.

        returns
        -------
        dict[str, path]
            mapeamento entre os nomes dos modelos e os caminhos exportados.
        """

        if self.model_exporter is None:
            return {}

        exported_models: dict[str, Path] = {}

        for model_name, model in training_result.models.items():
            filename = (
                f"{model_name}.joblib"
            )

            exported_models[model_name] = (
                self.model_exporter.save(
                    model,
                    filename,
                )
            )

        return exported_models

    def _load_data(
        self,
        source: Path | list[Path],
    ) -> pd.DataFrame:
        """carrega o dataset."""

        self.logger.info(
            f"carregando dados do {source}"
        )

        sources = source if isinstance(source, list) else [source]
        dataframes: list[pd.DataFrame] = []

        for input_path in sources:
            datasource = DataSourceFactory.create(input_path)
            dataframes.append(datasource.read(input_path))

        dataframe = pd.concat(
            dataframes,
            ignore_index=True,
        )

        self.logger.info(
            f"dataset carregado com sucesso "
            f"({len(dataframe):,} linhas)"
        )

        return dataframe