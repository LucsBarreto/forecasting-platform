"""
ponto de entrada da aplicação.

este módulo é responsável por compor e executar
a aplicação de forecasting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.core.utils.output_manager import OutputManager
from src.export.forecast_exporter import ForecastExporter
from src.export.model_exporter import ModelExporter
from src.feature_selection.selector import FeatureSelector
from src.ml.evaluation import ModelEvaluator
from src.ml.forecast import Forecaster
from src.ml.training.cross_validation import TemporalSplitter
from src.ml.training.model_selection import ModelSelector
from src.ml.training.training_manager import TrainingManager
from src.pipelines.evaluate_pipeline import EvaluatePipeline
from src.pipelines.predict_pipeline import PredictPipeline
from src.pipelines.train_pipeline import TrainingPipeline
from src.preprocessing.cleaning import CleaningProcessor
from src.preprocessing.datetime import DatetimeProcessor
from src.preprocessing.missing import MissingValueProcessor
from src.preprocessing.preprocessor import PreprocessingPipeline
from src.preprocessing.typing import TypeConverter


def main() -> None:
    """
    executa a aplicação de forecasting.

    responsabilidades
    ------------------
    - criar e configurar as dependências da aplicação.
    - configurar os pipelines de pré-processamento, treinamento,
      avaliação e predição.
    - executar o treinamento dos modelos.
    - executar as predições e a avaliação dos modelos.
    - salvar as métricas e previsões geradas.
    - atualizar o status da execução.
    """

    output_manager = OutputManager()

    output_manager.create_run()

    try:
        preprocessing = _create_preprocessing()

        training_pipeline = TrainingPipeline(
            output_manager=output_manager,
            preprocessing=preprocessing,
            feature_selector=FeatureSelector(
                features=_resolve_feature_columns(
                    _resolve_target_column(),
                ),
            ),
            temporal_splitter=TemporalSplitter(
                train_size=settings.forecast.validation.train_size,
                validation_size=settings.forecast.validation.validation_size,
                test_size=settings.forecast.validation.test_size,
                date_column=settings.data.date_column,
            ),
            training_manager=TrainingManager(
                selector=ModelSelector(),
            ),
            model_exporter=ModelExporter(
                output_directory=output_manager.get_models_dir(),
            ),
        )

        evaluator = _create_evaluator()

        evaluate_pipeline = EvaluatePipeline(
            evaluator=evaluator,
        )

        predict_pipeline = _create_predict_pipeline()

        source = _resolve_input_sources()

        training_result = training_pipeline.run(
            source,
            target_column=_resolve_target_column(),
        )

        models = training_result.models

        if not models:
            raise RuntimeError(
                "Training completed without producing any models."
            )

        test_data = training_result.test_data
        if test_data is None:
            raise RuntimeError(
                "Training completed without a temporal test dataset."
            )

        target_column = _resolve_target_column()
        feature_columns = training_result.selected_features

        features = test_data.loc[:, feature_columns]

        target = test_data[
            target_column
        ]

        prediction_result = predict_pipeline.run(
            models=models,
            features=features,
        )

        evaluation_result = evaluate_pipeline.run(
            y_true=target,
            predictions=prediction_result.predictions,
        )

        _save_metrics(
            output_manager,
            evaluation_result.metrics,
        )

        _save_forecasts(
            output_manager,
            test_data,
            prediction_result.predictions,
        )

        output_manager.update_status(
            "SUCCESS",
        )

    except Exception:
        output_manager.update_status(
            "FAILED",
        )
        raise


def _resolve_input_sources() -> Path | list[Path]:
    """
    resolve as fontes de dados de entrada configuradas.

    returns
    -------
    Path | list[Path]
        caminho de uma única fonte ou lista contendo todos os arquivos
        correspondentes encontrados no diretório de entrada.

    raises
    ------
    filenotfounderror
        se nenhum arquivo de entrada for encontrado no caminho configurado.
    """

    input_path = Path(
        settings.data.input_path,
    )

    if input_path.is_file():
        return input_path

    files = sorted(
        input_path.glob(
            settings.data.file_pattern,
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No input files found in: {input_path}"
        )

    return files if len(files) > 1 else files[0]


def _resolve_target_column() -> str:
    """
    resolve a coluna alvo utilizada pela aplicação.

    returns
    -------
    str
        nome da coluna alvo configurada.

    raises
    ------
    valueerror
        se nenhuma coluna alvo válida estiver configurada.
    """

    targets = getattr(settings.forecast, "targets", [])
    target_column = targets[0] if targets else None

    if not isinstance(
        target_column,
        str,
    ) or not target_column.strip():
        raise ValueError(
            "A valid target column must be configured."
        )

    return target_column


def _resolve_feature_columns(target_column: str) -> list[str]:
    """
    resolve as features configuradas ou as features numéricas para um alvo.

    parameters
    ----------
    target_column
        nome da coluna alvo que não deve ser utilizada como feature.

    returns
    -------
    list[str]
        lista contendo as colunas selecionadas para utilização como features.
    """

    configured = settings.features.selection.features
    if configured:
        return [feature for feature in configured if feature != target_column]

    numeric_types = {
        "int64",
        "float64",
        "int32",
        "float32",
    }
    features = [
        column
        for column, dtype in settings.data.schema_config.dtypes.items()
        if column != target_column and dtype in numeric_types
    ]
    return [settings.data.date_column, *features]


def _create_preprocessing() -> PreprocessingPipeline:
    """
    cria o pipeline completo de pré-processamento.

    returns
    -------
    PreprocessingPipeline
        pipeline configurado com os processadores de limpeza,
        valores ausentes, tipagem e datas.
    """

    return PreprocessingPipeline(
        cleaning=CleaningProcessor(),
        missing=MissingValueProcessor(),
        typing=TypeConverter(),
        datetime=DatetimeProcessor(),
    )


def _create_evaluator() -> ModelEvaluator:
    """
    cria o avaliador de modelos utilizando a configuração da aplicação.

    returns
    -------
    ModelEvaluator
        avaliador configurado com a métrica definida nas configurações.
    """

    return ModelEvaluator(
        metric=settings.models.evaluation.metric,
    )


def _create_predict_pipeline() -> PredictPipeline:
    """
    cria o pipeline de predição.

    returns
    -------
    PredictPipeline
        pipeline configurado para execução das predições.
    """

    return PredictPipeline(
        forecaster=Forecaster(),
    )


def _orchestrate_future_forecast(
    predict_pipeline: PredictPipeline,
    model: Any,
    history_frame: pd.DataFrame,
    selected_features: list[str],
    available_features: list[str] | None = None,
    horizon: int = 1,
    target_column: str = "target",
    time_column: str = "date",
) -> Any:
    """
    Composição mínima do entrypoint para o contrato de forecast futuro.

    O papel de main.py continua sendo apenas compor a chamada ao
    pipeline de predição, que mantém o detalhe de execução do contrato
    futuro internamente.
    """
    return predict_pipeline.run_future_forecast(
        model=model,
        history_frame=history_frame,
        selected_features=selected_features,
        available_features=available_features,
        horizon=horizon,
        target_column=target_column,
        time_column=time_column,
    )


def _save_metrics(
    output_manager: OutputManager,
    metrics: dict[str, float],
) -> Path:
    """
    salva as métricas de avaliação da execução atual.

    parameters
    ----------
    output_manager
        gerenciador de saída da execução atual.

    metrics
        métricas de avaliação indexadas pelo nome do modelo.

    returns
    -------
    Path
        caminho para o arquivo de métricas gerado.
    """

    import json

    output_path = (
        output_manager.get_metrics_dir()
        / "metrics.json"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def _build_future_date_matrix(
    test_data: Any,
    forecast_horizon: int,
) -> list[pd.Timestamp]:
    """constrói uma matriz de datas futuras mensais a partir do último registro do teste."""

    import pandas as pd

    last_date = pd.to_datetime(
        test_data[settings.data.date_column].max(),
    )

    start = last_date + pd.offsets.MonthBegin(1)
    future_dates = pd.date_range(
        start=start,
        periods=forecast_horizon,
        freq="MS",
    )

    return list(future_dates)


def _save_forecasts(
    output_manager: OutputManager,
    test_data: Any,
    predictions: dict[str, Any],
) -> tuple[Path, Path]:
    """
    salva o arquivo de avaliação/teste e também o arquivo futuro de previsão.

    parameters
    ----------
    output_manager
        gerenciador de saída da execução atual.

    test_data
        dados utilizados no período de teste.

    predictions
        previsões geradas pelos modelos, indexadas pelo nome do modelo.

    returns
    -------
    tuple[path, path]
        caminhos dos arquivos de teste e de previsões futuras.
    """

    import pandas as pd

    test_forecast = pd.DataFrame(
        {
            settings.data.date_column: test_data[
                settings.data.date_column
            ].to_numpy(),
            **{
                model_name: prediction.to_numpy()
                for model_name, prediction in predictions.items()
            },
        }
    )

    test_path = ForecastExporter(
        output_directory=output_manager.get_forecasts_dir(),
    ).export_csv(
        test_forecast,
        filename="test_forecast.csv",
    )

    max_horizon = int(settings.forecast.forecast_horizon.maximum)
    future_dates = _build_future_date_matrix(
        test_data,
        max_horizon,
    )

    future_forecast = pd.DataFrame(
        {
            settings.data.date_column: future_dates,
        }
    )

    for model_name, prediction in predictions.items():
        future_forecast[model_name] = float(
            prediction.to_numpy()[-1]
        )

    future_metadata = {
        "run_id": output_manager.run_id,
        "target": _resolve_target_column(),
        "models": list(predictions),
        "horizon": len(future_forecast),
        "period": {
            "start": future_forecast[settings.data.date_column].min(),
            "end": future_forecast[settings.data.date_column].max(),
        },
        "version": "2.0.0",
        "timestamps": {
            "started_at": datetime.now(timezone.utc),
            "finished_at": datetime.now(timezone.utc),
        },
        "forecast_rows": len(future_forecast),
    }

    future_path = ForecastExporter(
        output_directory=output_manager.get_forecasts_dir(),
    ).export_multimodel_forecast(
        future_forecast,
        filename="future_forecast.csv",
        metadata=future_metadata,
        date_column=settings.data.date_column,
    )

    return test_path, future_path


if __name__ == "__main__":
    main()
