"""testes do ponto de entrada da aplicação."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import main as main_module


def test_resolve_input_sources_returns_file(
    tmp_path: Path,
) -> None:
    """resolve uma origem de entrada explícita e retorna o arquivo correto."""

    source = tmp_path / "data.xlsb"
    source.touch()

    with patch.object(
        main_module.settings.data,
        "input_path",
        source,
    ):
        result = main_module._resolve_input_sources()

    assert result == source


def test_resolve_input_sources_returns_all_matching_files(
    tmp_path: Path,
) -> None:
    """resolve todos os arquivos correspondentes ao padrão configurado."""

    first = tmp_path / "first.xlsb"
    second = tmp_path / "second.xlsb"

    first.touch()
    second.touch()

    with patch.object(
        main_module.settings.data,
        "input_path",
        tmp_path,
    ), patch.object(
        main_module.settings.data,
        "file_pattern",
        "*.xlsb",
    ):
        result = main_module._resolve_input_sources()

    assert result == [first, second]


def test_resolve_input_sources_raises_when_no_files_exist(
    tmp_path: Path,
) -> None:
    """levanta erro quando não há arquivos de entrada compatíveis."""

    with patch.object(
        main_module.settings.data,
        "input_path",
        tmp_path,
    ), patch.object(
        main_module.settings.data,
        "file_pattern",
        "*.xlsb",
    ):
        with pytest.raises(
            FileNotFoundError,
            match="No input files found",
        ):
            main_module._resolve_input_sources()


def test_main_rejects_run_without_trained_models() -> None:
    """rejeita uma execução sem modelos treinados."""

    output_manager = MagicMock()

    training_pipeline = MagicMock()

    training_pipeline.run.return_value = MagicMock(
        models={},
        training_data=MagicMock(),
    )

    with patch(
        "main.OutputManager",
        return_value=output_manager,
    ), patch(
        "main.TrainingPipeline",
        return_value=training_pipeline,
    ), patch(
        "main.EvaluatePipeline",
    ), patch(
        "main.PredictPipeline",
    ), patch(
        "main._resolve_input_sources",
        return_value=Path("data.xlsb"),
    ), patch(
        "main._resolve_target_column",
        return_value="VOLUME",
    ), patch(
        "main._create_preprocessing",
        return_value=MagicMock(),
    ), patch(
        "main._create_evaluator",
        return_value=MagicMock(),
    ):

        with pytest.raises(
            RuntimeError,
            match="without producing any models",
        ):
            main_module.main()

    output_manager.create_run.assert_called_once()

    output_manager.update_status.assert_called_once_with("FAILED")


def test_main_marks_run_as_failed_on_error() -> None:
    """atualiza o status da execução como falha quando ocorre erro."""

    output_manager = MagicMock()

    with patch(
        "main.OutputManager",
        return_value=output_manager,
    ), patch(
        "main._resolve_input_sources",
        side_effect=RuntimeError(
            "input error",
        ),
    ):

        with pytest.raises(
            RuntimeError,
            match="input error",
        ):
            main_module.main()

    output_manager.create_run.assert_called_once()

    output_manager.update_status.assert_called_once_with(
        "FAILED",
    )


def test_main_propagates_training_error() -> None:
    """propaga erro de treinamento sem mascará-lo."""

    output_manager = MagicMock()
    training_pipeline = MagicMock()

    training_pipeline.run.side_effect = RuntimeError(
        "training error",
    )

    with patch(
        "main.OutputManager",
        return_value=output_manager,
    ), patch(
        "main.TrainingPipeline",
        return_value=training_pipeline,
    ), patch(
        "main.EvaluatePipeline",
    ), patch(
        "main.PredictPipeline",
    ), patch(
        "main._resolve_input_sources",
        return_value=Path("data.xlsb"),
    ), patch(
        "main._resolve_target_column",
        return_value="VOLUME",
    ), patch(
        "main._create_preprocessing",
        return_value=MagicMock(),
    ), patch(
        "main._create_evaluator",
        return_value=MagicMock(),
    ):
        with pytest.raises(
            RuntimeError,
            match="training error",
        ):
            main_module.main()

    output_manager.create_run.assert_called_once()

    output_manager.update_status.assert_called_once_with(
        "FAILED",
    )


def test_main_orchestrates_future_forecast_contract_via_predict_pipeline() -> None:
    """main.py must compose the future forecast call through the prediction pipeline without owning the forecast logic."""
    predict_pipeline = MagicMock()
    model = MagicMock()
    history_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "target": [10.0, 11.0, 12.0],
        }
    )

    result = main_module._orchestrate_future_forecast(
        predict_pipeline=predict_pipeline,
        model=model,
        history_frame=history_frame,
        selected_features=["lag_1", "lag_2"],
        available_features=["lag_1", "lag_2"],
        horizon=2,
        target_column="target",
        time_column="date",
    )

    assert result == predict_pipeline.run_future_forecast.return_value
    predict_pipeline.run_future_forecast.assert_called_once_with(
        model=model,
        history_frame=history_frame,
        selected_features=["lag_1", "lag_2"],
        available_features=["lag_1", "lag_2"],
        horizon=2,
        target_column="target",
        time_column="date",
    )