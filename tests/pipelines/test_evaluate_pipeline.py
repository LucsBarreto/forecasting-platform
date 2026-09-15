"""
Tests for EvaluatePipeline.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.pipelines.evaluate_pipeline import (
    EvaluationPipelineResult,
    EvaluatePipeline,
)


@pytest.fixture
def evaluator() -> MagicMock:
    """Return a mocked evaluator."""

    return MagicMock()


@pytest.fixture
def pipeline(
    evaluator: MagicMock,
) -> EvaluatePipeline:
    """Return a configured evaluation pipeline."""

    return EvaluatePipeline(
        evaluator=evaluator,
    )


@pytest.fixture
def y_true() -> pd.Series:
    """Return sample target values."""

    return pd.Series(
        [100, 200, 300, 400],
        name="target",
    )


def test_pipeline_name(
    pipeline: EvaluatePipeline,
) -> None:
    """Test pipeline name."""

    assert pipeline.name == "pipeline de avaliação"


def test_run_evaluates_all_models(
    pipeline: EvaluatePipeline,
    evaluator: MagicMock,
    y_true: pd.Series,
) -> None:
    """Test that all model predictions are evaluated."""

    linear_predictions = pd.Series(
        [110, 190, 310, 390],
    )

    random_forest_predictions = pd.Series(
        [105, 195, 305, 395],
    )

    predictions = {
        "linear_regression": linear_predictions,
        "random_forest": random_forest_predictions,
    }

    evaluator.evaluate.side_effect = [
        10.0,
        5.0,
    ]

    result = pipeline.run(
        y_true,
        predictions,
    )

    assert isinstance(
        result,
        EvaluationPipelineResult,
    )

    assert result.metrics == {
        "linear_regression": 10.0,
        "random_forest": 5.0,
    }

    assert evaluator.evaluate.call_count == 2

    first_call = evaluator.evaluate.call_args_list[0]
    second_call = evaluator.evaluate.call_args_list[1]

    assert first_call.args[0] is y_true
    assert first_call.args[1] is linear_predictions

    assert second_call.args[0] is y_true
    assert second_call.args[1] is random_forest_predictions


def test_run_returns_evaluator_metric(
    pipeline: EvaluatePipeline,
    evaluator: MagicMock,
    y_true: pd.Series,
) -> None:
    """Test that the evaluator metric is preserved."""

    predictions = {
        "baseline": pd.Series(
            [90, 210, 290, 410],
        ),
    }

    expected_metric = 10.0

    evaluator.evaluate.return_value = (
        expected_metric
    )

    result = pipeline.run(
        y_true,
        predictions,
    )

    assert result.metrics["baseline"] == (
        expected_metric
    )


def test_run_converts_metric_to_float(
    pipeline: EvaluatePipeline,
    evaluator: MagicMock,
    y_true: pd.Series,
) -> None:
    """Test that evaluator results are returned as floats."""

    predictions = {
        "baseline": pd.Series(
            [90, 210, 290, 410],
        ),
    }

    evaluator.evaluate.return_value = 10

    result = pipeline.run(
        y_true,
        predictions,
    )

    assert result.metrics["baseline"] == 10.0

    assert isinstance(
        result.metrics["baseline"],
        float,
    )


def test_run_does_not_modify_target(
    pipeline: EvaluatePipeline,
    evaluator: MagicMock,
    y_true: pd.Series,
) -> None:
    """Test that y_true is not modified."""

    original = y_true.copy()

    predictions = {
        "baseline": pd.Series(
            [90, 210, 290, 410],
        ),
    }

    evaluator.evaluate.return_value = 10.0

    pipeline.run(
        y_true,
        predictions,
    )

    pd.testing.assert_series_equal(
        y_true,
        original,
    )


def test_run_empty_target_raises_error(
    pipeline: EvaluatePipeline,
) -> None:
    """Test empty target validation."""

    with pytest.raises(
        ValueError,
        match="y_true cannot be empty",
    ):
        pipeline.run(
            pd.Series(
                dtype=float,
            ),
            {
                "baseline": pd.Series(
                    dtype=float,
                ),
            },
        )


def test_run_invalid_target_raises_error(
    pipeline: EvaluatePipeline,
) -> None:
    """Test invalid target validation."""

    with pytest.raises(
        TypeError,
        match="y_true must be a pandas Series",
    ):
        pipeline.run(
            [100, 200, 300],
            {
                "baseline": pd.Series(
                    [90, 210, 290],
                ),
            },
        )


def test_run_invalid_predictions_raises_error(
    pipeline: EvaluatePipeline,
    y_true: pd.Series,
) -> None:
    """Test invalid predictions validation."""

    with pytest.raises(
        TypeError,
        match="predictions must be a dictionary",
    ):
        pipeline.run(
            y_true,
            [],
        )


def test_run_empty_predictions_raises_error(
    pipeline: EvaluatePipeline,
    y_true: pd.Series,
) -> None:
    """Test empty predictions validation."""

    with pytest.raises(
        ValueError,
        match="predictions cannot be empty",
    ):
        pipeline.run(
            y_true,
            {},
        )


def test_run_propagates_evaluator_error(
    pipeline: EvaluatePipeline,
    evaluator: MagicMock,
    y_true: pd.Series,
) -> None:
    """Test evaluator errors are propagated."""

    evaluator.evaluate.side_effect = RuntimeError(
        "evaluation error",
    )

    predictions = {
        "baseline": pd.Series(
            [90, 210, 290, 410],
        ),
    }

    with pytest.raises(
        RuntimeError,
        match="evaluation error",
    ):
        pipeline.run(
            y_true,
            predictions,
        )