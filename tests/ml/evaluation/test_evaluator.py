"""
Tests for ModelEvaluator.
"""

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.evaluation import ModelEvaluator


@pytest.fixture
def y_true() -> pd.Series:
    """Return sample actual values."""

    return pd.Series(
        [100.0, 200.0, 300.0, 400.0],
        name="actual",
    )


@pytest.fixture
def y_pred() -> pd.Series:
    """Return sample predicted values."""

    return pd.Series(
        [90.0, 210.0, 330.0, 380.0],
        name="prediction",
    )


def test_evaluator_stores_metric() -> None:
    """Test that evaluator stores the configured metric."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    assert evaluator.metric == "mape"


def test_evaluate_returns_float(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Test that evaluation returns a float."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    assert isinstance(
        result,
        float,
    )


def test_mape_calculation(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Test MAPE calculation."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    expected = (
        (
            10.0
            + 5.0
            + 10.0
            + 5.0
        )
        / 4
    )

    assert result == pytest.approx(
        expected,
    )


def test_mape_ignores_zero_actual_values() -> None:
    """Test that zero actual values are excluded from MAPE."""

    y_true = pd.Series(
        [100.0, 0.0, 200.0],
    )

    y_pred = pd.Series(
        [90.0, 50.0, 220.0],
    )

    evaluator = ModelEvaluator(
        metric="mape",
    )

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    expected = (
        (
            10.0
            + 10.0
        )
        / 2
    )

    assert result == pytest.approx(
        expected,
    )


def test_mape_all_zero_actual_values_raise_error() -> None:
    """Test that MAPE rejects all-zero actual values."""

    y_true = pd.Series(
        [0.0, 0.0],
    )

    y_pred = pd.Series(
        [10.0, 20.0],
    )

    evaluator = ModelEvaluator(
        metric="mape",
    )

    with pytest.raises(
        DataValidationError,
        match="all actual values are zero",
    ):
        evaluator.evaluate(
            y_true,
            y_pred,
        )


def test_mae_calculation(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Test MAE calculation."""

    evaluator = ModelEvaluator(
        metric="mae",
    )

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    assert result == pytest.approx(
        17.5,
    )

def test_mse_calculation(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Test MSE calculation."""

    evaluator = ModelEvaluator(
        metric="mse",
    )

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    assert result == pytest.approx(
        375.0,
    )


def test_rmse_calculation(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Test RMSE calculation."""

    evaluator = ModelEvaluator(
        metric="rmse",
    )

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    assert result == pytest.approx(
        19.364916731037084,
    )


def test_metric_is_case_insensitive(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Test that metric names are case insensitive."""

    evaluator = ModelEvaluator(
        metric="MAPE",
    )

    result = evaluator.evaluate(
        y_true,
        y_pred,
    )

    assert result == pytest.approx(
        7.5,
    )


def test_unsupported_metric_raises_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Test that unsupported metrics are rejected."""

    evaluator = ModelEvaluator(
        metric="unsupported",
    )

    with pytest.raises(
        DataValidationError,
        match="Unsupported evaluation metric",
    ):
        evaluator.evaluate(
            y_true,
            y_pred,
        )


def test_invalid_actual_type_raises_error(
    y_pred: pd.Series,
) -> None:
    """Test that invalid actual values type is rejected."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    with pytest.raises(
        TypeError,
        match="y_true must be a pandas Series",
    ):
        evaluator.evaluate(
            [100.0, 200.0],  # type: ignore[arg-type]
            y_pred,
        )


def test_invalid_prediction_type_raises_error(
    y_true: pd.Series,
) -> None:
    """Test that invalid prediction type is rejected."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    with pytest.raises(
        TypeError,
        match="y_pred must be a pandas Series",
    ):
        evaluator.evaluate(
            y_true,
            [90.0, 210.0],  # type: ignore[arg-type]
        )


def test_empty_actual_values_raise_error(
    y_pred: pd.Series,
) -> None:
    """Test that empty actual values are rejected."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    y_true = pd.Series(
        dtype=float,
    )

    with pytest.raises(
        DataValidationError,
        match="Actual target cannot be empty",
    ):
        evaluator.evaluate(
            y_true,
            y_pred.iloc[:0],
        )


def test_empty_predictions_raise_error(
    y_true: pd.Series,
) -> None:
    """Test that empty predictions are rejected."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    y_pred = pd.Series(
        dtype=float,
    )

    with pytest.raises(
        DataValidationError,
        match="Predictions cannot be empty",
    ):
        evaluator.evaluate(
            y_true,
            y_pred,
        )


def test_different_lengths_raise_error(
    y_true: pd.Series,
) -> None:
    """Test that different lengths are rejected."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    y_pred = pd.Series(
        [90.0, 210.0],
    )

    with pytest.raises(
        DataValidationError,
        match="same number of values",
    ):
        evaluator.evaluate(
            y_true,
            y_pred,
        )


def test_invalid_actual_values_raise_error(
    y_pred: pd.Series,
) -> None:
    """Test that invalid actual numeric values are rejected."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    y_true = pd.Series(
        [100.0, float("nan"), 300.0, 400.0],
    )

    with pytest.raises(
        DataValidationError,
        match="invalid numeric values",
    ):
        evaluator.evaluate(
            y_true,
            y_pred,
        )


def test_invalid_prediction_values_raise_error(
    y_true: pd.Series,
) -> None:
    """Test that invalid prediction numeric values are rejected."""

    evaluator = ModelEvaluator(
        metric="mape",
    )

    y_pred = pd.Series(
        [90.0, 210.0, float("inf"), 380.0],
    )

    with pytest.raises(
        DataValidationError,
        match="invalid numeric values",
    ):
        evaluator.evaluate(
            y_true,
            y_pred,
        )