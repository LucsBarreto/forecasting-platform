"""
Tests for baseline forecasting model.
"""

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.models.baseline import BaselineModel


def create_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample training data."""

    X = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "COD CLIENTE": [
                "C001",
                "C001",
                "C001",
                "C002",
                "C002",
                "C002",
            ],
            "COD ITEM": [
                "P001",
                "P001",
                "P001",
                "P001",
                "P001",
                "P001",
            ],
        }
    )

    y = pd.Series(
        [
            10.0,
            20.0,
            30.0,
            100.0,
            200.0,
            300.0,
        ],
        name="VOLUME",
    )

    return X, y


def test_model_name() -> None:
    """Baseline model must expose its name."""

    model = BaselineModel()

    assert model.name == "baseline"


def test_fit_returns_model() -> None:
    """Fit must return the model instance."""

    X, y = create_dataframe()

    model = BaselineModel()

    result = model.fit(X, y)

    assert result is model


def test_global_baseline_predicts_last_value() -> None:
    """Global baseline must predict the last observed value."""

    X, y = create_dataframe()

    model = BaselineModel()

    model.fit(
        X,
        y,
    )

    prediction_DATA = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-04",
                    "2025-01-05",
                ]
            )
        }
    )

    predictions = model.predict(
        prediction_DATA,
    )

    assert predictions.tolist() == [
        300.0,
        300.0,
    ]


def test_grouped_baseline_predicts_last_group_value() -> None:
    """Grouped baseline must keep values isolated between groups."""

    X, y = create_dataframe()

    model = BaselineModel(
        group_columns=[
            "COD CLIENTE",
        ],
    )

    model.fit(
        X,
        y,
    )

    prediction_DATA = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-04",
                    "2025-01-04",
                ]
            ),
            "COD CLIENTE": [
                "C001",
                "C002",
            ],
        }
    )

    predictions = model.predict(
        prediction_DATA,
    )

    assert predictions.tolist() == [
        30.0,
        300.0,
    ]


def test_grouped_baseline_supports_multiple_group_columns() -> None:
    """Baseline must support multiple grouping columns."""

    X, y = create_dataframe()

    model = BaselineModel(
        group_columns=[
            "COD CLIENTE",
            "COD ITEM",
        ],
    )

    model.fit(
        X,
        y,
    )

    prediction_DATA = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-04",
                    "2025-01-04",
                ]
            ),
            "COD CLIENTE": [
                "C001",
                "C002",
            ],
            "COD ITEM": [
                "P001",
                "P001",
            ],
        }
    )

    predictions = model.predict(
        prediction_DATA,
    )

    assert predictions.tolist() == [
        30.0,
        300.0,
    ]


def test_prediction_preserves_index() -> None:
    """Predictions must preserve the input index."""

    X, y = create_dataframe()

    model = BaselineModel()

    model.fit(
        X,
        y,
    )

    prediction_DATA = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-04",
                    "2025-01-05",
                ]
            )
        },
        index=[10, 20],
    )

    predictions = model.predict(
        prediction_DATA,
    )

    pd.testing.assert_index_equal(
        predictions.index,
        prediction_DATA.index,
    )


def test_prediction_has_correct_length() -> None:
    """Prediction count must match input rows."""

    X, y = create_dataframe()

    model = BaselineModel()

    model.fit(
        X,
        y,
    )

    predictions = model.predict(
        X,
    )

    assert len(predictions) == len(X)


def test_predict_before_fit_raises_error() -> None:
    """Prediction before fitting must raise an error."""

    model = BaselineModel()

    X, _ = create_dataframe()

    with pytest.raises(
        DataValidationError,
        match="must be fitted",
    ):
        model.predict(X)


def test_empty_training_dataframe_raises_error() -> None:
    """Empty training dataframe must raise an error."""

    X, y = create_dataframe()

    X = X.iloc[0:0]
    y = y.iloc[0:0]

    model = BaselineModel()

    with pytest.raises(
        DataValidationError,
        match="cannot be empty",
    ):
        model.fit(X, y)


def test_mismatched_training_lengths_raise_error() -> None:
    """Features and target must have equal lengths."""

    X, y = create_dataframe()

    y = y.iloc[:-1]

    model = BaselineModel()

    with pytest.raises(
        DataValidationError,
        match="same length",
    ):
        model.fit(X, y)


def test_missing_date_column_raises_error() -> None:
    """Missing date column must raise an error."""

    X, y = create_dataframe()

    X = X.drop(
        columns=["DATA"],
    )

    model = BaselineModel()

    with pytest.raises(
        DataValidationError,
        match="Date column",
    ):
        model.fit(X, y)


def test_missing_group_column_raises_error() -> None:
    """Missing group column must raise an error."""

    X, y = create_dataframe()

    model = BaselineModel(
        group_columns=[
            "COD CLIENTE",
            "missing_column",
        ],
    )

    with pytest.raises(
        DataValidationError,
        match="Missing grouping columns",
    ):
        model.fit(X, y)


def test_missing_prediction_group_raises_error() -> None:
    """Unknown prediction group must raise an error."""

    X, y = create_dataframe()

    model = BaselineModel(
        group_columns=[
            "COD CLIENTE",
        ],
    )

    model.fit(X, y)

    prediction_DATA = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                ["2025-01-04"]
            ),
            "COD CLIENTE": [
                "C999",
            ],
        }
    )

    with pytest.raises(
        DataValidationError,
        match="No baseline value found",
    ):
        model.predict(prediction_DATA)


def test_get_params() -> None:
    """Model parameters must be exposed."""

    model = BaselineModel(
        date_column="date",
        target_column="target",
        group_columns=["group"],
    )

    params = model.get_params()

    assert params == {
        "date_column": "date",
        "target_column": "target",
        "group_columns": ["group"],
    }


def test_fit_predict() -> None:
    """fit_predict must fit and generate predictions."""

    X, y = create_dataframe()

    model = BaselineModel()

    predictions = model.fit_predict(
        X,
        y,
    )

    assert isinstance(
        predictions,
        pd.Series,
    )

    assert predictions.tolist() == [
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
    ]