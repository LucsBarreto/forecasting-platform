"""
Tests for the base model interface.
"""

import pandas as pd
import pytest

from src.ml.models.base_model import BaseModel


class DummyModel(BaseModel):
    """Concrete model used to test the BaseModel contract."""

    @property
    def name(self) -> str:
        """Return dummy model name."""

        return "dummy"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseModel:
        """Fit dummy model."""

        self.fitted = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Return deterministic predictions."""

        return pd.Series(
            [1.0] * len(X),
            index=X.index,
        )


def create_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample training data."""

    X = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [10.0, 20.0, 30.0],
        }
    )

    y = pd.Series(
        [100.0, 200.0, 300.0],
        name="target",
    )

    return X, y


def test_base_model_cannot_be_instantiated() -> None:
    """BaseModel must remain abstract."""

    with pytest.raises(TypeError):
        BaseModel()


def test_model_name() -> None:
    """Concrete models must expose a name."""

    model = DummyModel()

    assert model.name == "dummy"


def test_fit_returns_model() -> None:
    """Fit must return the fitted model instance."""

    X, y = create_dataframe()

    model = DummyModel()

    result = model.fit(X, y)

    assert result is model
    assert model.fitted is True


def test_predict_returns_series() -> None:
    """Predict must return a pandas Series."""

    X, _ = create_dataframe()

    model = DummyModel()

    result = model.predict(X)

    assert isinstance(result, pd.Series)


def test_predict_preserves_index() -> None:
    """Predictions must preserve the input index."""

    X, _ = create_dataframe()

    X.index = [10, 20, 30]

    model = DummyModel()

    result = model.predict(X)

    pd.testing.assert_index_equal(
        result.index,
        X.index,
    )


def test_predict_has_same_number_of_rows() -> None:
    """Prediction count must match input rows."""

    X, _ = create_dataframe()

    model = DummyModel()

    result = model.predict(X)

    assert len(result) == len(X)


def test_fit_predict() -> None:
    """fit_predict must fit and then predict."""

    X, y = create_dataframe()

    model = DummyModel()

    result = model.fit_predict(X, y)

    assert model.fitted is True
    assert isinstance(result, pd.Series)
    assert len(result) == len(X)


def test_get_params_returns_dictionary() -> None:
    """Base implementation must return a parameter dictionary."""

    model = DummyModel()

    result = model.get_params()

    assert isinstance(result, dict)
    assert result == {}