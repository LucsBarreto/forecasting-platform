"""
Tests for ModelTrainer.
"""

import pandas as pd
import pytest

from src.ml.models.base_model import BaseModel
from src.ml.training import ModelTrainer


class FakeModel(BaseModel):
    """Minimal model used to isolate ModelTrainer tests."""

    @property
    def name(self) -> str:
        """Return the fake model name."""

        return "fake_model"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseModel:
        """Return itself as a fitted model."""

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Return zero predictions."""

        return pd.Series(
            0.0,
            index=X.index,
        )


@pytest.fixture
def features() -> pd.DataFrame:
    """Return sample training features."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                ]
            ),
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [10.0, 20.0, 30.0, 40.0],
        }
    )


@pytest.fixture
def target() -> pd.Series:
    """Return sample training target."""

    return pd.Series(
        [100.0, 200.0, 300.0, 400.0],
        name="target",
    )


@pytest.fixture
def trainer() -> ModelTrainer:
    """Return a model trainer."""

    return ModelTrainer(
        model=FakeModel(),
    )


def test_trainer_stores_model() -> None:
    """Test that the trainer stores the configured model."""

    model = FakeModel()

    trainer = ModelTrainer(
        model=model,
    )

    assert trainer.model is model


def test_train_returns_model(
    trainer: ModelTrainer,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test that training returns a model."""

    result = trainer.train(
        features,
        target,
    )

    assert isinstance(
        result,
        BaseModel,
    )


def test_train_returns_configured_model(
    trainer: ModelTrainer,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test that training returns the configured model."""

    result = trainer.train(
        features,
        target,
    )

    assert result is trainer.model


def test_training_features_and_target_are_not_modified(
    trainer: ModelTrainer,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test that training does not modify input data."""

    original_features = features.copy(
        deep=True,
    )

    original_target = target.copy(
        deep=True,
    )

    trainer.train(
        features,
        target,
    )

    pd.testing.assert_frame_equal(
        features,
        original_features,
    )

    pd.testing.assert_series_equal(
        target,
        original_target,
    )


def test_empty_features_raise_error(
    trainer: ModelTrainer,
    target: pd.Series,
) -> None:
    """Test that empty features are rejected."""

    features = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="features cannot be empty",
    ):
        trainer.train(
            features,
            target,
        )


def test_empty_target_raises_error(
    trainer: ModelTrainer,
    features: pd.DataFrame,
) -> None:
    """Test that empty target is rejected."""

    target = pd.Series(
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="target cannot be empty",
    ):
        trainer.train(
            features,
            target,
        )


def test_mismatched_lengths_raise_error(
    trainer: ModelTrainer,
    features: pd.DataFrame,
) -> None:
    """Test that mismatched lengths are rejected."""

    target = pd.Series(
        [100.0, 200.0],
    )

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        trainer.train(
            features,
            target,
        )


def test_invalid_features_type_raises_error(
    trainer: ModelTrainer,
    target: pd.Series,
) -> None:
    """Test that invalid feature type is rejected."""

    with pytest.raises(
        TypeError,
        match="X must be a pandas DataFrame",
    ):
        trainer.train(
            [1, 2, 3],  # type: ignore[arg-type]
            target,
        )


def test_invalid_target_type_raises_error(
    trainer: ModelTrainer,
    features: pd.DataFrame,
) -> None:
    """Test that invalid target type is rejected."""

    with pytest.raises(
        TypeError,
        match="y must be a pandas Series",
    ):
        trainer.train(
            features,
            [1, 2, 3, 4],  # type: ignore[arg-type]
        )


def test_all_null_features_raise_error(
    trainer: ModelTrainer,
    target: pd.Series,
) -> None:
    """Test that completely null features are rejected."""

    features = pd.DataFrame(
        {
            "feature_1": [None, None],
            "feature_2": [None, None],
        }
    )

    with pytest.raises(
        ValueError,
        match="only null values",
    ):
        trainer.train(
            features,
            target.iloc[:2],
        )


def test_all_null_target_raises_error(
    trainer: ModelTrainer,
    features: pd.DataFrame,
) -> None:
    """Test that completely null target is rejected."""

    target = pd.Series(
        [None, None, None, None],
    )

    with pytest.raises(
        ValueError,
        match="only null values",
    ):
        trainer.train(
            features,
            target,
        )

