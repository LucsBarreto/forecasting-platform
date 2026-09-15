"""
End-to-end tests for the forecasting platform.

These tests validate the integration between:

TrainingPipeline
    ↓
PredictPipeline
    ↓
EvaluatePipeline

The input dataset is written to a temporary CSV file so that the
real TrainingPipeline data-loading flow is exercised.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ml.evaluation import ModelEvaluator
from src.pipelines.evaluate_pipeline import (
    EvaluatePipeline,
    EvaluationPipelineResult,
)
from src.pipelines.predict_pipeline import (
    PredictPipeline,
    PredictionPipelineResult,
)
from src.pipelines.train_pipeline import (
    TrainingPipeline,
    TrainingPipelineResult,
)


class FakeModel:
    """
    Simple deterministic model used by the end-to-end tests.

    The model predicts the mean of the training target for every
    input row.
    """

    def __init__(self) -> None:
        """Initialize the fake model."""

        self.mean_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "FakeModel":
        """Fit the model using the mean target value."""

        self.mean_ = float(
            y.mean(),
        )

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Generate deterministic predictions."""

        if self.mean_ is None:
            raise RuntimeError(
                "FakeModel must be fitted before prediction.",
            )

        return pd.Series(
            self.mean_,
            index=X.index,
        )


class FakePredictor:
    """
    Predictor adapter used by PredictPipeline.
    """

    @staticmethod
    def predict(
        model: FakeModel,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions using the trained model.

        PredictPipeline's single-model API expects a dataframe.
        """

        predictions = model.predict(
            features,
        )

        return pd.DataFrame(
            {
                "prediction": predictions,
            },
            index=features.index,
        )


class FakeFeatureSelector:
    """
    Feature selector used by the training integration test.
    """

    def select(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the configured feature columns."""

        return dataframe[
            [
                "feature_1",
                "feature_2",
            ]
        ].copy()


class FakeTemporalSplit:
    """
    Result produced by the fake temporal splitter.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Initialize temporal split."""

        self.train = train
        self.validation = validation
        self.test = test


class FakeTemporalSplitter:
    """
    Deterministic temporal splitter.

    The dataset is divided chronologically into:

    - 60% training
    - 20% validation
    - 20% test
    """

    def split(
        self,
        dataframe: pd.DataFrame,
    ) -> FakeTemporalSplit:
        """Split dataframe chronologically."""

        total_rows = len(dataframe)

        train_end = int(
            total_rows * 0.6,
        )

        validation_end = int(
            total_rows * 0.8,
        )

        return FakeTemporalSplit(
            train=dataframe.iloc[
                :train_end
            ].copy(),
            validation=dataframe.iloc[
                train_end:validation_end
            ].copy(),
            test=dataframe.iloc[
                validation_end:
            ].copy(),
        )


class FakeTrainingResult:
    """
    Result returned by FakeTrainingManager.
    """

    def __init__(
        self,
        models: dict[str, FakeModel],
    ) -> None:
        """Initialize training result."""

        self.models = models


class FakeTrainingManager:
    """
    Minimal training manager used by the integration test.
    """

    def __init__(
        self,
        model: FakeModel,
    ) -> None:
        """Initialize the training manager."""

        self.model = model

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: dict | None = None,
    ) -> FakeTrainingResult:
        """Train the fake model."""

        fitted_model = self.model.fit(
            X,
            y,
        )

        return FakeTrainingResult(
            models={
                "fake_model": fitted_model,
            },
        )


class FakePreprocessing:
    """
    Minimal preprocessing implementation.

    The processor preserves the dataset structure so that the
    downstream pipeline components can operate on it.
    """

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a copy of the input dataframe."""

        return dataframe.copy()


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """
    Return deterministic input data.
    """

    return pd.DataFrame(
        {
            "DATA": pd.date_range(
                "2026-01-01",
                periods=10,
                freq="D",
            ),
            "feature_1": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "feature_2": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
            ],
            "target": [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
            ],
        },
    )


@pytest.fixture
def source(
    tmp_path: Path,
    dataframe: pd.DataFrame,
) -> Path:
    """
    Create a temporary CSV input file.

    This allows the real TrainingPipeline._load_data()
    and DataSourceFactory to be exercised.
    """

    path = tmp_path / "data.csv"

    dataframe.to_csv(
        path,
        index=False,
    )

    return path


@pytest.fixture
def training_pipeline() -> TrainingPipeline:
    """
    Create a training pipeline for integration testing.
    """

    return TrainingPipeline(
        output_manager=None,  # type: ignore[arg-type]
        preprocessing=FakePreprocessing(),
        feature_engineering=None,
        feature_selector=FakeFeatureSelector(),
        temporal_splitter=FakeTemporalSplitter(),
        training_manager=FakeTrainingManager(
            FakeModel(),
        ),
        model_exporter=None,
    )


@pytest.fixture
def prediction_pipeline() -> PredictPipeline:
    """
    Create a prediction pipeline for integration testing.
    """

    return PredictPipeline(
        predictor=FakePredictor(),
    )


@pytest.fixture
def evaluation_pipeline() -> EvaluatePipeline:
    """
    Create an evaluation pipeline for integration testing.
    """

    return EvaluatePipeline(
        evaluator=ModelEvaluator(
            metric="mae",
        ),
    )


def test_end_to_end_training_prediction_and_evaluation(
    training_pipeline: TrainingPipeline,
    prediction_pipeline: PredictPipeline,
    evaluation_pipeline: EvaluatePipeline,
    source: Path,
) -> None:
    """
    Test the complete forecasting flow.

    The test validates that:

    1. The real datasource loads the CSV.
    2. TrainingPipeline preprocesses the data.
    3. Features are selected.
    4. Data is split chronologically.
    5. A model is trained.
    6. PredictPipeline generates predictions.
    7. EvaluatePipeline calculates the metric.
    """

    training_result = training_pipeline.run(
        source=source,
        target_column="target",
    )

    assert isinstance(
        training_result,
        TrainingPipelineResult,
    )

    assert "fake_model" in (
        training_result.models
    )

    assert isinstance(
        training_result.training_data,
        pd.DataFrame,
    )

    assert not training_result.training_data.empty

    assert isinstance(
        training_result.validation_data,
        pd.DataFrame,
    )

    assert isinstance(
        training_result.test_data,
        pd.DataFrame,
    )

    assert len(
        training_result.training_data,
    ) == 10

    assert len(
        training_result.validation_data,
    ) == 2

    assert len(
        training_result.test_data,
    ) == 2

    model = training_result.models[
        "fake_model"
    ]

    features = (
        training_result.training_data.drop(
            columns=["target"],
        )
    )

    prediction_result = prediction_pipeline.run(
        model=model,
        features=features,
        export=False,
    )

    assert isinstance(
        prediction_result,
        PredictionPipelineResult,
    )

    assert isinstance(
        prediction_result.predictions,
        pd.DataFrame,
    )

    assert len(
        prediction_result.predictions,
    ) == len(features)

    assert (
        "prediction"
        in prediction_result.predictions.columns
    )

    y_true = training_result.training_data[
        "target"
    ]

    y_pred = prediction_result.predictions[
        "prediction"
    ]

    evaluation_result = evaluation_pipeline.run(
        y_true=y_true,
        predictions={
            "fake_model": y_pred,
        },
    )

    assert isinstance(
        evaluation_result,
        EvaluationPipelineResult,
    )

    assert "fake_model" in (
        evaluation_result.metrics
    )

    metric = evaluation_result.metrics[
        "fake_model"
    ]

    assert isinstance(
        metric,
        float,
    )

    assert metric >= 0.0


def test_end_to_end_preserves_prediction_index(
    training_pipeline: TrainingPipeline,
    prediction_pipeline: PredictPipeline,
    source: Path,
) -> None:
    """
    Test that prediction indices remain aligned with input data.
    """

    training_result = training_pipeline.run(
        source=source,
        target_column="target",
    )

    model = training_result.models[
        "fake_model"
    ]

    features = (
        training_result.training_data.drop(
            columns=["target"],
        )
    )

    prediction_result = prediction_pipeline.run(
        model=model,
        features=features,
        export=False,
    )

    predictions = (
        prediction_result.predictions[
            "prediction"
        ]
    )

    pd.testing.assert_index_equal(
        predictions.index,
        features.index,
    )


def test_end_to_end_produces_valid_metric(
    training_pipeline: TrainingPipeline,
    prediction_pipeline: PredictPipeline,
    evaluation_pipeline: EvaluatePipeline,
    source: Path,
) -> None:
    """
    Test that the complete flow produces the expected MAE.

    The fake model is trained on the first six target values:

        100, 110, 120, 130, 140, 150

    Their mean is:

        125

    Predictions for all ten rows are therefore 125.

    The absolute errors are:

        25, 15, 5, 5, 15, 25, 35, 45, 55, 65

    Resulting in:

        MAE = 29.0
    """

    training_result = training_pipeline.run(
        source=source,
        target_column="target",
    )

    model = training_result.models[
        "fake_model"
    ]

    features = (
        training_result.training_data.drop(
            columns=["target"],
        )
    )

    prediction_result = prediction_pipeline.run(
        model=model,
        features=features,
        export=False,
    )

    predictions = (
        prediction_result.predictions[
            "prediction"
        ]
    )

    evaluation_result = evaluation_pipeline.run(
        y_true=training_result.training_data[
            "target"
        ],
        predictions={
            "fake_model": predictions,
        },
    )

    metric = evaluation_result.metrics[
        "fake_model"
    ]

    assert metric == pytest.approx(
        29.0,
    )

    assert metric != float("inf")

    assert metric == metric

