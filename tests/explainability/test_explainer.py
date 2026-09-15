"""
Tests for ModelExplainer.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions.modeling import ModelingError
from src.explainability import ModelExplainer


class MockEstimator:
    """Mock estimator exposing feature importance."""

    def __init__(
        self,
        importance: list[float],
    ) -> None:
        self.feature_importances_ = np.array(
            importance,
        )


class MockModel:
    """Mock model wrapper."""

    def __init__(
        self,
        importance: list[float],
    ) -> None:
        self.model = MockEstimator(
            importance,
        )


def test_feature_importance_returns_dataframe() -> None:
    """Test that feature importance returns a dataframe."""

    explainer = ModelExplainer()

    result = explainer.feature_importance(
        model=MockEstimator(
            [0.2, 0.5, 0.3],
        ),
        feature_names=[
            "feature_a",
            "feature_b",
            "feature_c",
        ],
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_feature_importance_returns_expected_columns() -> None:
    """Test the result columns."""

    explainer = ModelExplainer()

    result = explainer.feature_importance(
        model=MockEstimator(
            [0.2, 0.5],
        ),
        feature_names=[
            "feature_a",
            "feature_b",
        ],
    )

    assert list(
        result.columns,
    ) == [
        "feature",
        "importance",
    ]


def test_feature_importance_sorts_descending() -> None:
    """Test that features are sorted by importance."""

    explainer = ModelExplainer()

    result = explainer.feature_importance(
        model=MockEstimator(
            [0.2, 0.8, 0.5],
        ),
        feature_names=[
            "feature_a",
            "feature_b",
            "feature_c",
        ],
    )

    assert result["feature"].tolist() == [
        "feature_b",
        "feature_c",
        "feature_a",
    ]

    assert result["importance"].tolist() == [
        0.8,
        0.5,
        0.2,
    ]


def test_feature_importance_preserves_feature_association() -> None:
    """Test that importance values remain associated with features."""

    explainer = ModelExplainer()

    result = explainer.feature_importance(
        model=MockEstimator(
            [0.1, 0.7, 0.2],
        ),
        feature_names=[
            "VOLUME",
            "VALOR",
            "clientes",
        ],
    )

    expected = {
        "VOLUME": 0.1,
        "VALOR": 0.7,
        "clientes": 0.2,
    }

    actual = dict(
        zip(
            result["feature"],
            result["importance"],
        )
    )

    assert actual == expected


def test_feature_importance_accepts_model_wrapper() -> None:
    """Test that model wrappers are supported."""

    explainer = ModelExplainer()

    model = MockModel(
        [0.25, 0.75],
    )

    result = explainer.feature_importance(
        model=model,
        feature_names=[
            "feature_a",
            "feature_b",
        ],
    )

    assert result.iloc[0]["feature"] == "feature_b"
    assert result.iloc[0]["importance"] == 0.75


def test_empty_feature_names_raise_error() -> None:
    """Test that empty feature names are rejected."""

    explainer = ModelExplainer()

    with pytest.raises(
        ModelingError,
        match="No feature names were provided",
    ):
        explainer.feature_importance(
            model=MockEstimator(
                [],
            ),
            feature_names=[],
        )


def test_duplicated_feature_names_raise_error() -> None:
    """Test that duplicated feature names are rejected."""

    explainer = ModelExplainer()

    with pytest.raises(
        ModelingError,
        match="Duplicated feature names are not allowed",
    ):
        explainer.feature_importance(
            model=MockEstimator(
                [0.5, 0.5],
            ),
            feature_names=[
                "VOLUME",
                "VOLUME",
            ],
        )


def test_empty_feature_name_raises_error() -> None:
    """Test that empty feature names are rejected."""

    explainer = ModelExplainer()

    with pytest.raises(
        ModelingError,
        match="Feature names must be non-empty strings",
    ):
        explainer.feature_importance(
            model=MockEstimator(
                [0.5, 0.5],
            ),
            feature_names=[
                "VOLUME",
                "",
            ],
        )


def test_mismatched_importance_length_raises_error() -> None:
    """Test that mismatched importance length is rejected."""

    explainer = ModelExplainer()

    with pytest.raises(
        ModelingError,
        match="number of feature importance values",
    ):
        explainer.feature_importance(
            model=MockEstimator(
                [0.5, 0.5],
            ),
            feature_names=[
                "VOLUME",
                "VALOR",
                "clientes",
            ],
        )


def test_model_without_feature_importance_raises_error() -> None:
    """Test models without feature importance."""

    explainer = ModelExplainer()

    model = object()

    with pytest.raises(
        ModelingError,
        match="Model does not expose feature importance",
    ):
        explainer.feature_importance(
            model=model,
            feature_names=[
                "VOLUME",
            ],
        )


def test_none_model_raises_error() -> None:
    """Test that None models are rejected."""

    explainer = ModelExplainer()

    with pytest.raises(
        ModelingError,
        match="A model is required",
    ):
        explainer.feature_importance(
            model=None,
            feature_names=[
                "VOLUME",
            ],
        )