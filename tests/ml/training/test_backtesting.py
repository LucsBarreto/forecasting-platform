import pandas as pd
import pytest

from src.feature_selection.selector import FeatureSelector
from src.ml.training.backtesting import BacktestResult, Backtester, BacktestFold
from src.ml.training.final_training import FinalModelTrainer, FinalTrainingResult
from src.ml.training.model_selection import BacktestModelSelector


def test_backtester_generates_expanding_temporal_folds_with_contract_defaults():
    dataframe = pd.DataFrame(
        {
            "DATA": pd.date_range("2024-01-01", periods=40, freq="MS"),
            "VOLUME": list(range(40)),
        }
    )

    backtester = Backtester(
        dataframe=dataframe,
        date_column="DATA",
        horizon=3,
        n_folds=3,
        min_training_history=24,
        gap=0,
        strategy="expanding",
    )

    folds = backtester.generate_folds()

    assert len(folds) == 3
    assert len(folds[0].train) == 24
    assert len(folds[0].validation) == 3
    assert len(folds[1].train) == 27
    assert len(folds[2].train) == 30

    assert folds[0].train_start == 0
    assert folds[0].train_end == 24
    assert folds[0].validation_start == 24
    assert folds[0].validation_end == 27


def test_backtester_run_fold_executes_a_real_fold_with_train_only_feature_selection():
    dataframe = pd.DataFrame(
        {
            "DATA": pd.date_range("2024-01-01", periods=40, freq="MS"),
            "feature_1": list(range(40)),
            "feature_2": list(range(40, 80)),
            "target": list(range(100, 140)),
        }
    )

    backtester = Backtester(
        dataframe=dataframe,
        date_column="DATA",
        horizon=3,
        n_folds=3,
        min_training_history=24,
        gap=0,
        strategy="expanding",
        feature_selector=FeatureSelector(features=["feature_1"]),
    )

    fold = backtester.generate_folds()[0]

    class DummyModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return pd.Series([y for y in X["feature_1"]], index=X.index)

    class DummyEvaluator:
        def evaluate(self, y_true, y_pred):
            return float(0.5)

    executed = backtester.run_fold(
        fold,
        target_column="target",
        model=DummyModel(),
        evaluator=DummyEvaluator(),
    )

    assert executed.selected_features == ["feature_1"]
    assert executed.metrics == {"metric": 0.5}


def test_backtester_run_models_returns_comparable_results_for_each_candidate():
    dataframe = pd.DataFrame(
        {
            "DATA": pd.date_range("2024-01-01", periods=40, freq="MS"),
            "feature_1": list(range(40)),
            "feature_2": list(range(40, 80)),
            "target": list(range(100, 140)),
        }
    )

    backtester = Backtester(
        dataframe=dataframe,
        date_column="DATA",
        horizon=3,
        n_folds=3,
        min_training_history=24,
        gap=0,
        strategy="expanding",
        feature_selector=FeatureSelector(features=["feature_1"]),
    )

    class DummyModelA:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return pd.Series([value for value in X["feature_1"]], index=X.index)

    class DummyModelB:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return pd.Series([value + 1 for value in X["feature_1"]], index=X.index)

    class DummyEvaluator:
        metric = "metric"

        def evaluate(self, y_true, y_pred):
            return float(0.5)

    results = backtester.run_models(
        target_column="target",
        models={
            "model_a": DummyModelA(),
            "model_b": DummyModelB(),
        },
        evaluator=DummyEvaluator(),
    )

    assert set(results.keys()) == {"model_a", "model_b"}
    assert len(results["model_a"].folds) == 3
    assert len(results["model_b"].folds) == 3
    assert results["model_a"].aggregated_metrics["metric"]["mean"] == 0.5
    assert results["model_b"].aggregated_metrics["metric"]["mean"] == 0.5


def test_backtest_model_selector_selects_model_from_comparable_results():
    model_a = "model_a"
    model_b = "model_b"

    result_a = BacktestResult(
        folds=[],
        metrics_by_fold={},
        aggregated_metrics={
            "metric": {
                "mean": 2.0,
                "median": 2.0,
                "std": 0.0,
            }
        },
    )

    result_b = BacktestResult(
        folds=[],
        metrics_by_fold={},
        aggregated_metrics={
            "metric": {
                "mean": 1.0,
                "median": 1.0,
                "std": 0.0,
            }
        },
    )

    selector = BacktestModelSelector(metric="metric", objective="minimize")
    winner = selector.select({model_a: result_a, model_b: result_b})

    assert winner == model_b


def test_final_model_trainer_trains_winner_only_on_allowed_data_and_keeps_test_out():
    train_validation = pd.DataFrame(
        {
            "DATA": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "feature_1": [0, 1, 2, 3, 4, 5],
            "feature_2": [10, 11, 12, 13, 14, 15],
            "target": [100, 101, 102, 103, 104, 105],
        }
    )

    final_test = pd.DataFrame(
        {
            "DATA": pd.date_range("2024-07-01", periods=2, freq="MS"),
            "feature_1": [6, 7],
            "feature_2": [16, 17],
            "target": [106, 107],
        }
    )

    class DummyModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return pd.Series([float(row["feature_1"]) for _, row in X.iterrows()], index=X.index)

    class DummyEvaluator:
        metric = "metric"

        def evaluate(self, y_true, y_pred):
            return float(0.5)

    trainer = FinalModelTrainer(
        target_column="target",
        feature_selector=FeatureSelector(features=["feature_1"]),
        evaluator=DummyEvaluator(),
    )

    result = trainer.run(
        training_frame=train_validation,
        test_frame=final_test,
        model=DummyModel(),
    )

    assert isinstance(result, FinalTrainingResult)
    assert result.selected_features == ["feature_1"]
    assert result.training_rows == len(train_validation)
    assert result.test_rows == len(final_test)
    assert result.metric == 0.5
