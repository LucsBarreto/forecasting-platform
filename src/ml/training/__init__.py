"""
módulo de treinamento de modelos.
"""

from .backtesting import BacktestFold, Backtester, BacktestResult
from .cross_validation import TemporalSplit, TemporalSplitter
from .final_training import FinalModelTrainer, FinalTrainingResult
from .model_selection import BacktestModelSelector, ModelSelector
from .trainer import ModelTrainer
from .training_manager import TrainingManager, TrainingResult

__all__ = [
    "BacktestFold",
    "Backtester",
    "BacktestResult",
    "BacktestModelSelector",
    "FinalModelTrainer",
    "FinalTrainingResult",
    "ModelSelector",
    "ModelTrainer",
    "TemporalSplit",
    "TemporalSplitter",
    "TrainingManager",
    "TrainingResult",
]