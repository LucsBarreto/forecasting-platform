"""
módulo de treinamento de modelos.
"""

from .backtesting import BacktestFold, Backtester, BacktestResult
from .cross_validation import TemporalSplit, TemporalSplitter
from .model_selection import ModelSelector
from .trainer import ModelTrainer
from .training_manager import TrainingManager, TrainingResult

__all__ = [
    "BacktestFold",
    "Backtester",
    "BacktestResult",
    "ModelSelector",
    "ModelTrainer",
    "TemporalSplit",
    "TemporalSplitter",
    "TrainingManager",
    "TrainingResult",
]