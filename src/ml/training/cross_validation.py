"""
divisão temporal dos dados para treinamento e validação de modelos.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.exceptions.validation import DataValidationError


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """
    container para os conjuntos temporais de treino, validação e teste.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(slots=True)
class TemporalSplitter:
    """
    divide um dataframe em conjuntos cronológicos de treino, validação e teste.

    o divisor preserva a ordem temporal e nunca embaralha as observações.
    """

    train_size: float
    validation_size: float
    test_size: float
    date_column: str = "DATA"

    def split(
        self,
        dataframe: pd.DataFrame,
    ) -> TemporalSplit:
        """
        divide o dataframe cronologicamente.

        parameters
        ----------
        dataframe
            dataframe de entrada contendo a coluna de data.

        returns
        -------
        temporalsplit
            conjuntos cronológicos de treino, validação e teste.
        """

        self._validate_configuration()
        self._validate_dataframe(dataframe)

        data = dataframe.copy()

        data[self.date_column] = pd.to_datetime(
            data[self.date_column],
            errors="coerce",
        )

        if data[self.date_column].isna().any():
            raise DataValidationError(
                f"Date column '{self.date_column}' "
                "contains invalid dates."
            )

        data = data.sort_values(
            self.date_column,
            kind="stable",
        ).reset_index(drop=True)

        total_rows = len(data)

        train_end = int(
            total_rows * self.train_size
        )

        validation_end = train_end + int(
            total_rows * self.validation_size
        )

        if train_end <= 0:
            raise DataValidationError(
                "Training split cannot be empty."
            )

        if validation_end <= train_end:
            raise DataValidationError(
                "Validation split cannot be empty."
            )

        if validation_end >= total_rows:
            raise DataValidationError(
                "Test split cannot be empty."
            )

        return TemporalSplit(
            train=data.iloc[
                :train_end
            ].copy(),
            validation=data.iloc[
                train_end:validation_end
            ].copy(),
            test=data.iloc[
                validation_end:
            ].copy(),
        )

    def _validate_configuration(self) -> None:
        """valida as proporções dos conjuntos."""

        values = (
            self.train_size,
            self.validation_size,
            self.test_size,
        )

        if any(
            not isinstance(value, (int, float))
            for value in values
        ):
            raise TypeError(
                "Split sizes must be numeric."
            )

        if any(
            value <= 0 or value >= 1
            for value in values
        ):
            raise ValueError(
                "Split sizes must be greater than 0 "
                "and less than 1."
            )

        total = sum(values)

        if not abs(total - 1.0) < 1e-9:
            raise ValueError(
                "Train, validation and test sizes "
                "must sum to 1."
            )

        if not isinstance(
            self.date_column,
            str,
        ) or not self.date_column.strip():
            raise ValueError(
                "date_column must be a non-empty string."
            )

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """valida o dataframe de entrada."""

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise DataValidationError(
                "Cannot split an empty dataframe."
            )

        if self.date_column not in dataframe.columns:
            raise DataValidationError(
                f"Date column '{self.date_column}' "
                "was not found."
            )