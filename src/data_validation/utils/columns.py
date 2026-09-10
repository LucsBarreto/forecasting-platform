from src.core.exceptions import MissingColumnError

import pandas as pd


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    valida se todas as colunas necessárias estão no dataframe

    args:
        dataframe:
            input dataframe.
        required_columns:
            colunas que devem existir

    raises:
        missingcolumnerror:
            se alguma coluna não existe
    """

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise MissingColumnError(missing[0])