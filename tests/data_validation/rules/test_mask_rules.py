"""
Tests for mask-based validation rules.
"""

import pandas as pd
import pytest

from src.data_validation.rules.future_date_rule import FutureDateRule
from src.data_validation.rules.negative_sales_rule import NegativeSalesRule
from src.data_validation.rules.null_customer_rule import NullCustomerRule
from src.data_validation.rules.null_product_rule import NullProductRule
from src.data_validation.rules.positive_return_rule import PositiveReturnRule
from src.data_validation.rules.zero_value_rule import ZeroValueRule
from src.data_validation.rules.zero_volume_rule import ZeroVolumeRule


@pytest.mark.parametrize(
    ("rule_class", "data"),
    [
        (
            ZeroVolumeRule,
            {"VOLUME": [0, 10, 20]},
        ),
        (
            ZeroValueRule,
            {"VALOR": [0, 100, 200]},
        ),
        (
            NegativeSalesRule,
            {"VALOR": [-10, 100, 200]},
        ),
        (
            PositiveReturnRule,
            {"DEVOLUCAO": [10, 0, -5]},
        ),
        (
            NullCustomerRule,
            {"COD CLIENTE": [None, 2, 3]},
        ),
        (
            NullProductRule,
            {"COD ITEM": [None, 20, 30]},
        ),
    ],
)
def test_mask_rules_detect_invalid_rows(
    rule_class,
    data,
) -> None:
    """Test that mask rules detect their expected rows."""

    dataframe = pd.DataFrame(data)

    rule = rule_class()

    mask = rule.build_mask(dataframe)

    assert int(mask.sum()) == 1

