"""testes da configuração de features do projeto."""

from src.config.features import (
    FeatureSelectionSettings,
    FeatureSettings,
)


def test_feature_selection_defaults() -> None:
    """valida os valores padrões de seleção de features."""

    settings = FeatureSelectionSettings()

    assert settings.enabled is True
    assert settings.features == []


def test_feature_selection_accepts_features() -> None:
    """aceita a lista de features configuradas."""

    settings = FeatureSelectionSettings(
        features=[
            "lag_1",
            "rolling_mean_3",
            "month",
        ]
    )

    assert settings.enabled is True

    assert settings.features == [
        "lag_1",
        "rolling_mean_3",
        "month",
    ]


def test_feature_selection_can_be_disabled() -> None:
    """permite desabilitar a seleção de features."""

    settings = FeatureSelectionSettings(
        enabled=False,
    )

    assert settings.enabled is False


def test_feature_settings_contains_selection() -> None:
    """integra a configuração de features com a seleção ativa."""

    settings = FeatureSettings(
        temporal={
            "enabled": True,
        },
        lag={
            "enabled": True,
            "periods": [1, 2, 3],
        },
        rolling={
            "enabled": True,
            "windows": [3, 6],
        },
        trend={
            "enabled": True,
        },
        business={
            "enabled": True,
        },
        holidays={
            "enabled": True,
        },
        selection={
            "enabled": True,
            "features": [
                "lag_1",
                "rolling_mean_3",
            ],
        },
    )

    assert settings.selection.enabled is True

    assert settings.selection.features == [
        "lag_1",
        "rolling_mean_3",
    ]