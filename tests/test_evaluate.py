import numpy as np
import pandas as pd

from src.evaluate import (
    bias,
    bootstrap_ci,
    bootstrap_values,
    mae,
    paired_bootstrap_diff,
    smape,
    wape,
)


def test_metricas_calculadas_a_mao() -> None:
    y = np.array([100.0, 200.0])
    y_hat = np.array([110.0, 180.0])
    assert np.isclose(wape(y, y_hat), 30 / 300)
    assert np.isclose(bias(y, y_hat), -10 / 300)
    assert np.isclose(smape(y, y_hat), (20 / 210 + 40 / 380) / 2)
    assert np.isclose(mae(y, y_hat), 15.0)


def test_bootstrap_pareado_modelos_identicos() -> None:
    base = pd.DataFrame(
        {
            "data": pd.date_range("2026-01-01", periods=5).repeat(2),
            "canal": ["App", "Site"] * 5,
            "y": np.arange(10, 20, dtype=float),
            "y_hat": np.arange(10, 20, dtype=float) + 1,
        }
    )
    previsoes = pd.concat(
        [base.assign(modelo="a", seed=42), base.assign(modelo="b", seed=42)],
        ignore_index=True,
    )
    resultado = paired_bootstrap_diff(previsoes, "a", "b", n=100)
    assert resultado["diferenca"] == 0
    assert resultado["ic_low"] == 0
    assert resultado["ic_high"] == 0
    baixo, alto = bootstrap_ci(base, n=100)
    assert baixo <= wape(base.y, base.y_hat) <= alto


def test_bootstrap_uma_semana_repete_metrica_observada() -> None:
    base = pd.DataFrame(
        {
            "origem": pd.Timestamp("2026-05-31"),
            "data": pd.date_range("2026-06-01", periods=7),
            "y": np.arange(10, 17, dtype=float),
            "y_hat": np.arange(11, 18, dtype=float),
        }
    )
    valores = bootstrap_values(base, n=100)
    np.testing.assert_allclose(valores, wape(base["y"], base["y_hat"]))
