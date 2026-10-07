import numpy as np
import pandas as pd
import pytest

from src.combinacao import combinar_previsoes
from src.evaluate import wape


def _previsoes(sarima: list[float], lightgbm: list[float]) -> pd.DataFrame:
    datas = pd.date_range("2026-06-01", periods=7)
    real = np.arange(10.0, 80.0, 10.0)
    base = {
        "data": datas,
        "canal": "App",
        "origem": pd.Timestamp("2026-05-31"),
        "y": real,
    }
    return pd.concat(
        [
            pd.DataFrame(base).assign(modelo="sarima_log", seed=42, y_hat=sarima),
            pd.DataFrame(base).assign(modelo="lightgbm", seed=-1, y_hat=lightgbm),
        ],
        ignore_index=True,
    )


def test_combinacao_acerta_total_e_perfil() -> None:
    previsoes = _previsoes([40.0] * 7, list(range(1, 8)))
    combinado = combinar_previsoes(previsoes)
    assert wape(combinado["y"], combinado["y_hat"]) == pytest.approx(0.0)


def test_combinacao_preserva_total_sarima() -> None:
    sarima = list(np.arange(11.0, 18.0))
    previsoes = _previsoes(sarima, [3.0, 5.0, 2.0, 8.0, 1.0, 4.0, 7.0])
    combinado = combinar_previsoes(previsoes)
    assert combinado["y_hat"].sum() == pytest.approx(sum(sarima))


@pytest.mark.parametrize("perfil", [[0.0] * 7, [-1.0] * 7])
def test_perfil_nao_positivo_usa_distribuicao_uniforme(perfil: list[float]) -> None:
    previsoes = _previsoes([20.0] * 7, perfil)
    combinado = combinar_previsoes(previsoes)
    assert np.isfinite(combinado["y_hat"]).all()
    np.testing.assert_allclose(combinado["y_hat"], 20.0)
    assert combinado["y_hat"].sum() == pytest.approx(140.0)
