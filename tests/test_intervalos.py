import numpy as np
import pandas as pd
import pytest

from src.evaluate import cobertura, fva, pinball, totais_semanais, wape


def test_fva_regua_e_modelo_perfeito() -> None:
    assert fva(0.3, 0.3) == 0.0
    assert fva(0.0, 0.3) == 1.0


def test_wape_semanal_cancela_erros_da_semana() -> None:
    diario = pd.DataFrame({
        "origem": ["s1"] * 2 + ["s2"] * 2,
        "y": [100.0, 100.0, 50.0, 50.0],
        "y_hat": [120.0, 80.0, 60.0, 40.0],
    })
    semanal = totais_semanais(diario)

    assert wape(diario["y"], diario["y_hat"]) > 0
    assert wape(semanal["y"], semanal["y_hat"]) == 0.0


def test_cobertura_oito_de_dez() -> None:
    y = np.arange(10, dtype=float)
    inferior = np.full(10, 1.0)
    superior = np.full(10, 8.0)

    assert cobertura(y, inferior, superior) == pytest.approx(0.8)


def test_pinball_calculado_a_mao() -> None:
    # erro +2 com alpha 0.9 -> 1.8; erro -2 com alpha 0.9 -> 0.2
    assert pinball([10.0], [8.0], 0.9) == pytest.approx(1.8)
    assert pinball([10.0], [12.0], 0.9) == pytest.approx(0.2)


def test_intervalo_ordenado_em_dados_sinteticos() -> None:
    from src.features import build_features
    from src.intervalos import prever_quantis
    from tests.test_features import _daily_sintetico

    features = build_features(_daily_sintetico())
    origem = features["data"].sort_values().iloc[len(features) // 4]
    params = {"num_leaves": 4, "min_data_in_leaf": 5, "n_estimators": 20, "learning_rate": 0.1}
    quantis = prever_quantis(features, [origem], params)

    assert not quantis.empty
    assert (quantis["p10"] <= quantis["p90"]).all()


def test_conformal_alarga_ate_a_cobertura_alvo() -> None:
    from src.intervalos import ajuste_conformal, aplicar_conformal

    rng = np.random.default_rng(0)
    n = 50
    y = rng.uniform(100, 200, n)
    quantis = pd.DataFrame({
        "data": pd.date_range("2026-01-01", periods=n),
        "origem": "o1",
        "canal": "App",
        "y": y,
        "p10": y * 0.98,
        "p90": y * 1.02 + rng.normal(0, 30, n) ** 2 * np.sign(rng.normal(size=n)),
    })
    quantis["p90"] = quantis[["p10", "p90"]].max(axis=1)
    folga = ajuste_conformal(quantis, cobertura_alvo=0.8)
    calibrado = aplicar_conformal(quantis, folga)

    assert folga >= 0
    assert cobertura(calibrado["y"], calibrado["p10"], calibrado["p90"]) >= 0.8
