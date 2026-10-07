import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.evaluate import wape
from src.models import criar_modelo


def _features() -> pd.DataFrame:
    datas = pd.date_range("2026-01-01", periods=60)
    df = pd.MultiIndex.from_product(
        [datas, ["App", "Site"]], names=["data", "canal"]
    ).to_frame(index=False)
    df["receita"] = np.arange(100, 100 + len(df), dtype=float)
    df["pedidos"] = np.arange(1, len(df) + 1)
    df["ticket_medio"] = df["receita"] / df["pedidos"]
    df["taxa_desconto"] = 0.1
    return build_features(df).dropna().reset_index(drop=True)


@pytest.mark.parametrize(
    ("nome", "params"),
    [
        ("ridge", {"alpha": 1.0}),
        ("xgboost", {"n_estimators": 5, "max_depth": 2}),
        ("lightgbm", {"n_estimators": 5, "num_leaves": 4, "min_data_in_leaf": 5}),
    ],
)
def test_modelos_ml_preveem_na_escala_original(nome: str, params: dict) -> None:
    dados = _features()
    corte = int(len(dados) * 0.8)
    modelo = criar_modelo(nome, params=params, random_state=42).fit(dados.iloc[:corte])
    previsto = modelo.predict(dados.iloc[corte:])
    assert np.all(previsto >= 0)
    assert np.median(previsto) > 10


def _serie_semanal() -> tuple[pd.DataFrame, pd.DataFrame]:
    datas = pd.date_range("2025-10-01", periods=91)
    padrao = np.array([100.0, 120.0, 140.0, 160.0, 180.0, 130.0, 90.0])
    partes = []
    for canal, fator in [("App", 2.0), ("Site", 1.0)]:
        parte = pd.DataFrame(
            {"data": datas, "canal": canal, "y": np.resize(padrao, len(datas)) * fator}
        )
        partes.append(parte)
    dados = pd.concat(partes, ignore_index=True)
    for coluna in ["is_feriado", "is_vespera_feriado", "is_novembro", "is_evento"]:
        dados[coluna] = 0
    dados["dias_ate_evento_presente"] = 15
    dados["dias_ate_evento_promo"] = 15
    treino = dados.loc[dados["data"].le(datas[-8])].copy()
    teste = dados.loc[dados["data"].gt(datas[-8])].copy()
    return treino, teste


@pytest.mark.parametrize("nome", ["ets_log", "sarima_log", "sarimax_log"])
def test_modelos_estatisticos_recuperam_sazonalidade_sem_futuro(nome: str) -> None:
    treino, teste = _serie_semanal()
    modelo = criar_modelo(nome).fit(treino)
    previsto = modelo.predict(teste)
    alterado = teste.copy()
    alterado["y"] = alterado["y"] * 100

    assert wape(teste["y"], previsto) < 0.03
    np.testing.assert_allclose(previsto, modelo.predict(alterado))
