import numpy as np
import pandas as pd
import pytest

from src.features import build_features
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
