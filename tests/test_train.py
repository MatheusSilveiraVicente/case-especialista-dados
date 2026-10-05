import numpy as np
import pandas as pd

from src.models import NaiveSazonal
from src.train import (
    aplicar_regra_decisao,
    criar_fold,
    origens_avaliacao,
    origens_holdout,
    origens_pre_holdout,
    origens_tuning,
)


def _dados() -> pd.DataFrame:
    datas = pd.date_range("2025-11-01", "2026-06-30")
    df = pd.MultiIndex.from_product(
        [datas, ["App", "Site"]], names=["data", "canal"]
    ).to_frame(index=False)
    df["y"] = np.arange(len(df), dtype=float)
    df["lag_7"] = df.groupby("canal")["y"].shift(7)
    return df


def test_folds_respeitam_origem_e_holdout() -> None:
    df = _dados()
    for origem in origens_avaliacao():
        treino, teste = criar_fold(df, origem)
        assert treino["data"].max() <= origem
        assert teste["data"].min() > origem
        assert teste["data"].max() <= origem + pd.Timedelta(days=7)
        assert not treino["data"].isin(pd.date_range("2026-06-01", "2026-06-30")).any()
    assert min(origens_holdout()) == pd.Timestamp("2026-05-31")


def test_folds_tuning_avaliacao_disjuntos_e_cobrem_periodo() -> None:
    df = _dados()
    tuning = origens_tuning()
    avaliacao = origens_avaliacao()
    assert set(tuning).isdisjoint(avaliacao)
    assert sorted([*tuning, *avaliacao]) == origens_pre_holdout()

    datas = []
    for origem in [*tuning, *avaliacao]:
        _, teste = criar_fold(df, origem)
        datas.extend(teste["data"].drop_duplicates().tolist())
    esperado = list(pd.date_range("2025-12-31", "2026-05-31"))
    assert sorted(datas) == esperado


def test_naive_sazonal_reproduz_lag_7() -> None:
    df = _dados().dropna().reset_index(drop=True)
    modelo = NaiveSazonal().fit(df.iloc[:100])
    previsto = modelo.predict(df.iloc[100:120])
    np.testing.assert_array_equal(previsto, df.iloc[100:120]["lag_7"])


def _previsoes_decisao(valor_ridge: float, valor_bases: float) -> pd.DataFrame:
    partes = []
    for modelo in ["naive_sazonal", "media_movel_7", "ridge", "xgboost", "lightgbm"]:
        previsto = valor_ridge if modelo == "ridge" else valor_bases
        seed = -1 if modelo in {"xgboost", "lightgbm"} else 42
        partes.append(
            pd.DataFrame(
                {
                    "data": pd.date_range("2026-01-01", periods=7),
                    "origem": pd.Timestamp("2025-12-31"),
                    "modelo": modelo,
                    "seed": seed,
                    "y": 100.0,
                    "y_hat": previsto,
                }
            )
        )
    return pd.concat(partes, ignore_index=True)


def _resumo_decisao() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "modelo": ["ridge", "naive_sazonal", "media_movel_7", "xgboost", "lightgbm"],
            "conjunto": "teste",
            "granularidade": "total",
            "segmento": "todos",
            "wape": [0.0, 0.1, 0.1, 0.2, 0.2],
        }
    )


def test_decisao_empate_escolhe_mais_simples() -> None:
    decisao = aplicar_regra_decisao(
        _resumo_decisao(), _previsoes_decisao(100.0, 100.0), "teste"
    )
    assert decisao == {
        "candidato": "ridge",
        "empates": ["naive_sazonal", "media_movel_7"],
        "vencedor": "naive_sazonal",
    }


def test_decisao_vitoria_clara_mantem_candidato() -> None:
    decisao = aplicar_regra_decisao(
        _resumo_decisao(), _previsoes_decisao(100.0, 120.0), "teste"
    )
    assert decisao["vencedor"] == "ridge"
