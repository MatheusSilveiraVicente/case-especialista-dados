from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_daily
from src.features import COLUNAS_HISTORICAS, build_features, feature_columns
from src.train import origens_avaliacao, origens_holdout, origens_tuning


def _daily_sintetico() -> pd.DataFrame:
    datas = pd.date_range("2025-11-01", "2026-06-30", freq="D")
    indice = pd.MultiIndex.from_product(
        [datas, ["Site", "App"]], names=["data", "canal"]
    ).to_frame(index=False)
    indice["receita"] = np.arange(1, len(indice) + 1, dtype=float)
    indice["pedidos"] = np.arange(1, len(indice) + 1, dtype=int)
    indice["ticket_medio"] = indice["receita"] / indice["pedidos"]
    indice["taxa_desconto"] = 0.1 + indice["data"].dt.day / 1000
    return indice


def test_features_historicas_nao_usam_futuro() -> None:
    daily = _daily_sintetico()
    data_t = pd.Timestamp("2026-02-15")
    original = build_features(daily)

    perturbado = daily.copy()
    perturbado.loc[perturbado["data"] > data_t - pd.Timedelta(days=7), "receita"] += 1_000_000
    alterado = build_features(perturbado)

    filtro = original["data"].eq(data_t)
    pd.testing.assert_frame_equal(
        original.loc[filtro, COLUNAS_HISTORICAS].reset_index(drop=True),
        alterado.loc[filtro, COLUNAS_HISTORICAS].reset_index(drop=True),
    )


def _assert_sem_vazamento_todas_origens(daily: pd.DataFrame) -> None:
    original = build_features(daily)
    colunas_features = [
        *feature_columns()["categoricas"],
        *feature_columns()["numericas"],
    ]
    colunas_dado = daily.columns.difference(["data", "canal"])
    origens = [*origens_tuning(), *origens_avaliacao(), *origens_holdout()]

    for origem in origens:
        perturbado = daily.copy()
        futuro = perturbado["data"].gt(origem)
        perturbado.loc[futuro, colunas_dado] = (
            perturbado.loc[futuro, colunas_dado] * -123 + 987_654_321
        )
        alterado = build_features(perturbado)
        horizonte = original["data"].gt(origem) & original["data"].le(
            origem + pd.Timedelta(days=7)
        )
        pd.testing.assert_frame_equal(
            original.loc[horizonte, colunas_features].reset_index(drop=True),
            alterado.loc[horizonte, colunas_features].reset_index(drop=True),
            obj=f"features na origem {origem:%Y-%m-%d}",
        )


def test_sem_vazamento_todas_origens_dado_sintetico() -> None:
    _assert_sem_vazamento_todas_origens(_daily_sintetico())


def test_sem_vazamento_todas_origens_dado_real() -> None:
    path = Path("data/raw/vendas.csv")
    if not path.exists():
        pytest.skip("data/raw/vendas.csv ausente")

    _assert_sem_vazamento_todas_origens(load_daily(path, by=["canal"]))


def test_lag_7_corresponde_ao_alvo_do_mesmo_canal() -> None:
    daily = _daily_sintetico()
    resultado = build_features(daily)
    linha = resultado.loc[
        (resultado["data"] == "2025-11-15") & (resultado["canal"] == "App")
    ].squeeze()
    esperado = daily.loc[
        (daily["data"] == "2025-11-08") & (daily["canal"] == "App"), "receita"
    ].item()

    assert linha["lag_7"] == esperado


def test_calendario_eventos_e_feriados() -> None:
    resultado = build_features(_daily_sintetico())

    def valor(data: str, coluna: str):
        return resultado.loc[
            (resultado["data"] == data) & (resultado["canal"] == "Site"), coluna
        ].item()

    assert valor("2025-11-28", "is_evento") == 1
    assert valor("2026-06-05", "dias_ate_evento_presente") == 7
    assert valor("2026-02-16", "is_feriado") == 1
    assert valor("2026-06-04", "is_feriado") == 1
    assert valor("2025-12-25", "is_feriado") == 1
    assert valor("2025-12-25", "is_evento") == 0
    assert valor("2025-12-24", "is_vespera_feriado") == 1


@pytest.mark.parametrize(
    ("data", "coluna", "esperado"),
    [
        ("2026-06-05", "is_payday_5", 1),
        ("2026-06-22", "is_payday_20", 1),
        ("2026-06-22", "is_payday_5", 0),
        ("2026-06-08", "is_payday_5", 0),
        ("2026-06-08", "is_payday_20", 0),
    ],
)
def test_is_payday(data: str, coluna: str, esperado: int) -> None:
    resultado = build_features(_daily_sintetico())
    valor = resultado.loc[
        (resultado["data"] == data) & (resultado["canal"] == "Site"), coluna
    ].item()

    assert valor == esperado


def test_mantem_todas_as_linhas_e_ordena() -> None:
    daily = _daily_sintetico().sample(frac=1, random_state=42).reset_index(drop=True)
    resultado = build_features(daily)

    assert len(resultado) == len(daily)
    esperado = resultado.sort_values(["canal", "data"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(resultado, esperado)
    assert feature_columns()["categoricas"] == ["canal", "dia_semana"]
    removidas = {
        "mes",
        "semana_ano",
        "pedidos_lag_7",
        "media_14_lag7",
        "janela_presente",
        "janela_promo",
    }
    assert removidas.isdisjoint(resultado.columns)


def test_smoke_features_dado_real() -> None:
    path = Path("data/raw/vendas.csv")
    if not path.exists():
        pytest.skip("data/raw/vendas.csv ausente")

    resultado = build_features(load_daily(path, by=["canal"]))

    assert len(resultado) == 484
    for _, grupo in resultado.groupby("canal"):
        assert grupo["lag_14"].iloc[:14].isna().all()
        assert grupo["lag_14"].iloc[14:].notna().all()
