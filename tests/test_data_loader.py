from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    CATEGORIA_NAO_MAPEADA,
    clean,
    complete_grid,
    load_daily,
    load_raw,
    quality_report,
    to_daily,
)


def _salvar_csv_original(df: pd.DataFrame, path: Path) -> None:
    reverso = {
        "dt_hr": "dt_hr_venda",
        "canal": "DES_CANAL_VENDA_FINAL_AGRUP",
        "categoria": "DES_CATEGORIA_MATERIAL",
        "receita": "receita_aprovada",
        "pedidos": "nr_pedidos",
        "itens": "qt_material",
        "desconto": "vlr_venda_desconto",
    }
    original = df.rename(columns=reverso)
    original["receita_aprovada"] += 0.00000000000001
    original.to_csv(path, index=False)


def test_load_raw_renomeia_e_tipa(vendas_raw: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "vendas.csv"
    _salvar_csv_original(vendas_raw, path)

    resultado = load_raw(path)

    assert list(resultado.columns) == [
        "dt_hr",
        "canal",
        "categoria",
        "receita",
        "pedidos",
        "itens",
        "desconto",
    ]
    assert pd.api.types.is_datetime64_dtype(resultado["dt_hr"])
    assert resultado["pedidos"].dtype == "int64"
    assert resultado["itens"].dtype == "int64"
    assert resultado["receita"].dtype == "float64"
    assert resultado["desconto"].dtype == "float64"
    assert resultado.loc[0, "receita"] == 10.0


def test_clean_marca_e_agrupa_sem_remover_estorno(vendas_raw: pd.DataFrame) -> None:
    resultado = clean(vendas_raw)
    nao_mapeada = resultado.loc[
        (resultado["dt_hr"] == "2025-11-01")
        & (resultado["canal"] == "Site")
        & (resultado["categoria"] == CATEGORIA_NAO_MAPEADA)
    ].squeeze()

    assert len(resultado) == len(vendas_raw) - 1
    assert nao_mapeada["receita"] == 30.0
    assert nao_mapeada["pedidos"] == 3
    assert bool(nao_mapeada["categoria_invalida"])
    assert (resultado["receita"] == -5.0).sum() == 1
    assert bool(resultado.loc[resultado["receita"] == -5.0, "receita_negativa"].item())
    assert resultado["receita"].sum() == vendas_raw["receita"].sum()


def test_complete_grid_imputa_zero_e_preserva_receita(
    vendas_raw: pd.DataFrame,
) -> None:
    limpo = clean(vendas_raw)
    resultado = complete_grid(limpo)
    linha = resultado.loc[
        (resultado["dt_hr"] == "2025-11-02 12:00:00")
        & (resultado["canal"] == "Site")
        & (resultado["categoria"] == "Perfumaria")
    ].squeeze()

    assert len(resultado) == 72 * 2 * 3
    assert linha["receita"] == 0.0
    assert linha["pedidos"] == 0
    assert linha["itens"] == 0
    assert linha["desconto"] == 0.0
    assert bool(linha["linha_imputada"])
    assert not bool(linha["categoria_invalida"])
    assert resultado["receita"].sum() == limpo["receita"].sum()


def test_to_daily_calcula_metricas_e_divisao_por_zero(
    vendas_raw: pd.DataFrame,
) -> None:
    grade = complete_grid(clean(vendas_raw))
    resultado = to_daily(grade, by=["canal"])
    app_dia_1 = resultado.loc[
        (resultado["canal"] == "App") & (resultado["data"] == "2025-11-01")
    ].squeeze()
    site_dia_1 = resultado.loc[
        (resultado["canal"] == "Site") & (resultado["data"] == "2025-11-01")
    ].squeeze()
    site_dia_2 = resultado.loc[
        (resultado["canal"] == "Site") & (resultado["data"] == "2025-11-02")
    ].squeeze()

    assert app_dia_1["estorno"] == -5.0
    assert site_dia_1["receita"] == 80.0
    assert site_dia_1["taxa_desconto"] == pytest.approx(13.0 / 93.0)
    assert np.isnan(site_dia_2["ticket_medio"])
    assert np.isnan(site_dia_2["preco_medio"])
    assert pd.api.types.is_datetime64_dtype(resultado["data"])
    esperado = resultado.sort_values(["canal", "data"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(resultado, esperado)


def test_receita_invariante_em_todas_as_etapas(
    vendas_raw: pd.DataFrame,
) -> None:
    limpo = clean(vendas_raw)
    grade = complete_grid(limpo)
    diario = to_daily(grade)

    esperado = vendas_raw["receita"].sum()
    assert limpo["receita"].sum() == esperado
    assert grade["receita"].sum() == esperado
    assert diario["receita"].sum() == esperado


def test_quality_report(vendas_raw: pd.DataFrame) -> None:
    resultado = quality_report(vendas_raw)

    assert resultado["n_linhas"] == 7
    assert resultado["n_duplicatas_chave"] == 0
    assert resultado["n_nulos"] == 0
    assert resultado["n_categoria_invalida"] == 2
    assert resultado["pct_receita_categoria_invalida"] == pytest.approx(
        30.0 / 185.0 * 100
    )
    assert resultado["n_receita_negativa"] == 1
    assert resultado["soma_receita_negativa"] == -5.0
    assert resultado["n_receita_zero"] == 1
    assert resultado["n_horas_faltantes"] == 66
    assert resultado["pct_grade_preenchida"] == pytest.approx(5 / (72 * 2 * 2) * 100)


def test_load_daily_executa_pipeline(vendas_raw: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "vendas.csv"
    _salvar_csv_original(vendas_raw, path)

    resultado = load_daily(path, by=["canal"])

    assert len(resultado) == 3 * 2
    assert resultado["receita"].sum() == vendas_raw["receita"].sum()


def test_smoke_dado_real() -> None:
    path = Path("data/raw/vendas.csv")
    if not path.exists():
        pytest.skip("data/raw/vendas.csv ausente")

    relatorio = quality_report(load_raw(path))

    assert relatorio["n_linhas"] == 89566
    assert relatorio["n_categoria_invalida"] == 5341
    assert relatorio["n_receita_negativa"] == 3513
    assert relatorio["n_horas_faltantes"] == 0
