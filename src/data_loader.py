"""Carga, limpeza e agregacao dos dados de vendas."""

from pathlib import Path

import numpy as np
import pandas as pd


CATEGORIA_NAO_MAPEADA = "Nao mapeada"

COLUNAS_ORIGINAIS = {
    "dt_hr_venda": "dt_hr",
    "DES_CANAL_VENDA_FINAL_AGRUP": "canal",
    "DES_CATEGORIA_MATERIAL": "categoria",
    "receita_aprovada": "receita",
    "nr_pedidos": "pedidos",
    "qt_material": "itens",
    "vlr_venda_desconto": "desconto",
}
METRICAS = ["receita", "pedidos", "itens", "desconto", "estorno"]


def load_raw(path: str | Path = "data/raw/vendas.csv") -> pd.DataFrame:
    """Le o CSV e aplica nomes e tipos do contrato."""
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns=COLUNAS_ORIGINAIS)[list(COLUNAS_ORIGINAIS.values())]

    df["dt_hr"] = pd.to_datetime(df["dt_hr"], errors="raise")
    for coluna in ["pedidos", "itens"]:
        df[coluna] = pd.to_numeric(df[coluna], errors="raise").astype("int64")
    for coluna in ["receita", "desconto"]:
        df[coluna] = (
            pd.to_numeric(df[coluna], errors="raise").round(2).astype("float64")
        )
    return df


def _sem_letras(valor: object) -> bool:
    return not any(caractere.isalpha() for caractere in str(valor))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Trata categorias invalidas e adiciona campos derivados."""
    resultado = df.copy()
    resultado["categoria_invalida"] = resultado["categoria"].map(_sem_letras)
    resultado.loc[resultado["categoria_invalida"], "categoria"] = (
        CATEGORIA_NAO_MAPEADA
    )
    resultado["receita_negativa"] = resultado["receita"].lt(0)
    # Estorno calculado por linha original, antes de agrupar "Nao mapeada".
    resultado["estorno"] = resultado["receita"].where(resultado["receita_negativa"], 0.0)

    resultado = (
        resultado.groupby(
            ["dt_hr", "canal", "categoria"], as_index=False, sort=True, dropna=False
        )
        .agg(
            receita=("receita", "sum"),
            pedidos=("pedidos", "sum"),
            itens=("itens", "sum"),
            desconto=("desconto", "sum"),
            estorno=("estorno", "sum"),
            categoria_invalida=("categoria_invalida", "max"),
            receita_negativa=("receita_negativa", "max"),
        )
        .reset_index(drop=True)
    )
    resultado["data"] = resultado["dt_hr"].dt.normalize()
    resultado["hora"] = resultado["dt_hr"].dt.hour.astype("int64")
    resultado["receita_bruta"] = resultado["receita"] + resultado["desconto"]
    return resultado


def complete_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Completa a grade hora, canal e categoria com vendas iguais a zero."""
    horas = pd.date_range(df["dt_hr"].min(), df["dt_hr"].max(), freq="h")
    canais = sorted(df["canal"].unique())
    categorias = sorted(df["categoria"].unique())
    indice = pd.MultiIndex.from_product(
        [horas, canais, categorias], names=["dt_hr", "canal", "categoria"]
    )

    resultado = df.drop(columns=["data", "hora", "receita_bruta"]).copy()
    resultado["_presente"] = True
    resultado = resultado.set_index(["dt_hr", "canal", "categoria"]).reindex(indice)
    resultado["linha_imputada"] = resultado["_presente"].isna()
    resultado = resultado.drop(columns="_presente").reset_index()

    for coluna in METRICAS:
        resultado[coluna] = resultado[coluna].fillna(0)
    resultado[["pedidos", "itens"]] = resultado[["pedidos", "itens"]].astype(
        "int64"
    )
    resultado[["receita", "desconto", "estorno"]] = resultado[
        ["receita", "desconto", "estorno"]
    ].astype("float64")
    for coluna in ["categoria_invalida", "receita_negativa"]:
        resultado[coluna] = resultado[coluna].eq(True)

    resultado["data"] = resultado["dt_hr"].dt.normalize()
    resultado["hora"] = resultado["dt_hr"].dt.hour.astype("int64")
    resultado["receita_bruta"] = resultado["receita"] + resultado["desconto"]
    return resultado


def _razao_segura(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    return numerador.div(denominador).where(denominador.ne(0), np.nan)


def to_daily(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Agrega metricas diarias nas dimensoes solicitadas."""
    dimensoes = list(by or [])
    trabalho = df.copy()
    trabalho["data"] = pd.to_datetime(trabalho["data"]).dt.normalize()
    if "estorno" not in trabalho:
        trabalho["estorno"] = trabalho["receita"].where(
            trabalho["receita"].lt(0), 0.0
        )

    colunas_grupo = ["data", *dimensoes]
    resultado = (
        trabalho.groupby(colunas_grupo, as_index=False, sort=True, dropna=False)
        .agg(
            receita=("receita", "sum"),
            pedidos=("pedidos", "sum"),
            itens=("itens", "sum"),
            desconto=("desconto", "sum"),
            receita_bruta=("receita_bruta", "sum"),
            estorno=("estorno", "sum"),
        )
        .sort_values([*dimensoes, "data"])
        .reset_index(drop=True)
    )
    resultado["ticket_medio"] = _razao_segura(
        resultado["receita"], resultado["pedidos"]
    )
    resultado["preco_medio"] = _razao_segura(
        resultado["receita"], resultado["itens"]
    )
    resultado["taxa_desconto"] = _razao_segura(
        resultado["desconto"], resultado["receita_bruta"]
    )
    return resultado


def quality_report(df_raw: pd.DataFrame) -> dict:
    """Resume problemas de qualidade observados no dado bruto tipado."""
    categoria_invalida = df_raw["categoria"].map(_sem_letras)
    receita_total = df_raw["receita"].sum()
    receita_invalida = df_raw.loc[categoria_invalida, "receita"].sum()

    dt_min = df_raw["dt_hr"].min()
    dt_max = df_raw["dt_hr"].max()
    total_horas = len(pd.date_range(dt_min, dt_max, freq="h"))
    n_horas_faltantes = total_horas - df_raw["dt_hr"].nunique()

    validos = df_raw.loc[~categoria_invalida]
    n_categorias = validos["categoria"].nunique()
    n_canais = df_raw["canal"].nunique()
    tamanho_grade = total_horas * n_canais * n_categorias
    chaves_preenchidas = validos[
        ["dt_hr", "canal", "categoria"]
    ].drop_duplicates().shape[0]

    return {
        "n_linhas": int(len(df_raw)),
        "n_duplicatas_chave": int(
            df_raw.duplicated(["dt_hr", "canal", "categoria"]).sum()
        ),
        "n_nulos": int(df_raw.isna().sum().sum()),
        "n_categoria_invalida": int(categoria_invalida.sum()),
        "pct_receita_categoria_invalida": (
            float(receita_invalida / receita_total * 100)
            if receita_total != 0
            else np.nan
        ),
        "n_receita_negativa": int(df_raw["receita"].lt(0).sum()),
        "soma_receita_negativa": float(
            df_raw.loc[df_raw["receita"].lt(0), "receita"].sum()
        ),
        "n_receita_zero": int(df_raw["receita"].eq(0).sum()),
        "dt_min": dt_min,
        "dt_max": dt_max,
        "n_horas_faltantes": int(n_horas_faltantes),
        "pct_grade_preenchida": (
            float(chaves_preenchidas / tamanho_grade * 100)
            if tamanho_grade
            else np.nan
        ),
    }


def load_daily(
    path: str | Path = "data/raw/vendas.csv", by: list[str] | None = None
) -> pd.DataFrame:
    """Executa carga, limpeza, completude da grade e agregacao diaria."""
    return to_daily(complete_grid(clean(load_raw(path))), by=by)
