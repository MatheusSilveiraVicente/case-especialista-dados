"""Combina total semanal do SARIMA com perfil diario do LightGBM."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from src.evaluate import agregar_total, bias, paired_bootstrap_diff, wape


PASTA_DADOS = Path("data/processed")
PASTA_METRICAS = Path("reports/metrics")
PASTA_FIGURAS = Path("reports/figures")
SEED_LIGHTGBM = -1
MODELOS = ["sarima_log", "lightgbm", "combinacao", "media_movel_7"]


def combinar_previsoes(
    previsoes: pd.DataFrame, seed_lightgbm: int = SEED_LIGHTGBM
) -> pd.DataFrame:
    """Aplica o perfil LightGBM ao total SARIMA por origem e canal."""
    chaves = ["data", "canal", "origem"]
    sarima = previsoes.loc[
        previsoes["modelo"].eq("sarima_log") & previsoes["seed"].eq(42),
        chaves + ["y", "y_hat"],
    ].rename(columns={"y_hat": "y_hat_sarima"})
    lightgbm = previsoes.loc[
        previsoes["modelo"].eq("lightgbm")
        & previsoes["seed"].eq(seed_lightgbm),
        chaves + ["y", "y_hat"],
    ].rename(columns={"y": "y_lightgbm", "y_hat": "y_hat_lightgbm"})
    combinado = sarima.merge(
        lightgbm, on=chaves, how="outer", validate="one_to_one", indicator=True
    )
    if combinado.empty or combinado["_merge"].ne("both").any():
        raise ValueError("SARIMA e LightGBM precisam ter previsoes pareadas")
    if not np.allclose(combinado["y"], combinado["y_lightgbm"]):
        raise ValueError("Valores reais divergem entre SARIMA e LightGBM")

    grupos = ["origem", "canal"]
    soma_perfil = combinado.groupby(grupos)["y_hat_lightgbm"].transform("sum")
    n_dias = combinado.groupby(grupos)["data"].transform("nunique")
    perfil = combinado["y_hat_lightgbm"].div(soma_perfil)
    perfil = perfil.where(soma_perfil.gt(0), 1.0 / n_dias)
    total_sarima = combinado.groupby(grupos)["y_hat_sarima"].transform("sum")

    combinado["modelo"] = "combinacao"
    combinado["seed"] = seed_lightgbm
    combinado["y_hat"] = total_sarima * perfil
    return combinado[chaves + ["modelo", "seed", "y", "y_hat"]].sort_values(
        chaves, ignore_index=True
    )


def selecionar_modelos(previsoes: pd.DataFrame) -> pd.DataFrame:
    """Seleciona os tres modelos de referencia e acrescenta a combinacao."""
    filtros = (
        (previsoes["modelo"].eq("sarima_log") & previsoes["seed"].eq(42))
        | (previsoes["modelo"].eq("lightgbm") & previsoes["seed"].eq(-1))
        | (previsoes["modelo"].eq("media_movel_7") & previsoes["seed"].eq(42))
    )
    selecionadas = previsoes.loc[filtros].copy()
    return pd.concat(
        [selecionadas, combinar_previsoes(previsoes)], ignore_index=True
    )


def _resumir_recorte(previsoes: pd.DataFrame, recorte: str) -> pd.DataFrame:
    selecionadas = selecionar_modelos(previsoes)
    total = agregar_total(selecionadas)
    linhas = []
    for modelo in MODELOS:
        parte = total.loc[total["modelo"].eq(modelo)]
        semanas = parte.groupby("origem", as_index=False).agg(
            y=("y", "sum"), y_hat=("y_hat", "sum"), n_dias=("data", "nunique")
        )
        cheias = semanas.loc[semanas["n_dias"].eq(7)]
        linhas.append(
            {
                "recorte": recorte,
                "modelo": modelo,
                "n_blocos": len(semanas),
                "n_dias": parte["data"].nunique(),
                "wape_diario": wape(parte["y"], parte["y_hat"]),
                "vies": bias(parte["y"], parte["y_hat"]),
                "erro_semanal_semanas_cheias": wape(cheias["y"], cheias["y_hat"]),
                "erro_semanal_blocos_parciais": wape(
                    semanas["y"], semanas["y_hat"]
                ),
                "lightgbm_referencia": "seed_-1_media_seeds_42_a_46",
                "diferenca_wape_combinacao_menos_modelo": np.nan,
                "ic95_diferenca_wape_low": np.nan,
                "ic95_diferenca_wape_high": np.nan,
                "p_combinacao_menor": np.nan,
            }
        )

    resultado = pd.DataFrame(linhas)
    for modelo in ["sarima_log", "lightgbm"]:
        comparacao = paired_bootstrap_diff(
            total, "combinacao", modelo, n=2000, seed=42
        )
        mascara = resultado["modelo"].eq(modelo)
        resultado.loc[mascara, "diferenca_wape_combinacao_menos_modelo"] = (
            comparacao["diferenca"]
        )
        resultado.loc[mascara, "ic95_diferenca_wape_low"] = comparacao["ic_low"]
        resultado.loc[mascara, "ic95_diferenca_wape_high"] = comparacao["ic_high"]
        resultado.loc[mascara, "p_combinacao_menor"] = comparacao["p_a_menor_b"]
    return resultado


def gerar_resultados(
    walkforward: pd.DataFrame, holdout: pd.DataFrame
) -> pd.DataFrame:
    """Calcula metricas e intervalos nos tres recortes."""
    recortes = {
        "junho": holdout,
        "walkforward_avaliacao": walkforward,
        "dezesseis_semanas": pd.concat([walkforward, holdout], ignore_index=True),
    }
    return pd.concat(
        [_resumir_recorte(previsoes, nome) for nome, previsoes in recortes.items()],
        ignore_index=True,
    )


def gerar_figura_holdout(previsoes: pd.DataFrame, caminho: Path) -> None:
    """Compara as previsoes diarias de junho na receita total."""
    total = agregar_total(selecionar_modelos(previsoes))
    real = total.loc[total["modelo"].eq("combinacao"), ["data", "y"]]
    previsto = total.pivot(index="data", columns="modelo", values="y_hat")

    figura, eixo = plt.subplots(figsize=(12, 6))
    eixo.plot(real["data"], real["y"], color="#222222", linewidth=2.5, label="Real")
    estilos = {
        "sarima_log": ("SARIMA", "#4C78A8", "--"),
        "lightgbm": ("LightGBM", "#F58518", "-"),
        "combinacao": ("Combinação", "#54A24B", "-"),
    }
    for modelo, (rotulo, cor, estilo) in estilos.items():
        eixo.plot(
            previsto.index,
            previsto[modelo],
            color=cor,
            linestyle=estilo,
            linewidth=1.8,
            label=rotulo,
        )
    eixo.set_title("Junho de 2026 — receita diária real e prevista")
    eixo.set_xlabel("Data")
    eixo.set_ylabel("Receita (R$ mi)")
    eixo.yaxis.set_major_formatter(
        FuncFormatter(lambda valor, _: f"R$ {valor / 1e6:.1f} mi".replace(".", ","))
    )
    eixo.grid(axis="y", alpha=0.25)
    eixo.legend(ncol=4, frameon=False)
    figura.autofmt_xdate()
    figura.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(figura)


def main() -> None:
    walkforward = pd.read_parquet(PASTA_DADOS / "predicoes_walkforward.parquet")
    holdout = pd.read_parquet(PASTA_DADOS / "predicoes_holdout.parquet")
    resultado = gerar_resultados(walkforward, holdout)
    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(
        PASTA_METRICAS / "combinacao.csv", index=False, float_format="%.8f"
    )
    gerar_figura_holdout(holdout, PASTA_FIGURAS / "combinacao_holdout.png")


if __name__ == "__main__":
    main()
