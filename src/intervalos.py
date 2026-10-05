"""Metricas complementares e intervalo de previsao P10-P90."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_daily
from src.evaluate import (
    agregar_total,
    bias,
    cobertura,
    fva,
    pinball,
    rmse,
    selecionar_previsao_ensemble,
    totais_semanais,
    wape,
)
from src.features import build_features
from src.models import criar_modelo
from src.train import criar_fold, origens_avaliacao, origens_holdout, origens_tuning

PASTA_DADOS = Path("data/processed")
PASTA_METRICAS = Path("reports/metrics")
PASTA_FIGURAS = Path("reports/figures")
QUANTIS = {"p10": 0.1, "p90": 0.9}


def _carregar_previsoes() -> dict[str, pd.DataFrame]:
    conjuntos = {}
    for conjunto, arquivo in [("walkforward", "predicoes_walkforward.parquet"), ("holdout", "predicoes_holdout.parquet")]:
        previsoes = selecionar_previsao_ensemble(pd.read_parquet(PASTA_DADOS / arquivo))
        conjuntos[conjunto] = agregar_total(previsoes)
    return conjuntos


def metricas_extras(conjuntos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """FVA, erro semanal, RMSE e pior dia, na receita total diaria."""
    linhas = []
    for conjunto, total in conjuntos.items():
        wape_modelo = {modelo: wape(p["y"], p["y_hat"]) for modelo, p in total.groupby("modelo")}
        for modelo, parte in total.groupby("modelo"):
            semanal = totais_semanais(parte)
            erro = (parte["y"] - parte["y_hat"]).abs()
            pior = parte.loc[erro.idxmax()]
            linhas.append({
                "modelo": modelo,
                "conjunto": conjunto,
                "wape": wape_modelo[modelo],
                "fva_vs_naive": fva(wape_modelo[modelo], wape_modelo["naive_sazonal"]),
                "fva_vs_media_movel": fva(wape_modelo[modelo], wape_modelo["media_movel_7"]),
                "wape_semanal": wape(semanal["y"], semanal["y_hat"]),
                "bias_semanal": bias(semanal["y"], semanal["y_hat"]),
                "rmse": rmse(parte["y"], parte["y_hat"]),
                "erro_max_dia": float(erro.max()),
                "data_erro_max": pd.Timestamp(pior["data"]).date().isoformat(),
            })
    return pd.DataFrame(linhas)


def erro_por_horizonte(conjuntos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """WAPE e bias por dias de antecedencia (1 a 7)."""
    linhas = []
    for conjunto, total in conjuntos.items():
        total = total.assign(horizonte=(total["data"] - total["origem"]).dt.days)
        for (modelo, horizonte), parte in total.groupby(["modelo", "horizonte"]):
            linhas.append({
                "modelo": modelo,
                "conjunto": conjunto,
                "horizonte": int(horizonte),
                "wape": wape(parte["y"], parte["y_hat"]),
                "bias": bias(parte["y"], parte["y_hat"]),
                "n_dias": len(parte),
            })
    return pd.DataFrame(linhas)


def erro_por_semana(conjuntos: dict[str, pd.DataFrame], receita_diaria: pd.Series) -> pd.DataFrame:
    """WAPE por semana prevista e a virada de nivel da semana (|media da semana / media da anterior - 1|)."""
    linhas = []
    for conjunto, total in conjuntos.items():
        for origem, parte in total.groupby("origem"):
            inicio, fim = parte["data"].min(), parte["data"].max()
            anterior = receita_diaria[inicio - pd.Timedelta(days=7): inicio - pd.Timedelta(days=1)].mean()
            linha = {
                "conjunto": conjunto,
                "inicio": inicio.date().isoformat(),
                "fim": fim.date().isoformat(),
                "virada_nivel": abs(receita_diaria[inicio:fim].mean() / anterior - 1),
            }
            for modelo, p in parte.groupby("modelo"):
                linha[modelo] = wape(p["y"], p["y_hat"])
            linhas.append(linha)
    return pd.DataFrame(linhas)


def prever_quantis(features: pd.DataFrame, origens: list[pd.Timestamp], parametros: dict) -> pd.DataFrame:
    """Treina LightGBM quantilico P10 e P90 por origem e preve a semana seguinte."""
    partes = []
    for origem in origens:
        treino, teste = criar_fold(features, origem)
        previsto = {}
        for nome, alpha in QUANTIS.items():
            params = {**parametros, "objective": "quantile", "alpha": alpha}
            modelo = criar_modelo("lightgbm", params, random_state=42).fit(treino)
            previsto[nome] = modelo.predict(teste)
        baixo = np.minimum(previsto["p10"], previsto["p90"])
        alto = np.maximum(previsto["p10"], previsto["p90"])
        partes.append(pd.DataFrame({
            "data": teste["data"].to_numpy(),
            "canal": teste["canal"].to_numpy(),
            "origem": origem,
            "y": teste["y"].to_numpy(dtype=float),
            "p10": baixo,
            "p90": alto,
        }))
    return pd.concat(partes, ignore_index=True)


def resumir_intervalos(quantis: pd.DataFrame, ponto_total: pd.DataFrame, conjunto: str) -> list[dict]:
    """Cobertura, largura e perda quantilica no total diario e por canal."""
    total = quantis.groupby(["data", "origem"], as_index=False)[["y", "p10", "p90"]].sum()
    total = total.merge(ponto_total[["data", "y_hat"]], on="data", how="left")
    largura = total["p90"] - total["p10"]
    linhas = [{
        "conjunto": conjunto,
        "nivel": "total",
        "cobertura": cobertura(total["y"], total["p10"], total["p90"]),
        "largura_media_rs": float(largura.mean()),
        "largura_media_pct": float((largura / total["y_hat"]).mean()),
        "pinball_p10": pinball(total["y"], total["p10"], 0.1),
        "pinball_p90": pinball(total["y"], total["p90"], 0.9),
        "n_dias": len(total),
    }]
    for canal, parte in quantis.groupby("canal"):
        linhas.append({
            "conjunto": conjunto,
            "nivel": canal,
            "cobertura": cobertura(parte["y"], parte["p10"], parte["p90"]),
            "largura_media_rs": float((parte["p90"] - parte["p10"]).mean()),
            "largura_media_pct": np.nan,
            "pinball_p10": pinball(parte["y"], parte["p10"], 0.1),
            "pinball_p90": pinball(parte["y"], parte["p90"], 0.9),
            "n_dias": len(parte),
        })
    return linhas


def _total_diario(quantis: pd.DataFrame) -> pd.DataFrame:
    return quantis.groupby(["data", "origem"], as_index=False)[["y", "p10", "p90"]].sum()


def ajuste_conformal(quantis_calibracao: pd.DataFrame, cobertura_alvo: float = 0.8) -> float:
    """Folga (em log) que faz a faixa conter o real na fracao alvo dos dias de calibracao.

    Conformal quantile regression: escore = quanto o real ficou fora da faixa,
    em escala log; a folga e o quantil (com correcao de amostra finita) desses escores.
    """
    total = _total_diario(quantis_calibracao)
    real = np.log1p(total["y"])
    escore = np.maximum(np.log1p(total["p10"]) - real, real - np.log1p(total["p90"]))
    n = len(escore)
    nivel = min(1.0, np.ceil((n + 1) * cobertura_alvo) / n)
    return float(np.quantile(escore, nivel, method="higher"))


def aplicar_conformal(quantis: pd.DataFrame, folga: float) -> pd.DataFrame:
    """Alarga a faixa do total diario pela folga calibrada (multiplicativa)."""
    total = _total_diario(quantis)
    total["p10"] = np.expm1(np.log1p(total["p10"]) - folga).clip(lower=0)
    total["p90"] = np.expm1(np.log1p(total["p90"]) + folga)
    return total


def _metricas_faixa(total: pd.DataFrame, ponto_total: pd.DataFrame, conjunto: str, metodo: str) -> dict:
    total = total.merge(ponto_total[["data", "y_hat"]], on="data", how="left")
    largura = total["p90"] - total["p10"]
    return {
        "conjunto": conjunto,
        "nivel": "total",
        "metodo": metodo,
        "cobertura": cobertura(total["y"], total["p10"], total["p90"]),
        "largura_media_rs": float(largura.mean()),
        "largura_media_pct": float((largura / total["y_hat"]).mean()),
        "pinball_p10": pinball(total["y"], total["p10"], 0.1),
        "pinball_p90": pinball(total["y"], total["p90"], 0.9),
        "n_dias": len(total),
    }


def figura_intervalo(total: pd.DataFrame, ponto_total: pd.DataFrame) -> Path:
    total = total.groupby("data", as_index=False)[["y", "p10", "p90"]].sum()
    total = total.merge(ponto_total[["data", "y_hat"]], on="data", how="left")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(total["data"], total["p10"] / 1e6, total["p90"] / 1e6, color="#2F5D8A", alpha=0.18, label="Faixa provável (P10–P90)")
    ax.plot(total["data"], total["y_hat"] / 1e6, color="#2F5D8A", lw=2, label="Previsão LightGBM")
    ax.plot(total["data"], total["y"] / 1e6, color="#1E2A2F", lw=2.5, label="Receita real")
    ax.set_ylabel("Receita total (R$ mi)")
    ax.set_title("Junho/2026: previsão com faixa calibrada para 80%")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    caminho = PASTA_FIGURAS / "intervalo_holdout.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return caminho


def main() -> None:
    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)
    conjuntos = _carregar_previsoes()
    metricas_extras(conjuntos).to_csv(PASTA_METRICAS / "metricas_extras.csv", index=False, float_format="%.6f")
    erro_por_horizonte(conjuntos).to_csv(PASTA_METRICAS / "erro_por_horizonte.csv", index=False, float_format="%.6f")
    receita_diaria = load_daily().set_index("data")["receita"]
    erro_por_semana(conjuntos, receita_diaria).to_csv(PASTA_METRICAS / "erro_por_semana.csv", index=False, float_format="%.6f")

    features = build_features(load_daily(by=["canal"]))
    parametros = json.loads((PASTA_METRICAS / "melhores_hiperparametros.json").read_text(encoding="utf-8"))["lightgbm"]
    folga = ajuste_conformal(prever_quantis(features, origens_tuning(), parametros))
    linhas = []
    for conjunto, origens in [("walkforward", origens_avaliacao()), ("holdout", origens_holdout())]:
        quantis = prever_quantis(features, origens, parametros)
        ponto = conjuntos[conjunto].loc[conjuntos[conjunto]["modelo"].eq("lightgbm")]
        for linha in resumir_intervalos(quantis, ponto, conjunto):
            linhas.append({**linha, "metodo": "quantilico"})
        calibrado = aplicar_conformal(quantis, folga)
        linhas.append({**_metricas_faixa(calibrado, ponto, conjunto, "conformal"), "folga_log": folga})
        if conjunto == "holdout":
            figura_intervalo(calibrado, ponto)
    pd.DataFrame(linhas).to_csv(PASTA_METRICAS / "intervalos.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
