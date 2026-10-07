"""Visualizacoes do treinamento para audiencia nao tecnica."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import record_evaluation
from matplotlib.animation import FuncAnimation, PillowWriter

from src.evaluate import (
    agregar_total,
    paired_bootstrap_values,
    selecionar_previsao_ensemble,
    wape_total_por_data,
)
from src.models import LightGBMSemanal, XGBoostSemanal


PASTA_FIGURAS = Path("reports/figures")


def _walkforward_gif(features: pd.DataFrame, previsoes: pd.DataFrame) -> None:
    real = features.groupby("data", as_index=False)["y"].sum()
    previsto = agregar_total(
        previsoes.loc[previsoes["modelo"].eq("lightgbm") & previsoes["seed"].eq(-1)]
    )
    origens = sorted(previsto["origem"].unique())
    fig, ax = plt.subplots(figsize=(11, 5.5))

    def desenhar(i: int):
        ax.clear()
        origem = pd.Timestamp(origens[i])
        acumulado = previsto.loc[previsto["origem"].le(origem)]
        ax.plot(real["data"], real["y"] / 1e6, color="black", linewidth=1.5, label="Receita real")
        ax.plot(
            acumulado["data"],
            acumulado["y_hat"] / 1e6,
            color="#e67e22",
            linewidth=2,
            marker="o",
            markersize=2,
            label="Previsão LightGBM",
        )
        ax.axvspan(pd.Timestamp("2025-11-01"), origem, color="#3498db", alpha=0.14, label="Treino")
        ax.axvspan(origem, origem + pd.Timedelta(days=7), color="#f39c12", alpha=0.22, label="Semana prevista")
        erro = wape_total_por_data(
            acumulado["y"], acumulado["y_hat"], acumulado["data"]
        )
        ax.set_title(f"Walk-forward — origem {origem:%d/%m/%Y} — WAPE das semanas previstas {erro:.1%}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Receita total (R$ mi)")
        ax.legend(loc="upper right", ncol=2)
        ax.set_xlim(pd.Timestamp("2025-11-01"), pd.Timestamp("2026-06-30"))
        return ax.lines

    animacao = FuncAnimation(fig, desenhar, frames=len(origens), interval=550, blit=False)
    animacao.save(PASTA_FIGURAS / "walkforward.gif", writer=PillowWriter(fps=2), dpi=100)
    plt.close(fig)


def _walkforward_html(features: pd.DataFrame, previsoes: pd.DataFrame) -> None:
    real = features.groupby("data", as_index=False)["y"].sum()
    previsto = agregar_total(
        previsoes.loc[previsoes["modelo"].eq("lightgbm") & previsoes["seed"].eq(-1)]
    )
    origens = sorted(previsto["origem"].unique())
    quadros = []
    for origem in origens:
        parte = previsto.loc[previsto["origem"].le(origem)]
        quadros.append(
            {
                "name": str(pd.Timestamp(origem).date()),
                "data": [{"x": parte["data"].dt.strftime("%Y-%m-%d").tolist(), "y": (parte["y_hat"] / 1e6).tolist()}],
            }
        )
    dados = {
        "real_x": real["data"].dt.strftime("%Y-%m-%d").tolist(),
        "real_y": (real["y"] / 1e6).tolist(),
        "frames": quadros,
    }
    html = """<!doctype html><meta charset="utf-8"><title>Walk-forward interativo</title>
<div id="grafico" style="width:100%;height:650px"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><script>
const d = __DADOS__;
const real = {x:d.real_x,y:d.real_y,name:'Receita real',mode:'lines',line:{color:'black'}};
const prev = {x:d.frames[0].data[0].x,y:d.frames[0].data[0].y,name:'Previsão LightGBM',mode:'lines+markers'};
Plotly.newPlot('grafico',[real,prev],{title:'Walk-forward',xaxis:{title:'Data'},yaxis:{title:'Receita total (R$ mi)'},
sliders:[{active:0,steps:d.frames.map((f,i)=>({label:f.name,method:'animate',args:[[f.name],{mode:'immediate',frame:{duration:0,redraw:true}}]}))}]});
Plotly.addFrames('grafico',d.frames.map(f=>({name:f.name,data:[{},f.data[0]],traces:[0,1]})));
</script>""".replace("__DADOS__", json.dumps(dados, ensure_ascii=False))
    (PASTA_FIGURAS / "walkforward_interativo.html").write_text(html, encoding="utf-8")


def _boosting_passos(features: pd.DataFrame, modelo) -> None:
    holdout = features.loc[features["data"].gt("2026-05-31")].copy()
    real = holdout.groupby("data", as_index=False)["y"].sum()
    total_arvores = int(modelo.model.n_estimators_)
    passos = [1, min(10, total_arvores), min(50, total_arvores), total_arvores]
    fig, eixos = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, passo in zip(eixos.flat, passos):
        previsto = modelo.predict(holdout, num_iteration=passo)
        quadro = pd.DataFrame({"data": holdout["data"], "y_hat": previsto}).groupby("data", as_index=False).sum()
        ax.plot(real["data"], real["y"] / 1e6, color="black", label="Real")
        ax.plot(quadro["data"], quadro["y_hat"] / 1e6, color="#e67e22", label="Previsão")
        ax.set_title(f"{passo} árvore{'s' if passo != 1 else ''}")
        ax.set_ylabel("Receita (R$ mi)")
        ax.tick_params(axis="x", rotation=30)
    eixos[0, 0].legend()
    fig.suptitle("Cada árvore corrige a anterior — LightGBM")
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "boosting_passos.png", dpi=160)
    plt.close(fig)


def _curva_aprendizado(
    features: pd.DataFrame, parametros: dict[str, dict]
) -> None:
    treino = features.loc[features["data"].le("2026-05-24")]
    validacao = features.loc[features["data"].between("2026-05-25", "2026-05-31")]
    resultados = {}

    lgb = LightGBMSemanal(parametros["lightgbm"], random_state=42)
    lgb.categorias_ = {}
    x_treino = lgb._ajustar_categorias(treino, treino=True)
    x_val = lgb._ajustar_categorias(validacao, treino=False)
    historico_lgb: dict = {}
    y_treino = np.log1p(np.clip(treino["y"].to_numpy(dtype=float), 0, None))
    y_val = np.log1p(np.clip(validacao["y"].to_numpy(dtype=float), 0, None))
    lgb.model.fit(
        x_treino,
        y_treino,
        eval_X=(x_treino, x_val),
        eval_y=(y_treino, y_val),
        eval_metric="l2",
        callbacks=[record_evaluation(historico_lgb)],
    )
    resultados["LightGBM"] = (
        historico_lgb["training"]["l2"],
        historico_lgb["valid_1"]["l2"],
    )

    xgb = XGBoostSemanal(parametros["xgboost"], random_state=42)
    xgb.categorias_ = {}
    x_treino = xgb._ajustar_categorias(treino, treino=True)
    x_val = xgb._ajustar_categorias(validacao, treino=False)
    xgb.model.fit(x_treino, y_treino)
    iteracoes = range(1, int(xgb.model.n_estimators) + 1)
    resultados["XGBoost"] = (
        [
            np.mean(
                np.square(
                    y_treino
                    - xgb.model.predict(x_treino, iteration_range=(0, iteracao))
                )
            )
            for iteracao in iteracoes
        ],
        [
            np.mean(
                np.square(
                    y_val
                    - xgb.model.predict(x_val, iteration_range=(0, iteracao))
                )
            )
            for iteracao in iteracoes
        ],
    )

    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (nome, (erro_treino, erro_val)) in zip(eixos, resultados.items()):
        ax.plot(np.arange(1, len(erro_treino) + 1), erro_treino, label="Treino")
        ax.plot(np.arange(1, len(erro_val) + 1), erro_val, label="Validação")
        ax.set_title(nome)
        ax.set_xlabel("Número de árvores")
        ax.set_ylabel("Erro L2 no alvo log1p")
        ax.legend()
    fig.suptitle("Curva de aprendizado — última semana antes do holdout")
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "curva_aprendizado.png", dpi=160)
    plt.close(fig)


def _optuna_historico(historico: pd.DataFrame) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, nome in zip(eixos, ["lightgbm", "xgboost"]):
        parte = historico.loc[historico["modelo"].eq(nome)].sort_values("trial")
        ax.scatter(parte["trial"], parte["wape"], alpha=0.7, label="Tentativa")
        ax.plot(parte["trial"], parte["wape"].cummin(), color="black", label="Melhor até aqui")
        ax.set_title(nome)
        ax.set_xlabel("Tentativa")
        ax.set_ylabel("WAPE médio")
        ax.legend()
    fig.suptitle("Histórico do tuning Optuna")
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "optuna_historico.png", dpi=160)
    plt.close(fig)


def _torneio(resumo: pd.DataFrame) -> None:
    parte = resumo.loc[
        resumo["conjunto"].eq("holdout")
        & resumo["granularidade"].eq("total")
        & resumo["segmento"].eq("todos")
    ].sort_values("wape")
    erros = np.vstack(
        [parte["wape"] - parte["wape_ic_low"], parte["wape_ic_high"] - parte["wape"]]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(parte["modelo"], parte["wape"], yerr=erros, capsize=4, color="#5dade2")
    ax.set_title("Torneio no período de teste — receita total")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("WAPE (IC 95%)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "torneio_wape.png", dpi=160)
    plt.close(fig)


def _contraprova(previsoes: pd.DataFrame) -> None:
    total = agregar_total(selecionar_previsao_ensemble(previsoes))
    valores = paired_bootstrap_values(total, "xgboost", "lightgbm", n=2000, seed=42)
    baixo, alto = np.percentile(valores, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(valores, bins=35, color="#7fb3d5", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.5, label="Sem diferença")
    ax.axvspan(baixo, alto, color="#f5b041", alpha=0.25, label=f"IC 95% [{baixo:.3f}, {alto:.3f}]")
    ax.set_title("Contraprova bootstrap no período de teste")
    ax.set_xlabel("WAPE(XGBoost) − WAPE(LightGBM)")
    ax.set_ylabel("Reamostras")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "contraprova_bootstrap.png", dpi=160)
    plt.close(fig)


def gerar_visualizacoes(
    features: pd.DataFrame,
    walkforward: pd.DataFrame,
    holdout: pd.DataFrame,
    resumo: pd.DataFrame,
    contraprova: pd.DataFrame,
    historico: pd.DataFrame,
    parametros: dict[str, dict],
    finais: dict,
) -> None:
    """Gera todos os artefatos visuais exigidos pela spec."""
    del contraprova
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    _walkforward_gif(features, walkforward)
    _walkforward_html(features, walkforward)
    _boosting_passos(features, finais["lightgbm"])
    _curva_aprendizado(features, parametros)
    _optuna_historico(historico)
    _torneio(resumo)
    _contraprova(holdout)
