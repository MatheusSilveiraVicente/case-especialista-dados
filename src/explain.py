"""Explicabilidade e diagnosticos dos modelos no holdout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.evaluate import selecionar_previsao_ensemble, wape_total_por_data


PASTA_FIGURAS = Path("reports/figures")
PASTA_METRICAS = Path("reports/metrics")
NOMES_WATERFALL = {
    "media_7_lag7": "média_7",
    "dias_ate_evento_presente": "dias_até_presente",
    "dias_ate_evento_promo": "dias_até_promo",
    "taxa_desconto_media_7_lag7": "desconto_médio_7",
    "fourier_mes_sin1": "seno_mês_1",
    "fourier_mes_sin2": "seno_mês_2",
    "fourier_mes_cos1": "cosseno_mês_1",
    "fourier_mes_cos2": "cosseno_mês_2",
}
NOMES_NEGOCIO = {
    "dias_ate_evento_presente": "dias até a data de presente",
    "media_7_lag7": "nível da semana anterior",
    "lag_7": "venda do mesmo dia na semana anterior",
    "fourier_mes_sin1": "posição dentro do mês",
    "taxa_desconto_media_7_lag7": "desconto médio da semana anterior",
    "canal": "canal de venda",
    "share_canal_lag7": "participação recente do canal",
}


def _semana(features: pd.DataFrame, origem: pd.Timestamp) -> pd.DataFrame:
    fim = min(origem + pd.Timedelta(days=7), pd.Timestamp("2026-06-30"))
    return features.loc[
        features["data"].gt(origem) & features["data"].le(fim)
    ].copy()


def _shap_monetario(modelo: Any, dados: pd.DataFrame):
    x = modelo._ajustar_categorias(dados, treino=False)
    explicador = shap.TreeExplainer(modelo.model)
    valores_log = np.asarray(explicador.shap_values(x))
    base_log = np.asarray(explicador.expected_value).reshape(-1)[0]
    soma_log = valores_log.sum(axis=1)
    base = float(np.expm1(base_log))
    previsto = np.expm1(base_log + soma_log)
    fator = np.divide(
        previsto - base,
        soma_log,
        out=np.full_like(soma_log, np.exp(base_log)),
        where=np.abs(soma_log) > 1e-12,
    )
    valores = valores_log * fator[:, None]
    exibicao = x.copy()
    for coluna in modelo.categoricas:
        exibicao[coluna] = exibicao[coluna].cat.codes
    return exibicao, valores, np.full(len(exibicao), base)


def _salvar_shap(
    features: pd.DataFrame, modelos: dict[pd.Timestamp, Any]
) -> list[str]:
    exibicoes = []
    valores = []
    bases = []
    for origem, modelo in sorted(modelos.items()):
        exibicao, valor, base = _shap_monetario(modelo, _semana(features, origem))
        exibicoes.append(exibicao)
        valores.append(valor)
        bases.append(base)
    exibicao = pd.concat(exibicoes, ignore_index=True)
    valores_shap = np.vstack(valores)
    bases_shap = np.concatenate(bases)
    features_modelo = next(iter(modelos.values())).features
    importancia = pd.DataFrame(
        {
            "feature": features_modelo,
            "shap_abs_medio": np.abs(valores_shap).mean(axis=0),
        }
    ).sort_values("shap_abs_medio", ascending=False)
    importancia.to_csv(PASTA_METRICAS / "shap_importancia.csv", index=False)

    shap.summary_plot(
        valores_shap / 1e3,
        exibicao,
        show=False,
        max_display=20,
        rng=np.random.default_rng(42),
    )
    plt.title("Importância SHAP — LightGBM no período de teste")
    plt.gca().set_xlabel("Impacto SHAP (R$ mil)")
    figura = plt.gcf()
    if len(figura.axes) > 1:
        figura.axes[-1].set_ylabel("Valor da variável")
    for eixo in figura.axes:
        marcas = eixo.get_yticks()
        rotulos = [texto.get_text() for texto in eixo.get_yticklabels()]
        if rotulos == ["Low", "High"]:
            eixo.set_yticks(marcas, ["Baixo", "Alto"])
        for texto in eixo.texts:
            if texto.get_text() == "High":
                texto.set_text("Alto")
            elif texto.get_text() == "Low":
                texto.set_text("Baixo")
    plt.tight_layout()
    plt.savefig(PASTA_FIGURAS / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    for arquivo in PASTA_FIGURAS.glob("shap_dependence_*.png"):
        arquivo.unlink()
    top5 = importancia.head(5)["feature"].tolist()
    for feature in top5:
        shap.dependence_plot(
            feature,
            valores_shap / 1e3,
            exibicao,
            show=False,
            interaction_index=None,
        )
        plt.title(f"Efeito SHAP de {feature}")
        plt.xlabel(feature)
        plt.ylabel("Contribuição SHAP (R$ mil)")
        plt.tight_layout()
        plt.savefig(
            PASTA_FIGURAS / f"shap_dependence_{feature}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close()

    origem = pd.Timestamp("2026-06-07")
    semana = _semana(features, origem)
    alvo = semana["data"].eq("2026-06-12") & semana["canal"].eq("App")
    if alvo.any():
        _, valores_alvo, bases_alvo = _shap_monetario(
            modelos[origem], semana.loc[alvo]
        )
        contribuicoes = valores_alvo[0] / 1e3
        indices = np.argsort(np.abs(contribuicoes))[::-1][:9]
        restantes = np.setdiff1d(np.arange(len(contribuicoes)), indices)
        nomes = [NOMES_WATERFALL.get(features_modelo[i], features_modelo[i]) for i in indices]
        valores_exibidos = [contribuicoes[i] for i in indices]
        nomes.append("Outras variáveis")
        valores_exibidos.append(float(contribuicoes[restantes].sum()))

        fig, ax = plt.subplots(figsize=(15, 9))
        atual = bases_alvo[0] / 1e3
        for posicao, (nome, valor) in enumerate(zip(nomes, valores_exibidos)):
            seguinte = atual + valor
            esquerda = min(atual, seguinte)
            cor = "#e74c3c" if valor >= 0 else "#2980b9"
            ax.barh(posicao, abs(valor), left=esquerda, color=cor, height=0.55)
            ax.annotate(
                f"{valor:+.2f}",
                xy=((atual + seguinte) / 2, posicao - 0.34),
                va="bottom",
                ha="center",
                color=cor,
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
            )
            atual = seguinte
        base_mil = bases_alvo[0] / 1e3
        ax.axvline(base_mil, color="#7f8c8d", linestyle="--", linewidth=1)
        ax.axvline(atual, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(nomes)), nomes)
        ax.invert_yaxis()
        ax.set_title("Previsão do App em 12/06/2026")
        ax.set_xlabel("Contribuição para a previsão (R$ mil)")
        ax.set_ylabel("Variável")
        ax.annotate(
            f"Base: {base_mil:.2f}",
            xy=(base_mil, -0.65),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            color="#7f8c8d",
            fontsize=9,
        )
        ax.annotate(
            f"Previsão: {atual:.2f}",
            xy=(atual, -0.65),
            xytext=(8, 10),
            textcoords="offset points",
            ha="left",
            fontsize=9,
        )
        ax.margins(x=0.12)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            PASTA_FIGURAS / "shap_waterfall_namorados.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)
    return top5


def _prever_por_origem(
    modelos: dict[pd.Timestamp, Any], dados: pd.DataFrame
) -> np.ndarray:
    previsto = pd.Series(index=dados.index, dtype=float)
    for origem, modelo in modelos.items():
        fim = min(origem + pd.Timedelta(days=7), pd.Timestamp("2026-06-30"))
        mascara = dados["data"].gt(origem) & dados["data"].le(fim)
        previsto.loc[mascara] = modelo.predict(dados.loc[mascara])
    if previsto.isna().any():
        raise ValueError("Ha linhas do holdout sem modelo da origem")
    return previsto.to_numpy()


def _permutation_importance(
    holdout: pd.DataFrame,
    modelos_por_nome: dict[str, dict[pd.Timestamp, Any]],
) -> pd.DataFrame:
    gerador = np.random.default_rng(42)
    linhas = []
    for nome in ["lightgbm", "ridge"]:
        modelos = modelos_por_nome[nome]
        features_modelo = next(iter(modelos.values())).features
        referencia = wape_total_por_data(
            holdout["y"], _prever_por_origem(modelos, holdout), holdout["data"]
        )
        for feature in features_modelo:
            erros = []
            for _ in range(30):
                permutado = holdout.copy()
                permutado[feature] = gerador.permutation(
                    permutado[feature].to_numpy()
                )
                erros.append(
                    wape_total_por_data(
                        holdout["y"],
                        _prever_por_origem(modelos, permutado),
                        holdout["data"],
                    )
                )
            erros_array = np.asarray(erros)
            aumentos = erros_array - referencia
            linhas.append(
                {
                    "modelo": nome,
                    "feature": feature,
                    "wape_referencia": referencia,
                    "wape_permutado_media": float(erros_array.mean()),
                    "wape_permutado_std": float(erros_array.std(ddof=1)),
                    "aumento_wape_media": float(aumentos.mean()),
                    "aumento_wape_std": float(aumentos.std(ddof=1)),
                }
            )
    importancia = pd.DataFrame(linhas).sort_values(
        ["modelo", "aumento_wape_media"], ascending=[True, False]
    )
    importancia.to_csv(PASTA_METRICAS / "permutation_importance.csv", index=False)
    return importancia


def _fig_importancia_negocio(importancia: pd.DataFrame) -> None:
    top = (
        importancia.loc[importancia["modelo"].eq("lightgbm")]
        .nlargest(5, "aumento_wape_media")
        .sort_values("aumento_wape_media")
        .copy()
    )
    top["rotulo"] = top["feature"].map(NOMES_NEGOCIO).fillna(top["feature"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(top["rotulo"], top["aumento_wape_media"] * 100, color="#2F5D8A")
    ax.set_xlabel("Aumento do WAPE ao embaralhar a variável (p.p.)")
    ax.set_title("As variáveis que mais sustentam a previsão")
    ax.text(
        0,
        -0.2,
        "Referência: LightGBM seed 42; importância por permutação no diagnóstico de junho.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4D4D4D",
    )
    fig.subplots_adjust(bottom=0.24, left=0.33)
    fig.savefig(PASTA_FIGURAS / "importancia_negocio.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _nome_coeficiente(nome: str) -> str:
    return nome.replace("categoricas__", "").replace("numericas__", "")


def _salvar_coeficientes(modelo: Any) -> None:
    preparo = modelo.pipeline.named_steps["preparo"]
    nomes = [_nome_coeficiente(nome) for nome in preparo.get_feature_names_out()]
    coeficientes = pd.DataFrame(
        {
            "feature": nomes,
            "coeficiente_padronizado": modelo.pipeline.named_steps["modelo"].coef_,
        }
    )
    coeficientes["valor_absoluto"] = coeficientes["coeficiente_padronizado"].abs()
    coeficientes = coeficientes.sort_values("valor_absoluto", ascending=False)
    coeficientes.to_csv(PASTA_METRICAS / "ridge_coeficientes.csv", index=False)

    top = coeficientes.head(15).sort_values("coeficiente_padronizado")
    cores = np.where(top["coeficiente_padronizado"].ge(0), "#2e86c1", "#c0392b")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["coeficiente_padronizado"], color=cores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Coeficientes padronizados do Ridge — origem 31/05/2026")
    ax.set_xlabel("Coeficiente no alvo log1p")
    ax.set_ylabel("Variável")
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "ridge_coeficientes.png", dpi=160)
    plt.close(fig)


def _graficos_residuos(previsoes: pd.DataFrame) -> None:
    base = selecionar_previsao_ensemble(previsoes)
    base["dia_semana"] = pd.to_datetime(base["data"]).dt.dayofweek
    base["residuo"] = (base["y"] - base["y_hat"]) / 1e6
    resumo = base.groupby(["modelo", "dia_semana"], as_index=False)["residuo"].mean()
    nomes_dia = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for modelo, parte in resumo.groupby("modelo"):
        ax.plot(parte["dia_semana"], parte["residuo"], marker="o", label=modelo)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(7), nomes_dia)
    ax.set_title("Resíduo médio por dia da semana — período de teste")
    ax.set_xlabel("Dia da semana")
    ax.set_ylabel("Real − previsto (R$ mi)")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "residuos_dia_semana.png", dpi=160)
    plt.close(fig)

    total = base.groupby(["data", "modelo"], as_index=False)[["y", "y_hat"]].sum()
    real = total.groupby("data", as_index=False)["y"].first()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(real["data"], real["y"] / 1e6, color="black", linewidth=2.2, label="Real")
    for modelo, parte in total.groupby("modelo"):
        ax.plot(parte["data"], parte["y_hat"] / 1e6, linewidth=1.2, label=modelo)
    ax.set_title("Receita real e prevista no período de teste")
    ax.set_xlabel("Data")
    ax.set_ylabel("Receita total (R$ mi)")
    ax.legend(ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "real_vs_previsto_holdout.png", dpi=160)
    plt.close(fig)


def gerar_explicacoes(
    features: pd.DataFrame,
    previsoes: pd.DataFrame,
    modelos_por_nome: dict[str, dict[pd.Timestamp, Any]],
) -> list[str]:
    """Gera SHAP, permutacao, coeficientes e graficos de residuos."""
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)
    holdout = features.loc[
        features["data"].gt("2026-05-31")
        & features["data"].le("2026-06-30")
    ].copy()
    top5 = _salvar_shap(features, modelos_por_nome["lightgbm"])
    importancia = _permutation_importance(holdout, modelos_por_nome)
    _fig_importancia_negocio(importancia)
    _salvar_coeficientes(
        modelos_por_nome["ridge"][pd.Timestamp("2026-05-31")]
    )
    _graficos_residuos(previsoes)
    return top5
