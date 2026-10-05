"""Executa tuning, walk-forward, holdout, contraprova e explicabilidade."""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.data_loader import load_daily
from src.evaluate import (
    METRICAS,
    agregar_total,
    bootstrap_ci,
    paired_bootstrap_diff,
    selecionar_previsao_ensemble,
    wape,
    wape_total_por_data,
)
from src.features import build_features
from src.models import criar_modelo


SEEDS = range(42, 47)
MODELOS = ["naive_sazonal", "media_movel_7", "ridge", "xgboost", "lightgbm"]
PASTA_DADOS = Path("data/processed")
PASTA_METRICAS = Path("reports/metrics")
PASTA_MODELOS = Path("models")
LIMITE_WALKFORWARD = pd.Timestamp("2026-05-31")
ORDEM_SIMPLICIDADE = {
    "naive_sazonal": 0,
    "media_movel_7": 1,
    "ridge": 2,
    "xgboost": 3,
    "lightgbm": 3,
}


def origens_pre_holdout() -> list[pd.Timestamp]:
    """Origens semanais anteriores ao holdout."""
    origens: list[pd.Timestamp] = []
    origem = pd.Timestamp("2025-12-30")
    while origem < LIMITE_WALKFORWARD:
        origens.append(origem)
        origem += pd.Timedelta(days=7)
    return origens


def origens_tuning() -> list[pd.Timestamp]:
    """Origens de indice par reservadas ao tuning."""
    return origens_pre_holdout()[::2]


def origens_avaliacao() -> list[pd.Timestamp]:
    """Origens de indice impar reservadas a avaliacao."""
    return origens_pre_holdout()[1::2]


def origens_holdout() -> list[pd.Timestamp]:
    """Origens semanais congeladas para junho de 2026."""
    return [
        pd.Timestamp(data)
        for data in ["2026-05-31", "2026-06-07", "2026-06-14", "2026-06-21", "2026-06-28"]
    ]


def criar_fold(
    df: pd.DataFrame, origem: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa passado disponivel e os sete dias seguintes."""
    origem = pd.Timestamp(origem)
    limite = (
        LIMITE_WALKFORWARD
        if origem < LIMITE_WALKFORWARD
        else pd.Timestamp(df["data"].max())
    )
    fim = min(origem + pd.Timedelta(days=7), limite)
    treino = df.loc[df["data"].le(origem)].copy()
    teste = df.loc[df["data"].gt(origem) & df["data"].le(fim)].copy()
    return treino, teste


def _avaliar_parametros(
    df: pd.DataFrame,
    origens: list[pd.Timestamp],
    nome: str,
    params: dict[str, Any],
) -> float:
    erros = []
    for origem in origens:
        treino, teste = criar_fold(df, origem)
        modelo = criar_modelo(nome, params=params, random_state=42).fit(treino)
        erros.append(
            wape_total_por_data(teste["y"], modelo.predict(teste), teste["data"])
        )
    return float(np.mean(erros))


def _ajustar_ridge(df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    linhas = []
    for alpha in [0.1, 1, 3, 10, 30, 100]:
        erro = _avaliar_parametros(df, origens_tuning(), "ridge", {"alpha": alpha})
        linhas.append({"modelo": "ridge", "trial": len(linhas), "wape": erro, "alpha": alpha})
    historico = pd.DataFrame(linhas)
    melhor = historico.loc[historico["wape"].idxmin()]
    return {"alpha": float(melhor["alpha"])}, historico


def _sugerir_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 40),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
    }


def _sugerir_lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 4, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 40),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
    }


def _ajustar_arvore(
    df: pd.DataFrame, nome: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    sugerir = _sugerir_xgboost if nome == "xgboost" else _sugerir_lightgbm
    origens = origens_tuning()

    def objetivo(trial: optuna.Trial) -> float:
        return _avaliar_parametros(df, origens, nome, sugerir(trial))

    estudo = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    estudo.optimize(objetivo, n_trials=20, show_progress_bar=False)
    historico = pd.DataFrame(
        [
            {"modelo": nome, "trial": tentativa.number, "wape": tentativa.value, **tentativa.params}
            for tentativa in estudo.trials
        ]
    )
    melhores = dict(estudo.best_params)
    if nome == "lightgbm":
        melhores["bagging_freq"] = 1
    return melhores, historico


def ajustar_hiperparametros(
    df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Congela hiperparametros usando apenas folds anteriores a junho."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ridge, hist_ridge = _ajustar_ridge(df)
    xgb, hist_xgb = _ajustar_arvore(df, "xgboost")
    lgb, hist_lgb = _ajustar_arvore(df, "lightgbm")
    return (
        {"ridge": ridge, "xgboost": xgb, "lightgbm": lgb},
        pd.concat([hist_ridge, hist_xgb, hist_lgb], ignore_index=True),
    )


def _linhas_previsao(
    teste: pd.DataFrame,
    origem: pd.Timestamp,
    nome: str,
    seed: int,
    previsto: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data": teste["data"].to_numpy(),
            "canal": teste["canal"].to_numpy(),
            "origem": origem,
            "modelo": nome,
            "seed": seed,
            "y": teste["y"].to_numpy(dtype=float),
            "y_hat": previsto,
        }
    )


def prever_origens(
    df: pd.DataFrame,
    origens: list[pd.Timestamp],
    parametros: dict[str, dict[str, Any]],
    modelos_por_origem: dict[str, dict[pd.Timestamp, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retreina por origem e guarda seeds e media do ensemble."""
    partes = []
    tempos = []
    for nome in MODELOS:
        inicio = time.perf_counter()
        for origem in origens:
            treino, teste = criar_fold(df, origem)
            seeds = SEEDS if nome in {"xgboost", "lightgbm"} else [42]
            sementes = []
            for seed in seeds:
                modelo = criar_modelo(nome, parametros.get(nome), random_state=seed).fit(treino)
                if modelos_por_origem is not None and seed == 42:
                    modelos_por_origem.setdefault(nome, {})[pd.Timestamp(origem)] = modelo
                previsto = modelo.predict(teste)
                sementes.append(previsto)
                partes.append(_linhas_previsao(teste, origem, nome, seed, previsto))
            if len(sementes) > 1:
                partes.append(
                    _linhas_previsao(
                        teste, origem, nome, -1, np.mean(np.vstack(sementes), axis=0)
                    )
                )
        tempos.append({"modelo": nome, "segundos": time.perf_counter() - inicio})
    return pd.concat(partes, ignore_index=True), pd.DataFrame(tempos)


def _segmentar(df: pd.DataFrame, segmento: str) -> pd.DataFrame:
    if segmento == "todos":
        return df
    if segmento in {"Site", "App"}:
        return df.loc[df["canal"].eq(segmento)]
    if segmento == "evento/janela de evento":
        return df.loc[df["evento_janela"].eq(1)]
    return df.loc[df["evento_janela"].eq(0)]


def resumir_metricas(
    previsoes: pd.DataFrame, features: pd.DataFrame, conjunto: str
) -> pd.DataFrame:
    """Calcula metricas por granularidade e segmento."""
    eventos = features.assign(
        evento_janela=(
            features["is_evento"].eq(1)
            | features["dias_ate_evento_presente"].le(7)
            | features["dias_ate_evento_promo"].le(7)
        ).astype("int64")
    )[["data", "canal", "evento_janela"]]
    base = selecionar_previsao_ensemble(previsoes).merge(
        eventos, on=["data", "canal"], how="left"
    )
    linhas = []
    segmentos = ["todos", "Site", "App", "evento/janela de evento", "dia normal"]
    for modelo in MODELOS:
        por_modelo = base.loc[base["modelo"].eq(modelo)]
        for segmento in segmentos:
            parte = _segmentar(por_modelo, segmento)
            if parte.empty:
                continue
            for granularidade in ["canal", "total"]:
                avaliar = parte if granularidade == "canal" else agregar_total(parte)
                baixo, alto = bootstrap_ci(avaliar, "wape")
                linha = {
                    "modelo": modelo,
                    "conjunto": conjunto,
                    "granularidade": granularidade,
                    "segmento": segmento,
                    **{nome: funcao(avaliar["y"], avaliar["y_hat"]) for nome, funcao in METRICAS.items()},
                    "wape_ic_low": baixo,
                    "wape_ic_high": alto,
                }
                linhas.append(linha)
    return pd.DataFrame(linhas)


def gerar_contraprova(
    walkforward: pd.DataFrame, holdout: pd.DataFrame
) -> pd.DataFrame:
    pares = list(combinations(MODELOS, 2))
    linhas = []
    for conjunto, previsoes in [("walkforward", walkforward), ("holdout", holdout)]:
        base = agregar_total(selecionar_previsao_ensemble(previsoes))
        for modelo_a, modelo_b in pares:
            resultado = paired_bootstrap_diff(base, modelo_a, modelo_b)
            linhas.append({"conjunto": conjunto, "modelo_a": modelo_a, "modelo_b": modelo_b, **resultado})
    return pd.DataFrame(linhas)


def aplicar_regra_decisao(
    resumo: pd.DataFrame, previsoes: pd.DataFrame, conjunto: str
) -> dict[str, Any]:
    """Aplica a regra de WAPE pareado e simplicidade."""
    metricas = resumo.loc[
        resumo["conjunto"].eq(conjunto)
        & resumo["granularidade"].eq("total")
        & resumo["segmento"].eq("todos"),
        ["modelo", "wape"],
    ]
    candidato = str(metricas.loc[metricas["wape"].idxmin(), "modelo"])
    base = agregar_total(selecionar_previsao_ensemble(previsoes))
    mais_simples = [
        modelo for modelo in MODELOS
        if ORDEM_SIMPLICIDADE[modelo] < ORDEM_SIMPLICIDADE[candidato]
    ]
    empates = []
    for modelo in mais_simples:
        resultado = paired_bootstrap_diff(base, candidato, modelo)
        if resultado["ic_low"] <= 0 <= resultado["ic_high"]:
            empates.append(modelo)
    vencedor = min(
        [candidato, *empates],
        key=lambda modelo: (ORDEM_SIMPLICIDADE[modelo], MODELOS.index(modelo)),
    )
    return {"candidato": candidato, "empates": empates, "vencedor": vencedor}


def gerar_decisao(
    resumo: pd.DataFrame,
    walkforward: pd.DataFrame,
    holdout: pd.DataFrame,
    contraprova: pd.DataFrame,
) -> dict[str, Any]:
    """Registra a decisao no holdout e a checagem walk-forward."""
    decisao_holdout = aplicar_regra_decisao(resumo, holdout, "holdout")
    decisao_walkforward = aplicar_regra_decisao(
        resumo, walkforward, "walkforward"
    )
    vencedores = [decisao_holdout["vencedor"], decisao_walkforward["vencedor"]]
    regra4_vencedor = min(
        vencedores,
        key=lambda modelo: (ORDEM_SIMPLICIDADE[modelo], MODELOS.index(modelo)),
    )
    poder_estatistico = {}
    for conjunto, previsoes in [
        ("holdout", holdout),
        ("walkforward", walkforward),
    ]:
        intervalos = contraprova.loc[contraprova["conjunto"].eq(conjunto)]
        semi_larguras = (intervalos["ic_high"] - intervalos["ic_low"]) / 2
        poder_estatistico[conjunto] = {
            "n_blocos_semanas": int(previsoes["origem"].nunique()),
            "semi_largura_media_ic95_diferenca_wape": float(semi_larguras.mean()),
        }
    return {
        "holdout": decisao_holdout,
        "walkforward": decisao_walkforward,
        "walkforward_concorda": (
            decisao_holdout["vencedor"] == decisao_walkforward["vencedor"]
        ),
        "regra4_vencedor": regra4_vencedor,
        "poder_estatistico": poder_estatistico,
    }


def gerar_variabilidade(
    walkforward: pd.DataFrame, holdout: pd.DataFrame
) -> pd.DataFrame:
    linhas = []
    for conjunto, previsoes in [("walkforward", walkforward), ("holdout", holdout)]:
        arvores = previsoes.loc[
            previsoes["modelo"].isin(["xgboost", "lightgbm"]) & previsoes["seed"].ne(-1)
        ]
        for (modelo, seed), parte in arvores.groupby(["modelo", "seed"]):
            total = agregar_total(parte)
            linhas.append(
                {"modelo": modelo, "conjunto": conjunto, "seed": seed, "wape": wape(total["y"], total["y_hat"])}
            )
    return pd.DataFrame(linhas)


def treinar_finais(
    features: pd.DataFrame, parametros: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Treina modelos de arvore congelados ate 31/05 e os persiste."""
    treino = features.loc[features["data"].le("2026-05-31")]
    finais = {}
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    for nome in ["lightgbm", "xgboost"]:
        modelo = criar_modelo(nome, parametros[nome], random_state=42).fit(treino)
        finais[nome] = modelo
        if nome == "lightgbm":
            modelo.model.booster_.save_model(str(PASTA_MODELOS / "lightgbm_final.txt"))
        else:
            modelo.model.save_model(PASTA_MODELOS / "xgboost_final.json")
    return finais


def _salvar_tabela(df: pd.DataFrame, nome: str) -> None:
    df.to_csv(PASTA_METRICAS / nome, index=False, float_format="%.8f")


def main() -> None:
    """Roda o torneio completo e imprime o holdout total."""
    np.random.seed(42)
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)
    features = build_features(load_daily(by=["canal"]))

    parametros, historico_optuna = ajustar_hiperparametros(features)
    (PASTA_METRICAS / "melhores_hiperparametros.json").write_text(
        json.dumps(parametros, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    walkforward, tempos = prever_origens(features, origens_avaliacao(), parametros)
    modelos_holdout: dict[str, dict[pd.Timestamp, Any]] = {}
    holdout, _ = prever_origens(
        features, origens_holdout(), parametros, modelos_holdout
    )
    walkforward.to_parquet(PASTA_DADOS / "predicoes_walkforward.parquet", index=False)
    holdout.to_parquet(PASTA_DADOS / "predicoes_holdout.parquet", index=False)

    resumo = pd.concat(
        [
            resumir_metricas(walkforward, features, "walkforward"),
            resumir_metricas(holdout, features, "holdout"),
        ],
        ignore_index=True,
    )
    contraprova = gerar_contraprova(walkforward, holdout)
    decisao = gerar_decisao(resumo, walkforward, holdout, contraprova)
    variabilidade = gerar_variabilidade(walkforward, holdout)
    _salvar_tabela(resumo, "resumo_metricas.csv")
    _salvar_tabela(contraprova, "contraprova.csv")
    _salvar_tabela(variabilidade, "variabilidade_seeds.csv")
    _salvar_tabela(tempos, "tempos.csv")
    (PASTA_METRICAS / "decisao.json").write_text(
        json.dumps(decisao, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    finais = treinar_finais(features, parametros)
    from src.explain import gerar_explicacoes
    from src.visual import gerar_visualizacoes

    gerar_explicacoes(features, holdout, modelos_holdout)
    gerar_visualizacoes(
        features,
        walkforward,
        holdout,
        resumo,
        contraprova,
        historico_optuna,
        parametros,
        finais,
    )

    tabela = resumo.loc[
        resumo["conjunto"].eq("holdout")
        & resumo["granularidade"].eq("total")
        & resumo["segmento"].eq("todos"),
        ["modelo", "wape", "bias", "smape", "mae", "wape_ic_low", "wape_ic_high"],
    ].sort_values("wape")
    print("\nResumo holdout — total diario")
    print(tabela.to_string(index=False))
    print("\nContraprova")
    print(contraprova.to_string(index=False))


if __name__ == "__main__":
    main()
