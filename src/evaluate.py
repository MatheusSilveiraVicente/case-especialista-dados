"""Metricas e bootstrap em blocos para o torneio de modelos."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


Metric = Callable[[np.ndarray, np.ndarray], float]


def _vetores(y: object, y_hat: object) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y, dtype=float), np.asarray(y_hat, dtype=float)


def wape(y: object, y_hat: object) -> float:
    """Erro absoluto ponderado pelo volume real."""
    real, previsto = _vetores(y, y_hat)
    denominador = np.abs(real).sum()
    return float(np.abs(real - previsto).sum() / denominador) if denominador else np.nan


def bias(y: object, y_hat: object) -> float:
    """Vies com sinal: positivo indica superestimacao."""
    real, previsto = _vetores(y, y_hat)
    denominador = real.sum()
    return float((previsto - real).sum() / denominador) if denominador else np.nan


def smape(y: object, y_hat: object) -> float:
    """Erro percentual absoluto simetrico medio."""
    real, previsto = _vetores(y, y_hat)
    denominador = np.abs(real) + np.abs(previsto)
    parcelas = np.divide(
        2 * np.abs(real - previsto),
        denominador,
        out=np.zeros_like(denominador, dtype=float),
        where=denominador != 0,
    )
    return float(parcelas.mean())


def mae(y: object, y_hat: object) -> float:
    """Erro absoluto medio."""
    real, previsto = _vetores(y, y_hat)
    return float(np.abs(real - previsto).mean())


METRICAS: dict[str, Metric] = {
    "wape": wape,
    "bias": bias,
    "smape": smape,
    "mae": mae,
}


def _metrica(metric: str) -> Metric:
    if metric not in METRICAS:
        raise ValueError(f"Metrica desconhecida: {metric}")
    return METRICAS[metric]


def _coluna_bloco(df: pd.DataFrame) -> str:
    return "origem" if "origem" in df.columns else "data"


def _componentes_por_bloco(
    df: pd.DataFrame, metric: str, coluna_prevista: str = "y_hat"
) -> tuple[np.ndarray, np.ndarray]:
    real = df["y"].to_numpy(dtype=float)
    previsto = df[coluna_prevista].to_numpy(dtype=float)
    coluna = _coluna_bloco(df)
    trabalho = pd.DataFrame({"bloco": pd.to_datetime(df[coluna])})
    if metric == "wape":
        trabalho["numerador"] = np.abs(real - previsto)
        trabalho["denominador"] = np.abs(real)
    elif metric == "bias":
        trabalho["numerador"] = previsto - real
        trabalho["denominador"] = real
    elif metric == "mae":
        trabalho["numerador"] = np.abs(real - previsto)
        trabalho["denominador"] = 1.0
    elif metric == "smape":
        denominador = np.abs(real) + np.abs(previsto)
        trabalho["numerador"] = np.divide(
            2 * np.abs(real - previsto),
            denominador,
            out=np.zeros_like(real),
            where=denominador != 0,
        )
        trabalho["denominador"] = 1.0
    else:
        raise ValueError(metric)
    componentes = trabalho.groupby("bloco", sort=True)[["numerador", "denominador"]].sum()
    return componentes["numerador"].to_numpy(), componentes["denominador"].to_numpy()


def _bootstrap_componentes(
    numerador: np.ndarray, denominador: np.ndarray, n: int, seed: int
) -> np.ndarray:
    gerador = np.random.default_rng(seed)
    indices = gerador.integers(0, len(numerador), size=(n, len(numerador)))
    somas_denominador = denominador[indices].sum(axis=1)
    return np.divide(
        numerador[indices].sum(axis=1),
        somas_denominador,
        out=np.full(n, np.nan),
        where=somas_denominador != 0,
    )


def bootstrap_values(
    df: pd.DataFrame,
    metric: str = "wape",
    n: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Retorna metricas reamostrando semanas ou, na ausencia delas, datas."""
    if df.empty:
        raise ValueError("Bootstrap requer ao menos uma linha")
    _metrica(metric)
    numerador, denominador = _componentes_por_bloco(df, metric)
    return _bootstrap_componentes(numerador, denominador, n, seed)


def bootstrap_ci(
    df: pd.DataFrame,
    metric: str = "wape",
    n: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Calcula intervalo percentil de 95% por blocos."""
    valores = bootstrap_values(df, metric=metric, n=n, seed=seed)
    baixo, alto = np.nanpercentile(valores, [2.5, 97.5])
    return float(baixo), float(alto)


def _previsao_media(df: pd.DataFrame, modelo: str) -> pd.DataFrame:
    parte = df.loc[df["modelo"].eq(modelo)].copy()
    if parte.empty:
        raise ValueError(f"Modelo ausente: {modelo}")
    if "seed" in parte and parte["seed"].eq(-1).any():
        parte = parte.loc[parte["seed"].eq(-1)]
    chaves = [coluna for coluna in ["data", "canal", "origem"] if coluna in parte]
    return parte.groupby(chaves, as_index=False).agg(y=("y", "first"), y_hat=("y_hat", "mean"))


def paired_bootstrap_values(
    df: pd.DataFrame,
    modelo_a: str,
    modelo_b: str,
    metric: str = "wape",
    n: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """Retorna diferencas a-b com os mesmos blocos em cada reamostragem."""
    a = _previsao_media(df, modelo_a).rename(columns={"y_hat": "y_hat_a"})
    b = _previsao_media(df, modelo_b).rename(columns={"y_hat": "y_hat_b"})
    chaves = [
        coluna for coluna in ["data", "canal", "origem"]
        if coluna in a and coluna in b
    ]
    pares = a.merge(b[chaves + ["y_hat_b"]], on=chaves, how="inner")
    if pares.empty:
        raise ValueError("Modelos sem observacoes pareadas")

    _metrica(metric)
    numerador_a, denominador_a = _componentes_por_bloco(
        pares, metric, coluna_prevista="y_hat_a"
    )
    numerador_b, denominador_b = _componentes_por_bloco(
        pares, metric, coluna_prevista="y_hat_b"
    )
    gerador = np.random.default_rng(seed)
    indices = gerador.integers(0, len(numerador_a), size=(n, len(numerador_a)))
    valor_a = np.divide(
        numerador_a[indices].sum(axis=1),
        denominador_a[indices].sum(axis=1),
    )
    valor_b = np.divide(
        numerador_b[indices].sum(axis=1),
        denominador_b[indices].sum(axis=1),
    )
    return valor_a - valor_b


def paired_bootstrap_diff(
    df: pd.DataFrame,
    modelo_a: str,
    modelo_b: str,
    metric: str = "wape",
    n: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Resume a diferenca pareada a-b e a chance de a superar b."""
    funcao = _metrica(metric)
    a = _previsao_media(df, modelo_a)
    b = _previsao_media(df, modelo_b)
    chaves = [
        coluna for coluna in ["data", "canal", "origem"]
        if coluna in a and coluna in b
    ]
    pares = a.rename(columns={"y_hat": "y_hat_a"}).merge(
        b[chaves + ["y_hat"]].rename(columns={"y_hat": "y_hat_b"}), on=chaves
    )
    diferenca = funcao(pares["y"], pares["y_hat_a"]) - funcao(
        pares["y"], pares["y_hat_b"]
    )
    valores = paired_bootstrap_values(
        df, modelo_a, modelo_b, metric=metric, n=n, seed=seed
    )
    baixo, alto = np.nanpercentile(valores, [2.5, 97.5])
    return {
        "diferenca": float(diferenca),
        "ic_low": float(baixo),
        "ic_high": float(alto),
        "p_a_menor_b": float(np.mean(valores < 0)),
    }


def agregar_total(df: pd.DataFrame) -> pd.DataFrame:
    """Soma canais por data sem misturar origens, modelos ou seeds."""
    chaves = [
        coluna
        for coluna in ["data", "origem", "modelo", "seed"]
        if coluna in df.columns
    ]
    return df.groupby(chaves, as_index=False).agg(y=("y", "sum"), y_hat=("y_hat", "sum"))


def selecionar_previsao_ensemble(previsoes: pd.DataFrame) -> pd.DataFrame:
    """Seleciona media das arvores e seed fixa dos demais modelos."""
    arvore = previsoes["modelo"].isin(["xgboost", "lightgbm"])
    filtro = (arvore & previsoes["seed"].eq(-1)) | (
        ~arvore & previsoes["seed"].eq(42)
    )
    return previsoes.loc[filtro].copy()


def fva(wape_modelo: float, wape_regua: float) -> float:
    """Ganho sobre a regua: fracao do erro da regua que o modelo elimina."""
    return float(1 - wape_modelo / wape_regua) if wape_regua else np.nan


def totais_semanais(total_diario: pd.DataFrame) -> pd.DataFrame:
    """Soma real e previsto por semana prevista (coluna origem)."""
    return total_diario.groupby("origem", as_index=False)[["y", "y_hat"]].sum()


def rmse(y: object, y_hat: object) -> float:
    """Raiz do erro quadratico medio (pune erros grandes)."""
    real, previsto = _vetores(y, y_hat)
    return float(np.sqrt(np.mean((real - previsto) ** 2)))


def cobertura(y: object, inferior: object, superior: object) -> float:
    """Fracao dos valores reais dentro do intervalo [inferior, superior]."""
    real, baixo = _vetores(y, inferior)
    _, alto = _vetores(y, superior)
    return float(np.mean((real >= baixo) & (real <= alto)))


def pinball(y: object, quantil_previsto: object, alpha: float) -> float:
    """Perda quantilica media para o quantil alpha."""
    real, previsto = _vetores(y, quantil_previsto)
    erro = real - previsto
    return float(np.mean(np.maximum(alpha * erro, (alpha - 1) * erro)))


def wape_total_por_data(
    real: pd.Series, previsto: object, datas: pd.Series
) -> float:
    """Calcula WAPE depois de somar os canais de cada data."""
    quadro = pd.DataFrame(
        {"data": datas.to_numpy(), "y": real.to_numpy(), "y_hat": previsto}
    )
    total = quadro.groupby("data", as_index=False)[["y", "y_hat"]].sum()
    return wape(total["y"], total["y_hat"])
