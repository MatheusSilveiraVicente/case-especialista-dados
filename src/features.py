"""Features diarias de calendario, eventos e historico."""

import numpy as np
import pandas as pd

from src.calendario import EVENTOS, feriados_br


CATEGORICAS = ["canal", "dia_semana"]
NUMERICAS = [
    "dia_mes",
    "is_fim_semana",
    "is_feriado",
    "is_vespera_feriado",
    "is_payday_5",
    "is_payday_20",
    "is_novembro",
    "fourier_mes_sin1",
    "fourier_mes_cos1",
    "fourier_mes_sin2",
    "fourier_mes_cos2",
    "is_evento",
    "dias_ate_evento_presente",
    "dias_ate_evento_promo",
    "dias_desde_evento_promo",
    "lag_7",
    "lag_14",
    "media_7_lag7",
    "std_7_lag7",
    "ticket_lag_7",
    "taxa_desconto_media_7_lag7",
    "share_canal_lag7",
]
COLUNAS_HISTORICAS = [
    "lag_7",
    "lag_14",
    "media_7_lag7",
    "std_7_lag7",
    "ticket_lag_7",
    "taxa_desconto_media_7_lag7",
    "share_canal_lag7",
]


def feature_columns() -> dict[str, list[str]]:
    """Lista as features por tipo para consumo dos modelos."""
    return {"categoricas": CATEGORICAS.copy(), "numericas": NUMERICAS.copy()}


def _distancia_evento(
    datas: pd.Series, tipo: str, direcao: str = "proximo"
) -> pd.Series:
    eventos = sorted(
        pd.Timestamp(data) for data, (_, tipo_evento) in EVENTOS.items()
        if tipo_evento == tipo
    )

    def calcular(data: pd.Timestamp) -> int:
        if direcao == "proximo":
            distancias = [(evento - data).days for evento in eventos if evento >= data]
        else:
            distancias = [(data - evento).days for evento in eventos if evento <= data]
        return min(min(distancias), 15) if distancias else 15

    return datas.map(calcular).astype("int64")


def _adicionar_calendario(df: pd.DataFrame) -> None:
    datas = df["data"]
    df["dia_semana"] = datas.dt.dayofweek.astype("int64")
    df["dia_mes"] = datas.dt.day.astype("int64")
    df["is_fim_semana"] = df["dia_semana"].ge(5).astype("int64")

    anos = range(datas.dt.year.min(), datas.dt.year.max() + 2)
    feriados = feriados_br(anos)
    datas_date = datas.dt.date
    df["is_feriado"] = datas_date.isin(feriados).astype("int64")
    df["is_vespera_feriado"] = (datas + pd.Timedelta(days=1)).dt.date.isin(
        feriados
    ).astype("int64")
    # Separados: na EDA so o salario (dia 5) mostrou efeito; o adiantamento (dia 20) nao.
    df["is_payday_5"] = df["dia_mes"].isin([5, 6, 7]).astype("int64")
    df["is_payday_20"] = df["dia_mes"].isin([20, 21, 22]).astype("int64")
    df["is_novembro"] = datas.dt.month.eq(11).astype("int64")

    periodo = datas.dt.days_in_month
    angulo = 2 * np.pi * df["dia_mes"] / periodo
    df["fourier_mes_sin1"] = np.sin(angulo)
    df["fourier_mes_cos1"] = np.cos(angulo)
    df["fourier_mes_sin2"] = np.sin(2 * angulo)
    df["fourier_mes_cos2"] = np.cos(2 * angulo)


def _adicionar_eventos(df: pd.DataFrame) -> None:
    datas_eventos = {pd.Timestamp(data) for data in EVENTOS}
    df["is_evento"] = df["data"].isin(datas_eventos).astype("int64")
    df["dias_ate_evento_presente"] = _distancia_evento(df["data"], "presente")
    df["dias_ate_evento_promo"] = _distancia_evento(df["data"], "promocional")
    df["dias_desde_evento_promo"] = _distancia_evento(
        df["data"], "promocional", direcao="anterior"
    )


def _rolling_defasado(
    df: pd.DataFrame, coluna: str, janela: int, operacao: str
) -> pd.Series:
    return df.groupby("canal", sort=False)[coluna].transform(
        lambda serie: getattr(
            serie.shift(7).rolling(janela, min_periods=janela), operacao
        )()
    )


def _adicionar_historico(df: pd.DataFrame, target: str) -> None:
    por_canal = df.groupby("canal", sort=False)
    df["lag_7"] = por_canal[target].shift(7)
    df["lag_14"] = por_canal[target].shift(14)
    df["media_7_lag7"] = _rolling_defasado(df, target, 7, "mean")
    df["std_7_lag7"] = _rolling_defasado(df, target, 7, "std")
    df["ticket_lag_7"] = por_canal["ticket_medio"].shift(7)
    df["taxa_desconto_media_7_lag7"] = _rolling_defasado(
        df, "taxa_desconto", 7, "mean"
    )

    total_dia = df.groupby("data")["receita"].transform("sum")
    share = df["receita"].div(total_dia.where(total_dia.ne(0)))
    df["share_canal_lag7"] = share.groupby(df["canal"], sort=False).shift(7)


def build_features(daily: pd.DataFrame, target: str = "receita") -> pd.DataFrame:
    """Constroi features sem remover o inicio da serie por causa de NaN."""
    obrigatorias = {
        "data",
        "canal",
        "receita",
        "pedidos",
        "ticket_medio",
        "taxa_desconto",
        target,
    }
    ausentes = obrigatorias.difference(daily.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(ausentes)}")

    resultado = daily.copy()
    resultado["data"] = pd.to_datetime(resultado["data"]).dt.normalize()
    resultado = resultado.sort_values(["canal", "data"]).reset_index(drop=True)
    resultado["y"] = resultado[target]

    _adicionar_calendario(resultado)
    _adicionar_eventos(resultado)
    _adicionar_historico(resultado, target)

    colunas = ["data", "canal", "y", *CATEGORICAS[1:], *NUMERICAS]
    return resultado[colunas]
