"""Modelos com interface comum para previsao semanal direta."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.features import feature_columns


LOGGER = logging.getLogger(__name__)
EXOGENAS_ESTATISTICAS = [
    "is_feriado",
    "is_vespera_feriado",
    "is_novembro",
    "is_evento",
]


def _exogenas_estatisticas(df: pd.DataFrame) -> pd.DataFrame:
    x = df[EXOGENAS_ESTATISTICAS].astype(float).copy()
    x["pre_presente"] = df["dias_ate_evento_presente"].le(7).astype(float)
    x["pre_promocao"] = df["dias_ate_evento_promo"].le(7).astype(float)
    return x


class _EstatisticoPorCanal:
    nome = "estatistico"

    def fit(self, train_df: pd.DataFrame) -> "_EstatisticoPorCanal":
        self.ajustes_: dict[str, Any] = {}
        self.avisos_convergencia_: list[str] = []
        for canal, parte in train_df.sort_values("data").groupby("canal"):
            with warnings.catch_warnings(record=True) as capturados:
                warnings.simplefilter("always", ConvergenceWarning)
                self.ajustes_[str(canal)] = self._ajustar(parte)
            for aviso in capturados:
                if issubclass(aviso.category, ConvergenceWarning):
                    mensagem = f"{self.nome}/{canal}: {aviso.message}"
                    self.avisos_convergencia_.append(mensagem)
                    LOGGER.warning(mensagem)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        previsto = pd.Series(index=test_df.index, dtype=float)
        for canal, parte in test_df.groupby("canal"):
            ordenado = parte.sort_values("data")
            valores = self._prever(self.ajustes_[str(canal)], ordenado)
            previsto.loc[ordenado.index] = np.exp(valores)
        if previsto.isna().any():
            raise ValueError("Ha linhas sem previsao estatistica")
        return np.clip(previsto.loc[test_df.index].to_numpy(), 0, None)

    def _ajustar(self, treino: pd.DataFrame) -> Any:
        raise NotImplementedError

    def _prever(self, ajuste: Any, teste: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class ETSLogSemanal(_EstatisticoPorCanal):
    nome = "ets_log"

    def _ajustar(self, treino: pd.DataFrame) -> Any:
        alvo = np.log(treino["y"].clip(lower=1).to_numpy())
        return ExponentialSmoothing(
            alvo, trend=None, seasonal="add", seasonal_periods=7
        ).fit()

    def _prever(self, ajuste: Any, teste: pd.DataFrame) -> np.ndarray:
        return np.asarray(ajuste.forecast(len(teste)))


class SARIMALogSemanal(_EstatisticoPorCanal):
    nome = "sarima_log"

    def _ajustar(self, treino: pd.DataFrame) -> Any:
        alvo = np.log(treino["y"].clip(lower=1).to_numpy())
        return SARIMAX(
            alvo, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)
        ).fit(disp=False)

    def _prever(self, ajuste: Any, teste: pd.DataFrame) -> np.ndarray:
        return np.asarray(ajuste.forecast(len(teste)))


class SARIMAXLogSemanal(_EstatisticoPorCanal):
    nome = "sarimax_log"

    def _ajustar(self, treino: pd.DataFrame) -> Any:
        alvo = np.log(treino["y"].clip(lower=1).to_numpy())
        return SARIMAX(
            alvo,
            exog=_exogenas_estatisticas(treino).to_numpy(),
            order=(1, 0, 1),
            seasonal_order=(0, 1, 1, 7),
        ).fit(disp=False)

    def _prever(self, ajuste: Any, teste: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            ajuste.forecast(
                len(teste), exog=_exogenas_estatisticas(teste).to_numpy()
            )
        )


class NaiveSazonal:
    nome = "naive_sazonal"

    def fit(self, train_df: pd.DataFrame) -> "NaiveSazonal":
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return test_df["lag_7"].to_numpy(dtype=float)


class MediaMovel7:
    nome = "media_movel_7"

    def fit(self, train_df: pd.DataFrame) -> "MediaMovel7":
        ordenado = train_df.sort_values("data")
        self.medias_ = ordenado.groupby("canal")["y"].apply(
            lambda serie: float(serie.tail(7).mean())
        )
        self.media_global_ = float(ordenado["y"].tail(14).mean())
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return (
            test_df["canal"]
            .map(self.medias_)
            .fillna(self.media_global_)
            .to_numpy(dtype=float)
        )


class RidgeSemanal:
    nome = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        colunas = feature_columns()
        self.categoricas = colunas["categoricas"]
        self.numericas = colunas["numericas"]
        numerico = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        categorico = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preparo = ColumnTransformer(
            [("categoricas", categorico, self.categoricas), ("numericas", numerico, self.numericas)]
        )
        self.pipeline = Pipeline(
            [("preparo", preparo), ("modelo", Ridge(alpha=alpha))]
        )

    @property
    def features(self) -> list[str]:
        return [*self.categoricas, *self.numericas]

    def fit(self, train_df: pd.DataFrame) -> "RidgeSemanal":
        alvo = np.log1p(np.clip(train_df["y"].to_numpy(dtype=float), 0, None))
        self.pipeline.fit(train_df[self.features], alvo)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.clip(np.expm1(self.pipeline.predict(test_df[self.features])), 0, None)


class _ArvoreCategorias:
    nome = "arvore"

    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42) -> None:
        colunas = feature_columns()
        self.categoricas = colunas["categoricas"]
        self.numericas = colunas["numericas"]
        self.params = dict(params or {})
        self.random_state = random_state

    @property
    def features(self) -> list[str]:
        return [*self.categoricas, *self.numericas]

    def _ajustar_categorias(self, df: pd.DataFrame, treino: bool) -> pd.DataFrame:
        x = df[self.features].copy()
        for coluna in self.categoricas:
            if treino:
                categorias = sorted(x[coluna].dropna().unique().tolist())
                self.categorias_[coluna] = categorias
            x[coluna] = pd.Categorical(x[coluna], categories=self.categorias_[coluna])
        return x

    def fit(self, train_df: pd.DataFrame) -> "_ArvoreCategorias":
        self.categorias_: dict[str, list[Any]] = {}
        x = self._ajustar_categorias(train_df, treino=True)
        alvo = np.log1p(np.clip(train_df["y"].to_numpy(dtype=float), 0, None))
        self.model.fit(x, alvo)
        return self

    def predict(self, test_df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        x = self._ajustar_categorias(test_df, treino=False)
        return np.clip(np.expm1(self.model.predict(x, **kwargs)), 0, None)


class XGBoostSemanal(_ArvoreCategorias):
    nome = "xgboost"

    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42) -> None:
        super().__init__(params=params, random_state=random_state)
        configuracao = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": random_state,
            "n_jobs": 1,
            **self.params,
        }
        self.model = XGBRegressor(**configuracao)


class LightGBMSemanal(_ArvoreCategorias):
    nome = "lightgbm"

    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42) -> None:
        super().__init__(params=params, random_state=random_state)
        configuracao = {
            "objective": "regression",
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1,
            "bagging_seed": random_state,
            "feature_fraction_seed": random_state,
            "data_random_seed": random_state,
            **self.params,
        }
        self.model = LGBMRegressor(**configuracao)


def criar_modelo(
    nome: str, params: dict[str, Any] | None = None, random_state: int = 42
) -> (
    NaiveSazonal
    | MediaMovel7
    | ETSLogSemanal
    | SARIMALogSemanal
    | SARIMAXLogSemanal
    | RidgeSemanal
    | XGBoostSemanal
    | LightGBMSemanal
):
    """Cria um modelo pelo nome usado nos relatorios."""
    if nome == "naive_sazonal":
        return NaiveSazonal()
    if nome == "media_movel_7":
        return MediaMovel7()
    if nome == "ets_log":
        return ETSLogSemanal()
    if nome == "sarima_log":
        return SARIMALogSemanal()
    if nome == "sarimax_log":
        return SARIMAXLogSemanal()
    if nome == "ridge":
        return RidgeSemanal(alpha=float((params or {}).get("alpha", 1.0)))
    if nome == "xgboost":
        return XGBoostSemanal(params=params, random_state=random_state)
    if nome == "lightgbm":
        return LightGBMSemanal(params=params, random_state=random_state)
    raise ValueError(f"Modelo desconhecido: {nome}")
