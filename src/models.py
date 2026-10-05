"""Modelos com interface comum para previsao semanal direta."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.features import feature_columns


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
) -> NaiveSazonal | MediaMovel7 | RidgeSemanal | XGBoostSemanal | LightGBMSemanal:
    """Cria um modelo pelo nome usado nos relatorios."""
    if nome == "naive_sazonal":
        return NaiveSazonal()
    if nome == "media_movel_7":
        return MediaMovel7()
    if nome == "ridge":
        return RidgeSemanal(alpha=float((params or {}).get("alpha", 1.0)))
    if nome == "xgboost":
        return XGBoostSemanal(params=params, random_state=random_state)
    if nome == "lightgbm":
        return LightGBMSemanal(params=params, random_state=random_state)
    raise ValueError(f"Modelo desconhecido: {nome}")
