import json

import numpy as np
import pandas as pd

import src.train as train_module

from src.models import NaiveSazonal
from src.train import (
    MODELOS,
    aplicar_regra_decisao,
    criar_fold,
    gerar_contraprova_semanal,
    gerar_walkforward_inicio_serie,
    gerar_vies_segmento,
    origens_avaliacao,
    origens_holdout,
    origens_pre_holdout,
    origens_tuning,
)


def _dados() -> pd.DataFrame:
    datas = pd.date_range("2025-11-01", "2026-06-30")
    df = pd.MultiIndex.from_product(
        [datas, ["App", "Site"]], names=["data", "canal"]
    ).to_frame(index=False)
    df["y"] = np.arange(len(df), dtype=float)
    df["lag_7"] = df.groupby("canal")["y"].shift(7)
    return df


def test_folds_respeitam_origem_e_holdout() -> None:
    df = _dados()
    for origem in origens_avaliacao():
        treino, teste = criar_fold(df, origem)
        assert treino["data"].max() <= origem
        assert teste["data"].min() > origem
        assert teste["data"].max() <= origem + pd.Timedelta(days=7)
        assert not treino["data"].isin(pd.date_range("2026-06-01", "2026-06-30")).any()
    assert min(origens_holdout()) == pd.Timestamp("2026-05-31")


def test_folds_tuning_avaliacao_disjuntos_e_cobrem_periodo() -> None:
    df = _dados()
    tuning = origens_tuning()
    avaliacao = origens_avaliacao()
    assert set(tuning).isdisjoint(avaliacao)
    assert sorted([*tuning, *avaliacao]) == origens_pre_holdout()

    datas = []
    for origem in [*tuning, *avaliacao]:
        _, teste = criar_fold(df, origem)
        datas.extend(teste["data"].drop_duplicates().tolist())
    esperado = list(pd.date_range("2025-12-31", "2026-05-31"))
    assert sorted(datas) == esperado


def test_naive_sazonal_reproduz_lag_7() -> None:
    df = _dados().dropna().reset_index(drop=True)
    modelo = NaiveSazonal().fit(df.iloc[:100])
    previsto = modelo.predict(df.iloc[100:120])
    np.testing.assert_array_equal(previsto, df.iloc[100:120]["lag_7"])


def _previsoes_decisao(valor_ridge: float, valor_bases: float) -> pd.DataFrame:
    partes = []
    for modelo in ["naive_sazonal", "media_movel_7", "ridge", "xgboost", "lightgbm"]:
        previsto = valor_ridge if modelo == "ridge" else valor_bases
        seed = -1 if modelo in {"xgboost", "lightgbm"} else 42
        partes.append(
            pd.DataFrame(
                {
                    "data": pd.date_range("2026-01-01", periods=7),
                    "origem": pd.Timestamp("2025-12-31"),
                    "modelo": modelo,
                    "seed": seed,
                    "y": 100.0,
                    "y_hat": previsto,
                }
            )
        )
    return pd.concat(partes, ignore_index=True)


def _resumo_decisao() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "modelo": ["ridge", "naive_sazonal", "media_movel_7", "xgboost", "lightgbm"],
            "conjunto": "teste",
            "granularidade": "total",
            "segmento": "todos",
            "wape": [0.0, 0.1, 0.1, 0.2, 0.2],
        }
    )


def test_decisao_empate_escolhe_mais_simples() -> None:
    decisao = aplicar_regra_decisao(
        _resumo_decisao(), _previsoes_decisao(100.0, 100.0), "teste"
    )
    assert decisao == {
        "candidato": "ridge",
        "empates": ["naive_sazonal", "media_movel_7"],
        "vencedor": "naive_sazonal",
    }


def test_decisao_vitoria_clara_mantem_candidato() -> None:
    decisao = aplicar_regra_decisao(
        _resumo_decisao(), _previsoes_decisao(100.0, 120.0), "teste"
    )
    assert decisao["vencedor"] == "ridge"


def _previsoes_semanais(previsao_lightgbm: list[float] | None = None) -> pd.DataFrame:
    datas = [
        *pd.date_range("2026-01-01", periods=7),
        *pd.date_range("2026-01-08", periods=7),
        *pd.date_range("2026-01-15", periods=5),
    ]
    origens = [pd.Timestamp(data) - pd.Timedelta(days=1) for data in datas]
    origens = [origens[0]] * 7 + [origens[7]] * 7 + [origens[14]] * 5
    partes = []
    for modelo in MODELOS:
        seed = -1 if modelo in {"xgboost", "lightgbm"} else 42
        previsto = (
            previsao_lightgbm
            if modelo == "lightgbm" and previsao_lightgbm is not None
            else [100.0] * 19
        )
        partes.append(
            pd.DataFrame(
                {
                    "data": datas,
                    "canal": "Site",
                    "origem": origens,
                    "modelo": modelo,
                    "seed": seed,
                    "y": 100.0,
                    "y_hat": previsto,
                }
            )
        )
    return pd.concat(partes, ignore_index=True)


def test_contraprova_semanal_descarta_semana_parcial() -> None:
    previsoes = _previsoes_semanais()
    resultado = gerar_contraprova_semanal(previsoes, previsoes)
    assert resultado["n_semanas_cheias"].eq(2).all()


def test_contraprova_semanal_previsoes_iguais_tem_ic_nulo() -> None:
    previsoes = _previsoes_semanais()
    linha = gerar_contraprova_semanal(previsoes, previsoes).query(
        "recorte == 'junho' and modelo_b == 'media_movel_7'"
    ).iloc[0]
    assert linha["diferenca"] == 0
    assert linha["ic_low"] <= 0 <= linha["ic_high"]


def test_contraprova_semanal_compensa_erros_diarios() -> None:
    alternada = [90.0, 110.0, 90.0, 110.0, 90.0, 110.0, 100.0] * 2
    previsoes = _previsoes_semanais([*alternada, *([100.0] * 5)])
    linha = gerar_contraprova_semanal(previsoes, previsoes).query(
        "recorte == 'junho' and modelo_b == 'media_movel_7'"
    ).iloc[0]
    assert linha["wape_semanal_a"] == 0
    lightgbm = previsoes.query("modelo == 'lightgbm'")
    assert np.abs(lightgbm["y"] - lightgbm["y_hat"]).sum() > 0


def _resumo_segmentos() -> pd.DataFrame:
    linhas = []
    for conjunto in ["walkforward", "holdout"]:
        for modelo in MODELOS:
            for segmento in ["evento/janela de evento", "dia normal"]:
                linhas.append(
                    {
                        "modelo": modelo,
                        "conjunto": conjunto,
                        "granularidade": "total",
                        "segmento": segmento,
                        "bias": 0.1,
                    }
                )
    return pd.DataFrame(linhas)


def _previsoes_segmentos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datas = pd.to_datetime(["2026-01-01", "2026-01-02"])
    features = pd.DataFrame(
        {
            "data": datas,
            "canal": "Site",
            "is_evento": [1, 0],
            "dias_ate_evento_presente": [0, 8],
            "dias_ate_evento_promo": [0, 8],
        }
    )
    partes = []
    for modelo in MODELOS:
        seed = -1 if modelo in {"xgboost", "lightgbm"} else 42
        partes.append(
            pd.DataFrame(
                {
                    "data": datas,
                    "canal": "Site",
                    "origem": pd.Timestamp("2025-12-31"),
                    "modelo": modelo,
                    "seed": seed,
                    "y": 100.0,
                    "y_hat": [90.0, 100.0],
                }
            )
        )
    previsoes = pd.concat(partes, ignore_index=True)
    return previsoes.iloc[:0].copy(), previsoes, features


def test_vies_segmento_inclui_todos_os_modelos_nos_recortes_existentes() -> None:
    walkforward, holdout, features = _previsoes_segmentos()
    resultado = gerar_vies_segmento(
        _resumo_segmentos(), walkforward, holdout, features
    )
    existentes = resultado.loc[resultado["conjunto"].ne("dezesseis_semanas")]
    assert len(existentes) == 32
    assert set(existentes["modelo"]) == set(MODELOS)


def test_vies_segmento_inclui_dezesseis_semanas_e_dia_normal_sem_vies() -> None:
    walkforward, holdout, features = _previsoes_segmentos()
    resultado = gerar_vies_segmento(
        _resumo_segmentos(), walkforward, holdout, features
    )
    dezesseis = resultado.loc[resultado["conjunto"].eq("dezesseis_semanas")]
    assert len(resultado) == 48
    assert set(dezesseis["modelo"]) == set(MODELOS)
    assert set(dezesseis["segmento"]) == {
        "evento/janela de evento",
        "dia normal",
    }
    assert dezesseis.loc[
        dezesseis["modelo"].eq("lightgbm")
        & dezesseis["segmento"].eq("dia normal"),
        "vies",
    ].item() == 0


def _previsoes_inicio_sinteticas(
    df: pd.DataFrame, origens: list[pd.Timestamp]
) -> pd.DataFrame:
    partes = []
    for origem in origens:
        _, teste = criar_fold(df, origem)
        for modelo in MODELOS:
            seed = -1 if modelo in {"xgboost", "lightgbm"} else 42
            parte = teste[["data", "canal", "y"]].copy()
            parte["origem"] = origem
            parte["modelo"] = modelo
            parte["seed"] = seed
            parte["y_hat"] = parte["y"] * 0.9
            partes.append(parte)
    return pd.concat(partes, ignore_index=True)


def _executar_inicio_sintetico(monkeypatch, tmp_path) -> pd.DataFrame:
    (tmp_path / "melhores_hiperparametros.json").write_text(
        json.dumps({"ridge": {}, "xgboost": {}, "lightgbm": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(train_module, "PASTA_METRICAS", tmp_path)

    def prever(df, origens, parametros):
        return _previsoes_inicio_sinteticas(df, origens), pd.DataFrame()

    monkeypatch.setattr(train_module, "prever_origens", prever)
    return gerar_walkforward_inicio_serie(_dados())


def test_inicio_exclui_origem_sem_historico_minimo(monkeypatch, tmp_path) -> None:
    resultado = _executar_inicio_sintetico(monkeypatch, tmp_path)
    origens = resultado.loc[resultado["origem"].ne("TOTAL")]
    assert origens["n_dias_treino"].ge(origens["min_treino_dias"]).all()
    assert "2025-11-25" not in set(origens["origem"])


def test_inicio_treina_apenas_antes_da_janela_prevista() -> None:
    df = _dados()
    for origem in pd.date_range("2025-11-04", "2025-12-23", freq="7D"):
        treino, teste = criar_fold(df, origem)
        inicio_previsao = origem + pd.Timedelta(days=1)
        assert treino["data"].lt(inicio_previsao).all()
        assert teste["data"].ge(inicio_previsao).all()


def test_inicio_28_dias_tem_ao_menos_as_origens_de_42(
    monkeypatch, tmp_path
) -> None:
    resultado = _executar_inicio_sintetico(monkeypatch, tmp_path)
    origens = resultado.loc[resultado["origem"].ne("TOTAL")]
    contagens = origens.groupby("min_treino_dias")["origem"].nunique()
    assert contagens[28] >= contagens[42]


def test_inicio_marca_black_friday_apenas_na_janela_correta(
    monkeypatch, tmp_path
) -> None:
    resultado = _executar_inicio_sintetico(monkeypatch, tmp_path)
    origens = resultado.loc[resultado["origem"].ne("TOTAL")].copy()
    datas = pd.to_datetime(origens["origem"])
    esperado = datas.lt("2025-11-28") & datas.add(pd.Timedelta(days=7)).ge(
        "2025-11-28"
    )
    assert origens["contem_black_friday"].eq(esperado).all()
