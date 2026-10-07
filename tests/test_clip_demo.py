from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.clip_demo import (
    ATRIBUTOS,
    _combinacoes,
    atributos_simples,
    avaliar_zero_shot,
    gerar_banners,
    vizinhos,
)


def test_gerar_banners_deterministico_e_completo(tmp_path: Path) -> None:
    pasta_a = tmp_path / "a"
    pasta_b = tmp_path / "b"

    gabarito_a = gerar_banners(pasta_a, n=32, seed=42)
    gabarito_b = gerar_banners(pasta_b, n=32, seed=42)

    pd.testing.assert_frame_equal(gabarito_a, gabarito_b)
    for nome in gabarito_a["arquivo"]:
        assert (pasta_a / nome).read_bytes() == (pasta_b / nome).read_bytes()
    with Image.open(pasta_a / gabarito_a.loc[0, "arquivo"]) as imagem:
        assert imagem.size == (800, 300)
    assert set(gabarito_a["fundo"]) == {"claro", "escuro"}
    assert set(gabarito_a["cor_dominante"]) == {"rosa", "verde", "azul", "dourado"}
    assert set(gabarito_a["mensagem"]) == {"promocional", "lançamento"}
    assert set(gabarito_a["preco_visivel"]) == {"sim", "não"}
    assert set(gabarito_a["elemento_produto"]) == {"sim", "não"}
    assert list(gabarito_a.columns) == ["arquivo", *ATRIBUTOS]
    assert gabarito_a["arquivo"].str.match(r"banner_[0-9a-f]{10}\.png").all()
    assert (pasta_a / "gabarito.csv").is_file()


def test_gabarito_balanceado_e_atributos_independentes() -> None:
    combinacoes = _combinacoes(4_000, np.random.default_rng(42))
    gabarito = pd.DataFrame(combinacoes, columns=ATRIBUTOS)

    assert gabarito["cor_dominante"].value_counts().nunique() == 1
    for atributo in set(ATRIBUTOS) - {"cor_dominante"}:
        assert gabarito[atributo].value_counts().nunique() == 1

    indicadores = pd.get_dummies(gabarito, prefix_sep="=").astype(float)
    correlacoes = indicadores.corr()
    maior = 0.0
    for coluna_a in indicadores:
        atributo_a = coluna_a.split("=")[0]
        for coluna_b in indicadores:
            atributo_b = coluna_b.split("=")[0]
            if atributo_a != atributo_b:
                maior = max(maior, abs(correlacoes.loc[coluna_a, coluna_b]))
    assert maior < 0.06


def test_atributos_simples_separam_fundo_claro_e_escuro() -> None:
    imagens = [Image.new("RGB", (800, 300), "white"), Image.new("RGB", (800, 300), "black")]

    resultado = atributos_simples(imagens)

    assert resultado.loc[0, "brilho_medio"] > resultado.loc[1, "brilho_medio"]
    assert resultado.loc[0, "brilho_medio"] - resultado.loc[1, "brilho_medio"] > 0.9


def test_atributos_simples_detectam_preco_e_frasco(tmp_path: Path) -> None:
    gabarito = gerar_banners(tmp_path, n=32, seed=42)
    caminhos = [tmp_path / nome for nome in gabarito["arquivo"]]

    resultado = atributos_simples(caminhos).join(gabarito[ATRIBUTOS])

    preco = resultado.groupby("preco_visivel")["proporcao_caixa_preco"].mean()
    frasco = resultado.groupby("elemento_produto")["proporcao_frasco"].mean()
    assert preco["sim"] > preco["não"]
    assert frasco["sim"] > frasco["não"]


def test_avaliar_zero_shot_calcula_acuracia_manual() -> None:
    gabarito = pd.DataFrame(
        {
            "fundo": ["claro", "escuro", "claro", "escuro"],
            "mensagem": ["promocional", "lançamento", "promocional", "lançamento"],
        }
    )
    previsto = pd.DataFrame(
        {
            "fundo": ["claro", "claro", "claro", "escuro"],
            "fundo_probabilidade": [0.9, 0.6, 0.8, 0.7],
            "mensagem": ["promocional", "lançamento", "lançamento", "promocional"],
            "mensagem_probabilidade": [0.9, 0.8, 0.7, 0.6],
        }
    )

    resultado = avaliar_zero_shot(previsto, gabarito).set_index("atributo")

    assert resultado.loc["fundo", "acuracia"] == 0.75
    assert resultado.loc["mensagem", "acuracia"] == 0.5
    assert (resultado["n"] == 4).all()


def test_vizinhos_inclui_proprio_item_como_mais_similar() -> None:
    emb = np.array([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]])

    encontrados = vizinhos(emb, i=1, k=3)

    assert encontrados[0] == 1
    assert set(encontrados) == {0, 1, 2}


def test_linear_probe_separa_atributo_linear() -> None:
    from src.clip_demo import ATRIBUTOS, linear_probe

    rng = np.random.default_rng(0)
    n = 32
    gabarito = pd.DataFrame({a: np.tile(["x", "y"], n // 2) for a in ATRIBUTOS})
    emb = rng.normal(size=(n, 8))
    emb[:, 0] += np.where(gabarito[ATRIBUTOS[0]] == "x", 5.0, -5.0)

    resultado = linear_probe(emb, gabarito, particoes=3).set_index("atributo")

    assert resultado.loc[ATRIBUTOS[0], "acuracia_media"] == 1.0
    assert resultado.loc[ATRIBUTOS[0], "acuracia_min"] == 1.0
    assert resultado.loc[ATRIBUTOS[0], "acuracia_max"] == 1.0
    assert set(resultado.index) == set(ATRIBUTOS)


def test_embeddings_usa_encoder_mockado(monkeypatch) -> None:
    import torch
    import src.clip_demo as clip_demo

    class Entradas(dict):
        def to(self, _device):
            return self

    class Processador:
        def __call__(self, images, **_kwargs):
            return Entradas(pixel_values=torch.ones((len(images), 2)))

    class Modelo:
        device = "cpu"

        def get_image_features(self, pixel_values):
            return pixel_values

    monkeypatch.setattr(clip_demo, "carregar_modelo", lambda: (Modelo(), Processador()))

    vetores = clip_demo.embeddings([Image.new("RGB", (10, 10), "white")])

    assert vetores.shape == (1, 2)
    np.testing.assert_allclose(np.linalg.norm(vetores, axis=1), 1.0)
