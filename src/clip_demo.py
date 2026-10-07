"""Mini demo reprodutivel de embeddings CLIP com banners sinteticos."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


MODELO_CLIP = "openai/clip-vit-base-patch32"
MODELO_SIGLIP2 = "google/siglip2-base-patch16-224"
ATRIBUTOS = [
    "fundo",
    "cor_dominante",
    "mensagem",
    "preco_visivel",
    "elemento_produto",
]

# A ordem dos prompts corresponde a CLASSES_ZERO_SHOT.
ROTULOS_ZERO_SHOT = {
    "fundo": [
        "a banner with a light background",
        "a banner with a dark background",
    ],
    "cor_dominante": [
        "a predominantly pink banner",
        "a predominantly green banner",
        "a predominantly blue banner",
        "a predominantly gold banner",
    ],
    "mensagem": [
        "a promotional sale banner with a discount offer",
        "a new product launch banner",
    ],
    "preco_visivel": [
        "a banner with a visible price",
        "a banner without a visible price",
    ],
    "elemento_produto": [
        "a banner showing a perfume bottle product",
        "a banner without a product",
    ],
}
ROTULOS_ZERO_SHOT_PT = {
    "fundo": [
        "um banner com fundo claro",
        "um banner com fundo escuro",
    ],
    "cor_dominante": [
        "um banner predominantemente rosa",
        "um banner predominantemente verde",
        "um banner predominantemente azul",
        "um banner predominantemente dourado",
    ],
    "mensagem": [
        "um banner promocional com oferta de desconto",
        "um banner de lançamento de produto",
    ],
    "preco_visivel": [
        "um banner com preço visível",
        "um banner sem preço visível",
    ],
    "elemento_produto": [
        "um banner mostrando um frasco de perfume",
        "um banner sem produto",
    ],
}
CLASSES_ZERO_SHOT = {
    "fundo": ["claro", "escuro"],
    "cor_dominante": ["rosa", "verde", "azul", "dourado"],
    "mensagem": ["promocional", "lançamento"],
    "preco_visivel": ["sim", "não"],
    "elemento_produto": ["sim", "não"],
}

_MODELO = None
_PROCESSADOR = None
_MODELO_SIGLIP2 = None
_PROCESSADOR_SIGLIP2 = None


def _fonte(tamanho: int) -> ImageFont.ImageFont:
    candidatos = [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for candidato in candidatos:
        try:
            return ImageFont.truetype(candidato, tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


def _coluna_balanceada(
    valores: Sequence[str], n: int, rng: np.random.Generator
) -> np.ndarray:
    repeticoes, resto = divmod(n, len(valores))
    coluna = np.repeat(np.asarray(valores, dtype=object), repeticoes)
    if resto:
        coluna = np.concatenate([coluna, rng.choice(valores, resto, replace=False)])
    return rng.permutation(coluna)


def _combinacoes(n: int, rng: np.random.Generator) -> list[tuple[str, ...]]:
    """Sorteia cada atributo separadamente e preserva marginais balanceadas."""
    niveis = {
        "fundo": ["claro", "escuro"],
        "cor_dominante": ["rosa", "verde", "azul", "dourado"],
        "mensagem": ["promocional", "lançamento"],
        "preco_visivel": ["sim", "não"],
        "elemento_produto": ["sim", "não"],
    }
    colunas = [_coluna_balanceada(niveis[nome], n, rng) for nome in ATRIBUTOS]
    return list(zip(*colunas))


def _cores(fundo: str, cor: str) -> tuple[str, str, str]:
    paleta = {
        "rosa": ("#f7bad2", "#681d43", "#d62d79"),
        "verde": ("#bfe6c5", "#174c35", "#2f9e62"),
        "azul": ("#bddcf4", "#173f6b", "#3488c9"),
        "dourado": ("#ecd69d", "#604e1e", "#c79a28"),
    }
    claro, escuro, destaque = paleta[cor]
    return (claro, "#17202a", destaque) if fundo == "claro" else (escuro, "#ffffff", destaque)


def _desenhar_banner(caminho: Path, atributos: tuple[str, ...], variante: int) -> None:
    fundo, cor, mensagem, preco, produto = atributos
    cor_fundo, cor_texto, destaque = _cores(fundo, cor)
    imagem = Image.new("RGB", (800, 300), cor_fundo)
    desenho = ImageDraw.Draw(imagem)

    deslocamento = 15 + (variante % 5) * 8
    desenho.ellipse((520 + deslocamento, -120, 870 + deslocamento, 230), fill=destaque)
    desenho.rectangle((0, 260, 800, 300), fill=destaque)
    titulo = ["50% OFF", "LEVE 3 PAGUE 2"][variante % 2] if mensagem == "promocional" else ["NOVO", "LANÇAMENTO"][variante % 2]
    subtitulo = "OFERTA ESPECIAL" if mensagem == "promocional" else "UMA NOVA EXPERIÊNCIA"
    desenho.text((42, 50), titulo, font=_fonte(48), fill=cor_texto)
    desenho.text((45, 118), subtitulo, font=_fonte(21), fill=cor_texto)

    if preco == "sim":
        desenho.rounded_rectangle((42, 168, 240, 235), radius=12, fill="#ffffff", outline=destaque, width=3)
        desenho.text((61, 181), "R$ 79,90", font=_fonte(29), fill="#17202a")

    if produto == "sim":
        x = 620 + (variante % 3) * 12
        desenho.rounded_rectangle((x, 88, x + 98, 247), radius=20, fill="#f8f4ec", outline="#2b2b2b", width=4)
        desenho.rectangle((x + 28, 50, x + 70, 91), fill="#d7b65c", outline="#2b2b2b", width=3)
        desenho.rectangle((x + 37, 35, x + 61, 52), fill="#2b2b2b")
        desenho.text((x + 22, 153), "EAU", font=_fonte(16), fill="#2b2b2b")
        desenho.text((x + 14, 177), "PARFUM", font=_fonte(13), fill="#2b2b2b")

    imagem.save(caminho, format="PNG", optimize=False)


def gerar_banners(pasta: str | Path, n: int = 32, seed: int = 42) -> pd.DataFrame:
    """Gera banners 800x300 e seu gabarito de atributos conhecidos."""
    if n < 1:
        raise ValueError("n deve ser positivo")
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    combinacoes = _combinacoes(n, rng)
    linhas = []
    for i, atributos in enumerate(combinacoes):
        token = sha256(f"{seed}:{i}".encode()).hexdigest()[:10]
        nome = f"banner_{token}.png"
        _desenhar_banner(pasta / nome, atributos, int(rng.integers(0, 10_000)))
        linhas.append(dict(zip(ATRIBUTOS, atributos), arquivo=nome))

    gabarito = pd.DataFrame(linhas, columns=["arquivo", *ATRIBUTOS])
    gabarito.to_csv(pasta / "gabarito.csv", index=False, encoding="utf-8")
    return gabarito


def dispositivo() -> str:
    """GPU quando disponivel (ex.: Colab T4), senao CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def carregar_modelo():
    """Carrega CLIP ViT-B/32 e mantem uma unica instancia em memoria."""
    global _MODELO, _PROCESSADOR
    if _MODELO is None or _PROCESSADOR is None:
        from transformers import CLIPModel, CLIPProcessor

        _PROCESSADOR = CLIPProcessor.from_pretrained(MODELO_CLIP)
        _MODELO = CLIPModel.from_pretrained(MODELO_CLIP).to(dispositivo())
        _MODELO.eval()
    return _MODELO, _PROCESSADOR


def carregar_siglip2(local_files_only: bool = False):
    """Carrega o SigLIP 2 multilíngue e mantém uma instância em memória."""
    global _MODELO_SIGLIP2, _PROCESSADOR_SIGLIP2
    if _MODELO_SIGLIP2 is None or _PROCESSADOR_SIGLIP2 is None:
        from transformers import AutoModel, AutoProcessor

        _PROCESSADOR_SIGLIP2 = AutoProcessor.from_pretrained(
            MODELO_SIGLIP2, local_files_only=local_files_only
        )
        _MODELO_SIGLIP2 = AutoModel.from_pretrained(
            MODELO_SIGLIP2, local_files_only=local_files_only
        ).to(dispositivo())
        _MODELO_SIGLIP2.eval()
    return _MODELO_SIGLIP2, _PROCESSADOR_SIGLIP2


def _tensor(saida):
    if hasattr(saida, "pooler_output"):
        return saida.pooler_output
    return saida


def embeddings(imagens: Sequence[str | Path | Image.Image]) -> np.ndarray:
    """Calcula embeddings de imagem L2-normalizados."""
    if not imagens:
        raise ValueError("informe ao menos uma imagem")
    import torch

    modelo, processador = carregar_modelo()
    abertas = []
    for item in imagens:
        if isinstance(item, Image.Image):
            abertas.append(item.convert("RGB"))
        else:
            with Image.open(item) as imagem:
                abertas.append(imagem.convert("RGB"))
    entradas = processador(images=abertas, return_tensors="pt", padding=True).to(modelo.device)
    with torch.no_grad():
        vetores = _tensor(modelo.get_image_features(**entradas))
        vetores = vetores / vetores.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vetores.cpu().numpy()


def _abrir_imagens(
    imagens: Sequence[str | Path | Image.Image],
) -> list[Image.Image]:
    abertas = []
    for item in imagens:
        if isinstance(item, Image.Image):
            abertas.append(item.convert("RGB"))
        else:
            with Image.open(item) as imagem:
                abertas.append(imagem.convert("RGB"))
    return abertas


def zero_shot_siglip2(
    imagens: Sequence[str | Path | Image.Image],
    rotulos: dict[str, list[str]] = ROTULOS_ZERO_SHOT_PT,
    local_files_only: bool = False,
) -> pd.DataFrame:
    """Classifica atributos com SigLIP 2 e prompts em português."""
    if not imagens:
        raise ValueError("informe ao menos uma imagem")
    import torch

    modelo, processador = carregar_siglip2(local_files_only=local_files_only)
    abertas = _abrir_imagens(imagens)
    entradas_imagem = processador(images=abertas, return_tensors="pt").to(modelo.device)
    with torch.no_grad():
        vetores_imagem = _tensor(modelo.get_image_features(**entradas_imagem))
        vetores_imagem = vetores_imagem / vetores_imagem.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

    resultado: dict[str, np.ndarray] = {}
    for atributo, prompts in rotulos.items():
        entradas_texto = processador(
            text=prompts, return_tensors="pt", padding=True
        ).to(modelo.device)
        with torch.no_grad():
            vetores_texto = _tensor(modelo.get_text_features(**entradas_texto))
            vetores_texto = vetores_texto / vetores_texto.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            probabilidades = (vetores_imagem @ vetores_texto.T).softmax(dim=1)
        indices = probabilidades.argmax(dim=1).cpu().numpy()
        classes = CLASSES_ZERO_SHOT.get(atributo, prompts)
        resultado[atributo] = np.asarray(classes, dtype=object)[indices]
        resultado[f"{atributo}_probabilidade"] = (
            probabilidades.max(dim=1).values.cpu().numpy()
        )
    return pd.DataFrame(resultado)


def atributos_simples(
    imagens: Sequence[str | Path | Image.Image],
) -> pd.DataFrame:
    """Extrai medidas explicáveis de cor, luminosidade, texto, preço e frasco."""
    if not imagens:
        raise ValueError("informe ao menos uma imagem")
    linhas = []
    for imagem in _abrir_imagens(imagens):
        rgb = np.asarray(imagem.resize((800, 300)), dtype=np.float32) / 255.0
        cinza = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        dx = np.abs(np.diff(cinza, axis=1, prepend=cinza[:, :1]))
        dy = np.abs(np.diff(cinza, axis=0, prepend=cinza[:1, :]))
        regiao_preco = np.maximum(dx[160:245, 35:250], dy[160:245, 35:250])
        regiao_produto = rgb[30:255, 590:770]
        saturacao = regiao_produto.max(axis=2) - regiao_produto.min(axis=2)
        linhas.append(
            {
                "brilho_medio": float(cinza.mean()),
                "contraste": float(cinza.std()),
                "media_vermelho": float(rgb[:, :, 0].mean()),
                "media_verde": float(rgb[:, :, 1].mean()),
                "media_azul": float(rgb[:, :, 2].mean()),
                "proporcao_bordas_texto": float((np.maximum(dx, dy) > 0.12).mean()),
                "proporcao_caixa_preco": float((regiao_preco > 0.12).mean()),
                "proporcao_frasco": float(
                    ((regiao_produto.mean(axis=2) > 0.78) & (saturacao < 0.10)).mean()
                ),
            }
        )
    return pd.DataFrame(linhas)


def zero_shot(emb: np.ndarray, rotulos: dict[str, list[str]]) -> pd.DataFrame:
    """Classifica atributos comparando embeddings com prompts em ingles."""
    import torch

    modelo, processador = carregar_modelo()
    imagem = torch.as_tensor(emb, dtype=torch.float32, device=modelo.device)
    imagem = imagem / imagem.norm(dim=1, keepdim=True).clamp_min(1e-12)
    resultado: dict[str, np.ndarray] = {}
    for atributo, prompts in rotulos.items():
        if not prompts:
            raise ValueError(f"sem rotulos para {atributo}")
        entradas = processador(text=prompts, return_tensors="pt", padding=True).to(modelo.device)
        with torch.no_grad():
            texto = _tensor(modelo.get_text_features(**entradas))
            texto = texto / texto.norm(dim=1, keepdim=True).clamp_min(1e-12)
            escala = modelo.logit_scale.exp() if hasattr(modelo, "logit_scale") else 100.0
            probabilidades = (escala * imagem @ texto.T).softmax(dim=1)
        indices = probabilidades.argmax(dim=1).cpu().numpy()
        classes = CLASSES_ZERO_SHOT.get(atributo, prompts)
        if len(classes) != len(prompts):
            classes = prompts
        resultado[atributo] = np.asarray(classes, dtype=object)[indices]
        resultado[f"{atributo}_probabilidade"] = probabilidades.max(dim=1).values.cpu().numpy()
    return pd.DataFrame(resultado)


def avaliar_zero_shot(previsto: pd.DataFrame, gabarito: pd.DataFrame) -> pd.DataFrame:
    """Calcula a acuracia de cada atributo presente nas duas tabelas."""
    previsto = previsto.reset_index(drop=True)
    gabarito = gabarito.reset_index(drop=True)
    atributos = [
        coluna
        for coluna in ATRIBUTOS
        if coluna in previsto.columns and coluna in gabarito.columns
    ]
    if not atributos:
        raise ValueError("nenhum atributo comparavel")
    if len(previsto) != len(gabarito):
        raise ValueError("previsto e gabarito devem ter o mesmo numero de linhas")
    linhas = []
    for atributo in atributos:
        valido = previsto[atributo].notna() & gabarito[atributo].notna()
        acuracia = (previsto.loc[valido, atributo].to_numpy() == gabarito.loc[valido, atributo].to_numpy()).mean()
        linhas.append({"atributo": atributo, "acuracia": float(acuracia), "n": int(valido.sum())})
    return pd.DataFrame(linhas)


def clusters(emb: np.ndarray, k: int = 4, seed: int = 42) -> np.ndarray:
    """Agrupa os embeddings com k-means de seed fixa."""
    if k < 1 or k > len(emb):
        raise ValueError("k deve estar entre 1 e o numero de imagens")
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb)


def composicao_clusters(rotulos_cluster: Sequence[int], gabarito: pd.DataFrame) -> pd.DataFrame:
    """Resume quantidade e proporcao de cada atributo dentro dos clusters."""
    if len(rotulos_cluster) != len(gabarito):
        raise ValueError("clusters e gabarito devem ter o mesmo numero de linhas")
    base = gabarito.copy()
    base["cluster"] = np.asarray(rotulos_cluster)
    partes = []
    for atributo in ATRIBUTOS:
        contagem = base.groupby(["cluster", atributo], observed=True).size().rename("quantidade").reset_index()
        contagem["proporcao"] = contagem["quantidade"] / contagem.groupby("cluster")["quantidade"].transform("sum")
        contagem = contagem.rename(columns={atributo: "valor"})
        contagem.insert(1, "atributo", atributo)
        partes.append(contagem)
    return pd.concat(partes, ignore_index=True)


def linear_probe(
    emb: np.ndarray,
    gabarito: pd.DataFrame,
    folds: int = 4,
    seed: int = 42,
    particoes: int = 20,
) -> pd.DataFrame:
    """Avalia regressão logística nos embeddings em várias partições."""
    return _avaliar_classificador(
        emb, gabarito, folds=folds, seed=seed, particoes=particoes
    )


def probe_atributos_simples(
    atributos: pd.DataFrame,
    gabarito: pd.DataFrame,
    folds: int = 4,
    seed: int = 42,
    particoes: int = 20,
) -> pd.DataFrame:
    """Avalia as medidas quantitativas com a mesma validação do CLIP."""
    return _avaliar_classificador(
        atributos.to_numpy(),
        gabarito,
        folds=folds,
        seed=seed,
        particoes=particoes,
        padronizar=True,
    )


def _avaliar_classificador(
    features: np.ndarray,
    gabarito: pd.DataFrame,
    folds: int,
    seed: int,
    particoes: int,
    padronizar: bool = False,
) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if particoes < 1:
        raise ValueError("particoes deve ser positivo")
    if len(features) != len(gabarito):
        raise ValueError("features e gabarito devem ter o mesmo numero de linhas")

    linhas = []
    for atributo in ATRIBUTOS:
        estimador = LogisticRegression(max_iter=2000, C=10, random_state=42)
        if padronizar:
            estimador = make_pipeline(StandardScaler(), estimador)
        acuracias = []
        for deslocamento in range(particoes):
            cv = StratifiedKFold(
                folds, shuffle=True, random_state=seed + deslocamento
            )
            acuracias.append(
                cross_val_score(estimador, features, gabarito[atributo], cv=cv).mean()
            )
        linhas.append(
            {
                "atributo": atributo,
                "acuracia_media": float(np.mean(acuracias)),
                "acuracia_min": float(np.min(acuracias)),
                "acuracia_max": float(np.max(acuracias)),
                "particoes": particoes,
                "folds": folds,
                "exemplos_treino_por_fold": int(len(features) * (folds - 1) / folds),
            }
        )
    return pd.DataFrame(linhas)


def tabela_hibrida(
    zero_shot_resultado: pd.DataFrame,
    probe_clip: pd.DataFrame,
    probe_simples: pd.DataFrame,
) -> pd.DataFrame:
    """Combina as três réguas da demonstração em uma tabela."""
    zero = zero_shot_resultado[["atributo", "acuracia"]].rename(
        columns={"acuracia": "clip_zero_shot"}
    )
    clip = probe_clip.rename(
        columns={
            "acuracia_media": "clip_mais_24_rotulos_media",
            "acuracia_min": "clip_mais_24_rotulos_min",
            "acuracia_max": "clip_mais_24_rotulos_max",
        }
    )[[
        "atributo",
        "clip_mais_24_rotulos_media",
        "clip_mais_24_rotulos_min",
        "clip_mais_24_rotulos_max",
    ]]
    simples = probe_simples.rename(
        columns={
            "acuracia_media": "atributos_simples_media",
            "acuracia_min": "atributos_simples_min",
            "acuracia_max": "atributos_simples_max",
        }
    )[[
        "atributo",
        "atributos_simples_media",
        "atributos_simples_min",
        "atributos_simples_max",
    ]]
    return zero.merge(clip, on="atributo").merge(simples, on="atributo")


def medir_latencia(
    imagens: Sequence[str | Path | Image.Image],
    repeticoes: int = 5,
    aquecimentos: int = 1,
) -> pd.DataFrame:
    """Mede inferência aquecida em lote e um banner por chamada."""
    if not imagens:
        raise ValueError("informe ao menos uma imagem")
    if repeticoes < 1 or aquecimentos < 1:
        raise ValueError("repeticoes e aquecimentos devem ser positivos")
    carregar_modelo()
    for _ in range(aquecimentos):
        embeddings(imagens[: min(4, len(imagens))])

    medicoes: dict[str, list[float]] = {"lote": [], "individual": []}
    for _ in range(repeticoes):
        inicio = perf_counter()
        embeddings(imagens)
        medicoes["lote"].append((perf_counter() - inicio) * 1000 / len(imagens))

        inicio = perf_counter()
        for imagem in imagens:
            embeddings([imagem])
        medicoes["individual"].append(
            (perf_counter() - inicio) * 1000 / len(imagens)
        )

    linhas = []
    for modo, valores in medicoes.items():
        media = float(np.mean(valores))
        minimo = float(np.min(valores))
        maximo = float(np.max(valores))
        linhas.append(
            {
                "modo": modo,
                "repeticoes": repeticoes,
                "banners_por_repeticao": len(imagens),
                "ms_por_banner_media": media,
                "ms_por_banner_min": minimo,
                "ms_por_banner_max": maximo,
                "projecao_10_mil_minutos_media": media * 10_000 / 60_000,
                "projecao_10_mil_minutos_min": minimo * 10_000 / 60_000,
                "projecao_10_mil_minutos_max": maximo * 10_000 / 60_000,
            }
        )
    return pd.DataFrame(linhas)


def vizinhos(emb: np.ndarray, i: int, k: int = 3) -> np.ndarray:
    """Retorna os indices mais similares por cosseno, incluindo a consulta."""
    if not 0 <= i < len(emb):
        raise IndexError("indice de consulta fora do intervalo")
    if not 1 <= k <= len(emb):
        raise ValueError("k deve estar entre 1 e o numero de imagens")
    norma = np.linalg.norm(emb, axis=1, keepdims=True)
    normalizado = emb / np.clip(norma, 1e-12, None)
    similaridade = normalizado @ normalizado[i]
    similaridade[i] = np.inf
    return np.argsort(-similaridade, kind="stable")[:k]


def plotar_mapa_2d(
    emb: np.ndarray,
    rotulos_cluster: Sequence[int],
    gabarito: pd.DataFrame,
    saida: str | Path = "reports/figures/clip_mapa_2d.png",
) -> Path:
    """Salva PCA 2D por cluster, com marcador definido pela mensagem."""
    import matplotlib.pyplot as plt

    pontos = PCA(n_components=2).fit_transform(emb)
    figura, eixo = plt.subplots(figsize=(9, 6))
    marcadores = {"promocional": "o", "lançamento": "^"}
    clusters_arr = np.asarray(rotulos_cluster)
    for mensagem, marcador in marcadores.items():
        mascara = gabarito["mensagem"].eq(mensagem).to_numpy()
        dispersao = eixo.scatter(
            pontos[mascara, 0],
            pontos[mascara, 1],
            c=clusters_arr[mascara],
            cmap="tab10",
            marker=marcador,
            s=75,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label=mensagem,
        )
    eixo.set(title="Mapa 2D dos embeddings CLIP", xlabel="PCA 1", ylabel="PCA 2")
    eixo.legend(title="Mensagem")
    figura.colorbar(dispersao, ax=eixo, label="Cluster")
    figura.tight_layout()
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(saida, dpi=160, bbox_inches="tight")
    plt.close(figura)
    return saida


def plotar_vizinhos(
    imagens: Sequence[str | Path],
    emb: np.ndarray,
    i: int = 0,
    k: int = 3,
    saida: str | Path = "reports/figures/clip_vizinhos.png",
) -> Path:
    """Salva a consulta e k vizinhos visuais mais proximos."""
    import matplotlib.pyplot as plt

    encontrados = vizinhos(emb, i, min(k + 1, len(emb)))
    encontrados = [indice for indice in encontrados if indice != i][:k]
    indices = [i, *encontrados]
    figura, eixos = plt.subplots(len(indices), 1, figsize=(10, 2.5 * len(indices)))
    eixos = np.atleast_1d(eixos)
    consulta = emb[i] / max(np.linalg.norm(emb[i]), 1e-12)
    for posicao, (eixo, indice) in enumerate(zip(eixos, indices)):
        with Image.open(imagens[indice]) as imagem:
            eixo.imshow(imagem.convert("RGB"))
        if posicao == 0:
            titulo = "Consulta"
        else:
            candidato = emb[indice] / max(np.linalg.norm(emb[indice]), 1e-12)
            titulo = f"Vizinho {posicao} — similaridade {float(consulta @ candidato):.3f}"
        eixo.set_title(titulo)
        eixo.axis("off")
    figura.tight_layout()
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(saida, dpi=160, bbox_inches="tight")
    plt.close(figura)
    return saida


def executar_demo(
    pasta: str | Path = "data/banners_sinteticos",
    n: int = 32,
    seed: int = 42,
) -> dict[str, object]:
    """Executa a demo sintetica e grava suas metricas e figuras."""
    np.random.seed(seed)
    pasta = Path(pasta)
    gabarito = gerar_banners(pasta, n=n, seed=seed)
    caminhos = [pasta / nome for nome in gabarito["arquivo"]]
    emb = embeddings(caminhos)
    previsto = zero_shot(emb, ROTULOS_ZERO_SHOT)
    acuracia = avaliar_zero_shot(previsto, gabarito)
    probe_clip = linear_probe(emb, gabarito, seed=seed)
    simples = atributos_simples(caminhos)
    probe_simples = probe_atributos_simples(simples, gabarito, seed=seed)
    hibrido = tabela_hibrida(acuracia, probe_clip, probe_simples)
    rotulos_cluster = clusters(emb, k=4, seed=seed)
    composicao = composicao_clusters(rotulos_cluster, gabarito)

    pasta_metricas = Path("reports/metrics")
    pasta_metricas.mkdir(parents=True, exist_ok=True)
    acuracia.to_csv(pasta_metricas / "clip_zero_shot_acuracia.csv", index=False)
    probe_clip.to_csv(pasta_metricas / "clip_linear_probe.csv", index=False)
    simples.to_csv(pasta_metricas / "clip_atributos_simples.csv", index=False)
    hibrido.to_csv(pasta_metricas / "clip_hibrido.csv", index=False)
    composicao.to_csv(pasta_metricas / "clip_clusters.csv", index=False)
    plotar_mapa_2d(emb, rotulos_cluster, gabarito)
    plotar_vizinhos(caminhos, emb, i=0, k=3)
    return {
        "gabarito": gabarito,
        "embeddings": emb,
        "previsto": previsto,
        "acuracia": acuracia,
        "probe_clip": probe_clip,
        "atributos_simples": simples,
        "probe_simples": probe_simples,
        "hibrido": hibrido,
        "clusters": rotulos_cluster,
        "composicao": composicao,
    }
