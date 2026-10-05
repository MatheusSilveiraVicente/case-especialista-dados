"""Leitura de comportamento de consumo (notebook 02).

Base agregada: os resultados sao associacoes controladas por calendario,
hipoteses para discutir com o negocio, nao efeitos causais.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.data_loader import CATEGORIA_NAO_MAPEADA, clean, complete_grid, load_raw, to_daily

FIG_DIR = Path("reports/figures")
CORES = {"App": "#7a3e9d", "Site": "#2a9d8f", "baixo": "#9FB7C9", "alto": "#B5541C"}
DATAS_PRESENTE = {"Dia das Mães": "2026-05-10", "Dia dos Namorados": "2026-06-12"}

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})


def carregar_grade(path="data/raw/vendas.csv") -> pd.DataFrame:
    return complete_grid(clean(load_raw(path)))


def _salvar(fig, nome: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / nome, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return FIG_DIR / nome


def sensibilidade_desconto(g: pd.DataFrame) -> pd.DataFrame:
    """Variacao % associada a +1 p.p. de desconto, controlando dia da semana e mes.

    Regressao log-linear diaria com erros robustos a autocorrelacao (HAC, 7 dias).
    """
    d = to_daily(g)
    d["mes"] = d["data"].dt.strftime("%Y-%m")
    d["dia_semana"] = d["data"].dt.dayofweek
    alvos = {
        "Pedidos": np.log(d["pedidos"]),
        "Itens por pedido": np.log(d["itens"] / d["pedidos"]),
        "Preço por item": np.log(d["preco_medio"]),
        "Ticket médio": np.log(d["ticket_medio"]),
        "Receita": np.log(d["receita"]),
    }
    linhas = []
    for nome, serie in alvos.items():
        modelo = smf.ols(
            "y ~ taxa_desconto + C(dia_semana) + C(mes)", d.assign(y=serie)
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
        coef = modelo.params["taxa_desconto"] / 100
        baixo, alto = modelo.conf_int().loc["taxa_desconto"] / 100
        linhas.append({
            "indicador": nome,
            "efeito_pct": 100 * (np.exp(coef) - 1),
            "ic_baixo_pct": 100 * (np.exp(baixo) - 1),
            "ic_alto_pct": 100 * (np.exp(alto) - 1),
            "p_valor": modelo.pvalues["taxa_desconto"],
        })
    return pd.DataFrame(linhas)


def fig_sensibilidade(tabela: pd.DataFrame) -> Path:
    t = tabela.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cores = [CORES["alto"] if v >= 0 else "#2F5D8A" for v in t["efeito_pct"]]
    ax.barh(t["indicador"], t["efeito_pct"], color=cores,
            xerr=[t["efeito_pct"] - t["ic_baixo_pct"], t["ic_alto_pct"] - t["efeito_pct"]], capsize=3)
    ax.axvline(0, color="grey", lw=1)
    for y, (v, baixo, alto) in enumerate(zip(t["efeito_pct"], t["ic_baixo_pct"], t["ic_alto_pct"])):
        x = alto + 0.12 if v >= 0 else baixo - 0.12
        ax.text(x, y, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlim(t["ic_baixo_pct"].min() - 0.8, t["ic_alto_pct"].max() + 0.8)
    ax.set_xlabel("Variação associada a +1 p.p. de desconto (%)")
    ax.set_title("Desconto traz mais pedidos, mas de menor valor")
    return _salvar(fig, "consumo_01_sensibilidade_desconto.png")


def resposta_semanal_desconto(g: pd.DataFrame) -> pd.DataFrame:
    """Receita semanal vs desconto da propria semana e das duas anteriores (antecipacao de demanda)."""
    d = to_daily(g).set_index("data")
    w = d.resample("W-SUN")[["receita", "desconto", "receita_bruta"]].sum()
    w["taxa"] = w["desconto"] / w["receita_bruta"]
    w["taxa_sem_1"] = w["taxa"].shift(1)
    w["taxa_sem_2"] = w["taxa"].shift(2)
    modelo = smf.ols("np.log(receita) ~ taxa + taxa_sem_1 + taxa_sem_2", w.dropna()).fit()
    return pd.DataFrame({"coeficiente": modelo.params, "p_valor": modelo.pvalues}).drop("Intercept")


def mix_por_desconto(g: pd.DataFrame) -> pd.DataFrame:
    """Participacao media de cada categoria em dias de desconto baixo, medio e alto (tercis)."""
    gc = g[g["categoria"] != CATEGORIA_NAO_MAPEADA]
    dc = to_daily(gc, by=["categoria"])
    dc["participacao"] = dc["receita"] / dc.groupby("data")["receita"].transform("sum")
    taxa_dia = to_daily(g).set_index("data")["taxa_desconto"]
    dc["faixa"] = pd.qcut(dc["data"].map(taxa_dia), 3, labels=["baixo", "médio", "alto"])
    return dc.pivot_table(index="categoria", columns="faixa", values="participacao", observed=True) * 100


def fig_mix(mix: pd.DataFrame) -> Path:
    m = mix.sort_values("alto")
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(y + 0.2, m["baixo"], height=0.4, color=CORES["baixo"], label="Dias de desconto baixo")
    ax.barh(y - 0.2, m["alto"], height=0.4, color=CORES["alto"], label="Dias de desconto alto")
    ax.set_yticks(y, [c.title() for c in m.index])
    ax.set_xlabel("% da receita do dia")
    ax.set_title("Em promoção, o consumidor vai para o perfume")
    ax.legend(loc="lower right")
    return _salvar(fig, "consumo_02_mix_promocao.png")


def desconto_por_categoria(g: pd.DataFrame) -> pd.DataFrame:
    gc = g[g["categoria"] != CATEGORIA_NAO_MAPEADA]
    c = gc.groupby("categoria")[["desconto", "receita_bruta", "receita", "itens"]].sum()
    return pd.DataFrame({
        "taxa_desconto": c["desconto"] / c["receita_bruta"],
        "preco_cheio_por_item": c["receita_bruta"] / c["itens"],
        "preco_pago_por_item": c["receita"] / c["itens"],
        "participacao_receita": c["receita"] / c["receita"].sum(),
    }).sort_values("taxa_desconto")


def preco_por_item_mensal(g: pd.DataFrame) -> pd.DataFrame:
    """Preco pago e preco cheio por item: separa efeito desconto de efeito mix."""
    d = to_daily(g)
    m = d.groupby(d["data"].dt.strftime("%Y-%m"))[["receita", "receita_bruta", "itens"]].sum()
    return pd.DataFrame({
        "preco_pago_por_item": m["receita"] / m["itens"],
        "preco_cheio_por_item": m["receita_bruta"] / m["itens"],
    })


def curva_presente(g: pd.DataFrame, dias: int = 21) -> pd.DataFrame:
    """Participacao de perfumaria feminina, masculina e gifts nos dias que antecedem cada data."""
    gc = g[g["categoria"] != CATEGORIA_NAO_MAPEADA]
    dc = to_daily(gc, by=["categoria"])
    dc["participacao"] = dc["receita"] / dc.groupby("data")["receita"].transform("sum") * 100
    partes = []
    for nome, data in DATAS_PRESENTE.items():
        evento = pd.Timestamp(data)
        janela = dc[dc["data"].between(evento - pd.Timedelta(days=dias), evento)].copy()
        janela["dias_antes"] = (evento - janela["data"]).dt.days
        janela["data_comemorativa"] = nome
        partes.append(janela)
    curva = pd.concat(partes)
    return curva[curva["categoria"].isin(["PERFUMARIA FEMININA", "PERFUMARIA MASCULINA", "GIFTS"])]


def fig_presente(curva: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    cores = {"PERFUMARIA FEMININA": "#B5541C", "PERFUMARIA MASCULINA": "#2F5D8A", "GIFTS": "#8A8F95"}
    for ax, (nome, parte) in zip(axes, curva.groupby("data_comemorativa", sort=False)):
        for categoria, serie in parte.groupby("categoria"):
            serie = serie.sort_values("dias_antes", ascending=False)
            ax.plot(serie["dias_antes"], serie["participacao"], marker="o", ms=3,
                    color=cores[categoria], label=categoria.title())
        ax.invert_xaxis()
        ax.set_title(nome)
        ax.set_xlabel("Dias antes da data")
    axes[0].set_ylabel("% da receita do dia")
    axes[0].legend(fontsize=8)
    fig.suptitle("O presente tem destinatário: feminina no Dia das Mães, masculina nos Namorados", y=1.04)
    return _salvar(fig, "consumo_03_presente.png")


def salario_por_canal(g: pd.DataFrame) -> pd.DataFrame:
    """Indice da receita nos dias 5-7 vs demais, controlando mes e dia da semana, sem novembro."""
    d = to_daily(g, by=["canal"])
    d = d[d["data"].dt.month != 11].copy()
    d["mes"] = d["data"].dt.to_period("M")
    d["dia_semana"] = d["data"].dt.dayofweek
    d["indice"] = d["receita"] / d.groupby(["canal", "mes", "dia_semana"])["receita"].transform("mean")
    d["dias_5_a_7"] = d["data"].dt.day.isin([5, 6, 7])
    return d.groupby(["canal", "dias_5_a_7"])["indice"].mean().unstack()


def gerar_todas(path="data/raw/vendas.csv") -> dict:
    g = carregar_grade(path)
    resultados = {
        "sensibilidade": sensibilidade_desconto(g),
        "semanal": resposta_semanal_desconto(g),
        "mix": mix_por_desconto(g),
        "categorias": desconto_por_categoria(g),
        "preco_item": preco_por_item_mensal(g),
        "presente": curva_presente(g),
        "salario": salario_por_canal(g),
    }
    fig_sensibilidade(resultados["sensibilidade"])
    fig_mix(resultados["mix"])
    fig_presente(resultados["presente"])
    return resultados
