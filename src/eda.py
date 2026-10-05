"""Analises e figuras da EDA (notebook 01)."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import CATEGORIA_NAO_MAPEADA, clean, complete_grid, load_raw, to_daily

FIG_DIR = Path("reports/figures")
DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
CORES = {"App": "#7a3e9d", "Site": "#2a9d8f", "Total": "#264653"}
EVENTOS_PLOT = {
    "2025-11-11": "11.11",
    "2025-11-28": "Black Friday",
    "2025-12-25": "Natal",
    "2026-05-09": "Véspera\nDia das Mães",
    "2026-06-12": "Namorados",
}

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})


def carregar_grade(path="data/raw/vendas.csv") -> pd.DataFrame:
    """Grade horaria completa (hora x canal x categoria) com colunas auxiliares."""
    g = complete_grid(clean(load_raw(path)))
    g["dia_semana"] = g["dt_hr"].dt.dayofweek
    g["mes"] = g["dt_hr"].dt.to_period("M")
    return g


def _salvar(fig, nome):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / nome, bbox_inches="tight", dpi=220)
    return FIG_DIR / nome


def _mi(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"R$ {v / 1e6:.0f} mi"))


def kpis_gerais(g: pd.DataFrame) -> pd.Series:
    t = g[["receita", "pedidos", "itens", "desconto", "receita_bruta", "estorno"]].sum()
    return pd.Series({
        "receita_mi": t.receita / 1e6,
        "pedidos_mi": t.pedidos / 1e6,
        "ticket_medio": t.receita / t.pedidos,
        "preco_medio": t.receita / t.itens,
        "itens_por_pedido": t.itens / t.pedidos,
        "taxa_desconto": t.desconto / t.receita_bruta,
        "estorno_pct_receita": -t.estorno / t.receita,
    })


def fig_serie_diaria(g):
    d = to_daily(g)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d["data"], d["receita"], color=CORES["Total"], lw=1, alpha=0.6, label="Receita diária")
    ax.plot(d["data"], d["receita"].rolling(7, center=True).mean(), color="#e76f51", lw=2, label="Média 7 dias")
    for dt, nome in EVENTOS_PLOT.items():
        y = d.loc[d["data"] == dt, "receita"].iloc[0]
        ax.annotate(nome, (pd.Timestamp(dt), y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, arrowprops=dict(arrowstyle="-", alpha=0.4))
    _mi(ax)
    ax.set_ylim(0, d["receita"].max() * 1.15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
    ax.set_title("Receita diária: novembro é outro negócio; depois, sem tendência e com picos de data")
    ax.legend(loc="upper right")
    return _salvar(fig, "eda_01_serie_diaria.png")


def indice_dia_semana(g, excluir_novembro=True):
    d = to_daily(g, by=["canal"])
    if excluir_novembro:
        d = d[d["data"].dt.month != 11]
    d["dia_semana"] = d["data"].dt.dayofweek
    idx = d.groupby(["canal", "dia_semana"])["receita"].mean()
    idx = idx / idx.groupby("canal").transform("mean")
    return idx.unstack("canal")


def fig_dia_semana(g):
    idx = indice_dia_semana(g)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(7)
    for i, canal in enumerate(["App", "Site"]):
        ax.bar(x + (i - 0.5) * 0.38, idx[canal], width=0.38, color=CORES[canal], label=canal)
    ax.axhline(1, color="grey", lw=1, ls="--")
    ax.set_xticks(x, DIAS)
    ax.set_ylabel("Índice (1 = média do canal)")
    ax.set_title("Potencial por dia da semana (dez-jun): domingo é o dia fraco")
    ax.legend()
    return _salvar(fig, "eda_02_dia_semana.png")


def curva_intradia(g):
    h = g.groupby(["canal", "hora"])["receita"].sum()
    return (h / h.groupby("canal").transform("sum")).unstack("canal")


def fig_intradia(g):
    c = curva_intradia(g)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    for canal in ["App", "Site"]:
        axes[0].plot(c.index, c[canal] * 100, marker="o", ms=3, color=CORES[canal], label=canal)
    axes[0].set_xlabel("Hora do dia")
    axes[0].set_ylabel("% da receita do canal")
    axes[0].set_title("Curva de vendas ao longo do dia")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend()

    sub = g[g["mes"].dt.month != 11]
    hm = sub.groupby(["dia_semana", "hora"])["receita"].sum().unstack("hora")
    hm = hm / hm.values.sum() * 100
    axes[1].grid(False)
    im = axes[1].imshow(hm.values, aspect="auto", cmap="magma_r")
    axes[1].set_yticks(range(7), DIAS)
    axes[1].set_xticks(range(0, 24, 3), range(0, 24, 3))
    axes[1].set_xlabel("Hora do dia")
    axes[1].set_title("Mapa de calor dia × hora (% da receita, dez-jun)")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    return _salvar(fig, "eda_03_intradia.png")


def mensal(g):
    m = g.groupby("mes")[["receita", "pedidos", "itens", "desconto", "receita_bruta", "estorno"]].sum()
    m["ticket_medio"] = m.receita / m.pedidos
    m["preco_medio"] = m.receita / m.itens
    m["itens_por_pedido"] = m.itens / m.pedidos
    m["taxa_desconto"] = m.desconto / m.receita_bruta
    m["estorno_pct"] = -m.estorno / m.receita
    return m


def fig_preco_desconto(g):
    m = mensal(g)
    x = m.index.strftime("%b/%y")
    fig, ax1 = plt.subplots(figsize=(9, 3.8))
    ax1.plot(x, m["ticket_medio"], marker="o", color=CORES["Total"], lw=2, label="Ticket médio (R$)")
    ax1.plot(x, m["preco_medio"], marker="s", color=CORES["Site"], lw=2, label="Preço médio por item (R$)")
    ax1.set_ylabel("R$")
    ax2 = ax1.twinx()
    ax2.bar(x, m["taxa_desconto"] * 100, color="#e76f51", alpha=0.25, label="Desconto (% do bruto)")
    ax2.set_ylabel("% desconto")
    ax2.spines["right"].set_visible(True)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax1.set_title("Menos desconto, ticket maior: de R$ 59 em nov para R$ 93 em jun")
    return _salvar(fig, "eda_04_preco_desconto.png")


def indice_categoria_dia(g):
    sub = g[(g["categoria"] != CATEGORIA_NAO_MAPEADA) & (g["mes"].dt.month != 11)]
    t = sub.groupby(["categoria", "dia_semana"])["receita"].sum().unstack()
    return t.div(t.mean(axis=1), axis=0)


def fig_categoria_dia(g):
    t = indice_categoria_dia(g)
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(False)
    im = ax.imshow(t.values, cmap="RdBu", vmin=0.6, vmax=1.4, aspect="auto")
    ax.set_xticks(range(7), DIAS)
    ax.set_yticks(range(len(t)), [c.title() for c in t.index])
    for i in range(t.shape[0]):
        for j in range(7):
            ax.text(j, i, f"{t.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Índice (1 = média da categoria)")
    ax.set_title("Categoria × dia da semana (dez-jun)")
    return _salvar(fig, "eda_05_categoria_dia.png")


def efeito_payday(g):
    """Receita media em dias de payday vs demais, por mes (sem novembro e sem janelas de evento)."""
    d = to_daily(g)
    d = d[d["data"].dt.month != 11].copy()
    excluir = pd.to_datetime(["2025-12-01", "2025-12-25", "2026-03-15", "2026-04-05",
                              "2026-05-10", "2026-06-12"])
    perto = d["data"].apply(lambda x: (abs((excluir - x).days) <= 3).any())
    d = d[~perto]
    d["payday"] = d["data"].dt.day.isin([5, 6, 7, 20, 21, 22])
    d["dia_semana"] = d["data"].dt.dayofweek
    # controla dia da semana: indice da receita sobre a media do mesmo dia da semana no mes
    d["mes"] = d["data"].dt.to_period("M")
    base = d.groupby(["mes", "dia_semana"])["receita"].transform("mean")
    d["indice"] = d["receita"] / base
    return d.groupby("payday")["indice"].agg(["mean", "count"])


def share_canal_mensal(g):
    m = g.groupby(["mes", "canal"])["receita"].sum().unstack()
    return m.div(m.sum(axis=1), axis=0)


def eventos(g):
    d = to_daily(g).set_index("data")["receita"]
    mediana = d.median()
    linhas = []
    for dt, nome in {**EVENTOS_PLOT, "2026-05-10": "Dia das Mães"}.items():
        dt = pd.Timestamp(dt)
        mes = d[d.index.to_period("M") == dt.to_period("M")]
        linhas.append({"evento": nome.replace("\n", " "), "data": dt.date(),
                       "receita_mi": d[dt] / 1e6, "x_mediana_geral": d[dt] / mediana,
                       "x_media_mes": d[dt] / mes.mean()})
    return pd.DataFrame(linhas)


def gerar_todas(path="data/raw/vendas.csv"):
    g = carregar_grade(path)
    figs = [f(g) for f in (fig_serie_diaria, fig_dia_semana, fig_intradia,
                           fig_preco_desconto, fig_categoria_dia)]
    plt.close("all")
    return g, figs
