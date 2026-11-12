"""Etapa 2 — base final para o dashboard (dicionário e validação) e gráficos de apresentação do Notebook 2.

Só lê resultados já aprovados ou recalculados de forma idêntica; não treina nem altera modelos.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

REGIOES = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}
ORDEM_REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
ORDEM_PRIORIDADE = ["alta_consenso", "alta_divergente_revisar", "intermediaria", "menor"]
ROTULO_PRIORIDADE = {"alta_consenso": "Alta — consenso", "alta_divergente_revisar": "Alta — divergente (revisar)",
                     "intermediaria": "Intermediária", "menor": "Menor"}
COR_PRIORIDADE = {"alta_consenso": "#2a78d6", "alta_divergente_revisar": "#eb6834",
                  "intermediaria": "#a9a8a2", "menor": "#d6d5cf"}
TINTA, TINTA_2, GRADE = "#0b0b0b", "#52514e", "#e4e3de"

# --------------------------------------------------------------------------
# Dicionário da base final (etapa2/saidas/previsao2029/previsoes_2029_municipios.csv)
# --------------------------------------------------------------------------
NC = "NÃO CALIBRADO — não é probabilidade"
DICIONARIO_BASE_2029 = [
    # coluna, tipo, grupo, descrição, valores / unidade, uso no dashboard
    ("id_municipio", "texto (7 dígitos)", "identificação", "Código IBGE do município", "ler como texto",
     "chave; junção com mapas; o 1º dígito indica a região"),
    ("municipio", "texto", "identificação", "Nome do município (IBGE, Censo 2022)", "", "rótulo"),
    ("sigla_uf", "texto", "identificação", "Sigla da UF", "27 UFs", "filtro"),
    ("densidade_fev2026", "decimal", "situação observada",
     "Densidade de banda larga fixa publicada pela Anatel em fev/2026", "acessos por 100 domicílios",
     "medida (observada, não prevista)"),
    ("limiar_p25_origem", "decimal", "situação observada",
     "P25 nacional da densidade em fev/2026, congelado como limiar do alvo de 2029", "8,20 para todos",
     "linha de referência"),
    ("abaixo_limiar_em_fev2026", "booleano", "situação observada",
     "O município já está abaixo do limiar em fev/2026 (fato observado)", "True / False", "filtro"),
    ("escore_rf", "decimal [0, 1]", "técnico — " + NC,
     "Escore do Random Forest para estar abaixo do limiar em fev/2029. Serve só para ordenar; a calibração não "
     "foi homologada. Gravado com 6 algarismos significativos: use posicao_rf para ordenar",
     "0 a 1, sem unidade", "não exibir como % nem como chance; no máximo em tooltip técnico"),
    ("posicao_rf", "inteiro", "ranking e prioridade", "Posição no ranking do Random Forest (1 = maior risco relativo)",
     "1 a 5.570, sem empates", "ordenação"),
    ("posicao_b2", "inteiro", "ranking e prioridade", "Posição no ranking B2 (1 = menor densidade em fev/2026)",
     "1 a 5.570, sem empates (desempate pelo código IBGE)", "ordenação"),
    ("diferenca_posicoes", "inteiro", "ranking e prioridade", "posicao_b2 − posicao_rf",
     "> 0: o RF coloca o município mais à frente que o B2", "comparação RF × B2"),
    ("categoria_rf", "texto", "ranking e prioridade", "Categoria relativa no ranking do RF",
     "maior (top 25%), intermediario (25%–50%), menor (demais)", "filtro"),
    ("categoria_b2", "texto", "ranking e prioridade", "Categoria relativa no ranking B2", "mesmas de categoria_rf",
     "filtro"),
    ("concordancia_top25", "texto", "ranking e prioridade", "Em qual top 25% o município está",
     "ambos, so_rf, so_b2, nenhum", "comparação RF × B2"),
    ("classificacao_prioridade", "texto", "ranking e prioridade",
     "Prioridade relativa: cruzamento das categorias dos dois métodos, sem pesos",
     "alta_consenso, alta_divergente_revisar, intermediaria, menor", "indicador principal; cor do mapa"),
    ("meses_ausentes_36", "inteiro", "qualidade dos dados", "Meses sem registro nos 36 meses até fev/2026",
     "0 a 36", "indicador de qualidade (separado da prioridade)"),
    ("flag_causal_fev2026", "inteiro", "qualidade dos dados", "Sinalização causal de anomalia (regra D, L=12) em fev/2026",
     "0 normal; ±1 suspeita aberta; ±2 persistência reconhecida", "indicador de qualidade"),
    ("suspeitas_36m", "inteiro", "qualidade dos dados", "Meses com suspeita estatística aberta nos últimos 36 meses",
     "0 a 36", "indicador de qualidade"),
    ("persistencia_36m", "inteiro (0/1)", "qualidade dos dados", "Houve persistência reconhecida (±2) nos últimos 36 meses",
     "0 / 1", "indicador de qualidade"),
    ("historico_meses", "inteiro", "qualidade dos dados", "Idade da série do município em fev/2026", "meses",
     "indicador de qualidade"),
    ("tendencia_36", "decimal", "contexto", "Inclinação da densidade nos últimos 36 meses (observada até fev/2026)",
     "acessos/100 dom. por ano", "contexto (não é previsão)"),
    ("alerta_qualidade", "booleano", "qualidade dos dados",
     "Algum alerta: flag causal ≠ 0, mês ausente, suspeita em 36 meses, histórico < 36 meses ou 3+ features fora da "
     "faixa do treino", "True / False", "camada separada da prioridade; não altera a categoria"),
    ("n_features_fora_faixa_treino", "inteiro", "qualidade dos dados",
     "Nº de features fora da faixa p0,5–p99,5 das coortes de treino (condição atípica)", "0 a 28",
     "indicador de qualidade"),
]
COLUNAS_DICIONARIO = ["coluna", "tipo", "grupo", "descricao", "valores_unidade", "uso_dashboard"]


def dicionario_base_2029() -> pd.DataFrame:
    return pd.DataFrame(DICIONARIO_BASE_2029, columns=COLUNAS_DICIONARIO)


def ler_base_2029(caminho) -> pd.DataFrame:
    """Leitura recomendada da base final (código IBGE como texto)."""
    return pd.read_csv(caminho, dtype={"id_municipio": str}, encoding="utf-8")


def validar_base_dashboard(base: pd.DataFrame) -> pd.DataFrame:
    """Verificações para uso direto da base no dashboard. Todas devem ser True."""
    n = len(base)
    k1, k2 = math.ceil(0.25 * n), math.ceil(0.5 * n)
    cat = {"maior", "intermediario", "menor"}
    v = [
        ("5.570 municípios, um por linha", n == 5570 and base.id_municipio.is_unique),
        ("Colunas = dicionário (mesma ordem)", list(base.columns) == [c[0] for c in DICIONARIO_BASE_2029]),
        ("Sem valores ausentes", int(base.isna().sum().sum()) == 0),
        ("Código IBGE com 7 dígitos e região derivável",
         base.id_municipio.str.fullmatch(r"\d{7}").all() and base.id_municipio.str[0].isin(list(REGIOES)).all()),
        ("27 UFs", base.sigla_uf.nunique() == 27),
        ("Rankings sem empates (1 a 5.570)", all(sorted(base[c]) == list(range(1, n + 1)) for c in ("posicao_rf", "posicao_b2"))),
        ("Categorias válidas e com 25% / 25% / 50%",
         all(set(base[c]) == cat and (base[c] == "maior").sum() == k1 and (base[c] != "menor").sum() == k2
             for c in ("categoria_rf", "categoria_b2"))),
        ("Prioridade só com os 4 valores previstos", set(base.classificacao_prioridade) == set(ORDEM_PRIORIDADE)),
        ("Escore técnico entre 0 e 1", bool(base.escore_rf.between(0, 1).all())),
        ("Nenhuma coluna de probabilidade ou densidade futura",
         not any(("prob" in c) or ("2029" in c) for c in base.columns)),
        ("Limiar único de 8,20", bool(np.allclose(base.limiar_p25_origem, 8.20395, atol=1e-4))),
        ("Alerta de qualidade é coluna própria (booleana)", base.alerta_qualidade.dtype == bool),
    ]
    return pd.DataFrame(v, columns=["verificacao", "ok"])


def com_regiao(base: pd.DataFrame) -> pd.DataFrame:
    return base.assign(regiao=base.id_municipio.str[0].map(REGIOES))


def resumo_regional(base: pd.DataFrame) -> pd.DataFrame:
    b = com_regiao(base)
    t = pd.crosstab(b.regiao, b.classificacao_prioridade).reindex(index=ORDEM_REGIOES, columns=ORDEM_PRIORIDADE, fill_value=0)
    t["municipios"] = t.sum(axis=1)
    t["pct_alta_consenso"] = t.alta_consenso / t.municipios
    t["participacao_no_consenso"] = t.alta_consenso / t.alta_consenso.sum()
    return t


# --------------------------------------------------------------------------
# Gráficos de apresentação (matplotlib, paleta validada)
# --------------------------------------------------------------------------
def _eixo(ax, grade="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(TINTA_2)
    ax.spines["bottom"].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=8)
    if grade:
        ax.grid(axis=grade, color=GRADE, lw=0.6)
        ax.set_axisbelow(True)


NOMES_METODOS = {"random_forest": "Random Forest (selecionado)", "hist_gradient_boosting": "HistGradientBoosting",
                 "logistica": "Regressão logística L2", "B2_densidade_atual": "B2 — densidade atual",
                 "B3_tendencia": "B3 — tendência", "B1_persistencia": "B1 — persistência",
                 "B0_taxa_base": "B0 — taxa base"}


def grafico_desempenho(selecao: pd.DataFrame, selecao_coorte: pd.DataFrame, metricas_val: pd.DataFrame):
    """PR-AUC média em V1–V4 (barra) e faixa entre a pior e a melhor rodada (linha), modelos × baselines."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    melhores = selecao.sort_values("pr_auc_media", ascending=False).groupby("metodo").head(1)
    linhas = []
    for m in melhores.itertuples():
        g = selecao_coorte[(selecao_coorte.metodo == m.metodo) & (selecao_coorte.config == m.config)
                           & (selecao_coorte.grupo == "todos")]
        linhas.append(dict(metodo=m.metodo, tipo="modelo", media=g.pr_auc.mean(), minimo=g.pr_auc.min(), maximo=g.pr_auc.max()))
    b = metricas_val[(metricas_val.grupo == "todos") & metricas_val.metodo.str.startswith("B")]
    for nome, g in b.groupby("metodo"):
        linhas.append(dict(metodo=nome, tipo="baseline", media=g.pr_auc.mean(), minimo=g.pr_auc.min(), maximo=g.pr_auc.max()))
    t = pd.DataFrame(linhas).sort_values("media")
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(t))
    cores = ["#2a78d6" if m == "random_forest" else ("#9cc1ee" if tp == "modelo" else "#b9b8b2")
             for m, tp in zip(t.metodo, t.tipo)]
    ax.barh(y, t.media, color=cores, height=0.62)
    ax.hlines(y, t.minimo, t.maximo, color=TINTA, lw=1.4)
    for yi, (m, mn, mx) in enumerate(zip(t.media, t.minimo, t.maximo)):
        ax.text(mx + 0.01, yi, f"{m:.3f}".replace(".", ","), va="center", fontsize=8, color=TINTA)
    ax.set_yticks(y, [NOMES_METODOS[m] for m in t.metodo], fontsize=8.5, color=TINTA)
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("PR-AUC (média das rodadas 2019–2022; traço = pior e melhor rodada)", fontsize=8, color=TINTA_2)
    ax.set_title("Modelos × baselines na validação temporal (azul: modelos; cinza: baselines)", fontsize=10,
                 color=TINTA, loc="left")
    _eixo(ax, "x")
    fig.tight_layout()
    return fig


def grafico_validacao_teste(metricas_val: pd.DataFrame, teste: pd.DataFrame, dif: pd.DataFrame):
    """(a) PR-AUC por coorte (2019–2022 validação; 2023 teste); (b) diferença RF − B2 com IC 95%."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    tab = pd.concat([metricas_val, teste])
    tab = tab[tab.grupo == "todos"].pivot_table(index="origem", columns="metodo", values="pr_auc")
    anos = [o[:4] for o in tab.index]
    x = np.arange(len(anos))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    series = [("random_forest", "#2a78d6"), ("B2_densidade_atual", "#eb6834"),
              ("B3_tendencia", "#1baf7a"), ("B1_persistencia", "#eda100"), ("B0_taxa_base", "#8a8984")]
    for m, cor in series:
        a.plot(x, tab[m].values, "o-", color=cor, lw=2, ms=5, label=NOMES_METODOS[m].replace(" (selecionado)", ""))
    for ax in (a, b):
        ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#f1f0ec", zorder=0)
    a.text(x[-1], 0.66, "teste\nreservado", ha="center", fontsize=7.5, color=TINTA_2)
    a.set_xticks(x, anos)
    a.set_ylim(0, 0.7)
    a.set_ylabel("PR-AUC (todos os municípios)", fontsize=8, color=TINTA_2)
    a.set_xlabel("origem avaliada (alvo 36 meses depois)", fontsize=8, color=TINTA_2)
    a.set_title("(a) PR-AUC por coorte", fontsize=10, color=TINTA, loc="left")
    a.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.17))
    _eixo(a)
    d = dif[dif.grupo == "todos"].copy()
    d["ano"] = d.rodada.str[-7:-3]
    xb = np.arange(len(d))
    cores = ["#8a8984" if r.startswith("teste") else "#2a78d6" for r in d.rodada]
    b.axhline(0, color=TINTA_2, lw=0.9)
    for xi, (v, lo, hi, c) in enumerate(zip(d.dif_pr_auc, d.ic95_inf, d.ic95_sup, cores)):
        b.plot([xi, xi], [lo, hi], color=c, lw=2)
        b.plot(xi, v, "o", color=c, ms=7)
        b.text(xi + 0.12, v, f"{v:+.3f}".replace(".", ","), va="center", fontsize=8, color=TINTA)
    b.set_xticks(xb, d.ano)
    b.set_ylabel("PR-AUC do RF − PR-AUC de B2", fontsize=8, color=TINTA_2)
    b.set_title("(b) RF − B2, com IC 95% (bootstrap de municípios)", fontsize=10, color=TINTA, loc="left")
    _eixo(b)
    fig.tight_layout()
    return fig


def grafico_prioridade(base: pd.DataFrame):
    """(a) municípios por classe de prioridade; (b) alertas de qualidade como camada separada."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    cont = base.classificacao_prioridade.value_counts().reindex(ORDEM_PRIORIDADE)
    alerta = base[base.alerta_qualidade].classificacao_prioridade.value_counts().reindex(ORDEM_PRIORIDADE, fill_value=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(cont))[::-1]
    rot = [ROTULO_PRIORIDADE[c] for c in cont.index]
    a.barh(y, cont.values, color=[COR_PRIORIDADE[c] for c in cont.index], height=0.6)
    for yi, v in zip(y, cont.values):
        a.text(v + 40, yi, f"{v:,}".replace(",", "."), va="center", fontsize=8.5, color=TINTA)
    a.set_yticks(y, rot, fontsize=8.5, color=TINTA)
    a.set_xlim(0, cont.max() * 1.18)
    a.set_title("(a) Classificação de prioridade relativa — 5.570 municípios", fontsize=10, color=TINTA, loc="left")
    _eixo(a, "x")
    pct = alerta / cont
    b.barh(y, pct.values, color="#8a8984", height=0.6)
    for yi, (p, n) in enumerate(zip(pct.values, alerta.values)):
        b.text(p + 0.004, y[yi], f"{n} ({p:.0%})", va="center", fontsize=8.5, color=TINTA)
    b.set_yticks(y, rot, fontsize=8.5, color=TINTA)
    b.set_xlim(0, max(pct.max() * 1.35, 0.1))
    b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    b.set_title("(b) Com alerta de qualidade (camada separada)", fontsize=10, color=TINTA, loc="left")
    b.set_xlabel("parte da classe com algum alerta", fontsize=8, color=TINTA_2)
    _eixo(b, "x")
    fig.tight_layout()
    return fig


def grafico_regional(base: pd.DataFrame):
    """Parte dos municípios de cada região em prioridade alta (consenso e divergente)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    t = resumo_regional(base)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(t))[::-1]
    c = t.alta_consenso / t.municipios
    d = t.alta_divergente_revisar / t.municipios
    ax.barh(y, c, color=COR_PRIORIDADE["alta_consenso"], height=0.6, label=ROTULO_PRIORIDADE["alta_consenso"])
    ax.barh(y, d, left=c + 0.004, color=COR_PRIORIDADE["alta_divergente_revisar"], height=0.6,
            label=ROTULO_PRIORIDADE["alta_divergente_revisar"])
    for yi, (r, row) in zip(y, t.iterrows()):
        ax.text(c[r] + d[r] + 0.012, yi, f"{int(row.alta_consenso)} + {int(row.alta_divergente_revisar)} de {int(row.municipios):,}"
                .replace(",", "."), va="center", fontsize=8, color=TINTA)
    ax.set_yticks(y, t.index, fontsize=9, color=TINTA)
    ax.set_xlim(0, 0.75)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("parte dos municípios da região", fontsize=8, color=TINTA_2)
    ax.set_title("Municípios em prioridade alta, por região", fontsize=10, color=TINTA, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _eixo(ax, "x")
    fig.tight_layout()
    return fig
