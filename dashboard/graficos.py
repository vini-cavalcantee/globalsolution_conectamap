"""Gráficos Plotly do dashboard. Paleta semântica única (mesma cor = mesmo significado em todas as páginas)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dados import (PRIORIDADES, ROTULO_CATEGORIA, ROTULO_CONCORDANCIA, ROTULO_CURTO, ROTULO_PRIORIDADE,
                   subconjunto_geo)

# ---- todas as cores vêm do módulo central de paleta -------------------------------------------------
from paleta import (COR_ALERTA, COR_CATEGORIA, COR_CONCORDANCIA, COR_PRIORIDADE, GRADE, PETROLEO,  # noqa: E402,F401
                    TERRACOTA, TINTA, TINTA_2)
from paleta import BORDA_MARCA_CLARA as BORDA_CLARA  # noqa: E402
from paleta import FORA_FILTRO as COR_FORA  # noqa: E402
import paleta as P  # noqa: E402

SIMBOLO_CONCORDANCIA = {"ambos": "circle", "so_rf": "triangle-up", "so_b2": "diamond", "nenhum": "circle"}
_CLARAS = {P.MENOR, P.INTERMEDIARIA}
FONTE = dict(family="'Source Sans', 'Source Sans Pro', 'Source Sans 3', Arial, sans-serif", color=TINTA, size=13)


def _layout(fig, altura=320, legenda=False, margin=None, **kw):
    fig.update_layout(height=altura, margin=margin or dict(l=10, r=10, t=10, b=10), font=FONTE, showlegend=legenda,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(bgcolor="white", bordercolor=P.BORDA_MARCA_CLARA, font_size=13,
                                      font_family=FONTE["family"], font_color=TINTA), **kw)
    fig.update_xaxes(gridcolor=GRADE, zeroline=False, tickfont=dict(color=TINTA_2, size=12),
                     title_font=dict(color=TINTA_2, size=12))
    fig.update_yaxes(gridcolor=GRADE, zeroline=False, tickfont=dict(color=TINTA_2, size=12),
                     title_font=dict(color=TINTA_2, size=12))
    return fig


def _fmt(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def _pct(x) -> str:
    return f"{x:.1%}".replace(".", ",")


def _borda(cor: str):
    return dict(color=BORDA_CLARA, width=1) if cor in _CLARAS else dict(width=0)


# ------------------------------------------------------------------------------------------------------
# Visão geral
# ------------------------------------------------------------------------------------------------------
def fig_prioridades(base: pd.DataFrame):
    cont = base.classificacao_prioridade.value_counts().reindex(PRIORIDADES, fill_value=0)
    total = max(int(cont.sum()), 1)
    cores = [COR_PRIORIDADE[p] for p in cont.index]
    fig = go.Figure(go.Bar(
        x=cont.values, y=[ROTULO_CURTO[p] for p in cont.index], orientation="h", marker_color=cores,
        marker_line=dict(color=[BORDA_CLARA if c in _CLARAS else c for c in cores], width=1),
        text=[f"<b>{_fmt(v)}</b> · {_pct(v / total)}" for v in cont.values], textposition="outside", cliponaxis=False,
        textfont=dict(color=TINTA, size=13), customdata=[ROTULO_PRIORIDADE[p] for p in cont.index],
        hovertemplate="%{customdata}: %{x:,} municípios<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TINTA, size=13))
    fig.update_xaxes(range=[0, cont.max() * 2.5 if cont.max() else 1], showticklabels=False, showgrid=False)
    return _layout(fig, 250, separators=",.", bargap=0.35)


def fig_territorial(t: pd.DataFrame, modo: str = "Quantidade"):
    """Municípios prioritários em territórios COMPLETOS (ver dados.comparacao_territorial).
    Quantidade = número de municípios; % = percentual de municípios prioritários sobre o total da região, UF ou Brasil."""
    pct = modo != "Quantidade"
    rot = [f"<b>{n}</b>" if nv == "UF" else (n if nv == "Brasil" else f"{n}") for n, nv in zip(t.index, t.nivel)]
    fig = go.Figure()
    for p, col_n, col_p in (("alta_consenso", "alta_consenso", "pct_consenso"),
                            ("alta_divergente_revisar", "alta_divergente_revisar", "pct_divergente")):
        fig.add_bar(y=rot, x=t[col_p] if pct else t[col_n], orientation="h", name=ROTULO_CURTO[p],
                    marker_color=COR_PRIORIDADE[p], customdata=np.c_[t[col_n], t.municipios, t[col_p], t.index],
                    hovertemplate="%{customdata[3]} — " + ROTULO_PRIORIDADE[p] +
                                  "<br>%{customdata[0]:,} de %{customdata[1]:,} municípios "
                                  "(%{customdata[2]:.1%})<extra></extra>")
    tot = t.pct_prioritarios if pct else t.prioritarios
    fig.add_scatter(y=rot, x=tot, mode="text", showlegend=False, hoverinfo="skip", cliponaxis=False,
                    text=[f"  <b>{_pct(v)}</b>" if pct else f"  <b>{_fmt(v)}</b> de {_fmt(n)}" for v, n in
                          zip(tot, t.municipios)], textposition="middle right", textfont=dict(color=TINTA, size=13))
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal",
                                                   font=dict(size=12)), bargap=0.32)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TINTA, size=13))
    fig.update_xaxes(tickformat=".0%" if pct else ",d",
                     range=[0, tot.max() * (1.28 if pct else 1.5) if len(tot) and tot.max() else 1],
                     title="% de municípios prioritários" if pct
                     else "número de municípios prioritários")
    return _layout(fig, max(220, 46 * len(t) + 110), legenda=True, separators=",.", margin=dict(l=10, r=10, t=50, b=10))


# ------------------------------------------------------------------------------------------------------
# Mapa (geometrias, cache e enquadramento inalterados; só cores, rótulos e dicas)
# ------------------------------------------------------------------------------------------------------
MODO_PRIORIDADE, MODO_ALERTA = "Prioridade de acompanhamento", "Pontos de atenção nos dados"
MODO_RF, MODO_B2 = "Classificação pelo modelo preditivo", "Classificação pela densidade atual"
MODOS_PRINCIPAIS = [MODO_PRIORIDADE, MODO_ALERTA]
MODOS_AVANCADOS = [MODO_RF, MODO_B2]
MODOS_MAPA = {MODO_PRIORIDADE: ("classificacao_prioridade", COR_PRIORIDADE, ROTULO_CURTO),
              MODO_ALERTA: ("alerta_qualidade", COR_ALERTA, {True: "Com ponto de atenção", False: "Sem ponto de atenção"}),
              MODO_RF: ("categoria_rf", COR_CATEGORIA, ROTULO_CATEGORIA),
              MODO_B2: ("categoria_b2", COR_CATEGORIA, ROTULO_CATEGORIA)}


def _hover(d: pd.DataFrame):
    """Dica do mapa: primeiro o essencial (município, densidade, prioridade, atenção); posições ao final, menores."""
    custom = np.c_[d.municipio, d.sigla_uf, d.densidade_fev2026.map(lambda v: f"{v:.2f}".replace(".", ",")),
                   d.classificacao_prioridade.map(ROTULO_PRIORIDADE),
                   np.where(d.alerta_qualidade, "sim — conferir a série", "não"),
                   d.posicao_rf.map(_fmt), d.posicao_b2.map(_fmt), d.id_municipio]
    tpl = ("<b>%{customdata[0]} — %{customdata[1]}</b>"
           "<br>Densidade em fev/2026: <b>%{customdata[2]}</b> acessos por 100 domicílios"
           "<br>Prioridade: <b>%{customdata[3]}</b>"
           "<br>Ponto de atenção nos dados: %{customdata[4]}"
           "<br><span style='font-size:11px;color:" + TINTA_2 + "'>Posição nacional — modelo: %{customdata[5]} · "
           "densidade atual: %{customdata[6]} · IBGE %{customdata[7]}</span><extra></extra>")
    return custom, tpl


def enquadramento(geo: dict, altura: int, largura: int = 980) -> tuple[dict, float]:
    """Centro e zoom (Web Mercator) que enquadram as geometrias, calculados a partir dos próprios limites."""
    import math
    xs, ys = [], []
    for f in geo["features"]:
        g = f["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            for x, y in poly[0]:
                xs.append(x)
                ys.append(y)
    if not xs:
        return {"lon": -52.0, "lat": -14.5}, 3.0
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    merc = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))  # noqa: E731
    larg = max(x1 - x0, 0.05) / 360
    alt = max(merc(y1) - merc(y0), 0.001) / (2 * math.pi)
    zoom = math.log2(min(largura / 512 / larg, altura / 512 / alt)) - 0.15
    centro_lat = math.degrees(math.atan(math.sinh((merc(y0) + merc(y1)) / 2)))
    return {"lon": (x0 + x1) / 2, "lat": centro_lat}, round(max(min(zoom, 12), 1.5), 2)


def fig_mapa(territorio: pd.DataFrame, filtrado_ids: set, geo: dict, geo_uf: dict, modo: str = MODO_PRIORIDADE,
             altura: int = 620):
    """Municípios do recorte territorial; os que não passam nos demais filtros ficam em cinza-claro.
    As cores de prioridade nunca são alteradas pelos alertas (que têm modo próprio). Os municípios
    'Métodos divergem' recebem contorno escuro como segunda codificação (não dependem só da cor)."""
    col, cores, rotulos = MODOS_MAPA[modo]
    fig = go.Figure()
    geo_t = subconjunto_geo(geo, territorio.id_municipio)
    fora = territorio[~territorio.id_municipio.isin(filtrado_ids)]
    dentro = territorio[territorio.id_municipio.isin(filtrado_ids)]
    if len(fora):
        custom, tpl = _hover(fora)
        fig.add_choroplethmap(geojson=subconjunto_geo(geo, fora.id_municipio), locations=fora.id_municipio,
                              featureidkey="id", z=np.zeros(len(fora)), colorscale=[[0, COR_FORA], [1, COR_FORA]],
                              showscale=False, name="Fora do recorte", showlegend=True, marker_line_color=P.FUNDO,
                              marker_line_width=0.2, customdata=custom, hovertemplate=tpl)
    for valor, cor in cores.items():
        d = dentro[dentro[col] == valor]
        if not len(d):
            continue
        custom, tpl = _hover(d)
        destaque = modo == MODO_PRIORIDADE and valor == "alta_divergente_revisar"
        fig.add_choroplethmap(geojson=subconjunto_geo(geo, d.id_municipio), locations=d.id_municipio, featureidkey="id",
                              z=np.zeros(len(d)), colorscale=[[0, cor], [1, cor]], showscale=False,
                              name=f"{rotulos[valor]} ({_fmt(len(d))})", showlegend=True,
                              marker_line_color=PETROLEO if destaque else P.FUNDO,
                              marker_line_width=1.2 if destaque else 0.2, customdata=custom, hovertemplate=tpl)
    ufs = set(territorio.sigla_uf)
    contorno = {"type": "FeatureCollection", "features": [f for f in geo_uf["features"] if f["id"] in ufs]}
    fig.add_choroplethmap(geojson=contorno, locations=sorted(ufs), featureidkey="id", z=np.zeros(len(ufs)),
                          colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False, showlegend=False,
                          marker_line_color=TINTA_2, marker_line_width=0.8, hoverinfo="skip")
    # a seleção por clique não deve esmaecer os demais municípios (as cores de prioridade ficam intactas)
    fig.update_traces(selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)))
    centro, zoom = enquadramento(geo_t, altura)
    # estilo "white-bg": sem mapa de fundo nem servidor externo de blocos (funciona offline)
    fig.update_layout(map=dict(style="white-bg", center=centro, zoom=zoom),
                      legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=13), itemclick=False,
                                  itemdoubleclick=False))
    return _layout(fig, altura, legenda=True, margin=dict(l=0, r=0, t=0, b=44))


def fig_ufs(t: pd.DataFrame, modo: str = "Quantidade", destaque: list | None = None):
    """Estados completos; as UFs do filtro aparecem em negrito com uma seta."""
    pct = modo != "Quantidade"
    destaque = set(destaque or [])
    t = t.sort_values("pct_prioritarios" if pct else "prioritarios", ascending=True)
    rot = [f"<b>{u} ▸</b>" if u in destaque else u for u in t.index]
    fig = go.Figure()
    for p, col_n, col_p in (("alta_consenso", "alta_consenso", "pct_consenso"),
                            ("alta_divergente_revisar", "alta_divergente_revisar", "pct_divergente")):
        fig.add_bar(y=rot, x=t[col_p] if pct else t[col_n], orientation="h", name=ROTULO_CURTO[p],
                    marker_color=COR_PRIORIDADE[p], customdata=np.c_[t[col_n], t.municipios, t[col_p], t.index],
                    hovertemplate="%{customdata[3]} — " + ROTULO_PRIORIDADE[p] +
                                  ": %{customdata[0]:,} de %{customdata[1]:,} municípios (%{customdata[2]:.1%})<extra></extra>")
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, traceorder="normal"),
                      bargap=0.25)
    fig.update_xaxes(tickformat=".0%" if pct else ",d",
                     title="% de municípios prioritários do estado" if pct else "número de municípios prioritários")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=TINTA))
    return _layout(fig, max(260, 24 * len(t) + 100), legenda=True, separators=",.", margin=dict(l=10, r=10, t=40, b=10))


# ------------------------------------------------------------------------------------------------------
# Comparação dos critérios
# ------------------------------------------------------------------------------------------------------
def fig_dispersao(base: pd.DataFrame, destaque_ids: set | None = None, n_total: int = 5570):
    """Posição pela densidade atual × posição pelo modelo (rankings nacionais). Fora do recorte: cinza."""
    k = -(-n_total * 25 // 100)
    fig = go.Figure()
    dentro = base if destaque_ids is None else base[base.id_municipio.isin(destaque_ids)]
    if destaque_ids is not None:
        fora = base[~base.id_municipio.isin(destaque_ids)]
        fig.add_scattergl(x=fora.posicao_b2, y=fora.posicao_rf, mode="markers", name="Fora do recorte",
                          marker=dict(size=3, color=P.FORA_FILTRO), hoverinfo="skip")
    for c in ["nenhum", "ambos", "so_rf", "so_b2"]:
        d = dentro[dentro.concordancia_top25 == c]
        if not len(d):
            continue
        destacado = c in ("so_rf", "so_b2")
        fig.add_scattergl(
            x=d.posicao_b2, y=d.posicao_rf, mode="markers", name=f"{ROTULO_CONCORDANCIA[c]} ({_fmt(len(d))})",
            marker=dict(size=10 if destacado else 4, color=COR_CONCORDANCIA[c], symbol=SIMBOLO_CONCORDANCIA[c],
                        line=dict(width=1, color=PETROLEO) if destacado else dict(width=0)),
            customdata=np.c_[d.municipio, d.sigla_uf, d.densidade_fev2026, d.diferenca_posicoes],
            hovertemplate="<b>%{customdata[0]} — %{customdata[1]}</b><br>Posição pelo modelo: %{y:,}"
                          "<br>Posição pela densidade atual: %{x:,}<br>Densidade fev/2026: %{customdata[2]:.2f}"
                          "<extra></extra>")
    for eixo in ("x", "y"):
        (fig.add_vline if eixo == "x" else fig.add_hline)(k, line=dict(color=TINTA_2, width=1, dash="dash"))
    fig.add_annotation(x=k, y=0, text="limite dos 25% primeiros (1.393)", showarrow=False, xanchor="left",
                       yanchor="bottom", font=dict(size=11, color=TINTA_2), xshift=4, yshift=2)
    fig.update_xaxes(title="Posição pela densidade atual (1 = menor densidade em fev/2026)", range=[0, n_total + 50])
    fig.update_yaxes(title="Posição pelo modelo preditivo (1 = maior atenção)", range=[n_total + 50, 0])
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0))
    return _layout(fig, 520, legenda=True, separators=",.")


def fig_densidade_limiar(linha: pd.Series, mediana_nacional: float):
    """Densidade observada em fev/2026 comparada ao patamar de referência e à mediana nacional. Sem valores futuros."""
    d, p25 = float(linha.densidade_fev2026), float(linha.limiar_p25_origem)
    cor = COR_PRIORIDADE.get(linha.classificacao_prioridade, PETROLEO)
    xmax = max(d, mediana_nacional, p25) * 1.25
    fig = go.Figure(go.Bar(x=[d], y=[""], orientation="h", marker_color=cor, marker_line=_borda(cor), width=0.45,
                           text=[f"<b>{d:.2f}</b>".replace(".", ",")], textposition="outside", cliponaxis=False,
                           textfont=dict(color=TINTA, size=13),
                           hovertemplate="Densidade fev/2026: %{x:.2f}<extra></extra>"))
    for v, rot, dash in ((p25, f"patamar de referência ({p25:.2f})".replace(".", ","), "solid"),
                         (mediana_nacional, f"mediana do Brasil ({mediana_nacional:.1f})".replace(".", ","), "dot")):
        fig.add_vline(v, line=dict(color=PETROLEO, width=1.5, dash=dash))
        fig.add_annotation(x=v, y=1.0, yref="paper", yanchor="bottom", text=rot, showarrow=False, xanchor="center",
                           font=dict(size=11, color=TINTA_2))
    fig.update_xaxes(range=[0, xmax], title="acessos de banda larga fixa por 100 domicílios (fev/2026)")
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return _layout(fig, 170, margin=dict(l=10, r=10, t=30, b=40))


# ------------------------------------------------------------------------------------------------------
# Qualidade dos dados
# ------------------------------------------------------------------------------------------------------
def fig_alerta_territorio(t: pd.DataFrame, titulo_eixo: str = "% de municípios com ponto de atenção",
                          ordenar: bool = True):
    t = t.sort_values("pct_alerta") if ordenar else t.iloc[::-1]
    fig = go.Figure(go.Bar(
        y=t.index, x=t.pct_alerta, orientation="h", marker_color=COR_ALERTA[True],
        text=[f"{_fmt(a)} de {_fmt(n)}" for a, n in zip(t.com_alerta, t.municipios)], textposition="outside",
        cliponaxis=False, textfont=dict(color=TINTA), hovertemplate="%{y}: %{x:.1%} com ponto de atenção<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(t.pct_alerta.max() * 1.4, 0.05)], title=titulo_eixo)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=TINTA))
    return _layout(fig, max(260, 22 * len(t) + 70), separators=",.")


def fig_prioridade_alerta(tab: pd.DataFrame):
    """Parte de cada classe de prioridade com ponto de atenção — duas dimensões independentes."""
    rot = [ROTULO_CURTO[p] for p in tab.index]
    fig = go.Figure()
    fig.add_bar(y=rot, x=tab["Sem alerta"], orientation="h", name="Sem ponto de atenção", marker_color=P.MENOR,
                marker_line=dict(color=BORDA_CLARA, width=1), hovertemplate="%{y}: %{x:,} sem ponto de atenção<extra></extra>")
    fig.add_bar(y=rot, x=tab["Com alerta"], orientation="h", name="Com ponto de atenção", marker_color=COR_ALERTA[True],
                hovertemplate="%{y}: %{x:,} com ponto de atenção<extra></extra>")
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, traceorder="normal"))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TINTA))
    fig.update_xaxes(title="número de municípios")
    return _layout(fig, 280, legenda=True, separators=",.", margin=dict(l=10, r=10, t=40, b=10))
