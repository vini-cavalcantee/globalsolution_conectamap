"""ConectaMap — dashboard interativo (Etapa 3).

Execução, na pasta do projeto:  streamlit run dashboard/app.py

Usa só as saídas aprovadas da Etapa 2: nenhum modelo é treinado e nenhum ranking é recalculado.
Os códigos internos (alta_consenso, posicao_rf...) ficam nos dados; a interface usa rótulos amigáveis.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import componentes as K  # noqa: E402
import dados as D  # noqa: E402
import graficos as G  # noqa: E402
import paleta as P  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(page_title="ConectaMap — prioridades de conectividade", page_icon=str(ASSETS / "icone_conectamap.svg"),
                   layout="wide", initial_sidebar_state="expanded")
st.logo(str(ASSETS / "marca_conectamap.svg"), size="large", icon_image=str(ASSETS / "icone_conectamap.svg"))
K.estilo()


# ======================================================================================================
# Dados (com cache)
# ======================================================================================================
@st.cache_data(show_spinner="Carregando a base de 2029…")
def carregar():
    base = D.carregar_base()
    return base, D.validar_base(base)


@st.cache_data
def carregar_dicionario():
    return D.carregar_dicionario()


@st.cache_resource(show_spinner="Carregando a malha municipal…")
def carregar_geo():
    return D.carregar_geometrias(D.ARQ_GEO_MUN), D.carregar_geometrias(D.ARQ_GEO_UF)


@st.cache_data
def carregar_tecnico():
    return D.resumo_tecnico()


@st.cache_data
def metadados_geo():
    return json.loads(D.ARQ_GEO_META.read_text(encoding="utf-8"))


BASE, VALIDACAO = carregar()
if not VALIDACAO.ok.all():
    st.error("A base de 2029 não passou nas verificações de integridade. O dashboard não exibirá os dados.")
    st.dataframe(VALIDACAO, hide_index=True)
    st.stop()
NAC = D.indicadores(BASE)
PATAMAR = K.dec(BASE.limiar_p25_origem.iloc[0])
MEDIANA_BR = float(BASE.densidade_fev2026.median())


# ======================================================================================================
# Filtros globais (barra lateral)
# ======================================================================================================
CHAVES = {"f_regioes": [], "f_ufs": [], "f_prioridades": [], "f_alerta": "Todos"}


def limpar_filtros():
    for k, v in CHAVES.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v


def filtros_globais() -> D.Filtros:
    for k, v in CHAVES.items():
        st.session_state.setdefault(k, list(v) if isinstance(v, list) else v)
    with st.sidebar:
        K.titulo_filtros()
        regioes = st.multiselect("Região", D.ORDEM_REGIOES, key="f_regioes", placeholder="Todas as regiões")
        opcoes_uf = D.ufs_das_regioes(regioes)
        st.session_state.f_ufs = [u for u in st.session_state.f_ufs if u in opcoes_uf]
        ufs = st.multiselect("Estado (UF)", opcoes_uf, key="f_ufs", placeholder="Todos os estados")
        prioridades = st.multiselect("Prioridade de acompanhamento", D.PRIORIDADES, key="f_prioridades",
                                     format_func=D.ROTULO_PRIORIDADE.get, placeholder="Todas as prioridades")
        alerta = st.radio("Pontos de atenção nos dados", D.OPCOES_ALERTA, key="f_alerta",
                          help=D.TEXTO_ALERTA)
        st.button("Limpar filtros", on_click=limpar_filtros, width="stretch", icon=":material/filter_alt_off:")
        st.caption("Os filtros escolhem quais municípios aparecem. As posições continuam sendo as nacionais.")
    return D.Filtros(regioes, ufs, prioridades, alerta)


# ======================================================================================================
# Resumo de um município (usado no mapa e em Explorar municípios)
# ======================================================================================================
def _tendencia(v: float) -> str:
    return "em crescimento" if v > 0.25 else ("em queda" if v < -0.25 else "praticamente estável")


def painel_municipio(linha: pd.Series, chave: str):
    p = linha.classificacao_prioridade
    with st.container(border=True):
        st.markdown(f"### {linha.municipio} — {linha.sigla_uf}")
        st.caption(f"Região {linha.regiao} · código IBGE {linha.id_municipio}")
        c = st.columns(4)
        with c[0]:
            K.card("Densidade em fev/2026", K.dec(linha.densidade_fev2026), "acessos por 100 domicílios", G.PETROLEO)
        with c[1]:
            K.card("Patamar de referência (fev/2026)", PATAMAR,
                   "o município está " + ("abaixo" if linha.abaixo_limiar_em_fev2026 else "acima") + " do patamar",
                   P.REFERENCIA)
        with c[2]:
            K.card("Prioridade de acompanhamento", D.ROTULO_PRIORIDADE[p], "posição relativa até 2029",
                   G.COR_PRIORIDADE[p], texto=True)
        with c[3]:
            K.card("Tendência histórica (36 meses)", K.sinal(linha.tendencia_36, 2),
                   f"por ano — {_tendencia(linha.tendencia_36)}", P.INTERMEDIARIA)
        if linha.alerta_qualidade:
            K.atencao("Ponto de atenção nos dados")
            st.caption(D.TEXTO_ALERTA)
        st.plotly_chart(G.fig_densidade_limiar(linha, MEDIANA_BR), config={"displayModeBar": False},
                        key=f"dens_{chave}_{linha.id_municipio}")
        st.caption("Valores observados até fevereiro de 2026. O ConectaMap não estima a densidade de 2029.")
        with st.expander("Informações complementares — ordenações e indicadores técnicos"):
            a, b = st.columns(2)
            a.metric("Posição nacional — modelo preditivo", f"{K.fmt(linha.posicao_rf)}º",
                     help="Ordenação do Random Forest (1 = maior atenção)")
            a.caption(f"Classificação: {D.ROTULO_CATEGORIA[linha.categoria_rf]}")
            b.metric("Posição nacional — densidade atual", f"{K.fmt(linha.posicao_b2)}º",
                     help="Ordenação B2: 1 = menor densidade em fev/2026")
            b.caption(f"Classificação: {D.ROTULO_CATEGORIA[linha.categoria_b2]}")
            q = st.columns(4)
            q[0].metric("Variações incomuns (36 meses)", int(linha.suspeitas_36m),
                        help="Meses com variação estatisticamente incomum na série")
            q[1].metric("Variação que merece atenção", "sim" if linha.flag_causal_fev2026 != 0 else "não",
                        help="Sinalização causal aberta em fev/2026 (código técnico: "
                             f"{int(linha.flag_causal_fev2026)})")
            q[2].metric("Indicadores fora do padrão histórico", int(linha.n_features_fora_faixa_treino),
                        help="Quantos indicadores do município estão fora da faixa vista no histórico do modelo")
            q[3].metric("Meses sem dados (36 meses)", int(linha.meses_ausentes_36))
            st.caption(f"Índice de ordenação do modelo: {K.dec(linha.escore_rf, 4)} — serve só para ordenar; "
                       "não é probabilidade nem porcentagem de chance.")


# ======================================================================================================
# Comparação territorial (territórios completos; denominadores explícitos)
# ======================================================================================================
def bloco_territorial(f: D.Filtros, chave: str):
    """Municípios prioritários em territórios COMPLETOS. Com uma UF: UF × região × Brasil.
    Os filtros de prioridade e de pontos de atenção não alteram estes denominadores."""
    t = D.comparacao_territorial(BASE, f)
    modo = st.segmented_control("Medida", ["Nº de municípios", "% de municípios"], default="% de municípios",
                                key=f"{chave}_medida", label_visibility="collapsed") or "% de municípios"
    pct = not modo.startswith("Nº")
    if f.ufs:
        titulo = "UF selecionada × região completa × Brasil" if len(f.ufs) == 1 else "UFs selecionadas × regiões × Brasil"
    elif f.regioes:
        titulo = "Região completa × Brasil"
    else:
        titulo = "Regiões completas e Brasil"
    st.markdown(f"**{'Percentual de municípios prioritários' if pct else 'Número de municípios prioritários'}** "
                f"<span style='color:{P.TINTA_2};font-size:.88rem'>· {titulo}</span>", unsafe_allow_html=True)
    st.plotly_chart(G.fig_territorial(t, "%" if pct else "Quantidade"), config={"displayModeBar": False},
                    key=f"{chave}_terr")
    nota = ("Percentual de municípios prioritários em relação ao total de municípios da região ou estado (e do Brasil). "
            if pct else "Base: total de municípios de cada região ou estado (e do Brasil). ")
    nota += "Prioritários = alta prioridade + métodos divergem."
    if f.prioridades or f.alerta != "Todos":
        nota += " Os filtros de prioridade e de pontos de atenção não alteram estes percentuais."
    if f.ativos:
        r = D.resumo_recorte(BASE, f)
        nota += (f" No recorte filtrado: {K.fmt(r['prioritarios'])} de {K.fmt(r['municipios'])} municípios são "
                 f"prioritários ({K.pct(r['pct_prioritarios'])} do recorte)." if r["municipios"] else
                 " O recorte filtrado não tem municípios.")
    st.caption(nota)
    return pct


# ======================================================================================================
# Página 1 — Visão geral
# ======================================================================================================
def pagina_visao_geral():
    K.cabecalho_inicial(
        "Onde a conectividade precisa de mais atenção?",
        "Explore os municípios brasileiros priorizados para acompanhamento até 2029, com base na evolução da banda "
        "larga fixa.",
        ["Dados de referência: fevereiro de 2026", "Horizonte: fevereiro de 2029", "5.570 municípios"])
    f = FILTROS
    rec = D.aplicar_filtros(BASE, f)
    ind = D.indicadores(rec)
    if f.ativos:
        K.faixa_recorte(f.descricao(), len(rec))

    def nac(chave):
        return f"Brasil: {K.fmt(NAC[chave])}" if f.ativos else None

    c = st.columns(4)
    cartoes = [
        ("Municípios analisados", ind["municipios"], nac("municipios"), G.PETROLEO, None),
        ("Alta prioridade — métodos concordam", ind["alta_consenso"], nac("alta_consenso"),
         G.COR_PRIORIDADE["alta_consenso"], D.DESCRICAO_PRIORIDADE["alta_consenso"]),
        ("Métodos divergem — analisar", ind["alta_divergente_revisar"], nac("alta_divergente_revisar"),
         G.COR_PRIORIDADE["alta_divergente_revisar"], D.DESCRICAO_PRIORIDADE["alta_divergente_revisar"]),
        ("Com pontos de atenção nos dados", ind["alerta"], nac("alerta"), G.COR_ALERTA[True], D.TEXTO_ALERTA),
    ]
    for i, (col, (rot, val, n, cor, aj)) in enumerate(zip(c, cartoes)):
        with col:
            K.card(rot, K.fmt(val), n, cor, aj, destaque=(i == 1))
    K.indicadores_secundarios([
        (K.fmt(ind["abaixo_limiar"]), f"já estavam abaixo do patamar de referência ({PATAMAR}) em fev/2026"
         + (f" (Brasil: {K.fmt(NAC['abaixo_limiar'])})" if f.ativos else "")),
        (K.pct(NAC["concordancia_top25"]), "de concordância entre os rankings (Brasil)",
         D.TEXTO_CONCORDANCIA.format(pct=K.pct(NAC["concordancia_top25"]))),
    ])
    st.space("medium")

    territorial = D.aplicar_filtros(BASE, f, usar_prioridade=False)
    if territorial.empty:
        K.sem_resultados()
        K.rodape()
        return
    e, d = st.columns([1, 1.15], gap="large")
    with e:
        st.markdown("#### Como as prioridades se distribuem")
        st.plotly_chart(G.fig_prioridades(territorial), config={"displayModeBar": False}, key="vg_prior")
        K.legenda_prioridades(G.COR_PRIORIDADE, D.ROTULO_PRIORIDADE, D.DESCRICAO_PRIORIDADE)
        st.caption(f"Percentuais sobre os {K.fmt(len(territorial))} municípios "
                   + ("do recorte (o filtro de prioridade não se aplica a este gráfico)." if f.ativos else "do Brasil."))
    with d:
        st.markdown("#### Onde estão os municípios prioritários")
        bloco_territorial(f, "vg")

    K.nota("<b>Como ler:</b> o ConectaMap identifica municípios que merecem acompanhamento em relação à densidade de "
           "banda larga fixa. A análise considera dados até fevereiro de 2026 e apresenta uma priorização para 2029, "
           f"não uma garantia sobre a situação futura. O patamar de referência é de {PATAMAR} acessos por 100 "
           "domicílios — o nível que separava os 25% municípios de menor densidade em fevereiro de 2026.")
    st.space("small")
    a, b, _ = st.columns([1, 1, 2])
    a.page_link(PAG_MAPA, label="Ver no mapa de prioridades", icon=":material/map:")
    b.page_link(PAG_SOBRE, label="Entender a metodologia", icon=":material/info:")
    K.rodape()


# ======================================================================================================
# Página 2 — Mapa de prioridades
# ======================================================================================================
@st.cache_data(max_entries=24, show_spinner="Desenhando o mapa…")
def figura_mapa(regioes: tuple, ufs: tuple, prioridades: tuple, alerta: str, modo: str, altura: int):
    """Figura do mapa por combinação de filtros (cache: a mesma combinação não é redesenhada)."""
    geo, geo_uf = carregar_geo()
    f = D.Filtros(list(regioes), list(ufs), list(prioridades), alerta)
    territorio = D.aplicar_filtros(BASE, D.Filtros(f.regioes, f.ufs))
    filtrado = D.aplicar_filtros(BASE, f)
    return G.fig_mapa(territorio, set(filtrado.id_municipio), geo, geo_uf, modo, altura)


AVANCADO_PADRAO = "Não — usar a prioridade de acompanhamento"


def pagina_mapa():
    K.cabecalho_interno("Mapa de prioridades",
                        "Veja onde estão os municípios que mais merecem atenção. Passe o cursor sobre um município "
                        "para ver os dados ou clique para abrir o resumo.")
    f = FILTROS
    territorio = D.aplicar_filtros(BASE, D.Filtros(f.regioes, f.ufs))
    filtrado = D.aplicar_filtros(BASE, f)
    c1, c2 = st.columns([2.4, 1], vertical_alignment="bottom")
    with c1:
        modo = st.segmented_control("Colorir o mapa por", G.MODOS_PRINCIPAIS, default=G.MODO_PRIORIDADE,
                                    key="mapa_modo") or G.MODO_PRIORIDADE
    with c2:
        st.markdown(f"<div style='text-align:right;color:{P.TINTA_2};font-size:.95rem'><b style='color:{P.PETROLEO};"
                    f"font-size:1.35rem'>{K.fmt(len(filtrado))}</b> municípios no recorte</div>", unsafe_allow_html=True)
    with st.expander("Opções avançadas — classificação de cada critério"):
        avancado = st.radio("Colorir pela classificação de um único critério", [AVANCADO_PADRAO] + G.MODOS_AVANCADOS,
                            key="mapa_avancado", label_visibility="collapsed")
        st.caption("Mostra separadamente a classificação do modelo preditivo (Random Forest) ou da densidade atual "
                   "(B2), nas mesmas cores: alta (25% primeiros), intermediária e menor. A comparação completa está em "
                   "Explorar municípios → Comparação entre métodos.")
    if avancado != AVANCADO_PADRAO:
        modo = avancado
        st.caption(f"Modo avançado ativo: **{avancado.lower()}**.")
    elif modo == G.MODO_ALERTA:
        st.caption("Roxo: municípios com algum ponto de atenção nos dados. Esta camada não muda a prioridade.")
    else:
        st.caption("Os municípios com classificação divergente têm contorno escuro, além da cor.")
    if territorio.empty:
        K.sem_resultados()
        K.rodape()
        return
    if filtrado.empty:
        st.warning("Nenhum município da região ou UF selecionada atende aos demais filtros; todos aparecem em cinza.")
    # uma UF ocupa pouco espaço horizontal: altura menor, sem esticar as geometrias (o zoom é calculado pela forma)
    altura = 640 if not (f.regioes or f.ufs) else (500 if len(f.ufs) == 1 else 560)
    evento = st.plotly_chart(figura_mapa(tuple(f.regioes), tuple(f.ufs), tuple(f.prioridades), f.alerta, modo, altura),
                             config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True,
                                     "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"]},
                             on_select="rerun", selection_mode="points", key="mapa_brasil")
    st.caption("Municípios extensos ocupam mais área no mapa, mas isso não significa mais pessoas afetadas. "
               "Clique duas vezes numa área vazia para limpar a seleção; use a rolagem do mouse para aproximar.")
    escolhido = None
    try:
        pts = evento.selection.points if evento else []
        if pts:
            escolhido = pts[0].get("location")
    except AttributeError:
        pass
    if escolhido and escolhido in set(BASE.id_municipio):
        painel_municipio(BASE.loc[BASE.id_municipio == escolhido].iloc[0], "mapa")

    st.markdown("#### Comparação entre regiões e estados")
    e, d = st.columns([1, 1.15], gap="large")
    with e:
        pct = bloco_territorial(f, "mapa")
    with d:
        t_ufs, destaque, titulo = D.ufs_de_contexto(BASE, f)
        st.markdown(f"**Por estado** <span style='color:{P.TINTA_2};font-size:.88rem'>· {titulo}</span>",
                    unsafe_allow_html=True)
        st.plotly_chart(G.fig_ufs(t_ufs, "%" if pct else "Quantidade", destaque), config={"displayModeBar": False},
                        key="mapa_uf")
        if destaque:
            st.caption("A UF filtrada aparece em negrito, com seta.")
    K.rodape()


# ======================================================================================================
# Página 3 — Explorar municípios
# ======================================================================================================
COLCFG = {
    D.COL_POS_RF: st.column_config.NumberColumn(format="%d", help="Posição nacional pelo modelo preditivo (Random Forest)"),
    D.COL_POS_B2: st.column_config.NumberColumn(format="%d", help="Posição nacional pela densidade atual (B2)"),
    "Densidade fev/2026": st.column_config.NumberColumn(format="%.2f", help="Acessos por 100 domicílios (observado)"),
    "Ponto de atenção nos dados": st.column_config.CheckboxColumn(help=D.TEXTO_ALERTA),
    "Tendência (por ano)": st.column_config.NumberColumn(format="%+.2f", help="Variação anual média nos 36 meses até fev/2026"),
    D.COL_INDICE: st.column_config.NumberColumn(format="%.4f", help="Valor técnico de ordenação. Não é probabilidade."),
}
_CODIGO_POR_ROTULO = {v: k for k, v in D.ROTULO_PRIORIDADE.items()}


def com_cor_prioridade(df: pd.DataFrame, coluna: str = "Prioridade"):
    """Fundo claro (texto escuro) na coluna de prioridade, nas cores da paleta central."""
    def _cor(v):
        return f"background-color: {P.FUNDO_TABELA.get(_CODIGO_POR_ROTULO.get(v), P.FUNDO)}; color: {P.TINTA}"
    return df.style.map(_cor, subset=[coluna]) if coluna in df.columns else df


ORDEM_RANKING = ["Município", "UF", "Prioridade", "Densidade fev/2026", "Tendência (por ano)",
                 "Ponto de atenção nos dados", D.COL_POS_RF, D.COL_POS_B2, "Código IBGE"]


def pagina_explorar():
    K.cabecalho_interno("Explorar municípios",
                        "Busque um município, veja sua posição nas ordenações nacionais e compare os critérios de "
                        "priorização.")
    f = FILTROS
    rec = D.aplicar_filtros(BASE, f)
    if f.ativos:
        K.faixa_recorte(f.descricao(), len(rec))
    abas = st.tabs(["Ranking e busca", "Detalhes do município", "Comparação entre métodos", "Municípios divergentes"])

    with abas[0]:
        c = st.columns([2, 1.4, 0.9])
        texto = c[0].text_input("Buscar por nome ou código IBGE", key="rk_busca", placeholder="ex.: Itati ou 4310652")
        criterio = c[1].segmented_control("Ordenar por", [D.NOME_METODO["rf"], D.NOME_METODO["b2"]],
                                          default=D.NOME_METODO["rf"], key="rk_metodo") or D.NOME_METODO["rf"]
        n = c[2].selectbox("Mostrar", [10, 25, 50, 100, "Todos"], index=1, key="rk_n")
        tab = D.ranking(D.buscar(rec, texto), "rf" if criterio == D.NOME_METODO["rf"] else "b2")
        if tab.empty:
            K.sem_resultados()
        else:
            mostrar = tab if n == "Todos" else tab.head(int(n))
            exib = D.tabela_exibicao(mostrar)
            ordem = ORDEM_RANKING if criterio == D.NOME_METODO["rf"] else \
                [c for c in ORDEM_RANKING if c != D.COL_POS_B2][:6] + [D.COL_POS_B2, D.COL_POS_RF, "Código IBGE"]
            sel = st.dataframe(com_cor_prioridade(exib), hide_index=True, column_config=COLCFG, column_order=ordem,
                               height=min(38 + 35 * len(mostrar), 720), on_select="rerun",
                               selection_mode="single-row", key="rk_tabela")
            linhas = sel.selection.rows if sel and hasattr(sel, "selection") else []
            if linhas:
                st.session_state["rk_mun"] = mostrar.id_municipio.iloc[linhas[0]]
                st.success(f"{mostrar.municipio.iloc[linhas[0]]} selecionado — veja a aba **Detalhes do município**.",
                           icon=":material/check_circle:")
            st.caption(f"{K.fmt(len(mostrar))} de {K.fmt(len(tab))} municípios, em ordem de posição nacional "
                       f"({criterio.lower()}). Com filtros, a primeira linha pode não ser a posição 1. "
                       "Selecione uma linha para ver os detalhes.")
            st.download_button("Baixar a tabela filtrada (CSV)",
                               D.tabela_exibicao(tab).to_csv(index=False).encode("utf-8-sig"),
                               file_name="conectamap_municipios_2029.csv", mime="text/csv", icon=":material/download:")

    with abas[1]:
        opcoes = D.ranking(rec if not rec.empty else BASE, "rf")
        rotulos = dict(zip(opcoes.id_municipio, opcoes.municipio + " — " + opcoes.sigla_uf + " (" + opcoes.id_municipio + ")"))
        if st.session_state.get("rk_mun") not in rotulos:
            st.session_state.pop("rk_mun", None)
        sel = st.selectbox("Município (digite para buscar)", list(rotulos), format_func=rotulos.get, key="rk_mun",
                           help="Lista do recorte filtrado, em ordem de posição nacional pelo modelo preditivo")
        if sel:
            painel_municipio(BASE.loc[BASE.id_municipio == sel].iloc[0], "explorar")

    with abas[2]:
        st.markdown("O projeto utiliza dois critérios de priorização: **um modelo preditivo** que considera o histórico "
                    "municipal (Random Forest) e **uma referência baseada na densidade atual** (B2). "
                    + D.TEXTO_DIVERGENCIA)
        c = st.columns(5)
        for col, (rot, val, cor) in zip(c, [
                ("25% primeiros — modelo preditivo", K.fmt(NAC["top25_rf"]), G.PETROLEO),
                ("25% primeiros — densidade atual", K.fmt(NAC["top25_b2"]), G.PETROLEO),
                ("Nos dois critérios", K.fmt(NAC["top25_ambos"]), G.COR_PRIORIDADE["alta_consenso"]),
                ("Só pelo modelo / só pela densidade", f"{NAC['so_rf']} / {NAC['so_b2']}",
                 G.COR_PRIORIDADE["alta_divergente_revisar"]),
                ("Concordância entre os rankings", K.pct(NAC["concordancia_top25"]), P.INTERMEDIARIA)]):
            with col:
                K.card(rot, val, "Brasil", cor,
                       D.TEXTO_CONCORDANCIA.format(pct=val) if rot.startswith("Concordância") else None)
        if f.ativos:
            ir = D.indicadores(rec)
            st.caption(f"No recorte: {K.fmt(ir['top25_ambos'])} nos dois critérios, {K.fmt(ir['so_rf'])} só pelo modelo "
                       f"e {K.fmt(ir['so_b2'])} só pela densidade atual. Os 25% primeiros são sempre os nacionais.")
        st.plotly_chart(G.fig_dispersao(BASE, set(rec.id_municipio) if f.ativos else None),
                        config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}, key="cmp_disp")
        st.caption("Cada ponto é um município. Perto da diagonal, os dois critérios dão posições parecidas. Triângulos e "
                   "losangos marcam os municípios que só um critério colocou entre os 25% primeiros.")
        st.markdown("##### Maiores diferenças de posição")
        dif = rec[rec.concordancia_top25 != "nenhum"].assign(_d=lambda d: d.diferenca_posicoes.abs()) \
            .sort_values("_d", ascending=False).head(15)
        if dif.empty:
            K.sem_resultados()
        else:
            st.dataframe(D.tabela_exibicao(dif).assign(**{"Diferença de posições": dif.diferenca_posicoes.values}),
                         hide_index=True, column_config=COLCFG,
                         column_order=["Município", "UF", D.COL_POS_RF, D.COL_POS_B2, "Diferença de posições",
                                       "Critérios no top 25%", "Densidade fev/2026", "Tendência (por ano)",
                                       "Ponto de atenção nos dados"])
            st.caption("Diferença positiva: o modelo coloca o município mais à frente do que a densidade atual.")
        with st.expander("Leitura técnica: algum critério é melhor?"):
            te = carregar_tecnico()["teste"]
            st.markdown(
                f"- No teste reservado (origem 2023, alvo 2026), a PR-AUC foi **{K.dec(te['pr_auc_rf'], 3)}** para o Random "
                f"Forest e **{K.dec(te['pr_auc_b2'], 3)}** para o B2 (densidade atual). A diferença, "
                f"{K.sinal(te['dif'])}, tem intervalo de 95% de {K.sinal(te['ic_inf'])} a {K.sinal(te['ic_sup'])}." +
                "\n- **O Random Forest não demonstrou vantagem robusta sobre o B2.** Uma diferença de posição não "
                "comprova que um critério prevê melhor; por isso a prioridade final usa os dois.")

    with abas[3]:
        div = rec[rec.classificacao_prioridade == "alta_divergente_revisar"]
        tot = int((BASE.classificacao_prioridade == "alta_divergente_revisar").sum())
        st.markdown(f"##### Municípios com classificação divergente entre os métodos — {K.fmt(len(div))}"
                    + (f" no recorte (Brasil: {K.fmt(tot)})" if f.ativos else ""))
        st.markdown(D.TEXTO_DIVERGENCIA + " Muitas dessas localidades tiveram mudança brusca recente na série — isso "
                    "não significa, por si só, problema nos dados.")
        escolha = st.segmented_control("Mostrar", ["Todos", "Só pelo modelo preditivo", "Só pela densidade atual"],
                                       default="Todos", key="div_tipo") or "Todos"
        if escolha != "Todos":
            div = div[div.concordancia_top25 == ("so_rf" if escolha == "Só pelo modelo preditivo" else "so_b2")]
        c = st.columns(3)
        with c[0]:
            K.card("Só pelo modelo preditivo", K.fmt((div.concordancia_top25 == "so_rf").sum()), None,
                   G.COR_PRIORIDADE["alta_divergente_revisar"])
        with c[1]:
            K.card("Só pela densidade atual", K.fmt((div.concordancia_top25 == "so_b2").sum()), None,
                   G.COR_PRIORIDADE["alta_divergente_revisar"])
        with c[2]:
            K.card("Destes, com ponto de atenção nos dados", K.fmt(div.alerta_qualidade.sum()), "mostrado à parte",
                   G.COR_ALERTA[True])
        if div.empty:
            K.sem_resultados()
        else:
            t = div.sort_values(["concordancia_top25", "posicao_rf"])
            st.dataframe(pd.DataFrame({
                "Município": t.municipio, "UF": t.sigla_uf, "Selecionado": t.concordancia_top25.map(D.ROTULO_CONCORDANCIA),
                "Ponto de atenção nos dados": t.alerta_qualidade,
                "Densidade fev/2026": t.densidade_fev2026, "Tendência (por ano)": t.tendencia_36,
                D.COL_POS_RF: t.posicao_rf, D.COL_POS_B2: t.posicao_b2,
                "Variações incomuns (36 meses)": t.suspeitas_36m,
                "Indicadores fora do padrão": t.n_features_fora_faixa_treino}), hide_index=True, column_config=COLCFG)
    K.rodape()


# ======================================================================================================
# Página 4 — Sobre o ConectaMap
# ======================================================================================================
def pagina_sobre():
    K.cabecalho_interno("Sobre o ConectaMap", "O que o projeto faz, de onde vêm os dados e como ler os resultados.")
    tec = carregar_tecnico()
    te = tec["teste"]

    st.markdown("#### O que é o ConectaMap?")
    st.markdown("O ConectaMap ajuda a identificar municípios brasileiros que merecem atenção no planejamento da "
                "conectividade de banda larga fixa. A partir do histórico de cada município até fevereiro de 2026, ele "
                "ordena os 5.570 municípios pelo risco relativo de estarem abaixo de um patamar de referência em "
                "fevereiro de 2029.")

    st.markdown("#### De onde vêm os dados?")
    a, b = st.columns(2)
    with a, st.container(border=True, height="stretch"):
        st.markdown("**Anatel**")
        st.markdown("Série histórica da densidade de acessos de banda larga fixa por 100 domicílios, por município. "
                    "É a única fonte usada pelo modelo preditivo principal.")
    with b, st.container(border=True, height="stretch"):
        st.markdown("**IBGE**")
        st.markdown("Identificação dos municípios, dados do Censo 2022 tratados no pré-processamento e limites "
                    "municipais usados no mapa (malha de 2022). **As variáveis do Censo 2022 não entraram no "
                    "treinamento do modelo preditivo principal.**")

    st.markdown("#### Como as prioridades são definidas?")
    passos = [
        ("1. Data de referência", "Data de referência adotada pelo projeto: fevereiro de 2026."),
        ("2. Horizonte", "36 meses: a pergunta é sobre fevereiro de 2029."),
        ("3. Patamar de referência", f"{PATAMAR} acessos por 100 domicílios — o valor que separava os 25% de menor "
                                     "densidade do país em fev/2026 (P25), mantido fixo."),
        ("4. Dois critérios", "Um modelo preditivo (Random Forest) que considera o histórico municipal e uma referência "
                              "pela densidade atual (B2)."),
        ("5. Categorias", "Alta prioridade quando os dois critérios colocam o município entre os 25% primeiros; "
                          "classificação divergente quando só um coloca."),
    ]
    cols = st.columns(len(passos))
    for col, (t, txt) in zip(cols, passos):
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"**{t}**")
            st.markdown(f"<span style='font-size:.9rem'>{txt}</span>", unsafe_allow_html=True)

    st.markdown("#### Como interpretar os resultados?")
    e, d = st.columns([1.3, 1], gap="large")
    with e:
        K.legenda_prioridades(G.COR_PRIORIDADE, D.ROTULO_PRIORIDADE, D.DESCRICAO_PRIORIDADE)
    with d:
        K.atencao("Ponto de atenção nos dados")
        st.markdown(f"<span style='font-size:.9rem'>{D.TEXTO_ALERTA} É uma camada separada: não muda a prioridade."
                    "</span>", unsafe_allow_html=True)
    st.caption("O indicador é densidade de acessos de banda larga fixa — não mede velocidade, qualidade do serviço, "
               "cobertura nem o número de pessoas sem internet.")

    st.markdown("#### Qualidade dos dados")
    f = FILTROS
    rec = D.aplicar_filtros(BASE, f, usar_prioridade=False)
    if f.ativos:
        K.faixa_recorte(D.Filtros(f.regioes, f.ufs, [], f.alerta).descricao() + " (sem o filtro de prioridade)", len(rec))
    if rec.empty:
        K.sem_resultados()
    else:
        alta = rec.classificacao_prioridade.isin(["alta_consenso", "alta_divergente_revisar"])
        div = rec.classificacao_prioridade == "alta_divergente_revisar"
        c = st.columns(4)
        for col, (rot, val, nac) in zip(c, [
                ("Municípios com ponto de atenção", K.fmt(rec.alerta_qualidade.sum()), f"de {K.fmt(len(rec))} municípios"),
                ("Entre os prioritários", K.fmt((rec.alerta_qualidade & alta).sum()), f"de {K.fmt(alta.sum())} prioritários"),
                ("Entre os de classificação divergente", K.fmt((rec.alerta_qualidade & div).sum()), f"de {K.fmt(div.sum())}"),
                ("Com variação incomum (36 meses)", K.fmt((rec.suspeitas_36m > 0).sum()), "um dos tipos de ponto de atenção")]):
            with col:
                K.card(rot, val, nac, G.COR_ALERTA[True])
        e, d = st.columns([1, 1], gap="large")
        with e:
            st.markdown("**Prioridade × pontos de atenção** — são dimensões independentes")
            tab = D.prioridade_x_alerta(rec)
            st.plotly_chart(G.fig_prioridade_alerta(tab), config={"displayModeBar": False}, key="q_pxa")
        with d:
            por = st.segmented_control("Ver por", ["Região", "Estado"], default="Região", key="q_por") or "Região"
            if por == "Região":
                t = D.comparacao_territorial(BASE, f)
                st.plotly_chart(G.fig_alerta_territorio(t, ordenar=False), config={"displayModeBar": False},
                                key="q_terr")
            else:
                t, _, titulo = D.ufs_de_contexto(BASE, f)
                st.plotly_chart(G.fig_alerta_territorio(t), config={"displayModeBar": False}, key="q_terr")
            st.caption("Percentual em relação ao total de municípios de cada região ou estado; os filtros de prioridade e "
                       "de pontos de atenção não alteram estes valores.")
        with st.expander("Tabela dos indicadores de qualidade"):
            exib = tab.assign(**{"% com alerta": tab["% com alerta"].map(lambda v: "—" if pd.isna(v) else K.pct(v))})
            st.dataframe(exib.rename(index=D.ROTULO_PRIORIDADE).rename_axis("Prioridade")
                         .rename(columns={"Sem alerta": "Sem ponto de atenção", "Com alerta": "Com ponto de atenção",
                                          "% com alerta": "% com ponto de atenção"}))
            st.dataframe(D.indicadores_qualidade(rec).rename(columns={
                "indicador": "Indicador", "municipios": "Municípios", "em_prioridade_alta": "Entre os prioritários"}),
                hide_index=True)
            st.caption("São sinais estatísticos para conferência da série, não erros comprovados da Anatel.")

    st.markdown("#### Detalhes técnicos e limitações")
    pr = tec["pr_auc_validacao"]
    with st.expander("PR-AUC — capacidade de ordenar"):
        st.markdown("A PR-AUC mede o quanto uma ordenação coloca no topo os municípios que de fato ficaram abaixo do "
                    "patamar. Quanto maior, melhor; o mínimo de referência é a proporção de casos (cerca de 13% a 15%).")
        st.dataframe(pd.DataFrame({"Origem avaliada": [o[:4] for o in pr.index],
                                   "Random Forest": [K.dec(v, 3) for v in pr["random_forest"]],
                                   "B2 — densidade atual": [K.dec(v, 3) for v in pr["B2_densidade_atual"]]}),
                     hide_index=True)
    with st.expander("Brier score — qualidade das probabilidades"):
        st.markdown(f"No teste reservado, o Brier do Random Forest foi {K.dec(tec['brier_teste_rf'], 3)}, contra "
                    f"{K.dec(tec['brier_teste_b0'], 3)} da taxa base (menor é melhor). Mesmo assim, as probabilidades não "
                    "passaram no critério de calibração (abaixo) e não são divulgadas.")
    with st.expander("Validação temporal"):
        st.markdown("O modelo foi escolhido em quatro rodadas (origens 2019 a 2022), sempre treinando só com origens "
                    "anteriores cujo resultado já era conhecido. Os mesmos municípios se repetem em todas as origens, "
                    "então as rodadas não são independentes.")
    with st.expander("Teste reservado (2023)"):
        st.markdown(f"Avaliado uma única vez, depois de congelado o protocolo: PR-AUC de {K.dec(te['pr_auc_rf'], 3)} "
                    f"(Random Forest) contra {K.dec(te['pr_auc_b2'], 3)} (B2). Diferença de {K.sinal(te['dif'])}, com "
                    f"intervalo de 95% de {K.sinal(te['ic_inf'])} a {K.sinal(te['ic_sup'])}: **sem vantagem robusta do "
                    "Random Forest**.")
    with st.expander("Calibração — por que não há probabilidades"):
        esc = tec["escore_medio"]
        v = esc[esc.uso == "validacao"]
        st.markdown("Pela regra fixada antes da análise, nenhuma versão (bruta ou recalibrada) atingiu os critérios nas "
                    "quatro rodadas. O índice médio do modelo ficou abaixo da frequência observada, e a diferença "
                    "cresceu com o tempo; por isso o dashboard usa posições e categorias relativas.")
        st.dataframe(pd.DataFrame({"Origem": [r[-7:-3] for r in v.rodada],
                                   "Frequência observada": [K.dec(x, 3) for x in v.prevalencia],
                                   "Índice médio do modelo": [K.dec(x, 3) for x in v.prob_media]}), hide_index=True)
        st.dataframe(tec["decisao"].rename(columns={"versao": "Versão", "calibracao_geral": "Calibração geral",
                                                    "inclinacao": "Inclinação", "decil_superior": "Faixa de maior risco",
                                                    "brier_vs_B2p": "Brier ≤ referência", "aprovada": "Aprovada"}),
                     hide_index=True)
    with st.expander("Mudança de distribuição"):
        m = tec["densidade_mediana"]
        st.markdown(f"A densidade mediana passou de {K.dec(m['2014'], 1)} (2014) para {K.dec(m['2023'], 1)} (2023) e "
                    f"{K.dec(m['2026'], 1)} (2026); {K.pct(tec['fora_faixa_2026'])} dos municípios de 2026 estão fora da "
                    "faixa vista no treinamento. Como o modelo aprendeu com níveis mais baixos, a extrapolação para 2029 "
                    "tem confiabilidade limitada.")
    with st.expander("Limitações"):
        st.markdown(
            "- Horizonte de três anos: mudanças entre 2026 e 2029 (tecnologia, políticas, coleta da Anatel) podem alterar o quadro.\n"
            "- Poucas coortes independentes: 10 origens anuais com os mesmos municípios.\n"
            "- Teste reservado de uma única origem (2023).\n"
            "- Densidades de 2026 acima da faixa do treinamento.\n"
            "- Sem superioridade robusta do Random Forest sobre o B2.\n"
            "- Sem probabilidades calibradas: o índice do modelo serve só para ordenar.\n"
            "- Observações estatisticamente incomuns podem afetar a posição de alguns municípios.\n"
            "- A classificação é uma priorização relativa, não uma garantia sobre 2029.")
    with st.expander("Fontes, integridade e dicionário de dados"):
        meta = metadados_geo()
        st.markdown(
            "- **Base de 2029:** `etapa2/saidas/previsao2029/previsoes_2029_municipios.csv` (Etapa 2, aprovada).\n"
            f"- **Malha municipal:** {meta['fonte']}; ano {meta['ano_malha']}; {meta['sistema_referencia']}.\n"
            f"- **Preparação da malha:** {meta['preparacao']}. {meta.get('excluidos', '')}")
        st.dataframe(VALIDACAO.rename(columns={"verificacao": "Verificação da base", "ok": "OK"}), hide_index=True)
        st.dataframe(carregar_dicionario(), hide_index=True)
    K.rodape()


# ======================================================================================================
# Navegação
# ======================================================================================================
# a página inicial é a Visão geral; a variável de ambiente só é usada pelos testes automatizados
INICIAL = os.environ.get("CONECTAMAP_PAGINA_INICIAL", "visao-geral")
PAGINAS = {caminho: st.Page(funcao, title=titulo, icon=icone, url_path=caminho, default=(caminho == INICIAL))
           for funcao, titulo, icone, caminho in [
               (pagina_visao_geral, "Visão geral", ":material/space_dashboard:", "visao-geral"),
               (pagina_mapa, "Mapa de prioridades", ":material/map:", "mapa"),
               (pagina_explorar, "Explorar municípios", ":material/travel_explore:", "explorar"),
               (pagina_sobre, "Sobre o ConectaMap", ":material/info:", "sobre")]}
PAG_MAPA, PAG_SOBRE = PAGINAS["mapa"], PAGINAS["sobre"]
navegacao = st.navigation(list(PAGINAS.values()))
FILTROS = filtros_globais()
navegacao.run()
