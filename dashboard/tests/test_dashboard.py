"""Testes do dashboard (Etapa 3): integridade da base, indicadores, filtros, mapa e execução das páginas.

Execução, na pasta do projeto:  python -m pytest -q dashboard/tests
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DASH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASH))

import dados as D  # noqa: E402
import graficos as G  # noqa: E402


@pytest.fixture(scope="module")
def base():
    return D.carregar_base()


@pytest.fixture(scope="module")
def geo():
    return D.carregar_geometrias(D.ARQ_GEO_MUN), D.carregar_geometrias(D.ARQ_GEO_UF)


# ---------------------------------------------------------------------------------------------------
# Integridade da base
# ---------------------------------------------------------------------------------------------------
def test_base_integra_e_identica_a_aprovada(base):
    v = D.validar_base(base)
    assert v.ok.all(), v[~v.ok]
    assert D.sha256(D.ARQ_BASE) == D.SHA256_APROVADO[D.ARQ_BASE.name]
    assert D.sha256(D.ARQ_DICIONARIO) == D.SHA256_APROVADO[D.ARQ_DICIONARIO.name]
    assert pd.api.types.is_string_dtype(base.id_municipio) and base.id_municipio.str.len().eq(7).all()
    assert base.id_municipio.is_unique and len(base) == 5570


def test_validacao_detecta_problemas(base):
    ruim = base.copy()
    ruim.loc[ruim.index[0], "posicao_rf"] = 2                       # posição repetida
    ruim.loc[ruim.index[1], "classificacao_prioridade"] = "outra"   # categoria desconhecida
    v = D.validar_base(ruim, caminho=None).set_index("verificacao").ok
    assert not v["Rankings RF e B2 completos e sem posições repetidas"]
    assert not v["Categorias reconhecidas"]
    assert not D.validar_base(pd.concat([base, base.head(1)]), caminho=None).ok.all()


def test_regiao_derivada_confere_com_uf(base):
    assert (base.regiao == base.sigla_uf.map(D.REGIAO_POR_UF)).all()
    assert base.regiao.value_counts().to_dict() == {"Nordeste": 1794, "Sudeste": 1668, "Sul": 1191,
                                                    "Centro-Oeste": 467, "Norte": 450}
    assert len(D.REGIAO_POR_UF) == 27 and set(base.sigla_uf) == set(D.REGIAO_POR_UF)


def test_dicionario_cobre_todas_as_colunas(base):
    dic = D.carregar_dicionario()
    assert dic.coluna.tolist() == D.COLUNAS
    assert "NÃO CALIBRADO" in dic.set_index("coluna").loc["escore_rf", "grupo"]


# ---------------------------------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------------------------------
def test_indicadores_nacionais(base):
    i = D.indicadores(base)
    assert (i["municipios"], i["alta_consenso"], i["alta_divergente_revisar"], i["intermediaria"], i["menor"]) == \
        (5570, 1362, 62, 1750, 2396)
    assert (i["alerta"], i["abaixo_limiar"]) == (859, 1393)
    assert (i["top25_rf"], i["top25_b2"], i["top25_ambos"], i["so_rf"], i["so_b2"]) == (1393, 1393, 1362, 31, 31)
    assert round(i["concordancia_top25"], 3) == 0.978


def test_indicadores_de_recorte_vazio_nao_quebram(base):
    i = D.indicadores(base.iloc[0:0])
    assert i["municipios"] == 0 and i["alta_consenso"] == 0 and np.isnan(i["concordancia_top25"])


# ---------------------------------------------------------------------------------------------------
# Filtros: selecionam linhas e preservam os rankings nacionais
# ---------------------------------------------------------------------------------------------------
CASOS = [D.Filtros(ufs=["SP"]), D.Filtros(regioes=["Norte"]), D.Filtros(regioes=["Sul", "Sudeste"], ufs=["RS"]),
         D.Filtros(prioridades=["alta_divergente_revisar"]), D.Filtros(alerta="Com alerta"),
         D.Filtros(regioes=["Nordeste"], prioridades=["alta_consenso"], alerta="Sem alerta"),
         D.Filtros(ufs=["DF"], prioridades=["alta_consenso"])]


@pytest.mark.parametrize("f", CASOS, ids=lambda f: f.descricao())
def test_filtros_preservam_rankings(base, f):
    rec = D.aplicar_filtros(base, f)
    ref = base.set_index("id_municipio")
    cols = ["posicao_rf", "posicao_b2", "categoria_rf", "categoria_b2", "classificacao_prioridade", "escore_rf"]
    pd.testing.assert_frame_equal(rec.set_index("id_municipio")[cols], ref.loc[rec.id_municipio, cols])
    if f.ufs:
        assert set(rec.sigla_uf) <= set(f.ufs)
    if f.regioes:
        assert set(rec.regiao) <= set(f.regioes)
    if f.prioridades:
        assert set(rec.classificacao_prioridade) <= set(f.prioridades)
    if f.alerta != "Todos":
        assert rec.alerta_qualidade.eq(f.alerta == "Com alerta").all()
    i = D.indicadores(rec)
    assert sum(i[p] for p in D.PRIORIDADES) == len(rec)


def test_filtro_sem_resultado_e_contagens(base):
    assert D.aplicar_filtros(base, D.Filtros(ufs=["DF"], prioridades=["alta_consenso"])).empty
    assert len(D.aplicar_filtros(base, D.Filtros(ufs=["SP"]))) == 645
    assert len(D.aplicar_filtros(base, D.Filtros(alerta="Com alerta"))) == 859
    assert len(D.aplicar_filtros(base, D.Filtros(alerta="Sem alerta"))) == 5570 - 859
    sem_prioridade = D.aplicar_filtros(base, D.Filtros(ufs=["PI"], prioridades=["menor"]), usar_prioridade=False)
    assert len(sem_prioridade) == 224 and sem_prioridade.classificacao_prioridade.nunique() == 4


def test_top25_nao_e_recalculado_no_recorte(base):
    rec = D.aplicar_filtros(base, D.Filtros(ufs=["SC"]))
    i = D.indicadores(rec)
    assert i["top25_rf"] == (rec.categoria_rf == "maior").sum() < np.ceil(0.25 * len(rec))   # SC tem poucos no top nacional


def test_ranking_usa_posicao_nacional_e_nao_o_escore(base):
    r = D.ranking(D.aplicar_filtros(base, D.Filtros(ufs=["BA"])), "rf")
    assert r.posicao_rf.is_monotonic_increasing and r.posicao_rf.iloc[0] > 1
    assert D.ranking(base, "rf", 10).posicao_rf.tolist() == list(range(1, 11))
    assert D.ranking(base, "b2", 5).posicao_b2.tolist() == list(range(1, 6))
    empatados = base[base.escore_rf.duplicated(keep=False)]
    if len(empatados):
        assert D.ranking(empatados, "rf").posicao_rf.is_unique


def test_busca_por_nome_e_codigo(base):
    r = D.buscar(base, "itati")
    assert "Itati" in r.municipio.tolist() and r.municipio.str.lower().str.contains("itati").all()
    assert D.buscar(base, "boqueirao do piaui").municipio.tolist() == ["Boqueirão do Piauí"]
    codigo = base.loc[base.municipio == "Itati", "id_municipio"].iloc[0]
    assert D.buscar(base, codigo).id_municipio.tolist() == [codigo]
    assert D.buscar(base, "43").id_municipio.str.startswith("43").all()
    assert D.buscar(base, "  ").equals(base) and D.buscar(base, "zzzz").empty


def test_ufs_das_regioes():
    assert D.ufs_das_regioes(["Sul"]) == ["PR", "RS", "SC"]
    assert len(D.ufs_das_regioes([])) == 27


def test_tabela_exportada_sem_probabilidade(base):
    t = D.tabela_exibicao(D.ranking(base, "rf", 25))
    assert t[D.COL_POS_RF].tolist() == list(range(1, 26))
    assert not any("probabil" in c.lower() or "chance" in c.lower() for c in t.columns if c != D.COL_INDICE)
    assert t.columns[-1] == D.COL_INDICE and "não é probabilidade" in D.COL_INDICE
    csv = t.to_csv(index=False)
    assert "2708204" in csv                                          # código IBGE preservado como texto


def test_resumos_territoriais(base):
    r = D.resumo_territorial(base, "regiao")
    assert r.municipios.sum() == 5570 and r.alta_consenso.sum() == 1362 and r.alta_divergente_revisar.sum() == 62
    assert list(r.index) == D.ORDEM_REGIOES
    u = D.resumo_territorial(base, "sigla_uf")
    assert len(u) == 27 and u.com_alerta.sum() == 859
    pxa = D.prioridade_x_alerta(base)
    assert pxa["Com alerta"].sum() == 859 and pxa.loc["alta_consenso", "Total"] == 1362
    q = D.indicadores_qualidade(base).set_index("indicador")
    assert q.loc["Algum ponto de atenção nos dados", "municipios"] == 859


def test_resultados_do_teste_lidos_dos_arquivos():
    t = D.resultados_teste()
    assert round(t["pr_auc_rf"], 3) == 0.521 and round(t["pr_auc_b2"], 3) == 0.516
    assert t["ic_inf"] < 0 < t["ic_sup"]


# ---------------------------------------------------------------------------------------------------
# Mapa
# ---------------------------------------------------------------------------------------------------
def test_geometrias_conferem_com_a_base(base, geo):
    v = D.validar_geometrias(geo[0], base)
    assert v.ok.all(), v[~v.ok]
    assert {f["id"] for f in geo[1]["features"]} == set(base.sigla_uf)


CAPITAIS = [("3550308", -46.63, -23.55), ("5300108", -47.88, -15.79), ("1302603", -60.02, -3.12),
            ("3106200", -43.94, -19.92), ("5208707", -49.25, -16.68), ("4106902", -49.27, -25.43),
            ("4314902", -51.18, -30.05), ("2211001", -42.80, -5.09)]


@pytest.mark.parametrize("codigo,lon,lat", CAPITAIS)
def test_municipios_bem_posicionados(geo, codigo, lon, lat):
    assert D.ponto_no_municipio(geo[0], codigo, lon, lat)


def test_ponto_fora_do_municipio(geo):
    assert not D.ponto_no_municipio(geo[0], "3550308", -43.94, -19.92)     # BH não está em São Paulo


def test_metadados_da_malha():
    import json
    m = json.loads(D.ARQ_GEO_META.read_text(encoding="utf-8"))
    assert m["municipios"] == 5570 and m["ano_malha"] == 2022 and "IBGE" in m["fonte"]
    for nome, h in m["sha256_saidas"].items():
        assert D.sha256(D.DIR_ASSETS / nome) == h


# ---------------------------------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------------------------------
def test_mapa_cores_e_legenda(base, geo):
    rec = D.aplicar_filtros(base, D.Filtros(ufs=["PI"]))
    fig = G.fig_mapa(rec, set(rec.id_municipio), *geo, modo=G.MODO_PRIORIDADE)
    nomes = [t.name for t in fig.data if t.showlegend]
    assert nomes == ["Alta prioridade (109)", "Métodos divergem (5)", "Intermediária (86)", "Menor prioridade (24)"]
    assert not any("_" in n for n in nomes)                                     # nenhum código interno na legenda
    cores = {t.name.rsplit(" (", 1)[0]: t.colorscale[0][1] for t in fig.data if t.showlegend}
    assert cores["Alta prioridade"] == "#1E5A8A"
    divergem = [t for t in fig.data if t.showlegend and t.name.startswith("Métodos divergem")][0]
    assert divergem.marker.line.width > 1                                       # 2ª codificação além da cor
    fig_alerta = G.fig_mapa(rec, set(rec.id_municipio), *geo, modo=G.MODO_ALERTA)
    assert {t.colorscale[0][1] for t in fig_alerta.data if t.showlegend} <= set(G.COR_ALERTA.values())
    filtrado = D.aplicar_filtros(base, D.Filtros(ufs=["PI"], prioridades=["alta_divergente_revisar"]))
    fig_f = G.fig_mapa(rec, set(filtrado.id_municipio), *geo)
    assert fig_f.data[0].name == "Fora do recorte" and len(fig_f.data[0].locations) == 224 - 5


def test_graficos_constroem_em_varios_recortes(base):
    for f in CASOS:
        rec = D.aplicar_filtros(base, f, usar_prioridade=False)
        if rec.empty:
            continue
        G.fig_prioridades(rec)
        G.fig_territorial(D.comparacao_territorial(base, f), "Quantidade")
        G.fig_territorial(D.comparacao_territorial(base, f), "%")
        t_ufs, dest, _ = D.ufs_de_contexto(base, f)
        G.fig_ufs(t_ufs, "%", dest)
        G.fig_alerta_territorio(D.comparacao_territorial(base, f), ordenar=False)
        G.fig_prioridade_alerta(D.prioridade_x_alerta(rec))
    G.fig_dispersao(base, set(base.id_municipio[:100]))
    G.fig_densidade_limiar(base.iloc[0], float(base.densidade_fev2026.median()))


def test_enquadramento_do_mapa(geo):
    c, z = G.enquadramento(geo[0], 640)
    assert -60 < c["lon"] < -45 and -20 < c["lat"] < -5 and 2.5 < z < 4.5
    c2, z2 = G.enquadramento(D.subconjunto_geo(geo[0], ["5300108"]), 560)
    assert z2 > z


# ---------------------------------------------------------------------------------------------------
# Aplicação: todas as páginas executam sem exceção (inclusive com filtros e recorte vazio)
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pagina", ["visao-geral", "mapa", "explorar", "sobre"])
def test_paginas_executam(pagina, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CONECTAMAP_PAGINA_INICIAL", pagina)
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.multiselect(key="f_ufs").set_value(["PI"]).run()
    assert not at.exception
    at.sidebar.multiselect(key="f_prioridades").set_value(["alta_consenso"]).run()
    assert not at.exception
    at.sidebar.multiselect(key="f_ufs").set_value(["DF"]).run()             # recorte vazio
    assert not at.exception
    at.sidebar.button[0].click().run()                                       # limpar filtros
    assert not at.exception
    assert at.session_state["f_ufs"] == [] and at.session_state["f_prioridades"] == []


def test_cartoes_da_visao_geral(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CONECTAMAP_PAGINA_INICIAL", "visao-geral")
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=120)
    at.run()
    html = " ".join(m.value for m in at.markdown)
    for valor in ["5.570", "1.362", "62", "859"]:
        assert f'class="val">{valor}<' in html, valor
    for valor in ["1.393", "97,8%"]:                                            # indicadores secundários
        assert f"<b>{valor}</b>" in html, valor
    assert "Onde a conectividade precisa de mais atenção?" in html
    at.sidebar.multiselect(key="f_ufs").set_value(["PI"]).run()
    html = " ".join(m.value for m in at.markdown)
    assert "Recorte filtrado" in html and 'class="val">224<' in html and "Brasil: 5.570" in html


# ---------------------------------------------------------------------------------------------------
# Identidade visual e linguagem (revisão de UX)
# ---------------------------------------------------------------------------------------------------
def test_paleta_aprovada():
    import paleta as P
    assert G.COR_PRIORIDADE == {"alta_consenso": "#1E5A8A", "alta_divergente_revisar": "#71849E",
                                "intermediaria": "#86BCD8", "menor": "#DFE7ED"}
    assert P.ATENCAO == "#7856A6" and P.PETROLEO == "#17324D" and P.TERRACOTA == "#C96D46"
    assert G.COR_CATEGORIA["maior"] == G.COR_PRIORIDADE["alta_consenso"]          # mesma semântica nos modos avançados
    assert P.TERRACOTA not in set(G.COR_PRIORIDADE.values()) | set(G.COR_CATEGORIA.values()) | set(G.COR_CONCORDANCIA.values())
    # texto escuro legível sobre os fundos claros da paleta (WCAG AA ≥ 4,5)
    for fundo in [P.MENOR, P.INTERMEDIARIA, P.FUNDO_LATERAL, P.ATIVO_LATERAL, *P.FUNDO_TABELA.values()]:
        assert P.contraste(P.TINTA, fundo) >= 4.5, fundo
    assert P.contraste(P.PETROLEO, P.ATIVO_LATERAL) >= 4.5                       # página ativa na barra lateral


def test_paleta_centralizada():
    import re
    for arq in ("app.py", "graficos.py"):
        codigo = (DASH / arq).read_text(encoding="utf-8")
        assert not re.findall(r"#[0-9A-Fa-f]{6}", codigo), arq                    # nenhuma cor fora de paleta.py


def test_marca_conecta_map():
    import componentes as K
    m = K.marca("g")
    assert '<span class="c">Conecta</span><span class="m">Map</span>' in m          # juntas, sem espaço
    assert ".cm-marca .c {color: #17324D;}" in K.CSS and ".cm-marca .m {color: #C96D46;}" in K.CSS
    for arq in ("marca_conectamap.svg", "icone_conectamap.svg"):
        svg = (DASH / "assets" / arq).read_text(encoding="utf-8")
        assert "#17324D" in svg and "#C96D46" in svg


def test_rotulos_amigaveis():
    assert D.ROTULO_PRIORIDADE["alta_consenso"] == "Alta prioridade — métodos concordam"
    assert D.ROTULO_PRIORIDADE["alta_divergente_revisar"] == "Métodos divergem — analisar"
    assert set(D.ROTULO_PRIORIDADE) == set(D.PRIORIDADES)                       # códigos originais preservados
    assert G.MODOS_PRINCIPAIS == ["Prioridade de acompanhamento", "Pontos de atenção nos dados"]
    assert set(G.MODOS_AVANCADOS) == {"Classificação pelo modelo preditivo", "Classificação pela densidade atual"}
    app = (DASH / "app.py").read_text(encoding="utf-8")
    for nome in ("Visão geral", "Mapa de prioridades", "Explorar municípios", "Sobre o ConectaMap",
                 "Data de referência adotada pelo projeto: fevereiro de 2026."):
        assert nome in app
    assert "último mês com dados observados" not in app


def test_modos_avancados_do_mapa(base, geo):
    rec = D.aplicar_filtros(base, D.Filtros(ufs=["SE"]))
    for modo, col in ((G.MODO_RF, "categoria_rf"), (G.MODO_B2, "categoria_b2")):
        fig = G.fig_mapa(rec, set(rec.id_municipio), *geo, modo=modo)
        total = sum(len(t.locations) for t in fig.data if t.showlegend)
        assert total == len(rec)
        n_alta = [len(t.locations) for t in fig.data if t.showlegend and t.name.startswith("Alta (25%")]
        assert n_alta == [int((rec[col] == "maior").sum())] or (rec[col] == "maior").sum() == 0


# ---------------------------------------------------------------------------------------------------
# Comparação territorial (correção da versão 3): territórios completos, denominadores explícitos
# ---------------------------------------------------------------------------------------------------
def _ref(base, mascara):
    t = base[mascara]
    p = t.classificacao_prioridade.isin(["alta_consenso", "alta_divergente_revisar"])
    return int(p.sum()), len(t)


def test_territorial_brasil(base):
    t = D.comparacao_territorial(base, D.Filtros())
    assert list(t.index) == D.ORDEM_REGIOES + ["Brasil"]
    assert (t.loc["Brasil", "prioritarios"], t.loc["Brasil", "municipios"]) == _ref(base, base.regiao.notna())
    assert round(t.loc["Brasil", "pct_prioritarios"], 3) == 0.256
    assert t.loc[D.ORDEM_REGIOES, "municipios"].sum() == 5570


def test_territorial_nordeste(base):
    t = D.comparacao_territorial(base, D.Filtros(regioes=["Nordeste"]))
    assert list(t.index) == ["Nordeste", "Brasil"]
    assert (t.loc["Nordeste", "prioritarios"], t.loc["Nordeste", "municipios"]) == _ref(base, base.regiao == "Nordeste")
    assert round(t.loc["Nordeste", "pct_prioritarios"], 3) == 0.425


def test_territorial_piaui(base):
    t = D.comparacao_territorial(base, D.Filtros(ufs=["PI"]))
    assert list(t.index) == ["Piauí (PI)", "Nordeste", "Brasil"] and list(t.nivel) == ["UF", "Região", "Brasil"]
    assert (t.loc["Piauí (PI)", "prioritarios"], t.loc["Piauí (PI)", "municipios"]) == _ref(base, base.sigla_uf == "PI")
    assert round(t.loc["Piauí (PI)", "pct_prioritarios"], 3) == 0.509
    assert round(t.loc["Nordeste", "pct_prioritarios"], 3) == 0.425          # a região NÃO repete o valor da UF
    assert round(t.loc["Brasil", "pct_prioritarios"], 3) == 0.256
    ufs, destaque, titulo = D.ufs_de_contexto(base, D.Filtros(ufs=["PI"]))
    assert destaque == ["PI"] and len(ufs) == 9 and "Nordeste" in titulo         # contexto: UFs da mesma região


def test_territorial_outra_regiao(base):
    t = D.comparacao_territorial(base, D.Filtros(ufs=["RS"]))
    assert list(t.index) == ["Rio Grande do Sul (RS)", "Sul", "Brasil"]
    assert (t.loc["Sul", "prioritarios"], t.loc["Sul", "municipios"]) == _ref(base, base.regiao == "Sul")
    assert t.loc["Rio Grande do Sul (RS)", "pct_prioritarios"] != t.loc["Sul", "pct_prioritarios"]


def test_territorial_multiplas_ufs(base):
    t = D.comparacao_territorial(base, D.Filtros(ufs=["SP", "PI"]))
    assert list(t.index) == ["Piauí (PI)", "São Paulo (SP)", "Nordeste", "Sudeste", "Brasil"]
    for nome, uf in (("Piauí (PI)", "PI"), ("São Paulo (SP)", "SP")):
        assert (t.loc[nome, "prioritarios"], t.loc[nome, "municipios"]) == _ref(base, base.sigla_uf == uf)
    r = D.resumo_recorte(base, D.Filtros(ufs=["SP", "PI"]))                      # recorte: denominador próprio
    assert (r["prioritarios"], r["municipios"]) == _ref(base, base.sigla_uf.isin(["SP", "PI"]))


@pytest.mark.parametrize("extra", [dict(prioridades=["menor"]), dict(prioridades=["alta_consenso"]),
                                   dict(alerta="Com alerta"), dict(alerta="Sem alerta")])
def test_territorial_ignora_filtros_de_prioridade_e_qualidade(base, extra):
    ref = D.comparacao_territorial(base, D.Filtros(ufs=["PI"]))
    t = D.comparacao_territorial(base, D.Filtros(ufs=["PI"], **extra))
    pd.testing.assert_frame_equal(t, ref)                                        # denominadores não mudam
    r = D.resumo_recorte(base, D.Filtros(ufs=["PI"], **extra))
    assert r["municipios"] == len(D.aplicar_filtros(base, D.Filtros(ufs=["PI"], **extra)))


def test_territorial_recorte_vazio(base):
    f = D.Filtros(ufs=["PI"], prioridades=["menor"], alerta="Com alerta")
    assert D.aplicar_filtros(base, f).empty
    r = D.resumo_recorte(base, f)
    assert r["municipios"] == 0 and np.isnan(r["pct_prioritarios"])
    t = D.comparacao_territorial(base, f)
    assert t.loc["Piauí (PI)", "municipios"] == 224                              # o território continua completo
    G.fig_territorial(t, "%")


def test_pagina_mapa_piaui_mostra_uf_regiao_brasil(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CONECTAMAP_PAGINA_INICIAL", "mapa")
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=120)
    at.run()
    at.sidebar.multiselect(key="f_ufs").set_value(["PI"]).run()
    assert not at.exception
    textos = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "UF selecionada × região completa × Brasil" in textos
    assert "municípios no recorte" in textos and "municípios destacados" not in textos
    assert "50,9% do recorte" in textos
