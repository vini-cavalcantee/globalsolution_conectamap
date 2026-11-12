"""Etapa 2 — previsão de 2029 (E21–E27)."""
import math

import numpy as np
import pandas as pd
import pytest

from conectamap import painel as PN, previsao as PV, treino as TR


@pytest.fixture(scope="module")
def hist(dados_reais):
    return PN.montar_paineis(PN.origens_fevereiro(2014, 2023), dados=dados_reais)


@pytest.fixture(scope="module")
def p26(dados_reais):
    return PN.montar_painel(PV.ORIGEM_OPERACIONAL, dados=dados_reais)


@pytest.fixture(scope="module")
def operacional(hist, p26, dados_reais):
    m, tr = PV.treinar_operacional(hist)
    return m, tr, PV.prever_2029(m, p26, dados_reais["ibge"])


# E21 — verificação do protocolo antes da previsão
def test_E21_verificacao_protocolo(hist, p26):
    v = PV.verificar_protocolo(hist, p26)
    assert v.ok.all(), v[~v.ok]
    assert p26.densidade_alvo.isna().all() and not p26.alvo_observavel.any()


# E22 — o treino operacional reproduz o procedimento congelado e só usa as 10 coortes 2014–2023
def test_E22_treino_operacional(hist, operacional):
    m, tr, _ = operacional
    prot = TR.ler_protocolo(PV.CAMINHO_PROTOCOLO)
    ref = TR.construir_modelo(prot["algoritmo"], prot["hiperparametros"]).named_steps["modelo"].get_params()
    got = m.named_steps["modelo"].get_params()
    assert {k: v for k, v in got.items() if k != "n_jobs"} == {k: v for k, v in ref.items() if k != "n_jobs"}
    assert sorted(tr.origem.unique()) == PN.origens_fevereiro(2014, 2023)
    assert len(tr) == 55686 and (tr.mes_alvo <= "2026-02").all()
    assert list(m.feature_names_in_) == PN.FEATURES_MODELO_PRINCIPAL
    # uma origem não permitida no conjunto de entrada não entra no treino
    extra = hist[hist.origem == "2023-02"].assign(origem="2024-02")
    _, tr2 = PV.treinar_operacional(pd.concat([hist, extra]))
    assert len(tr2) == len(tr)


# E23 — a previsão depende só das features da origem 2026-02 e é reproduzível
def test_E23_previsao_so_origem_e_reproduzivel(operacional, p26, dados_reais):
    m, _, prev = operacional
    lixo = p26.assign(densidade_alvo=123.0, alvo_p25_origem=1.0)       # colunas de alvo não são lidas
    prev2 = PV.prever_2029(m, lixo, dados_reais["ibge"])
    pd.testing.assert_frame_equal(prev, prev2)
    assert len(prev) == 5570 and prev.id_municipio.is_unique
    assert not any("prob" in c for c in prev.columns)                  # nenhuma coluna de probabilidade divulgada


# E24 — rankings, categorias e cruzamento RF × B2
def test_E24_rankings_categorias(operacional):
    _, _, p = operacional
    n, k = len(p), math.ceil(0.25 * len(p))
    for c in ("posicao_rf", "posicao_b2"):
        assert sorted(p[c]) == list(range(1, n + 1))
    assert (p.sort_values("posicao_rf").escore_rf.diff().dropna() <= 0).all()
    assert (p.sort_values("posicao_b2").densidade_fev2026.diff().dropna() >= 0).all()
    assert (p.categoria_rf == "maior").sum() == k and (p.categoria_b2 == "maior").sum() == k
    assert ((p.categoria_rf == "intermediario").sum() == math.ceil(0.5 * n) - k)
    assert (p.diferenca_posicoes == p.posicao_b2 - p.posicao_rf).all()
    ambos = (p.categoria_rf == "maior") & (p.categoria_b2 == "maior")
    assert ((p.concordancia_top25 == "ambos") == ambos).all()
    assert ((p.classificacao_prioridade == "alta_consenso") == ambos).all()


def test_E24b_ranking_empates_deterministico():
    v = pd.Series([0.5, 0.9, 0.5, 0.1])
    ids = pd.Series(["3", "2", "1", "4"])
    assert list(PV._ranking(v, ids)) == [3, 1, 2, 4]                    # empate: menor código primeiro
    assert list(PV.categoria(np.arange(1, 9), 8)) == ["maior"] * 2 + ["intermediario"] * 2 + ["menor"] * 4


# E25 — métricas de calibração e regra de divulgação pré-registrada
def test_E25_calibracao_e_regra():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.9, 200_000)
    y = rng.uniform(size=p.size) < p
    m = PV.metricas_calibracao(y, p)
    assert abs(m["calibracao_geral"]) < 0.005 and abs(m["inclinacao"] - 1) < 0.05 and abs(m["decil_sup_dif"]) < 0.01
    m2 = PV.metricas_calibracao(y, p * 0.6)                             # subestimação sistemática
    assert m2["calibracao_geral"] < -0.1

    def tab(cal, incl, dec, brier):
        linhas = []
        for r in ("V1", "V2", "V3", "V4"):
            linhas += [dict(rodada=r, versao="rf_bruto", grupo="todos", calibracao_geral=cal, inclinacao=incl,
                            decil_sup_dif=dec, brier=brier),
                       dict(rodada=r, versao="rf_platt", grupo="todos", calibracao_geral=cal, inclinacao=incl,
                            decil_sup_dif=dec, brier=brier),
                       dict(rodada=r, versao="B2p_densidade", grupo="todos", calibracao_geral=0, inclinacao=1,
                            decil_sup_dif=0, brier=0.09)]
        return pd.DataFrame(linhas)
    assert PV.decidir_divulgacao(tab(0.01, 1.0, 0.02, 0.08)).aprovada.all()
    assert not PV.decidir_divulgacao(tab(-0.03, 1.0, 0.02, 0.08)).aprovada.any()
    assert not PV.decidir_divulgacao(tab(0.01, 0.7, 0.02, 0.08)).aprovada.any()
    assert not PV.decidir_divulgacao(tab(0.01, 1.0, 0.02, 0.10)).aprovada.any()


# E26 — o calibrador de cada rodada é ajustado numa coorte anterior, nunca na avaliada nem em 2023
def test_E26_calibrador_temporal(hist):
    prot = TR.ler_protocolo(PV.CAMINHO_PROTOCOLO)
    f = TR.folds_validacao().iloc[0]                                    # avalia 2019, treina 2014–2016
    sem_teste = hist[hist.origem != TR.ORIGEM_TESTE]
    p = PV.previsoes_calibracao(sem_teste, prot, f)
    assert p.coorte_calibrador.iloc[0] == f.origens_treino.split(",")[-1] < f.origem_avaliada
    assert (p.origem == f.origem_avaliada).all()
    # mudar o alvo da coorte avaliada não muda nenhuma probabilidade (o alvo avaliado não é usado no ajuste)
    lixo = sem_teste.copy()
    lixo.loc[lixo.origem == f.origem_avaliada, TR.ALVO] = 1 - lixo.loc[lixo.origem == f.origem_avaliada, TR.ALVO]
    p2 = PV.previsoes_calibracao(lixo, prot, f)
    for c in ("prob", "prob_platt", "prob_b2p"):
        np.testing.assert_array_equal(p[c].values, p2[c].values)


# E27 — B2p usa só a densidade atual e é monótono nela
def test_E27_b2p(hist):
    tr = hist[hist.origem.isin(["2014-02", "2015-02"]) & hist.elegivel_treino.astype(bool)]
    m = PV.ajustar_b2p(tr)
    assert m.n_features_in_ == 1
    d = pd.DataFrame({"densidade_atual": [0.5, 2, 5, 20, 80]})
    assert (np.diff(PV.prever_b2p(m, d)) < 0).all()


# E28 — a base final recalculada é idêntica, byte a byte, ao CSV aprovado de 2029
def test_E28_base_final_reproduz_aprovada(operacional, hist, p26, dados_reais):
    aprovado = PV.DIR_SAIDA / "previsoes_2029_municipios.csv"
    if not aprovado.exists():
        pytest.skip("saídas aprovadas ausentes")
    base = PV.previsao_final(operacional[0], p26, hist, dados_reais["ibge"])
    assert PV.texto_csv(base).encode("utf-8") == aprovado.read_bytes()


# E29 — base pronta para o dashboard: dicionário completo e verificações de uso direto
def test_E29_base_dashboard():
    from conectamap import apresentacao as AP
    aprovado = PV.DIR_SAIDA / "previsoes_2029_municipios.csv"
    if not aprovado.exists():
        pytest.skip("saídas aprovadas ausentes")
    base = AP.ler_base_2029(aprovado)
    v = AP.validar_base_dashboard(base)
    assert v.ok.all(), v[~v.ok]
    dic = AP.dicionario_base_2029()
    assert dic.coluna.tolist() == base.columns.tolist() and dic.notna().all().all()
    assert "NÃO CALIBRADO" in dic.set_index("coluna").loc["escore_rf", "grupo"]
    reg = AP.resumo_regional(base)
    assert reg.municipios.sum() == 5570 and reg.alta_consenso.sum() == (base.classificacao_prioridade == "alta_consenso").sum()
