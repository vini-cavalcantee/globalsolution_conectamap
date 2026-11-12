"""Etapa 2 — proposta metodológica consolidada (E12–E15): features do modelo principal, regra conservadora
de backtesting, métricas e baselines. Nenhum modelo é treinado."""
import math

import numpy as np
import pandas as pd
import pytest

from conectamap import avaliacao as AV, painel as PN


# E12 — modelo principal só com features temporais da Anatel
def test_E12_features_modelo_principal(dados_reais):
    p = PN.montar_painel("2020-02", dados=dados_reais)
    assert PN.FEATURES_MODELO_PRINCIPAL == PN.colunas_features(p, incluir_censo=False, incluir_L6=False)
    assert len(PN.FEATURES_MODELO_PRINCIPAL) == 30
    assert not [c for c in PN.FEATURES_MODELO_PRINCIPAL
                if c.startswith("censo_") or c.endswith("_L6") or "retro" in c or "alvo" in c or "limiar" in c]
    # o Censo está ausente em todas as origens históricas de treino (2014–2023) pelo critério principal
    treino = PN.montar_paineis(PN.origens_fevereiro(2014, 2023), dados=dados_reais)
    assert treino.filter(like="censo_").isna().all().all()
    assert PN.ALVO_PRINCIPAL == "alvo_p25_origem"


# E13 — regra conservadora: alvo de treino estritamente anterior à origem avaliada
def test_E13_regra_conservadora():
    atual, cons = PN.folds_backtesting(), PN.folds_backtesting(regra="conservadora")
    for r in cons.itertuples():
        limite = pd.Period("2026-02" if r.papel == "previsao" else r.origem_avaliada, "M")
        assert all(pd.Period(o, "M") + 36 < limite for o in r.origens_treino.split(","))
    # com origens anuais, a regra conservadora perde exatamente a coorte mais recente de cada rodada
    assert (atual.n_origens_treino - cons.n_origens_treino == 1).all()
    assert cons.loc[cons.papel == "previsao", "origens_treino"].iloc[0].split(",")[-1] == "2022-02"
    with pytest.raises(ValueError):
        PN.folds_backtesting(regra="outra")


# E14 — métricas calculadas à mão
def test_E14_metricas():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 1])
    s = np.array([.9, .8, .7, .6, .5, .4, .3, .2])
    ids = [f"35000{i:02d}" for i in range(8)]
    # positivos nas posições 1, 3 e 8: AP = (1/1 + 2/3 + 3/8) / 3
    assert AV.pr_auc(y, s) == pytest.approx((1 + 2 / 3 + 3 / 8) / 3)
    prec, rec = AV.top_fracao(y, s, ids)                          # top ceil(0,25 × 8) = 2
    assert (prec, rec) == (pytest.approx(0.5), pytest.approx(1 / 3))
    assert AV.brier([1, 0], [0.8, 0.3]) == pytest.approx(((0.2) ** 2 + 0.3 ** 2) / 2)
    with pytest.raises(ValueError):
        AV.brier([1, 0], [1.2, 0.0])
    # empates: escore constante -> AP = prevalência; top pelo valor esperado (2 vagas × 3/8)
    assert AV.pr_auc(y, np.zeros(8)) == pytest.approx(3 / 8)
    assert AV.top_fracao(y, np.zeros(8), ids) == (pytest.approx(3 / 8), pytest.approx(0.25))
    # o resultado não depende da ordem nem dos códigos dos municípios
    perm = np.random.default_rng(0).permutation(8)
    assert AV.top_fracao(y[perm], np.zeros(8)[perm], list(reversed(ids))) == AV.top_fracao(y, np.zeros(8), ids)
    assert math.isnan(AV.pr_auc([0, 0], [1, 2]))


# E15 — baselines e avaliação separada de quem estava acima do P25 na origem
def test_E15_baselines():
    c = pd.DataFrame({"id_municipio": ["3500001", "3500002", "3500003", "3500004"],
                      "densidade_atual": [1.0, 3.0, 10.0, 20.0], "tendencia_36": [0.0, -1.0, np.nan, 2.0],
                      "abaixo_p25_na_origem": [1.0, 0.0, 0.0, 0.0], "alvo_p25_origem": [1.0, 1.0, 0.0, 0.0]})
    b = AV.scores_baselines(c, prevalencia_treino=0.15)
    assert list(b) == AV.BASELINES
    assert np.allclose(b["B3_tendencia"][0], [-1.0, 0.0, -10.0, -26.0])     # projeção 3 anos; NaN -> atual
    assert np.allclose(b["B2_densidade_atual"][0], [-1, -3, -10, -20]) and b["B2_densidade_atual"][1] is None
    assert np.allclose(b["B0_taxa_base"][1], 0.15)
    r = AV.avaliar_coorte(c, *b["B1_persistencia"]).set_index("grupo")
    assert r.loc["todos", "n"] == 4 and r.loc["acima_p25_na_origem", "n"] == 3
    # no subgrupo acima do P25, a persistência dá o mesmo escore a todos: desempenho = acaso (k/n)
    assert r.loc["acima_p25_na_origem", "recall_top25"] == pytest.approx(1 / 3)
    assert r.loc["acima_p25_na_origem", "pr_auc"] == pytest.approx(r.loc["acima_p25_na_origem", "prevalencia"])
    r3 = AV.avaliar_coorte(c, b["B3_tendencia"][0]).set_index("grupo")
    assert r3.loc["acima_p25_na_origem", "recall_top25"] == 1          # projeção capta a queda do 3500002
    assert np.isnan(r3.loc["todos", "brier"])
