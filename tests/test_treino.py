"""Etapa 2 — treinamento e avaliação (E16–E20). Usa regressão logística nos testes por velocidade."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from conectamap import painel as PN, treino as TR

LR = ("logistica", {"C": 0.01})


@pytest.fixture(scope="module")
def paineis(dados_reais):
    return PN.montar_paineis(PN.origens_fevereiro(2014, 2023), dados=dados_reais)


# E16 — transformações ajustadas só com o treino da rodada
def test_E16_transformacoes_so_no_treino(paineis):
    f = TR.folds_validacao().iloc[0]                              # avalia 2019-02, treina 2014–2016
    _, m1 = TR.prever_rodada(*LR, paineis, f, PN.FEATURES_MODELO_PRINCIPAL)
    lixo = paineis.copy()
    fora = ~lixo.origem.isin(f.origens_treino.split(","))
    for c in PN.FEATURES_MODELO_PRINCIPAL:
        lixo.loc[fora, c] = 1e6                                   # tudo fora do treino vira valor absurdo
    _, m2 = TR.prever_rodada(*LR, lixo, f, PN.FEATURES_MODELO_PRINCIPAL)
    np.testing.assert_array_equal(m1.named_steps["imputacao"].statistics_, m2.named_steps["imputacao"].statistics_)
    np.testing.assert_array_equal(m1.named_steps["escala"].mean_, m2.named_steps["escala"].mean_)
    np.testing.assert_array_equal(m1.named_steps["modelo"].coef_, m2.named_steps["modelo"].coef_)


# E17 — dados de cada rodada e isolamento da coorte de teste
def test_E17_rodadas_e_teste_isolado(paineis):
    for f in TR.folds_validacao().itertuples():
        Xtr, ytr, av = TR.dados_rodada(paineis, f.origens_treino.split(","), f.origem_avaliada, PN.FEATURES_MODELO_PRINCIPAL)
        origens = paineis.loc[Xtr.index, "origem"]
        assert set(origens) == set(f.origens_treino.split(",")) and (av.origem == f.origem_avaliada).all()
        assert paineis.loc[Xtr.index, "elegivel_treino"].all() and ytr.isin([0, 1]).all()
        assert all(pd.Period(o, "M") + 36 <= pd.Period(f.origem_avaliada, "M") for o in set(origens))
    with pytest.raises(ValueError):
        TR.dados_rodada(paineis, ["2020-02", "2023-02"], "2022-02", PN.FEATURES_MODELO_PRINCIPAL)
    with pytest.raises(ValueError):
        TR.selecionar(paineis, algoritmos=("logistica",), folds=PN.folds_backtesting().query("papel == 'teste'"))
    # a seleção não muda se a coorte de 2023 for trocada por valores absurdos
    lixo = paineis.copy()
    t = lixo.origem == "2023-02"
    lixo.loc[t, PN.FEATURES_MODELO_PRINCIPAL] = -999.0
    lixo.loc[t, "alvo_p25_origem"] = 1.0
    _, r1 = TR.selecionar(paineis, algoritmos=("logistica",))
    _, r2 = TR.selecionar(lixo, algoritmos=("logistica",))
    pd.testing.assert_frame_equal(r1, r2)


# E18 — protocolo congelado antes do teste
def test_E18_protocolo(tmp_path, paineis):
    with pytest.raises(RuntimeError):
        TR.avaliar_teste(paineis, tmp_path / "nao_existe.json")
    real = PN.C.PACOTE / "etapa2" / "protocolo_treino_congelado.json"
    prot = TR.ler_protocolo(real)
    assert prot["algoritmo"] == "random_forest" and prot["features"] == PN.FEATURES_MODELO_PRINCIPAL
    assert prot["origem_teste"] == "2023-02" and prot["alvo"] == "alvo_p25_origem"
    copia = tmp_path / "p.json"
    shutil.copy(real, copia)
    d = json.loads(copia.read_text(encoding="utf-8"))
    d["hiperparametros"]["min_samples_leaf"] = 1                 # alteração depois de congelado
    copia.write_text(json.dumps(d, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    with pytest.raises(RuntimeError):
        TR.ler_protocolo(copia)


# E19 — sensibilidade do alvo: informação retrospectiva só no alvo
def test_E19_alvo_suspeito(paineis):
    m = TR.marcar_alvo_suspeito(paineis)
    assert len(m) == len(paineis) and not set(m.columns) & set(PN.FEATURES_MODELO_PRINCIPAL)
    ind = pd.read_csv(PN.C.DIR_TRATADOS / "anatel_indicadores_qualidade.csv", dtype={"id_municipio": str},
                      usecols=["id_municipio", "competencia", "retro_classificacao", "retro_evento_coletivo"])
    alvo = paineis[["id_municipio", "mes_alvo"]].merge(ind, left_on=["id_municipio", "mes_alvo"],
                                                      right_on=["id_municipio", "competencia"], how="left")
    esperado = alvo.retro_classificacao.isin(TR.CLASSES_COM_RETORNO) & ~alvo.retro_evento_coletivo.fillna(False).astype(bool)
    np.testing.assert_array_equal(m.alvo_em_episodio_com_retorno.values, esperado.values)
    ok = m.alvo_robusto.notna() & ~m.alvo_em_episodio_com_retorno
    # o alvo robusto usa o MESMO limiar congelado da origem
    assert ((m.densidade_alvo_robusta < paineis.limiar_p25_origem.values)[ok] == (m.alvo_robusto[ok] == 1)).all()


# E20 — reprodutibilidade do treino
def test_E20_reprodutibilidade(paineis):
    f = TR.folds_validacao().iloc[[3]]
    a = TR.avaliar_config("random_forest", {"max_features": "sqrt", "min_samples_leaf": 300}, paineis, f,
                          PN.FEATURES_MODELO_PRINCIPAL)
    b = TR.avaliar_config("random_forest", {"max_features": "sqrt", "min_samples_leaf": 300}, paineis, f,
                          PN.FEATURES_MODELO_PRINCIPAL)
    pd.testing.assert_frame_equal(a, b)
