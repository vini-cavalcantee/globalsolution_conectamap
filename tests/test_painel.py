"""Etapa 2 — testes do painel temporal (E01–E12)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from conectamap import painel as PN
from conftest import sintetico

CHAVES_FEATURES = None


def _feats(p):
    return p[PN.colunas_features(p, incluir_censo=True, incluir_L6=True)]


# --------------------------------------------------------------------------
# E01 — integridade das junções
# --------------------------------------------------------------------------
def test_E01_juncoes_reais(dados_reais):
    d = dados_reais
    assert len(d["ids"]) == 5570 and len(set(d["ids"])) == 5570
    assert all(isinstance(i, str) and len(i) == 7 for i in d["ids"])
    assert d["n_registros"] == 1_105_770 - 5_570              # todos os registros até 2026-02
    assert list(d["ibge"].index) == list(d["ids"])
    p = PN.montar_paineis(["2019-02", "2020-02"], dados=d)
    assert len(p) == 2 * 5570 and not p.duplicated(["id_municipio", "origem"]).any()
    assert p.id_municipio.map(type).eq(str).all()


def test_E01b_juncao_recusa_duplicidade_e_divergencia():
    base, ind, aus, ibge = sintetico({"3500001": lambda c: 10.0, "3500002": lambda c: 5.0})
    with pytest.raises(ValueError):
        PN.integrar(pd.concat([base, base.iloc[[100]]]), ind, aus, ibge)
    with pytest.raises(ValueError):
        PN.integrar(base, ind, aus, ibge.iloc[:1])


# --------------------------------------------------------------------------
# E02 — correspondência origem × alvo
# --------------------------------------------------------------------------
def test_E02_origem_alvo_reais(dados_reais):
    d = dados_reais
    bruto = pd.read_csv(PN.C.DIR_TRATADOS / "anatel_densidade_tratada.csv", dtype={"id_municipio": str},
                        usecols=["id_municipio", "competencia", "densidade_original"])
    for origem in ("2013-02", "2018-02", "2023-02"):
        p = PN.montar_painel(origem, dados=d)
        alvo = str(pd.Period(origem, "M") + 36)
        assert (p.mes_alvo == alvo).all()
        ref = bruto[bruto.competencia == alvo].set_index("id_municipio").densidade_original
        ok = p.obs_no_alvo
        assert np.allclose(p.loc[ok, "densidade_alvo"].values, ref.reindex(p.loc[ok, "id_municipio"]).values)
        assert set(p.loc[ok, "id_municipio"]) == set(ref.index)
    # município com meses ausentes: Mojuí dos Campos (PA) não tem 2019-02
    p16 = PN.montar_painel("2016-02", dados=d).set_index("id_municipio")
    p19 = PN.montar_painel("2019-02", dados=d).set_index("id_municipio")
    assert not p16.loc["1504752", "obs_no_alvo"] and not p16.loc["1504752", "elegivel_treino"]
    assert np.isnan(p16.loc["1504752", "alvo_p25_origem"])
    assert not p19.loc["1504752", "obs_na_origem"] and not p19.loc["1504752", "elegivel_features"]


# --------------------------------------------------------------------------
# E03 — nenhuma informação posterior à origem nas features
# --------------------------------------------------------------------------
@pytest.mark.parametrize("origem", ["2011-02", "2016-02", "2021-02", "2023-02"])
def test_E03_sem_dados_pos_origem(dados_reais, origem):
    d = dados_reais
    t = int(np.where(d["meses"] == pd.Period(origem, "M"))[0][0])
    rng = np.random.default_rng(1)
    perturbado = dict(d)
    for k in ("M", "M_original", "flag12", "flag6", "refins"):
        X = d[k].copy()
        X[:, t + 1:] = rng.normal(50, 30, size=X[:, t + 1:].shape)      # futuro inteiramente trocado
        perturbado[k] = X
    A = d["ausente"].copy()
    A[:, t + 1:] = ~A[:, t + 1:]
    perturbado["ausente"] = A
    p0, p1 = PN.montar_painel(origem, dados=d), PN.montar_painel(origem, dados=perturbado)
    pd.testing.assert_frame_equal(_feats(p0), _feats(p1))
    assert (p0.limiar_p25_origem == p1.limiar_p25_origem).all()
    if p0.alvo_observavel.iloc[0]:
        assert not np.allclose(p0.densidade_alvo.fillna(-1), p1.densidade_alvo.fillna(-1))   # o alvo, sim, muda


def test_E03b_colunas_de_feature_sem_alvo_nem_retro(dados_reais):
    p = PN.montar_painel("2020-02", dados=dados_reais, limiar_absoluto=8.2)
    cols = PN.colunas_features(p, incluir_censo=True, incluir_L6=True)
    assert not [c for c in cols if any(s in c for s in ("alvo", "retro", "limiar", "elegivel", "obs_no"))]
    assert "abaixo_p25_na_origem" in cols          # posição na origem: causal


# --------------------------------------------------------------------------
# E04 — horizonte de 36 meses e observabilidade
# --------------------------------------------------------------------------
def test_E04_horizonte(dados_reais):
    d = dados_reais
    assert PN.montar_painel("2023-02", dados=d).alvo_observavel.all()
    p = PN.montar_painel("2023-03", dados=d)                   # alvo seria 2026-03: fora do limite
    assert not p.alvo_observavel.any() and p.densidade_alvo.isna().all() and p.alvo_p25_origem.isna().all()
    assert (PN.montar_painel("2020-02", horizonte=24, dados=d).mes_alvo == "2022-02").all()
    prev = PN.montar_painel("2026-02", dados=d)
    assert (prev.mes_alvo == "2029-02").all() and prev.densidade_alvo.isna().all() and prev.elegivel_features.sum() == 5570


# --------------------------------------------------------------------------
# E05 — disponibilidade temporal do IBGE
# --------------------------------------------------------------------------
def test_E05_censo(dados_reais):
    d = dados_reais
    censo = [f"censo_{v}" for v in PN.VARIAVEIS_CENSO]
    for origem in [f"{a}-02" for a in range(2011, 2024)]:          # todas as origens de treino
        p = PN.montar_painel(origem, dados=d, criterio_censo="publicacao")
        assert p[censo].isna().all().all() and (p.n_variaveis_censo == 0).all()
    assert PN.montar_painel("2022-07", dados=d, criterio_censo="conteudo")[censo].isna().all().all()
    assert PN.montar_painel("2022-08", dados=d, criterio_censo="conteudo")[censo].notna().all().all()
    assert PN.montar_painel("2011-02", dados=d, criterio_censo="nenhum")[censo].notna().all().all()
    p26 = PN.montar_painel("2026-02", dados=d, criterio_censo="publicacao")
    assert p26[censo].notna().all().all()
    assert np.allclose(p26.censo_pop_total.values, d["ibge"].pop_total.values)
    # 9515 (atualizada em 22/12/2023) entra em 2024-01; 9923 (14/11/2024) só em 2024-12
    p24 = PN.montar_painel("2024-01", dados=d, criterio_censo="publicacao")
    assert p24.censo_idade_mediana.notna().all() and p24.censo_pop_total.isna().all()


def test_E05b_metadados_iguais_aos_da_etapa_1_5():
    arq = PN.C.DIR_METADADOS / "metadados_fontes.csv"
    if not arq.exists():
        pytest.skip("auditoria técnica ausente")
    m = pd.read_csv(arq, dtype=str).set_index("chave")
    for r in PN.METADADOS_CENSO.itertuples():
        assert m.at[r.chave, "data_atualizacao_sidra"] == r.data_atualizacao_sidra
        assert m.at[r.chave, "data_referencia"] == r.data_referencia


# --------------------------------------------------------------------------
# E06 — março de 2026 fora do painel
# --------------------------------------------------------------------------
def test_E06_marco_2026(dados_reais):
    assert str(dados_reais["meses"][-1]) == "2026-02"
    with pytest.raises(ValueError):
        PN.montar_painel("2026-03", dados=dados_reais)
    with pytest.raises(ValueError):
        PN.montar_painel("2010-12", dados=dados_reais)
    base, ind, aus, ibge = sintetico({"3500001": lambda c: 999.0 if c == "2026-03" else 10.0})
    d = PN.integrar(base, ind, aus, ibge)
    assert np.nanmax(d["M"]) == 10.0                            # o valor de 2026-03 não entrou


# --------------------------------------------------------------------------
# E07 — reprodutibilidade
# --------------------------------------------------------------------------
def test_E07_reprodutibilidade(dados_reais):
    h = lambda df: hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    origens = PN.origens_fevereiro(2013, 2023)
    a = PN.montar_paineis(origens, dados=dados_reais)
    b = PN.montar_paineis(origens, dados=PN.carregar_bases())
    assert h(a) == h(b)


# --------------------------------------------------------------------------
# E08 — cálculo das features (dados sintéticos)
# --------------------------------------------------------------------------
def test_E08_features_sinteticas():
    def linear(c):        # cresce 1 por ano (1/12 por mês)
        p = pd.Period(c, "M")
        return 1.0 + (p.year - 2007) + (p.month - 1) / 12
    flags = {("3500003", "2019-12"): 1, ("3500003", "2020-01"): 1, ("3500003", "2020-02"): 2}
    base, ind, aus, ibge = sintetico({"3500001": lambda c: 10.0, "3500002": linear, "3500003": lambda c: 5.0},
                                     ausentes=[("3500001", "2019-06"), ("3500001", "2019-07")], flags=flags)
    d = PN.integrar(base, ind, aus, ibge)
    p = PN.montar_painel("2020-02", dados=d).set_index("id_municipio")
    assert p.loc["3500002", "tendencia_24"] == pytest.approx(1.0)
    assert p.loc["3500002", "variacao_abs_12"] == pytest.approx(1.0)
    assert p.loc["3500001", "volatilidade_12"] == pytest.approx(0.0)
    assert p.loc["3500001", "n_meses_ausentes_36"] == 2 and p.loc["3500001", "n_obs_12"] == 10
    assert p.loc["3500003", "n_suspeita_12"] == 2 and p.loc["3500003", "persistencia_36"] == 1
    assert p.loc["3500003", "flag_causal_atual"] == 2 and p.loc["3500003", "meses_desde_ultima_sinalizacao"] == 0
    # origem em 2011-02: a defasagem de 12 meses cai em 2010 (trimestral) e usa dez/2009 (tolerância de 2 meses)
    p11 = PN.montar_painel("2011-02", dados=d).set_index("id_municipio")
    assert p11.loc["3500002", "densidade_lag_12"] == pytest.approx(linear("2009-12"))
    assert p11.loc["3500002", "fracao_trimestral_36"] == pytest.approx(34 / 36)


# --------------------------------------------------------------------------
# E09 — backtesting: treino só com alvos conhecidos na data de validação
# --------------------------------------------------------------------------
def test_E09_folds():
    f = PN.folds_backtesting()
    for r in f.itertuples():
        pv = pd.Period(r.origem_avaliada, "M")
        for o in r.origens_treino.split(","):
            assert pd.Period(o, "M") + 36 <= pv if r.papel != "previsao" else pd.Period(o, "M") + 36 <= pd.Period("2026-02", "M")
    assert "2023-02" not in set(f[f.papel == "validacao"].origem_avaliada)
    assert all("2023-02" not in r.origens_treino for r in f[f.papel != "previsao"].itertuples())
    assert f[f.papel == "previsao"].origens_treino.iloc[0].split(",")[-1] == "2023-02"


# --------------------------------------------------------------------------
# E10 — limiar P25 só com dados da origem
# --------------------------------------------------------------------------
def test_E10_limiar_p25_origem(dados_reais):
    p = PN.montar_painel("2018-02", dados=dados_reais)
    assert p.limiar_p25_origem.iloc[0] == pytest.approx(np.nanquantile(p.densidade_atual, 0.25))
    ok = p.elegivel_treino
    assert (p.loc[ok, "alvo_p25_origem"] == (p.loc[ok, "densidade_alvo"] < p.limiar_p25_origem.iloc[0])).all()


# --------------------------------------------------------------------------
# E11 — bases tratadas e Notebook 1 não são alterados
# --------------------------------------------------------------------------
def test_E11_bases_tratadas_intactas(dados_reais):
    arqs = sorted(PN.C.DIR_TRATADOS.glob("*")) + [PN.C.PACOTE / "ConectaMap_Pre_Processamento.ipynb"]
    antes = {a.name: hashlib.sha256(a.read_bytes()).hexdigest() for a in arqs}
    PN.montar_paineis(PN.origens_fevereiro(2011, 2023) + ["2026-02"], dados=PN.carregar_bases())
    depois = {a.name: hashlib.sha256(a.read_bytes()).hexdigest() for a in arqs}
    assert antes == depois
