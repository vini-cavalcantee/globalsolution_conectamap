"""Etapa 2 — treinamento e avaliação (segue PROPOSTA_METODOLOGICA_TREINAMENTO.md, sem alterá-la).

Ordem obrigatória:
1. baselines e modelos avaliados nas rodadas de validação V1–V4 (regra atual);
2. seleção pela PR-AUC média de V1–V4 (desempate: recall no top 25%);
3. sensibilidades com o procedimento selecionado;
4. `congelar_protocolo` grava algoritmo, hiperparâmetros, features e sementes;
5. `avaliar_teste` só roda com o protocolo congelado e o avalia UMA vez na origem 2023-02.

Toda transformação (imputação, escala, transformação log) fica DENTRO do pipeline e é ajustada só com as
linhas de treino de cada rodada. A coorte de teste não entra em nenhuma função de seleção.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import avaliacao as AV
from . import painel as PN

SEMENTE = 42
ORIGEM_TESTE = "2023-02"
ALVO = PN.ALVO_PRINCIPAL


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def _log_simetrico(X):
    return np.sign(X) * np.log1p(np.abs(X))


def construir_modelo(algoritmo: str, params: dict) -> Pipeline:
    if algoritmo == "logistica":
        return Pipeline([
            ("imputacao", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("log", FunctionTransformer(_log_simetrico)),          # caudas longas (densidades até ~400)
            ("escala", StandardScaler()),
            ("modelo", LogisticRegression(C=params["C"], max_iter=2000, random_state=SEMENTE)),
        ])
    if algoritmo == "random_forest":
        return Pipeline([
            ("imputacao", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("modelo", RandomForestClassifier(n_estimators=200, min_samples_leaf=params["min_samples_leaf"],
                                              max_features=params["max_features"], n_jobs=-1, random_state=SEMENTE)),
        ])
    if algoritmo == "hist_gradient_boosting":   # trata ausentes nativamente
        return Pipeline([("modelo", HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"], max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"], max_iter=params["max_iter"], l2_regularization=1.0,
            early_stopping=False, random_state=SEMENTE))])
    raise ValueError(algoritmo)


# Grade final. A 1ª grade (C 0,01–1; min_samples_leaf 20–100; learning_rate 0,05–0,1, folhas 15–31) teve o
# melhor resultado na borda mais regularizada; a grade foi ampliada nessa direção usando SÓ as rodadas V1–V4.
GRADES = {
    "logistica": {"C": [0.001, 0.003, 0.01, 0.1, 1.0]},
    "random_forest": {"min_samples_leaf": [20, 100, 300], "max_features": ["sqrt", 0.5]},
    "hist_gradient_boosting": {"learning_rate": [0.02, 0.05, 0.1], "max_leaf_nodes": [7, 15],
                               "min_samples_leaf": [100, 300], "max_iter": [200]},
}


def configuracoes(algoritmo):
    g = GRADES[algoritmo]
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


# --------------------------------------------------------------------------
# Dados de uma rodada
# --------------------------------------------------------------------------
def dados_rodada(paineis: pd.DataFrame, origens_treino, origem_avaliada, features, alvo=ALVO, excluir=None):
    """Linhas elegíveis de treino (origens da rodada) e de avaliação (origem avaliada).
    `excluir`: máscara booleana opcional (sensibilidades do alvo) aplicada às duas partes."""
    ok = paineis.elegivel_treino.astype(bool) & paineis[alvo].notna()
    if excluir is not None:
        ok &= ~excluir
    tr = paineis[ok & paineis.origem.isin(origens_treino)]
    av = paineis[ok & (paineis.origem == origem_avaliada)]
    if ORIGEM_TESTE in set(tr.origem):
        raise ValueError("a origem de teste não pode entrar no treino")
    return tr[features], tr[alvo].astype(int), av


def prever_rodada(algoritmo, params, paineis, fold, features, alvo=ALVO, excluir=None):
    Xtr, ytr, av = dados_rodada(paineis, fold.origens_treino.split(","), fold.origem_avaliada, features, alvo, excluir)
    m = construir_modelo(algoritmo, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, ytr)
    if "n_jobs" in m.named_steps["modelo"].get_params():
        # predição em 1 thread: a média das árvores sempre na mesma ordem -> resultados idênticos byte a byte
        # (o treino não muda; sem isso, a ordem das threads altera o 17º dígito das probabilidades)
        m.named_steps["modelo"].set_params(n_jobs=1)
    p = m.predict_proba(av[features])[:, 1]
    return av.assign(prob=p, prevalencia_treino=float(ytr.mean())), m


# --------------------------------------------------------------------------
# Avaliação
# --------------------------------------------------------------------------
def metricas(av: pd.DataFrame, escore, prob, rodada, metodo, alvo=ALVO) -> pd.DataFrame:
    r = AV.avaliar_coorte(av.assign(**{alvo: av[alvo]}), escore, prob, alvo=alvo)
    return r.assign(rodada=rodada, origem=av.origem.iloc[0], metodo=metodo)


def avaliar_baselines(paineis, folds, alvo=ALVO, excluir=None) -> pd.DataFrame:
    out = []
    for f in folds.itertuples():
        _, ytr, av = dados_rodada(paineis, f.origens_treino.split(","), f.origem_avaliada,
                                  PN.FEATURES_MODELO_PRINCIPAL, alvo, excluir)
        for nome, (esc, prob) in AV.scores_baselines(av, float(ytr.mean())).items():
            out.append(metricas(av, esc, prob, f.rodada, nome, alvo))
    return pd.concat(out, ignore_index=True)


def avaliar_config(algoritmo, params, paineis, folds, features, alvo=ALVO, excluir=None, guardar=False):
    out, previsoes = [], []
    for f in folds.itertuples():
        av, _ = prever_rodada(algoritmo, params, paineis, f, features, alvo, excluir)
        out.append(metricas(av, av.prob.values, av.prob.values, f.rodada, algoritmo, alvo))
        if guardar:
            previsoes.append(av[["id_municipio", "origem", alvo, "abaixo_p25_na_origem", "densidade_atual",
                                 "tendencia_36", "prob"]].assign(rodada=f.rodada))
    res = pd.concat(out, ignore_index=True).assign(config=json.dumps(params, sort_keys=True))
    return (res, pd.concat(previsoes, ignore_index=True)) if guardar else res


def folds_validacao(regra="atual"):
    f = PN.folds_backtesting(regra=regra)
    return f[f.papel == "validacao"].reset_index(drop=True)


def selecionar(paineis, features=None, algoritmos=tuple(GRADES), folds=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia todas as configurações em V1–V4 e devolve (todas as métricas, resumo por configuração).
    Seleção: maior PR-AUC média (grupo 'todos') nas 4 rodadas; desempate pelo recall no top 25%."""
    features = PN.FEATURES_MODELO_PRINCIPAL if features is None else features
    folds = folds_validacao() if folds is None else folds
    if (folds.origem_avaliada == ORIGEM_TESTE).any():
        raise ValueError("a origem de teste não pode ser usada na seleção")
    sem_teste = paineis[paineis.origem != ORIGEM_TESTE]          # a coorte de teste nem chega às funções
    todas = [avaliar_config(a, c, sem_teste, folds, features) for a in algoritmos for c in configuracoes(a)]
    todas = pd.concat(todas, ignore_index=True)
    resumo = (todas[todas.grupo == "todos"].groupby(["metodo", "config"])
              .agg(pr_auc_media=("pr_auc", "mean"), recall_top25_media=("recall_top25", "mean"),
                   brier_medio=("brier", "mean"), pr_auc_min=("pr_auc", "min"))
              .reset_index().sort_values(["pr_auc_media", "recall_top25_media"], ascending=False).reset_index(drop=True))
    return todas, resumo


def calibracao(previsoes: pd.DataFrame, alvo=ALVO, n_faixas=10) -> pd.DataFrame:
    """Probabilidade média prevista × frequência observada por faixa de probabilidade (decis)."""
    p = previsoes.copy()
    p["faixa"] = pd.qcut(p.prob.rank(method="first"), n_faixas, labels=False)
    return p.groupby("faixa").agg(prob_media=("prob", "mean"), freq_observada=(alvo, "mean"), n=("prob", "size")).reset_index()


def bootstrap_diferenca(y, esc_a, esc_b, n=500, semente=SEMENTE):
    """IC 95% da diferença de PR-AUC (a − b), reamostrando municípios da coorte."""
    from sklearn.metrics import average_precision_score
    y, a, b = map(np.asarray, (y, esc_a, esc_b))
    rng = np.random.default_rng(semente)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        d.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --------------------------------------------------------------------------
# Sensibilidades
# --------------------------------------------------------------------------
def features_L6():
    troca = {"flag_causal_atual": "flag_causal_atual_L6", "n_suspeita_12": "n_suspeita_12_L6",
             "n_suspeita_36": "n_suspeita_36_L6", "persistencia_36": "persistencia_36_L6"}
    return [troca.get(f, f) for f in PN.FEATURES_MODELO_PRINCIPAL]


CLASSES_COM_RETORNO = {"atipico_retorno_consistente", "atipico_com_retorno"}


def marcar_alvo_suspeito(paineis: pd.DataFrame, pasta=None) -> pd.DataFrame:
    """SÓ PARA SENSIBILIDADE DO ALVO: lê as colunas retrospectivas dos indicadores no mês-alvo.
    Devolve, por (id_municipio, origem): alvo_em_episodio_com_retorno (S-A1), alvo_pendente_fim_serie (S-A3),
    densidade_alvo_robusta (S-A2: mediana dez–fev do ano-alvo, sem os meses de S-A1) e alvo_robusto.
    Nada disso entra nas features."""
    pasta = PN.C.DIR_TRATADOS if pasta is None else pasta
    ind = pd.read_csv(pasta / "anatel_indicadores_qualidade.csv", dtype={"id_municipio": str},
                      usecols=["id_municipio", "competencia", "dentro_limite_observacao", "retro_classificacao",
                               "retro_evento_coletivo"])
    base = pd.read_csv(pasta / "anatel_densidade_tratada.csv", dtype={"id_municipio": str},
                       usecols=["id_municipio", "competencia", "densidade_original"])
    t = base.merge(ind, on=["id_municipio", "competencia"])
    t = t[t.dentro_limite_observacao.astype(bool)]
    t["suspeito"] = t.retro_classificacao.isin(CLASSES_COM_RETORNO) & ~t.retro_evento_coletivo.astype(bool)
    t["pendente"] = t.retro_classificacao.eq("pendente_fim_serie")
    chave = t.set_index(["id_municipio", "competencia"])
    p = paineis[["id_municipio", "origem", "mes_alvo", "limiar_p25_origem"]].copy()
    idx = pd.MultiIndex.from_arrays([p.id_municipio, p.mes_alvo])
    p["alvo_em_episodio_com_retorno"] = chave.suspeito.reindex(idx).fillna(False).astype(bool).values
    p["alvo_pendente_fim_serie"] = chave.pendente.reindex(idx).fillna(False).astype(bool).values
    # alvo robusto: dez(ano-1), jan, fev do ano-alvo, sem meses suspeitos
    vals = []
    for desloc in (-2, -1, 0):
        m = (pd.PeriodIndex(p.mes_alvo, freq="M") + desloc).astype(str)
        i2 = pd.MultiIndex.from_arrays([p.id_municipio, m])
        v = chave.densidade_original.reindex(i2).values.astype(float)
        s = chave.suspeito.reindex(i2).fillna(False).astype(bool).values
        vals.append(np.where(s, np.nan, v))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p["densidade_alvo_robusta"] = np.nanmedian(np.vstack(vals), axis=0)
    p["alvo_robusto"] = np.where(np.isnan(p.densidade_alvo_robusta), np.nan,
                                 (p.densidade_alvo_robusta < p.limiar_p25_origem).astype(float))
    return p.drop(columns=["mes_alvo", "limiar_p25_origem"])


# --------------------------------------------------------------------------
# Protocolo congelado e teste reservado
# --------------------------------------------------------------------------
def congelar_protocolo(caminho: Path, algoritmo, params, features, extra=None) -> dict:
    prot = dict(algoritmo=algoritmo, hiperparametros=params, features=list(features), alvo=ALVO,
                origem_teste=ORIGEM_TESTE, regra_backtesting="atual", semente=SEMENTE,
                criterio_selecao="PR-AUC média V1–V4 (desempate: recall top 25%)", **(extra or {}))
    texto = json.dumps(prot, ensure_ascii=False, sort_keys=True, indent=1)
    prot["sha256_protocolo"] = hashlib.sha256(texto.encode()).hexdigest()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(prot, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    return prot


def ler_protocolo(caminho: Path) -> dict:
    if not Path(caminho).exists():
        raise RuntimeError("protocolo não congelado: o teste reservado não pode ser executado")
    prot = json.loads(Path(caminho).read_text(encoding="utf-8"))
    h = prot.pop("sha256_protocolo")
    if hashlib.sha256(json.dumps(prot, ensure_ascii=False, sort_keys=True, indent=1).encode()).hexdigest() != h:
        raise RuntimeError("protocolo alterado depois de congelado")
    prot["sha256_protocolo"] = h
    return prot


def avaliar_teste(paineis, caminho_protocolo, regra="atual"):
    """Avaliação única na origem 2023-02 com o protocolo congelado (modelo + baselines)."""
    prot = ler_protocolo(caminho_protocolo)
    f = PN.folds_backtesting(regra=regra)
    f = f[f.papel == "teste"].iloc[0]
    av, modelo = prever_rodada(prot["algoritmo"], prot["hiperparametros"], paineis, f, prot["features"])
    res = [metricas(av, av.prob.values, av.prob.values, f.rodada, prot["algoritmo"])]
    for nome, (esc, prob) in AV.scores_baselines(av, float(av.prevalencia_treino.iloc[0])).items():
        res.append(metricas(av, esc, prob, f.rodada, nome))
    return pd.concat(res, ignore_index=True).assign(regra=regra), av, modelo


# --------------------------------------------------------------------------
# Sensibilidades com o procedimento selecionado (sempre em V1–V4; nunca no teste)
# --------------------------------------------------------------------------
def resumo_por_metodo(tab: pd.DataFrame) -> pd.DataFrame:
    """Média das 4 rodadas por método e grupo (a tabela por coorte continua disponível)."""
    return (tab.groupby(["metodo", "grupo"])[["prevalencia", "pr_auc", "precisao_top25", "recall_top25", "brier"]]
            .mean().reset_index())


def rodar_sensibilidades(paineis, paineis_censo_nenhum, algoritmo, params, marcas) -> pd.DataFrame:
    """Executa as sensibilidades previstas; devolve métricas por coorte com a coluna `sensibilidade`."""
    sem_teste = paineis[paineis.origem != ORIGEM_TESTE]
    feats = PN.FEATURES_MODELO_PRINCIPAL
    out = []

    def junta(tab, nome):
        out.append(tab.assign(sensibilidade=nome))

    # principal (referência para comparação)
    fv = folds_validacao()
    junta(avaliar_config(algoritmo, params, sem_teste, fv, feats), "principal")
    junta(avaliar_baselines(sem_teste, fv), "principal")
    # 1. backtesting conservador
    fc = folds_validacao("conservadora")
    junta(avaliar_config(algoritmo, params, sem_teste, fc, feats), "backtesting_conservador")
    junta(avaliar_baselines(sem_teste, fc), "backtesting_conservador")
    # 2. flags com L=6
    junta(avaliar_config(algoritmo, params, sem_teste, fv, features_L6()), "flags_L6")
    # 3a. S-A1: exclui pares com mês-alvo em episódio retrospectivo com retorno (fora de evento coletivo)
    m = sem_teste.merge(marcas, on=["id_municipio", "origem"], how="left")
    excl = m.alvo_em_episodio_com_retorno.fillna(False).astype(bool).values
    excl = pd.Series(excl, index=sem_teste.index)
    junta(avaliar_config(algoritmo, params, sem_teste, fv, feats, excluir=excl), "alvo_SA1_exclui_suspeitos")
    junta(avaliar_baselines(sem_teste, fv, excluir=excl), "alvo_SA1_exclui_suspeitos")
    # 3b. S-A2: alvo robusto (mediana dez–fev sem meses suspeitos), mesmo limiar congelado
    rob = sem_teste.copy()
    rob["alvo_robusto"] = m.alvo_robusto.values
    junta(avaliar_config(algoritmo, params, rob, fv, feats, alvo="alvo_robusto"), "alvo_SA2_robusto")
    junta(avaliar_baselines(rob, fv, alvo="alvo_robusto"), "alvo_SA2_robusto")
    return pd.concat(out, ignore_index=True)


def rodar_hibrido_exploratorio(paineis_censo_nenhum, algoritmo, params) -> pd.DataFrame:
    """EXPLORATÓRIO: Censo 2022 pelo critério 'nenhum' (anacrônico) em todas as origens. Não é validação histórica."""
    sem_teste = paineis_censo_nenhum[paineis_censo_nenhum.origem != ORIGEM_TESTE]
    feats = PN.FEATURES_MODELO_PRINCIPAL + [f"censo_{v}" for v in PN.VARIAVEIS_CENSO]
    return avaliar_config(algoritmo, params, sem_teste, folds_validacao(), feats).assign(sensibilidade="hibrido_censo_nenhum_EXPLORATORIO")


def comparar_com_baseline(previsoes: pd.DataFrame, baseline="B2_densidade_atual", n_boot=500, alvo=ALVO) -> pd.DataFrame:
    """Diferença de PR-AUC (modelo − baseline) por rodada e grupo, com IC 95% por bootstrap de municípios."""
    linhas = []
    for rodada, d in previsoes.groupby("rodada", sort=True):
        for grupo, g in (("todos", d), ("acima_p25_na_origem", d[d.abaixo_p25_na_origem == 0])):
            esc_b = AV.scores_baselines(g.assign(densidade_atual=g.densidade_atual, tendencia_36=g.tendencia_36),
                                        0.0)[baseline][0]
            media, lo, hi = bootstrap_diferenca(g[alvo].values, g.prob.values, esc_b, n=n_boot)
            linhas.append(dict(rodada=rodada, grupo=grupo, baseline=baseline, dif_pr_auc=AV.pr_auc(g[alvo], g.prob)
                               - AV.pr_auc(g[alvo], esc_b), ic95_inf=lo, ic95_sup=hi))
    return pd.DataFrame(linhas)
