"""Etapa 2 — métricas e baselines FIXADOS ANTES DO TREINAMENTO (nenhum modelo é treinado aqui).

Métricas (por coorte = origem avaliada; nunca somadas entre coortes):
- prevalencia            proporção de positivos na coorte avaliada
- pr_auc                 precisão média (average precision): soma, sobre os positivos em ordem de risco,
                         da precisão acumulada no ponto em que cada positivo aparece, dividida pelo nº de
                         positivos. Empates de escore são avaliados juntos (bloco), sem desempate arbitrário
- precisao_top25 /       os k = ceil(25% × n) municípios de maior risco da coorte. Empate no corte: o bloco
  recall_top25           empatado contribui pelo VALOR ESPERADO (vagas restantes × proporção de positivos do
                         bloco), equivalente a desempatar ao acaso, sem depender do código do município
- brier                  média de (p − y)², só para escores que são probabilidades em [0, 1]
Toda métrica é calculada também no subgrupo de municípios ACIMA do P25 na origem (abaixo_p25_na_origem == 0).

Baselines (sem treinamento; usam só colunas do painel disponíveis na origem):
- B0 taxa_base       probabilidade constante = prevalência média das origens de treino da rodada (referência do Brier)
- B1 persistencia    positivo se abaixo do P25 na origem (escore 0/1; Brier com o próprio 0/1)
- B2 densidade_atual risco = −densidade_atual (menor densidade, maior risco); não é probabilidade (sem Brier)
- B3 tendencia       projeção linear = densidade_atual + 3 × tendencia_36 (acessos/100 dom. por ano × 3 anos);
                     risco = −projeção; sem tendência calculável, usa a densidade atual; sem Brier
"""
import math

import numpy as np
import pandas as pd

FRACAO_TOP = 0.25
METRICAS = ["prevalencia", "pr_auc", "precisao_top25", "recall_top25", "brier"]
BASELINES = ["B0_taxa_base", "B1_persistencia", "B2_densidade_atual", "B3_tendencia"]


def pr_auc(y, escore) -> float:
    """Precisão média com tratamento de empates em bloco."""
    y = np.asarray(y, float)
    s = np.asarray(escore, float)
    if y.sum() == 0:
        return float("nan")
    d = pd.DataFrame({"y": y, "s": s}).groupby("s", sort=True)["y"].agg(["sum", "count"]).iloc[::-1]
    tp = d["sum"].cumsum().values
    n = d["count"].cumsum().values
    prec = tp / n
    return float((prec * d["sum"].values).sum() / y.sum())


def top_fracao(y, escore, ids=None, fracao=FRACAO_TOP):
    """Precisão e recall nos k = ceil(fração × n) de maior risco; empates no corte pelo valor esperado.
    `ids` é aceito por compatibilidade e não influencia o resultado."""
    y = np.asarray(y, float)
    s = np.asarray(escore, float)
    k = math.ceil(fracao * len(y))
    blocos = pd.DataFrame({"s": s, "y": y}).groupby("s", sort=True)["y"].agg(["sum", "count"]).iloc[::-1]
    vagas, acertos = k, 0.0
    for soma, cont in zip(blocos["sum"].values, blocos["count"].values):
        if vagas <= 0:
            break
        usar = min(vagas, cont)
        acertos += soma * usar / cont
        vagas -= usar
    return float(acertos / k), (float(acertos / y.sum()) if y.sum() else float("nan"))


def brier(y, prob) -> float:
    p = np.asarray(prob, float)
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("Brier só para probabilidades em [0, 1]")
    return float(np.mean((p - np.asarray(y, float)) ** 2))


def avaliar(y, escore, ids, prob=None) -> dict:
    prec, rec = top_fracao(y, escore, ids)
    return dict(n=len(y), prevalencia=float(np.mean(y)), pr_auc=pr_auc(y, escore), precisao_top25=prec,
                recall_top25=rec, brier=brier(y, prob) if prob is not None else float("nan"))


def scores_baselines(coorte: pd.DataFrame, prevalencia_treino: float) -> dict:
    """{nome: (escore de risco, probabilidade ou None)} para uma coorte (linhas elegíveis de uma origem)."""
    atual = coorte.densidade_atual.astype(float).values
    proj = atual + 3 * coorte.tendencia_36.astype(float).values
    proj = np.where(np.isnan(proj), atual, proj)
    b1 = coorte.abaixo_p25_na_origem.astype(float).values
    return {
        "B0_taxa_base": (np.full(len(coorte), prevalencia_treino), np.full(len(coorte), prevalencia_treino)),
        "B1_persistencia": (b1, b1),
        "B2_densidade_atual": (-atual, None),
        "B3_tendencia": (-proj, None),
    }


def avaliar_coorte(coorte: pd.DataFrame, escore, prob=None, alvo="alvo_p25_origem") -> pd.DataFrame:
    """Métricas na coorte inteira e no subgrupo acima do P25 na origem."""
    c = coorte.reset_index(drop=True)
    y = c[alvo].astype(float).values
    escore = np.asarray(escore, float)
    prob = None if prob is None else np.asarray(prob, float)
    acima = (c.abaixo_p25_na_origem == 0).values
    linhas = [dict(grupo="todos", **avaliar(y, escore, c.id_municipio, prob)),
              dict(grupo="acima_p25_na_origem", **avaliar(y[acima], escore[acima], c.id_municipio[acima],
                                                         None if prob is None else prob[acima]))]
    return pd.DataFrame(linhas)
