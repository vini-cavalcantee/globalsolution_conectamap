"""B3 — Séries original, retrospectiva e causal + testes de causalidade.

- densidade_original: valores da fonte, nunca alterados.
- densidade_retro:    destino de tratamentos retrospectivos (podem usar o futuro).
                      Etapa 1: IGUAL à original, status "nao_tratada".
- densidade_causal:   base das features. Só usa informação até a competência.
                      Etapas 1 e 1.5: IGUAL à original (sem correção), acompanhada
                      de uma sinalização causal calculada mês a mês com dados até
                      aquele mês.

Estados da sinalização causal (flag_causal):
   0  normal (dentro da referência ou sem referência suficiente)
  +1  desvio de alta em aberto        -1  desvio de queda em aberto
  +2  persistência de alta reconhecida -2  persistência de queda reconhecida
      (mês de transição: o desvio durou L meses sem retorno; a partir do mês
       seguinte a referência passa a incluir o novo nível. Persistência NÃO é
       prova de validade da fonte.)

Correção da Etapa 1.5: na Etapa 1, o mês de transição recebia 0 (igual a um mês
normal) mesmo continuando fora da referência, e a série retrospectiva o incluía
no episódio. Agora ele recebe ±2.
"""
import math

import numpy as np
import pandas as pd

from . import config as C
from .anomalias import indice_mes


# --------------------------------------------------------------------------
# Sinalização causal (só passado e presente)
# --------------------------------------------------------------------------
def _causal_municipio(v, i0, iend, p):
    lnK, lnB = math.log(p["K"]), math.log(p["fator_banda_saida"])
    ok = ~np.isnan(v)
    flag = np.zeros(len(v), dtype=np.int8)
    base_m = np.full(len(v), np.nan)
    aberto, base_ab, ini, cand, dir_ = False, None, None, [], 0
    for m in range(i0, iend + 1):
        x = v[m]
        if x != x:
            continue
        if aberto:
            if abs(math.log((x + 1) / (base_ab + 1))) <= lnB:
                aberto = False                      # voltou: meses do episódio seguem fora da referência
            elif m - ini >= p["L"]:
                for q in cand:                      # decisão tomada em m (dados <= m): persistência reconhecida
                    ok[q] = True
                flag[m] = 2 * dir_                  # mês de transição continua sinalizado (Etapa 1.5)
                base_m[m] = base_ab
                aberto = False
            else:
                flag[m] = dir_
                base_m[m] = base_ab
                ok[m] = False
                cand.append(m)
            continue
        ref = [k for k in range(m - 1, max(-1, m - p["janela_ref_meses"] - 1), -1) if ok[k]][: p["max_obs_ref"]]
        if len(ref) < p["min_obs_ref"]:
            continue
        base = float(np.median(v[ref]))
        lr = math.log((x + 1) / (base + 1))
        up = lr >= lnK and (x - base) >= p["A"]
        dn = lr <= -lnK and (base - x) >= p["A"] and base >= p["base_min_queda"]
        if up or dn:
            aberto, base_ab, ini, cand, dir_ = True, base, m, [m], (1 if up else -1)
            flag[m] = dir_
            base_m[m] = base
            ok[m] = False
    return flag, base_m


ESTADOS_CAUSAIS = {0: "normal", 1: "desvio_alta_aberto", -1: "desvio_queda_aberto",
                   2: "persistencia_alta_reconhecida", -2: "persistencia_queda_reconhecida"}


def sinalizar_causal(M, meses, params, ate=None):
    """Retorna (flags, bases) para os meses do período mensal até `ate` (ou o limite)."""
    i0 = indice_mes(meses, C.PARAMETROS_APROVADOS["inicio_periodo_mensal"])
    iend = indice_mes(meses, ate or C.PARAMETROS_APROVADOS["limite_observacao"])
    iend = min(iend, M.shape[1] - 1)
    F = np.zeros((M.shape[0], iend + 1), dtype=np.int8)
    B = np.full((M.shape[0], iend + 1), np.nan)
    for r in range(M.shape[0]):
        f, b = _causal_municipio(M[r, : iend + 1], i0, iend, params)
        F[r], B[r] = f, b
    return F, B


# --------------------------------------------------------------------------
# Estrutura das três séries
# --------------------------------------------------------------------------
def construir_series(anatel: pd.DataFrame, reg_log: pd.DataFrame, ids, meses, params):
    """Tabela longa com as três representações. A série causal é calculada
    EXCLUSIVAMENTE a partir de `densidade_original`."""
    s = anatel.rename(columns={"densidade": "densidade_original"}).copy()
    s["competencia"] = s.ano.astype(str) + "-" + s.mes.astype(str).str.zfill(2)
    lim = C.PARAMETROS_APROVADOS["limite_observacao"]
    s["dentro_limite_observacao"] = s.competencia <= lim
    s["densidade_retro"] = s["densidade_original"]
    s["status_retro"] = "nao_tratada"                       # Etapa 1: nenhuma correção aprovada
    s["densidade_causal"] = s["densidade_original"]
    s["status_causal"] = "sem_correcao_etapa1"
    s = s.merge(reg_log[["id_municipio", "competencia", "episodio_id", "classificacao_provisoria"]],
                on=["id_municipio", "competencia"], how="left")
    s = s.rename(columns={"episodio_id": "retro_episodio_id", "classificacao_provisoria": "retro_classificacao"})
    return s


def flags_causais_da_tabela(series: pd.DataFrame, meses, params):
    """Recalcula as flags causais lendo SÓ a coluna densidade_original da tabela."""
    from .anomalias import montar_matriz
    base = series[["ano", "mes", "sigla_uf", "id_municipio", "densidade_original"]].rename(
        columns={"densidade_original": "densidade"})
    _, _, _, M = montar_matriz(base)
    return sinalizar_causal(M, meses, params)


# --------------------------------------------------------------------------
# Testes de causalidade (8.3)
# --------------------------------------------------------------------------
def teste_truncamento(M, meses, params, datas):
    F_full, _ = sinalizar_causal(M, meses, params)
    res = []
    for d in datas:
        t = indice_mes(meses, d)
        F_t, _ = sinalizar_causal(M[:, : t + 1], meses, params, ate=d)
        iguais = np.array_equal(F_full[:, : t + 1], F_t)
        res.append(dict(data_corte=d, flags_ate_t_identicas=bool(iguais), n_flags_ate_t=int((F_t != 0).sum())))
    return pd.DataFrame(res)


def teste_perturbacao(M, meses, params, datas, semente=42):
    rng = np.random.default_rng(semente)
    F_ref, B_ref = sinalizar_causal(M, meses, params)
    res = []
    for d in datas:
        t = indice_mes(meses, d)
        P = M.copy()
        fut = P[:, t + 1:]
        fut *= rng.uniform(0.1, 10, size=fut.shape)          # altera todo o futuro
        fut[rng.random(fut.shape) < 0.1] = np.nan             # e apaga 10% dele
        F_p, B_p = sinalizar_causal(P, meses, params)
        iguais = np.array_equal(F_ref[:, : t + 1], F_p[:, : t + 1]) and \
            np.array_equal(np.nan_to_num(B_ref[:, : t + 1], nan=-1), np.nan_to_num(B_p[:, : t + 1], nan=-1))
        mudou_futuro = not np.array_equal(F_ref[:, t + 1:], F_p[:, t + 1:])
        res.append(dict(data_corte=d, flags_e_bases_ate_t_identicas=bool(iguais),
                        futuro_foi_alterado_e_flags_futuras_mudaram=bool(mudou_futuro)))
    return pd.DataFrame(res)
