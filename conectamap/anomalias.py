"""B1/B2 — Detecção e classificação de episódios anômalos (regra D revisada).

IMPORTANTE (Etapas 1 e 1.5): este módulo DETECTA e CLASSIFICA. Ele não altera
nenhuma densidade e não propõe valores substitutos (a interpolação não está
aprovada; simulações ficam em calibracao.py, separadas). O "valor_aplicado" fica
vazio e o status de aplicação é "nao_autorizada_nao_aplicada".

As classes são ESTATÍSTICAS: não provam erro nem validade da fonte.

Resumo da regra (escala log com +1, para não explodir perto de zero):
  desvio(m) = ln((x_m + 1) / (base + 1))
  base      = mediana das até 6 observações válidas mais recentes nos 12 meses
              do calendário anteriores a m (mínimo 3). Meses de episódios já
              marcados não entram na referência (evita contaminação).
  entrada   = alta:  desvio >= ln K e x - base >= A
              queda: desvio <= -ln K e base - x >= A e base >= base_min_queda
  saída     = primeiro mês com |desvio| <= ln(2) (histerese: entra a K, sai a 2)
  retorno confirmado = mais 1 observação seguida dentro da banda
  persistente = sem retorno em até L meses do calendário (persistência observada;
              a validade da fonte NÃO é avaliada)
  pendente  = série termina (limite de observação) antes de retorno ou de L meses
"""
from datetime import date
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------
# Estrutura de dados: matriz município x mês do calendário (NaN = sem dado)
# --------------------------------------------------------------------------
def montar_matriz(anatel: pd.DataFrame):
    """Matriz município x mês do calendário. Ausências viram NaN (nunca zero).

    Falha de forma explícita (ValueError) diante de registros duplicados ou
    competências inválidas, em vez de sobrescrever valores em silêncio.
    """
    a = anatel.copy()
    dup = a.duplicated(["id_municipio", "ano", "mes"], keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} registros duplicados (município, ano, mês); "
                         "resolva antes de montar a matriz")
    if not a["mes"].between(1, 12).all() or (a["ano"] < 2007).any():
        raise ValueError("competência inválida (mês fora de 1–12 ou ano < 2007)")
    a["t"] = (a.ano - 2007) * 12 + (a.mes - 1)
    ids = np.sort(a.id_municipio.unique())
    pos = {k: i for i, k in enumerate(ids)}
    T = int(a.t.max()) + 1
    M = np.full((len(ids), T), np.nan)
    M[a.id_municipio.map(pos).values, a.t.values] = a.densidade.values
    meses = pd.period_range("2007-01", periods=T, freq="M")
    uf = a.drop_duplicates("id_municipio").set_index("id_municipio").loc[ids, "sigla_uf"].values
    return ids, uf, meses, M


def indice_mes(meses, comp: str) -> int:
    return int(np.where(meses == pd.Period(comp, "M"))[0][0])


# --------------------------------------------------------------------------
# Detecção (por município)
# --------------------------------------------------------------------------
def _detectar_municipio(v, i0, i1, p):
    lnK, lnB = math.log(p["K"]), math.log(p["fator_banda_saida"])
    A, L = p["A"], p["L"]
    jan, maxo, mino = p["janela_ref_meses"], p["max_obs_ref"], p["min_obs_ref"]
    ok = ~np.isnan(v)          # validade para referência
    eps = []
    m = i0
    while m <= i1:
        x = v[m]
        if x != x:  # NaN
            m += 1
            continue
        ref = [k for k in range(m - 1, max(-1, m - jan - 1), -1) if ok[k]][:maxo]
        if not ref:
            m += 1
            continue
        base = float(np.median(v[ref]))
        lr = math.log((x + 1) / (base + 1))
        up = lr >= lnK and (x - base) >= A
        dn = lr <= -lnK and (base - x) >= A and base >= p["base_min_queda"]
        if not (up or dn):
            m += 1
            continue
        prev = next((k for k in range(m - 1, -1, -1) if v[k] == v[k]), None)
        info = dict(ini=m, dir=1 if up else -1, base=base, ref_ini=min(ref), ref_fim=max(ref),
                    n_ref=len(ref), prev=prev, ref=sorted(ref))
        if len(ref) < mino:
            info.update(tipo="sem_referencia", dentro=[m], ret=None, ret_status=None)
            eps.append(info)
            m += 1
            continue
        dentro, k, ret = [], m, None
        while k <= i1:
            y = v[k]
            if y != y:
                k += 1
                continue
            if abs(math.log((y + 1) / (base + 1))) <= lnB:
                ret = k
                break
            if k - m >= L:
                break
            dentro.append(k)
            k += 1
        if ret is not None:
            nxt = [q for q in range(ret + 1, i1 + 1) if v[q] == v[q]][: p["obs_confirmacao_retorno"] - 1]
            if len(nxt) < p["obs_confirmacao_retorno"] - 1:
                rs = "nao_confirmado_fim"
            elif all(abs(math.log((v[q] + 1) / (base + 1))) <= lnB for q in nxt):
                rs = "confirmado"
            else:
                rs = "instavel"
            for q in dentro:
                ok[q] = False
            info.update(tipo="retornou", dentro=dentro, ret=ret, ret_status=rs)
            eps.append(info)
            m = ret + 1
        elif k > i1:
            for q in dentro:
                ok[q] = False
            info.update(tipo="pendente_fim", dentro=dentro, ret=None, ret_status=None)
            eps.append(info)
            break
        else:
            dentro.append(k)                     # L+1 meses fora da banda
            info.update(tipo="patamar", dentro=dentro, ret=None, ret_status=None)
            eps.append(info)                     # valores seguem válidos (novo nível)
            m = k + 1
    return eps, ok


def detectar(M, meses, params, inicio=None, limite=None):
    """Roda a detecção em todos os municípios. Usa só meses <= limite."""
    inicio = inicio or C.PARAMETROS_APROVADOS["inicio_periodo_mensal"]
    limite = limite or C.PARAMETROS_APROVADOS["limite_observacao"]
    i0, i1 = indice_mes(meses, inicio), indice_mes(meses, limite)
    Mv = M[:, : i1 + 1]                          # nada depois do limite entra na análise
    todos, OK = [], np.zeros_like(Mv, dtype=bool)
    for r in range(Mv.shape[0]):
        eps, ok = _detectar_municipio(Mv[r], i0, i1, params)
        OK[r] = ok
        for e in eps:
            e["mun"] = r
            todos.append(e)
    return todos, OK, (i0, i1)


# --------------------------------------------------------------------------
# Classificação (níveis de evidência)
# --------------------------------------------------------------------------
CLASSES = ["atipico_retorno_consistente", "atipico_com_retorno", "ambiguo",
           "persistente_sem_retorno", "pendente_fim_serie", "sem_referencia_suficiente"]
# Nomenclatura da Etapa 1 -> Etapa 1.5 (os critérios de cada classe estão em classificar())
NOMES_ETAPA1 = {"anomalia_estatisticamente_confirmada": "atipico_retorno_consistente",
                "anomalia_provavel": "atipico_com_retorno", "caso_ambiguo": "ambiguo",
                "mudanca_patamar": "persistente_sem_retorno", "pendente_fim_serie": "pendente_fim_serie",
                "sem_referencia_suficiente": "sem_referencia_suficiente"}
ELEGIVEIS = CLASSES[:2]   # candidatas a tratamento (sujeito à política e à revisão humana)


def classificar(eps, M, OK, meses, ids, uf, params, janela):
    i0, i1 = janela
    p = params
    lnB = math.log(p["fator_banda_saida"])
    linhas = []
    for e in eps:
        v = M[e["mun"]]
        dentro = [q for q in e["dentro"] if v[q] == v[q]]
        lrs = [math.log((v[q] + 1) / (e["base"] + 1)) for q in dentro]
        jmax = int(np.argmax(np.abs(lrs))) if lrs else 0
        mag = math.exp(abs(lrs[jmax])) if lrs else np.nan
        # entrada/saída abrupta x gradual
        ent = sai = None
        if e["prev"] is not None and lrs:
            passo = math.log((v[e["ini"]] + 1) / (v[e["prev"]] + 1))
            ent = abs(passo) >= p["fracao_rampa"] * abs(lrs[0]) and np.sign(passo) == np.sign(lrs[0])
        if e.get("ret") is not None and lrs:
            ult = dentro[-1]
            passo = math.log((v[e["ret"]] + 1) / (v[ult] + 1))
            sai = abs(passo) >= p["fracao_rampa"] * abs(lrs[-1]) and np.sign(passo) == -np.sign(lrs[-1])
        fim = dentro[-1] if dentro else e["ini"]
        linhas.append(dict(
            mun=e["mun"], id_municipio=ids[e["mun"]], sigla_uf=uf[e["mun"]],
            inicio=str(meses[e["ini"]]), fim=str(meses[fim]), ini_idx=e["ini"], fim_idx=fim,
            direcao="alta" if e["dir"] > 0 else "queda", tipo=e["tipo"],
            duracao_meses=fim - e["ini"] + 1, n_obs=len(dentro),
            base_ref=e["base"], ref_inicio=str(meses[e["ref_ini"]]), ref_fim=str(meses[e["ref_fim"]]),
            n_obs_ref=e["n_ref"], magnitude_vezes=mag,
            valor_extremo=v[dentro[jmax]] if dentro else np.nan,
            entrada=None if ent is None else ("abrupta" if ent else "gradual"),
            saida=None if sai is None else ("abrupta" if sai else "gradual"),
            retorno=e["ret_status"] if e["tipo"] == "retornou" else
                    ("sem_retorno_em_L" if e["tipo"] == "patamar" else
                     ("aberto_no_limite" if e["tipo"] == "pendente_fim" else "nao_avaliado")),
            mes_retorno=str(meses[e["ret"]]) if e.get("ret") is not None else "",
            ret_idx=e.get("ret"), prev_idx=e["prev"], dentro_idx=dentro, ref_idx=e["ref"]))
    ep = pd.DataFrame(linhas)
    if ep.empty:
        return ep
    for col in ("ret_idx", "prev_idx"):      # índices opcionais: int ou None (nunca float)
        ep[col] = pd.Series([None if (x is None or x != x) else int(x) for x in ep[col]], dtype=object)
    # Evento coletivo: muitos inícios no mesmo mês (acima do p95 mensal).
    # Etapa 1.5: NÃO é evidência de artefato local; bloqueia tratamento automático (H10).
    validos = ep[ep.tipo != "sem_referencia"]
    cont = validos.groupby("ini_idx").size().reindex(range(i0, i1 + 1), fill_value=0)
    lim_choque = max(float(np.percentile(cont.values, p["percentil_choque_comum"])),
                     float(p.get("min_inicios_evento_coletivo", 10)))
    ep["inicios_no_mes"] = ep.ini_idx.map(cont).fillna(0).astype(int)
    ep["limite_evento_coletivo"] = lim_choque
    ep["evento_coletivo"] = ep.inicios_no_mes > lim_choque
    # sinal extra 2: pico implausível na própria série (só altas)
    Mv = M[:, : i1 + 1]
    maxv = np.where(OK, Mv, np.nan)
    max_fora = np.nanmax(np.where(np.isnan(maxv), -np.inf, maxv), axis=1)
    ep["max_fora_episodios"] = ep.mun.map(lambda r: max_fora[r])
    ep["implausivel"] = (ep.direcao == "alta") & (ep.valor_extremo > p["fator_implausivel"] * ep.max_fora_episodios)
    ep["ida_e_volta_curta"] = ep.duracao_meses <= p["duracao_ida_volta"]

    def nivel(r):
        if r.tipo == "sem_referencia":
            return "sem_referencia_suficiente", "Menos de 3 observações válidas nos 12 meses anteriores."
        if r.tipo == "pendente_fim" or r.retorno == "nao_confirmado_fim":
            return "pendente_fim_serie", "Episódio aberto ou retorno sem confirmação até o limite de observação."
        if r.tipo == "patamar":
            return "persistente_sem_retorno", (f"Sem retorno em até {p['L']} meses: persistência observada. "
                                               "Não há evidência de retorno nem de artefato temporário; a validade da fonte não é avaliada.")
        abrupto = r.entrada == "abrupta" and r.saida == "abrupta"
        extra = r.ida_e_volta_curta or r.implausivel
        if r.retorno == "confirmado" and abrupto and r.magnitude_vezes >= p["k_confirmada"] and extra:
            sinais = [s for s, f in [("ida e volta curta", r.ida_e_volta_curta), ("pico implausível", r.implausivel)] if f]
            return "atipico_retorno_consistente", (
                f"Retorno confirmado, entrada e saída abruptas, magnitude {r.magnitude_vezes:.1f}x; sinais: {', '.join(sinais)}.")
        if r.retorno == "confirmado" and abrupto and r.magnitude_vezes >= p["k_provavel"] \
                and r.duracao_meses <= p["duracao_max_provavel"]:
            return "atipico_com_retorno", (
                f"Retorno confirmado, entrada e saída abruptas, magnitude {r.magnitude_vezes:.1f}x, {r.duracao_meses} mês(es).")
        motivos = []
        if r.retorno == "instavel": motivos.append("retorno instável")
        if r.entrada == "gradual": motivos.append("entrada gradual (rampa)")
        if r.saida == "gradual": motivos.append("saída gradual")
        if r.duracao_meses > p["duracao_max_provavel"]: motivos.append(f"duração de {r.duracao_meses} meses")
        if r.magnitude_vezes < p["k_provavel"]: motivos.append(f"magnitude {r.magnitude_vezes:.1f}x")
        return "ambiguo", "Retornou, mas: " + ("; ".join(motivos) or "sem sinal extra suficiente") + "."
    cj = ep.apply(nivel, axis=1, result_type="expand")
    ep["classificacao"], ep["justificativa"] = cj[0], cj[1]
    ep["candidato_tratamento"] = ep.classificacao.isin(ELEGIVEIS) & ~ep.evento_coletivo
    ep["motivo_nao_candidato"] = np.select(
        [ep.classificacao.isin(ELEGIVEIS) & ep.evento_coletivo, ~ep.classificacao.isin(ELEGIVEIS)],
        ["evento_coletivo (H10): exige investigação conjunta", "classe não elegível"], default="")
    ep["metodo_proposto"] = np.where(ep.candidato_tratamento, "a_definir_na_etapa2", "nenhum")
    # identificador estável (independe da ordem de processamento)
    ep.insert(0, "chave_episodio", ep.id_municipio + "_" + ep.inicio + "_" + ep.direcao)
    ep.insert(0, "episodio_id", [f"E{i:05d}" for i in range(len(ep))])
    return ep


def rodar(M, meses, ids, uf, params, limite=None):
    eps, OK, jan = detectar(M, meses, params, limite=limite)
    return classificar(eps, M, OK, meses, ids, uf, params, jan), OK, jan


# --------------------------------------------------------------------------
# B2 — Log por registro
# --------------------------------------------------------------------------
def log_registros(ep: pd.DataFrame, M, meses, params) -> pd.DataFrame:
    linhas = []
    hoje = date.today().isoformat()
    for r in ep.itertuples():
        v = M[r.mun]
        for q in r.dentro_idx:
            linhas.append(dict(
                id_municipio=r.id_municipio, sigla_uf=r.sigla_uf, competencia=str(meses[q]),
                valor_original=v[q], episodio_id=r.episodio_id, chave_episodio=r.chave_episodio,
                episodio_inicio=r.inicio, episodio_fim=r.fim,
                direcao=r.direcao, desvio_vezes=math.exp(abs(math.log((v[q] + 1) / (r.base_ref + 1)))),
                referencia_historica=r.base_ref, janela_referencia=f"{r.ref_inicio} a {r.ref_fim} ({r.n_obs_ref} obs)",
                duracao_meses=r.duracao_meses, evidencia_retorno=r.retorno, classificacao_provisoria=r.classificacao,
                evento_coletivo=bool(r.evento_coletivo), candidato_tratamento=bool(r.candidato_tratamento),
                etapa="deteccao_e_classificacao", status_revisao="aguardando_revisao",
                metodo_proposto=r.metodo_proposto, valor_proposto=np.nan,
                status_aprovacao="nao_aprovada", valor_aplicado=np.nan,
                status_aplicacao="nao_autorizada_nao_aplicada",
                versao_regra=C.VERSAO_REGRA_D,
                parametros=f"K={params['K']};A={params['A']};L={params['L']}",
                data_processamento=hoje, revisor=""))
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Sensibilidade (grade K x A x L)
# --------------------------------------------------------------------------
_G = {}


def _uma_config(kal):
    K, A, L = kal
    p = dict(_G["base"], K=K, A=A, L=L)
    ep, _, _ = rodar(_G["M"], _G["meses"], _G["ids"], _G["uf"], p)
    pr = ep[ep.inicio >= C.PARAMETROS_APROVADOS["inicio_revisao_prioritaria"]]
    linha = dict(K=K, A=A, L=L, episodios=len(ep), municipios=ep.id_municipio.nunique(),
                 episodios_2018_mais=len(pr),
                 registros_em_episodios=int(ep.n_obs.sum()),
                 registros_candidatos=int(ep[ep.candidato_tratamento].n_obs.sum()),
                 episodios_candidatos=int(ep.candidato_tratamento.sum()),
                 episodios_evento_coletivo=int(ep.evento_coletivo.sum()),
                 duracao_mediana=float(ep.duracao_meses.median()),
                 altas=int((ep.direcao == "alta").sum()), quedas=int((ep.direcao == "queda").sum()))
    for c in CLASSES:
        linha[c] = int((ep.classificacao == c).sum())
    return linha


def grade_sensibilidade(M, meses, ids, uf, base_params, processos=4):
    _G.update(M=M, meses=meses, ids=ids, uf=uf, base=base_params)
    combos = [(K, A, L) for K in base_params["grade_K"] for A in base_params["grade_A"] for L in base_params["grade_L"]]
    try:
        with Pool(processos) as pool:
            res = pool.map(_uma_config, combos)
    except Exception:                      # ambientes sem multiprocessing (ex.: Windows sem guarda)
        res = [_uma_config(c) for c in combos]
    return pd.DataFrame(res)


# --------------------------------------------------------------------------
# Comparação com a regra antiga (v1)
# --------------------------------------------------------------------------
def comparar_regra_antiga(log_antigo: pd.DataFrame, reg: pd.DataFrame, limite: str, inicio: str):
    la = log_antigo.copy()
    la["id_municipio"] = la["id_municipio"].astype(str).str.zfill(7)
    la["competencia"] = la.ano.astype(int).astype(str) + "-" + la.mes.astype(int).astype(str).str.zfill(2)
    lim_ok = la.competencia <= limite
    ini_ok = la.competencia >= inicio
    m = la.merge(reg[["id_municipio", "competencia", "classificacao_provisoria", "episodio_id"]],
                 on=["id_municipio", "competencia"], how="left")
    m["situacao_na_regra_nova"] = np.select(
        [~ini_ok.values, ~lim_ok.values, m.episodio_id.notna()],
        ["fora_do_escopo_temporal (antes de 2011)", "fora_do_limite_de_observacao (mar/2026)", "detectado"],
        default="nao_detectado")
    m.loc[m.situacao_na_regra_nova == "detectado", "situacao_na_regra_nova"] = \
        "detectado: " + m.classificacao_provisoria.astype(str)
    return m
