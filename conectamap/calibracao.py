"""Etapa 1.5 — calibração da Regra D, comparação entre versões, impacto potencial
dos candidatos, lacunas temporais e evento de jan/2021.

Nada aqui altera densidades oficiais. As simulações (ex.: interpolação) são
gravadas em saidas/etapa1_5/simulacoes/ e marcadas como SIMULAÇÃO.
"""
import csv
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import anomalias as AN
from . import config as C

# --------------------------------------------------------------------------
# 1) Grade K x A x L com métricas de calibração
# --------------------------------------------------------------------------
_G = {}


def _casos_ref(ep):
    out = {}
    for chave in C.CASOS_REFERENCIA:
        cod, mes = chave.split("_")
        e = ep[(ep.id_municipio == cod) & (ep.inicio <= mes) & (ep.fim >= mes)]
        out[chave] = e.classificacao.iloc[0] if len(e) else "sem_episodio"
    return out


def _config(kal):
    K, A, L = kal
    p = dict(_G["base"], K=K, A=A, L=L)
    ep, _, _ = AN.rodar(_G["M"], _G["meses"], _G["ids"], _G["uf"], p)
    meses, pop = _G["meses"], _G["pop"]
    cand = ep[ep.candidato_tratamento]
    regs = {(int(r.mun), int(q)) for r in cand.itertuples() for q in r.dentro_idx}
    peq_mun = pop < 5000
    ep_peq = ep.mun.map(lambda r: bool(peq_mun[r]))
    res = dict(K=K, A=A, L=L, episodios=len(ep), municipios=ep.id_municipio.nunique(),
               episodios_candidatos=int(ep.candidato_tratamento.sum()), registros_candidatos=len(regs),
               municipios_com_candidatos=cand.id_municipio.nunique(),
               episodios_evento_coletivo=int(ep.evento_coletivo.sum()),
               episodios_base_menor_2=int((ep.base_ref < 2).sum()),
               pct_episodios_em_municipios_pequenos=float(ep_peq.mean()) if len(ep) else np.nan,
               sobrerrepresentacao_pequenos=(float(ep_peq.mean()) / float(peq_mun.mean())) if len(ep) else np.nan,
               registros_candidatos_em_fevereiro=sum(1 for _, q in regs if meses[q].month == 2),
               registros_candidatos_em_setembro=sum(1 for _, q in regs if meses[q].month == 9))
    for c in AN.CLASSES:
        res[c] = int((ep.classificacao == c).sum())
    por_ano = ep.groupby(ep.inicio.str[:4]).size().to_dict()
    por_uf = ep.groupby("sigla_uf").size().to_dict()
    return res, por_ano, por_uf, _casos_ref(ep), regs


def grade_calibracao(M, meses, ids, uf, pop_por_mun, base_params, processos=4):
    _G.update(M=M, meses=meses, ids=ids, uf=uf, base=base_params, pop=pop_por_mun)
    combos = [(K, A, L) for K in base_params["grade_K"] for A in base_params["grade_A"] for L in base_params["grade_L"]]
    try:
        with Pool(processos) as pool:
            out = pool.map(_config, combos)
    except Exception:
        out = [_config(c) for c in combos]
    resumo = pd.DataFrame([o[0] for o in out])
    anos = pd.DataFrame([o[1] for o in out], index=pd.MultiIndex.from_tuples(combos, names=["K", "A", "L"])).fillna(0).astype(int)
    ufs = pd.DataFrame([o[2] for o in out], index=anos.index).fillna(0).astype(int)
    casos = pd.DataFrame([o[3] for o in out], index=anos.index)
    regs = {c: o[4] for c, o in zip(combos, out)}
    # estabilidade: Jaccard médio dos registros candidatos com os vizinhos de um passo na grade
    g = {k: base_params[f"grade_{k}"] for k in "KAL"}
    est = []
    for (K, A, L) in combos:
        viz = []
        for nome, val in (("K", K), ("A", A), ("L", L)):
            i = g[nome].index(val)
            for j in (i - 1, i + 1):
                if 0 <= j < len(g[nome]):
                    viz.append({**dict(K=K, A=A, L=L), nome: g[nome][j]})
        jac = []
        for v in viz:
            a, b = regs[(K, A, L)], regs[(v["K"], v["A"], v["L"])]
            jac.append(len(a & b) / len(a | b) if (a | b) else 1.0)
        est.append(float(np.mean(jac)))
    resumo["estabilidade_jaccard_vizinhos"] = est
    return resumo, anos, ufs, casos


# --------------------------------------------------------------------------
# 2) Transição Etapa 1 -> Etapa 1.5
# --------------------------------------------------------------------------
def transicoes(ep1: pd.DataFrame, ep15: pd.DataFrame):
    a = ep1.assign(chave_episodio=ep1.id_municipio + "_" + ep1.inicio + "_" + ep1.direcao)
    b = ep15[["chave_episodio", "classificacao", "candidato_tratamento", "evento_coletivo", "fim", "duracao_meses"]]
    m = a.merge(b, on="chave_episodio", how="outer", suffixes=("_etapa1", "_etapa1_5"), indicator=True)
    m["classe_etapa1_nome_novo"] = m.classificacao_etapa1.map(AN.NOMES_ETAPA1)
    m["candidato_etapa1"] = m.metodo_proposto.eq("interpolacao_linear")
    m["candidato_etapa1_5"] = m.candidato_tratamento.fillna(False).astype(bool)

    def motivo(r):
        if r._merge != "both":
            return "episódio só em uma das versões"
        mudou_classe = r.classe_etapa1_nome_novo != r.classificacao_etapa1_5
        if mudou_classe and r.choque_comum:
            return "choque comum deixou de contar como evidência de artefato"
        if mudou_classe:
            return "outro (investigar)"
        if r.candidato_etapa1 and not r.candidato_etapa1_5 and r.evento_coletivo:
            return "mesma classe; evento coletivo bloqueia tratamento (H10)"
        if r.candidato_etapa1 != r.candidato_etapa1_5:
            return "mesma classe; elegibilidade mudou (investigar)"
        return "sem mudança (apenas nomenclatura)"
    m["motivo"] = m.apply(motivo, axis=1)
    matriz = (m.groupby(["classificacao_etapa1", "classificacao_etapa1_5", "candidato_etapa1", "candidato_etapa1_5", "motivo"],
                        dropna=False).size().rename("episodios").reset_index())
    return m, matriz


# --------------------------------------------------------------------------
# 3) Impacto potencial dos candidatos (sem aplicar nada)
# --------------------------------------------------------------------------
PERIODOS_CRITICOS = {"choque_2021 (2020-12 a 2021-03)": ("2020-12", "2021-03"),
                     "instabilidade (2025-11 a 2026-02)": ("2025-11", "2026-02")}


def impacto_candidatos(ep: pd.DataFrame, reg: pd.DataFrame, candidato: pd.Series, classes_cand, rotulo: str):
    """ep/reg de uma versão; `candidato` = máscara booleana dos episódios candidatos."""
    e = ep[candidato].copy()
    r = reg[reg.episodio_id.isin(e.episodio_id)].copy()
    comp = r.competencia
    n_cls = int(ep.classificacao.isin(classes_cand).sum())
    linhas = [
        ("1. Episódios candidatos x classes elegíveis",
         f"{len(e)} candidatos; {n_cls} episódios nas classes elegíveis; diferença {n_cls - len(e)} "
         f"(evento coletivo); classes dos candidatos: {e.classificacao.value_counts().to_dict()}"),
        ("2. Registros por categoria", r.merge(e[["episodio_id", "classificacao"]], on="episodio_id")
         .classificacao.value_counts().to_dict()),
        ("3. Municípios afetados", e.id_municipio.nunique()),
        ("4. Episódios com retorno estável (confirmado)", int((e.retorno == "confirmado").sum())),
        ("5. Episódios com duração > 3 meses", int((e.duracao_meses > 3).sum())),
    ]
    for nome, (a, b) in PERIODOS_CRITICOS.items():
        linhas.append((f"6. Registros em {nome}", int(((comp >= a) & (comp <= b)).sum())))
    linhas.append(("6. Registros em meses de fevereiro (âncora da previsão)", int(comp.str.endswith("-02").sum())))
    linhas.append(("6. Registros em meses de setembro (âncora do teste limpo)", int(comp.str.endswith("-09").sum())))
    col = "evento_coletivo" if "evento_coletivo" in e else "choque_comum"
    linhas.append(("6. Episódios em mês de evento coletivo", int(e[col].sum())))
    linhas.append(("7. Episódios com referência < 2 (perto de zero)", int((e.base_ref < 2).sum())))
    linhas.append(("7. Registros com valor original < 1", int((r.valor_original < 1).sum())))
    return pd.DataFrame(linhas, columns=["pergunta", rotulo])


def simular_tendencias(M, meses, ids, ep_cand: pd.DataFrame, limite="2026-02"):
    """SIMULAÇÃO: troca os registros candidatos por interpolação linear e mede
    quanto as variações de 12 e 24 meses mudariam. Nunca altera M."""
    pos = {k: i for i, k in enumerate(ids)}
    il = AN.indice_mes(meses, limite)
    linhas, det = [], []
    for cod, grupo in ep_cand.groupby("id_municipio"):
        r = pos[cod]
        v = M[r, : il + 1].copy()
        s = v.copy()
        for e in grupo.itertuples():
            a, b = AN.indice_mes(meses, e.inicio), AN.indice_mes(meses, e.fim)
            prev = next((k for k in range(a - 1, -1, -1) if v[k] == v[k]), None)
            nxt = next((k for k in range(b + 1, il + 1) if v[k] == v[k]), None)
            if prev is None or nxt is None:
                continue
            for q in range(a, b + 1):
                if v[q] == v[q]:
                    s[q] = v[prev] + (v[nxt] - v[prev]) * (q - prev) / (nxt - prev)
        for h in (12, 24):
            o = np.log((v[h:] + 1) / (v[:-h] + 1))
            x = np.log((s[h:] + 1) / (s[:-h] + 1))
            dif = np.abs(x - o)
            ok = ~np.isnan(dif)
            n_mud = int((dif[ok] > math.log(1.5)).sum())
            idx = np.where(ok & (np.nan_to_num(dif) > math.log(1.5)))[0] + h
            n_fev = int(sum(1 for q in idx if meses[q].month == 2))
            det.append(dict(id_municipio=cod, horizonte_meses=h, meses_com_variacao_alterada_mais_50pct=n_mud,
                            fevereiros_alterados_mais_50pct=n_fev,
                            maior_alteracao_vezes=float(np.exp(np.nanmax(dif))) if ok.any() else np.nan))
    d = pd.DataFrame(det)
    for h in (12, 24):
        x = d[d.horizonte_meses == h]
        linhas.append(dict(horizonte_meses=h, municipios_simulados=len(x),
                           municipios_com_tendencia_alterada_mais_50pct=int((x.meses_com_variacao_alterada_mais_50pct > 0).sum()),
                           municipio_meses_alterados_mais_50pct=int(x.meses_com_variacao_alterada_mais_50pct.sum()),
                           municipio_fevereiros_alterados_mais_50pct=int(x.fevereiros_alterados_mais_50pct.sum())))
    res = pd.DataFrame(linhas)
    res.insert(0, "natureza", "SIMULACAO — interpolação linear hipotética, não aplicada")
    d.insert(0, "natureza", "SIMULACAO")
    return res, d


# --------------------------------------------------------------------------
# 4) Lacunas temporais
# --------------------------------------------------------------------------
def meses_ausentes(anatel: pd.DataFrame, inicio="2011-01", fim="2026-02") -> pd.DataFrame:
    """Lacuna = competência do calendário sem registro, entre max(1ª observação do
    município, `inicio`) e `fim`. (Etapa 1 contava só saltos depois da 1ª
    observação de 2011 e por isso subestimava as lacunas.)"""
    t = anatel.ano * 12 + anatel.mes - 1
    ti = int(inicio[:4]) * 12 + int(inicio[5:]) - 1
    tf = int(fim[:4]) * 12 + int(fim[5:]) - 1
    linhas = []
    for cod, tt in t.groupby(anatel.id_municipio):
        pres = set(tt)
        for k in range(max(int(tt.min()), ti), tf + 1):
            if k not in pres:
                linhas.append((cod, k))
    return pd.DataFrame(linhas, columns=["id_municipio", "t"])


def lacunas(anatel: pd.DataFrame, caminho_bruto, nomes):
    aus = meses_ausentes(anatel)
    uf = anatel.drop_duplicates("id_municipio").set_index("id_municipio").sigla_uf
    t0 = (anatel.ano * 12 + anatel.mes - 1).groupby(anatel.id_municipio).min()
    linhas = []
    for cod, g in aus.groupby("id_municipio"):
        ks = sorted(g.t)
        blocos, ini, ant = [], ks[0], ks[0]
        for k in ks[1:] + [None]:
            if k is None or k != ant + 1:
                blocos.append((ini, ant))
                ini = k
            ant = k if k is not None else ant
        for a, b in blocos:
            linhas.append(dict(id_municipio=cod, municipio=nomes.get(cod, ""), sigla_uf=uf[cod],
                               de=f"{a // 12}-{a % 12 + 1:02d}", ate=f"{b // 12}-{b % 12 + 1:02d}", meses_ausentes=b - a + 1,
                               inicio_da_serie_no_municipio=f"{t0[cod] // 12}-{t0[cod] % 12 + 1:02d}"))
    lac = pd.DataFrame(linhas)
    # verificação independente no arquivo bruto (módulo csv, sem pandas)
    vistos, n_linhas = set(), 0
    with open(caminho_bruto, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            n_linhas += 1
            vistos.add((row[3].zfill(7), int(row[0]) * 12 + int(row[1]) - 1))
    ausentes_no_bruto = all((r.id_municipio, r.t) not in vistos for r in aus.itertuples())
    verif = dict(meses_ausentes=len(aus), municipios=aus.id_municipio.nunique(), intervalos=len(lac),
                 linhas_de_dados_no_bruto=n_linhas, registros_lidos=len(anatel), leitura_sem_perda=n_linhas == len(anatel),
                 lacunas_confirmadas_ausentes_no_bruto=ausentes_no_bruto,
                 juncao_envolvida="nenhuma (a série da Anatel é lida diretamente, sem junção)")
    return lac, verif


def impacto_lacunas_pares(anatel: pd.DataFrame, horizonte=36):
    pres = set(zip(anatel.id_municipio, anatel.ano, anatel.mes))
    ids = anatel.id_municipio.unique()
    linhas = []
    for mes, anos in ((2, range(2013, 2024)), (9, range(2012, 2023))):
        for t in anos:
            alvo_a = t + horizonte // 12
            sem_origem = [i for i in ids if (i, t, mes) not in pres]
            sem_alvo = [i for i in ids if (i, alvo_a, mes) not in pres]
            linhas.append(dict(ancora=f"{mes:02d}", origem=f"{t}-{mes:02d}", alvo=f"{alvo_a}-{mes:02d}",
                               municipios_sem_dado_na_origem=len(sem_origem), municipios_sem_dado_no_alvo=len(sem_alvo),
                               exemplos=";".join(sorted(set(sem_origem + sem_alvo))[:5])))
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# 5) Evento de jan/2021
# --------------------------------------------------------------------------
def evento_2021(ep15: pd.DataFrame, afetados: list, flags_causais, ids, meses):
    j = ep15[(ep15.inicio >= "2020-12") & (ep15.inicio <= "2021-02")]
    tab = j.groupby(["inicio", "classificacao", "evento_coletivo", "candidato_tratamento"]).size().rename("episodios").reset_index()
    af = set(afetados)
    resto21 = ep15[(ep15.inicio >= "2021-02") & (ep15.inicio <= "2021-12")]
    n_af, n_nao = len(af), len(ids) - len(af)
    taxa = dict(
        episodios_fev_dez_2021_em_afetados_por_100=100 * resto21.id_municipio.isin(af).sum() / max(n_af, 1),
        episodios_fev_dez_2021_em_nao_afetados_por_100=100 * (~resto21.id_municipio.isin(af)).sum() / max(n_nao, 1))
    i = AN.indice_mes(meses, "2021-01")
    pos = np.isin(ids, list(af))
    flags = dict(afetados=n_af, afetados_com_sinal_causal_em_jan_2021=int((flags_causais[pos, i] != 0).sum()))
    return tab, taxa, flags


def efeito_piso_A(ep: pd.DataFrame, A_alto=20):
    """Quantos episódios da configuração atual sumiriam com um piso A mais alto,
    por faixa de densidade de referência (baixa densidade = persona do projeto)."""
    dif = np.where(ep.direcao == "alta", ep.valor_extremo - ep.base_ref, ep.base_ref - ep.valor_extremo)
    e = ep.assign(diferenca_absoluta=dif, faixa_referencia=pd.cut(ep.base_ref, [0, 10, 20, 40, np.inf],
                                                                    labels=["<10", "10–20", "20–40", ">=40"], right=False))
    e["some_com_A_alto"] = e.diferenca_absoluta < A_alto
    return e.groupby(["faixa_referencia", "direcao"], observed=True).agg(
        episodios=("episodio_id", "size"), sumiriam_com_A_alto=("some_com_A_alto", "sum")).reset_index()


# --------------------------------------------------------------------------
# 6) Política de tratamento PROPOSTA (não aplicada) — para aprovação
# --------------------------------------------------------------------------
def politica_tratamento() -> pd.DataFrame:
    P = [
        ("Todas as categorias", "—",
         "densidade_original nunca muda; toda decisão fica no log (episódio, registro, revisor, data, versão)",
         "sobrescrever o valor original; corrigir só por ser extremo", "—", "—", "—"),
        ("atipico_retorno_consistente (fora de evento coletivo)",
         "critérios da regra + consenso 'A — forte evidência' na amostra revisada da categoria",
         "densidade_retro = ausente (NaN) com flag 'excluido_por_episodio'; interpolação só em análise de sensibilidade "
         "e apenas para episódios de até 2 meses com vizinhos válidos dentro da banda",
         "interpolar episódios > 2 meses; preencher sem vizinho posterior válido; aplicar à densidade_causal",
         "Sim: aprovação da categoria pela amostra; caso a caso se > 2 meses",
         "remove picos das variações de 12/24 meses; alvo retro na âncora fica ausente (regra de elegibilidade no NB02)",
         "baixo a médio: pode remover evento real curto"),
        ("atipico_com_retorno (fora de evento coletivo)",
         "critérios da regra; tratamento só se a revisão mostrar consenso A/B majoritário na categoria",
         "manter original com flag; NB02 roda com e sem esses registros (sensibilidade)",
         "tratamento automático sem aprovação da categoria",
         "Sim: aprovação da categoria", "pequeno se mantido; mensurável na sensibilidade", "médio"),
        ("ambiguo", "—", "manter original com flag 'ambiguo'; só sensibilidade",
         "qualquer substituição automática", "Opcional (casos extremos)", "nenhum direto", "baixo"),
        ("persistente_sem_retorno", "—",
         "manter original; flag de persistência; na série causal, referência atualizada após L meses (estado ±2)",
         "reverter ao nível antigo; interpolar; tratar como 'legítimo comprovado'",
         "Opcional: revisar casos > 10× ou ligados a evento coletivo",
         "features mostram o salto (é a informação disponível na época)", "médio: pode ser artefato persistente"),
        ("pendente_fim_serie", "—",
         "manter original com flag 'aberto'; NB02 testa o alvo fev/2026 com e sem esses municípios",
         "usar mar/2026 para confirmar ou corrigir; tratar como anomalia ou como patamar",
         "Não (reavaliar quando houver dados novos)", "afeta alvos e features da última origem", "alto para o teste fev/2026"),
        ("sem_referencia_suficiente", "—", "manter original com flag", "qualquer substituição", "Não",
         "nenhum direto", "baixo"),
        ("Episódio em evento coletivo (qualquer classe; ex.: jan/2021)", "investigação conjunta do mês (H10)",
         "manter original com flag 'evento_coletivo'; NB02 com e sem as origens/alvos afetados",
         "correção automática (individual ou em massa); atribuir causa sem documento",
         "Sim: decisão conjunta do grupo", "pode deslocar tendências de muitos municípios ao mesmo tempo", "alto"),
        ("Mês ausente (lacuna)", "ausência confirmada no arquivo bruto",
         "manter ausente (NaN); par origem/alvo inelegível se faltar o mês da âncora; features com mínimo de observações",
         "imputar zero; interpolar; copiar mês vizinho sem aprovação", "Não", "reduz pares elegíveis (poucos casos)", "baixo"),
        ("densidade_causal (features)", "—",
         "derivar só de densidade_original + flags causais (estados ±1/±2) calculadas com dados até t; opção a decidir: "
         "usar a referência causal (base_causal) no lugar de meses com desvio aberto",
         "usar densidade_retro, classificações retrospectivas ou mar/2026", "Não (regra fixa testada)",
         "protege contra vazamento", "baixo"),
    ]
    return pd.DataFrame(P, columns=["categoria", "evidencia_necessaria", "tratamento_recomendado", "tratamento_proibido",
                                    "revisao_humana", "impacto_nas_features", "risco_metodologico"])
