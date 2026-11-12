"""Etapa 2 — painel temporal (origem × alvo em 36 meses), sem treinamento de modelos.

Entradas: SOMENTE as bases exportadas pelo Notebook 1 em `dados_tratados/` (não são alteradas).
Regras principais:
- limite de observação: 2026-02. 2026-03 não entra no painel (nem como feature nem como alvo);
- features usam apenas meses <= origem (a matriz é fatiada até a coluna da origem antes de qualquer cálculo);
- `densidade_retro` e as colunas `retro_*` dos indicadores não são lidas;
- variáveis do Censo só entram quando disponíveis na origem pelo critério escolhido;
- o alvo usa `densidade_original` no mês origem + horizonte; se esse mês não existe ou passa do limite, o alvo fica vazio.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import disponibilidade as DISP

VERSAO_PAINEL = "etapa2-painel-v1"
LIMITE = C.PARAMETROS_APROVADOS["limite_observacao"]            # 2026-02
INICIO_MENSAL = C.PARAMETROS_APROVADOS["inicio_periodo_mensal"]  # 2011-01
HORIZONTE = C.PARAMETROS_APROVADOS["horizonte_meses"]           # 36
TOLERANCIA_LAG = 2   # meses: valor defasado = última observação em [t-k-2, t-k] (cobre os anos trimestrais)

# Datas registradas no projeto para as tabelas do Censo (fonte: metadados da Etapa 1.5,
# auditoria_tecnica/saidas/etapa1_5/metadados/metadados_fontes.csv). A atualização do SIDRA é um
# LIMITE SUPERIOR da disponibilidade; a primeira divulgação não foi verificada.
METADADOS_CENSO = pd.DataFrame([
    dict(chave="ibge_9923", tabela="9923", data_referencia="2022-07-31", data_atualizacao_sidra="14/11/2024", data_primeira_divulgacao="não verificada"),
    dict(chave="ibge_9515", tabela="9515", data_referencia="2022-07-31", data_atualizacao_sidra="22/12/2023", data_primeira_divulgacao="não verificada"),
    dict(chave="ibge_6805", tabela="6805", data_referencia="2022-07-31", data_atualizacao_sidra="23/02/2024", data_primeira_divulgacao="não verificada"),
    dict(chave="ibge_10295", tabela="10295", data_referencia="2022-07-31", data_atualizacao_sidra="09/10/2025", data_primeira_divulgacao="não verificada"),
    dict(chave="ibge_10125", tabela="10125", data_referencia="2022-07-31", data_atualizacao_sidra="23/05/2025", data_primeira_divulgacao="não verificada"),
])
VARIAVEIS_CENSO = list(DISP.FONTES_VARIAVEIS)


# --------------------------------------------------------------------------
# 1. Carregamento e integração
# --------------------------------------------------------------------------
def carregar_bases(pasta=None) -> dict:
    """Lê as bases tratadas e monta matrizes município × mês (2007-01 .. 2026-02).
    Só leitura; falha (ValueError) diante de duplicidade ou divergência de municípios."""
    pasta = C.DIR_TRATADOS if pasta is None else pasta
    ids_str = {"id_municipio": str}
    base = pd.read_csv(pasta / "anatel_densidade_tratada.csv", dtype=ids_str,
                       usecols=["id_municipio", "sigla_uf", "competencia", "periodicidade", "dentro_limite_observacao",
                                "densidade_original", "densidade_causal"])
    ind = pd.read_csv(pasta / "anatel_indicadores_qualidade.csv", dtype=ids_str,
                      usecols=["id_municipio", "competencia", "dentro_limite_observacao", "flag_causal",
                               "flag_causal_L6", "referencia_insuficiente"])   # nenhuma coluna retro_* é lida
    aus = pd.read_csv(pasta / "anatel_meses_ausentes.csv", dtype=ids_str)
    ibge = pd.read_csv(pasta / "ibge_municipal_2022_tratada.csv", dtype=ids_str)
    return integrar(base, ind, aus, ibge)


def integrar(base, ind, aus, ibge) -> dict:
    base = base[base.dentro_limite_observacao.astype(bool)]          # 2026-03 fora do painel
    ind = ind[ind.dentro_limite_observacao.astype(bool)]
    for nome, df in (("base", base), ("indicadores", ind)):
        if df.duplicated(["id_municipio", "competencia"]).any():
            raise ValueError(f"{nome}: registros duplicados (município, competência)")
    if not base[["id_municipio", "competencia"]].reset_index(drop=True).equals(ind[["id_municipio", "competencia"]].reset_index(drop=True)):
        raise ValueError("base e indicadores não têm as mesmas chaves na mesma ordem")
    if ibge.id_municipio.duplicated().any():
        raise ValueError("IBGE: município duplicado")
    ids = np.sort(base.id_municipio.unique())
    if set(ibge.id_municipio) != set(ids):
        raise ValueError("IBGE e Anatel com conjuntos de municípios diferentes")
    if base.competencia.max() > LIMITE:
        raise ValueError("competência posterior ao limite de observação no painel")
    meses = pd.period_range("2007-01", LIMITE, freq="M")
    col = {str(m): j for j, m in enumerate(meses)}
    lin = {k: i for i, k in enumerate(ids)}
    r, c = base.id_municipio.map(lin).values, base.competencia.map(col).values

    def matriz(valores, dtype=float):
        m = np.full((len(ids), len(meses)), np.nan)
        m[r, c] = valores
        return m
    flag12 = matriz(pd.to_numeric(ind.flag_causal, errors="coerce").values)
    flag6 = matriz(pd.to_numeric(ind.flag_causal_L6, errors="coerce").values)
    refins = matriz(ind.referencia_insuficiente.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).values)
    ausente = np.zeros((len(ids), len(meses)), bool)
    a = aus[aus.competencia <= LIMITE]
    ausente[a.id_municipio.map(lin).values, a.competencia.map(col).values] = True
    uf = base.drop_duplicates("id_municipio").set_index("id_municipio").sigla_uf.reindex(ids).values
    return dict(ids=ids, uf=uf, meses=meses, M=matriz(base.densidade_causal.values),
                M_original=matriz(base.densidade_original.values), flag12=flag12, flag6=flag6,
                refins=refins, ausente=ausente, ibge=ibge.set_index("id_municipio").reindex(ids),
                n_registros=len(base))


_CACHE = {}


def dados_padrao() -> dict:
    if "dados" not in _CACHE:
        _CACHE["dados"] = carregar_bases()
    return _CACHE["dados"]


# --------------------------------------------------------------------------
# 2. Features causais (só colunas <= origem)
# --------------------------------------------------------------------------
def _ultimo_valido(X, ini, fim):
    """Último valor não ausente de cada linha entre as colunas ini..fim (inclusive); NaN se nenhum."""
    ini = max(ini, 0)
    if fim < ini:
        return np.full(X.shape[0], np.nan)
    bloco = X[:, ini:fim + 1]
    ok = ~np.isnan(bloco)
    pos = np.where(ok.any(1), bloco.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
    return np.where(pos >= 0, bloco[np.arange(len(bloco)), np.maximum(pos, 0)], np.nan)


def _inclinacao_anual(bloco, min_obs):
    """Inclinação OLS (por ano) dos valores válidos de cada linha; NaN com menos de `min_obs` pontos."""
    x = np.arange(bloco.shape[1], dtype=float)
    ok = ~np.isnan(bloco)
    n = ok.sum(1)
    xm = np.where(ok, x, 0).sum(1) / np.maximum(n, 1)
    ym = np.nansum(bloco, 1) / np.maximum(n, 1)
    dx = np.where(ok, x - xm[:, None], 0)
    dy = np.where(ok, bloco - ym[:, None], 0)
    sxx = (dx ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (dx * dy).sum(1) / sxx * 12
    return np.where((n >= min_obs) & (sxx > 0), b, np.nan)


def _volatilidade(bloco, min_dif):
    """Desvio-padrão das variações ln((x+1)/(x_ant+1)) entre observações válidas consecutivas."""
    out = np.full(bloco.shape[0], np.nan)
    for i, linha in enumerate(bloco):
        v = linha[~np.isnan(linha)]
        if len(v) - 1 >= min_dif:
            out[i] = np.std(np.diff(np.log1p(v)), ddof=1)
    return out


def features_causais(dados: dict, t: int) -> pd.DataFrame:
    """Features na coluna de origem t. As matrizes são cortadas em t+1 ANTES de qualquer cálculo."""
    M = dados["M"][:, : t + 1]
    F12, F6 = dados["flag12"][:, : t + 1], dados["flag6"][:, : t + 1]
    RI, AU = dados["refins"][:, : t + 1], dados["ausente"][:, : t + 1]
    i_mensal = int(np.where(dados["meses"] == pd.Period(INICIO_MENSAL, "M"))[0][0])
    f = {"densidade_atual": M[:, t]}
    for k in (12, 24, 36):
        lag = _ultimo_valido(M, t - k - TOLERANCIA_LAG, t - k)
        f[f"densidade_lag_{k}"] = lag
        f[f"variacao_abs_{k}"] = M[:, t] - lag
        f[f"variacao_log_{k}"] = np.log1p(M[:, t]) - np.log1p(lag)
    for k in (12, 36):
        bloco = M[:, max(t - k + 1, 0): t + 1]
        with warnings.catch_warnings():   # linhas sem nenhuma observação na janela -> NaN, sem aviso
            warnings.simplefilter("ignore", RuntimeWarning)
            f[f"media_{k}"] = np.nanmean(bloco, 1)
            f[f"mediana_{k}"] = np.nanmedian(bloco, 1)
        f[f"n_obs_{k}"] = (~np.isnan(bloco)).sum(1)
        f[f"volatilidade_{k}"] = _volatilidade(bloco, 4 if k == 12 else 8)
    for k in (24, 36):
        f[f"tendencia_{k}"] = _inclinacao_anual(M[:, max(t - k + 1, 0): t + 1], 6)
    janela36 = slice(max(t - 35, 0), t + 1)
    f["n_meses_ausentes_36"] = AU[:, janela36].sum(1)
    f["fracao_trimestral_36"] = np.mean(np.arange(max(t - 35, 0), t + 1) < i_mensal) * np.ones(len(M))
    primeira = np.argmax(~np.isnan(M), axis=1).astype(float)
    primeira[np.isnan(M).all(1)] = np.nan
    f["meses_desde_primeira_obs"] = t - primeira
    for sufixo, Fm in (("", F12), ("_L6", F6)):
        f[f"flag_causal_atual{sufixo}"] = Fm[:, t]
        f[f"n_suspeita_12{sufixo}"] = (np.abs(Fm[:, max(t - 11, 0): t + 1]) == 1).sum(1)
        f[f"n_suspeita_36{sufixo}"] = (np.abs(Fm[:, janela36]) == 1).sum(1)
        f[f"persistencia_36{sufixo}"] = (np.abs(Fm[:, janela36]) == 2).any(1).astype(int)
    sinal = np.abs(F12) >= 1
    ult = np.where(sinal.any(1), t - (sinal.shape[1] - 1 - np.argmax(sinal[:, ::-1], axis=1)), np.nan)
    f["meses_desde_ultima_sinalizacao"] = ult
    f["referencia_insuficiente_atual"] = RI[:, t]
    return pd.DataFrame(f)


# --------------------------------------------------------------------------
# 3. Censo — disponibilidade na origem
# --------------------------------------------------------------------------
def inventario_censo() -> pd.DataFrame:
    return DISP.inventario_variaveis(VARIAVEIS_CENSO, METADADOS_CENSO)


def variaveis_censo_disponiveis(ancora: str, criterio: str) -> list:
    d = DISP.disponivel(inventario_censo(), ancora, criterio)
    return list(d[d].index)


# --------------------------------------------------------------------------
# 4. Painel de uma origem
# --------------------------------------------------------------------------
def montar_painel(ancora: str, horizonte: int = HORIZONTE, criterio_censo: str = "publicacao",
                  dados: dict | None = None, limiar_absoluto: float | None = None) -> pd.DataFrame:
    """Painel de uma origem: uma linha por município, com features até a origem e o alvo em origem + horizonte.

    ancora           competência de origem 'AAAA-MM', entre 2011-01 e 2026-02
    horizonte        meses até o alvo (36)
    criterio_censo   'publicacao' (principal), 'conteudo' ou 'nenhum' (sensibilidade)
    limiar_absoluto  se informado, acrescenta o alvo pelo limiar absoluto fixo
    """
    dados = dados_padrao() if dados is None else dados
    p = pd.Period(ancora, "M")
    if p > pd.Period(LIMITE, "M"):
        raise ValueError(f"origem {ancora} depois do limite de observação {LIMITE}")
    if p < pd.Period(INICIO_MENSAL, "M"):
        raise ValueError(f"origem {ancora} antes do período mensal ({INICIO_MENSAL})")
    if criterio_censo not in DISP.CRITERIOS:
        raise ValueError(f"critério desconhecido: {criterio_censo}")
    meses = dados["meses"]
    t = int(np.where(meses == p)[0][0])
    alvo_mes = p + horizonte
    alvo_observavel = alvo_mes <= pd.Period(LIMITE, "M")

    painel = pd.DataFrame({"id_municipio": dados["ids"], "sigla_uf": dados["uf"], "origem": str(p),
                           "mes_alvo": str(alvo_mes), "horizonte_meses": horizonte})
    feats = features_causais(dados, t)
    painel = pd.concat([painel, feats], axis=1)

    # elegibilidade
    painel["obs_na_origem"] = ~np.isnan(feats.densidade_atual.values)
    painel["alvo_observavel"] = bool(alvo_observavel)
    if alvo_observavel:
        ta = int(np.where(meses == alvo_mes)[0][0])
        dens_alvo = dados["M_original"][:, ta]
    else:
        dens_alvo = np.full(len(painel), np.nan)
    painel["densidade_alvo"] = dens_alvo
    painel["obs_no_alvo"] = ~np.isnan(dens_alvo)
    painel["elegivel_features"] = painel.obs_na_origem
    painel["elegivel_treino"] = painel.obs_na_origem & painel.obs_no_alvo

    # alvo 1: P25 da origem, congelado (calculado só com municípios observados na origem)
    limiar = float(np.nanquantile(feats.densidade_atual.values, 0.25))
    painel["limiar_p25_origem"] = limiar
    painel["abaixo_p25_na_origem"] = np.where(painel.obs_na_origem, feats.densidade_atual < limiar, np.nan)
    painel["alvo_p25_origem"] = np.where(painel.elegivel_treino, dens_alvo < limiar, np.nan)
    # alvo 2: limiar absoluto fixo (opcional)
    if limiar_absoluto is not None:
        painel["limiar_absoluto"] = float(limiar_absoluto)
        painel["alvo_absoluto"] = np.where(painel.elegivel_treino, dens_alvo < limiar_absoluto, np.nan)

    # Censo: só as variáveis disponíveis na origem pelo critério; as demais ficam vazias
    disp = variaveis_censo_disponiveis(str(p), criterio_censo)
    for v in VARIAVEIS_CENSO:
        painel[f"censo_{v}"] = dados["ibge"][v].values if v in disp else np.nan
    painel["criterio_censo"] = criterio_censo
    painel["n_variaveis_censo"] = len(disp)
    painel["versao_painel"] = VERSAO_PAINEL
    return painel


def montar_paineis(ancoras, horizonte=HORIZONTE, criterio_censo="publicacao", dados=None, limiar_absoluto=None):
    return pd.concat([montar_painel(a, horizonte, criterio_censo, dados, limiar_absoluto) for a in ancoras],
                     ignore_index=True)


def colunas_features(painel: pd.DataFrame, incluir_censo=True, incluir_L6=False) -> list:
    """Colunas que podem ser usadas como feature (nada do alvo, nada retrospectivo)."""
    proibidas = {"id_municipio", "sigla_uf", "origem", "mes_alvo", "horizonte_meses", "obs_na_origem",
                 "alvo_observavel", "densidade_alvo", "obs_no_alvo", "elegivel_features", "elegivel_treino",
                 "limiar_p25_origem", "alvo_p25_origem", "limiar_absoluto", "alvo_absoluto", "criterio_censo",
                 "n_variaveis_censo", "versao_painel"}
    cols = [c for c in painel.columns if c not in proibidas]
    if not incluir_censo:
        cols = [c for c in cols if not c.startswith("censo_")]
    if not incluir_L6:
        cols = [c for c in cols if not c.endswith("_L6")]
    return cols


# --------------------------------------------------------------------------
# 5. Backtesting temporal (desenho; nenhum modelo é treinado aqui)
# --------------------------------------------------------------------------
def origens_fevereiro(inicio=2014, fim=2023):
    return [f"{a}-02" for a in range(inicio, fim + 1)]


REGRAS_BACKTESTING = {
    # hipótese: os dados da Anatel de uma competência estão disponíveis na data de origem dessa mesma competência
    "atual": "alvo de treino <= origem avaliada (mesma hipótese de disponibilidade usada nas features)",
    # alternativa: o alvo de treino precisa ser ANTERIOR à origem avaliada
    "conservadora": "alvo de treino < origem avaliada (exige ao menos um mês entre o último alvo de treino e a origem)",
}


def _alvo_conhecido(origem_treino, data_avaliada, horizonte, regra):
    alvo = pd.Period(origem_treino, "M") + horizonte
    return alvo <= data_avaliada if regra == "atual" else alvo < data_avaliada


def folds_backtesting(origens_treino_min="2014-02", validacoes=("2019-02", "2020-02", "2021-02", "2022-02"),
                      teste="2023-02", horizonte=HORIZONTE, regra="atual") -> pd.DataFrame:
    """Janela expansiva: em cada rodada, o treino só usa origens cujo alvo (origem + horizonte) já seria
    conhecido na data da origem de validação. O teste fica fora de qualquer seleção.
    regra='atual': alvo de treino <= origem avaliada; regra='conservadora': alvo de treino < origem avaliada."""
    if regra not in REGRAS_BACKTESTING:
        raise ValueError(f"regra desconhecida: {regra}")
    linhas = []
    p0, pt = pd.Period(origens_treino_min, "M"), pd.Period(teste, "M")
    todas = [str(p0 + 12 * i) for i in range((pt - p0).n // 12 + 1)]   # mesma posição no ano (fevereiro)
    for papel, v in [("validacao", x) for x in validacoes] + [("teste", teste)]:
        pv = pd.Period(v, "M")
        treino = [o for o in todas if _alvo_conhecido(o, pv, horizonte, regra)]
        linhas.append(dict(rodada=f"{papel}_{v}", papel=papel, origem_avaliada=v, mes_alvo_avaliado=str(pv + horizonte),
                           origens_treino=",".join(treino), n_origens_treino=len(treino),
                           ultimo_alvo_treino=str(pd.Period(treino[-1], "M") + horizonte) if treino else ""))
    final = [o for o in todas if _alvo_conhecido(o, pd.Period(LIMITE, "M"), horizonte, regra)]
    linhas.append(dict(rodada="final_2026-02", papel="previsao", origem_avaliada="2026-02", mes_alvo_avaliado="2029-02",
                       origens_treino=",".join(final), n_origens_treino=len(final),
                       ultimo_alvo_treino=str(pd.Period(final[-1], "M") + horizonte) if final else ""))
    out = pd.DataFrame(linhas)
    if regra != "atual":
        out.insert(1, "regra", regra)
    return out


# Modelo principal: SOMENTE features temporais da Anatel (lista congelada antes do treino).
# Sem Censo (ausente em todas as origens históricas de treino pelo critério de publicação) e sem L=6.
FEATURES_MODELO_PRINCIPAL = [
    "densidade_atual", "densidade_lag_12", "variacao_abs_12", "variacao_log_12", "densidade_lag_24", "variacao_abs_24",
    "variacao_log_24", "densidade_lag_36", "variacao_abs_36", "variacao_log_36", "media_12", "mediana_12", "n_obs_12",
    "volatilidade_12", "media_36", "mediana_36", "n_obs_36", "volatilidade_36", "tendencia_24", "tendencia_36",
    "n_meses_ausentes_36", "fracao_trimestral_36", "meses_desde_primeira_obs", "flag_causal_atual", "n_suspeita_12",
    "n_suspeita_36", "persistencia_36", "meses_desde_ultima_sinalizacao", "referencia_insuficiente_atual",
    "abaixo_p25_na_origem",
]
ALVO_PRINCIPAL = "alvo_p25_origem"


FEATURES_DESCRICAO = {
    "densidade_atual": ("nível", "Densidade na origem (acessos por 100 domicílios)"),
    "densidade_lag_12/24/36": ("nível", "Densidade 12, 24 e 36 meses antes (última observação em até 2 meses antes da data exata)"),
    "media_12/36, mediana_12/36": ("histórico", "Média e mediana das observações válidas nos últimos 12 e 36 meses"),
    "variacao_abs_12/24/36": ("variação", "Densidade atual menos a defasada"),
    "variacao_log_12/24/36": ("variação", "ln(1 + atual) − ln(1 + defasada)"),
    "tendencia_24/36": ("tendência", "Inclinação da reta de mínimos quadrados, em acessos/100 dom. por ano (mín. 6 obs.)"),
    "volatilidade_12/36": ("volatilidade", "Desvio-padrão das variações log entre observações consecutivas"),
    "n_obs_12/36": ("qualidade", "Observações válidas na janela"),
    "n_meses_ausentes_36": ("qualidade", "Meses mensais ausentes na janela de 36 meses"),
    "fracao_trimestral_36": ("qualidade", "Parte da janela de 36 meses no período trimestral (antes de 2011)"),
    "meses_desde_primeira_obs": ("qualidade", "Idade da série do município na origem"),
    "referencia_insuficiente_atual": ("qualidade", "Menos de 3 observações nos 12 meses anteriores à origem"),
    "flag_causal_atual": ("anomalia causal", "Flag da regra D causal na origem (0, ±1, ±2), K=5, A=10, L=12"),
    "n_suspeita_12/36": ("anomalia causal", "Meses com suspeita aberta (±1) nos últimos 12 e 36 meses"),
    "persistencia_36": ("anomalia causal", "Houve persistência reconhecida (±2) nos últimos 36 meses"),
    "meses_desde_ultima_sinalizacao": ("anomalia causal", "Meses desde a última flag diferente de zero (vazio se nunca)"),
    "*_L6": ("anomalia causal", "As mesmas flags com L=6 (sensibilidade)"),
    "abaixo_p25_na_origem": ("posição", "Município abaixo do P25 nacional na própria origem"),
    "censo_*": ("Censo 2022", "18 variáveis do IBGE; vazias quando não disponíveis na origem pelo critério"),
}


def dicionario_features() -> pd.DataFrame:
    return pd.DataFrame([dict(feature=k, grupo=g, descricao=d) for k, (g, d) in FEATURES_DESCRICAO.items()])


# --------------------------------------------------------------------------
# 6. Resumos para o relatório (descritivos; não envolvem modelos)
# --------------------------------------------------------------------------
def resumo_origens(paineis: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for o, p in paineis.groupby("origem", sort=True):
        e = p[p.elegivel_treino.astype(bool)]
        abaixo = e.abaixo_p25_na_origem == 1
        linhas.append(dict(
            origem=o, mes_alvo=p.mes_alvo.iloc[0], municipios=len(p), elegiveis_features=int(p.elegivel_features.sum()),
            alvo_observavel=bool(p.alvo_observavel.iloc[0]), elegiveis_treino=int(len(e)),
            janela_36_mensal_completa=round(float((p.n_obs_36 == 36).mean()), 3),
            fracao_trimestral_36=round(float(p.fracao_trimestral_36.iloc[0]), 3),
            limiar_p25_origem=round(float(p.limiar_p25_origem.iloc[0]), 2),
            prevalencia_alvo_p25=round(float(e.alvo_p25_origem.mean()), 3) if len(e) else np.nan,
            permanecem_abaixo=round(float(e.loc[abaixo, "alvo_p25_origem"].mean()), 3) if abaixo.any() else np.nan,
            positivos_ja_abaixo_na_origem=round(float(e.loc[e.alvo_p25_origem == 1, "abaixo_p25_na_origem"].mean()), 3)
            if (e.alvo_p25_origem == 1).any() else np.nan,
            variaveis_censo=int(p.n_variaveis_censo.iloc[0])))
    return pd.DataFrame(linhas)


def comparar_limiares(paineis: pd.DataFrame, limiares_absolutos=(5.0, 10.0, 15.0)) -> pd.DataFrame:
    """Prevalência do alvo por origem: P25 congelado na origem × limiares absolutos fixos."""
    linhas = []
    for o, p in paineis.groupby("origem", sort=True):
        e = p[p.elegivel_treino.astype(bool)]
        if not len(e):
            continue
        d = dict(origem=o, limiar_p25=round(float(e.limiar_p25_origem.iloc[0]), 2),
                 prev_p25_origem=round(float(e.alvo_p25_origem.mean()), 3))
        for v in limiares_absolutos:
            d[f"prev_abs_{v:g}"] = round(float((e.densidade_alvo < v).mean()), 3)
        linhas.append(d)
    return pd.DataFrame(linhas)
