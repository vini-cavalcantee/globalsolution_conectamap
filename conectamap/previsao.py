"""Etapa 2 — previsão de fevereiro de 2029 e avaliação de confiabilidade.

Usa o protocolo congelado (`etapa2/protocolo_treino_congelado.json`) SEM alterá-lo.
Não há nova busca de hiperparâmetros nem novo algoritmo. A coorte de teste (2023-02) não participa de
nenhuma decisão desta etapa; qualquer número calculado nela aqui é rotulado como PÓS-TESTE.

CRITÉRIO DE DIVULGAÇÃO DE PROBABILIDADES — fixado ANTES de qualquer cálculo de calibração:
as probabilidades só são divulgadas como estimativas quantitativas se uma versão do modelo (bruta ou com
Platt), em TODAS as 4 rodadas de validação (V1–V4), tiver:
  (a) |probabilidade média − frequência observada| <= 0,02 na coorte (calibração geral);
  (b) inclinação de calibração entre 0,80 e 1,25 (logística de y em logit(p));
  (c) |média prevista − frequência observada| <= 0,05 no decil de maior risco;
  (d) Brier não pior que o do baseline probabilístico B2p (logística só com a densidade atual).
Se nenhuma versão cumprir tudo, os escores ficam como valores técnicos e a divulgação é por categorias
relativas (maior / intermediário / menor).
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import avaliacao as AV
from . import painel as PN
from . import treino as TR

ORIGEM_OPERACIONAL = "2026-02"
CAMINHO_PROTOCOLO = PN.C.PACOTE / "etapa2" / "protocolo_treino_congelado.json"
CRITERIOS_DIVULGACAO = dict(max_abs_calibracao_geral=0.02, inclinacao_min=0.80, inclinacao_max=1.25,
                            max_abs_decil_superior=0.05, brier_nao_pior_que="B2p")
FAIXAS_CATEGORIA = (0.25, 0.50)    # maior: top 25%; intermediário: 25%–50%; menor: demais (relativas ao conjunto)
EPS = 1e-4


# --------------------------------------------------------------------------
# 1. Verificação do protocolo antes da previsão
# --------------------------------------------------------------------------
def verificar_protocolo(paineis_hist: pd.DataFrame, painel_2026: pd.DataFrame) -> pd.DataFrame:
    """Lista de verificações; todas precisam ser True para gerar a previsão."""
    prot = TR.ler_protocolo(CAMINHO_PROTOCOLO)
    modelo = TR.construir_modelo(prot["algoritmo"], prot["hiperparametros"])
    passos = [n for n, _ in modelo.steps]
    imp = modelo.named_steps.get("imputacao")
    final = PN.folds_backtesting().query("papel == 'previsao'").iloc[0]
    origens_final = final.origens_treino.split(",")
    elegiveis = paineis_hist[paineis_hist.origem.isin(origens_final)].groupby("origem").elegivel_treino.sum()
    feats = painel_2026[prot["features"]]
    t = PN.dados_padrao()["meses"]
    checks = [
        ("Protocolo íntegro (assinatura do conteúdo)", True, prot["sha256_protocolo"][:12]),
        ("Features = FEATURES_MODELO_PRINCIPAL (30, só Anatel)", prot["features"] == PN.FEATURES_MODELO_PRINCIPAL,
         len(prot["features"])),
        ("Algoritmo e hiperparâmetros congelados", prot["algoritmo"] == "random_forest"
         and prot["hiperparametros"] == {"max_features": "sqrt", "min_samples_leaf": 300}, json.dumps(prot["hiperparametros"])),
        ("Pré-processamento: imputação pela mediana do treino + indicadores de ausência",
         passos == ["imputacao", "modelo"] and imp.strategy == "median" and imp.add_indicator, " → ".join(passos)),
        ("Horizonte de 36 meses", (painel_2026.horizonte_meses == 36).all() and (painel_2026.mes_alvo == "2029-02").all(),
         painel_2026.mes_alvo.iloc[0]),
        ("Origem 2026-02 sem alvo observado", painel_2026.densidade_alvo.isna().all()
         and painel_2026.alvo_p25_origem.isna().all() and not painel_2026.alvo_observavel.any(), "alvo vazio"),
        ("Nenhuma informação posterior à origem (grade termina em 2026-02)", str(t[-1]) == ORIGEM_OPERACIONAL, str(t[-1])),
        ("Censo fora das features", not any(c.startswith("censo_") for c in prot["features"]), "0 variáveis"),
        ("10 coortes 2014–2023 elegíveis (alvo de treino ≤ 2026-02)", origens_final == PN.origens_fevereiro(2014, 2023)
         and (elegiveis > 5500).all(), f"{len(origens_final)} coortes, {int(elegiveis.sum()):,} pares".replace(",", ".")),
        ("Todos os municípios com features na origem 2026-02", bool(painel_2026.elegivel_features.all()),
         int(painel_2026.elegivel_features.sum())),
        ("Ausências em 2026-02 só na feature estruturalmente vazia (nunca sinalizado), tratada pela imputação "
         "+ indicador do protocolo", set(feats.columns[feats.isna().any()]) <= {"meses_desde_ultima_sinalizacao"},
         f"{feats.meses_desde_ultima_sinalizacao.isna().mean():.1%} sem sinalização anterior"),
    ]
    return pd.DataFrame(checks, columns=["verificacao", "ok", "detalhe"])


# --------------------------------------------------------------------------
# 2. Baseline probabilístico e calibração (só rodadas de validação)
# --------------------------------------------------------------------------
def ajustar_b2p(treino: pd.DataFrame, alvo=TR.ALVO) -> LogisticRegression:
    """B2p: regressão logística (L2, C=1, sem busca) com UMA feature, log(1 + densidade atual)."""
    m = LogisticRegression(C=1.0, max_iter=1000)
    m.fit(np.log1p(treino[["densidade_atual"]].values), treino[alvo].astype(int).values)
    return m


def prever_b2p(m, dados) -> np.ndarray:
    return m.predict_proba(np.log1p(dados[["densidade_atual"]].values))[:, 1]


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ajustar_platt(prob_cal, y_cal) -> LogisticRegression:
    """Platt (sigmoide) sobre logit(p), ajustado num conjunto de calibração SEPARADO da avaliação."""
    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(_logit(prob_cal).reshape(-1, 1), np.asarray(y_cal, int))
    return m


def aplicar_platt(m, prob) -> np.ndarray:
    return m.predict_proba(_logit(prob).reshape(-1, 1))[:, 1]


def metricas_calibracao(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    incl = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y.astype(int)).coef_[0, 0]
    corte = np.quantile(p, 0.9)
    top = p >= corte
    return dict(prevalencia=y.mean(), prob_media=p.mean(), calibracao_geral=p.mean() - y.mean(),
                inclinacao=float(incl), decil_sup_prob=p[top].mean(), decil_sup_obs=y[top].mean(),
                decil_sup_dif=p[top].mean() - y[top].mean(), brier=float(np.mean((p - y) ** 2)),
                pr_auc=AV.pr_auc(y, p))


def previsoes_calibracao(paineis, prot, fold, alvo=TR.ALVO) -> pd.DataFrame:
    """Para uma rodada: RF bruto (protocolo), RF + Platt e B2p, todos avaliados na origem da rodada.
    O Platt é ajustado na ÚLTIMA coorte de treino da rodada, prevista por um RF treinado nas coortes
    anteriores (fora da amostra); a origem avaliada nunca entra no ajuste do calibrador."""
    origens = fold.origens_treino.split(",")
    av, _ = TR.prever_rodada(prot["algoritmo"], prot["hiperparametros"], paineis, fold, prot["features"])
    cal_fold = pd.Series(dict(origens_treino=",".join(origens[:-1]), origem_avaliada=origens[-1]))
    cal, _ = TR.prever_rodada(prot["algoritmo"], prot["hiperparametros"], paineis, cal_fold, prot["features"])
    platt = ajustar_platt(cal.prob.values, cal[alvo].values)
    tr = paineis[paineis.origem.isin(origens) & paineis.elegivel_treino.astype(bool)]
    b2p = ajustar_b2p(tr)
    return av.assign(prob_platt=aplicar_platt(platt, av.prob.values), prob_b2p=prever_b2p(b2p, av),
                     coorte_calibrador=origens[-1])


VERSOES = {"rf_bruto": "prob", "rf_platt": "prob_platt", "B2p_densidade": "prob_b2p"}


def tabela_calibracao(prev: pd.DataFrame, rodada: str, alvo=TR.ALVO) -> pd.DataFrame:
    linhas = []
    for nome, col in VERSOES.items():
        for grupo, g in (("todos", prev), ("acima_p25_na_origem", prev[prev.abaixo_p25_na_origem == 0])):
            linhas.append(dict(rodada=rodada, versao=nome, grupo=grupo, **metricas_calibracao(g[alvo], g[col])))
    return pd.DataFrame(linhas)


def decidir_divulgacao(cal: pd.DataFrame) -> pd.DataFrame:
    """Aplica o critério pré-registrado (grupo 'todos', 4 rodadas de validação)."""
    t = cal[cal.grupo == "todos"]
    ref = t[t.versao == "B2p_densidade"].set_index("rodada").brier
    linhas = []
    for versao in ("rf_bruto", "rf_platt"):
        v = t[t.versao == versao].set_index("rodada")
        crit = pd.DataFrame({
            "calibracao_geral": v.calibracao_geral.abs() <= CRITERIOS_DIVULGACAO["max_abs_calibracao_geral"],
            "inclinacao": v.inclinacao.between(CRITERIOS_DIVULGACAO["inclinacao_min"], CRITERIOS_DIVULGACAO["inclinacao_max"]),
            "decil_superior": v.decil_sup_dif.abs() <= CRITERIOS_DIVULGACAO["max_abs_decil_superior"],
            "brier_vs_B2p": v.brier <= ref.reindex(v.index)})
        linhas.append(dict(versao=versao, **{k: f"{int(c.sum())}/{len(c)}" for k, c in crit.items()},
                           aprovada=bool(crit.all().all())))
    return pd.DataFrame(linhas)


def curva_calibracao(prev: pd.DataFrame, col: str, alvo=TR.ALVO, n=10) -> pd.DataFrame:
    p = prev.assign(faixa=pd.qcut(prev[col].rank(method="first"), n, labels=False))
    return p.groupby("faixa").agg(prob_media=(col, "mean"), freq_observada=(alvo, "mean")).reset_index()


# --------------------------------------------------------------------------
# 3. Treinamento operacional e previsão de 2029
# --------------------------------------------------------------------------
def treinar_operacional(paineis_hist: pd.DataFrame, features=None):
    """RF do protocolo treinado nas 10 coortes 2014–2023 (regra atual). Devolve o pipeline ajustado."""
    prot = TR.ler_protocolo(CAMINHO_PROTOCOLO)
    features = prot["features"] if features is None else features
    origens = PN.folds_backtesting().query("papel == 'previsao'").iloc[0].origens_treino.split(",")
    tr = paineis_hist[paineis_hist.origem.isin(origens) & paineis_hist.elegivel_treino.astype(bool)
                      & paineis_hist[TR.ALVO].notna()]
    m = TR.construir_modelo(prot["algoritmo"], prot["hiperparametros"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(tr[features], tr[TR.ALVO].astype(int))
    if "n_jobs" in m.named_steps["modelo"].get_params():
        m.named_steps["modelo"].set_params(n_jobs=1)          # predição determinística
    return m, tr


def _ranking(valor_risco: pd.Series, ids: pd.Series) -> np.ndarray:
    """Posição 1 = maior risco; empates por código IBGE crescente (determinístico)."""
    ordem = pd.DataFrame({"r": -valor_risco.values, "id": ids.values}).sort_values(["r", "id"]).index
    pos = np.empty(len(ordem), int)
    pos[ordem] = np.arange(1, len(ordem) + 1)
    return pos


def categoria(posicao: np.ndarray, n: int) -> np.ndarray:
    a, b = (math.ceil(f * n) for f in FAIXAS_CATEGORIA)
    return np.where(posicao <= a, "maior", np.where(posicao <= b, "intermediario", "menor"))


def prever_2029(modelo, painel_2026: pd.DataFrame, ibge: pd.DataFrame, features=None) -> pd.DataFrame:
    features = TR.ler_protocolo(CAMINHO_PROTOCOLO)["features"] if features is None else features
    p = painel_2026.reset_index(drop=True)
    escore = modelo.predict_proba(p[features])[:, 1]
    n = len(p)
    out = pd.DataFrame({
        "id_municipio": p.id_municipio, "municipio": p.id_municipio.map(ibge.municipio.str.replace(r" \([A-Z]{2}\)$", "", regex=True)),
        "sigla_uf": p.sigla_uf, "densidade_fev2026": p.densidade_atual, "limiar_p25_origem": p.limiar_p25_origem,
        "abaixo_limiar_em_fev2026": p.abaixo_p25_na_origem.astype(bool),
        "escore_rf": escore})
    out["posicao_rf"] = _ranking(out.escore_rf, out.id_municipio)
    out["posicao_b2"] = _ranking(-out.densidade_fev2026, out.id_municipio)
    out["diferenca_posicoes"] = out.posicao_b2 - out.posicao_rf          # > 0: RF põe o município mais à frente
    out["categoria_rf"] = categoria(out.posicao_rf.values, n)
    out["categoria_b2"] = categoria(out.posicao_b2.values, n)
    top = {"rf": out.categoria_rf == "maior", "b2": out.categoria_b2 == "maior"}
    out["concordancia_top25"] = np.select([top["rf"] & top["b2"], top["rf"], top["b2"]], ["ambos", "so_rf", "so_b2"], "nenhum")
    out["classificacao_prioridade"] = np.select(
        [top["rf"] & top["b2"], top["rf"] | top["b2"], (out.categoria_rf == "menor") & (out.categoria_b2 == "menor")],
        ["alta_consenso", "alta_divergente_revisar", "menor"], "intermediaria")
    # indicadores de qualidade (todos disponíveis na origem)
    out["meses_ausentes_36"] = p.n_meses_ausentes_36.values
    out["flag_causal_fev2026"] = p.flag_causal_atual.values
    out["suspeitas_36m"] = p.n_suspeita_36.values
    out["persistencia_36m"] = p.persistencia_36.values
    out["historico_meses"] = p.meses_desde_primeira_obs.values
    out["tendencia_36"] = p.tendencia_36.values
    out["alerta_qualidade"] = (out.flag_causal_fev2026.fillna(0).ne(0) | (out.meses_ausentes_36 > 0)
                               | (out.suspeitas_36m > 0) | (out.historico_meses < 36))
    return out.sort_values("posicao_rf").reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Comparação, qualidade, mudança de distribuição e estabilidade
# --------------------------------------------------------------------------
def comparar_rankings(prev: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr  # noqa: PLC0415  (dependência do scikit-learn)
    c = prev.concordancia_top25.value_counts()
    k = int((prev.categoria_rf == "maior").sum())
    return pd.DataFrame([
        ("Municípios no top 25% (RF)", k), ("Municípios no top 25% (B2)", int((prev.categoria_b2 == "maior").sum())),
        ("Selecionados por ambos", int(c.get("ambos", 0))), ("Só pelo Random Forest", int(c.get("so_rf", 0))),
        ("Só pelo B2", int(c.get("so_b2", 0))), ("Concordância do top 25%", f"{c.get('ambos', 0) / k:.1%}"),
        ("Correlação de Spearman entre as posições", round(spearmanr(prev.posicao_rf, prev.posicao_b2).correlation, 3)),
        ("Mediana de |diferença de posições| no top 25% de algum método",
         int(prev[prev.concordancia_top25 != "nenhum"].diferenca_posicoes.abs().median())),
    ], columns=["medida", "valor"])


def comparar_top25(a: pd.DataFrame, b: pd.DataFrame, col_a="posicao_rf", col_b="posicao_rf") -> float:
    k = math.ceil(0.25 * len(a))
    return len(set(a.loc[a[col_a] <= k, "id_municipio"]) & set(b.loc[b[col_b] <= k, "id_municipio"])) / k


FEATURES_DRIFT = ["densidade_atual", "tendencia_36", "volatilidade_12", "n_meses_ausentes_36", "n_suspeita_36",
                  "flag_causal_atual", "meses_desde_primeira_obs"]


def drift_2026(paineis_hist: pd.DataFrame, painel_2026: pd.DataFrame, features=FEATURES_DRIFT) -> pd.DataFrame:
    """Quantis por origem e parte dos municípios de 2026 fora da faixa central (p1–p99) do treino."""
    tr = paineis_hist[paineis_hist.elegivel_treino.astype(bool)]
    linhas = []
    for f in features:
        lo, hi = np.nanquantile(tr[f], [0.01, 0.99])
        h = tr.groupby("origem")[f].median()
        x = painel_2026[f]
        linhas.append(dict(feature=f, mediana_2014=h.iloc[0], mediana_2023=h.iloc[-1], mediana_2026=x.median(),
                           p90_treino=np.nanquantile(tr[f], 0.9), p90_2026=x.quantile(0.9),
                           max_treino=np.nanmax(tr[f]), acima_max_treino_2026=float((x > np.nanmax(tr[f])).mean()),
                           fora_p1_p99_treino_2026=float(((x < lo) | (x > hi)).mean())))
    return pd.DataFrame(linhas)


ESTRUTURAIS = ("meses_desde_primeira_obs", "meses_desde_ultima_sinalizacao")   # crescem com o calendário / vazias por construção


def features_fora_faixa(painel_2026, paineis_hist, features) -> pd.Series:
    """Nº de features (exceto as estruturais) fora da faixa p0,5–p99,5 das coortes de treino, por município."""
    f = [c for c in features if c not in ESTRUTURAIS]
    tr = paineis_hist[paineis_hist.elegivel_treino.astype(bool)]
    lo, hi = tr[f].quantile(0.005), tr[f].quantile(0.995)
    x = painel_2026[f]
    return pd.Series(((x < lo) | (x > hi)).sum(axis=1).values, index=painel_2026.id_municipio.values)


def qualidade_entradas(prev: pd.DataFrame, painel_2026: pd.DataFrame, paineis_hist: pd.DataFrame, features) -> pd.DataFrame:
    tr = paineis_hist[paineis_hist.elegivel_treino.astype(bool)]
    fora = features_fora_faixa(painel_2026, paineis_hist, features).values
    top = prev.set_index("id_municipio").categoria_rf.reindex(painel_2026.id_municipio).values == "maior"
    nao_estr = [c for c in features if c not in ESTRUTURAIS]

    def lin(nome, m):
        m = np.asarray(m, bool)
        return dict(indicador=nome, municipios=int(m.sum()), no_top25_rf=int((m & top).sum()))
    return pd.DataFrame([
        lin("Feature ausente (exceto a estrutural 'meses desde a última sinalização')", painel_2026[nao_estr].isna().any(axis=1)),
        lin("Sem sinalização anterior (feature estrutural vazia; imputada + indicador)",
            painel_2026.meses_desde_ultima_sinalizacao.isna()),
        lin("Mês ausente nos últimos 36 meses", painel_2026.n_meses_ausentes_36 > 0),
        lin("Sinalização causal aberta em fev/2026 (flag ≠ 0)", painel_2026.flag_causal_atual.fillna(0) != 0),
        lin("Suspeita estatística nos últimos 36 meses", painel_2026.n_suspeita_36 > 0),
        lin("Histórico menor que 36 meses", painel_2026.meses_desde_primeira_obs < 36),
        lin("Idade da série acima do máximo do treino (efeito do calendário)",
            painel_2026.meses_desde_primeira_obs > tr.meses_desde_primeira_obs.max()),
        lin("Densidade acima do p99,5 do treino (valor extremo)",
            painel_2026.densidade_atual > tr.densidade_atual.quantile(0.995)),
        lin("Alguma feature não estrutural fora da faixa p0,5–p99,5 do treino", fora > 0),
        lin("Três ou mais features fora dessa faixa (condição atípica)", fora >= 3),
    ])


# --------------------------------------------------------------------------
# 5. Execução completa (usada pelo Notebook 2) e gráficos
# --------------------------------------------------------------------------
CORES = {"rf_bruto": "#2a78d6", "B2p_densidade": "#eb6834", "rf_platt": "#1baf7a"}   # paleta categorical validada
ROTULOS = {"rf_bruto": "Random Forest (escore bruto)", "rf_platt": "Random Forest + Platt",
           "B2p_densidade": "B2p (logística da densidade)"}
DIR_SAIDA = PN.C.PACOTE / "etapa2" / "saidas" / "previsao2029"


def analise_calibracao(paineis_hist: pd.DataFrame):
    """V1–V4 (decisão) + descrição PÓS-TESTE em 2023 (sem efeito sobre a decisão)."""
    prot = TR.ler_protocolo(CAMINHO_PROTOCOLO)
    sem_teste = paineis_hist[paineis_hist.origem != TR.ORIGEM_TESTE]
    prev, tabs = [], []
    for f in TR.folds_validacao().itertuples():
        p = previsoes_calibracao(sem_teste, prot, f)
        prev.append(p.assign(rodada=f.rodada, uso="validacao"))
        tabs.append(tabela_calibracao(p, f.rodada).assign(uso="validacao"))
    cal = pd.concat(tabs, ignore_index=True)
    decisao = decidir_divulgacao(cal)          # decidida ANTES de olhar 2023
    ft = PN.folds_backtesting().query("papel == 'teste'").iloc[0]
    p = previsoes_calibracao(paineis_hist, prot, ft)
    prev.append(p.assign(rodada=ft.rodada, uso="pos_teste"))
    cal = pd.concat([cal, tabela_calibracao(p, ft.rodada).assign(uso="pos_teste")], ignore_index=True)
    curvas = pd.concat([curva_calibracao(g, col).assign(rodada=r, uso=g.uso.iloc[0], versao=v)
                        for r, g in pd.concat(prev).groupby("rodada") for v, col in VERSOES.items()], ignore_index=True)
    return cal, decisao, curvas


def escore_medio(cal: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Escore médio do RF × frequência observada por coorte; em 2026 o alvo é desconhecido."""
    t = cal[(cal.grupo == "todos") & (cal.versao == "rf_bruto")][["rodada", "uso", "prevalencia", "prob_media"]]
    return pd.concat([t, pd.DataFrame([dict(rodada="previsao_2026-02", uso="operacional", prevalencia=np.nan,
                                            prob_media=prev.escore_rf.mean())])], ignore_index=True)


def estabilidade(prev: pd.DataFrame, prev_l6: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr  # noqa: PLC0415
    a = prev.set_index("id_municipio")
    b = prev_l6.set_index("id_municipio").reindex(a.index)
    k = math.ceil(0.25 * len(a))
    return pd.DataFrame([
        dict(comparacao="RF principal (L=12) × RF sensibilidade (L=6)", top25_comum=comparar_top25(prev, prev_l6),
             entram_ou_saem=int(k - round(comparar_top25(prev, prev_l6) * k)),
             spearman=spearmanr(a.posicao_rf, b.posicao_rf).correlation,
             max_dif_escore=float((a.escore_rf - b.escore_rf).abs().max())),
        dict(comparacao="RF principal (L=12) × B2 (densidade atual)", top25_comum=comparar_top25(prev, prev, "posicao_rf", "posicao_b2"),
             entram_ou_saem=int(k - round(comparar_top25(prev, prev, "posicao_rf", "posicao_b2") * k)),
             spearman=spearmanr(a.posicao_rf, a.posicao_b2).correlation, max_dif_escore=np.nan)])


def executar(salvar=True, dados=None) -> dict:
    """Roda toda a etapa: verificação, calibração, treino operacional, previsão, comparação e confiabilidade."""
    dados = PN.dados_padrao() if dados is None else dados
    hist = PN.montar_paineis(PN.origens_fevereiro(2014, 2023), dados=dados)
    p26 = PN.montar_painel(ORIGEM_OPERACIONAL, dados=dados)
    r = {"verificacao": verificar_protocolo(hist, p26)}
    if not r["verificacao"].ok.all():
        raise RuntimeError("verificação do protocolo falhou; previsão interrompida:\n"
                           + r["verificacao"][~r["verificacao"].ok].to_string())
    r["calibracao"], r["decisao"], r["curvas"] = analise_calibracao(hist)
    r["divulgar_probabilidades"] = bool(r["decisao"].aprovada.any())
    modelo, tr = treinar_operacional(hist)
    ibge = dados["ibge"]
    prev = previsao_final(modelo, p26, hist, ibge)
    feats = TR.ler_protocolo(CAMINHO_PROTOCOLO)["features"]
    if r["divulgar_probabilidades"]:      # só entra se o critério pré-registrado aprovar
        raise NotImplementedError("critério aprovado: incluir a probabilidade calibrada antes de divulgar")
    m6, _ = treinar_operacional(hist, TR.features_L6())
    prev_l6 = prever_2029(m6, p26, ibge, TR.features_L6())
    r.update(previsao=prev, comparacao=comparar_rankings(prev), estabilidade=estabilidade(prev, prev_l6),
             qualidade=qualidade_entradas(prev, p26, hist, feats), drift=drift_2026(hist, p26),
             divergencias=prev[prev.concordancia_top25.isin(["so_rf", "so_b2"])].sort_values(["concordancia_top25", "posicao_rf"]),
             n_treino=len(tr), prevalencia_treino=float(tr[TR.ALVO].mean()), hist=hist, p26=p26,
             escore_medio=escore_medio(r["calibracao"], prev))
    if salvar:
        salvar_saidas(r)
    return r


def previsao_final(modelo, painel_2026, paineis_hist, ibge) -> pd.DataFrame:
    """Base final de 2029: previsão + nº de features fora da faixa do treino, que entra no alerta de qualidade."""
    feats = TR.ler_protocolo(CAMINHO_PROTOCOLO)["features"]
    prev = prever_2029(modelo, painel_2026, ibge, feats)
    prev["n_features_fora_faixa_treino"] = prev.id_municipio.map(features_fora_faixa(painel_2026, paineis_hist, feats))
    prev["alerta_qualidade"] = prev.alerta_qualidade | (prev.n_features_fora_faixa_treino >= 3)
    return prev


def texto_csv(df: pd.DataFrame) -> str:
    """Formato único de gravação (UTF-8, vírgula, ponto decimal, 6 algarismos significativos, quebra \\n)."""
    return df.to_csv(index=False, float_format="%.6g", lineterminator="\n")


def tabelas_saida(r: dict) -> dict:
    """{nome do arquivo: texto CSV} de todas as tabelas da previsão."""
    div = ["id_municipio", "municipio", "sigla_uf", "densidade_fev2026", "tendencia_36", "escore_rf",
           "posicao_rf", "posicao_b2", "diferenca_posicoes", "concordancia_top25", "alerta_qualidade"]
    return {
        "previsoes_2029_municipios.csv": texto_csv(r["previsao"]),
        "comparacao_rf_b2_2029.csv": texto_csv(pd.concat([
            r["comparacao"].assign(secao="comparacao_top25"),
            r["estabilidade"].melt(id_vars="comparacao", var_name="medida", value_name="valor")
            .assign(medida=lambda d: d.comparacao + " | " + d.medida).drop(columns="comparacao")
            .assign(secao="estabilidade")])),
        "divergencias_rf_b2_2029.csv": texto_csv(r["divergencias"][div]),
        "verificacao_protocolo.csv": texto_csv(r["verificacao"]),
        "calibracao_por_coorte.csv": texto_csv(r["calibracao"]),
        "decisao_divulgacao_probabilidades.csv": texto_csv(r["decisao"]),
        "confiabilidade_entradas_2026.csv": texto_csv(pd.concat([
            r["qualidade"].assign(secao="qualidade_entradas"), r["drift"].assign(secao="mudanca_distribuicao"),
            r["escore_medio"].assign(secao="escore_medio_rf")])),
    }


def _gravar(nome: str, texto: str):
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    with open(DIR_SAIDA / nome, "w", encoding="utf-8", newline="") as f:
        f.write(texto)


def salvar_saidas(r: dict):
    for nome, texto in tabelas_saida(r).items():
        _gravar(nome, texto)
    graficos(r, salvar=True)


def conferir_ou_salvar(r: dict) -> pd.DataFrame:
    """Confere byte a byte as tabelas recalculadas com as saídas aprovadas, sem regravá-las.
    Só grava um arquivo se ele não existir (por exemplo, numa cópia do projeto sem as saídas)."""
    linhas = []
    for nome, texto in tabelas_saida(r).items():
        alvo = DIR_SAIDA / nome
        if alvo.exists():
            igual = hashlib.sha256(alvo.read_bytes()).hexdigest() == hashlib.sha256(texto.encode("utf-8")).hexdigest()
            linhas.append(dict(arquivo=nome, situacao="idêntico ao aprovado" if igual else "DIFERENTE do aprovado"))
        else:
            _gravar(nome, texto)
            linhas.append(dict(arquivo=nome, situacao="gerado (não existia)"))
    return pd.DataFrame(linhas)


def _estilo(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def grafico_calibracao(cur: pd.DataFrame):
    """Curvas de calibração por coorte (V1–V4 e pós-teste)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    rods = sorted(cur.rodada.unique(), key=lambda x: (x.startswith("teste"), x))      # validação primeiro
    fig, axs = plt.subplots(1, len(rods), figsize=(3.2 * len(rods), 3.4), sharex=True, sharey=True)
    for ax, rod in zip(axs, rods):
        for v in VERSOES:
            g = cur[(cur.rodada == rod) & (cur.versao == v)]
            ax.plot(g.prob_media, g.freq_observada, "o-", ms=4, lw=1.6, color=CORES[v], label=ROTULOS[v])
        ax.plot([0, 0.8], [0, 0.8], ":", color="#8a8984", lw=1)
        pos = cur[cur.rodada == rod].uso.iloc[0] == "pos_teste"
        if pos:
            ax.set_facecolor("#f1f0ec")
        ax.set_title(rod.split("_")[-1] + (" (pós-teste)" if pos else ""), fontsize=9)
        ax.set_xlabel("probabilidade média prevista", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("frequência observada", fontsize=8)
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Calibração por coorte (decis) — a decisão usa só 2019–2022", fontsize=10)
    fig.tight_layout()
    return fig


def grafico_rankings(p: pd.DataFrame):
    """Dispersão das posições RF × B2, destacando o top 25% e as divergências."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    k = math.ceil(0.25 * len(p))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    cor = p.concordancia_top25.map({"ambos": "#2a78d6", "so_rf": "#1baf7a", "so_b2": "#eb6834", "nenhum": "#cfcdc6"})
    nomes = {"ambos": "top 25% nos dois", "so_rf": "só no RF", "so_b2": "só no B2", "nenhum": "fora do top 25%"}
    for c in ("nenhum", "ambos", "so_rf", "so_b2"):
        g = p[p.concordancia_top25 == c]
        ax.scatter(g.posicao_b2, g.posicao_rf, s=4 if c in ("nenhum", "ambos") else 12, color=cor[g.index],
                   label=f"{nomes[c]} ({len(g)})", linewidths=0)
    for v in (k,):
        ax.axhline(v, color="#52514e", lw=0.8, ls="--")
        ax.axvline(v, color="#52514e", lw=0.8, ls="--")
    ax.set_xlabel("posição no ranking B2 (1 = menor densidade em fev/2026)")
    ax.set_ylabel("posição no ranking do Random Forest (1 = maior escore)")
    ax.set_title("Rankings de 2029: Random Forest × B2", fontsize=10)
    ax.legend(fontsize=8, frameon=False, markerscale=2, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def grafico_drift(h: pd.DataFrame, p26: pd.DataFrame):
    """Densidade e tendência por origem de treino × origem da previsão."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, f, tit in zip(axs, ("densidade_atual", "tendencia_36"),
                          ("Densidade na origem (acessos/100 dom.)", "Tendência de 36 meses (acessos/100 dom. por ano)")):
        grupos = [g[f].dropna().values for _, g in h[h.elegivel_treino.astype(bool)].groupby("origem")] + [p26[f].dropna().values]
        rot = [o[:4] for o in sorted(h.origem.unique())] + ["2026"]
        bp = ax.boxplot(grupos, tick_labels=rot, showfliers=False, patch_artist=True, widths=0.6,
                        medianprops=dict(color="#0b0b0b", lw=1.2))
        for i, b in enumerate(bp["boxes"]):
            b.set(facecolor="#eb6834" if i == len(grupos) - 1 else "#2a78d6", alpha=0.85, lw=0)
        ax.set_title(tit, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].plot(range(1, len(rot) + 1), [h[h.origem == o].limiar_p25_origem.iloc[0] for o in sorted(h.origem.unique())]
                + [p26.limiar_p25_origem.iloc[0]], "D", color="#0b0b0b", ms=3, label="P25 da origem")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Origens de treino (2014–2023, azul) × origem da previsão (2026, laranja); caixas sem valores extremos",
                 fontsize=9)
    fig.tight_layout()
    return fig


def graficos(r: dict, salvar=False) -> dict:
    figs = {"calibracao_por_coorte_2029.png": grafico_calibracao(r["curvas"]),
            "ranking_rf_vs_b2_2029.png": grafico_rankings(r["previsao"]),
            "mudanca_distribuicao_2026.png": grafico_drift(r["hist"], r["p26"])}
    if salvar:
        DIR_SAIDA.mkdir(parents=True, exist_ok=True)
        for nome, f in figs.items():
            f.savefig(DIR_SAIDA / nome, dpi=130, metadata={"Software": None})
    return figs
