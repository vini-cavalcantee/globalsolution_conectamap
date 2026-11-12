"""Funções auxiliares do notebook de entrega (ConectaMap_Pre_Processamento.ipynb).

Este módulo NÃO cria regras novas: organiza, para apresentação, o que já está implementado em
`fontes`, `anomalias`, `series` e `calibracao` (Etapas 1 e 1.5) e monta as bases finais exportadas.
"""
import hashlib

import numpy as np
import pandas as pd

from . import anomalias as AN
from . import calibracao as CB
from . import config as C
from . import fontes as F
from . import series as S

TABELAS_IBGE = ["ibge_9923", "ibge_9515", "ibge_6805", "ibge_10295", "ibge_10125"]
NOMES_TABELAS = {
    "anatel": "Anatel — densidade de banda larga fixa",
    "ibge_9923": "IBGE 9923 — população urbana e rural",
    "ibge_9515": "IBGE 9515 — envelhecimento, idade mediana, razão de sexo",
    "ibge_6805": "IBGE 6805 — esgotamento sanitário dos domicílios",
    "ibge_10295": "IBGE 10295 — rendimento domiciliar per capita",
    "ibge_10125": "IBGE 10125 — pessoas com deficiência",
}


# --------------------------------------------------------------------------
# 2. Carregamento
# --------------------------------------------------------------------------
def ler_anatel_como_esta() -> pd.DataFrame:
    """Leitura do CSV da Anatel sem nenhum tratamento (tipos inferidos pelo pandas)."""
    return pd.read_csv(C.DIR_BRUTOS / C.ARQUIVOS["anatel"])


def ler_sidra_como_esta(chave: str) -> pd.DataFrame:
    """Linhas de dados de uma tabela do SIDRA como texto, antes da conversão numérica."""
    lay = F.LAYOUT_SIDRA[chave]
    bruto = pd.read_csv(C.DIR_BRUTOS / C.ARQUIVOS[chave], sep=";", encoding="utf-8-sig", dtype=str, header=None,
                        skiprows=lay["skiprows"], names=lay["colunas"], engine="python")
    return bruto[bruto["id_municipio"].astype(str).str.fullmatch(r"\d{7}", na=False)].reset_index(drop=True)


def hashes_brutos() -> pd.Series:
    return pd.Series(F.hashes_brutos(), name="sha256")


def resumo_fontes(anatel_bruta: pd.DataFrame, sidra_brutas: dict) -> pd.DataFrame:
    linhas = [dict(fonte=NOMES_TABELAS["anatel"], arquivo=C.ARQUIVOS["anatel"], registros=len(anatel_bruta),
                   colunas=anatel_bruta.shape[1], municipios=anatel_bruta.id_municipio.nunique(),
                   periodo=f"{anatel_bruta.ano.min()}-{anatel_bruta[anatel_bruta.ano == anatel_bruta.ano.min()].mes.min():02d} a "
                           f"{anatel_bruta.ano.max()}-{anatel_bruta[anatel_bruta.ano == anatel_bruta.ano.max()].mes.max():02d}")]
    for k, t in sidra_brutas.items():
        linhas.append(dict(fonte=NOMES_TABELAS[k], arquivo=C.ARQUIVOS[k], registros=len(t), colunas=t.shape[1],
                           municipios=t.id_municipio.nunique(), periodo="Censo 2022 (referência 31/07/2022)"))
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# 3. Exploração inicial
# --------------------------------------------------------------------------
def perfil_colunas(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"tipo": df.dtypes.astype(str), "preenchidos": df.notna().sum(), "ausentes": df.isna().sum(),
                         "exemplo": df.iloc[0]}).rename_axis("coluna").reset_index()


def simbolos_sidra(simbolos: dict) -> pd.DataFrame:
    """Contagem dos símbolos especiais do SIDRA ('-' = zero absoluto; 'X', '..', '...' = sem valor)."""
    linhas = []
    for k, cols in simbolos.items():
        for col, cont in cols.items():
            for s, n in cont.items():
                if n:
                    linhas.append(dict(tabela=k, coluna=col, simbolo=s, ocorrencias=n,
                                       tratamento="zero" if s in F.SIMBOLOS_ZERO else "ausente (NaN)"))
    return pd.DataFrame(linhas, columns=["tabela", "coluna", "simbolo", "ocorrencias", "tratamento"])


def verificar_identificadores(anatel: pd.DataFrame, tabelas: dict) -> pd.DataFrame:
    ids_a = set(anatel.id_municipio)
    prefixo_uf = anatel.drop_duplicates("id_municipio").assign(p=lambda d: d.id_municipio.str[:2]).groupby("p").sigla_uf.nunique()
    linhas = [
        ("Anatel: códigos com 7 dígitos", bool(anatel.id_municipio.str.fullmatch(r"\d{7}").all())),
        ("Anatel: municípios distintos", len(ids_a)),
        ("Anatel: cada município com uma única UF", bool((anatel.groupby("id_municipio").sigla_uf.nunique() == 1).all())),
        ("Anatel: prefixo do código (2 dígitos) corresponde a uma única UF", bool((prefixo_uf == 1).all())),
        ("Anatel: registros duplicados (município, ano, mês)", int(anatel.duplicated(["id_municipio", "ano", "mes"]).sum())),
    ]
    for k, t in tabelas.items():
        linhas.append((f"{k}: mesmo conjunto de municípios da Anatel", set(t.id_municipio) == ids_a and t.id_municipio.is_unique))
    return pd.DataFrame(linhas, columns=["verificacao", "resultado"])


def competencias_por_ano(anatel: pd.DataFrame) -> pd.DataFrame:
    g = anatel.groupby("ano")
    return pd.DataFrame({"competencias": g.mes.nunique(), "meses": g.mes.apply(lambda m: ",".join(map(str, sorted(m.unique())))),
                         "registros": g.size(), "municipios": g.id_municipio.nunique()}).reset_index()


def municipios_por_competencia(anatel: pd.DataFrame) -> pd.Series:
    comp = pd.PeriodIndex(anatel.ano.astype(str) + "-" + anatel.mes.astype(str).str.zfill(2), freq="M")
    return anatel.groupby(comp).id_municipio.nunique()


# --------------------------------------------------------------------------
# 4. Limpeza e padronização
# --------------------------------------------------------------------------
def periodicidade(ano: pd.Series) -> pd.Series:
    """Até 2010 a Anatel publicava competências trimestrais; a partir de 2011, mensais."""
    inicio = int(C.PARAMETROS_APROVADOS["inicio_periodo_mensal"][:4])
    return pd.Series(np.where(ano < inicio, "trimestral", "mensal"), index=ano.index)


def montar_base_ibge(tabelas: dict, anatel: pd.DataFrame) -> pd.DataFrame:
    """Une as 5 tabelas do Censo por município e calcula as proporções usadas no projeto
    (mesma construção do notebook técnico da Etapa 1.5)."""
    pop, demo, esgoto, renda, pcd = (tabelas[k] for k in TABELAS_IBGE)
    ibge = (pop[["id_municipio", "municipio", "pop_total", "pop_urbana", "pop_rural"]]
            .merge(demo.drop(columns="municipio"), on="id_municipio", validate="one_to_one")
            .merge(esgoto.drop(columns="municipio"), on="id_municipio", validate="one_to_one")
            .merge(renda.drop(columns="municipio"), on="id_municipio", validate="one_to_one")
            .merge(pcd.drop(columns=["municipio", "grupo_idade"]), on="id_municipio", validate="one_to_one"))
    ibge["pct_rural"] = ibge.pop_rural / ibge.pop_total
    ibge["pct_esgoto_rede_ou_fossa_ligada"] = ibge.dom_rede_ou_fossa_ligada / ibge.dom_total
    ibge["pct_sem_banheiro"] = ibge.dom_sem_banheiro / ibge.dom_total
    ibge["pct_deficiencia"] = ibge.pop_com_deficiencia / ibge.pop_2anos_mais
    ibge["sigla_uf"] = ibge.id_municipio.map(anatel.drop_duplicates("id_municipio").set_index("id_municipio").sigla_uf)
    return ibge


def meses_ausentes(anatel: pd.DataFrame) -> pd.DataFrame:
    """Competências mensais sem registro (entre a 1ª observação do município, ou 2011-01, e 2026-02).
    Ficam apenas identificadas: nenhum valor é preenchido."""
    aus = CB.meses_ausentes(anatel, C.PARAMETROS_APROVADOS["inicio_periodo_mensal"], C.PARAMETROS_APROVADOS["limite_observacao"])
    aus["competencia"] = [f"{t // 12}-{t % 12 + 1:02d}" for t in aus.t]
    uf = anatel.drop_duplicates("id_municipio").set_index("id_municipio").sigla_uf
    return aus.assign(sigla_uf=aus.id_municipio.map(uf))[["id_municipio", "sigla_uf", "competencia"]]


# --------------------------------------------------------------------------
# 5. Qualidade e anomalias
# --------------------------------------------------------------------------
def flags_em_tabela(Fm, Bm, ids, meses, nome_flag="flag_causal", nome_ref="referencia_causal") -> pd.DataFrame:
    """Converte as matrizes da sinalização causal (município × mês) em tabela longa, só no período
    em que a regra é calculada (2011-01 até o limite de observação)."""
    i0 = AN.indice_mes(meses, C.PARAMETROS_APROVADOS["inicio_periodo_mensal"])
    comp = [str(m) for m in meses[i0:Fm.shape[1]]]
    t = pd.DataFrame({"id_municipio": np.repeat(ids, len(comp)), "competencia": np.tile(comp, len(ids)),
                      nome_flag: Fm[:, i0:].ravel()})
    if Bm is not None:
        t[nome_ref] = Bm[:, i0:].ravel()
    return t


def obs_validas_12m(M, ids, meses) -> pd.DataFrame:
    """Quantidade de observações válidas nos 12 meses anteriores (não inclui o próprio mês).
    Indica referência histórica insuficiente (< 3), como na regra D. Só usa o passado."""
    i0 = AN.indice_mes(meses, C.PARAMETROS_APROVADOS["inicio_periodo_mensal"])
    iend = AN.indice_mes(meses, C.PARAMETROS_APROVADOS["limite_observacao"])
    val = (~np.isnan(M[:, : iend + 1])).astype(int)
    acum = np.concatenate([np.zeros((val.shape[0], 1), int), np.cumsum(val, axis=1)], axis=1)
    j = np.arange(i0, iend + 1)
    cont = acum[:, j] - acum[:, np.maximum(j - 12, 0)]
    comp = [str(meses[k]) for k in j]
    return pd.DataFrame({"id_municipio": np.repeat(ids, len(comp)), "competencia": np.tile(comp, len(ids)),
                         "obs_validas_12m_anteriores": cont.ravel()})


ESTADOS = {0: "normal", 1: "suspeita_alta_aberta", -1: "suspeita_queda_aberta",
           2: "persistencia_alta_reconhecida", -2: "persistencia_queda_reconhecida"}


def montar_indicadores(series: pd.DataFrame, flags12: pd.DataFrame, flags6: pd.DataFrame, obs12: pd.DataFrame,
                       reg_log: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por registro da base Anatel com os indicadores de qualidade (causais e retrospectivos)."""
    ind = series[["id_municipio", "competencia", "dentro_limite_observacao"]].copy()
    ind = (ind.merge(flags12, on=["id_municipio", "competencia"], how="left")
              .merge(flags6, on=["id_municipio", "competencia"], how="left")
              .merge(obs12, on=["id_municipio", "competencia"], how="left"))
    for c in ("flag_causal", "flag_causal_L6", "obs_validas_12m_anteriores"):
        ind[c] = ind[c].astype("Int8" if c.startswith("flag") else "Int16")
    ind.insert(ind.columns.get_loc("flag_causal") + 1, "estado_causal", ind.flag_causal.map(ESTADOS))
    ind["referencia_insuficiente"] = (ind.obs_validas_12m_anteriores < C.PARAMETROS_CANDIDATOS["min_obs_ref"]).astype("boolean")
    lg = reg_log[["id_municipio", "competencia", "episodio_id", "classificacao_provisoria", "evento_coletivo"]].rename(
        columns={"episodio_id": "retro_episodio_id", "classificacao_provisoria": "retro_classificacao",
                 "evento_coletivo": "retro_evento_coletivo"})
    ind = ind.merge(lg, on=["id_municipio", "competencia"], how="left")
    ind["retro_suspeita"] = ind.retro_episodio_id.notna()
    ind["retro_evento_coletivo"] = ind.retro_evento_coletivo.fillna(False).astype(bool)
    return ind


def resumo_sinalizacoes(ind: pd.DataFrame, ep: pd.DataFrame, ep6: pd.DataFrame) -> pd.DataFrame:
    d = ind[ind.dentro_limite_observacao]
    def lin(nome, mask, tipo):
        return dict(indicador=nome, tipo=tipo, registros=int(mask.sum()), municipios=int(d.loc[mask, "id_municipio"].nunique()))
    linhas = [
        lin("Suspeita estatística aberta (flag_causal = ±1)", d.flag_causal.abs().eq(1).fillna(False), "causal"),
        lin("Persistência reconhecida após L meses (flag_causal = ±2)", d.flag_causal.abs().eq(2).fillna(False), "causal"),
        lin("Referência histórica insuficiente (< 3 obs. em 12 meses)", d.referencia_insuficiente.fillna(False), "causal"),
        lin("Em episódio da análise retrospectiva", d.retro_suspeita, "retrospectivo"),
        lin("  dos quais em mês de evento coletivo", d.retro_evento_coletivo, "retrospectivo"),
        lin("  dos quais suspeita individual (fora de evento coletivo)", d.retro_suspeita & ~d.retro_evento_coletivo, "retrospectivo"),
        lin("Sensibilidade L = 6: suspeita aberta (flag_causal_L6 = ±1)", d.flag_causal_L6.abs().eq(1).fillna(False), "causal"),
    ]
    t = pd.DataFrame(linhas)
    t.loc[len(t)] = dict(indicador="Episódios retrospectivos (K=5, A=10, L=12) | (L=6)", tipo="retrospectivo",
                         registros=f"{len(ep)} | {len(ep6)}", municipios=f"{ep.id_municipio.nunique()} | {ep6.id_municipio.nunique()}")
    return t


# --------------------------------------------------------------------------
# 6–7. Resultado, exportação e conferência
# --------------------------------------------------------------------------
DICIONARIO = [
    ("anatel_densidade_tratada.csv", "Base temporal da Anatel: um registro por município e competência do arquivo original", [
        ("id_municipio", "Código IBGE do município (7 dígitos, texto)"),
        ("sigla_uf", "Sigla da UF"), ("ano", "Ano da competência"), ("mes", "Mês da competência"),
        ("competencia", "Competência no formato AAAA-MM"),
        ("periodicidade", "trimestral (2007–2010) ou mensal (a partir de 2011)"),
        ("dentro_limite_observacao", "True até 2026-02; False em 2026-03 (só auditoria)"),
        ("densidade_original", "Acessos de banda larga fixa por 100 domicílios, como publicado"),
        ("densidade_retro", "Versão retrospectiva para auditoria; igual à original (nenhuma correção aprovada)"),
        ("status_retro", "Situação da série retrospectiva"),
        ("densidade_causal", "Versão para features; igual à original (nenhuma correção aprovada)"),
        ("status_causal", "Situação da série causal")]),
    ("anatel_indicadores_qualidade.csv", "Indicadores de qualidade por registro (mesmas linhas da base temporal)", [
        ("id_municipio", "Código IBGE do município"), ("competencia", "AAAA-MM"),
        ("dentro_limite_observacao", "True até 2026-02"),
        ("flag_causal", "Regra D causal (K=5, A=10, L=12), só com dados até o mês: 0 normal, ±1 suspeita aberta, ±2 persistência reconhecida; vazio fora de 2011-01..2026-02"),
        ("estado_causal", "Descrição de flag_causal"),
        ("referencia_causal", "Referência histórica usada nos meses com flag ±1/±2"),
        ("flag_causal_L6", "Mesma regra com L=6 (sensibilidade)"),
        ("obs_validas_12m_anteriores", "Observações válidas nos 12 meses anteriores"),
        ("referencia_insuficiente", "Menos de 3 observações nos 12 meses anteriores"),
        ("retro_episodio_id", "Episódio da análise retrospectiva (usa meses posteriores; NÃO usar como feature)"),
        ("retro_classificacao", "Classe estatística do episódio retrospectivo (não é erro comprovado)"),
        ("retro_evento_coletivo", "Episódio iniciado em mês de evento coletivo (muitos municípios ao mesmo tempo)"),
        ("retro_suspeita", "Registro dentro de episódio retrospectivo")]),
    ("anatel_meses_ausentes.csv", "Competências mensais sem registro no arquivo original (não preenchidas)", [
        ("id_municipio", "Código IBGE"), ("sigla_uf", "Sigla da UF"), ("competencia", "AAAA-MM ausente")]),
    ("ibge_municipal_2022_tratada.csv", "Base municipal do Censo 2022 (5 tabelas do SIDRA unidas)", [
        ("id_municipio", "Código IBGE"), ("municipio", "Nome do município (UF)"), ("sigla_uf", "Sigla da UF"),
        ("pop_total / pop_urbana / pop_rural", "Pessoas (tabela 9923)"), ("pct_rural", "pop_rural / pop_total"),
        ("ind_envelhecimento / idade_mediana / razao_sexo", "Tabela 9515 (razões, não percentuais)"),
        ("dom_total / dom_rede_ou_fossa_ligada / dom_sem_banheiro", "Domicílios (tabela 6805)"),
        ("pct_esgoto_rede_ou_fossa_ligada / pct_sem_banheiro", "Proporções sobre dom_total"),
        ("moradores_renda / renda_media / renda_mediana", "Rendimento per capita, R$ nominais de 2022 (tabela 10295)"),
        ("pop_2anos_mais / pop_com_deficiencia", "Pessoas (tabela 10125)"), ("pct_deficiencia", "pop_com_deficiencia / pop_2anos_mais")]),
]


def dicionario() -> pd.DataFrame:
    return pd.DataFrame([dict(arquivo=a, finalidade=f, coluna=c, descricao=d) for a, f, cols in DICIONARIO for c, d in cols])


def exportar(bases: dict) -> pd.DataFrame:
    C.DIR_TRATADOS.mkdir(parents=True, exist_ok=True)
    linhas = []
    for nome, df in bases.items():
        caminho = C.DIR_TRATADOS / nome
        df.to_csv(caminho, index=False)
        linhas.append(dict(arquivo=f"dados_tratados/{nome}", linhas=len(df), colunas=df.shape[1],
                           tamanho_mb=round(caminho.stat().st_size / 1e6, 1),
                           sha256=hashlib.sha256(caminho.read_bytes()).hexdigest()[:16]))
    return pd.DataFrame(linhas)


def conferir_com_etapa_1_5(ep, reg_log, ep6, ausentes, hashes_inicio) -> pd.DataFrame:
    """Compara as saídas deste notebook com os arquivos produzidos pelo notebook técnico da Etapa 1.5."""
    aud, cal, meta = C.DIR_AUDITORIA, C.DIR_SAIDAS / "calibracao", C.DIR_METADADOS
    ep_ant = pd.read_csv(aud / "episodios_regra_D.csv", dtype={"id_municipio": str})
    log_ant = pd.read_csv(aud / "log_auditoria_registros.csv", dtype={"id_municipio": str})
    ep6_ant = pd.read_csv(cal / "episodios_config_recomendada_L6.csv", dtype={"id_municipio": str})
    lac_ant = pd.read_csv(aud / "lacunas_temporais.csv", dtype={"id_municipio": str})
    h_ant = pd.read_csv(meta / "hashes_dados_brutos.csv", index_col=0).iloc[:, 0]
    chave = ["chave_episodio", "classificacao", "evento_coletivo", "inicio", "fim"]
    col_meses = [c for c in lac_ant.columns if "meses" in c and "aus" in c]
    total_lac = int(lac_ant[col_meses[0]].sum()) if col_meses else None
    linhas = [
        ("Dados brutos com o mesmo SHA-256 registrado na Etapa 1.5", hashes_inicio.sort_index().equals(h_ant.sort_index())),
        ("Episódios (K=5, A=10, L=12): mesmos episódios e classes", ep[chave].astype(str).reset_index(drop=True)
         .equals(ep_ant[chave].astype(str).reset_index(drop=True))),
        ("Registros em episódios: mesmos registros e classes", reg_log[["id_municipio", "competencia", "episodio_id", "classificacao_provisoria"]]
         .astype(str).reset_index(drop=True).equals(log_ant[["id_municipio", "competencia", "episodio_id", "classificacao_provisoria"]]
                                                    .astype(str).reset_index(drop=True))),
        ("Episódios com L=6 (sensibilidade): mesmos episódios e classes", ep6[chave].astype(str).reset_index(drop=True)
         .equals(ep6_ant[chave].astype(str).reset_index(drop=True))),
        ("Meses ausentes: mesma contagem", (len(ausentes), total_lac) if total_lac != len(ausentes) else True),
    ]
    return pd.DataFrame(linhas, columns=["conferencia", "resultado"])


def conferir_limite_causal(M, meses, params, Fm, datas=("2016-06", "2021-01", "2025-06")) -> pd.DataFrame:
    """Recalcula a sinalização causal só com dados até cada data e compara com a sinalização completa:
    se forem iguais até a data, nenhum mês posterior influenciou as flags."""
    linhas = []
    for d in datas:
        i = AN.indice_mes(meses, d)
        Ft, _ = S.sinalizar_causal(M[:, : i + 1], meses[: i + 1], params, ate=d)
        linhas.append(dict(dados_ate=d, flags_identicas_ate_a_data=bool(np.array_equal(Ft, Fm[:, : i + 1]))))
    return pd.DataFrame(linhas)
