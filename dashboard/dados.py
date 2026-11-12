"""Leitura, validação, filtros e indicadores do dashboard (funções puras, sem Streamlit).

Regras: a base aprovada só é LIDA; nenhum escore, ranking ou categoria é recalculado. Os filtros
selecionam linhas e preservam as posições nacionais.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent                       # pasta do projeto
DIR_PREVISAO = RAIZ / "etapa2" / "saidas" / "previsao2029"
ARQ_BASE = DIR_PREVISAO / "previsoes_2029_municipios.csv"
ARQ_DICIONARIO = DIR_PREVISAO / "dicionario_previsoes_2029.csv"
ARQ_TESTE = RAIZ / "etapa2" / "saidas" / "treino" / "teste_2023_metricas.csv"
ARQ_DIF_TESTE = RAIZ / "etapa2" / "saidas" / "treino" / "comparacao_modelo_vs_B2_teste.csv"
DIR_ASSETS = Path(__file__).resolve().parent / "assets"
ARQ_GEO_MUN = DIR_ASSETS / "municipios_2022.geojson"
ARQ_GEO_UF = DIR_ASSETS / "ufs_2022.geojson"
ARQ_GEO_META = DIR_ASSETS / "geometrias_metadados.json"

# assinaturas dos arquivos aprovados na Etapa 2 (o dashboard confere antes de usar)
SHA256_APROVADO = {
    ARQ_BASE.name: "11fc88e8986944104450d8a3456312492c8fc32bb791cd0f6843b183259a47dc",
    ARQ_DICIONARIO.name: "95420052b0098bcb400d599af16fd3c43a71144a9a255689464f33a331503815",
}

COLUNAS = ["id_municipio", "municipio", "sigla_uf", "densidade_fev2026", "limiar_p25_origem", "abaixo_limiar_em_fev2026",
           "escore_rf", "posicao_rf", "posicao_b2", "diferenca_posicoes", "categoria_rf", "categoria_b2",
           "concordancia_top25", "classificacao_prioridade", "meses_ausentes_36", "flag_causal_fev2026",
           "suspeitas_36m", "persistencia_36m", "historico_meses", "tendencia_36", "alerta_qualidade",
           "n_features_fora_faixa_treino"]

# --- regiões: 1º dígito do código IBGE, conferido com a UF -------------------------------------------
REGIAO_POR_DIGITO = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}
REGIAO_POR_UF = {**dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "Norte"),
                 **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
                 **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
                 **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
                 **dict.fromkeys(["DF", "GO", "MS", "MT"], "Centro-Oeste")}
ORDEM_REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

# --- rótulos amigáveis -------------------------------------------------------------------------------
PRIORIDADES = ["alta_consenso", "alta_divergente_revisar", "intermediaria", "menor"]
# Rótulos de APRESENTAÇÃO. Os códigos originais (alta_consenso etc.) ficam só nos dados e na lógica.
ROTULO_PRIORIDADE = {"alta_consenso": "Alta prioridade — métodos concordam",
                     "alta_divergente_revisar": "Métodos divergem — analisar",
                     "intermediaria": "Prioridade intermediária", "menor": "Menor prioridade"}
ROTULO_CURTO = {"alta_consenso": "Alta prioridade", "alta_divergente_revisar": "Métodos divergem",
                "intermediaria": "Intermediária", "menor": "Menor prioridade"}
DESCRICAO_PRIORIDADE = {
    "alta_consenso": "Está entre os 25% que mais merecem atenção pelos dois critérios do projeto.",
    "alta_divergente_revisar": "Está entre os 25% por apenas um dos critérios; vale uma análise complementar.",
    "intermediaria": "Está entre 25% e 50% da ordenação em pelo menos um dos critérios.",
    "menor": "Fica fora da metade que mais merece atenção nos dois critérios.",
}
ROTULO_CATEGORIA = {"maior": "Alta (25% primeiros)", "intermediario": "Intermediária (25%–50%)", "menor": "Menor (demais)"}
ROTULO_CONCORDANCIA = {"ambos": "Alta nos dois critérios", "so_rf": "Só pelo modelo preditivo",
                       "so_b2": "Só pela densidade atual", "nenhum": "Fora dos 25% primeiros"}
NOME_METODO = {"rf": "Modelo preditivo", "b2": "Densidade atual"}
NOME_METODO_TECNICO = {"rf": "Random Forest", "b2": "B2"}
OPCOES_ALERTA = ["Todos", "Com alerta", "Sem alerta"]
TEXTO_ALERTA = ("Indica variações ou características incomuns nos dados, que podem merecer conferência. "
                "Não significa necessariamente que a informação esteja incorreta.")
TEXTO_CONCORDANCIA = ("Os dois critérios selecionam {pct} dos mesmos municípios entre os 25% primeiros do ranking. "
                      "Esse valor representa a concordância das listas, não a precisão da previsão.")
TEXTO_DIVERGENCIA = ("Dois critérios analisam os municípios. Quando eles não chegam à mesma classificação, a localidade "
                     "é sinalizada para análise complementar.")


# ======================================================================================================
# Leitura e validação
# ======================================================================================================
def sha256(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def carregar_base(caminho: Path = ARQ_BASE) -> pd.DataFrame:
    """Lê a base aprovada (id_municipio como texto) e acrescenta a região, validada pela UF."""
    base = pd.read_csv(caminho, dtype={"id_municipio": str}, encoding="utf-8")
    regiao_codigo = base.id_municipio.str[0].map(REGIAO_POR_DIGITO)
    regiao_uf = base.sigla_uf.map(REGIAO_POR_UF)
    if not regiao_codigo.equals(regiao_uf):
        raise ValueError("região derivada do código IBGE não confere com a UF")
    return base.assign(regiao=regiao_codigo)


def carregar_dicionario(caminho: Path = ARQ_DICIONARIO) -> pd.DataFrame:
    return pd.read_csv(caminho, encoding="utf-8")


def validar_base(base: pd.DataFrame, caminho: Path | None = ARQ_BASE) -> pd.DataFrame:
    """Verificações de integridade (todas devem ser True para o app exibir os dados)."""
    n = len(base)
    k25, k50 = -(-n * 25 // 100), -(-n * 50 // 100)
    v = [
        ("Arquivo idêntico ao aprovado (SHA-256)",
         caminho is None or sha256(caminho) == SHA256_APROVADO.get(Path(caminho).name)),
        ("5.570 municípios, sem duplicidade", n == 5570 and base.id_municipio.is_unique),
        ("Códigos IBGE com 7 dígitos", bool(base.id_municipio.str.fullmatch(r"\d{7}").all())),
        ("Colunas esperadas", [c for c in base.columns if c != "regiao"] == COLUNAS),
        ("Sem valores ausentes", int(base[COLUNAS].isna().sum().sum()) == 0),
        ("Rankings RF e B2 completos e sem posições repetidas",
         all(sorted(base[c]) == list(range(1, n + 1)) for c in ("posicao_rf", "posicao_b2"))),
        ("Categorias reconhecidas", set(base.classificacao_prioridade) <= set(PRIORIDADES)
         and set(base.categoria_rf) | set(base.categoria_b2) <= set(ROTULO_CATEGORIA)
         and set(base.concordancia_top25) <= set(ROTULO_CONCORDANCIA)),
        ("Top 25% com 1.393 municípios em cada método",
         all((base[c] == "maior").sum() == k25 and (base[c] != "menor").sum() == k50 for c in ("categoria_rf", "categoria_b2"))),
        ("Escore técnico entre 0 e 1 (não é probabilidade)", bool(base.escore_rf.between(0, 1).all())),
        ("Região derivada do código confere com a UF", "regiao" in base and base.regiao.notna().all()),
    ]
    return pd.DataFrame(v, columns=["verificacao", "ok"])


def carregar_geometrias(caminho: Path = ARQ_GEO_MUN) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def validar_geometrias(geo: dict, base: pd.DataFrame) -> pd.DataFrame:
    ids = [f["id"] for f in geo["features"]]
    uf_geo = {f["id"]: f["properties"]["uf"] for f in geo["features"]}
    uf_base = dict(zip(base.id_municipio, base.sigla_uf))
    v = [
        ("Uma geometria por município (5.570)", len(ids) == 5570 and len(set(ids)) == 5570),
        ("Códigos da malha = códigos da base", set(ids) == set(base.id_municipio)),
        ("UF da malha = UF da base", all(uf_geo[i] == uf_base[i] for i in uf_base)),
        ("Coordenadas dentro do território brasileiro",
         all(-74.5 <= x <= -28.5 and -34.5 <= y <= 5.5 for x, y in _pontos(geo))),
    ]
    return pd.DataFrame(v, columns=["verificacao", "ok"])


def _pontos(geo: dict):
    for f in geo["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for anel in poly:
                yield from anel


def ponto_no_municipio(geo: dict, id_municipio: str, lon: float, lat: float) -> bool:
    """Ponto-em-polígono (ray casting) para conferir a posição de municípios conhecidos."""
    f = next(f for f in geo["features"] if f["id"] == id_municipio)
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]

    def dentro(anel):
        d = False
        for (x1, y1), (x2, y2) in zip(anel, anel[1:] + anel[:1]):
            if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
                d = not d
        return d
    return any(dentro(p[0]) and not any(dentro(b) for b in p[1:]) for p in polys)


def subconjunto_geo(geo: dict, ids) -> dict:
    ids = set(ids)
    return {"type": "FeatureCollection", "features": [f for f in geo["features"] if f["id"] in ids]}


def resultados_teste(caminho_teste: Path = ARQ_TESTE, caminho_dif: Path = ARQ_DIF_TESTE) -> dict:
    """PR-AUC do teste reservado (2023) e diferença RF − B2, lidos das saídas aprovadas."""
    t = pd.read_csv(caminho_teste, dtype={"regra": str, "sensibilidade": str})
    t = t[(t.grupo == "todos") & (t.regra == "atual") & (t.sensibilidade == "principal")].set_index("metodo")
    d = pd.read_csv(caminho_dif)
    d = d[d.grupo == "todos"].iloc[0]
    return dict(pr_auc_rf=float(t.loc["random_forest", "pr_auc"]), pr_auc_b2=float(t.loc["B2_densidade_atual", "pr_auc"]),
                prevalencia=float(t.loc["random_forest", "prevalencia"]), dif=float(d.dif_pr_auc),
                ic_inf=float(d.ic95_inf), ic_sup=float(d.ic95_sup))


# ======================================================================================================
# Filtros (selecionam linhas; os rankings nacionais permanecem)
# ======================================================================================================
@dataclass
class Filtros:
    regioes: list = field(default_factory=list)
    ufs: list = field(default_factory=list)
    prioridades: list = field(default_factory=list)
    alerta: str = "Todos"

    @property
    def ativos(self) -> bool:
        return bool(self.regioes or self.ufs or self.prioridades or self.alerta != "Todos")

    def descricao(self) -> str:
        partes = []
        if self.regioes:
            partes.append("Região: " + ", ".join(self.regioes))
        if self.ufs:
            partes.append("UF: " + ", ".join(self.ufs))
        if self.prioridades:
            partes.append("Prioridade: " + ", ".join(ROTULO_PRIORIDADE[p] for p in self.prioridades))
        if self.alerta != "Todos":
            partes.append("Pontos de atenção nos dados: " + self.alerta.lower())
        return " · ".join(partes) if partes else "Brasil — sem filtros"


def aplicar_filtros(base: pd.DataFrame, f: Filtros, usar_prioridade: bool = True) -> pd.DataFrame:
    m = pd.Series(True, index=base.index)
    if f.regioes:
        m &= base.regiao.isin(f.regioes)
    if f.ufs:
        m &= base.sigla_uf.isin(f.ufs)
    if usar_prioridade and f.prioridades:
        m &= base.classificacao_prioridade.isin(f.prioridades)
    if f.alerta == "Com alerta":
        m &= base.alerta_qualidade
    elif f.alerta == "Sem alerta":
        m &= ~base.alerta_qualidade
    return base[m]


def ufs_das_regioes(regioes) -> list:
    return sorted(u for u, r in REGIAO_POR_UF.items() if not regioes or r in regioes)


def buscar(base: pd.DataFrame, texto: str) -> pd.DataFrame:
    """Busca por nome (sem diferenciar acentos e maiúsculas) ou por código IBGE."""
    t = (texto or "").strip()
    if not t:
        return base
    if t.isdigit():
        return base[base.id_municipio.str.startswith(t)]
    alvo = _sem_acentos(t)
    return base[base.municipio.map(_sem_acentos).str.contains(alvo, regex=False)]


def _sem_acentos(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower()) if unicodedata.category(c) != "Mn")


def ranking(base: pd.DataFrame, metodo: str = "rf", n: int | None = None) -> pd.DataFrame:
    """Ordena pela posição NACIONAL do método escolhido (nunca pelo escore arredondado)."""
    col = "posicao_rf" if metodo == "rf" else "posicao_b2"
    r = base.sort_values(col)
    return r if n is None else r.head(n)


# ======================================================================================================
# Indicadores
# ======================================================================================================
def indicadores(base: pd.DataFrame) -> dict:
    cont = base.classificacao_prioridade.value_counts()
    top_rf = base.categoria_rf == "maior"
    top_b2 = base.categoria_b2 == "maior"
    algum = int((top_rf | top_b2).sum())
    return dict(
        municipios=len(base),
        **{p: int(cont.get(p, 0)) for p in PRIORIDADES},
        abaixo_limiar=int(base.abaixo_limiar_em_fev2026.sum()),
        alerta=int(base.alerta_qualidade.sum()),
        top25_rf=int(top_rf.sum()), top25_b2=int(top_b2.sum()), top25_ambos=int((top_rf & top_b2).sum()),
        so_rf=int((top_rf & ~top_b2).sum()), so_b2=int((~top_rf & top_b2).sum()),
        concordancia_top25=(int((top_rf & top_b2).sum()) / int(top_rf.sum())) if top_rf.any() else float("nan"),
        em_algum_top25=algum,
    )


def resumo_territorial(base: pd.DataFrame, por: str = "regiao") -> pd.DataFrame:
    """Contagem por classe de prioridade e parte dos municípios do território em prioridade alta."""
    ordem = ORDEM_REGIOES if por == "regiao" else sorted(base[por].unique())
    t = pd.crosstab(base[por], base.classificacao_prioridade).reindex(columns=PRIORIDADES, fill_value=0)
    t = t.reindex([o for o in ordem if o in t.index])
    t["municipios"] = t[PRIORIDADES].sum(axis=1)
    t["prioritarios"] = t.alta_consenso + t.alta_divergente_revisar
    t["pct_consenso"] = t.alta_consenso / t.municipios
    t["pct_divergente"] = t.alta_divergente_revisar / t.municipios
    t["pct_prioritarios"] = t.prioritarios / t.municipios
    t["com_alerta"] = base.groupby(por).alerta_qualidade.sum().reindex(t.index)
    t["pct_alerta"] = t.com_alerta / t.municipios
    return t


# ======================================================================================================
# Comparação territorial — denominadores explícitos (correção da versão 3)
# ======================================================================================================
# Regra: os gráficos territoriais SEMPRE usam o território COMPLETO (todos os municípios da UF, da região
# ou do Brasil). Filtros de prioridade e de pontos de atenção não entram nesses denominadores; o filtro de
# região/UF só escolhe QUAIS territórios aparecem. Os números do recorte filtrado são mostrados à parte,
# com denominador = municípios do recorte.
def _linha_territorio(base: pd.DataFrame, mascara: pd.Series, nome: str, nivel: str) -> dict:
    t = base[mascara]
    c = t.classificacao_prioridade.value_counts()
    n = len(t)
    cons, div = int(c.get("alta_consenso", 0)), int(c.get("alta_divergente_revisar", 0))
    alerta = int(t.alerta_qualidade.sum())
    return dict(territorio=nome, nivel=nivel, municipios=n, alta_consenso=cons, alta_divergente_revisar=div,
                prioritarios=cons + div, pct_consenso=cons / n if n else float("nan"),
                pct_divergente=div / n if n else float("nan"), pct_prioritarios=(cons + div) / n if n else float("nan"),
                com_alerta=alerta, pct_alerta=alerta / n if n else float("nan"))


NOME_UF = {"AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará",
           "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão", "MG": "Minas Gerais",
           "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco",
           "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RO": "Rondônia",
           "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo",
           "TO": "Tocantins"}


def comparacao_territorial(base: pd.DataFrame, f: "Filtros") -> pd.DataFrame:
    """Territórios completos a comparar, conforme o filtro territorial:
    - UF(s) selecionada(s): cada UF + a(s) região(ões) a que pertence(m) + Brasil;
    - região(ões) sem UF: cada região selecionada + Brasil;
    - sem filtro territorial: as 5 regiões + Brasil."""
    linhas = []
    if f.ufs:
        for uf in sorted(f.ufs, key=lambda u: (ORDEM_REGIOES.index(REGIAO_POR_UF[u]), u)):
            linhas.append(_linha_territorio(base, base.sigla_uf == uf, f"{NOME_UF[uf]} ({uf})", "UF"))
        regioes = [r for r in ORDEM_REGIOES if r in {REGIAO_POR_UF[u] for u in f.ufs}]
    else:
        regioes = [r for r in ORDEM_REGIOES if not f.regioes or r in f.regioes]
    for r in regioes:
        linhas.append(_linha_territorio(base, base.regiao == r, r, "Região"))
    linhas.append(_linha_territorio(base, pd.Series(True, index=base.index), "Brasil", "Brasil"))
    return pd.DataFrame(linhas).set_index("territorio")


def ufs_de_contexto(base: pd.DataFrame, f: "Filtros") -> tuple[pd.DataFrame, list, str]:
    """UFs completas para o gráfico por estado e o título que descreve o conjunto mostrado.
    Com UF selecionada, mostra as UFs da mesma região (com a selecionada destacada), sem repetir a mesma barra."""
    if f.ufs:
        regioes = sorted({REGIAO_POR_UF[u] for u in f.ufs}, key=ORDEM_REGIOES.index)
        titulo = "Estados da região " + " e ".join(regioes) + " (todos os municípios de cada estado)"
    elif f.regioes:
        regioes = [r for r in ORDEM_REGIOES if r in f.regioes]
        titulo = "Estados da região " + " e ".join(regioes) + " (todos os municípios de cada estado)"
    else:
        regioes = ORDEM_REGIOES
        titulo = "Todos os estados (todos os municípios de cada estado)"
    ufs = sorted(u for u, r in REGIAO_POR_UF.items() if r in regioes)
    t = pd.DataFrame([_linha_territorio(base, base.sigla_uf == u, u, "UF") for u in ufs]).set_index("territorio")
    return t, list(f.ufs), titulo


def resumo_recorte(base: pd.DataFrame, f: "Filtros") -> dict:
    """Números do recorte filtrado (denominador = municípios do recorte)."""
    rec = aplicar_filtros(base, f)
    n = len(rec)
    prior = int(rec.classificacao_prioridade.isin(["alta_consenso", "alta_divergente_revisar"]).sum())
    return dict(municipios=n, prioritarios=prior, pct_prioritarios=prior / n if n else float("nan"))


def prioridade_x_alerta(base: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(base.classificacao_prioridade, base.alerta_qualidade).reindex(index=PRIORIDADES, columns=[False, True],
                                                                                    fill_value=0)
    t.columns = ["Sem alerta", "Com alerta"]
    t["Total"] = t.sum(axis=1)
    t["% com alerta"] = t["Com alerta"] / t["Total"].where(t["Total"] > 0)
    return t


def indicadores_qualidade(base: pd.DataFrame) -> pd.DataFrame:
    linhas = [
        ("Algum ponto de atenção nos dados", base.alerta_qualidade),
        ("Variação estatisticamente incomum nos últimos 36 meses", base.suspeitas_36m > 0),
        ("Variação que merece atenção em fev/2026 (sinalização causal)", base.flag_causal_fev2026 != 0),
        ("Mudança persistente reconhecida nos últimos 36 meses", base.persistencia_36m > 0),
        ("3 ou mais indicadores fora do padrão histórico", base.n_features_fora_faixa_treino >= 3),
        ("Mês ausente nos últimos 36 meses", base.meses_ausentes_36 > 0),
        ("Histórico menor que 36 meses", base.historico_meses < 36),
    ]
    alta = base.classificacao_prioridade.isin(["alta_consenso", "alta_divergente_revisar"])
    return pd.DataFrame([dict(indicador=n, municipios=int(m.sum()), em_prioridade_alta=int((m & alta).sum()))
                         for n, m in linhas])


COL_POS_RF, COL_POS_B2 = "Posição — modelo preditivo", "Posição — densidade atual"
COL_INDICE = "Índice de ordenação do modelo (técnico, não é probabilidade)"


def tabela_exibicao(base: pd.DataFrame) -> pd.DataFrame:
    """Colunas amigáveis para tabelas e para o CSV exportado (rankings nacionais preservados)."""
    return pd.DataFrame({
        "Município": base.municipio, "UF": base.sigla_uf, "Região": base.regiao, "Código IBGE": base.id_municipio,
        "Prioridade": base.classificacao_prioridade.map(ROTULO_PRIORIDADE),
        "Densidade fev/2026": base.densidade_fev2026, "Tendência (por ano)": base.tendencia_36,
        "Ponto de atenção nos dados": base.alerta_qualidade,
        COL_POS_RF: base.posicao_rf, COL_POS_B2: base.posicao_b2,
        "Critérios no top 25%": base.concordancia_top25.map(ROTULO_CONCORDANCIA), COL_INDICE: base.escore_rf,
    })


# ======================================================================================================
# Resultados técnicos (lidos das saídas aprovadas da Etapa 2, para a página Sobre)
# ======================================================================================================
DIR_TREINO = RAIZ / "etapa2" / "saidas" / "treino"


def resumo_tecnico() -> dict:
    val = pd.read_csv(DIR_TREINO / "metricas_validacao_por_coorte.csv")
    val = val[(val.grupo == "todos") & val.metodo.isin(["random_forest", "B2_densidade_atual"])]
    pr_val = val.pivot_table(index="origem", columns="metodo", values="pr_auc")
    t = pd.read_csv(ARQ_TESTE, dtype={"regra": str, "sensibilidade": str})
    t = t[(t.grupo == "todos") & (t.regra == "atual") & (t.sensibilidade == "principal")].set_index("metodo")
    conf = pd.read_csv(DIR_PREVISAO / "confiabilidade_entradas_2026.csv")
    esc = conf[conf.secao == "escore_medio_rf"][["rodada", "uso", "prevalencia", "prob_media"]]
    drift = conf[(conf.secao == "mudanca_distribuicao") & (conf.feature == "densidade_atual")].iloc[0]
    decisao = pd.read_csv(DIR_PREVISAO / "decisao_divulgacao_probabilidades.csv")
    return dict(pr_auc_validacao=pr_val, teste=resultados_teste(), brier_teste_rf=float(t.loc["random_forest", "brier"]),
                brier_teste_b0=float(t.loc["B0_taxa_base", "brier"]), escore_medio=esc, decisao=decisao,
                densidade_mediana={"2014": float(drift.mediana_2014), "2023": float(drift.mediana_2023),
                                   "2026": float(drift.mediana_2026)},
                fora_faixa_2026=float(drift.fora_p1_p99_treino_2026))
