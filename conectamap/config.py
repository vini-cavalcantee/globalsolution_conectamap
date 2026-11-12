"""B12 — Parâmetros centralizados do pré-processamento do ConectaMap.

Separação obrigatória:
- PARAMETROS_APROVADOS: decisões já aprovadas pelo grupo (H1, H4, H11...).
- PARAMETROS_CANDIDATOS: valores em calibração. NÃO são definitivos (H5).
"""
from pathlib import Path

VERSAO_METODOLOGIA = "etapa1_5-2026.11"  # Etapa 1.5: revisão, testes e calibração (sem correção)
VERSAO_REGRA_D = "D-rev3-candidata"     # rev3: transição causal ±2, evento coletivo, nomenclatura

# --------------------------------------------------------------------------
# Caminhos — resolvidos a partir da pasta do projeto (a que contém `conectamap/`),
# para funcionar tanto no notebook de entrega (raiz) quanto na auditoria técnica.
# --------------------------------------------------------------------------
PACOTE = Path(__file__).resolve().parent.parent
DIR_BRUTOS = PACOTE / "dados_brutos"                # entradas — somente leitura
DIR_TRATADOS = PACOTE / "dados_tratados"           # bases finais do notebook de entrega
RAIZ = PACOTE / "auditoria_tecnica"                # materiais técnicos das Etapas 1 e 1.5
DIR_SAIDAS = RAIZ / "saidas" / "etapa1_5"          # saídas da Etapa 1.5
DIR_SAIDAS_ETAPA1 = RAIZ / "saidas" / "etapa1"     # saídas da Etapa 1 (somente leitura)
DIR_METADADOS = DIR_SAIDAS / "metadados"
DIR_AUDITORIA = DIR_SAIDAS / "auditoria"
DIR_EDA = DIR_SAIDAS / "eda_temporal"
DIR_REVISAO = DIR_SAIDAS / "revisao"                    # material CEGO entregue aos revisores
DIR_REVISAO_TECNICA = DIR_SAIDAS / "revisao_tecnica"    # classificação do algoritmo (NÃO entregar aos revisores)
DIR_REVISAO_INTERNA = DIR_SAIDAS / "revisao_interna"    # formulário consolidado interno (NÃO distribuir antes da revisão)
DIR_REVISAO_COMPLEMENTAR = DIR_SAIDAS / "revisao_complementar"  # rodada complementar (C05 + K01–K24), material cego
DIR_VERSAO_ANTERIOR = RAIZ / "versoes" / "v1"      # notebook e saídas preservados

ARQUIVOS = {
    "anatel": "br_anatel_banda_larga_fixa_densidade_municipio.csv",
    "ibge_9923": "tabela9923.csv",
    "ibge_9515": "tabela9515.csv",
    "ibge_6805": "tabela6805.csv",
    "ibge_10295": "tabela10295.csv",
    "ibge_10125": "tabela10125.csv",
}
LOG_REGRA_ANTIGA = DIR_VERSAO_ANTERIOR / "saidas_v1" / "log_correcoes_anatel.csv"

# --------------------------------------------------------------------------
# Parâmetros APROVADOS
# --------------------------------------------------------------------------
PARAMETROS_APROVADOS = {
    "n_municipios": 5570,
    "competencia_final": "2026-02",          # H1: competência final provisória
    "limite_observacao": "2026-02",          # nenhuma observação posterior entra na análise principal
    "competencia_auditoria_extra": "2026-03",  # preservada só para auditoria/sensibilidade
    "inicio_periodo_mensal": "2011-01",      # H4: detecção só no período mensal
    "inicio_revisao_prioritaria": "2018-01", # H4: prioridade de revisão
    "horizonte_meses": 36,                   # previsão de 3 anos
    "semente": 42,
}

# --------------------------------------------------------------------------
# Parâmetros CANDIDATOS da regra D (em calibração — não definitivos)
# --------------------------------------------------------------------------
PARAMETROS_CANDIDATOS = {
    # grade de calibração
    "grade_K": [3, 5, 10],          # magnitude de entrada (vezes a referência)
    "grade_A": [5, 10, 20],         # diferença absoluta mínima (acessos/100 dom.)
    "grade_L": [3, 6, 12],          # duração máxima para haver retorno (meses)
    # configuração de COMPARAÇÃO com a Etapa 1 (exploratória, não aprovada)
    "K": 5, "A": 10, "L": 12,
    # referência histórica
    "janela_ref_meses": 12,         # meses do calendário olhados para trás
    "max_obs_ref": 6,               # até 6 observações válidas mais recentes
    "min_obs_ref": 3,               # abaixo disso: "sem referência suficiente"
    # entrada e saída do episódio (histerese)
    "fator_banda_saida": 2.0,       # retorno = valor dentro de [base/2, base*2] (escala log+1)
    "obs_confirmacao_retorno": 2,   # retorno confirmado = 2 observações seguidas na banda
    "base_min_queda": 5.0,          # quedas só avaliadas com base >= 5
    "fracao_rampa": 0.5,            # entrada/saída abrupta se o último passo explica >= 50% do desvio (log)
    # níveis de evidência
    "k_confirmada": 10,             # magnitude mínima para "estatisticamente confirmada"
    "k_provavel": 5,                # magnitude mínima para "provável"
    "duracao_max_provavel": 3,      # meses
    "duracao_ida_volta": 2,         # sinal extra: ida e volta em até 2 meses (evento coletivo não conta)
    "percentil_choque_comum": 95,   # evento coletivo: inícios no mês acima do p95 (bloqueia tratamento)
    "min_inicios_evento_coletivo": 10,  # e no mínimo 10 inícios no mês (evita "coletivo" com 1 caso)
    "fator_implausivel": 3.0,       # sinal extra: pico > 3x o máximo fora de episódios
}

# Configuração RECOMENDADA PROVISORIAMENTE na Etapa 1.5 (depende da revisão humana)
CONFIG_RECOMENDADA_PROVISORIA = {"K": 5, "A": 10, "L": 6}

# Parâmetros da amostra de revisão manual
VERSAO_MATERIAL_REVISAO = "cego-v3"   # encerramento da Etapa 1.5: sem outros períodos/faixas cinza; rodada complementar

# Controles complementares K01–K24 (aprovados em 2026-11-10). Não estimam recall nem taxa de falsos negativos.
PARAMETROS_CONTROLES_K = {
    "semente": 20261110,            # registrada; fluxos independentes [semente, 1..4]
    "inicio": "2018-01",
    "fim": "2026-02",
    "tamanho_janela": 24,
    "min_obs_validas": 22,
    "n_aleatorio": 18,
    "n_limiar": 6,
    "cortes_densidade": (5, 20),     # faixas pela mediana do município em 2018-01..2026-02
    "fator_limiar": 2.0,             # "perto do limiar": desvio >= x2 sem satisfazer a entrada da regra
    "configuracao_referencia": "K=5, A=10, L=12",
}

PARAMETROS_AMOSTRA = {
    "n_por_estrato": 4,
    "semente": 42,
    "meses_contexto": 12,
    "inicio_estratos": "2018-01",   # estratos sorteados no período prioritário
}

# Casos históricos citados nas auditorias (fixos na amostra)
# modo: "episodio" (usa o episódio que contém mes_ref), "controle_mar2026"
# (evento fora do limite de observação), "controle_pre2011" (fora do escopo
# temporal), "controle" (sem episódio esperado).
CASOS_HISTORICOS = {
    "1507755": ("PA — sequência anômala de 2025", "episodio", "2025-08"),
    "3115805": ("MG — padrão alternado 2013–2015", "episodio", "2014-01"),
    "3556008": ("SP Urupês — alta gradual 2022–2026", "episodio", "2022-10"),
    "3300225": ("RJ Areal — oscilações extremas 2023–2025", "episodio", "2024-10"),
    "4200200": ("SC Agrolândia — alta gradual 2015–2017", "controle", "2016-07"),
    "3510906": ("SP — correção antiga de set/2007 (período trimestral)", "controle_pre2011", "2007-09"),
    "4319711": ("RS — 47.996,81 em mar/2026", "controle_mar2026", "2026-03"),
    "5201207": ("GO — 476,55 em mar/2026", "controle_mar2026", "2026-03"),
    "3501152": ("SP Alumínio — 134,61 em mar/2026", "controle_mar2026", "2026-03"),
    "4323309": ("RS — queda 38,2 → 1,9 em mar/2026", "controle_mar2026", "2026-03"),
}

# Definições usadas na análise exploratória (reprodução do diagnóstico)
PARAMETROS_EDA = {
    "fator_abrupto": 5,             # queda/subida abrupta: 5x a mediana de referência
    "base_min_queda_eda": 10,       # quedas só com base >= 10 (como no diagnóstico)
    "dif_min_subida_eda": 20,       # subidas com diferença >= 20 (como no diagnóstico)
    "variacao_forte": 0.5,          # variação mensal > 50%
    "corte_descritivo": 7.62,       # corte DESCRITIVO (Q1 mar/2026) — não é target
}


DIR_CALIBRACAO = DIR_SAIDAS / "calibracao"
DIR_TESTES = DIR_SAIDAS / "testes"
DIR_SIMULACOES = DIR_SAIDAS / "simulacoes"   # cenários simulados — NUNCA séries oficiais

# Casos de referência para a sensibilidade (comportamento, não rótulo de verdade)
CASOS_REFERENCIA = {
    "1507755_2025-08": "PA Sapucaia — sequência alta 2025",
    "1507755_2023-09": "PA Sapucaia — pico isolado 2023",
    "3115805_2014-01": "MG Centralina — padrão alternado",
    "2508604_2024-07": "PB Lucena — quedas alternadas",
    "3556008_2022-10": "SP Urupês — alta e persistência",
    "4200200_2016-07": "SC Agrolândia — alta gradual",
    "3300225_2024-10": "RJ Areal — pico extremo",
    "5207808_2025-10": "GO Firminópolis — alta no fim da série",
}


def criar_pastas_saida():
    for d in (DIR_METADADOS, DIR_AUDITORIA, DIR_EDA, DIR_REVISAO / "graficos", DIR_REVISAO_TECNICA / "graficos", DIR_REVISAO_INTERNA, DIR_CALIBRACAO,
              DIR_TESTES, DIR_SIMULACOES):
        d.mkdir(parents=True, exist_ok=True)
