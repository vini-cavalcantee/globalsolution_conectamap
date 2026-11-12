"""Disponibilidade temporal das variáveis do IBGE (Censo 2022) — especificação (encerramento da Etapa 1.5).

Três critérios para decidir se uma variável pode entrar como feature numa origem (mês de previsão) t:

- "nenhum":     sempre disponível. HIPÓTESE explícita de estabilidade estrutural (o retrato de 2022 vale
                para origens anteriores). Anacrônico; só para análise de sensibilidade.
- "conteudo":   disponível a partir do mês seguinte à data de referência (31/07/2022 -> 2022-08).
                O CONTEÚDO já descreve a realidade em t, mas o dado ainda não tinha sido publicado.
- "publicacao": disponível a partir do mês seguinte à data em que o dado comprovadamente existia.
                Como a primeira divulgação NÃO foi verificada, usa-se o LIMITE SUPERIOR conhecido
                (data de atualização do SIDRA registrada pelo grupo). É conservador: a disponibilidade
                real pode ter sido anterior, nunca posterior. Sem data conhecida -> indisponível.

Nada aqui monta o painel nem cria features: só define e verifica a regra.
"""
import numpy as np
import pandas as pd

CRITERIOS = ("nenhum", "conteudo", "publicacao")

# Variáveis efetivamente montadas no Notebook 01 (tabela `ibge`) e as fontes de que dependem.
FONTES_VARIAVEIS = {
    "pop_total": ["ibge_9923"], "pop_urbana": ["ibge_9923"], "pop_rural": ["ibge_9923"], "pct_rural": ["ibge_9923"],
    "ind_envelhecimento": ["ibge_9515"], "idade_mediana": ["ibge_9515"], "razao_sexo": ["ibge_9515"],
    "dom_total": ["ibge_6805"], "dom_rede_ou_fossa_ligada": ["ibge_6805"], "dom_sem_banheiro": ["ibge_6805"],
    "pct_esgoto_rede_ou_fossa_ligada": ["ibge_6805"], "pct_sem_banheiro": ["ibge_6805"],
    "moradores_renda": ["ibge_10295"], "renda_media": ["ibge_10295"], "renda_mediana": ["ibge_10295"],
    "pop_2anos_mais": ["ibge_10125"], "pop_com_deficiencia": ["ibge_10125"], "pct_deficiencia": ["ibge_10125"],
}
IDENTIFICADORES = {"id_municipio", "municipio", "sigla_uf"}


def _mes_seguinte(data_txt):
    """'14/11/2024' ou '2022-07-31' -> Period('2024-12'). Vazio/não verificada -> None."""
    if data_txt is None or (isinstance(data_txt, float) and np.isnan(data_txt)):
        return None
    t = str(data_txt).strip()
    if not t or "verific" in t.lower() or "aplica" in t.lower():
        return None
    d = pd.to_datetime(t, dayfirst="/" in t, errors="coerce")
    return None if pd.isna(d) else d.to_period("M") + 1


def inventario_variaveis(colunas, metadados: pd.DataFrame, fontes=None) -> pd.DataFrame:
    """Uma linha por variável: fontes, data de referência, disponibilidade por critério e status das datas."""
    fontes = FONTES_VARIAVEIS if fontes is None else fontes
    meta = metadados.set_index("chave")
    linhas = []
    for v in colunas:
        if v not in fontes:
            continue
        fs = fontes[v]
        ref = [_mes_seguinte(meta.at[f, "data_referencia"]) for f in fs]
        pub = [_mes_seguinte(meta.at[f, "data_atualizacao_sidra"]) for f in fs]
        prim = [str(meta.at[f, "data_primeira_divulgacao"]) for f in fs]
        linhas.append(dict(
            variavel=v, fontes="; ".join(fs), tabelas_sidra="; ".join(str(meta.at[f, "tabela"]) for f in fs),
            data_referencia="; ".join(sorted({str(meta.at[f, "data_referencia"]) for f in fs})),
            disponivel_conteudo_a_partir=str(max(ref)) if all(ref) else "",
            data_atualizacao_sidra_limite_superior="; ".join(str(meta.at[f, "data_atualizacao_sidra"]) for f in fs),
            disponivel_publicacao_a_partir=str(max(pub)) if all(pub) else "",
            primeira_divulgacao="; ".join(sorted(set(prim))),
            status_data_publicacao=("limite superior (atualização do SIDRA); primeira divulgação não verificada"
                                    if all(pub) else "sem data conhecida: indisponível no critério publicação"),
        ))
    return pd.DataFrame(linhas)


def disponivel(inv: pd.DataFrame, origem, criterio: str) -> pd.Series:
    """Booleano por variável: a variável pode ser usada como feature na origem `origem` ('AAAA-MM')?"""
    if criterio not in CRITERIOS:
        raise ValueError(f"critério desconhecido: {criterio}; use {CRITERIOS}")
    o = pd.Period(origem, "M")
    if criterio == "nenhum":
        return pd.Series(True, index=inv.variavel.values)
    col = "disponivel_conteudo_a_partir" if criterio == "conteudo" else "disponivel_publicacao_a_partir"
    return pd.Series([bool(x) and pd.Period(x, "M") <= o for x in inv[col]], index=inv.variavel.values)


def origens_x_criterios(inv: pd.DataFrame, origens, horizonte: int, limite_alvo: str) -> pd.DataFrame:
    """Para cada origem: o alvo (origem + horizonte) é observável até `limite_alvo`? Quantas variáveis do
    Censo estariam disponíveis em cada critério?"""
    lim = pd.Period(limite_alvo, "M")
    linhas = []
    for o in origens:
        p = pd.Period(o, "M")
        d = dict(origem=str(p), mes_alvo=str(p + horizonte), alvo_observavel=(p + horizonte) <= lim)
        for c in CRITERIOS:
            d[f"variaveis_{c}"] = int(disponivel(inv, str(p), c).sum())
        linhas.append(d)
    return pd.DataFrame(linhas)
