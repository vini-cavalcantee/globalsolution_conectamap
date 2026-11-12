"""Leitura das 6 fontes e registro de metadados (B9)."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def sha256(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def hashes_brutos() -> dict:
    return {nome: sha256(C.DIR_BRUTOS / arq) for nome, arq in C.ARQUIVOS.items()}


def padronizar_id(serie: pd.Series) -> pd.Series:
    s = serie.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(7)


# --------------------------------------------------------------------------
# Anatel
# --------------------------------------------------------------------------
COLUNAS_ANATEL = ["ano", "mes", "sigla_uf", "id_municipio", "densidade"]


def carregar_anatel() -> pd.DataFrame:
    """Lê o CSV original da Anatel (ponto decimal), sem nenhum tratamento."""
    df = pd.read_csv(C.DIR_BRUTOS / C.ARQUIVOS["anatel"],
                     dtype={"id_municipio": str, "sigla_uf": str})
    df["id_municipio"] = padronizar_id(df["id_municipio"])
    return df


# --------------------------------------------------------------------------
# SIDRA / IBGE
# --------------------------------------------------------------------------
SIMBOLOS_ZERO = {"-"}                    # legenda: zero absoluto
SIMBOLOS_NAN = {"X", "..", "...", ""}    # inibido / não se aplica / não disponível

LAYOUT_SIDRA = {
    "ibge_9923": dict(skiprows=5, colunas=["id_municipio", "municipio", "pop_total", "pop_urbana", "pop_rural"]),
    "ibge_9515": dict(skiprows=4, colunas=["id_municipio", "municipio", "ind_envelhecimento", "idade_mediana", "razao_sexo"]),
    "ibge_6805": dict(skiprows=5, colunas=["id_municipio", "municipio", "dom_total", "dom_rede_ou_fossa_ligada", "dom_sem_banheiro"]),
    "ibge_10295": dict(skiprows=7, colunas=["id_municipio", "municipio", "moradores_renda", "renda_media", "renda_mediana"]),
    "ibge_10125": dict(skiprows=5, colunas=["id_municipio", "municipio", "grupo_idade", "pop_2anos_mais", "pop_com_deficiencia"]),
}
NAO_NUMERICAS = {"id_municipio", "municipio", "grupo_idade"}


def para_numero(serie: pd.Series) -> pd.Series:
    s = serie.astype(str).str.strip()
    s = s.where(~s.isin(SIMBOLOS_ZERO), "0")
    s = s.where(~s.isin(SIMBOLOS_NAN), np.nan)
    tem_virgula = s.str.contains(",", regex=False, na=False)
    s = s.where(~tem_virgula, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def carregar_sidra(chave: str):
    """Retorna (tabela numérica, contagem de símbolos por coluna)."""
    lay = LAYOUT_SIDRA[chave]
    bruto = pd.read_csv(C.DIR_BRUTOS / C.ARQUIVOS[chave], sep=";", encoding="utf-8-sig",
                        dtype=str, header=None, skiprows=lay["skiprows"],
                        names=lay["colunas"], engine="python")
    df = bruto[bruto["id_municipio"].astype(str).str.fullmatch(r"\d{7}", na=False)].copy()
    df["id_municipio"] = padronizar_id(df["id_municipio"])
    numericas = [c for c in lay["colunas"] if c not in NAO_NUMERICAS]
    simbolos = {c: {s: int((df[c].astype(str).str.strip() == s).sum())
                    for s in sorted(SIMBOLOS_ZERO | SIMBOLOS_NAN) if s}
                for c in numericas}
    for c in numericas:
        df[c] = para_numero(df[c])
    return df.reset_index(drop=True), simbolos


def texto_rodape(chave: str) -> str:
    """Notas e legenda do rodapé do arquivo SIDRA (evidência para os metadados)."""
    txt = (C.DIR_BRUTOS / C.ARQUIVOS[chave]).read_text(encoding="utf-8-sig")
    i = txt.find('"Notas"')
    return txt[i:] if i >= 0 else ""


def legenda_sidra(chave: str) -> dict:
    rod = texto_rodape(chave)
    pares = re.findall(r'"(-|X|\.\.\.|\.\.)";"([^\n"]+)', rod)
    return {s: t.strip() for s, t in pares}


# --------------------------------------------------------------------------
# B9 — Registro de metadados
# --------------------------------------------------------------------------
NV = "não verificada"
FONTE_PRINT = "tela do SIDRA (print enviado pelo grupo em 09/11/2026)"


def registro_metadados() -> pd.DataFrame:
    """Metadados das 6 fontes. Datas desconhecidas ficam como 'não verificada'."""
    rod = {k: texto_rodape(k) for k in LAYOUT_SIDRA}

    def tipo(chave):
        t = rod[chave].lower()
        if "preliminares da amostra" in t:
            return "Censo — resultados preliminares da amostra", "amostral (estimativa)"
        if "universo" in t:
            return "Censo — dados do universo", "censitária"
        return NV, NV

    ref_10295 = "31 de julho de 2022" in rod["ibge_10295"]
    linhas = [dict(
        chave="anatel", fonte="Densidade de banda larga fixa por município",
        orgao="Anatel (obtida pela Base dos Dados)", tabela="br_anatel_banda_larga_fixa.densidade_municipio",
        arquivo=C.ARQUIVOS["anatel"], granularidade="município × competência mensal (trimestral em 2007–2010)",
        unidade="acessos de banda larga fixa por 100 domicílios",
        periodo_referencia="2007-03 a 2026-03 (conforme o arquivo)",
        data_referencia="por competência (ano/mês)", status_data_referencia="verificada no arquivo",
        data_atualizacao_sidra="não se aplica", data_atualizacao_fonte=NV,
        data_primeira_divulgacao=NV, tipo_levantamento="registro administrativo",
        natureza="administrativa", observacoes=("Não é % de cobertura, disponibilidade nem qualidade. "
            "Sem coluna de status/revisão; não há evidência no arquivo de que mar/2026 seja preliminar."))]
    info = {
        "ibge_9923": ("População residente, por situação do domicílio", "pessoas", "14/11/2024"),
        "ibge_9515": ("Índice de envelhecimento, idade mediana e razão de sexo",
                      "razão (IE: 65+ por 100 de 0–14; RS: homens por 100 mulheres); anos (idade mediana)", "22/12/2023"),
        "ibge_6805": ("Domicílios particulares permanentes ocupados, por tipo de esgotamento sanitário", "domicílios", "23/02/2024"),
        "ibge_10295": ("Rendimento domiciliar per capita médio e mediano", "R$ nominais de 2022", "09/10/2025"),
        "ibge_10125": ("Pessoas de 2 anos ou mais, total e com deficiência", "pessoas", "23/05/2025"),
    }
    obs = {
        "ibge_9923": "Símbolo '-' = zero absoluto (legenda). Urbana + rural = total.",
        "ibge_9515": "IE = (P65+ / P0-14) x 100; RS = (H/M) x 100 (notas 2–4). Não são percentuais.",
        "ibge_6805": ("Categoria usada: 'Rede geral, rede pluvial ou fossa ligada à rede' -> "
                      "pct_esgoto_rede_ou_fossa_ligada (H8). Não equivale a todo esgotamento adequado."),
        "ibge_10295": "Exclui pensionistas e empregados domésticos e parentes (nota 2). Sem atualização monetária.",
        "ibge_10125": "Deficiência = 'muita dificuldade' ou 'não consegue' em ao menos um domínio (nota 2). Ruído maior em municípios pequenos.",
    }
    for k, (nome, unidade, atualiz) in info.items():
        tl, nat = tipo(k)
        linhas.append(dict(
            chave=k, fonte=nome, orgao="IBGE — Censo Demográfico 2022 (SIDRA)", tabela=k.split("_")[1],
            arquivo=C.ARQUIVOS[k], granularidade="município (5.570)", unidade=unidade,
            periodo_referencia="Censo 2022",
            data_referencia="2022-07-31",
            status_data_referencia=("verificada nas notas deste arquivo" if (k == "ibge_10295" and ref_10295)
                                    else "data de referência do Censo 2022; consta nas notas da tabela 10295, não neste arquivo"),
            data_atualizacao_sidra=atualiz, data_atualizacao_fonte=FONTE_PRINT,
            data_primeira_divulgacao=NV, tipo_levantamento=tl, natureza=nat, observacoes=obs[k]))
    df = pd.DataFrame(linhas)
    df["nota_disponibilidade"] = ("A data de atualização do SIDRA é apenas um limite superior: "
                                  "o dado existia, no máximo, nessa data. Não equivale à primeira divulgação.")
    df.loc[df.chave == "anatel", "nota_disponibilidade"] = "Disponibilidade por competência não verificada."
    return df
