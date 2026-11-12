"""Testes da Etapa 2 (painel). Rodar na pasta do projeto:  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pasta do projeto (contém conectamap/)

from conectamap import painel as PN  # noqa: E402


def competencias():
    """Calendário da Anatel: trimestral 2007–2010, mensal 2011-01..2026-03."""
    tri = [f"{a}-{m:02d}" for a in range(2007, 2011) for m in (3, 6, 9, 12)]
    men = [str(p) for p in pd.period_range("2011-01", "2026-03", freq="M")]
    return tri + men


def sintetico(series: dict, ausentes=(), flags=None):
    """Bases no formato exportado pelo Notebook 1 a partir de {id: função(competencia)->densidade}."""
    linhas, ind = [], []
    aus = set(ausentes)
    for cod, fn in series.items():
        for c in competencias():
            if (cod, c) in aus:
                continue
            v = fn(c)
            dentro = c <= "2026-02"
            linhas.append(dict(id_municipio=cod, sigla_uf="SP", competencia=c,
                               periodicidade="trimestral" if c < "2011-01" else "mensal",
                               dentro_limite_observacao=dentro, densidade_original=v, densidade_causal=v))
            fl = (flags or {}).get((cod, c), 0) if "2011-01" <= c <= "2026-02" else np.nan
            ind.append(dict(id_municipio=cod, competencia=c, dentro_limite_observacao=dentro, flag_causal=fl,
                            flag_causal_L6=fl, referencia_insuficiente=False if "2011-01" <= c <= "2026-02" else np.nan))
    base, ind = pd.DataFrame(linhas), pd.DataFrame(ind)
    a = pd.DataFrame([dict(id_municipio=k, sigla_uf="SP", competencia=c) for k, c in aus],
                     columns=["id_municipio", "sigla_uf", "competencia"])
    ibge = pd.DataFrame({"id_municipio": list(series)})
    for v in PN.VARIAVEIS_CENSO:
        ibge[v] = 1.0
    return base, ind, a, ibge


@pytest.fixture(scope="session")
def dados_reais():
    if not (PN.C.DIR_TRATADOS / "anatel_densidade_tratada.csv").exists():
        pytest.skip("dados_tratados/ ausente: rode o Notebook 1")
    return PN.carregar_bases()
