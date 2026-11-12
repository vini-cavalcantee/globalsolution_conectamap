"""Componentes visuais do dashboard: marca ConectaMap, cabeçalhos, cartões, notas e rodapé."""
from __future__ import annotations

import html

import streamlit as st

import paleta as P

PETROLEO, TERRACOTA = P.PETROLEO, P.TERRACOTA

ICONE_SVG = (
    '<svg viewBox="0 0 64 64" aria-hidden="true" focusable="false">'
    '<path d="M32 4C20.4 4 11 13.2 11 24.6 11 40.3 32 60 32 60s21-19.7 21-35.4C53 13.2 43.6 4 32 4z" fill="#17324D"/>'
    '<g stroke="#FFFFFF" stroke-width="2.6" stroke-linecap="round" fill="none" opacity="0.9">'
    '<line x1="23" y1="29" x2="32" y2="17"/><line x1="32" y1="17" x2="41" y2="29"/><line x1="23" y1="29" x2="41" y2="29"/></g>'
    '<circle cx="32" cy="17" r="4.6" fill="#C96D46" stroke="#FFFFFF" stroke-width="1.6"/>'
    '<circle cx="23" cy="29" r="4" fill="#FFFFFF"/><circle cx="41" cy="29" r="4" fill="#FFFFFF"/></svg>')

CSS_MODELO = """
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 2.5rem; max-width: 1240px;}
h1, h2, h3, h4 {color: @PETROLEO; letter-spacing: -0.01em;}
/* barra lateral: cinza-azulado muito claro, página ativa em azul */
[data-testid="stSidebar"] {background: @FUNDO_LATERAL; border-right: 1px solid @BORDA;}
[data-testid="stSidebarNavLink"] {border-radius: 8px;}
[data-testid="stSidebarNavLink"][aria-current="page"] {background: @ATIVO_LATERAL; box-shadow: inset 3px 0 0 @ALTA;}
[data-testid="stSidebarNavLink"][aria-current="page"] span {color: @PETROLEO !important; font-weight: 600;}
[data-testid="stSidebarNavSeparator"] {border-color: @BORDA;}
.cm-filtros-titulo {font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: @TINTA_2; margin: .2rem 0 .1rem 0;}
/* marca */
.cm-marca {display: inline-flex; align-items: center; gap: .45rem; font-weight: 700; letter-spacing: -0.02em;
  line-height: 1; white-space: nowrap;}
.cm-marca svg {height: 1.15em; width: 1.15em; flex: none;}
.cm-marca .c {color: #17324D;} .cm-marca .m {color: #C96D46;}
.cm-marca.g {font-size: 2.45rem;} .cm-marca.p {font-size: 1.02rem;}
/* cabeçalho da página inicial */
.cm-hero {padding: .2rem 0 .9rem 0; border-bottom: 1px solid @BORDA; margin-bottom: 1.1rem;}
.cm-hero .tag {font-size: .92rem; color: @TINTA_2; margin: .35rem 0 1.1rem 0;}
.cm-hero h1 {font-size: 1.75rem; line-height: 1.25; margin: 0; padding: 0; color: @PETROLEO; font-weight: 700;}
.cm-hero .sub {font-size: 1.05rem; color: #34414C; margin: .35rem 0 .55rem 0; max-width: 54rem;}
.cm-contexto {display: inline-flex; flex-wrap: wrap; gap: .4rem; font-size: .82rem; color: #34414C;}
.cm-contexto span {background: @FUNDO_SUAVE; border: 1px solid @BORDA; border-radius: 999px; padding: .12rem .65rem;}
/* cabeçalho das páginas internas */
.cm-pagina {margin: 0 0 .9rem 0; padding-bottom: .6rem; border-bottom: 1px solid @BORDA;}
.cm-pagina .t {font-size: 1.55rem; font-weight: 700; color: @PETROLEO; margin: .45rem 0 .15rem 0; line-height: 1.25;}
.cm-pagina .o {font-size: .98rem; color: #4A5661; margin: 0;}
/* cartões */
.cm-card {background: #FFFFFF; border: 1px solid @BORDA; border-radius: 12px; padding: .85rem 1rem .8rem 1rem;
  height: 100%; box-shadow: 0 1px 2px rgba(23,50,77,.05);}
.cm-card.destaque {background: @DESTAQUE_SUAVE; border-color: #BFD3E4;}
.cm-card .barra {height: 4px; width: 2.4rem; border-radius: 2px; margin-bottom: .55rem;}
.cm-card .rot {font-size: .86rem; color: #4A5661; line-height: 1.3; min-height: 2.25em;}
.cm-card .val {font-size: 2.05rem; font-weight: 700; color: @PETROLEO; line-height: 1.1; margin-top: .15rem;
  font-variant-numeric: tabular-nums;}
.cm-card .val.txt {font-size: 1.15rem; line-height: 1.3;}
.cm-card .nac {font-size: .8rem; color: #66727D; margin-top: .3rem; min-height: 1.1em;}
.cm-sec {display: flex; flex-wrap: wrap; gap: .5rem 1.4rem; font-size: .9rem; color: #34414C; margin: .7rem 0 0 0;}
.cm-sec b {color: @PETROLEO; font-variant-numeric: tabular-nums;}
.cm-info {color: @ALTA; cursor: help; font-size: .95rem; margin-left: .15rem;}
/* faixas, notas e legendas */
.cm-recorte {font-size: .86rem; color: @PETROLEO; background: #EAF1F7; border-left: 3px solid @ALTA;
  padding: .45rem .75rem; border-radius: 4px; margin: 0 0 .9rem 0;}
.cm-nota {font-size: .92rem; color: #34414C; background: @FUNDO_SUAVE; border-radius: 8px; padding: .65rem .85rem;}
.cm-legenda-item {display: flex; gap: .55rem; align-items: flex-start; margin-bottom: .4rem; font-size: .9rem;
  color: #34414C;}
.cm-legenda-item .q {width: 14px; height: 14px; border-radius: 3px; flex: none; margin-top: .2rem;
  border: 1px solid rgba(23,50,77,.3);}
.cm-legenda-item b {color: @TINTA;}
.cm-atencao {display: inline-flex; gap: .4rem; align-items: center; font-size: .86rem; color: #4B3470;
  background: #F1ECF7; border: 1px solid #DCD0EC; border-radius: 6px; padding: .3rem .6rem;}
.cm-atencao .p {width: 9px; height: 9px; border-radius: 50%; background: @ATENCAO; flex: none;}
.cm-chip {display: inline-flex; gap: .35rem; align-items: center; font-size: .82rem; color: @TINTA;
  border: 1px solid @BORDA; border-radius: 999px; padding: .12rem .55rem; background: #FFFFFF;}
.cm-chip .q {width: 10px; height: 10px; border-radius: 2px; border: 1px solid rgba(23,50,77,.3);}
.cm-rodape {margin-top: 2.2rem; padding-top: .7rem; border-top: 1px solid @BORDA; font-size: .8rem; color: #66727D;
  display: flex; flex-wrap: wrap; gap: .4rem 1rem; align-items: center;}
@media (max-width: 640px) {
  .cm-marca.g {font-size: 1.9rem;} .cm-hero h1 {font-size: 1.4rem;} .cm-card .val {font-size: 1.7rem;}
}
</style>
"""
CSS = CSS_MODELO
for _nome in ("PETROLEO", "FUNDO_LATERAL", "ATIVO_LATERAL", "BORDA", "ALTA", "TINTA_2", "TINTA", "FUNDO_SUAVE",
              "DESTAQUE_SUAVE", "ATENCAO"):
    CSS = CSS.replace("@" + _nome + ";", getattr(P, _nome) + ";").replace("@" + _nome + " ", getattr(P, _nome) + " ") \
        .replace("@" + _nome + "}", getattr(P, _nome) + "}")


def estilo():
    st.markdown(CSS, unsafe_allow_html=True)


def marca(tamanho: str = "p") -> str:
    """Assinatura ConectaMap: 'Conecta' em azul-petróleo + 'Map' em terracota, sem espaço, com ícone discreto."""
    return f'<span class="cm-marca {tamanho}">{ICONE_SVG}<span><span class="c">Conecta</span><span class="m">Map</span></span></span>'


def cabecalho_inicial(titulo: str, subtitulo: str, contexto: list[str]):
    chips = "".join(f"<span>{html.escape(c)}</span>" for c in contexto)
    st.markdown(
        f'<div class="cm-hero">{marca("g")}<div class="tag">Inteligência territorial para a conectividade</div>'
        f'<h1>{html.escape(titulo)}</h1><p class="sub">{html.escape(subtitulo)}</p>'
        f'<div class="cm-contexto">{chips}</div></div>', unsafe_allow_html=True)


def cabecalho_interno(titulo: str, objetivo: str):
    st.markdown(f'<div class="cm-pagina">{marca("p")}<p class="t">{html.escape(titulo)}</p>'
                f'<p class="o">{html.escape(objetivo)}</p></div>', unsafe_allow_html=True)


def card(rotulo: str, valor: str, nacional: str | None = None, cor: str | None = None, ajuda: str | None = None,
         texto: bool = False, destaque: bool = False):
    barra = f'<div class="barra" style="background:{cor}"></div>' if cor else '<div class="barra"></div>'
    nac = html.escape(nacional) if nacional else "&nbsp;"
    titulo = f' title="{html.escape(ajuda)}"' if ajuda else ""
    st.markdown(f'<div class="cm-card{" destaque" if destaque else ""}"{titulo}>{barra}<div class="rot">{html.escape(rotulo)}</div>'
                f'<div class="val{" txt" if texto else ""}">{html.escape(valor)}</div><div class="nac">{nac}</div></div>',
                unsafe_allow_html=True)


def indicadores_secundarios(itens: list[tuple]):
    """Itens (valor, rótulo) ou (valor, rótulo, ajuda); com ajuda, um ícone ⓘ mostra o texto ao passar o mouse."""
    def _item(v, r, ajuda=None):
        info = (f' <span class="cm-info" title="{html.escape(ajuda)}" aria-label="{html.escape(ajuda)}" role="img"'
                f' tabindex="0">ⓘ</span>') if ajuda else ""
        return f"<span><b>{html.escape(v)}</b> {html.escape(r)}{info}</span>"
    st.markdown('<div class="cm-sec">' + "".join(_item(*i) for i in itens) + "</div>", unsafe_allow_html=True)


def faixa_recorte(descricao: str, n: int):
    st.markdown(f'<div class="cm-recorte"><b>Recorte filtrado:</b> {html.escape(descricao)} — {fmt(n)} municípios. '
                'Abaixo de cada número aparece o total do Brasil.</div>', unsafe_allow_html=True)


def nota(texto: str):
    st.markdown(f'<div class="cm-nota">{texto}</div>', unsafe_allow_html=True)


def atencao(texto: str):
    st.markdown(f'<span class="cm-atencao"><span class="p"></span>{html.escape(texto)}</span>', unsafe_allow_html=True)


def legenda_prioridades(cores: dict, rotulos: dict, descricoes: dict):
    itens = "".join(f'<div class="cm-legenda-item"><div class="q" style="background:{cores[p]}"></div>'
                    f'<div><b>{html.escape(rotulos[p])}</b> — {html.escape(descricoes[p])}</div></div>' for p in rotulos)
    st.markdown(itens, unsafe_allow_html=True)


def chip(cor: str, texto: str) -> str:
    return f'<span class="cm-chip"><span class="q" style="background:{cor}"></span>{html.escape(texto)}</span>'


def titulo_filtros():
    st.markdown('<div class="cm-filtros-titulo">Filtros</div>', unsafe_allow_html=True)


def sem_resultados():
    st.info("Nenhum município corresponde aos filtros escolhidos. Ajuste ou limpe os filtros na barra lateral.",
            icon=":material/search_off:")


def rodape():
    st.markdown(f'<div class="cm-rodape">{marca("p")}<span>Global Solution FIAP 2026.2 · CitizenTech</span>'
                '<span>Dados: Anatel (banda larga fixa) e IBGE (municípios e malha 2022)</span>'
                '<span>Data de referência: fev/2026 · Horizonte: fev/2029</span></div>', unsafe_allow_html=True)


def fmt(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def pct(x, casas: int = 1) -> str:
    return f"{x:.{casas}%}".replace(".", ",")


def dec(x, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


def sinal(x, casas: int = 3) -> str:
    return f"{x:+.{casas}f}".replace(".", ",").replace("-", "−")
