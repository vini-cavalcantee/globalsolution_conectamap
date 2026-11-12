"""Paleta única do ConectaMap (versão 3). Todas as cores da interface vêm daqui.

Identidade predominantemente azul; o terracota aparece só na palavra "Map" da marca.
"""

# ---- marca -------------------------------------------------------------------------------------------
PETROLEO = "#17324D"          # "Conecta", títulos, navegação
TERRACOTA = "#C96D46"         # apenas "Map" na marca

# ---- dados (semântica fixa em mapas, gráficos, legendas, cartões e tabelas) -----------------------------
ALTA = "#1E5A8A"              # Alta prioridade — métodos concordam
DIVERGE = "#71849E"           # Métodos divergem — analisar
INTERMEDIARIA = "#86BCD8"     # Prioridade intermediária
MENOR = "#DFE7ED"             # Menor prioridade
ATENCAO = "#7856A6"           # Pontos de atenção nos dados

COR_PRIORIDADE = {"alta_consenso": ALTA, "alta_divergente_revisar": DIVERGE,
                  "intermediaria": INTERMEDIARIA, "menor": MENOR}
# classificação de um único critério (modo avançado): mesma escala alta → intermediária → menor
COR_CATEGORIA = {"maior": ALTA, "intermediario": INTERMEDIARIA, "menor": MENOR}
COR_ALERTA = {True: ATENCAO, False: MENOR}
COR_CONCORDANCIA = {"ambos": ALTA, "so_rf": DIVERGE, "so_b2": DIVERGE, "nenhum": MENOR}

# fundos claros para identificar a prioridade em tabelas (texto escuro por cima)
FUNDO_TABELA = {"alta_consenso": "#D9E6F1", "alta_divergente_revisar": "#E4E8EE",
                "intermediaria": "#E9F3F9", "menor": "#FFFFFF"}

# ---- neutros e interface ---------------------------------------------------------------------------
FUNDO = "#FFFFFF"
FUNDO_LATERAL = "#EEF3F7"     # barra lateral: cinza-azulado muito claro
ATIVO_LATERAL = "#D6E4F0"     # página ativa na navegação
FUNDO_SUAVE = "#F3F6F8"       # notas, selos
DESTAQUE_SUAVE = "#F2F7FB"    # cartão de alta prioridade (destaque discreto)
BORDA = "#DFE5EA"
BORDA_MARCA_CLARA = "#9DAEBD"  # contorno de marcas claras sobre fundo branco
FORA_FILTRO = "#F4F6F8"
TINTA = "#1E2A35"
TINTA_2 = "#55626E"
GRADE = "#E7ECF0"
REFERENCIA = "#A9B6C2"        # barras de referência (região completa, Brasil)

# ---- verificação simples de contraste (WCAG) --------------------------------------------------------


def _lum(hexcor: str) -> float:
    r, g, b = (int(hexcor[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contraste(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)
