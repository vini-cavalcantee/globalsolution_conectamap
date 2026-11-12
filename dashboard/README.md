# ConectaMap — Dashboard interativo (Etapa 3)

Aplicação em Streamlit que apresenta a **priorização dos municípios brasileiros para fevereiro de 2029**. Ela usa só as saídas aprovadas da Etapa 2:
- não treina modelos;
- não recalcula escores, rankings nem categorias;
- não estima a densidade de 2029.

## Como executar
Na **raiz do repositório**, com Python 3.10 ou superior (testado no 3.13):

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

**No navegador:** o app abre em `http://localhost:8501`.

**Rede:** nenhuma etapa da execução depende de internet. O mapa é desenhado só com a malha local, sem mapa de fundo nem servidor de blocos. O navegador precisa ter WebGL, o que é padrão nos navegadores atuais.

**Tema visual:** fica em `.streamlit/config.toml`, na raiz do repositório, e só é aplicado quando o comando é executado a partir dessa pasta. Fora dela, o app funciona com o tema padrão do Streamlit.

**Testes** (o `pytest` não faz parte do `requirements.txt` do app):
```bash
pip install pytest
python -m pytest -q dashboard/tests
```
São 56 casos e levam cerca de 12 s.

## Identidade visual e linguagem
- **Marca:** ConectaMap, com "Conecta" em azul-petróleo (`#17324D`) e "Map" em terracota (`#C96D46`), juntos e com um ícone de localização e conexão. O terracota aparece só na marca.
- **Paleta:** todas as cores ficam num único módulo, `dashboard/paleta.py`. Nenhuma outra cor está escrita nos demais arquivos, e um teste confere isso. As cores têm significado fixo e são as mesmas em mapas, gráficos, legendas, cartões e tabelas:

  | Significado | Cor |
  |---|---|
  | Alta prioridade — métodos concordam | azul profundo `#1E5A8A` |
  | Métodos divergem — analisar | azul-acinzentado `#71849E` |
  | Prioridade intermediária | azul-claro `#86BCD8` |
  | Menor prioridade | cinza azulado `#DFE7ED` |
  | Pontos de atenção nos dados | roxo `#7856A6` |

- **Acessibilidade:** as categorias têm nome escrito nas legendas e nas dicas do mapa, e não dependem só da cor.
  - Os municípios com classificação divergente recebem contorno escuro no mapa; na dispersão, aparecem como triângulos e losangos.
  - As marcas claras têm borda.
  - O texto escuro sobre os fundos claros atinge contraste WCAG AA de pelo menos 4,5, o que também é verificado em teste.
- **Linguagem:** a interface usa termos simples (modelo preditivo, densidade atual, patamar de referência, pontos de atenção nos dados, classificação divergente). Os nomes técnicos (Random Forest, B2, P25) ficam nas áreas de comparação e em "Sobre o ConectaMap". As colunas dos arquivos não mudaram.

## Comparação territorial (regra dos denominadores)
- **Os gráficos territoriais usam sempre territórios completos:** todos os municípios da UF, da região ou do Brasil.
  - Com **uma UF** filtrada, a comparação é UF × região completa × Brasil. Por exemplo, Piauí 50,9%, Nordeste 42,5% e Brasil 25,6%, todos calculados a partir da base.
  - Com **várias UFs**, aparecem cada UF, as regiões delas e o Brasil.
  - Com **região** filtrada, aparecem a região completa e o Brasil.
  - **Sem filtro**, aparecem as cinco regiões e o Brasil.
- **O gráfico por estado** mostra os estados completos da região correspondente, com a UF filtrada destacada.
- **Filtros de prioridade e de pontos de atenção** não alteram esses denominadores.
- **O recorte filtrado** aparece à parte, com denominador próprio ("x de N municípios do recorte").

## Páginas
| Página | Conteúdo |
|---|---|
| **Visão geral** | Marca e pergunta-guia: "Onde a conectividade precisa de mais atenção?". Quatro indicadores principais (municípios analisados, alta prioridade, métodos divergem e pontos de atenção), mais dois secundários (abaixo do patamar e concordância entre os rankings, com um ícone ⓘ que explica que o valor mede a concordância das listas, não a precisão da previsão). Também traz a distribuição das prioridades, os prioritários por região (em número de municípios ou em % de municípios), uma ressalva curta e links para o mapa e a metodologia |
| **Mapa de prioridades** | O contador mostra os municípios no recorte. Duas visões principais: **Prioridade de acompanhamento**, a padrão, e **Pontos de atenção nos dados**. Em "Opções avançadas", aparece a classificação de cada critério isolado. A dica do mapa mostra primeiro município, densidade, prioridade e ponto de atenção; as posições ficam ao final. Ao clicar num município, abre-se o resumo dele. Abaixo do mapa ficam a comparação territorial (territórios completos) e os estados |
| **Explorar municípios** | Quatro abas: (1) **Ranking e busca**, com busca por nome ou código, ordenação pelo modelo ou pela densidade atual, exportação em CSV e seleção da linha que abre os detalhes; (2) **Detalhes do município**, que mostra densidade, patamar, prioridade e tendência e deixa ordenações e indicadores técnicos numa seção recolhida; (3) **Comparação entre métodos**, com indicadores, dispersão, maiores diferenças e uma leitura técnica; (4) **Municípios divergentes** (classificação divergente entre os métodos) |
| **Sobre o ConectaMap** | (1) O que é; (2) de onde vêm os dados; (3) como as prioridades são definidas; (4) como interpretar; (5) qualidade dos dados; (6) detalhes técnicos e limitações, em seções recolhidas: PR-AUC, Brier, validação temporal, teste reservado, calibração, mudança de distribuição, limitações, fontes e dicionário |

## Filtros
Os filtros globais ficam na barra lateral e valem para todas as páginas:
- região;
- estado (UF), com as opções seguindo a região escolhida;
- prioridade de acompanhamento;
- pontos de atenção nos dados;
- botão **Limpar filtros**.

**Como os filtros se comportam**
- **Os filtros só selecionam municípios.** As posições continuam sendo as nacionais, e os 25% primeiros não são recalculados dentro do recorte.
- **Recorte e Brasil aparecem separados:** com filtros ativos, os cartões mostram o valor do recorte e, abaixo, o total do Brasil.
- **Recorte vazio:** quando nenhum município atende aos filtros, aparece um aviso, sem erro.

## Dados
| Arquivo | Uso |
|---|---|
| `etapa2/saidas/previsao2029/previsoes_2029_municipios.csv` | Base de 2029, com 5.570 municípios e 22 colunas. Antes de usá-la, o app confere a assinatura SHA-256 com a da versão aprovada |
| `etapa2/saidas/previsao2029/dicionario_previsoes_2029.csv` | Significado e unidade de cada coluna |
| `etapa2/saidas/treino/teste_2023_metricas.csv`, `comparacao_modelo_vs_B2_teste.csv` | Resultado do teste reservado, exibido na página de metodologia |
| `dashboard/assets/municipios_2022.geojson`, `ufs_2022.geojson` | Malha municipal e contorno das UFs |

**Regras de leitura dos dados**
- `id_municipio` é lido como texto, com 7 dígitos.
- A ordenação usa sempre `posicao_rf` ou `posicao_b2`, nunca o `escore_rf` arredondado.
- O `escore_rf` aparece só como **valor técnico, não calibrado**: nunca como probabilidade ou percentual.
- A região é derivada do primeiro dígito do código IBGE, e o app confere essa correspondência com a UF ao carregar a base.

## Mapa: fonte e preparação das geometrias
**Fonte:** IBGE, **Malha Municipal 2022**, na versão simplificada distribuída pelo pacote **geobr** (IPEA), release v1.7.0 no GitHub. São 27 GeoPackages, um por UF, em SIRGAS 2000 (EPSG:4674). Os servidores do IBGE não estavam acessíveis no ambiente de preparação; o geobr redistribui a malha oficial com os códigos IBGE de 7 dígitos.

**Preparação:** feita uma única vez por `dashboard/preparar_geometrias.py`.
1. A leitura usa sqlite3 e shapely, sem GDAL nem geopandas.
2. Cada município é simplificado com tolerância de 0,01 grau (cerca de 1 km), preservando a topologia do próprio polígono.
3. As coordenadas são arredondadas em 3 casas decimais.
4. Saem `municipios_2022.geojson` (4,4 MB) e `ufs_2022.geojson`, este com a união dos municípios por UF.
5. Origem, procedimento e SHA-256 ficam registrados em `assets/geometrias_metadados.json`.

**Áreas excluídas:** os códigos 4300001 e 4300002 (Lagoa Mirim e Lagoa dos Patos), que são áreas de água da malha e não municípios.

**Validação**
- **Correspondência com a base:** há exatamente 5.570 geometrias, uma por município, com os mesmos códigos e as mesmas UFs da base.
- **Coordenadas:** todas dentro do território brasileiro.
- **Posicionamento:** testado com oito capitais, conferindo se as coordenadas conhecidas de cada uma caem dentro do polígono do município certo.

**Desempenho:** o arquivo é lido uma vez, com cache. Cada camada do mapa recebe só as geometrias dos seus municípios, cerca de 5 MB para o Brasil inteiro e menos de 1 MB para uma UF. Cada combinação de filtros fica em cache.

**Para refazer a malha** (opcional; precisa de `shapely` e acesso ao GitHub):
```bash
python dashboard/preparar_geometrias.py <pasta_para_os_gpkg>
```

## Estrutura
```
dashboard/
  app.py                  navegação, filtros globais e as quatro páginas
  dados.py                leitura, validação, filtros, buscas e indicadores (funções puras, sem Streamlit)
  graficos.py             gráficos Plotly (paleta única por categoria)
  componentes.py          cabeçalho, cartões e notas
  preparar_geometrias.py  preparação única da malha municipal
  assets/                 malha municipal, contorno das UFs e metadados
  tests/test_dashboard.py testes automatizados
```

## Limitações conhecidas
- **Sem gráfico histórico:** o painel do município não mostra a série histórica de densidade. Exigiria carregar a base temporal da Anatel (185 MB), e isso não é obrigatório para a primeira versão.
- **Malha simplificada:** a simplificação de cerca de 1 km pode criar pequenas frestas entre municípios vizinhos. A posição e a correspondência dos códigos não são afetadas.
- **Formato dos números nas tabelas:** as tabelas interativas usam ponto decimal; os cartões e os textos usam vírgula.
- **Dependência de WebGL:** o mapa precisa de WebGL. Em computadores sem aceleração gráfica, o primeiro desenho do Brasil inteiro pode levar alguns segundos.
