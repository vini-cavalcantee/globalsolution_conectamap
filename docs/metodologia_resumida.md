# ConectaMap — Metodologia resumida

Este documento resume o caminho dos dados brutos até a priorização de fevereiro de 2029. Os detalhes completos, com código e resultados, estão nos dois notebooks da raiz do repositório.

## 1. Fontes de dados
| Fonte | Conteúdo | Uso no projeto |
|---|---|---|
| **Anatel** (obtida pela Base dos Dados, tabela `br_anatel_banda_larga_fixa.densidade_municipio`) | Densidade de acessos de banda larga fixa por 100 domicílios, por município e competência, de 2007 a 2026 (trimestral até 2010, mensal a partir de 2011) | Única fonte das variáveis do modelo e do alvo |
| **IBGE — Censo Demográfico 2022** (SIDRA, tabelas 9923, 9515, 6805, 10295 e 10125) | População urbana e rural, idade, saneamento, rendimento e pessoas com deficiência | Tratada no pré-processamento; não entra no modelo principal |
| **IBGE — Malha Municipal 2022** (versão simplificada redistribuída pelo pacote geobr) | Geometrias dos 5.570 municípios | Mapa do dashboard |

## 2. Pré-processamento (Notebook 1)
- **Padronização:** os códigos IBGE de 7 dígitos são tratados como texto, e a Anatel e o SIDRA são convertidos para o mesmo formato.
- **Organização temporal:** a base fica com um registro por município e competência (1.105.770 registros). As 160 competências mensais sem registro no arquivo original são identificadas e **não são preenchidas**.
- **Limite de observação:** fevereiro de 2026. Março de 2026 fica na base só para auditoria e não entra na modelagem.
- **Qualidade:** observações atípicas são sinalizadas por uma regra estatística (regra D, com K=5, A=10, L=12) **sem alterar os valores publicados**. Nenhuma correção foi aprovada. Os sinais são suspeitas estatísticas, não erros comprovados da Anatel.
- **Sinalização causal:** cada sinal usa só dados até a própria competência, para poder entrar nas variáveis do modelo sem olhar o futuro.

## 3. Painel temporal
- **Origens:** fevereiro de cada ano de 2014 a 2023 para treino e avaliação, e fevereiro de 2026 para a previsão.
- **Variáveis:** em cada origem, 30 variáveis calculadas só com meses até a origem: nível atual, defasagens de 12, 24 e 36 meses, variações, médias, medianas, volatilidade, tendência, meses ausentes, histórico disponível e os sinais de qualidade.
- **Fonte das variáveis:** todas vêm da série temporal da Anatel. O Censo foi testado só como análise exploratória e acrescentou praticamente nada.

## 4. Definição do alvo
- **Horizonte:** 36 meses (por exemplo, de fevereiro de 2026 para fevereiro de 2029).
- **Alvo:** o município fica **abaixo do P25 nacional calculado na origem** no mês-alvo.
- **Limiar de 2029:** o P25 de fevereiro de 2026, **8,20 acessos por 100 domicílios**, congelado.
- **Treino:** 10 coortes (2014 a 2023), com 55.686 pares município × origem que têm alvo observado.

## 5. Treinamento
- **Algoritmos comparados:** regressão logística, Random Forest e HistGradientBoosting, com transformações ajustadas só nos dados de treino de cada rodada.
- **Validação temporal:** quatro rodadas (origens avaliadas de 2019 a 2022), com treino em janela expansiva desde 2014.
- **Critério de escolha:** maior PR-AUC média.
- **Modelo escolhido:** Random Forest com `max_features="sqrt"` e `min_samples_leaf=300` (PR-AUC média de 0,572).
- **Protocolo congelado:** o protocolo foi congelado antes do teste (`etapa2/protocolo_treino_congelado.json`), e a origem de 2023 foi avaliada uma única vez.
- **Baselines:**
  - **B0:** taxa base;
  - **B1:** persistência;
  - **B2:** ordenação pela densidade atual;
  - **B3:** tendência.

## 6. Avaliação
- **Validação:** o Random Forest superou B0, B1 e B3 com folga. Sobre B2, a vantagem foi pequena e consistente (PR-AUC média de 0,572 contra 0,537).
- **Teste reservado:** origem 2023, alvo em fevereiro de 2026.

  | Medida | Random Forest | B2 |
  |---|---|---|
  | PR-AUC | 0,521 | 0,516 |
  | Recall no top 25% | 0,826 | 0,812 |

  A diferença de PR-AUC foi de +0,006, com IC 95% de −0,010 a +0,023, ou seja, **não distinguível de zero**.
- **Leitura:** o modelo **não demonstrou vantagem robusta** sobre a simples ordenação pela densidade atual.
- **Calibração:** um critério de divulgação foi fixado antes do cálculo. Nem o Random Forest bruto nem a versão com calibração de Platt o cumpriram nas quatro rodadas. Por isso, **os escores não são divulgados como probabilidades**.

## 7. Previsão final (fevereiro de 2029)
- **Execução:** o modelo do protocolo foi treinado nas coortes de 2014 a 2023 e aplicado aos 5.570 municípios na origem de fevereiro de 2026.
- **Categorias:** cada município recebe uma posição no ranking do Random Forest e outra no ranking pela densidade atual (B2). A prioridade combina os dois:

  | Classificação | Critério | Municípios |
  |---|---|---|
  | Alta prioridade — consenso | Entre os 25% primeiros nos dois rankings | 1.362 |
  | Classificação divergente | Entre os 25% primeiros em só um dos rankings | 62 |
  | Intermediária | Entre 25% e 50% da ordenação em pelo menos um dos rankings | 1.750 |
  | Menor prioridade | Fora da metade superior nos dois rankings | 2.396 |

- **Concordância:** os dois rankings selecionam 97,8% dos mesmos municípios nos 25% primeiros.
- **Indicador de qualidade:** separado da prioridade; aponta 859 municípios com algum ponto de atenção nos dados.
- **Arquivo final:** `etapa2/saidas/previsao2029/previsoes_2029_municipios.csv`, com dicionário na mesma pasta.

## 8. Limitações principais
1. **Natureza do resultado:** é uma **priorização relativa**, não uma previsão exata nem uma garantia. Nenhuma densidade de 2029 é estimada.
2. **Superioridade:** não há superioridade robusta do Random Forest sobre a densidade atual. A lista de consenso não depende da escolha do método.
3. **Escores sem calibração:** `escore_rf` é um valor técnico de ordenação.
4. **Mudança estrutural:** a densidade de 2026 está acima de toda a faixa histórica usada no treino, o que limita a extrapolação.
5. **Poucas coortes independentes:** 10 origens anuais com os mesmos municípios e um único ano de teste.
6. **Pontos de atenção:** as divergências entre os métodos se concentram em municípios com pontos de atenção nos dados, que pedem conferência das séries.
