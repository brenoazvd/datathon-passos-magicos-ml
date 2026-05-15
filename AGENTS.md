# Skill — Datathon Passos Mágicos ML

## 1. Objetivo do projeto

Este repositório contém a solução do Datathon Passos Mágicos, com foco em transformar dados históricos educacionais em inteligência acionável para apoio à decisão.

A entrega deve conectar:

- análise exploratória e storytelling executivo;
- modelagem de risco educacional;
- aplicação Streamlit para consulta de score e fatores de risco;
- apresentação gerencial e vídeo final.

O objetivo central não é apenas prever risco, mas apoiar uma atuação preventiva da Passos Mágicos.

Frase-guia:

> O principal valor do projeto não está apenas em prever risco, mas em transformar dados educacionais dispersos em uma ferramenta de cuidado preventivo. O modelo ajuda a Passos Mágicos a olhar antes, priorizar melhor e agir com mais evidência.

## 2. Estrutura esperada do projeto

Respeitar a organização atual do repositório. O projeto principal fica em `datathon_passos/`.

Fluxo geral:

- `datathon_passos/data/raw/` ou equivalente: dados originais, quando existir;
- `datathon_passos/data/bronze/`: dados brutos organizados, quando existir;
- `datathon_passos/data/silver/`: dados tratados e padronizados;
- `datathon_passos/data/gold/`: bases analíticas finais;
- `datathon_passos/notebooks/01_ingestao_padronizacao.ipynb`: ingestão, padronização e geração Silver/Gold;
- `datathon_passos/notebooks/02_eda_storytelling.ipynb`: EDA e storytelling executivo;
- `datathon_passos/notebooks/03_modelagem_risco.ipynb`: modelagem de risco;
- `datathon_passos/src/features/risk_target.py`: lógica de target de risco;
- `datathon_passos/app/streamlit_app.py`: aplicação final para consumo do modelo.

Não refazer a ingestão se as bases Silver/Gold já existirem e estiverem atualizadas.

## 3. Regra geral de atuação

Sempre trabalhar com foco em avaliação de Datathon:

- clareza;
- reprodutibilidade;
- justificativa metodológica;
- narrativa executiva;
- ausência de vazamento de dados;
- conexão entre EDA, modelo, app e recomendações.

Não fazer mudanças cosméticas que escondam problemas reais dos dados.

Não inventar conclusões sem evidência.

Toda conclusão executiva deve estar conectada a:

- tabela;
- métrica;
- gráfico;
- diagnóstico de qualidade;
- ou regra metodológica documentada.

## 4. Notebook 02 — EDA e Storytelling Executivo

O notebook `datathon_passos/notebooks/02_eda_storytelling.ipynb` já foi estruturado para responder às 11 perguntas do enunciado em formato executivo.

Cada pergunta deve manter a estrutura:

- pergunta de negócio;
- evidência analisada;
- tabela/gráfico;
- leitura dos dados;
- implicação prática.

Não transformar as perguntas em checklist seco. O notebook deve contar uma história:

problema -> evidência -> interpretação -> decisão prática.

As seções finais obrigatórias são:

- `## Síntese executiva dos achados`
- `## Recomendações de negócio`
- `## Ponte para a modelagem de risco`

A conclusão da EDA deve conectar naturalmente com o notebook `datathon_passos/notebooks/03_modelagem_risco.ipynb`.

## 5. Decisão metodológica importante — variável `fase`

A variável original `fase` não deve ser usada diretamente como dimensão analítica principal sem tratamento.

Diagnóstico encontrado:

- a coluna original era `fase`;
- em 2022, vinha como valores numéricos: `0`, `1`, `1.0`, `2`, `2.0` etc.;
- em 2023, vinha como categorias textuais: `FASE 1`, `FASE 2`, `FASE 3`, `ALFA` etc.;
- em 2024, muitos valores eram turmas, não fases: `1A`, `1B`, `4M`, `7E`, `8E` etc.

Problema:

A coluna misturava fase, turma e mudança de nomenclatura por ano. Isso gerava fragmentação artificial e muitos `NaN` em análises do tipo fase x ano.

Números validados:

- `fase` original: 96 categorias únicas;
- após padronização: 11 categorias analíticas;
- registros por ano:
- 2022: 1349;
- 2023: 1014;
- 2024: 1156;
- percentual de `fase` nula:
- 2022: 12.9%;
- 2023: 0.0%;
- 2024: 0.0%;
- percentual de `IDA` nulo:
- 2022: 12.90%;
- 2023: 7.59%;
- 2024: 8.74%.

Colunas criadas/esperadas:

- `fase_original`;
- `fase_norm`;
- `fase_analitica`;
- `turma_extraida`;
- `tipo_categoria_fase`.

Regras aplicadas:

- `1`, `1.0`, `FASE 1`, `1A` -> `FASE 1`;
- `1A`, `1B`, `8E` preservam a turma em `turma_extraida`;
- `ALFA` continua como `ALFA`;
- casos inseguros -> `OUTROS/NAO_CLASSIFICADO`.

Regra obrigatória:

Quando a análise envolver fase, usar `fase_analitica`, não `fase` original.

Não preencher `NaN` artificialmente.

Os `NaN` restantes devem ser interpretados como:

- ausência real de observação válida;
- ausência de IDA;
- categoria inexistente naquele ano;
- ou limitação de cobertura.

A correção da fase deve ser apresentada como decisão metodológica de qualidade de dados, não como maquiagem visual.

## 6. Notebook 03 — Modelagem de Risco

O notebook `datathon_passos/notebooks/03_modelagem_risco.ipynb` deve ser continuação lógica do storytelling.

Objetivo:

Transformar os sinais identificados na EDA em um modelo de risco para apoiar priorização preventiva.

A modelagem deve deixar explícito:

1. Problema de negócio.
2. Construção do target.
3. Distribuição das classes.
4. Features usadas.
5. Tratamento de categóricas e numéricas.
6. Split treino/teste.
7. Baseline.
8. Modelo principal.
9. Métricas.
10. Threshold.
11. Interpretação dos fatores.
12. Saída para o app.

Regras:

- não alterar `datathon_passos/src/features/risk_target.py` sem necessidade real;
- não mudar target sem justificar tecnicamente;
- não usar variáveis com vazamento de dados;
- não usar informação futura indevida;
- não otimizar apenas por acurácia;
- priorizar recall/F1 da classe de risco;
- interpretar falsos negativos como alunos em risco que o modelo não sinalizou;
- interpretar falsos positivos como alunos sinalizados que demandam avaliação da equipe;
- justificar threshold com critério de negócio.

Como o objetivo é prevenção, o threshold pode privilegiar recall da classe de risco, desde que isso seja explicado.

## 7. Métricas esperadas na modelagem

Sempre que possível, apresentar:

- matriz de confusão;
- classification report;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- PR-AUC, especialmente se houver desbalanceamento;
- análise de threshold.

A análise de threshold deve comparar, por exemplo:

- 0.3;
- 0.4;
- 0.5;
- 0.6;
- 0.7.

Para cada threshold, mostrar:

- precision da classe de risco;
- recall da classe de risco;
- F1 da classe de risco;
- quantidade de alunos sinalizados.

## 8. Interpretação do modelo

Se o modelo principal for Random Forest, apresentar feature importance.

Regras de interpretação:

- não afirmar causalidade;
- dizer "fatores associados ao risco" ou "variáveis relevantes para o modelo";
- explicar que o modelo apoia triagem e priorização, não decisão automática;
- conectar os fatores do modelo com os achados da EDA.

## 9. Aplicação Streamlit

A aplicação final deve mostrar:

- score/probabilidade de risco;
- classificação final conforme threshold escolhido;
- principais fatores associados;
- leitura executiva simples para apoiar a equipe.

O app deve ser apresentado como ferramenta de apoio à decisão, não como diagnóstico definitivo.

## 10. README e entrega final

O README deve deixar claro:

- objetivo do projeto;
- estrutura do repositório;
- como executar;
- requisitos;
- notebooks principais;
- app Streamlit;
- principais decisões metodológicas;
- link do deploy, quando disponível;
- limitações conhecidas.

Artefatos finais esperados:

- GitHub organizado;
- notebook de EDA/storytelling;
- notebook de modelagem;
- app Streamlit;
- apresentação gerencial;
- vídeo de até 5 minutos.

## 11. Regras de qualidade antes de qualquer commit

Antes de finalizar alterações:

- executar notebooks afetados via `nbconvert`;
- garantir que não há erro de execução;
- verificar encoding quebrado, como `MÃ`, `Ã§`, `?` indevido;
- garantir que gráficos têm título claro;
- garantir que tabelas têm interpretação;
- garantir que não há arquivos desnecessários no commit;
- não alterar modelagem/target quando a tarefa for apenas EDA;
- não alterar EDA quando a tarefa for apenas app ou documentação, salvo necessidade justificada.

O warning `MissingIDFieldWarning` do nbconvert não bloqueia a entrega se o notebook executa corretamente. Ele é um aviso estrutural do formato `.ipynb`, não erro de análise.

## 12. Tom da documentação e narrativa

Usar tom:

- executivo;
- claro;
- técnico quando necessário;
- orientado à decisão;
- sem exageros.

Evitar:

- linguagem genérica;
- conclusões não sustentadas;
- "modelo resolve o problema";
- esconder limitações;
- excesso de jargão.

Mensagem central do projeto:

A Passos Mágicos possui dados ricos sobre a trajetória dos alunos. Quando esses dados são organizados e analisados em conjunto, eles permitem identificar padrões de vulnerabilidade e apoiar uma atuação mais preventiva.
