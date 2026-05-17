# Datathon Passos Mágicos - Dados, EDA, Modelo e App

Este projeto implementa uma solução para o Datathon Passos Mágicos com foco em transformar dados educacionais históricos em inteligência acionável para apoio à decisão.

O objetivo central não é apenas prever risco, mas apoiar uma atuação preventiva: olhar antes, priorizar melhor e agir com mais evidência.

## Objetivo

Construir um fluxo reprodutível de dados e modelagem para:

- consolidar bases históricas da Passos Mágicos;
- padronizar informações de alunos, anos, fases e indicadores;
- responder às perguntas analíticas do enunciado;
- criar um modelo de risco de defasagem/piora educacional;
- disponibilizar o score em um app Streamlit simples para consulta.

## Fluxo do projeto

O pipeline foi organizado em camadas:

```text
Bronze -> Silver -> Gold -> Modelagem -> App Streamlit -> Apresentação
```

Arquivos principais:

- `notebooks/01_ingestao_padronizacao.ipynb`
  - Ingestão, padronização, reconciliação de bases e geração das saídas Silver/Gold.
- `notebooks/02_eda_storytelling.ipynb`
  - Análise exploratória com storytelling executivo e resposta às 11 perguntas do enunciado.
- `notebooks/03_modelagem_risco.ipynb`
  - Baseline, modelo principal, métricas, threshold e aplicação do score em 2024.
- `app/streamlit_app.py`
  - App para consulta do score de risco por aluno.
- `apresentacao/datathon_passos_apresentacao.pptx`
  - Apresentação final do projeto.

## Estrutura

```text
datathon_passos/
├── app/
│   └── streamlit_app.py
├── apresentacao/
│   ├── datathon_passos_apresentacao.pptx
│   ├── roteiro_apresentacao.md
│   └── preview/
├── data/
│   ├── silver/
│   └── gold/
├── notebooks/
│   ├── 01_ingestao_padronizacao.ipynb
│   ├── 02_eda_storytelling.ipynb
│   └── 03_modelagem_risco.ipynb
├── src/
│   ├── features/
│   │   └── risk_target.py
│   └── pipeline/
│       ├── relational_features.py
│       └── schema_map.py
├── requirements.txt
└── README.md
```

## Bases geradas

Principais saídas do pipeline:

- `data/silver/base_unificada_validada_enriquecida.csv`
- `data/silver/base_unificada_invalidos.csv`
- `data/gold/base_modelagem_risco.csv`
- `data/gold/base_modelagem_risco_scored.csv`

A base Silver concentra os dados tratados e padronizados. A base Gold é usada para análise, modelagem e geração do score.

## EDA e storytelling

O notebook `02_eda_storytelling.ipynb` organiza a análise em uma narrativa de negócio:

```text
problema -> evidência -> interpretação -> decisão prática
```

Ele responde às 11 perguntas do enunciado e inclui:

- visão geral da base;
- análise dos principais indicadores educacionais;
- comparações por ano, fase, turma e grupos disponíveis;
- diagnóstico de qualidade da variável `fase`;
- síntese executiva dos achados;
- recomendações de negócio;
- ponte para a modelagem de risco.

### Decisão metodológica sobre fase

A coluna original `fase` apresentou formatos diferentes ao longo dos anos:

- em 2022, valores numéricos como `1`, `1.0`, `2`;
- em 2023, categorias textuais como `FASE 1`, `FASE 2`, `ALFA`;
- em 2024, muitos valores representavam turmas, como `1A`, `4M`, `7E`.

Por isso, a análise passou a usar uma fase padronizada, reduzindo fragmentação e evitando comparar turma como se fosse fase.

Essa correção foi tratada como decisão de qualidade de dados, não como ajuste visual.

## Modelagem de risco

O notebook `03_modelagem_risco.ipynb` constrói um modelo para prever risco de defasagem ou piora educacional.

Modelos utilizados:

- baseline: `LogisticRegression`;
- modelo principal: `RandomForestClassifier`.

Métricas avaliadas:

- acurácia;
- ROC-AUC;
- PR-AUC;
- precision;
- recall;
- F1-score;
- matriz de confusão;
- análise de threshold.

### Uso de 2024

O ano de 2024 é usado para aplicação do score, não para avaliação supervisionada.

Motivo: dentro da base disponível, 2024 ainda não possui um ano posterior para confirmar o desfecho real de piora/defasagem. Por isso:

- 2022 e 2023 são usados para treino/teste com target conhecido;
- 2024 recebe score para apoiar priorização de acompanhamento.

## App Streamlit

O app permite consultar:

- score de risco por aluno;
- classificação operacional;
- fatores que mais chamaram atenção;
- listas de prioridade para acompanhamento.

App publicado:

[https://datathon-paapps-magicos-ml-jcpp4d6sagrv8qxwfscu73.streamlit.app/](https://datathon-paapps-magicos-ml-jcpp4d6sagrv8qxwfscu73.streamlit.app/)

Rodar localmente:

```powershell
cd C:\Users\00157NLUC-BrenoR\Datathon
.\.venv\Scripts\Activate.ps1
streamlit run .\datathon_passos\app\streamlit_app.py
```

## Apresentação final

Arquivos da apresentação:

- `apresentacao/datathon_passos_apresentacao.pptx`
- `apresentacao/roteiro_apresentacao.md`
- `apresentacao/preview/montage.png`

A apresentação foi montada para explicar o projeto em linguagem simples:

1. problema;
2. organização dos dados;
3. principais achados da EDA;
4. modelagem de risco;
5. aplicação do score em 2024;
6. uso do app;
7. valor prático para a Passos Mágicos.

## Como executar os notebooks

Instalar dependências:

```powershell
cd C:\Users\00157NLUC-BrenoR\Datathon\datathon_passos
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Executar em sequência:

1. `notebooks/01_ingestao_padronizacao.ipynb`
2. `notebooks/02_eda_storytelling.ipynb`
3. `notebooks/03_modelagem_risco.ipynb`

Se as bases Silver/Gold já existirem, não é necessário reprocessar toda a ingestão para usar o app ou revisar os resultados.

## Interpretação do score

O score é uma ferramenta de triagem.

Ele não substitui a avaliação pedagógica, psicossocial ou institucional da equipe. A leitura correta é:

- score alto: aluno merece atenção mais cedo;
- score intermediário: acompanhar evolução;
- score baixo: manter monitoramento regular.

O modelo ajuda a organizar prioridades, não a tomar decisão automática.

## Fechamento

O principal valor do projeto está em transformar dados educacionais dispersos em uma ferramenta de cuidado preventivo.

A Passos Mágicos pode usar essa estrutura para olhar antes, priorizar melhor e agir com mais evidência.
