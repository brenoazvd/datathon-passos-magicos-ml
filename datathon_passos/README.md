# Datathon Passos Magicos - Pipeline de Dados e Risco

Este projeto implementa a primeira parte do case da Passos Magicos com foco em:

- ingestao e padronizacao de dados (base antiga + base nova),
- reconciliacao por aluno/ano,
- enriquecimento com tabelas relacionais,
- criacao da base gold para modelagem de risco de defasagem,
- app Streamlit para inspecao rapida.

## Objetivo

Construir um fluxo reprodutivel `Bronze -> Silver -> Gold` para responder perguntas analiticas e suportar modelagem preditiva do Datathon.

## Estrutura do Projeto

- `src/pipeline/schema_map.py`
  - Normalizacao das planilhas PEDE (2022-2024).
  - Normalizacao da base antiga consolidada (2020-2022).
  - Reconciliacao das fontes com prioridade para base nova em 2022.
  - Validacao com Pydantic.
- `src/pipeline/relational_features.py`
  - Extracao de features relacionais por `ra + ano_referencia`:
    - frequencia (`TbDiarioFrequencia`),
    - notas/faltas por fase (`TbFaseNotaAluno`),
    - historico escolar (`TbHistoricoNotas`).
- `src/features/risk_target.py`
  - Construcao do target temporal de risco (`target_risco`).
- `notebooks/01_ingestao_padronizacao.ipynb`
  - Notebook principal da etapa 1 (ingestao, reconciliacao, enriquecimento, geracao silver/gold).
- `app/streamlit_app.py`
  - App para leitura da base gold, score preditivo de risco e fatores principais.
- `data/silver/`
  - Saidas intermediarias validadas.
- `data/gold/`
  - Base final para modelagem.

## Fontes de Dados Utilizadas

- Base nova:
  - `.../DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx` (abas 2022, 2023, 2024)
- Base antiga consolidada:
  - `.../DATATHON/Bases antigas/PEDE_PASSOS_DATASET_FIAP.csv`
- Tabelas relacionais:
  - `.../Bases antigas/Base de dados - Passos Magicos/...`
  - principais: `TbAluno`, `TbDiarioFrequencia`, `TbDiarioAula`, `TbFaseNotaAluno`, `TbAlunoTurmaHistorico`, `TbHistoricoNotas`

## Regras de Reconciliacao

- Chave principal: `ra + ano_referencia`.
- Em 2022, quando existe conflito entre base antiga e nova:
  - mantem base nova (prioridade maior),
  - mantem base antiga apenas para complementar alunos ausentes na nova.

## Status Atual (Etapa 1)

- Pipeline funcional e validado.
- Silver enriquecido gerado.
- Gold de risco gerado e utilizavel para modelagem baseline.

Arquivos de saida principais:

- `data/silver/base_unificada_validada_enriquecida.csv`
- `data/silver/base_unificada_invalidos.csv`
- `data/gold/base_modelagem_risco.csv`

## Resultados EDA e Modelagem

Com a etapa de engenharia concluida, o projeto agora inclui:

- `notebooks/02_eda_storytelling.ipynb`
  - Respostas as **11 perguntas do enunciado oficial** com tabelas e graficos.
  - Analises em cima dos CSVs existentes (`silver` e `gold`), sem recomecar pipeline.
- `notebooks/03_modelagem_risco.ipynb`
  - Baseline: `LogisticRegression`.
  - Modelo principal: `RandomForestClassifier`.
  - Metricas: `ROC-AUC`, `Recall`, `F1` e `Matriz de Confusao`.
  - Importancia de variaveis e geracao de score (`data/gold/base_modelagem_risco_scored.csv`).
- `app/streamlit_app.py`
  - App atualizado para carregar a base gold pronta.
  - Exibe score de risco por aluno e fatores principais (proxy explicativa baseada em importancia + desvio da mediana).

### Guia rapido de interpretacao dos indicadores

- `IAN` (Adequacao de Nivel):
  - Mede adequacao do aluno ao nivel esperado.
  - Neste projeto, a regra de risco considera **aumento de IAN em t+1** como sinal de piora (`target_risco=1`).
- `IDA` (Desempenho Academico):
  - Resume desempenho escolar; em geral, valores maiores indicam melhor desempenho.
  - Queda de `IDA` em t+1 entra como criterio de risco.
- `IEG` (Engajamento):
  - Participacao e envolvimento nas atividades.
  - Normalmente, valores maiores sugerem melhor engajamento.
- `IAA` (Autoavaliacao):
  - Percepcao do proprio aluno sobre seu desenvolvimento.
- `IPS` (Psicossocial):
  - Sinais comportamentais/emocionais acompanhados pela equipe.
- `IPP` (Psicopedagogico):
  - Avaliacao psicopedagogica de aprendizagem e desenvolvimento.
- `IPV` (Ponto de Virada):
  - Indicador de progresso/mudanca relevante ao longo do ciclo.
- `INDE` (Indice global):
  - Medida consolidada do desenvolvimento educacional.

Uso pratico no projeto:

- `Taxa media de risco (real)`: media de `target_risco`.
- `Score de risco (previsto)`: probabilidade estimada pelo modelo para cada aluno.
- `Taxa de risco por grupo`: media do risco por recorte (ano, fase, pedra, turma).

## Como Executar

### 1) Instalar dependencias

```powershell
cd C:\Users\00157NLUC-BrenoR\Datathon\datathon_passos
py -3.14 -m pip install --upgrade pip
py -3.14 -m pip install -r requirements.txt
```

### 2) Rodar pipeline no notebook

Abra e execute:

- `notebooks/01_ingestao_padronizacao.ipynb`

### 3) Rodar notebooks de analise e modelagem

Abra e execute em sequencia:

- `notebooks/02_eda_storytelling.ipynb`
- `notebooks/03_modelagem_risco.ipynb`

### 4) Rodar app Streamlit

```powershell
cd C:\Users\00157NLUC-BrenoR\Datathon\datathon_passos
streamlit run app\streamlit_app.py
```

## Proximos Passos Recomendados

1. Consolidar narrativa executiva (slides + video) usando as evidencias de `02` e `03`.
2. Ajustar threshold de classificacao por objetivo operacional (priorizar recall ou precisao).
3. Publicar deploy do Streamlit no Community Cloud com o modelo de risco.

## Notas para Continuidade em Novo Chat

- O projeto esta em: `C:\Users\00157NLUC-BrenoR\Datathon\datathon_passos`.
- A etapa 1 (engenharia de dados) esta concluida.
- O foco pendente e analise (EDA) e modelagem final.
- Se houver erro de import no Jupyter, reiniciar kernel e reexecutar a celula de imports.
