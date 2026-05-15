# Datathon Passos Mágicos - Pipeline de Dados e Risco

Este projeto implementa a primeira parte do case da Passos Mágicos com foco em:

- ingestão e padronização de dados (base antiga + base nova),
- reconciliação por aluno/ano,
- enriquecimento com tabelas relacionais,
- criação da base gold para modelagem de risco de defasagem/piora educacional,
- app Streamlit para consulta do score de risco.

## Objetivo

Construir um fluxo reprodutível `Bronze -> Silver -> Gold` para responder perguntas analíticas e suportar modelagem preditiva do Datathon.

## Estrutura do Projeto

- `src/pipeline/schema_map.py`
  - Normalização das planilhas PEDE (2022-2024).
  - Normalização da base antiga consolidada (2020-2022).
  - Reconciliação das fontes com prioridade para base nova em 2022.
  - Validação com Pydantic.
- `src/pipeline/relational_features.py`
  - Extração de features relacionais por `ra + ano_referencia`:
    - frequência (`TbDiarioFrequencia`),
    - notas/faltas por fase (`TbFaseNotaAluno`),
    - histórico escolar (`TbHistoricoNotas`).
- `src/features/risk_target.py`
  - Construção do target temporal de risco (`target_risco`).
- `notebooks/01_ingestao_padronizacao.ipynb`
  - Notebook principal da etapa 1 (ingestão, reconciliação, enriquecimento, geração silver/gold).
- `app/streamlit_app.py`
  - App para leitura da base gold, score preditivo de risco e fatores principais.
- `data/silver/`
  - Saídas intermediárias validadas.
- `data/gold/`
  - Base final para modelagem.

## Fontes de Dados Utilizadas

- Base nova:
  - `.../DATATHON/BASE DE DADOS PEDE 2024 - DATATHON.xlsx` (abas 2022, 2023, 2024)
- Base antiga consolidada:
  - `.../DATATHON/Bases antigas/PEDE_PASSOS_DATASET_FIAP.csv`
- Tabelas relacionais:
  - `.../Bases antigas/Base de dados - Passos Mágicos/...`
  - principais: `TbAluno`, `TbDiarioFrequencia`, `TbDiarioAula`, `TbFaseNotaAluno`, `TbAlunoTurmaHistorico`, `TbHistoricoNotas`

## Regras de Reconciliação

- Chave principal: `ra + ano_referencia`.
- Em 2022, quando existe conflito entre base antiga e nova:
  - mantém base nova (prioridade maior),
  - mantém base antiga apenas para complementar alunos ausentes na nova.

## Status Atual (Etapa 1)

- Pipeline funcional e validado.
- Silver enriquecido gerado.
- Gold de risco gerado para modelagem supervisionada.

Arquivos de saída principais:

- `data/silver/base_unificada_validada_enriquecida.csv`
- `data/silver/base_unificada_invalidos.csv`
- `data/gold/base_modelagem_risco.csv`

## Resultados EDA e Modelagem

Com a etapa de engenharia concluída, o projeto agora inclui:

- `notebooks/02_eda_storytelling.ipynb`
  - Respostas às **11 perguntas do enunciado oficial** com tabelas e gráficos.
  - Análises em cima dos CSVs existentes (`silver` e `gold`), sem recomeçar pipeline.
- `notebooks/03_modelagem_risco.ipynb`
  - Baseline: `LogisticRegression`.
  - Modelo principal: `RandomForestClassifier`.
  - Métricas: `ROC-AUC`, `Recall`, `F1` e `Matriz de Confusão`.
  - Treino/teste com anos rotulados comparáveis (`2022-2023`).
  - Aplicação do score nos alunos de `2024`, sem tratar 2024 como target real conhecido.
  - Importância de variáveis e geração de score (`data/gold/base_modelagem_risco_scored.csv`).
- `app/streamlit_app.py`
  - App atualizado para carregar a base com score.
  - Exibe score de risco por aluno, classificação pelo threshold e fatores principais (proxy explicativa baseada em importância + desvio da mediana).

### Decisão metodológica da modelagem

O modelo segue o enunciado oficial do Datathon: prever **risco de defasagem/piora educacional** antes de queda de desempenho ou aumento de defasagem.

Importante:

- O score não é uma previsão direta de evasão ou retenção.
- O `target_risco` é um proxy temporal construído a partir de piora futura nos indicadores.
- O ano de `2024` não possui target real conhecido dentro da base, porque não há observação posterior disponível.
- Por isso, `2024` é usado como base de aplicação/scoring, enquanto a avaliação supervisionada usa anos anteriores rotulados.

### Guia rápido de interpretação dos indicadores

- `IAN` (Adequação de Nível):
  - Mede adequação do aluno ao nível esperado.
  - Neste projeto, entra na regra temporal do `target_risco` conforme a lógica operacional definida em `src/features/risk_target.py`.
- `IDA` (Desempenho Acadêmico):
  - Resume desempenho escolar; em geral, valores maiores indicam melhor desempenho.
  - Queda de `IDA` em t+1 entra como critério de risco.
- `IEG` (Engajamento):
  - Participação e envolvimento nas atividades.
  - Normalmente, valores maiores sugerem melhor engajamento.
- `IAA` (Autoavaliação):
  - Percepção do próprio aluno sobre seu desenvolvimento.
- `IPS` (Psicossocial):
  - Sinais comportamentais/emocionais acompanhados pela equipe.
- `IPP` (Psicopedagógico):
  - Avaliação psicopedagógica de aprendizagem e desenvolvimento.
- `IPV` (Ponto de Virada):
  - Indicador de progresso/mudança relevante ao longo do ciclo.
- `INDE` (Índice global):
  - Medida consolidada do desenvolvimento educacional.

Uso prático no projeto:

- `Taxa média de risco (real)`: média de `target_risco` apenas nos anos com rótulo conhecido.
- `Score de risco (previsto)`: probabilidade estimada pelo modelo para cada aluno, incluindo aplicação em 2024.
- `Taxa de risco por grupo`: média do risco por recorte (ano, fase, pedra, turma).

## Como Executar

### 1) Instalar dependências

```powershell
cd C:\Users\00157NLUC-BrenoR\Datathon\datathon_passos
py -3.14 -m pip install --upgrade pip
py -3.14 -m pip install -r requirements.txt
```

### 2) Rodar pipeline no notebook

Abra e execute:

- `notebooks/01_ingestao_padronizacao.ipynb`

### 3) Rodar notebooks de análise e modelagem

Abra e execute em sequência:

- `notebooks/02_eda_storytelling.ipynb`
- `notebooks/03_modelagem_risco.ipynb`

### 4) Rodar app Streamlit

```powershell
cd C:\Users\00157NLUC-BrenoR\Datathon\datathon_passos
streamlit run app\streamlit_app.py
```

## Próximos Passos Recomendados

1. Consolidar narrativa executiva (slides + vídeo) usando as evidências de `02` e `03`.
2. Ajustar threshold de classificação por objetivo operacional (priorizar recall ou precisão).
3. Publicar deploy do Streamlit no Community Cloud com o modelo de risco.

## Notas para Continuidade em Novo Chat

- O projeto está em: `C:\Users\00157NLUC-BrenoR\Datathon\datathon_passos`.
- A etapa 1 (engenharia de dados) está concluída.
- O foco pendente é análise (EDA) e modelagem final.
- Se houver erro de import no Jupyter, reiniciar kernel e reexecutar a célula de imports.
