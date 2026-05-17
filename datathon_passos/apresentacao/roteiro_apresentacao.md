# Roteiro da apresentação

Tempo alvo: 4 a 5 minutos.

Use como guia, não como texto decorado. A ideia é soar natural: explicar o problema, mostrar o que foi feito e fechar com o valor prático para a Passos Mágicos.

## Slide 1 — Abertura

Aqui eu começo apresentando o projeto: a ideia não é criar um modelo para substituir a equipe, mas organizar os dados para ajudar a Passos Mágicos a olhar antes, priorizar melhor e agir com mais evidência.

## Slide 2 — Problema

O ponto central do trabalho é identificar sinais de risco antes da piora ficar muito clara. No Datathon, esse risco foi tratado como defasagem ou piora educacional, então a pergunta prática foi: quem precisa de atenção primeiro?

## Slide 3 — Base confiável

Antes de analisar ou modelar, a base precisou ser organizada. O projeto seguiu o fluxo Bronze, Silver e Gold para não trabalhar com planilhas soltas. Isso dá mais segurança para a EDA, para o modelo e para o app.

## Slide 4 — Cobertura dos dados

Aqui é importante explicar que nem todo ano tem o mesmo papel. Os anos de 2022 e 2023 foram usados para treino e teste porque têm desfecho conhecido. Já 2024 foi usado para aplicar o score nos alunos atuais, ou seja, para priorização.

## Slide 5 — Ajuste da variável fase

Esse foi um cuidado importante de qualidade de dados. A coluna de fase vinha com formatos diferentes: número, texto e até turma misturada. Se eu usasse isso direto, a análise ficaria enganosa. Por isso, a fase foi padronizada e a turma foi separada quando aparecia.

## Slide 6 — Principais sinais da EDA

A análise mostrou que o risco não depende de um indicador só. Ele aparece como combinação de desempenho, engajamento, defasagem e contexto. A EDA ajudou a entender os sinais, mas o modelo entra para priorizar aluno por aluno.

## Slide 7 — Modelo

O modelo principal foi Random Forest, com baseline de regressão logística. Eu não olhei só acurácia, porque em prevenção isso pode enganar. O mais importante foi avaliar recall e F1 da classe de risco, para reduzir a chance de deixar aluno em risco passar sem atenção.

## Slide 8 — Ponto de corte

O threshold escolhido foi 0,40 porque aumenta o recall. Isso sinaliza mais alunos e dá mais trabalho para a equipe, mas combina melhor com a ideia de prevenção: primeiro sinaliza, depois a equipe avalia.

## Slide 9 — Aplicação em 2024

Depois de validar o modelo, ele foi aplicado nos alunos de 2024. O resultado não diz que o aluno necessariamente vai ter problema. Ele mostra quem tem mais sinais de atenção e ajuda a equipe a começar por uma lista mais organizada.

## Slide 10 — App Streamlit

O app foi feito para traduzir o modelo em algo mais simples de usar: escolher ano e aluno, ver o score, a situação e os fatores que mais chamaram atenção. A proposta é apoiar a conversa da equipe, não dar uma decisão automática.

## Slide 11 — Fechamento

O valor do projeto não está só em prever risco. Está em transformar dados educacionais dispersos em uma ferramenta de cuidado preventivo. O modelo ajuda a Passos Mágicos a olhar antes, priorizar melhor e agir com mais evidência.
