from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    "ano_referencia",
    "idade",
    "ano_ingresso",
    "inde",
    "ian",
    "ida",
    "ieg",
    "iaa",
    "ips",
    "ipp",
    "ipv",
    "defasagem",
    "matematica",
    "portugues",
]
TARGET_COL = "target_risco"
TRAIN_YEARS = [2022, 2023]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "base_modelagem_risco.csv"
DEFAULT_SCORED_PATH = PROJECT_ROOT / "data" / "gold" / "base_modelagem_risco_scored.csv"
DEFAULT_DATA_PATH = DEFAULT_SCORED_PATH if DEFAULT_SCORED_PATH.exists() else DEFAULT_GOLD_PATH


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_resource
def train_risk_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    if TARGET_COL not in df.columns:
        raise ValueError("A base precisa ter target_risco para treinar o modelo explicativo.")

    train_df = df[df[TARGET_COL].notna()].copy()
    if "ano_referencia" in train_df.columns:
        filtered = train_df[train_df["ano_referencia"].isin(TRAIN_YEARS)].copy()
        if filtered[TARGET_COL].nunique(dropna=True) >= 2:
            train_df = filtered

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = train_df[feature_cols].copy()
    y = train_df[TARGET_COL].astype(int)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X, y)

    importances = model.named_steps["model"].feature_importances_
    stats = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
            "median": [X[c].median() for c in feature_cols],
            "std": [X[c].std() if pd.notna(X[c].std()) and X[c].std() != 0 else 1.0 for c in feature_cols],
            "corr_target": [X[c].corr(y) if pd.notna(X[c].corr(y)) else 0.0 for c in feature_cols],
        }
    ).sort_values("importance", ascending=False)

    return model, stats


def explain_main_factors(row: pd.Series, feature_stats: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if feature_stats.empty:
        return pd.DataFrame()

    hidden_features = {"ano_referencia", "ano_ingresso"}
    rows = []
    for _, stat in feature_stats.iterrows():
        feature = stat["feature"]
        if feature in hidden_features:
            continue

        median = float(stat["median"])
        std = float(stat["std"])
        importance = float(stat["importance"])

        value = row.get(feature, np.nan)
        if pd.isna(value):
            value = median

        z_score = abs((float(value) - median) / std)
        contribution = z_score * importance

        rows.append(
            {
                "feature": feature,
                "valor_aluno": float(value),
                "mediana_base": median,
                "desvio": float(value) - median,
                "std": std,
                "importancia_global": importance,
                "contribuicao_proxy": contribution,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("contribuicao_proxy", ascending=False).head(top_n)
    return out


FEATURE_LABELS = {
    "ano_referencia": "Ano de referência",
    "idade": "Idade",
    "ano_ingresso": "Ano de ingresso",
    "inde": "Índice geral do aluno",
    "ian": "Adequação ao nível",
    "ida": "Desempenho acadêmico",
    "ieg": "Engajamento",
    "iaa": "Autoavaliação",
    "ips": "Aspectos psicossociais",
    "ipp": "Aspectos psicopedagógicos",
    "ipv": "Ponto de virada",
    "defasagem": "Defasagem",
    "matematica": "Matemática",
    "portugues": "Português",
}

HIGHER_IS_BETTER = {"inde", "ian", "ida", "ieg", "iaa", "ips", "ipp", "ipv", "matematica", "portugues"}
HIGHER_NEEDS_ATTENTION = {"defasagem", "idade"}

FACTOR_HINTS = {
    "inde": "Resumo geral do desenvolvimento do aluno.",
    "ian": "Mostra se o aluno está adequado ao nível esperado.",
    "ida": "Resume sinais de desempenho acadêmico.",
    "ieg": "Mostra participação e engajamento nas atividades.",
    "iaa": "Ajuda a entender como o aluno percebe o próprio desenvolvimento.",
    "ips": "Traz sinais psicossociais acompanhados pela equipe.",
    "ipp": "Traz sinais psicopedagógicos de aprendizagem e desenvolvimento.",
    "ipv": "Indica sinais ligados ao ponto de virada do aluno.",
    "defasagem": "Ajuda a observar distância em relação ao nível esperado.",
    "idade": "Pode indicar trajetória escolar mais longa ou diferente do grupo.",
    "matematica": "Mostra o desempenho em matemática.",
    "portugues": "Mostra o desempenho em português.",
}


def risk_status(score: float, threshold: float) -> str:
    if score >= max(0.70, threshold + 0.20):
        return "Prioridade alta"
    if score >= threshold:
        return "Atenção"
    return "Monitorar"


def recommended_action(status: str) -> str:
    if status == "Prioridade alta":
        return "Avaliar primeiro e combinar leitura pedagógica, psicossocial e psicopedagógica."
    if status == "Atenção":
        return "Acompanhar de perto e verificar sinais recentes de queda ou dificuldade."
    return "Manter acompanhamento de rotina."


def format_number(value: float) -> str:
    return f"{value:.1f}"


def interpret_factor(feature: str, value: float, reference: float, std: float) -> tuple[str, str, str]:
    diff = value - reference
    tolerance = max(abs(std) * 0.15, 0.2)

    if abs(diff) <= tolerance:
        return (
            "Dentro do esperado",
            "neutral",
            "O valor está próximo da referência dos alunos usados no modelo.",
        )

    if feature in HIGHER_IS_BETTER:
        if diff < 0:
            return (
                "Pede atenção",
                "attention",
                "O valor está abaixo da referência. Vale verificar se houve queda recente ou dificuldade específica.",
            )
        return (
            "Sinal favorável",
            "positive",
            "O valor está acima da referência. Este indicador, isoladamente, não é o ponto de maior preocupação.",
        )

    if feature in HIGHER_NEEDS_ATTENTION:
        if diff > 0:
            return (
                "Pede atenção",
                "attention",
                "O valor está acima da referência. Vale entender se isso reflete defasagem ou trajetória diferente do grupo.",
            )
        return (
            "Sinal favorável",
            "positive",
            "O valor está abaixo da referência. Este indicador, isoladamente, tende a ser menos preocupante.",
        )

    return (
        "Verificar com a equipe",
        "neutral",
        "O valor está diferente da referência e precisa ser lido junto com o contexto do aluno.",
    )


def friendly_factor_table(factors: pd.DataFrame) -> pd.DataFrame:
    if factors.empty:
        return factors

    out = factors.copy()
    out["Fator"] = out["feature"].map(FEATURE_LABELS).fillna(out["feature"])
    interpreted = out.apply(
        lambda r: interpret_factor(r["feature"], r["valor_aluno"], r["mediana_base"], r["std"]),
        axis=1,
    )
    out["Como interpretar"] = [item[0] for item in interpreted]
    out["Classe visual"] = [item[1] for item in interpreted]
    out["Mensagem"] = [item[2] for item in interpreted]
    out["Valor do aluno"] = out["valor_aluno"].map(format_number)
    out["Referência da base"] = out["mediana_base"].map(format_number)
    out["Diferença"] = out["desvio"].map(lambda value: f"{value:+.1f}")
    out["O que é"] = out["feature"].map(FACTOR_HINTS).fillna("Indicador usado no modelo.")
    out["Prioridade interna"] = out["contribuicao_proxy"]
    return out[
        [
            "Fator",
            "Valor do aluno",
            "Referência da base",
            "Diferença",
            "Como interpretar",
            "Classe visual",
            "Mensagem",
            "O que é",
            "Prioridade interna",
        ]
    ]


def render_factor_cards(factors: pd.DataFrame, max_cards: int = 6) -> None:
    friendly = friendly_factor_table(factors)
    if friendly.empty:
        st.info("Não foi possível calcular os fatores de atenção para a base carregada.")
        return

    order = {"attention": 0, "neutral": 1, "positive": 2}
    friendly["_ordem"] = friendly["Classe visual"].map(order).fillna(3)
    friendly = friendly.sort_values(["_ordem", "Prioridade interna"], ascending=[True, False])

    attention_count = int((friendly["Classe visual"] == "attention").sum())
    if attention_count:
        st.markdown(
            f"""
            <div class="factor-summary attention-summary">
                <strong>{attention_count} ponto(s) pedem atenção.</strong>
                Comece por eles antes de olhar os sinais favoráveis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="factor-summary positive-summary">
                <strong>Nenhum fator isolado apareceu claramente em alerta.</strong>
                Neste caso, o risco estimado pode vir da combinação dos indicadores, não de um único ponto óbvio.
            </div>
            """,
            unsafe_allow_html=True,
        )

    friendly = friendly.head(max_cards)
    cols = st.columns(2)
    for idx, item in friendly.iterrows():
        css_class = escape(str(item["Classe visual"]))
        badge = escape(str(item["Como interpretar"]))
        factor = escape(str(item["Fator"]))
        value = escape(str(item["Valor do aluno"]))
        reference = escape(str(item["Referência da base"]))
        diff = escape(str(item["Diferença"]))
        message = escape(str(item["Mensagem"]))
        hint = escape(str(item["O que é"]))

        card_html = f"""
        <div class="factor-card {css_class}">
            <div class="factor-topline">
                <div class="factor-name">{factor}</div>
                <div class="factor-badge">{badge}</div>
            </div>
            <div class="factor-hint">{hint}</div>
            <div class="factor-numbers">
                <div>
                    <span>Aluno</span>
                    <strong>{value}</strong>
                </div>
                <div>
                    <span>Referência</span>
                    <strong>{reference}</strong>
                </div>
                <div>
                    <span>Diferença</span>
                    <strong>{diff}</strong>
                </div>
            </div>
            <div class="factor-message">{message}</div>
        </div>
        """
        cols[idx % 2].markdown(card_html, unsafe_allow_html=True)


st.set_page_config(page_title="Datathon Passos Mágicos - Risco", layout="wide")
st.markdown(
    """
    <style>
    .factor-card {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 18px;
        padding: 18px 18px 16px;
        margin: 0 0 16px 0;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02)),
            rgba(20, 24, 31, 0.96);
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
    }
    .factor-card.attention {
        border-left: 7px solid #f59e0b;
    }
    .factor-card.positive {
        border-left: 7px solid #22c55e;
    }
    .factor-card.neutral {
        border-left: 7px solid #60a5fa;
    }
    .factor-topline {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
        margin-bottom: 8px;
    }
    .factor-name {
        font-size: 1.08rem;
        font-weight: 800;
        letter-spacing: .01em;
        color: #f8fafc;
    }
    .factor-badge {
        border-radius: 999px;
        padding: 5px 11px;
        font-size: .78rem;
        font-weight: 800;
        white-space: nowrap;
        color: #0f172a;
        background: #e2e8f0;
    }
    .attention .factor-badge {
        background: #fbbf24;
    }
    .positive .factor-badge {
        background: #86efac;
    }
    .neutral .factor-badge {
        background: #93c5fd;
    }
    .factor-hint {
        color: #cbd5e1;
        font-size: .92rem;
        line-height: 1.35;
        min-height: 40px;
    }
    .factor-numbers {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 14px 0;
    }
    .factor-numbers div {
        border-radius: 12px;
        padding: 10px 11px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.16);
    }
    .factor-numbers span {
        display: block;
        color: #94a3b8;
        font-size: .76rem;
        margin-bottom: 4px;
    }
    .factor-numbers strong {
        color: #ffffff;
        font-size: 1.02rem;
    }
    .factor-message {
        color: #e5e7eb;
        line-height: 1.42;
        font-size: .94rem;
        border-top: 1px solid rgba(148, 163, 184, 0.14);
        padding-top: 12px;
    }
    .factor-summary {
        border-radius: 16px;
        padding: 14px 16px;
        margin: 8px 0 16px 0;
        line-height: 1.45;
        border: 1px solid rgba(255,255,255,.10);
    }
    .factor-summary strong {
        display: block;
        margin-bottom: 4px;
        color: #ffffff;
    }
    .attention-summary {
        background: linear-gradient(135deg, rgba(245,158,11,.20), rgba(15,23,42,.80));
        border-left: 7px solid #f59e0b;
        color: #fde68a;
    }
    .positive-summary {
        background: linear-gradient(135deg, rgba(34,197,94,.18), rgba(15,23,42,.80));
        border-left: 7px solid #22c55e;
        color: #bbf7d0;
    }
    @media (max-width: 760px) {
        .factor-topline {
            align-items: flex-start;
            flex-direction: column;
        }
        .factor-numbers {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Painel de Prioridade de Acompanhamento")
st.caption(
    "O painel ajuda a identificar quais alunos podem precisar de atenção antes de uma piora educacional."
)

with st.sidebar:
    st.header("Configurações")
    data_path_input = st.text_input("Base de dados", value=str(DEFAULT_DATA_PATH))
    st.markdown(
        "O risco exibido é uma estimativa para apoiar triagem. "
        "A decisão final deve sempre considerar a avaliação da equipe."
    )

data_path = Path(data_path_input)

if not data_path.exists():
    st.error(f"Arquivo não encontrado: {data_path}")
    st.stop()

df_view = load_dataset(str(data_path))

missing_cols = [c for c in ["ra", "ano_referencia"] if c not in df_view.columns]
if missing_cols:
    st.error(f"Colunas obrigatórias ausentes na base: {missing_cols}")
    st.stop()

if TARGET_COL not in df_view.columns and "score_risco" not in df_view.columns:
    st.error("A base precisa ter score_risco pronto ou target_risco para treinar o modelo localmente.")
    st.stop()

model, feature_stats = train_risk_model(df_view) if TARGET_COL in df_view.columns else (None, pd.DataFrame())

if "score_risco" not in df_view.columns:
    X_data = df_view[[c for c in FEATURE_COLS if c in df_view.columns]].copy()
    df_view["score_risco"] = model.predict_proba(X_data)[:, 1]

df_view["score_risco_pct"] = df_view["score_risco"] * 100

threshold = 0.4
if "threshold_modelo" in df_view.columns and df_view["threshold_modelo"].notna().any():
    threshold = float(df_view["threshold_modelo"].dropna().iloc[0])
if "pred_risco_threshold" not in df_view.columns:
    df_view["pred_risco_threshold"] = (df_view["score_risco"] >= threshold).astype(int)

df_view["situacao"] = df_view["score_risco"].apply(lambda value: risk_status(float(value), threshold))
df_view["acao_sugerida"] = df_view["situacao"].apply(recommended_action)

st.info(
    "Como ler: quanto maior o percentual, maior a prioridade para acompanhamento. "
    "Para 2024, ainda não existe resultado futuro observado; por isso o painel mostra uma estimativa preventiva."
)

st.subheader("Escolha o ano e o aluno")
anos = sorted(df_view["ano_referencia"].dropna().astype(int).unique().tolist())
selected_ano = st.selectbox("Ano de referência", anos, index=len(anos) - 1)

df_year = df_view[df_view["ano_referencia"] == selected_ano].copy()
df_year["aluno_label"] = df_year["ra"].astype(str) + " | " + df_year["nome"].fillna("sem nome").astype(str)

c1, c2, c3, c4 = st.columns(4)
sinalizados = int(df_year["pred_risco_threshold"].sum())
pct_sinalizados = sinalizados / len(df_year) * 100 if len(df_year) else 0
tem_target_real = TARGET_COL in df_year.columns and df_year[TARGET_COL].notna().any()

c1.metric("Alunos no ano", int(df_year["ra"].nunique()))
c2.metric("Sinalizados para atenção", f"{sinalizados} ({pct_sinalizados:.1f}%)")
c3.metric("Risco médio estimado", f"{df_year['score_risco_pct'].mean():.1f}%")
c4.metric("Resultado observado", "Disponível" if tem_target_real else "Ainda não")

selected_label = st.selectbox("Aluno", sorted(df_year["aluno_label"].unique().tolist()))
row = df_year[df_year["aluno_label"] == selected_label].iloc[0]

st.markdown("### Leitura do aluno selecionado")
target_value = row.get(TARGET_COL, np.nan)
status = risk_status(float(row["score_risco"]), threshold)

r1, r2, r3 = st.columns([1, 1, 2])
r1.metric("Risco estimado", f"{row['score_risco_pct']:.1f}%")
r2.metric("Situação", status)
r3.write("**Ação sugerida**")
r3.write(recommended_action(status))

st.progress(float(row["score_risco"]))
if pd.isna(target_value):
    st.caption("Este aluno está em um ano de aplicação do modelo. Ainda não há resultado futuro observado para comparar.")
else:
    st.caption("Este aluno pertence a um ano histórico usado para validar o modelo.")

main_factors = explain_main_factors(row, feature_stats, top_n=12)
st.markdown("### O que mais chamou atenção")
st.caption(
    "Compare o valor do aluno com a referência dos alunos usados no modelo. "
    "Os cartões abaixo indicam onde vale olhar com mais cuidado."
)
render_factor_cards(main_factors)
with st.expander("Ver tabela técnica dos fatores"):
    st.dataframe(main_factors, use_container_width=True, hide_index=True)

st.markdown("### Lista de prioridade")
top_n = st.slider("Quantidade", min_value=5, max_value=50, value=20, step=5)

top_cols = [c for c in ["ra", "nome", "ano_referencia", "fase_analitica", "pedra"] if c in df_year.columns]
top_risk = df_year.sort_values("score_risco", ascending=False).head(top_n).copy()
top_risk["risco_estimado"] = top_risk["score_risco_pct"].map(lambda value: f"{value:.1f}%")
top_risk = (
    top_risk.loc[:, top_cols + ["risco_estimado", "situacao", "acao_sugerida"]]
    .rename(
        columns={
            "ra": "RA",
            "nome": "Aluno",
            "ano_referencia": "Ano",
            "fase_analitica": "Fase",
            "pedra": "Pedra",
            "risco_estimado": "Risco estimado",
            "situacao": "Situação",
            "acao_sugerida": "Ação sugerida",
        }
    )
    .head(top_n)
)
st.dataframe(top_risk, use_container_width=True, hide_index=True)

st.markdown("### Observação importante")
st.write(
    "Este painel não substitui a decisão da equipe. Ele organiza os sinais dos dados para ajudar a Passos Mágicos "
    "a olhar antes, priorizar melhor e agir com mais evidência."
)
