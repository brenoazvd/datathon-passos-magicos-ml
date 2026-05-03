from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "base_modelagem_risco.csv"


@st.cache_data
def load_gold_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_resource
def train_risk_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)

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
    rows = []
    for _, stat in feature_stats.iterrows():
        feature = stat["feature"]
        median = float(stat["median"])
        std = float(stat["std"])
        importance = float(stat["importance"])
        corr_target = float(stat["corr_target"])

        value = row.get(feature, np.nan)
        if pd.isna(value):
            value = median

        z_score = abs((float(value) - median) / std)
        contribution = z_score * importance

        local_direction = (float(value) - median) * np.sign(corr_target)
        impact_label = "Aumenta risco" if local_direction > 0 else "Reduz risco"

        rows.append(
            {
                "feature": feature,
                "valor_aluno": float(value),
                "mediana_base": median,
                "importancia_global": importance,
                "contribuicao_proxy": contribution,
                "impacto_estimado": impact_label,
            }
        )

    out = pd.DataFrame(rows).sort_values("contribuicao_proxy", ascending=False).head(top_n)
    return out


st.set_page_config(page_title="Datathon Passos Mágicos - Risco", layout="wide")
st.title("Predição de Risco de Defasagem - Passos Mágicos")
st.caption("Aplicação baseada na base Gold já processada (sem reprocessar Bronze/Silver).")

gold_path_input = st.text_input("Caminho da base Gold (CSV)", value=str(DEFAULT_GOLD_PATH))
gold_path = Path(gold_path_input)

if not gold_path.exists():
    st.error(f"Arquivo não encontrado: {gold_path}")
    st.stop()

df_gold = load_gold_dataset(str(gold_path))

missing_cols = [c for c in FEATURE_COLS + [TARGET_COL, "ra", "ano_referencia"] if c not in df_gold.columns]
if missing_cols:
    st.error(f"Colunas obrigatórias ausentes na base Gold: {missing_cols}")
    st.stop()

model, feature_stats = train_risk_model(df_gold)
X_gold = df_gold[[c for c in FEATURE_COLS if c in df_gold.columns]].copy()
df_view = df_gold.copy()
df_view["score_risco"] = model.predict_proba(X_gold)[:, 1]
df_view["score_risco_pct"] = df_view["score_risco"] * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", int(len(df_view)))
c2.metric("Alunos únicos", int(df_view["ra"].nunique()))
c3.metric("Taxa média de risco (real)", f"{df_view[TARGET_COL].mean() * 100:.1f}%")
c4.metric("Score médio previsto", f"{df_view['score_risco_pct'].mean():.1f}%")

st.subheader("Seleção de aluno para score individual")
anos = sorted(df_view["ano_referencia"].dropna().astype(int).unique().tolist())
selected_ano = st.selectbox("Ano de referência", anos, index=len(anos) - 1)

df_year = df_view[df_view["ano_referencia"] == selected_ano].copy()
df_year["aluno_label"] = df_year["ra"].astype(str) + " | " + df_year["nome"].fillna("sem nome").astype(str)

selected_label = st.selectbox("Aluno", sorted(df_year["aluno_label"].unique().tolist()))
row = df_year[df_year["aluno_label"] == selected_label].iloc[0]

st.markdown("### Score de risco do aluno selecionado")
st.metric(
    "Probabilidade prevista de risco",
    f"{row['score_risco_pct']:.1f}%",
    delta=f"Target real: {int(row[TARGET_COL])}",
)
st.progress(float(row["score_risco"]))

main_factors = explain_main_factors(row, feature_stats, top_n=6)
st.markdown("### Fatores principais (proxy explicativa)")
st.caption(
    "A proxy combina importância global do RandomForest com distância do aluno em relação à mediana da base."
)
st.dataframe(main_factors, use_container_width=True)
st.bar_chart(main_factors.set_index("feature")["contribuicao_proxy"])

st.markdown("### Top alunos por score previsto")
top_n = st.slider("Quantidade", min_value=5, max_value=50, value=20, step=5)
top_risk = (
    df_view.sort_values("score_risco", ascending=False)
    .loc[:, ["ra", "nome", "ano_referencia", "pedra", TARGET_COL, "score_risco_pct"]]
    .head(top_n)
)
st.dataframe(top_risk, use_container_width=True)
