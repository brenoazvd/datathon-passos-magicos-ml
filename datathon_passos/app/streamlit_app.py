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
st.title("Score de Risco de Defasagem - Passos Mágicos")
st.caption(
    "Triagem preventiva baseada no modelo do Datathon. "
    "O score estima risco de defasagem/piora educacional e não prevê evasão ou retenção diretamente."
)

data_path_input = st.text_input("Caminho da base com score (CSV)", value=str(DEFAULT_DATA_PATH))
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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros exibidos", int(len(df_view)))
c2.metric("Alunos únicos", int(df_view["ra"].nunique()))
if TARGET_COL in df_view.columns and df_view[TARGET_COL].notna().any():
    labeled = df_view[df_view[TARGET_COL].notna()]
    c3.metric("Taxa de risco rotulada", f"{labeled[TARGET_COL].mean() * 100:.1f}%")
else:
    c3.metric("Taxa de risco rotulada", "N/D")
c4.metric("Score médio estimado", f"{df_view['score_risco_pct'].mean():.1f}%")

st.subheader("Seleção de aluno para score individual")
anos = sorted(df_view["ano_referencia"].dropna().astype(int).unique().tolist())
selected_ano = st.selectbox("Ano de referência", anos, index=len(anos) - 1)

df_year = df_view[df_view["ano_referencia"] == selected_ano].copy()
df_year["aluno_label"] = df_year["ra"].astype(str) + " | " + df_year["nome"].fillna("sem nome").astype(str)

selected_label = st.selectbox("Aluno", sorted(df_year["aluno_label"].unique().tolist()))
row = df_year[df_year["aluno_label"] == selected_label].iloc[0]

st.markdown("### Score de risco do aluno selecionado")
target_value = row.get(TARGET_COL, np.nan)
target_delta = "Sem target real" if pd.isna(target_value) else f"Target real: {int(target_value)}"
st.metric(
    "Probabilidade estimada de risco",
    f"{row['score_risco_pct']:.1f}%",
    delta=target_delta,
)
st.progress(float(row["score_risco"]))
st.caption(
    f"Classificação pelo threshold {threshold:.2f}: "
    f"{'Sinalizado para triagem' if int(row['pred_risco_threshold']) == 1 else 'Não sinalizado'}."
)

main_factors = explain_main_factors(row, feature_stats, top_n=6)
st.markdown("### Fatores principais (proxy explicativa)")
st.caption(
    "A proxy combina importância global do RandomForest com distância do aluno em relação à mediana da base."
)
if main_factors.empty:
    st.info("Fatores explicativos indisponíveis para a base carregada.")
else:
    st.dataframe(main_factors, use_container_width=True)
    st.bar_chart(main_factors.set_index("feature")["contribuicao_proxy"])

st.markdown("### Top alunos por score previsto")
top_n = st.slider("Quantidade", min_value=5, max_value=50, value=20, step=5)
top_cols = [
    c
    for c in ["ra", "nome", "ano_referencia", "fase_analitica", "pedra", TARGET_COL, "score_risco_pct", "pred_risco_threshold"]
    if c in df_view.columns
]
top_risk = (
    df_view.sort_values("score_risco", ascending=False)
    .loc[:, top_cols]
    .head(top_n)
)
st.dataframe(top_risk, use_container_width=True)
