from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _to_ra_from_name(name: pd.Series) -> pd.Series:
    s = name.astype("string").str.strip()
    num = s.str.extract(r"(?i)^aluno[-_ ]?(\d+)$", expand=False)
    return ("RA-" + num.astype("string")).where(num.notna(), s)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=",", low_memory=False)


def _find_file(base_dir: Path, filename: str) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Pasta base relacional nao encontrada: {base_dir}")
    target = filename.lower()
    hits = [p for p in base_dir.rglob("*.csv") if p.name.lower() == target]
    if not hits:
        # fallback tolerante a variacoes de codificacao/nome
        hits = [p for p in base_dir.rglob("*.csv") if p.name.lower().startswith(Path(filename).stem.lower())]
    if not hits:
        raise FileNotFoundError(f"Arquivo nao encontrado: {filename}")
    # prioriza 'Originais anonimizados'
    hits = sorted(hits, key=lambda p: ("originais anonimizados" not in str(p).lower(), len(str(p))))
    return hits[0]


def build_relational_features(base_relacional_dir: str | Path) -> pd.DataFrame:
    base = Path(base_relacional_dir)

    tb_aluno = _read_csv(_find_file(base, "TbAluno.csv"))
    tb_aluno["ra"] = _to_ra_from_name(tb_aluno["NomeAluno"])
    aluno_map = tb_aluno[["IdAluno", "ra"]].dropna().drop_duplicates()

    # Diario/frequencia por ano
    tb_freq = _read_csv(_find_file(base, "TbDiarioFrequencia.csv"))
    tb_aula = _read_csv(_find_file(base, "TbDiarioAula.csv"))
    freq = tb_freq.merge(tb_aula[["IdDiarioAula", "DataAula"]], on="IdDiarioAula", how="left")
    freq = freq.merge(aluno_map, on="IdAluno", how="left")
    freq["ano_referencia"] = pd.to_datetime(freq["DataAula"], errors="coerce").dt.year
    st = freq["StPresencaFalta"].astype("string").str.upper().str.strip()
    freq["is_presenca"] = (st == "P").astype("float")
    freq["is_falta"] = st.str.startswith("F").astype("float")

    feat_freq = (
        freq.dropna(subset=["ra", "ano_referencia"])
        .groupby(["ra", "ano_referencia"], as_index=False)
        .agg(
            freq_registros=("IdDiarioFrequencia", "count"),
            freq_tx_presenca=("is_presenca", "mean"),
            freq_tx_falta=("is_falta", "mean"),
        )
    )

    # Fase nota por ano (via historico de turma)
    tb_fase_aluno = _read_csv(_find_file(base, "TbFaseNotaAluno.csv"))
    tb_aluno_turma_hist = _read_csv(_find_file(base, "TbAlunoTurmaHistorico.csv"))
    tb_aluno_turma_hist["ano_referencia"] = pd.to_datetime(tb_aluno_turma_hist["DataOcorrencia"], errors="coerce").dt.year

    fase = tb_fase_aluno.merge(
        tb_aluno_turma_hist[["IdAluno", "IdTurma", "ano_referencia"]],
        on=["IdAluno", "IdTurma"],
        how="left",
    )
    fase = fase.merge(aluno_map, on="IdAluno", how="left")

    for c in ["NotaFase", "Faltas", "QuantAulasDadas"]:
        fase[c] = pd.to_numeric(fase[c], errors="coerce")

    feat_fase = (
        fase.dropna(subset=["ra", "ano_referencia"])
        .groupby(["ra", "ano_referencia"], as_index=False)
        .agg(
            fase_nota_media=("NotaFase", "mean"),
            fase_faltas_total=("Faltas", "sum"),
            fase_aulas_total=("QuantAulasDadas", "sum"),
        )
    )

    # Historico notas por ano
    tb_hist = _read_csv(_find_file(base, "TbHistoricoNotas.csv"))
    hist = tb_hist.merge(aluno_map, on="IdAluno", how="left")
    hist["ano_referencia"] = pd.to_numeric(hist["AnoConclusao"], errors="coerce")
    hist["NotaFinal"] = pd.to_numeric(hist["NotaFinal"], errors="coerce")
    hist["QuantidadeFaltasAnual"] = pd.to_numeric(hist["QuantidadeFaltasAnual"], errors="coerce")

    feat_hist = (
        hist.dropna(subset=["ra", "ano_referencia"])
        .groupby(["ra", "ano_referencia"], as_index=False)
        .agg(
            hist_nota_final_media=("NotaFinal", "mean"),
            hist_faltas_anual_media=("QuantidadeFaltasAnual", "mean"),
            hist_disciplinas=("IdDisciplina", "nunique"),
        )
    )

    features = feat_freq.merge(feat_fase, on=["ra", "ano_referencia"], how="outer")
    features = features.merge(feat_hist, on=["ra", "ano_referencia"], how="outer")
    features["ano_referencia"] = features["ano_referencia"].astype("Int64")
    return features


def enrich_with_relational_features(df_silver: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
    out = df_silver.merge(df_features, on=["ra", "ano_referencia"], how="left")
    return out
