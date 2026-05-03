from __future__ import annotations

import re
import unicodedata
from typing import Dict

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

SCHEMA_CANONICAL = [
    "ra",
    "nome",
    "ano_referencia",
    "fase",
    "turma",
    "idade",
    "genero",
    "ano_ingresso",
    "instituicao_ensino",
    "pedra",
    "inde",
    "ian",
    "ida",
    "ieg",
    "iaa",
    "ips",
    "ipp",
    "ipv",
    "atingiu_pv",
    "defasagem",
    "fase_ideal",
    "matematica",
    "portugues",
    "ingles",
    "escola",
    "status_ativo",
]

OLD_BASE_MAP_BY_YEAR: Dict[int, Dict[str, str]] = {
    2020: {
        "nome": "nome",
        "instituicao_ensino_aluno_2020": "instituicao_ensino",
        "idade_aluno_2020": "idade",
        "anos_pm_2020": "ano_ingresso",
        "fase_turma_2020": "turma",
        "pedra_2020": "pedra",
        "inde_2020": "inde",
        "ian_2020": "ian",
        "ida_2020": "ida",
        "ieg_2020": "ieg",
        "iaa_2020": "iaa",
        "ips_2020": "ips",
        "ipp_2020": "ipp",
        "ipv_2020": "ipv",
        "ponto_virada_2020": "atingiu_pv",
    },
    2021: {
        "nome": "nome",
        "instituicao_ensino_aluno_2021": "instituicao_ensino",
        "fase_2021": "fase",
        "turma_2021": "turma",
        "pedra_2021": "pedra",
        "inde_2021": "inde",
        "ian_2021": "ian",
        "ida_2021": "ida",
        "ieg_2021": "ieg",
        "iaa_2021": "iaa",
        "ips_2021": "ips",
        "ipp_2021": "ipp",
        "ipv_2021": "ipv",
        "ponto_virada_2021": "atingiu_pv",
        "nivel_ideal_2021": "fase_ideal",
        "defasagem_2021": "defasagem",
    },
    2022: {
        "nome": "nome",
        "instituicao_ensino_aluno_2021": "instituicao_ensino",
        "fase_2022": "fase",
        "turma_2022": "turma",
        "pedra_2022": "pedra",
        "inde_2022": "inde",
        "ian_2022": "ian",
        "ida_2022": "ida",
        "ieg_2022": "ieg",
        "iaa_2022": "iaa",
        "ips_2022": "ips",
        "ipp_2022": "ipp",
        "ipv_2022": "ipv",
        "ponto_virada_2022": "atingiu_pv",
        "nivel_ideal_2022": "fase_ideal",
        "defasagem_2022": "defasagem",
        "nota_mat_2022": "matematica",
        "nota_port_2022": "portugues",
        "nota_ing_2022": "ingles",
        "ano_ingresso_2022": "ano_ingresso",
    },
}

COLUMN_MAP_BY_YEAR: Dict[int, Dict[str, str]] = {
    2022: {
        "ra": "ra",
        "nome": "nome",
        "fase": "fase",
        "turma": "turma",
        "idade 22": "idade",
        "genero": "genero",
        "ano ingresso": "ano_ingresso",
        "instituicao de ensino": "instituicao_ensino",
        "pedra 22": "pedra",
        "inde 22": "inde",
        "ian": "ian",
        "ida": "ida",
        "ieg": "ieg",
        "iaa": "iaa",
        "ips": "ips",
        "ipv": "ipv",
        "atingiu pv": "atingiu_pv",
        "defas": "defasagem",
        "fase ideal": "fase_ideal",
        "matem": "matematica",
        "portug": "portugues",
        "ingles": "ingles",
    },
    2023: {
        "ra": "ra",
        "nome anonimizado": "nome",
        "fase": "fase",
        "turma": "turma",
        "idade": "idade",
        "genero": "genero",
        "ano ingresso": "ano_ingresso",
        "instituicao de ensino": "instituicao_ensino",
        "pedra 2023": "pedra",
        "inde 2023": "inde",
        "inde 23": "inde",
        "ian": "ian",
        "ida": "ida",
        "ieg": "ieg",
        "iaa": "iaa",
        "ips": "ips",
        "ipp": "ipp",
        "ipv": "ipv",
        "atingiu pv": "atingiu_pv",
        "defasagem": "defasagem",
        "fase ideal": "fase_ideal",
        "mat": "matematica",
        "por": "portugues",
        "ing": "ingles",
    },
    2024: {
        "ra": "ra",
        "nome anonimizado": "nome",
        "fase": "fase",
        "turma": "turma",
        "idade": "idade",
        "genero": "genero",
        "ano ingresso": "ano_ingresso",
        "instituicao de ensino": "instituicao_ensino",
        "pedra 2024": "pedra",
        "inde 2024": "inde",
        "ian": "ian",
        "ida": "ida",
        "ieg": "ieg",
        "iaa": "iaa",
        "ips": "ips",
        "ipp": "ipp",
        "ipv": "ipv",
        "atingiu pv": "atingiu_pv",
        "defasagem": "defasagem",
        "fase ideal": "fase_ideal",
        "mat": "matematica",
        "por": "portugues",
        "ing": "ingles",
        "escola": "escola",
        "ativo/ inativo": "status_ativo",
        "ativo/ inativo.1": "status_ativo",
    },
}


def _normalize_colname(col: str) -> str:
    text = str(col).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_year(sheet_name: str) -> int:
    match = re.search(r"(20\d{2})", sheet_name)
    if not match:
        raise ValueError(f"Nao foi possivel identificar o ano na aba: {sheet_name}")
    return int(match.group(1))


def normalize_sheet(df_raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    year = _extract_year(sheet_name)
    mapping = COLUMN_MAP_BY_YEAR.get(year, {})
    rename_dict = {}
    for col in df_raw.columns:
        key = _normalize_colname(col)
        if key in mapping:
            rename_dict[col] = mapping[key]
    selected = df_raw.rename(columns=rename_dict)
    if selected.columns.duplicated().any():
        # Algumas abas trazem colunas equivalentes (ex.: INDE 2023 e INDE 23).
        # Mantemos apenas a primeira ocorrencia apos o mapeamento canonical.
        selected = selected.loc[:, ~selected.columns.duplicated()]

    if "ra" not in selected.columns:
        raise ValueError(f"Aba {sheet_name} sem coluna RA apos mapeamento")

    selected["ano_referencia"] = year

    for col in SCHEMA_CANONICAL:
        if col not in selected.columns:
            selected[col] = pd.NA

    out = selected[SCHEMA_CANONICAL].copy()

    numeric_cols = [
        "idade", "ano_ingresso", "inde", "ian", "ida", "ieg", "iaa",
        "ips", "ipp", "ipv", "defasagem", "fase_ideal", "matematica",
        "portugues", "ingles",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ra"] = out["ra"].astype("string").str.strip()
    out["nome"] = out["nome"].astype("string").str.strip()
    out = out[out["ra"].notna()]
    out = out.drop_duplicates(subset=["ra", "ano_referencia"], keep="last")
    return out


def load_and_normalize_workbook(workbook_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(workbook_path)
    normalized = []
    for sheet in xls.sheet_names:
        raw = xls.parse(sheet)
        normalized.append(normalize_sheet(raw, sheet))
    return pd.concat(normalized, ignore_index=True)


def _normalize_old_columns(df_old: pd.DataFrame) -> pd.DataFrame:
    rename = {c: _normalize_colname(c) for c in df_old.columns}
    out = df_old.rename(columns=rename).copy()
    if "nome" in out.columns:
        nome = out["nome"].astype("string").str.strip()
        extracted = nome.str.extract(r"(?i)^aluno[-_ ]?(\d+)$", expand=False)
        ra_from_nome = "RA-" + extracted.astype("string")
        out["ra"] = ra_from_nome.where(extracted.notna(), nome)
    else:
        raise ValueError("Base antiga sem coluna NOME para chave de aluno.")
    return out


def _normalize_old_year(df_old_norm: pd.DataFrame, year: int) -> pd.DataFrame:
    mapping = OLD_BASE_MAP_BY_YEAR[year]
    selected = df_old_norm.rename(columns={k: v for k, v in mapping.items() if k in df_old_norm.columns}).copy()
    selected["ano_referencia"] = year

    for col in SCHEMA_CANONICAL:
        if col not in selected.columns:
            selected[col] = pd.NA

    out = selected[SCHEMA_CANONICAL].copy()
    numeric_cols = [
        "idade", "ano_ingresso", "inde", "ian", "ida", "ieg", "iaa",
        "ips", "ipp", "ipv", "defasagem", "fase_ideal", "matematica",
        "portugues", "ingles",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ra"] = out["ra"].astype("string").str.strip()
    out["nome"] = out["nome"].astype("string").str.strip()
    out = out[out["ra"].notna()]
    out = out.drop_duplicates(subset=["ra", "ano_referencia"], keep="last")
    return out


def load_and_normalize_old_base(old_base_path: str) -> pd.DataFrame:
    if old_base_path.lower().endswith(".csv"):
        old_raw = pd.read_csv(old_base_path, sep=";")
    else:
        old_raw = pd.read_excel(old_base_path)

    old_norm = _normalize_old_columns(old_raw)
    frames = [_normalize_old_year(old_norm, yr) for yr in [2020, 2021, 2022]]
    return pd.concat(frames, ignore_index=True)


def reconcile_bases(df_new: pd.DataFrame, df_old: pd.DataFrame) -> pd.DataFrame:
    new_df = df_new.copy()
    old_df = df_old.copy()

    new_df["source_system"] = "nova_2022_2024"
    new_df["source_priority"] = 2
    old_df["source_system"] = "antiga_2020_2022"
    old_df["source_priority"] = 1

    merged = pd.concat([new_df, old_df], ignore_index=True, sort=False)
    merged = merged.sort_values(["ra", "ano_referencia", "source_priority"], ascending=[True, True, False])
    merged = merged.drop_duplicates(subset=["ra", "ano_referencia"], keep="first").reset_index(drop=True)
    return merged


class AlunoPEDEModel(BaseModel):
    ra: str = Field(min_length=1)
    nome: str | None = None
    ano_referencia: int = Field(ge=2020, le=2035)
    fase: str | None = None
    turma: str | None = None
    idade: float | None = Field(default=None, ge=0, le=30)
    genero: str | None = None
    ano_ingresso: float | None = Field(default=None, ge=1990, le=2035)
    instituicao_ensino: str | None = None
    pedra: str | None = None
    inde: float | None = Field(default=None, ge=0, le=10)
    ian: float | None = Field(default=None, ge=0, le=10)
    ida: float | None = Field(default=None, ge=0, le=10)
    ieg: float | None = Field(default=None, ge=0, le=10)
    iaa: float | None = Field(default=None, ge=0, le=15)
    ips: float | None = Field(default=None, ge=0, le=10)
    ipp: float | None = Field(default=None, ge=0, le=10)
    ipv: float | None = Field(default=None, ge=0, le=15)
    atingiu_pv: str | None = None
    defasagem: float | None = None
    fase_ideal: float | None = None
    matematica: float | None = Field(default=None, ge=0, le=10)
    portugues: float | None = Field(default=None, ge=0, le=10)
    ingles: float | None = Field(default=None, ge=0, le=10)
    escola: str | None = None
    status_ativo: str | None = None

    @field_validator("ra")
    @classmethod
    def validate_ra(cls, value: str) -> str:
        v = str(value).strip()
        if not v:
            raise ValueError("ra vazio")
        return v

    @field_validator("fase", "turma", mode="before")
    @classmethod
    def coerce_text_fields(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None


def _clean_record_for_validation(record: dict) -> dict:
    cleaned = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def validate_normalized_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows = []
    invalid_rows = []

    for idx, row in df.iterrows():
        rec = _clean_record_for_validation(row.to_dict())
        try:
            validated = AlunoPEDEModel(**rec)
            valid_rows.append(validated.model_dump())
        except ValidationError as exc:
            invalid_rows.append(
                {
                    "row_index": idx,
                    "ra": rec.get("ra"),
                    "ano_referencia": rec.get("ano_referencia"),
                    "errors": exc.json(),
                }
            )

    valid_df = pd.DataFrame(valid_rows)
    if valid_df.empty:
        valid_df = pd.DataFrame(columns=SCHEMA_CANONICAL)
    else:
        for col in SCHEMA_CANONICAL:
            if col not in valid_df.columns:
                valid_df[col] = pd.NA
        valid_df = valid_df[SCHEMA_CANONICAL]

    invalid_df = pd.DataFrame(invalid_rows)
    return valid_df, invalid_df
