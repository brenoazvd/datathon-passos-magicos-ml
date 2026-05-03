from __future__ import annotations

import pandas as pd


def build_risk_dataset(df: pd.DataFrame, queda_ida_min: float = 0.7, aumento_ian_min: float = 0.7) -> pd.DataFrame:
    """Cria target temporal de risco por aluno.

    Regra de risco em t+1:
    - queda de IDA maior/igual a `queda_ida_min`, OU
    - aumento de IAN maior/igual a `aumento_ian_min`, OU
    - aumento da defasagem.
    """
    req = {"ra", "ano_referencia", "ida", "ian", "defasagem"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(missing)}")

    d = df.copy().sort_values(["ra", "ano_referencia"])
    d["ida_next"] = d.groupby("ra")["ida"].shift(-1)
    d["ian_next"] = d.groupby("ra")["ian"].shift(-1)
    d["defasagem_next"] = d.groupby("ra")["defasagem"].shift(-1)

    d["delta_ida_next"] = d["ida_next"] - d["ida"]
    d["delta_ian_next"] = d["ian_next"] - d["ian"]
    d["delta_defasagem_next"] = d["defasagem_next"] - d["defasagem"]

    risk = (
        (d["delta_ida_next"] <= -abs(queda_ida_min))
        | (d["delta_ian_next"] >= abs(aumento_ian_min))
        | (d["delta_defasagem_next"] > 0)
    )
    d["target_risco"] = risk.astype("Int64")

    d = d[d["ida_next"].notna() | d["ian_next"].notna() | d["defasagem_next"].notna()].copy()
    return d
