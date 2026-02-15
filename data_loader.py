"""Unified data loader supporting CSV, XLSX, JSON, and XML tabular data."""

from __future__ import annotations

import io
from pathlib import PurePath
from typing import Any

import pandas as pd
import streamlit as st

from feature_engine import ColumnProfile, profile_dataframe, profile_summary_text


# ── public helpers ──────────────────────────────────────────────


def load_dataframe(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Detect format from the file extension and return a DataFrame."""
    name = uploaded_file.name.lower()
    ext = PurePath(name).suffix

    loaders: dict[str, Any] = {
        ".csv": _load_csv,
        ".tsv": _load_tsv,
        ".xlsx": _load_xlsx,
        ".xls": _load_xlsx,
        ".json": _load_json,
        ".xml": _load_xml,
    }

    loader = loaders.get(ext)
    if loader is None:
        supported = ", ".join(sorted(loaders))
        raise ValueError(f"Unsupported file format '{ext}'. Supported: {supported}")

    return loader(uploaded_file)


def dataframe_summary(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile] | None = None,
    max_rows: int = 5,
) -> str:
    """Return a compact text summary of a DataFrame for the LLM context.

    When *profiles* is provided the summary includes semantic type info.
    """
    lines: list[str] = []
    lines.append(f"Dimensão: {df.shape[0]} linhas × {df.shape[1]} colunas")
    lines.append("")
    lines.append("Colunas e tipos:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        nunique = df[col].nunique()
        lines.append(f"  • {col} ({df[col].dtype}) — {non_null} não-nulo, {nunique} únicos")
    lines.append("")

    # Perfis semânticos
    if profiles:
        lines.append(profile_summary_text(profiles))
        lines.append("")

    lines.append(f"Primeiras {max_rows} linhas:")
    lines.append(df.head(max_rows).to_string(index=False))
    lines.append("")
    lines.append("Estatísticas básicas:")
    lines.append(df.describe(include="all").to_string())
    return "\n".join(lines)


# ── private loaders ─────────────────────────────────────────────


def _load_csv(f: Any) -> pd.DataFrame:
    """Load CSV, auto-detecting separator."""
    raw = f.read()
    f.seek(0)
    # Try to detect the separator from first line
    first_line = raw.decode("utf-8", errors="replace").split("\n")[0]
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line:
        sep = ";"
    else:
        sep = ","
    return pd.read_csv(io.BytesIO(raw), sep=sep, encoding="utf-8")


def _load_tsv(f: Any) -> pd.DataFrame:
    return pd.read_csv(f, sep="\t")


def _load_xlsx(f: Any) -> pd.DataFrame:
    return pd.read_excel(f, engine="openpyxl")


def _load_json(f: Any) -> pd.DataFrame:
    """Load JSON — supports records, columns, or plain array format."""
    return pd.read_json(f)


def _load_xml(f: Any) -> pd.DataFrame:
    raw = f.read()
    return pd.read_xml(io.BytesIO(raw))
