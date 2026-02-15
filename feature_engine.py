"""Feature engineering — semantic column detection and transformations.

Detects the *meaning* behind each column (tags, categories, booleans,
dates hidden in strings, numbers hidden behind currency symbols, etc.)
and exposes helpers that the LLM-generated code can call at runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# Semantic type definitions
# ─────────────────────────────────────────────────────────────────

SEMANTIC_TYPES = (
    "boolean",          # yes/no, true/false, Y/N, 1/0, si/no …
    "categorical",      # low-cardinality string / int labels
    "ordinal",          # Likert scales, ratings, ordered categories
    "multi_value_tags", # comma/pipe/semicolon-separated tags in one cell
    "numeric_string",   # "$1,200" or "45%" — number hiding in a string
    "date_string",      # parseable date currently stored as object/str
    "free_text",        # long prose, comments, open-ended responses
    "identifier",       # unique / near-unique keys, not useful for analysis
    "numeric",          # already numeric (int / float)
    "datetime",         # already datetime64
    "unknown",
)


@dataclass
class ColumnProfile:
    """Rich metadata about a single column."""
    name: str
    pandas_dtype: str
    semantic_type: str
    nullable: bool
    n_unique: int
    n_total: int
    sample_values: list[Any] = field(default_factory=list)
    # Extras filled by specific detectors
    tag_separator: str | None = None       # for multi_value_tags
    tag_vocabulary: list[str] | None = None
    boolean_map: dict[str, bool] | None = None
    ordinal_order: list[str] | None = None
    numeric_prefix: str = ""               # e.g. "$" for currency
    numeric_suffix: str = ""               # e.g. "%" for percentages
    categories: list[str] | None = None
    date_format_hint: str | None = None

    def summary_line(self) -> str:
        """One-line plain-English summary used in the LLM system prompt."""
        parts = [f"{self.name} — {self.semantic_type}"]
        if self.semantic_type == "boolean" and self.boolean_map:
            truthy = [k for k, v in self.boolean_map.items() if v]
            falsy = [k for k, v in self.boolean_map.items() if not v]
            parts.append(f"  true={truthy}, false={falsy}")
        elif self.semantic_type == "multi_value_tags":
            parts.append(f"  sep='{self.tag_separator}', {len(self.tag_vocabulary or [])} unique tags")
            if self.tag_vocabulary:
                shown = self.tag_vocabulary[:12]
                parts.append(f"  tags: {shown}{'…' if len(self.tag_vocabulary or []) > 12 else ''}")
        elif self.semantic_type == "categorical":
            parts.append(f"  {self.n_unique} categories")
            if self.categories:
                shown = self.categories[:10]
                parts.append(f"  values: {shown}{'…' if len(self.categories or []) > 10 else ''}")
        elif self.semantic_type == "ordinal" and self.ordinal_order:
            parts.append(f"  order: {self.ordinal_order}")
        elif self.semantic_type == "numeric_string":
            parts.append(f"  prefix='{self.numeric_prefix}' suffix='{self.numeric_suffix}'")
        elif self.semantic_type == "date_string":
            parts.append(f"  date pattern hint: {self.date_format_hint}")
        elif self.semantic_type == "free_text":
            parts.append(f"  avg length ~{self._avg_len():.0f} chars")
        return "\n    ".join(parts)

    def _avg_len(self) -> float:
        lens = [len(str(v)) for v in self.sample_values if pd.notna(v)]
        return sum(lens) / max(len(lens), 1)


# ─────────────────────────────────────────────────────────────────
# Public API — profiling
# ─────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame) -> dict[str, ColumnProfile]:
    """Analyse every column and return a dict of ColumnProfiles."""
    profiles: dict[str, ColumnProfile] = {}
    for col in df.columns:
        profiles[col] = _profile_column(df, col)
    return profiles


def profile_summary_text(profiles: dict[str, ColumnProfile]) -> str:
    """Multi-line human-readable summary for the LLM system prompt."""
    lines = ["Perfis semânticos das colunas (auto-detectados):"]
    for p in profiles.values():
        lines.append(f"  • {p.summary_line()}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Public API — transformations (available to generated code)
# ─────────────────────────────────────────────────────────────────

def to_boolean(series: pd.Series, true_values: list[str] | None = None) -> pd.Series:
    """Convert a boolean-like column to actual bool (True/False/NaN).

    Auto-detects common patterns if *true_values* is not given.
    """
    s = series.astype(str).str.strip().str.lower()
    if true_values is None:
        true_values = ["yes", "y", "true", "t", "1", "si", "sí", "oui",
                       "sim", "ja", "da", "tak", "yeah", "yep", "x"]
    true_set = {v.lower() for v in true_values}
    # Map values
    mapped = s.map(lambda v: True if v in true_set else (
        False if v and v != "nan" else None
    ))
    return mapped.astype("boolean")


def explode_tags(
    df: pd.DataFrame,
    column: str,
    sep: str | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Explode a multi-value/tags column into one-hot indicator columns.

    Returns a *new* DataFrame with the original column kept and new
    binary columns ``{prefix}_{tag}`` appended.

    If *sep* is None it is auto-detected from the data.
    """
    if sep is None:
        sep = _detect_tag_separator(df[column])
    if prefix is None:
        prefix = column

    tags_series = df[column].astype(str).str.split(sep)
    tags_series = tags_series.apply(
        lambda lst: [t.strip() for t in lst if t.strip() and t.strip().lower() != "nan"]
        if isinstance(lst, list) else []
    )
    # Build indicator columns
    mlb_data: dict[str, list[int]] = {}
    all_tags = sorted({tag for tags in tags_series for tag in tags})
    for tag in all_tags:
        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", tag).strip("_")
        col_name = f"{prefix}_{safe_name}"
        mlb_data[col_name] = [int(tag in row_tags) for row_tags in tags_series]

    return pd.concat([df, pd.DataFrame(mlb_data, index=df.index)], axis=1)


def encode_categorical(
    series: pd.Series,
    method: str = "onehot",
    prefix: str | None = None,
) -> pd.DataFrame | pd.Series:
    """Encode a categorical column.

    Methods:
        'onehot'    — returns DataFrame of indicator columns
        'label'     — returns integer-encoded Series (0, 1, 2, …)
        'frequency' — returns Series with value counts / total
    """
    if method == "label":
        cat = series.astype("category")
        return cat.cat.codes.rename(series.name)
    elif method == "frequency":
        freq = series.value_counts(normalize=True)
        return series.map(freq).rename(series.name)
    else:  # onehot
        return pd.get_dummies(series, prefix=prefix or series.name, dtype=int)


def parse_numeric_strings(series: pd.Series) -> pd.Series:
    """Strip currency symbols, %, commas, spaces and convert to float."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[€£¥₹$R\s]", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_dates(series: pd.Series, fmt: str | None = None) -> pd.Series:
    """Parse string dates into datetime, optionally with explicit format."""
    if fmt and fmt != "auto":
        return pd.to_datetime(series, format=fmt, errors="coerce")
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except Exception:
        return pd.to_datetime(series, errors="coerce")


def extract_date_features(series: pd.Series, prefix: str | None = None) -> pd.DataFrame:
    """From a datetime Series, extract year, month, day, weekday, hour."""
    dt = pd.to_datetime(series, errors="coerce")
    p = prefix or series.name or "date"
    return pd.DataFrame({
        f"{p}_year": dt.dt.year,
        f"{p}_month": dt.dt.month,
        f"{p}_day": dt.dt.day,
        f"{p}_weekday": dt.dt.day_name(),
        f"{p}_hour": dt.dt.hour,
    })


def bin_numeric(
    series: pd.Series,
    bins: int | list[float] = 5,
    labels: list[str] | None = None,
) -> pd.Series:
    """Bin a numeric column into categories."""
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def normalize(series: pd.Series, method: str = "minmax") -> pd.Series:
    """Normalize numeric column: 'minmax' (0-1) or 'zscore'."""
    if method == "zscore":
        return (series - series.mean()) / series.std()
    # minmax
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0.0
    return (series - mn) / (mx - mn)


def make_ordinal(
    series: pd.Series,
    order: list[str] | None = None,
) -> pd.Series:
    """Convert a column to an ordered Categorical.

    If *order* is not given, attempts to infer common ordinal patterns.
    """
    if order is None:
        order = _infer_ordinal_order(series)
    if order:
        cat_type = pd.CategoricalDtype(categories=order, ordered=True)
        return series.astype(str).str.strip().str.lower().astype(cat_type)
    return series.astype("category")


# ─────────────────────────────────────────────────────────────────
# Private — column profiling
# ─────────────────────────────────────────────────────────────────

def _profile_column(df: pd.DataFrame, col: str) -> ColumnProfile:
    s = df[col]
    n_total = len(s)
    n_unique = s.nunique()
    nullable = bool(s.isna().any())
    sample = s.dropna().head(20).tolist()
    dtype = str(s.dtype)

    base = ColumnProfile(
        name=col,
        pandas_dtype=dtype,
        semantic_type="unknown",
        nullable=nullable,
        n_unique=n_unique,
        n_total=n_total,
        sample_values=sample,
    )

    # ── Already datetime ─────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(s):
        base.semantic_type = "datetime"
        return base

    # ── Already numeric ──────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(s):
        if n_unique == 2 and set(s.dropna().unique()) <= {0, 1, 0.0, 1.0}:
            base.semantic_type = "boolean"
            base.boolean_map = {"1": True, "0": False}
        elif n_unique <= _categorical_threshold(n_total):
            base.semantic_type = "categorical"
            base.categories = sorted(s.dropna().unique().astype(str).tolist())
        else:
            base.semantic_type = "numeric"
        return base

    # ── String-based detection ───────────────────────────────────
    str_vals = s.dropna().astype(str).str.strip()
    lowered = str_vals.str.lower()

    # Boolean?
    bool_map = _detect_boolean(lowered)
    if bool_map is not None:
        base.semantic_type = "boolean"
        base.boolean_map = bool_map
        return base

    # Numeric hiding in strings?
    if _looks_numeric_string(str_vals):
        base.semantic_type = "numeric_string"
        base.numeric_prefix, base.numeric_suffix = _detect_numeric_affixes(str_vals)
        return base

    # Date hiding in strings?
    date_fmt = _detect_date_string(str_vals)
    if date_fmt:
        base.semantic_type = "date_string"
        base.date_format_hint = date_fmt
        return base

    # Multi-value / tags?
    tag_sep = _detect_tag_separator(s)
    if tag_sep:
        tags = str_vals.str.split(tag_sep).explode().str.strip()
        tags = tags[tags != ""]
        vocab = sorted(tags.unique().tolist())
        avg_per_row = str_vals.str.split(tag_sep).apply(len).mean()
        if avg_per_row > 1.2 and len(vocab) >= 2:
            base.semantic_type = "multi_value_tags"
            base.tag_separator = tag_sep
            base.tag_vocabulary = vocab
            return base

    # Ordinal?
    ordinal_order = _infer_ordinal_order(s)
    if ordinal_order:
        base.semantic_type = "ordinal"
        base.ordinal_order = ordinal_order
        base.categories = ordinal_order
        return base

    # Categorical vs free-text vs identifier
    if n_unique <= _categorical_threshold(n_total):
        base.semantic_type = "categorical"
        base.categories = sorted(str_vals.unique().tolist())
    elif n_unique / max(n_total, 1) > 0.9:
        avg_len = str_vals.str.len().mean()
        if avg_len > 50:
            base.semantic_type = "free_text"
        else:
            base.semantic_type = "identifier"
    else:
        avg_len = str_vals.str.len().mean()
        if avg_len > 80:
            base.semantic_type = "free_text"
        else:
            base.semantic_type = "categorical"
            base.categories = sorted(str_vals.unique().tolist())

    return base


# ─────────────────────────────────────────────────────────────────
# Private — detection helpers
# ─────────────────────────────────────────────────────────────────

_BOOLEAN_TRUE = {"yes", "y", "true", "t", "1", "si", "sí", "oui", "sim",
                 "ja", "da", "tak", "yeah", "yep", "x"}
_BOOLEAN_FALSE = {"no", "n", "false", "f", "0", "não", "nao", "non",
                  "nein", "nie", "nope", "nah", ""}

_ORDINAL_PATTERNS: list[list[str]] = [
    # Likert 5-point
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"],
    ["very poor", "poor", "average", "good", "excellent"],
    ["very low", "low", "medium", "high", "very high"],
    ["never", "rarely", "sometimes", "often", "always"],
    # 3-point
    ["low", "medium", "high"],
    ["small", "medium", "large"],
    ["bad", "average", "good"],
    ["disagree", "neutral", "agree"],
    # Education
    ["high school", "bachelor", "master", "phd"],
    ["high school", "bachelors", "masters", "doctorate"],
    # Satisfaction short
    ["poor", "fair", "good", "very good", "excellent"],
    # Frequency
    ["daily", "weekly", "monthly", "yearly"],
    # Priority / severity
    ["critical", "high", "medium", "low"],
    ["p0", "p1", "p2", "p3", "p4"],
    # ── Português (BR) ──
    ["discordo totalmente", "discordo", "neutro", "concordo", "concordo totalmente"],
    ["muito insatisfeito", "insatisfeito", "neutro", "satisfeito", "muito satisfeito"],
    ["péssimo", "ruim", "regular", "bom", "excelente"],
    ["muito baixo", "baixo", "médio", "alto", "muito alto"],
    ["nunca", "raramente", "às vezes", "frequentemente", "sempre"],
    ["baixo", "médio", "alto"],
    ["pequeno", "médio", "grande"],
    ["ruim", "regular", "bom"],
    ["discordo", "neutro", "concordo"],
    ["ensino médio", "graduação", "mestrado", "doutorado"],
    ["ruim", "razoável", "bom", "muito bom", "excelente"],
    ["diário", "semanal", "mensal", "anual"],
    ["crítico", "alto", "médio", "baixo"],
]


def _categorical_threshold(n_total: int) -> int:
    """Max unique values to still consider a column categorical."""
    if n_total <= 50:
        return max(n_total // 2, 10)
    if n_total <= 500:
        return 30
    return min(50, int(n_total * 0.05))


def _detect_boolean(lowered: pd.Series) -> dict[str, bool] | None:
    """Return a mapping if the column looks boolean, else None."""
    uniq = set(lowered.unique())
    # Remove NaN-like
    uniq.discard("nan")
    uniq.discard("")
    if len(uniq) > 3 or len(uniq) < 1:
        return None

    true_found = uniq & _BOOLEAN_TRUE
    false_found = uniq & _BOOLEAN_FALSE
    if true_found and false_found and (true_found | false_found) == uniq:
        mapping: dict[str, bool] = {}
        for v in true_found:
            mapping[v] = True
        for v in false_found:
            mapping[v] = False
        return mapping
    return None


def _looks_numeric_string(str_vals: pd.Series) -> bool:
    """True if >60% of non-empty values are numbers once cleaned."""
    cleaned = (
        str_vals
        .str.replace(r"[€£¥₹$R\s,%]", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    pct = numeric.notna().sum() / max(len(cleaned), 1)
    return pct > 0.6 and len(cleaned) > 0


def _detect_numeric_affixes(str_vals: pd.Series) -> tuple[str, str]:
    """Find common leading prefix (e.g. $) and trailing suffix (e.g. %)."""
    prefix = ""
    suffix = ""
    sample = str_vals.head(50).tolist()

    # Prefix: first non-digit, non-space, non-minus character at start
    pfx_counts: dict[str, int] = {}
    sfx_counts: dict[str, int] = {}
    for v in sample:
        v = v.strip()
        if v and not v[0].isdigit() and v[0] not in "-+":
            m = re.match(r"^([^0-9\-+]+)", v)
            if m:
                pfx_counts[m.group(1).strip()] = pfx_counts.get(m.group(1).strip(), 0) + 1
        if v and not v[-1].isdigit():
            m = re.search(r"([^0-9]+)$", v)
            if m:
                sfx_counts[m.group(1).strip()] = sfx_counts.get(m.group(1).strip(), 0) + 1

    if pfx_counts:
        top_pfx = max(pfx_counts, key=pfx_counts.get)  # type: ignore[arg-type]
        if pfx_counts[top_pfx] > len(sample) * 0.4:
            prefix = top_pfx
    if sfx_counts:
        top_sfx = max(sfx_counts, key=sfx_counts.get)  # type: ignore[arg-type]
        if sfx_counts[top_sfx] > len(sample) * 0.4:
            suffix = top_sfx

    return prefix, suffix


def _detect_date_string(str_vals: pd.Series) -> str | None:
    """Try to parse a sample; return a format hint if >50% succeed."""
    sample = str_vals.head(40)
    if sample.empty:
        return None
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except Exception:
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
        except Exception:
            return None
    pct = parsed.notna().sum() / len(sample)
    if pct > 0.5:
        # Try to guess the format from a successful value
        first_ok = sample[parsed.notna()].iloc[0] if parsed.notna().any() else None
        return _guess_date_format(first_ok) if first_ok else "auto"
    return None


def _guess_date_format(val: str) -> str:
    """Return a strftime-ish hint for a date string."""
    val = val.strip()
    # ISO
    if re.match(r"\d{4}-\d{2}-\d{2}", val):
        return "%Y-%m-%d"
    # US
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}", val):
        return "%m/%d/%Y"
    if re.match(r"\d{1,2}/\d{1,2}/\d{2}$", val):
        return "%m/%d/%y"
    # EU
    if re.match(r"\d{1,2}\.\d{1,2}\.\d{4}", val):
        return "%d.%m.%Y"
    if re.match(r"\d{1,2}-\d{1,2}-\d{4}", val):
        return "%d-%m-%Y"
    return "auto"


def _detect_tag_separator(series: pd.Series) -> str | None:
    """Detect whether a string column uses a common multi-value separator."""
    str_vals = series.dropna().astype(str)
    if str_vals.empty:
        return None

    sample = str_vals.head(100)
    for sep in [",", ";", "|", " / ", " // "]:
        hits = sample.str.contains(re.escape(sep), regex=True).sum()
        if hits / len(sample) > 0.25:
            # Confirm that splitting yields >1 item on average
            avg_parts = sample.str.split(sep).apply(len).mean()
            if avg_parts > 1.3:
                return sep
    return None


def _infer_ordinal_order(series: pd.Series) -> list[str] | None:
    """Check if the column's unique values match a known ordinal pattern."""
    uniq = set(series.dropna().astype(str).str.strip().str.lower().unique())
    if not uniq or len(uniq) > 15:
        return None

    for pattern in _ORDINAL_PATTERNS:
        pattern_set = set(pattern)
        if uniq <= pattern_set and len(uniq) >= 2:
            # Return in the correct order, filtering to only values present
            return [p for p in pattern if p in uniq]
    return None
