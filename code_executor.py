"""Safe(ish) execution of LLM-generated Streamlit/Plotly code."""

from __future__ import annotations

import contextlib
import io
import traceback
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import feature_engine as fe


def execute_generated_code(
    code: str,
    df: pd.DataFrame,
    *,
    container: Any = None,
) -> dict:
    """Execute LLM-generated code in a restricted namespace.

    The code has access to:
      - df            : the user's DataFrame (a copy — safe to mutate)
      - pd            : pandas
      - px            : plotly.express
      - go            : plotly.graph_objects
      - pio           : plotly.io
      - st            : streamlit  (writes go to *container* if provided)
      - update_data() : call with a new DataFrame to persist it back

    Returns a dict with keys:
      - success          (bool)
      - error            (str | None)
      - figures          (list[go.Figure])  — any plotly figures created
      - show_only_figures(list[go.Figure])  — figures only from fig.show()
      - updated_df       (pd.DataFrame | None) — if the code updated the data
    """
    figures: list[go.Figure] = []
    show_only_figures: list[go.Figure] = []  # from fig.show(), not yet rendered
    _original_plotly_show = go.Figure.show
    # Holder for the updated DataFrame (set by update_data() or df reassignment)
    updated_df_holder: list[pd.DataFrame | None] = [None]

    # Monkey-patch Figure.show so generated code that calls fig.show()
    # appends the figure instead.
    def _capture_show(self: go.Figure, *a: Any, **kw: Any) -> None:  # noqa: ARG001
        figures.append(self)
        show_only_figures.append(self)

    def _update_data(new_df: pd.DataFrame) -> None:
        """Persist a transformed DataFrame back to the session.

        Call this from generated code to replace the loaded dataset.
        """
        if not isinstance(new_df, pd.DataFrame):
            raise TypeError("update_data() expects a pandas DataFrame.")
        updated_df_holder[0] = new_df

    target = container or st

    # Wrap st.plotly_chart to also capture rendered figures
    class _StWrapper:
        """Proxy that intercepts plotly_chart calls to capture figures."""

        def __init__(self, real_st: Any) -> None:
            self._real = real_st

        def plotly_chart(self, figure_or_data: Any, **kwargs: Any) -> Any:
            if isinstance(figure_or_data, go.Figure):
                figures.append(figure_or_data)
            return self._real.plotly_chart(figure_or_data, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    st_proxy = _StWrapper(target)

    df_copy = df.copy()
    # Snapshot shape/columns before execution so we can detect in-place mutations
    _original_shape = df_copy.shape
    _original_columns = list(df_copy.columns)

    namespace: dict[str, Any] = {
        "df": df_copy,
        "pd": pd,
        "px": px,
        "go": go,
        "pio": pio,
        "st": st_proxy,
        # Data update function
        "update_data": _update_data,
        # Feature engineering helpers
        "fe": fe,
        "to_boolean": fe.to_boolean,
        "explode_tags": fe.explode_tags,
        "encode_categorical": fe.encode_categorical,
        "parse_numeric_strings": fe.parse_numeric_strings,
        "parse_dates": fe.parse_dates,
        "extract_date_features": fe.extract_date_features,
        "bin_numeric": fe.bin_numeric,
        "normalize": fe.normalize,
        "make_ordinal": fe.make_ordinal,
        "__builtins__": _safe_builtins(),
    }

    stderr_capture = io.StringIO()

    def _resolve_updated_df() -> pd.DataFrame | None:
        """Check if the data was updated — either via update_data() or
        by reassigning / mutating ``df`` in the namespace."""
        # Explicit update_data() takes priority
        if updated_df_holder[0] is not None:
            return updated_df_holder[0]
        # Check if the code reassigned ``df`` to a new object
        ns_df = namespace.get("df")
        if ns_df is not None and isinstance(ns_df, pd.DataFrame):
            if ns_df is not df_copy:
                # df was reassigned to a brand-new DataFrame
                return ns_df
            # Same object — but check if it was mutated in-place
            # (columns added/removed/renamed, rows added/dropped)
            if (
                ns_df.shape != _original_shape
                or list(ns_df.columns) != _original_columns
            ):
                return ns_df
        return None

    try:
        go.Figure.show = _capture_show  # type: ignore[assignment]
        with contextlib.redirect_stderr(stderr_capture):
            exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102
        return {
            "success": True,
            "error": None,
            "figures": figures,
            "show_only_figures": show_only_figures,
            "updated_df": _resolve_updated_df(),
        }
    except Exception:
        tb = traceback.format_exc()
        return {
            "success": False,
            "error": tb,
            "figures": figures,
            "show_only_figures": show_only_figures,
            "updated_df": None,
        }
    finally:
        go.Figure.show = _original_plotly_show  # type: ignore[assignment]


def extract_code_block(text: str) -> str:
    """Pull the first ```python ... ``` block from LLM output."""
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        # Skip optional language tag on same line
        newline = text.index("\n", start)
        start = newline + 1
        end = text.index("```", start)
        return text[start:end].strip()
    return text.strip()


# ── helpers ──────────────────────────────────────────────────────


def _safe_builtins() -> dict:
    """Provide a subset of builtins for generated code."""
    import builtins

    allowed = [
        "abs", "all", "any", "bool", "dict", "dir", "enumerate", "filter",
        "float", "format", "frozenset", "getattr", "hasattr", "hash", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max",
        "min", "next", "object", "pow", "print", "property", "range",
        "repr", "reversed", "round", "set", "slice", "sorted", "str",
        "sum", "tuple", "type", "zip", "True", "False", "None",
        "ValueError", "TypeError", "KeyError", "IndexError", "Exception",
        "RuntimeError", "StopIteration", "AttributeError", "ZeroDivisionError",
    ]
    safe = {k: getattr(builtins, k, None) for k in allowed}
    # Allow imports for numpy and common libs
    safe["__import__"] = _restricted_import
    return safe


def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Only allow importing a curated set of packages."""
    import builtins as _builtins

    allowed_modules = {
        "math", "statistics", "collections", "itertools", "functools",
        "re", "json", "datetime", "textwrap", "io",
        "numpy", "pandas", "plotly", "plotly.express", "plotly.graph_objects",
        "plotly.graph_objs", "plotly.subplots", "plotly.io",
        "streamlit", "streamlit.components", "streamlit.components.v1",
        "feature_engine",
    }
    if name not in allowed_modules:
        raise ImportError(f"Import of '{name}' is not allowed in generated code.")
    return _builtins.__import__(name, *args, **kwargs)
