"""Plot gallery — persist generated plots with metadata."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

GALLERY_DIR = Path(__file__).parent / "gallery_store"
INDEX_FILE = GALLERY_DIR / "_index.json"


def _ensure_dir() -> None:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text("[]", encoding="utf-8")


def load_index() -> list[dict[str, Any]]:
    _ensure_dir()
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def save_index(index: list[dict[str, Any]]) -> None:
    _ensure_dir()
    INDEX_FILE.write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")


def save_plot(
    *,
    title: str,
    description: str,
    code: str,
    plot_html: str,
    user_query: str,
    column_context: str = "",
) -> str:
    """Save a plot entry and return its id."""
    _ensure_dir()
    entry_id = f"plot_{int(time.time() * 1000)}"

    # Save the HTML rendering
    html_path = GALLERY_DIR / f"{entry_id}.html"
    html_path.write_text(plot_html, encoding="utf-8")

    # Save the code
    code_path = GALLERY_DIR / f"{entry_id}.py"
    code_path.write_text(code, encoding="utf-8")

    entry = {
        "id": entry_id,
        "title": title,
        "description": description,
        "user_query": user_query,
        "column_context": column_context,
        "code_file": str(code_path.name),
        "html_file": str(html_path.name),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    index = load_index()
    index.insert(0, entry)  # newest first
    save_index(index)
    return entry_id


def get_plot_entry(entry_id: str) -> dict[str, Any] | None:
    for e in load_index():
        if e["id"] == entry_id:
            return e
    return None


def get_plot_html(entry_id: str) -> str:
    path = GALLERY_DIR / f"{entry_id}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def get_plot_code(entry_id: str) -> str:
    path = GALLERY_DIR / f"{entry_id}.py"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def delete_plot(entry_id: str) -> None:
    index = load_index()
    index = [e for e in index if e["id"] != entry_id]
    save_index(index)
    for ext in (".html", ".py"):
        p = GALLERY_DIR / f"{entry_id}{ext}"
        if p.exists():
            p.unlink()
