import os
import json
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv

load_dotenv()

CANONICAL_FIELDS = [
    "order_date", "order_id",
    "item_name", "category", "quantity",
]

_HEURISTIC_KEYWORDS = {
    "order_date": ["date"],
    "order_id": ["order no", "order id", "bill", "ticket", "invoice", "receipt"],
    "item_name": ["item", "dish", "food", "product", "menu"],
    "category": ["category", "cuisine", "type"],
    "quantity": ["qty", "quantity", "count"],
}

_model = None


def _get_model():
    global _model
    if _model is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel("gemini-1.5-flash")
    return _model


def map_columns(columns: list[str]) -> dict[str, str]:
    """Map arbitrary spreadsheet column names to our canonical order schema.

    Restaurants won't all name columns the same way ("Dish" vs "Item Name"),
    so we ask Gemini to do the mapping and fall back to keyword matching if
    the API key is missing or the call fails.
    """
    model = _get_model()
    if model:
        try:
            mapping = _map_with_gemini(model, columns)
            if mapping:
                return mapping
        except Exception:
            pass
    return _map_with_heuristics(columns)


def _map_with_gemini(model, columns: list[str]) -> dict[str, str]:
    prompt = (
        "You are mapping a restaurant's order spreadsheet to a canonical schema.\n"
        f"Canonical fields: {CANONICAL_FIELDS}\n"
        f"Spreadsheet columns: {columns}\n"
        "Map each spreadsheet column to the single best-matching canonical field. "
        "Omit columns that don't match any canonical field. "
        "Respond with ONLY a JSON object (no markdown, no explanation) mapping "
        "original column name -> canonical field name."
    )
    response = model.generate_content(
        prompt,
        request_options=types.RequestOptions(timeout=10),
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    mapping = json.loads(text.strip())
    return {
        col: field for col, field in mapping.items()
        if col in columns and field in CANONICAL_FIELDS
    }


def _map_with_heuristics(columns: list[str]) -> dict[str, str]:
    mapping = {}
    used_fields = set()
    for col in columns:
        low = col.lower()
        for field, keywords in _HEURISTIC_KEYWORDS.items():
            if field in used_fields:
                continue
            if any(kw in low for kw in keywords):
                mapping[col] = field
                used_fields.add(field)
                break
    return mapping
