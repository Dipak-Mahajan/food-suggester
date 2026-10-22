import os
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv

load_dotenv()

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


def generate_reason(selected_names: list[str], suggested_name: str, stars: float) -> str:
    model = _get_model()
    if not model:
        return _fallback_reason(selected_names, suggested_name, stars)

    selected_str = ", ".join(selected_names) if selected_names else "popular items"
    prompt = (
        f"A customer ordered: {selected_str}.\n"
        f"We're suggesting '{suggested_name}' (rated {stars}/5 stars based on sales).\n"
        f"Write one short, friendly sentence (max 20 words) explaining why '{suggested_name}' "
        f"pairs well with their order. Be specific and appetizing. No emojis."
    )
    try:
        response = model.generate_content(
            prompt,
            request_options=types.RequestOptions(timeout=5),
        )
        return response.text.strip()
    except Exception:
        return _fallback_reason(selected_names, suggested_name, stars)


def _fallback_reason(selected_names: list[str], suggested_name: str, stars: float) -> str:
    if not selected_names:
        return f"One of our most popular items with {stars}/5 stars."
    return f"Customers who ordered {selected_names[0]} frequently also loved {suggested_name}."
