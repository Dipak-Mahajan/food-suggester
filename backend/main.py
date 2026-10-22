from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from recommender import FoodRecommender, REQUIRED_COLUMNS
from ai_reason import generate_reason
from sheet_mapper import map_columns
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
import pandas as pd

app = FastAPI(title="Food Suggestor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = FoodRecommender()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


class SuggestRequest(BaseModel):
    selected_item_ids: list[int] = []
    top_n: int = 6


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/items")
def get_all_items():
    return recommender.all_items_with_stats()


@app.get("/api/trends")
def get_trends(window: str = "month", top_n: int = 10):
    if window not in {"week", "month", "6months"}:
        raise HTTPException(400, "window must be one of: week, month, 6months")
    return {
        "window": window,
        "has_dates": recommender.has_dates,
        "items": recommender.get_trends(window, top_n),
    }


@app.post("/api/upload-orders")
async def upload_orders(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Please upload an Excel file (.xlsx or .xls)")

    contents = await file.read()
    try:
        raw_df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not read Excel file: {e}")

    mapping = map_columns(list(raw_df.columns))
    df = raw_df.rename(columns=mapping)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            400,
            f"Could not find a column for: {missing}. "
            f"Detected mapping: {mapping}",
        )

    global recommender
    recommender = FoodRecommender(df)

    return {
        "status": "ok",
        "rows": len(df),
        "distinct_items": df["item_name"].nunique(),
        "distinct_orders": df["order_id"].nunique() if "order_id" in df.columns else None,
        "column_mapping": mapping,
    }


@app.post("/api/suggest")
def suggest(req: SuggestRequest):
    if req.selected_item_ids:
        suggestions = recommender.get_suggestions(req.selected_item_ids, req.top_n)
        selected_names = [
            recommender.food_map[i]["name"]
            for i in req.selected_item_ids
            if i in recommender.food_map
        ]
    else:
        suggestions = recommender.get_popular_items(req.top_n)
        selected_names = []

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {
            ex.submit(generate_reason, selected_names, s["name"], s["stars"]): i
            for i, s in enumerate(suggestions)
        }
        for future in as_completed(futures):
            suggestions[futures[future]]["reason"] = future.result()

    return {"suggestions": suggestions}
