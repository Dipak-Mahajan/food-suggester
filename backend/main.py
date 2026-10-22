from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from recommender import FoodRecommender, REQUIRED_COLUMNS
from ai_reason import generate_reason
from sheet_mapper import map_columns
from mock_data import generate_sales_data
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
import secrets
import pandas as pd
import db

app = FastAPI(title="Food Suggestor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

db.init_db()

# One FoodRecommender (pandas + similarity matrix) per restaurant, built
# lazily from the DB and cached by restaurant id — rebuilding is cheap at
# demo scale (a few MB, sub-second) so there's no need for LRU eviction yet.
_recommender_cache: dict[int, FoodRecommender] = {}

DEMO_RESTAURANTS = [
    {"name": "Spice Garden", "seed": 42, "n_orders": 5000, "username": "admin", "password": "spicegarden123"},
    {"name": "Urban Bites", "seed": 99, "n_orders": 4500, "username": "admin", "password": "urbanbites123"},
]


def _seed_demo_restaurants():
    if db.list_restaurants():
        return
    for spec in DEMO_RESTAURANTS:
        restaurant = db.create_restaurant(spec["name"])
        orders = generate_sales_data(n_orders=spec["n_orders"], seed=spec["seed"])
        db.replace_orders(restaurant["id"], orders)
        db.set_credentials(restaurant["id"], spec["username"], spec["password"])


_seed_demo_restaurants()


def _get_restaurant_or_404(slug: str) -> dict:
    restaurant = db.get_restaurant_by_slug(slug)
    if not restaurant:
        raise HTTPException(404, f"Unknown restaurant: {slug}")
    return restaurant


def _get_recommender(restaurant: dict) -> FoodRecommender:
    rid = restaurant["id"]
    if rid not in _recommender_cache:
        _recommender_cache[rid] = FoodRecommender(db.load_orders(rid))
    return _recommender_cache[rid]


# In-memory session tokens: viewing the menu never needs auth, but uploading
# or changing sold-out status does, scoped to that one restaurant's login.
_sessions: dict[str, int] = {}


def require_auth(slug: str, authorization: str = Header(None)) -> dict:
    restaurant = _get_restaurant_or_404(slug)
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token or _sessions.get(token) != restaurant["id"]:
        raise HTTPException(401, "Staff login required for this restaurant")
    return restaurant


class SuggestRequest(BaseModel):
    selected_item_ids: list[int] = []
    top_n: int = 6


class SoldOutRequest(BaseModel):
    item_name: str
    sold_out: bool


class CreateRestaurantRequest(BaseModel):
    name: str
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/restaurants")
def list_restaurants():
    return db.list_restaurants()


@app.post("/api/restaurants")
def create_restaurant(req: CreateRestaurantRequest):
    name = req.name.strip()
    username = req.username.strip()
    if not name or not username or not req.password:
        raise HTTPException(400, "Restaurant name, username, and password are required")
    restaurant = db.create_restaurant(name)
    db.set_credentials(restaurant["id"], username, req.password)
    return restaurant


@app.post("/api/{slug}/login")
def login(slug: str, req: LoginRequest):
    restaurant = _get_restaurant_or_404(slug)
    if not db.verify_credentials(restaurant["id"], req.username, req.password):
        raise HTTPException(401, "Invalid username or password")
    token = secrets.token_urlsafe(32)
    _sessions[token] = restaurant["id"]
    return {"token": token}


@app.get("/api/{slug}/items")
def get_all_items(slug: str):
    restaurant = _get_restaurant_or_404(slug)
    sold_out = db.get_sold_out_items(restaurant["id"])
    items = _get_recommender(restaurant).all_items_with_stats()
    for item in items:
        item["sold_out"] = item["name"] in sold_out
    return items


@app.post("/api/{slug}/sold-out")
def set_sold_out(slug: str, req: SoldOutRequest, restaurant: dict = Depends(require_auth)):
    db.set_sold_out(restaurant["id"], req.item_name, req.sold_out)
    return {"status": "ok", "sold_out_items": sorted(db.get_sold_out_items(restaurant["id"]))}


@app.get("/api/{slug}/trends")
def get_trends(slug: str, window: str = "month", top_n: int = 10):
    if window not in {"week", "month", "6months"}:
        raise HTTPException(400, "window must be one of: week, month, 6months")
    restaurant = _get_restaurant_or_404(slug)
    recommender = _get_recommender(restaurant)
    return {
        "window": window,
        "has_dates": recommender.has_dates,
        "items": recommender.get_trends(window, top_n),
    }


@app.get("/api/{slug}/combos")
def get_combos(slug: str, top_n: int = 8):
    restaurant = _get_restaurant_or_404(slug)
    sold_out = db.get_sold_out_items(restaurant["id"])
    return {"combos": _get_recommender(restaurant).get_frequent_pairs(top_n, exclude_names=sold_out)}


@app.post("/api/{slug}/upload-orders")
async def upload_orders(slug: str, file: UploadFile = File(...), restaurant: dict = Depends(require_auth)):
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

    db.replace_orders(restaurant["id"], df)
    _recommender_cache.pop(restaurant["id"], None)  # rebuild from DB on next request

    return {
        "status": "ok",
        "rows": len(df),
        "distinct_items": df["item_name"].nunique(),
        "distinct_orders": df["order_id"].nunique() if "order_id" in df.columns else None,
        "column_mapping": mapping,
    }


@app.post("/api/{slug}/suggest")
def suggest(slug: str, req: SuggestRequest):
    restaurant = _get_restaurant_or_404(slug)
    recommender = _get_recommender(restaurant)
    sold_out = db.get_sold_out_items(restaurant["id"])

    if req.selected_item_ids:
        suggestions = recommender.get_suggestions(req.selected_item_ids, req.top_n, exclude_names=sold_out)
        selected_names = [
            recommender.food_map[i]["name"]
            for i in req.selected_item_ids
            if i in recommender.food_map
        ]
    else:
        suggestions = recommender.get_popular_items(req.top_n, exclude_names=sold_out)
        selected_names = []

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {
            ex.submit(generate_reason, selected_names, s["name"], s["stars"]): i
            for i, s in enumerate(suggestions)
        }
        for future in as_completed(futures):
            suggestions[futures[future]]["reason"] = future.result()

    return {"suggestions": suggestions}
