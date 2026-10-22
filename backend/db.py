"""SQLite persistence for multi-restaurant data: which restaurants exist,
their uploaded orders, and their sold-out flags. Free, zero-infra, file-based
— deliberately simple since the recommender itself stays in-memory (pandas)
and is rebuilt from these rows on demand.
"""
import os
import re
import datetime
import hashlib
import secrets
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "app.db")

ORDER_COLUMNS = ["order_id", "order_date", "item_name", "category", "quantity"]


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS restaurants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant_id INTEGER NOT NULL REFERENCES restaurants(id),
            order_id INTEGER,
            order_date TEXT,
            item_name TEXT NOT NULL,
            category TEXT,
            quantity INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_orders_restaurant ON orders(restaurant_id);

        CREATE TABLE IF NOT EXISTS sold_out_items (
            restaurant_id INTEGER NOT NULL REFERENCES restaurants(id),
            item_name TEXT NOT NULL,
            PRIMARY KEY (restaurant_id, item_name)
        );

        CREATE TABLE IF NOT EXISTS restaurant_credentials (
            restaurant_id INTEGER PRIMARY KEY REFERENCES restaurants(id),
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "restaurant"


def create_restaurant(name: str) -> dict:
    base_slug = slugify(name)
    slug = base_slug
    conn = get_connection()
    try:
        n = 1
        while conn.execute("SELECT 1 FROM restaurants WHERE slug = ?", (slug,)).fetchone():
            n += 1
            slug = f"{base_slug}-{n}"
        conn.execute(
            "INSERT INTO restaurants (slug, name, created_at) VALUES (?, ?, ?)",
            (slug, name, datetime.datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return get_restaurant_by_slug(slug)


def list_restaurants() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT id, slug, name FROM restaurants ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_restaurant_by_slug(slug: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT id, slug, name FROM restaurants WHERE slug = ?", (slug,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def replace_orders(restaurant_id: int, df: pd.DataFrame):
    """Upload replaces a restaurant's whole order history (matches the
    single-tenant app's existing upload semantics)."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM orders WHERE restaurant_id = ?", (restaurant_id,))
        conn.commit()
        if len(df):
            df = df.copy()
            df["restaurant_id"] = restaurant_id
            for col in ORDER_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            df[["restaurant_id", *ORDER_COLUMNS]].to_sql(
                "orders", conn, if_exists="append", index=False
            )
    finally:
        conn.close()


def load_orders(restaurant_id: int) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        f"SELECT {', '.join(ORDER_COLUMNS)} FROM orders WHERE restaurant_id = ?",
        conn, params=(restaurant_id,),
    )
    conn.close()
    return df


def set_sold_out(restaurant_id: int, item_name: str, sold_out: bool):
    conn = get_connection()
    try:
        if sold_out:
            conn.execute(
                "INSERT OR IGNORE INTO sold_out_items (restaurant_id, item_name) VALUES (?, ?)",
                (restaurant_id, item_name),
            )
        else:
            conn.execute(
                "DELETE FROM sold_out_items WHERE restaurant_id = ? AND item_name = ?",
                (restaurant_id, item_name),
            )
        conn.commit()
    finally:
        conn.close()


def get_sold_out_items(restaurant_id: int) -> set:
    conn = get_connection()
    rows = conn.execute(
        "SELECT item_name FROM sold_out_items WHERE restaurant_id = ?", (restaurant_id,)
    ).fetchall()
    conn.close()
    return {r["item_name"] for r in rows}


def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), 100_000
    ).hex()


def set_credentials(restaurant_id: int, username: str, password: str):
    """Staff login for this restaurant — required to upload orders or
    toggle sold-out items; browsing the menu never needs this."""
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO restaurant_credentials (restaurant_id, username, password_hash, salt)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(restaurant_id) DO UPDATE SET
                username=excluded.username,
                password_hash=excluded.password_hash,
                salt=excluded.salt
            """,
            (restaurant_id, username, password_hash, salt),
        )
        conn.commit()
    finally:
        conn.close()


def verify_credentials(restaurant_id: int, username: str, password: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT username, password_hash, salt FROM restaurant_credentials WHERE restaurant_id = ?",
        (restaurant_id,),
    ).fetchone()
    conn.close()
    if not row or row["username"] != username:
        return False
    return _hash_password(password, row["salt"]) == row["password_hash"]
