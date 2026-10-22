import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from mock_data import generate_sales_data

CATEGORY_EMOJI = {
    "Pizza": "🍕", "Burger": "🍔", "Sides": "🍟", "Salad": "🥗",
    "Pasta": "🍝", "Japanese": "🍱", "Mexican": "🌮", "Dessert": "🍰",
    "Drinks": "🥤",
}
DEFAULT_EMOJI = "🍽️"

REQUIRED_COLUMNS = ["item_name"]

WINDOW_DAYS = {"week": 7, "month": 30, "6months": 182}


class FoodRecommender:
    """Builds its item catalog and similarity model from an orders DataFrame.

    Expected (canonical) columns: order_date, order_id, item_name, category,
    quantity. Only item_name is required; the rest degrade gracefully when
    missing. `order_id` identifies one ticket/bill (which items were ordered
    together) — not a customer identity, which most restaurants can't
    reliably provide for walk-in customers anyway. Star ratings are always
    derived from order frequency, never uploaded. One instance = one
    restaurant's data; there's no multi-restaurant concept.
    """

    def __init__(self, orders_df: pd.DataFrame = None):
        self.orders_df = orders_df if orders_df is not None else generate_sales_data()
        if "order_id" not in self.orders_df.columns:
            # Back-compat with older sheets that used customer_id as the basket key.
            if "customer_id" in self.orders_df.columns:
                self.orders_df = self.orders_df.rename(columns={"customer_id": "order_id"})
            else:
                self.orders_df = self.orders_df.assign(
                    order_id=range(1, len(self.orders_df) + 1)
                )
        self._build_catalog()
        self._build_matrix()

    def _build_catalog(self):
        df = self.orders_df
        if "category" not in df.columns:
            df = df.assign(category="Uncategorized")
        items = (
            df[["item_name", "category"]]
            .fillna({"category": "Uncategorized"})
            .drop_duplicates(subset="item_name")
            .reset_index(drop=True)
        )
        items["id"] = items.index + 1
        items["emoji"] = items["category"].map(CATEGORY_EMOJI).fillna(DEFAULT_EMOJI)

        self.food_map = {
            int(row.id): {
                "id": int(row.id),
                "name": row.item_name,
                "category": row.category,
                "emoji": row.emoji,
            }
            for row in items.itertuples()
        }
        name_to_id = {row.item_name: int(row.id) for row in items.itertuples()}

        self.orders_df = df.copy()
        self.orders_df["item_id"] = self.orders_df["item_name"].map(name_to_id)

        if "order_date" in self.orders_df.columns:
            self.orders_df["_parsed_date"] = pd.to_datetime(
                self.orders_df["order_date"], errors="coerce"
            )
            self.has_dates = bool(self.orders_df["_parsed_date"].notna().any())
        else:
            self.has_dates = False

    def _build_matrix(self):
        self.matrix = self.orders_df.pivot_table(
            index="order_id", columns="item_id", aggfunc="size", fill_value=0
        )
        item_vectors = self.matrix.T.values
        self.similarity = pd.DataFrame(
            cosine_similarity(item_vectors),
            index=self.matrix.columns,
            columns=self.matrix.columns,
        )

    def _order_count(self, df: pd.DataFrame) -> pd.Series:
        """Orders per item_id, weighted by quantity when available (a row
        with quantity=3 counts as 3 orders, not 1 — this is the single
        source of both 'popular' ranking and the derived star rating)."""
        if "quantity" in df.columns:
            return df.groupby("item_id")["quantity"].sum()
        return df.groupby("item_id").size()

    def get_item_stats(self, item_id: int) -> dict:
        """Purchase count and a star rating derived purely from how often
        the item is ordered relative to the busiest item — no uploaded
        rating column needed."""
        rows = self.orders_df[self.orders_df["item_id"] == item_id]
        count = int(self._order_count(rows).sum()) if len(rows) else 0

        all_counts = self._order_count(self.orders_df)
        busiest = int(all_counts.max()) if len(all_counts) else 0
        popularity = count / busiest if busiest else 0
        stars = round(1 + 4 * popularity, 1)
        return {"purchase_count": count, "stars": stars}

    def get_popular_items(self, top_n: int = 6) -> list[dict]:
        """Return globally most popular items (highest order volume)."""
        counts = self._order_count(self.orders_df).sort_values(ascending=False)
        result = []
        for item_id in counts.index[:top_n]:
            stats = self.get_item_stats(item_id)
            food = self.food_map[item_id]
            result.append({**food, **stats, "reason": None})
        return result

    def get_suggestions(self, selected_item_ids: list[int], top_n: int = 6) -> list[dict]:
        """Given selected items, suggest similar ones using item-item CF."""
        valid_ids = [i for i in selected_item_ids if i in self.similarity.columns]
        if not valid_ids:
            return self.get_popular_items(top_n)

        # Average similarity across all selected items
        scores = self.similarity[valid_ids].mean(axis=1)
        # Exclude already-selected items
        scores = scores.drop(index=[i for i in valid_ids if i in scores.index], errors="ignore")
        top_ids = scores.nlargest(top_n).index.tolist()

        result = []
        for item_id in top_ids:
            stats = self.get_item_stats(item_id)
            food = self.food_map[item_id]
            sim_score = float(scores[item_id])
            result.append({**food, **stats, "similarity": round(sim_score, 3), "reason": None})
        return result

    def all_items_with_stats(self) -> list[dict]:
        result = []
        for food in self.food_map.values():
            stats = self.get_item_stats(food["id"])
            result.append({**food, **stats})
        return result

    def _period_df(self, window: str, periods_ago: int) -> pd.DataFrame:
        """Orders within one window-length period, `periods_ago` periods back."""
        if not self.has_dates:
            return self.orders_df if periods_ago == 0 else self.orders_df.iloc[0:0]
        days = WINDOW_DAYS.get(window, 30)
        end = datetime.datetime.now() - datetime.timedelta(days=days * periods_ago)
        start = end - datetime.timedelta(days=days)
        dates = self.orders_df["_parsed_date"]
        return self.orders_df[(dates > start) & (dates <= end)]

    def get_trends(self, window: str = "month", top_n: int = 10) -> list[dict]:
        """Top items for a time window, with % change vs the prior equivalent
        period (e.g. this week vs last week) — the "is this trending" signal."""
        current_df = self._period_df(window, periods_ago=0)
        previous_df = self._period_df(window, periods_ago=1)

        current_counts = self._order_count(current_df).sort_values(ascending=False)
        previous_counts = self._order_count(previous_df)

        result = []
        for item_id, count in current_counts.head(top_n).items():
            food = self.food_map.get(int(item_id))
            if not food:
                continue
            prev_count = int(previous_counts.get(item_id, 0))
            change_pct = round((count - prev_count) / prev_count * 100, 1) if prev_count else None

            result.append({
                **food,
                "order_count": int(count),
                "previous_count": prev_count,
                "change_pct": change_pct,
            })
        return result
