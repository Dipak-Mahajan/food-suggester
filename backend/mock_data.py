import pandas as pd
import numpy as np
import datetime

FOOD_ITEMS = [
    {"id": 1, "name": "Margherita Pizza",   "category": "Pizza"},
    {"id": 2, "name": "Pepperoni Pizza",    "category": "Pizza"},
    {"id": 3, "name": "BBQ Chicken Pizza",  "category": "Pizza"},
    {"id": 4, "name": "Veggie Burger",      "category": "Burger"},
    {"id": 5, "name": "Classic Cheeseburger","category": "Burger"},
    {"id": 6, "name": "Garlic Bread",       "category": "Sides"},
    {"id": 7, "name": "Caesar Salad",       "category": "Salad"},
    {"id": 8, "name": "Chicken Wings",      "category": "Sides"},
    {"id": 9, "name": "Pasta Carbonara",    "category": "Pasta"},
    {"id": 10,"name": "Mac & Cheese",       "category": "Pasta"},
    {"id": 11,"name": "Sushi Platter",      "category": "Japanese"},
    {"id": 12,"name": "Miso Soup",          "category": "Japanese"},
    {"id": 13,"name": "Tacos",              "category": "Mexican"},
    {"id": 14,"name": "Nachos",             "category": "Mexican"},
    {"id": 15,"name": "Cheesecake",         "category": "Dessert"},
    {"id": 16,"name": "Chocolate Lava Cake","category": "Dessert"},
    {"id": 17,"name": "Iced Coffee",        "category": "Drinks"},
    {"id": 18,"name": "Fresh Lemonade",     "category": "Drinks"},
]

# Co-purchase patterns: pairs frequently bought together
CO_PURCHASE_PATTERNS = [
    ([1, 6, 17], 0.8),   # Margherita + Garlic Bread + Iced Coffee
    ([2, 8, 17], 0.75),  # Pepperoni + Wings + Iced Coffee
    ([3, 8, 18], 0.7),   # BBQ Chicken + Wings + Lemonade
    ([4, 7, 18], 0.65),  # Veggie Burger + Salad + Lemonade
    ([5, 6, 17], 0.72),  # Cheeseburger + Garlic Bread + Iced Coffee
    ([9, 7, 17], 0.6),   # Carbonara + Salad + Coffee
    ([10, 15, 17], 0.68),# Mac n Cheese + Cheesecake + Coffee
    ([11, 12, 18], 0.8), # Sushi + Miso + Lemonade
    ([13, 14, 18], 0.78),# Tacos + Nachos + Lemonade
    ([15, 16, 17], 0.7), # Cheesecake + Lava Cake + Coffee
    ([1, 2, 6], 0.5),    # Pizza lovers
    ([4, 13, 7], 0.55),  # Healthy options
]


def generate_sales_data(n_orders=500, seed=42) -> pd.DataFrame:
    """Fallback demo dataset, shaped like a restaurant's uploaded order sheet.

    One `order_id` = one ticket/bill, which may have several item lines —
    that's the join key for "what's ordered together," not a customer identity.
    """
    rng = np.random.RandomState(seed)
    id_to_item = {item["id"]: item for item in FOOD_ITEMS}
    today = datetime.date.today()
    records = []
    for order_id in range(1, n_orders + 1):
        chosen = rng.choice(len(CO_PURCHASE_PATTERNS),
                             size=rng.randint(1, 4), replace=False)
        ordered = set()
        for idx in chosen:
            items, prob = CO_PURCHASE_PATTERNS[idx]
            for item_id in items:
                if rng.random() < prob:
                    ordered.add(item_id)
        extras = rng.choice(len(FOOD_ITEMS), size=rng.randint(0, 3), replace=False)
        for e in extras:
            ordered.add(FOOD_ITEMS[e]["id"])

        for item_id in ordered:
            item = id_to_item[item_id]
            order_date = today - datetime.timedelta(days=int(rng.randint(0, 180)))
            records.append({
                "order_date": order_date.isoformat(),
                "order_id": order_id,
                "item_name": item["name"],
                "category": item["category"],
                "quantity": int(rng.randint(1, 4)),
            })
    return pd.DataFrame(records)
