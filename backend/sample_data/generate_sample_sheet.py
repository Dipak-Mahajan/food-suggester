"""Generates sample_orders.xlsx: a dummy single-restaurant order sheet with
intentionally different column names than our canonical schema, so the
upload flow's Gemini/heuristic column-mapping has something to map.

Shaped like a real POS "order details" export: one Order No per ticket,
with one row per item line on that ticket. No ratings, no customer
identity, no multi-restaurant column — just what a single restaurant can
realistically hand over.

Run once: venv/bin/python sample_data/generate_sample_sheet.py
"""
import os
import datetime
import numpy as np
import pandas as pd

from mock_data import FOOD_ITEMS

rng = np.random.RandomState(7)
today = datetime.date.today()

records = []
for order_no in range(1, 251):
    order_date = (today - datetime.timedelta(days=int(rng.randint(0, 180)))).isoformat()
    n_items = int(rng.randint(1, 5))
    item_idxs = rng.choice(len(FOOD_ITEMS), size=n_items, replace=False)
    for idx in item_idxs:
        item = FOOD_ITEMS[idx]
        records.append({
            "Order Date": order_date,
            "Order No": order_no,
            "Dish": item["name"],
            "Food Category": item["category"],
            "Qty": int(rng.randint(1, 4)),
        })

df = pd.DataFrame(records)
out_path = os.path.join(os.path.dirname(__file__), "sample_orders.xlsx")
df.to_excel(out_path, index=False, engine="openpyxl")
print(f"Wrote {len(df)} rows across {df['Order No'].nunique()} orders to {out_path}")
