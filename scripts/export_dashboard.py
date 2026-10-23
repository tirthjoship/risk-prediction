"""Aggregate the raw DataCo file into the small cube behind the Overview tab.

Line items are grouped to order level first (one row per "Order Id"), then
counted by month, market, shipping mode and customer segment. The cube holds
counts only: no customer, order or product identifiers and no raw rows.

Usage:
    python scripts/export_dashboard.py [path/to/DataCoSupplyChainDataset.csv]
    DATACO_RAW_CSV=path/to/file.csv python scripts/export_dashboard.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "dashboard" / "order_cube.csv"

USE_COLUMNS = [
    "Order Id",
    "order date (DateOrders)",
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Late_delivery_risk",
]


def build_order_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse line items to one row per order."""
    frame = raw.copy()
    frame["order_date"] = pd.to_datetime(
        frame["order date (DateOrders)"], format="%m/%d/%Y %H:%M"
    )
    grouped = frame.groupby("Order Id", sort=True).agg(
        order_date=("order_date", "min"),
        shipping_mode=("Shipping Mode", "first"),
        market=("Market", "first"),
        customer_segment=("Customer Segment", "first"),
        late=("Late_delivery_risk", "max"),
    )
    return grouped.reset_index(drop=True)


def build_cube(orders: pd.DataFrame) -> pd.DataFrame:
    """Count orders and late orders by month, market, mode and segment."""
    frame = orders.copy()
    frame["month"] = frame["order_date"].dt.strftime("%Y-%m")
    frame["year"] = frame["order_date"].dt.year
    cube = (
        frame.groupby(
            ["month", "year", "market", "shipping_mode", "customer_segment"],
            as_index=False,
        )
        .agg(orders=("late", "size"), late_orders=("late", "sum"))
        .sort_values(["month", "market", "shipping_mode", "customer_segment"])
        .reset_index(drop=True)
    )
    cube["late_orders"] = cube["late_orders"].astype(int)
    return cube


def export(raw_path: Path, out_path: Path) -> pd.DataFrame:
    """Read the raw CSV, build the cube and write it."""
    raw = pd.read_csv(raw_path, encoding="latin-1", usecols=USE_COLUMNS)
    cube = build_cube(build_order_table(raw))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cube.to_csv(out_path, index=False)
    return cube


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", nargs="?", help="path to the raw DataCo CSV")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output cube path")
    args = parser.parse_args()
    raw_path = Path(args.raw or os.environ.get("DATACO_RAW_CSV", DEFAULT_RAW))
    cube = export(raw_path, Path(args.out))
    print(
        f"Wrote {len(cube):,} rows, {int(cube['orders'].sum()):,} orders, "
        f"{int(cube['late_orders'].sum()):,} late, to {args.out}"
    )


if __name__ == "__main__":
    main()
