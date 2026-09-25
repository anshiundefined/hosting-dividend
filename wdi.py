"""Minimal World Bank WDI client (API v2, no key needed). Caches raw pulls in data/."""
from __future__ import annotations

import pandas as pd

from utils import DATA, get

BASE = "https://api.worldbank.org/v2"


def countries() -> pd.DataFrame:
    """Economies only (aggregates such as 'World' or 'South Asia' removed)."""
    js = get(f"{BASE}/country?format=json&per_page=500").json()
    rows = []
    for c in js[1]:
        if c["region"]["value"].strip() == "Aggregates":
            continue
        rows.append({"iso3": c["id"], "country": c["name"], "region": c["region"]["value"].strip(),
                     "income": c["incomeLevel"]["value"].strip()})
    return pd.DataFrame(rows)


def indicator(code: str, start: int, end: int) -> pd.DataFrame:
    url = f"{BASE}/country/all/indicator/{code}?format=json&per_page=20000&date={start}:{end}"
    page, pages, rows = 1, 1, []
    while page <= pages:
        js = get(f"{url}&page={page}").json()
        meta, data = js[0], js[1] or []
        pages = int(meta.get("pages", 1))
        for r in data:
            if r.get("countryiso3code"):
                rows.append({"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]})
        page += 1
    return pd.DataFrame(rows, columns=["iso3", "year", "value"])


def panel(codes: dict[str, str], start: int, end: int, cache: str = "wdi_panel.csv") -> pd.DataFrame:
    """Return a tidy country-year panel with one column per indicator (named by the dict keys)."""
    meta = countries()
    df = None
    for name, code in codes.items():
        s = indicator(code, start, end).rename(columns={"value": name})
        df = s if df is None else df.merge(s, on=["iso3", "year"], how="outer")
    df = meta.merge(df, on="iso3", how="inner").sort_values(["iso3", "year"]).reset_index(drop=True)
    df.to_csv(DATA / cache, index=False)
    return df
