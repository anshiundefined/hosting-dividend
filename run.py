"""
The hosting dividend: do mega-events (Olympics, FIFA World Cup, Commonwealth Games) raise
the host's income or tourism?

For each host: (1) a classic synthetic control, (2) a machine-learning counterfactual via
a generalised synthetic control / interactive fixed-effects
factor model with cross-validated rank (Xu 2017), and (3) placebo-in-space inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls

import wdi
from utils import DATA, HIGHLIGHT, MUTED, PALETTE, md_table, savefig, setup, write_results

SRC = "World Bank WDI (GDP per capita, constant 2015 US$; international tourism arrivals)"
# host, event, event year, year awarded
EVENTS = [
    ("GRC", "Athens Olympics", 2004, 1997), ("KOR", "World Cup (co-host)", 2002, 1996),
    ("DEU", "World Cup", 2006, 2000), ("CHN", "Beijing Olympics", 2008, 2001),
    ("ZAF", "World Cup", 2010, 2004), ("IND", "Delhi Commonwealth Games", 2010, 2003),
    ("GBR", "London Olympics", 2012, 2005), ("BRA", "World Cup + Rio Olympics", 2014, 2007),
    ("RUS", "World Cup + Sochi Winter Olympics", 2014, 2007), ("JPN", "Tokyo Olympics", 2021, 2013),
    ("QAT", "World Cup", 2022, 2010),
]
# every country that hosted (or was awarded) a comparable event 1988-2034 is removed from the donor pool
EVER_HOSTS = {"GRC", "KOR", "DEU", "CHN", "ZAF", "IND", "GBR", "BRA", "RUS", "JPN", "QAT", "AUS", "ESP", "USA",
              "ITA", "FRA", "CAN", "MEX", "NOR", "SWE", "SAU", "MAR", "PRT", "URY", "ARG", "PRY", "MYS", "NZL",
              "SCO", "ARE", "BEL", "NLD", "CHE", "AUT", "UKR", "POL"}
PRE = 10     # pre-award years used to fit
POST = 8     # years after the event shown


def load() -> pd.DataFrame:
    df = wdi.panel({"gdppc": "NY.GDP.PCAP.KD", "arrivals": "ST.INT.ARVL", "pop": "SP.POP.TOTL"}, 1980, 2024)
    df = df[df.groupby("iso3")["pop"].transform("max") > 1e6]
    df["ln_gdppc"] = np.log(df["gdppc"])
    df["ln_arr"] = np.log(df["arrivals"].where(df["arrivals"] > 0))
    return df


def synth_weights(y_pre: np.ndarray, Y0_pre: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one that best reproduce the treated unit's pre-period path.
    Solved as NNLS with the adding-up constraint imposed through a heavily weighted extra row."""
    J = Y0_pre.shape[1]
    big = 1e3 * max(1.0, np.abs(Y0_pre).max())
    A = np.vstack([Y0_pre, big * np.ones((1, J))])
    b = np.concatenate([y_pre, [big]])
    w, _ = nnls(A, b, maxiter=50 * J)
    return w / w.sum() if w.sum() > 0 else np.full(J, 1 / J)


def factor_counterfactual(panel: pd.DataFrame, host: str, first_post: int) -> tuple[pd.Series, int]:
    """Generalised synthetic control / interactive fixed effects (Xu 2017).

    Y_it = u_i + v_t + lambda_i' f_t + e_it. Time effects and latent factors f_t are learned from
    donors only (SVD); the host's loadings are fitted on its pre-period; the number of factors k
    is chosen by leave-one-year-out cross-validation on the host's pre-period.
    """
    Y = panel.values.astype(float)
    hi = panel.index.get_loc(host)
    post = np.asarray(panel.columns >= first_post)
    D = np.delete(Y, hi, axis=0)
    v = D.mean(axis=0)
    Rd = D - v - (D - v).mean(axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(Rd, full_matrices=False)
    yh = Y[hi] - v
    pre_idx = np.where(~post)[0]

    def fit_predict(k: int, train: np.ndarray) -> np.ndarray:
        F = np.column_stack([np.ones(len(v))] + [Vt[j] for j in range(k)])   # intercept = unit effect
        beta, *_ = np.linalg.lstsq(F[train], yh[train], rcond=None)
        return F @ beta

    kmax = max(0, min(5, len(pre_idx) - 3))
    cv = {}
    for k in range(kmax + 1):
        errs = [(fit_predict(k, np.setdiff1d(pre_idx, [t]))[t] - yh[t]) ** 2 for t in pre_idx]
        cv[k] = np.mean(errs)
    k = min(cv, key=cv.get)
    return pd.Series(v + fit_predict(k, pre_idx), index=panel.columns), k


def run_event(df: pd.DataFrame, var: str, host: str, event_year: int, award: int, rng, pre_len: int = PRE) -> dict | None:
    last = int(df.dropna(subset=[var])["year"].max())
    y0, y1 = award - pre_len, min(event_year + POST, last)
    if y1 <= event_year:
        return None
    wide = df.pivot(index="iso3", columns="year", values=var).loc[:, y0:y1]
    wide = wide.interpolate(axis=1, limit=2, limit_area="inside")          # fill short reporting gaps
    if host not in wide.index or wide.loc[host].isna().any():
        return None
    donors = wide.drop(index=[i for i in wide.index if i in EVER_HOSTS]).dropna()
    if len(donors) < 15:
        return None
    # anchor every series at its award-year value so we compare growth paths
    panel = pd.concat([wide.loc[[host]], donors])
    panel = panel.sub(panel[award], axis=0)
    pre = panel.columns < award
    w = synth_weights(panel.loc[host, pre].values, panel.loc[donors.index, pre].values.T)
    sc = pd.Series(panel.loc[donors.index].values.T @ w, index=panel.columns)
    mc, k = factor_counterfactual(panel, host, award)
    actual = panel.loc[host]

    # placebo-in-space: pretend each donor hosted; compare post/pre RMSPE ratios
    def ratio(gap):
        return np.sqrt(np.mean(gap[~pre] ** 2)) / max(np.sqrt(np.mean(gap[pre] ** 2)), 1e-6)
    host_ratio = ratio((actual - sc).values)
    ratios = []
    for dn in donors.index:
        others = donors.index.drop(dn)
        wp = synth_weights(panel.loc[dn, pre].values, panel.loc[others, pre].values.T)
        ratios.append(ratio(panel.loc[dn].values - panel.loc[others].values.T @ wp))
    p = (1 + np.sum(np.array(ratios) >= host_ratio)) / (1 + len(ratios))
    top = pd.Series(w, index=donors.index).sort_values(ascending=False).head(4)
    return {"actual": actual, "sc": sc, "mc": mc, "p": p, "weights": top, "pre_rmse": np.sqrt(np.mean((actual - sc)[pre] ** 2)), "k": k}


def main() -> None:
    setup()
    rng = np.random.default_rng(0)
    df = load()
    names = df.drop_duplicates("iso3").set_index("iso3")["country"]
    md, rows, gaps = [], [], []

    for var, label, pre_len in [("ln_gdppc", "GDP per capita", PRE), ("ln_arr", "Tourist arrivals", 6)]:
        results = {}
        for host, event, ey, award in EVENTS:
            r = run_event(df, var, host, ey, award, rng, pre_len)
            if r is None:
                continue
            results[host] = (event, ey, award, r)
            post = r["actual"].index >= ey
            gap_sc = 100 * (np.exp((r["actual"] - r["sc"])[post]) - 1).mean()
            gap_mc = 100 * (np.exp((r["actual"] - r["mc"])[post]) - 1).mean()
            rows.append({"Outcome": label, "Host": names.get(host, host), "Event": f"{event} {ey}",
                         "Avg effect, synth control (%)": gap_sc, "Avg effect, factor model (%)": gap_mc,
                         "Placebo p-value": r["p"], "Factors (CV)": r["k"], "Pre-fit RMSE": r["pre_rmse"],
                         "Top donors": ", ".join(f"{names.get(i, i)} ({v:.2f})" for i, v in r["weights"].items() if v > 0.02)})
            for yr in r["actual"].index:
                gaps.append({"Outcome": label, "host": host, "rel": yr - ey,
                             "sc": 100 * (np.exp(r["actual"][yr] - r["sc"][yr]) - 1),
                             "mc": 100 * (np.exp(r["actual"][yr] - r["mc"][yr]) - 1)})
        if not results:
            continue
        n = len(results)
        cols = 4
        fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(15, 3.3 * np.ceil(n / cols)), squeeze=False)
        for ax, (host, (event, ey, award, r)) in zip(axes.flat, results.items()):
            idx = r["actual"].index
            ax.plot(idx, 100 * (np.exp(r["actual"]) - 1), color=HIGHLIGHT if host == "IND" else "#111", lw=2.2, label="Actual")
            ax.plot(idx, 100 * (np.exp(r["sc"]) - 1), color=PALETTE[0], lw=1.8, ls="--", label="Synthetic control")
            ax.plot(idx, 100 * (np.exp(r["mc"]) - 1), color=PALETTE[1], lw=1.8, ls=":", label="Factor model (GSC)")
            ax.axvline(award, color=MUTED, lw=1)
            ax.axvline(ey, color="#555", lw=1, ls="--")
            ax.set_title(f"{names.get(host, host)}: {event}\n(placebo p = {r['p']:.2f})", fontsize=9.5)
            ax.tick_params(labelsize=8)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7.5)
        fig.suptitle(f"{label}: host vs counterfactual (% change since award year; grey = award, dashed = event)",
                     fontweight="bold", y=1.0)
        fig.tight_layout()
        rows_fig = savefig(fig, f"0{1 if var == 'ln_gdppc' else 2}_{var}_by_host", SRC)
        md.append(f"**{label} by host**\n\n![{label}]({rows_fig})\n")

    tab = pd.DataFrame(rows)
    G = pd.DataFrame(gaps)
    ev = G[(G["rel"] >= -10) & (G["rel"] <= POST)].groupby(["Outcome", "rel"])[["sc", "mc"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, label in zip(axes, ["GDP per capita", "Tourist arrivals"]):
        e = ev[ev["Outcome"] == label]
        if e.empty:
            ax.axis("off"); continue
        ax.plot(e["rel"], e["sc"], color=PALETTE[0], lw=2.4, marker="o", ms=3, label="Synthetic control")
        ax.plot(e["rel"], e["mc"], color=PALETTE[1], lw=2.4, marker="s", ms=3, label="Factor model (GSC)")
        ax.axhline(0, color="#333", lw=1); ax.axvline(0, color="#555", ls="--", lw=1)
        ax.set_xlabel("Years relative to the event")
        ax.set_ylabel("Host minus counterfactual (%)")
        ax.set_title(f"{label}: average across hosts")
        ax.legend()
    ev_fig = savefig(fig, "03_event_study_average", SRC)

    head = ["### Headline numbers\n"]
    for label in ["GDP per capita", "Tourist arrivals"]:
        t = tab[tab["Outcome"] == label]
        if t.empty:
            continue
        sig = int((t["Placebo p-value"] <= 0.10).sum())
        head.append(f"- **{label}:** average post-event effect across {len(t)} hosts is "
                    f"**{t['Avg effect, synth control (%)'].mean():+.1f}%** (synthetic control) and "
                    f"**{t['Avg effect, factor model (%)'].mean():+.1f}%** (factor model). "
                    f"{sig} of {len(t)} hosts show an effect larger than 90% of placebo countries.")
    ind = tab[(tab["Host"].str.contains("India")) & (tab["Outcome"] == "GDP per capita")]
    if not ind.empty:
        r = ind.iloc[0]
        head.append(f"- **India (Delhi 2010):** {r['Avg effect, synth control (%)']:+.1f}% vs synthetic India "
                    f"(placebo p = {r['Placebo p-value']:.2f}).")
    head.append("\n### Host-by-host estimates\n")
    head.append(md_table(tab, "{:.2f}"))
    head.append("\n### Figures\n")
    head.append(f"**Average event study**\n\n![Average event study]({ev_fig})\n")
    tab.to_csv(DATA / "estimates.csv", index=False)
    write_results("\n".join(head + md))
    print("done")


if __name__ == "__main__":
    main()
