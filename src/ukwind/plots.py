"""Figures saved to results/figures (PNG, 150 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"actual": "#1f1f1f", "naive_D-2": "#b8b8b8", "naive_D-7": "#8c8c8c",
          "LEAR": "#2b6cb0", "LightGBM": "#d9822b", "physical": "#2b6cb0",
          "physical+ML": "#d9822b", "observed": "#1f1f1f"}
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def _save(fig, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / name)
    plt.close(fig)


def _pick_week(y: pd.Series) -> pd.Timestamp:
    """The test week with the largest price range (the most informative to look at)."""
    rng = y.resample("7D").agg(lambda s: s.max() - s.min())
    return rng.idxmax()


def price_week(y, preds, folder):
    start = _pick_week(y)
    sl = slice(start, start + pd.Timedelta(days=7))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(y[sl], color=COLORS["actual"], lw=1.6, label="actual")
    for k in ("naive_D-7", "LEAR", "LightGBM"):
        if k in preds:
            ax.plot(preds[k][sl], color=COLORS[k], lw=1.1, label=k)
    ax.set_ylabel("GBP/MWh")
    ax.set_title("Day-ahead price forecasts, most volatile test week")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    _save(fig, folder, "price_week.png")


def mae_by_hour(y, preds, folder, tz):
    hour = y.index.tz_convert(tz).hour
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for k, p in preds.items():
        e = (y - p).abs()
        ax.plot(e.groupby(hour).mean(), marker="o", ms=3, color=COLORS.get(k), label=k)
    ax.set_xlabel("Hour (UK local)")
    ax.set_ylabel("MAE, GBP/MWh")
    ax.set_title("Price forecast error by hour")
    ax.legend(frameon=False, ncol=2)
    _save(fig, folder, "price_mae_by_hour.png")


def shap_summary(sv, Xs, folder):
    import shap
    shap.summary_plot(sv, Xs, show=False, max_display=12, plot_size=(7, 4.2))
    fig = plt.gcf()
    fig.suptitle("LightGBM price model: SHAP values (GBP/MWh)", fontsize=9)
    _save(fig, folder, "price_shap.png")


def generation_week(obs, phys, corr, folder, name):
    te = obs.dropna().index
    test = obs[corr.notna()]
    start = (test if len(test) else obs).resample("7D").std().idxmax()
    sl = slice(start, start + pd.Timedelta(days=7))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(obs[sl], color=COLORS["observed"], lw=1.6, label="observed")
    ax.plot(phys[sl], color=COLORS["physical"], lw=1.1, label="physical")
    ax.plot(corr[sl], color=COLORS["physical+ML"], lw=1.1, label="physical+ML")
    ax.set_ylabel("Capacity factor")
    ax.set_ylim(0, 1)
    ax.set_title(f"Day-ahead generation forecast, reference {name}")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _save(fig, folder, "generation_week.png")


def capture_monthly(monthly: pd.DataFrame, folder):
    """Bars: baseload vs capture price. Line: capture discount (baseload - capture).

    The discount is used instead of the capture-rate ratio, which explodes when the
    monthly baseload price is close to zero.
    """
    m = monthly.astype(float)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(m))
    ax.bar(x - 0.2, m["baseload_price"], 0.4, color="#b8b8b8", label="baseload price")
    ax.bar(x + 0.2, m["capture_price"], 0.4, color="#2b6cb0", label="wind capture price")
    ax.plot(x, m["baseload_price"] - m["capture_price"], color="#d9822b", marker="o", ms=3,
            label="capture discount")
    ax.axhline(0, color="#1f1f1f", lw=0.6)
    ax.set_ylabel("GBP/MWh")
    ax.set_xticks(x, [d.strftime("%b %y") for d in m.index], rotation=45)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("Portfolio capture price vs baseload by month")
    _save(fig, folder, "capture_monthly.png")


def revenue_scatter(daily: pd.DataFrame, folder):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(daily["real"] / 1e3, daily["exp"] / 1e3, s=8, alpha=0.6, color="#2b6cb0")
    lim = [min(daily[["real", "exp"]].min()) / 1e3, max(daily[["real", "exp"]].max()) / 1e3]
    ax.plot(lim, lim, color="#1f1f1f", lw=0.8)
    ax.set_xlabel("Realised daily revenue, GBP k")
    ax.set_ylabel("Day-ahead expected revenue, GBP k")
    ax.set_title("Daily revenue: forecast vs realised")
    _save(fig, folder, "revenue_daily.png")
