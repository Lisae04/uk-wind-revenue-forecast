"""End-to-end run: sites -> data -> generation -> price -> revenue -> report."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import generation as gen
from . import plots
from . import price as pr
from . import revenue as rv
from .config import reference_mode
from .sites import build_sites, summarise
from .timeutils import utc_ts


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data
def circular_mean_deg(frames, col):
    rad = np.deg2rad(np.column_stack([f[col].to_numpy(float) for f in frames]))
    return (np.rad2deg(np.arctan2(np.nanmean(np.sin(rad), 1), np.nanmean(np.cos(rad), 1))) + 360) % 360


def average_weather(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = sum(f.drop(columns=[c for c in f if c.startswith("wd")]) for f in frames) / len(frames)
    for c in ("wd100_act", "wd100_fc"):
        out[c] = circular_mean_deg(frames, c)
    return out[frames[0].columns]


def national_reference(system, weather, farms):
    """GB transmission-metered wind CF; capacity proxy = trailing 365-day 99.5th percentile."""
    out = system["wind_outturn_mw"]
    cap = out.rolling("365D", min_periods=24 * 30).quantile(0.995).shift(1)
    cf = (out / cap).clip(0, 1)
    w = average_weather([weather[f] for f in farms["farm_id"]])
    return {"GB_national": {"weather": w, "cf_obs": cf}}


def load_data(cfg, synthetic: bool):
    proc = cfg["paths"]["processed"]
    if synthetic:
        from .synthetic import make_all, synthetic_gwa
        farms_tmp = pd.read_csv(cfg["portfolio"]["farms_file"])
        farms_tmp = farms_tmp[farms_tmp.option == cfg["portfolio"]["option"]]
        cell_ids = sorted({c for s in farms_tmp["cells"] for c in s.split(";")})
        cells, farms = build_sites(cfg, gwa=synthetic_gwa(cell_ids))
        weather, system, refs = make_all(cfg, cells, farms)
        mode = "bmu"
    else:
        from .fetch_elexon import fetch_bmu_output, fetch_system
        from .fetch_weather import fetch_site_weather
        cells, farms = build_sites(cfg)
        weather = {}
        for f in farms.itertuples():
            log(f"weather for {f.name}")
            weather[f.farm_id] = fetch_site_weather(f.lat, f.lon, cfg, cfg["paths"]["cache"])
            log(f"  filled {weather[f.farm_id].attrs.get('filled', 0)} values, "
                f"{weather[f.farm_id].attrs.get('still_missing', 0)} still missing")
        log("Elexon prices, demand/wind forecasts and wind outturn")
        system = fetch_system(cfg)
        mode = reference_mode(cfg)
        refs = {}
        if mode == "bmu":
            p = cfg["period"]
            for b in cfg["reference_farms"]["bmus"]:
                log(f"reference BMU {b['bmu']}")
                mwh = fetch_bmu_output(b["bmu"], p["start"], p["end"], cfg["paths"]["cache"])
                refs[b["bmu"]] = {"weather": fetch_site_weather(b["lat"], b["lon"], cfg, cfg["paths"]["cache"]),
                                  "cf_obs": (mwh.clip(lower=0) / float(b["capacity_mw"])).clip(0, 1)}
    if mode == "national":
        refs = national_reference(system, weather, farms)

    cells.to_csv(proc / "cells.csv", index=False)
    farms.to_csv(proc / "farms.csv", index=False)
    system.to_parquet(proc / "system.parquet")
    for fid, w in weather.items():
        w.to_parquet(proc / f"weather_{fid}.parquet")
    return cells, farms, weather, system, refs, mode


# ------------------------------------------------------------------ generation
def run_generation(cfg, cells, farms, weather, refs):
    t = cfg["turbine"]
    test_start = utc_ts(cfg["period"]["test_start"])
    phys = gen.portfolio_physical(cells, farms, weather, t)
    check = gen.long_term_check(phys["act"], farms)

    # reference farms: level-match, residual target, features
    X_ref, r_ref, rows = [], [], []
    for name, ref in refs.items():
        w, cf_obs = ref["weather"], ref["cf_obs"].reindex(ref["weather"].index)
        train = np.asarray(w.index < test_start)
        cls = gen.reference_class(np.nanmean(gen.hub_wind(w, "act", t)), t)
        k = gen.level_match(w, cf_obs, train, cls, t)
        cf_fc = gen.reference_cf(w, "fc", k, cls, t)
        X = gen.correction_features(w, cf_fc, t, cfg["market"]["tz"])
        X_ref.append(X.assign(ref=name))
        r_ref.append(cf_obs - cf_fc)
        rows.append({"reference": name, "iec_class": cls, "k_level": k})
    X_ref = pd.concat(X_ref)
    r_ref = pd.concat(r_ref)
    ref_names = X_ref.pop("ref")

    # portfolio features (forecast weather only)
    X_port = {}
    for f in farms.itertuples():
        cf_fc = phys["fc"][f.name] / f.mw
        X_port[f.name] = gen.correction_features(weather[f.farm_id], cf_fc, t, cfg["market"]["tz"])

    # walk-forward on the pooled references (one time-sorted frame, positional alignment)
    order = np.argsort(X_ref.index.values, kind="stable")
    Xs, rs, names = X_ref.iloc[order], r_ref.iloc[order], ref_names.iloc[order].to_numpy()
    gm = cfg["generation_model"]
    r_hat, r_port, model = gen.walk_forward_correction(Xs, rs, test_start, gm["retrain_every_days"],
                                                       gm["lgbm"], X_port)

    # reference evaluation (out of sample)
    ref_eval = []
    for name, ref in refs.items():
        msk = names == name
        Xr = Xs[msk]
        obs = ref["cf_obs"].reindex(Xr.index)
        phys_cf = Xr["cf_phys"]
        rh = pd.Series(r_hat.to_numpy()[msk], index=Xr.index)
        corr = (phys_cf + rh).clip(0, 1)
        train = np.asarray(Xr.index < test_start)
        clim = gen.climatology_cf(obs, train, cfg["market"]["tz"])
        te = Xr.index >= test_start
        tab = ev.metrics_table(obs[te], {"climatology": clim[te], "physical": phys_cf[te],
                                         "physical+ML": corr[te]}, scale_ref="climatology")
        tab.insert(0, "reference", name)
        ref_eval.append(tab)
        if name == list(refs)[0]:
            plots.generation_week(obs, phys_cf, corr, cfg["paths"]["results"] / "figures", name)
    ref_eval = pd.concat(ref_eval)

    # portfolio forecasts
    cap_cf = 1 - float(t["losses"])
    fc_ml = {}
    for f in farms.itertuples():
        cf = phys["fc"][f.name] / f.mw
        rh = r_port[f.name]
        fc_ml[f.name] = np.where(rh.notna(), (cf + rh).clip(0, cap_cf), cf) * f.mw
    fc_ml = pd.DataFrame(fc_ml, index=phys["fc"].index)

    port = pd.DataFrame({"gen_act": phys["act"].sum(axis=1), "gen_fc_phys": phys["fc"].sum(axis=1),
                         "gen_fc_ml": fc_ml.sum(axis=1)})
    te = port.index >= test_start
    mw = farms["mw"].sum()
    port_eval = ev.metrics_table(port.loc[te, "gen_act"] / mw,
                                 {"physical": port.loc[te, "gen_fc_phys"] / mw,
                                  "physical+ML": port.loc[te, "gen_fc_ml"] / mw})
    try:
        import shap  # noqa: F401
        imp, _, _ = pr.shap_importance(model, X_port[farms["name"].iloc[0]][te])
    except Exception:
        imp = pd.Series(dtype=float)
    return {"phys": phys, "port": port, "check": check, "ref_eval": ref_eval,
            "port_eval": port_eval, "refs": pd.DataFrame(rows), "shap_gen": imp}


# ------------------------------------------------------------------ price
def run_price(cfg, system):
    m, pm = cfg["market"], cfg["price_model"]
    test_start = cfg["period"]["test_start"]
    df = pr.build_dataset(system, m["tz"], m["gate_closure_local"])
    pr.assert_no_lookahead(df, m["tz"], m["gate_closure_local"])
    preds = pr.naive_forecasts(df)
    log("price: LEAR")
    preds["LEAR"] = pr.lear_forecast(df, test_start, pm["lear_window_days"], pm["lear_recalibrate_every_days"])
    log("price: LightGBM")
    preds["LightGBM"], model = pr.lgbm_forecast(df, test_start, pm["lgbm_retrain_every_days"], pm["lgbm"])
    te = df.index >= utc_ts(test_start)
    y = df.loc[te, "price"]
    P = {k: v[te] for k, v in preds.items()}
    table = ev.metrics_table(y, P, scale_ref="naive_D-7")
    dm = ev.dm_matrix(y, P)
    imp, sv, Xs = pr.shap_importance(model, df.loc[te, pr.FEATURES])
    figs = cfg["paths"]["results"] / "figures"
    plots.price_week(y, P, figs)
    plots.mae_by_hour(y, P, figs, m["tz"])
    plots.shap_summary(sv, Xs, figs)
    return {"df": df, "preds": preds, "table": table, "dm": dm, "shap": imp}


# ------------------------------------------------------------------ revenue
def run_revenue(cfg, G, P):
    te = G["port"].index >= utc_ts(cfg["period"]["test_start"])
    port = G["port"][te]
    price = P["df"].loc[te, "price"]
    real = rv.capture_stats(port["gen_act"], price)
    monthly = rv.by_period(port["gen_act"], price)
    combos = []
    for g in ("gen_fc_phys", "gen_fc_ml"):
        for pm in ("naive_D-7", "LEAR", "LightGBM"):
            d = rv.daily_revenue_error(port[g], P["preds"][pm][te], port["gen_act"], price)
            combos.append({"generation": g.replace("gen_fc_", ""), "price": pm,
                           "daily_rev_MAE_gbp": d["error"].abs().mean(),
                           "daily_rev_bias_gbp": d["error"].mean(),
                           "MAE_%_of_mean_daily_rev": 100 * d["error"].abs().mean() / d["real"].mean()})
    combos = pd.DataFrame(combos)
    cann = rv.cannibalisation(P["df"]["price"], P["df"]["wind_share_fc"])
    figs = cfg["paths"]["results"] / "figures"
    plots.capture_monthly(monthly, figs)
    best = rv.daily_revenue_error(port["gen_fc_ml"], P["preds"]["LightGBM"][te], port["gen_act"], price)
    plots.revenue_scatter(best, figs)
    return {"realised": real, "monthly": monthly, "forecast_value": combos, "cannibalisation": cann}


# ------------------------------------------------------------------ report
def _md(df, floatfmt=".3f"):
    return df.to_markdown(floatfmt=floatfmt)


def write_report(cfg, farms, cells, G, P, R, mode, synthetic):
    res = cfg["paths"]["results"]
    G["check"].to_csv(res / "generation_longterm_check.csv", index=False)
    G["ref_eval"].to_csv(res / "generation_reference_metrics.csv")
    G["port_eval"].to_csv(res / "generation_portfolio_metrics.csv")
    P["table"].to_csv(res / "price_metrics.csv")
    P["dm"].to_csv(res / "price_dm_pvalues.csv")
    P["shap"].to_csv(res / "price_shap_importance.csv")
    R["monthly"].to_csv(res / "revenue_monthly.csv")
    R["forecast_value"].to_csv(res / "revenue_forecast_value.csv", index=False)
    with open(res / "revenue_realised.json", "w") as f:
        json.dump({**R["realised"], **{f"cannibalisation_{k}": v for k, v in R["cannibalisation"].items()}},
                  f, indent=2)

    banner = ("> **SYNTHETIC DATA RUN.** All numbers below come from invented data and only show "
              "that the pipeline works. Run `make all` with real data for findings.\n\n") if synthetic else ""
    r = R["realised"]
    lines = [
        "# Results summary", "", banner,
        f"Portfolio: GA option {cfg['portfolio']['option']}, {len(farms)} farms, "
        f"{len(cells)} cells, {farms['mw'].sum():.0f} MW. Out-of-sample period from "
        f"{cfg['period']['test_start']} to {cfg['period']['end']}. ML reference: **{mode}**.", "",
        "## 1. Long-term check against the GA (capstone) capacity factors", "",
        _md(G["check"].set_index("farm")), "",
        "## 2. Generation forecast (reference farm, out of sample, CF units)", "",
        _md(G["ref_eval"]), "",
        "Portfolio forecast vs ERA5-driven proxy (no metered output exists for planned farms):", "",
        _md(G["port_eval"]), "",
        "## 3. Price forecast (GBP/MWh, out of sample)", "", _md(P["table"], ".2f"), "",
        "Diebold-Mariano one-sided p-values (column more accurate than row):", "",
        _md(P["dm"], ".3f"), "",
        "Top LightGBM drivers (mean |SHAP|, GBP/MWh):", "",
        _md(P["shap"].head(8).to_frame(), ".2f"), "",
        "## 4. Revenue", "",
        f"- Energy: {r['energy_mwh'] / 1000:,.1f} GWh",
        f"- Revenue: GBP {r['revenue_gbp'] / 1e6:,.2f} m",
        f"- Capture price: {r['capture_price']:.2f} GBP/MWh vs baseload {r['baseload_price']:.2f} "
        f"(capture rate {100 * r['capture_rate']:.1f}%)",
        f"- Energy produced at negative prices: {r.get('energy_at_negative_price_mwh', 0):,.0f} MWh "
        f"over {r.get('negative_price_hours', 0)} hours",
        f"- Price falls by {abs(R['cannibalisation']['slope_gbp_per_10pp']):.1f} GBP/MWh per +10 pp "
        f"forecast wind share (R2 {R['cannibalisation']['r2']:.2f})", "",
        "Daily revenue forecast error by model combination:", "",
        _md(R["forecast_value"].set_index(["generation", "price"]), ".1f"), "",
    ]
    (res / "summary.md").write_text("\n".join(lines))
    return res / "summary.md"


def run(cfg, synthetic: bool = False):
    if synthetic:   # keep demo output away from real results so it is never mistaken for findings
        for k in ("processed", "results"):
            cfg["paths"][k] = cfg["paths"][k] / "synthetic_demo"
            cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    log("sites and data")
    cells, farms, weather, system, refs, mode = load_data(cfg, synthetic)
    log(summarise(cells, farms))
    log("generation model")
    G = run_generation(cfg, cells, farms, weather, refs)
    log("price models")
    P = run_price(cfg, system)
    log("revenue")
    R = run_revenue(cfg, G, P)
    path = write_report(cfg, farms, cells, G, P, R, mode, synthetic)
    log(f"done -> {path}")
    return G, P, R
