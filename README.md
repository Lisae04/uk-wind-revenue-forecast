# UK Wind Portfolio: Day-Ahead Generation, Price and Revenue Forecasting

> **Headline:** _fill in after the first real run, e.g. "LightGBM cuts day-ahead GB price MAE by X% vs a seasonal naive benchmark (DM p < 0.01); the 156 MW portfolio captured Y% of baseload price, losing £Z/MWh to wind cannibalisation."_

How much is a planned onshore wind portfolio worth in the GB power market, and how well can that value be forecast a day ahead?

This repository takes the **156 MW, four-farm portfolio** selected by the multi-objective genetic algorithm in my BREE 495 capstone (UK onshore wind site selection), and turns it into a trading-style forecasting system:

```
forecast weather (D-1)  ──►  physical power curve  ──►  + ML correction  ──►  portfolio MWh
Elexon demand & wind    ──►  price models (naive / LEAR / LightGBM)    ──►  GBP/MWh
                                             MWh × GBP/MWh  ──►  revenue, capture price, forecast value
```

Everything is evaluated **out of sample** on a walk-forward basis, using only information available before the GB day-ahead gate closure.

---

## 1. Portfolio

GA_07 V3, option 1 (scenario `balanced-res-flag`, 150 MW target). One 6 MW turbine per 1 km cell.

| Farm | Local authority | MW | Cells | GA capacity factor |
|---|---|---|---|---|
| 3815 | North Norfolk | 24 | 4 | 0.407 |
| 4530 | Melton | 24 | 4 | 0.408 |
| 20201 | County Durham | 48 | 8 | 0.485 |
| 34424 | North Yorkshire | 60 | 10 | 0.456 |

The five GA options are all in `data/reference/ga_portfolio_farms.csv`. Change `portfolio.option` in `config.yaml` to model another one.

## 2. Data

| Source | Dataset | Use | Resolution | Units |
|---|---|---|---|---|
| Capstone (Global Wind Atlas) | `england_wind.csv` | long-term mean wind per cell, at 100 m | 1 km | m/s |
| Open-Meteo archive | ERA5 reanalysis | "actual" weather | hourly | m/s, °C, hPa |
| Open-Meteo Previous Runs | archived NWP at a fixed lead time | day-ahead weather forecast | hourly | m/s, °C, hPa |
| Elexon Insights | `MID` (APXMIDP) | GB reference price (target) | 30 min → hourly | GBP/MWh |
| Elexon Insights | `NDF` | day-ahead national demand forecast | 30 min → hourly | MW |
| Elexon Insights | `WINDFOR` | transmission-metered wind forecast | → hourly | MW |
| Elexon Insights | `FUELHH` (WIND) | wind outturn (national reference) | 30 min → hourly | MW |
| Elexon Insights | `B1610` | metered output per BM unit (reference farms) | 30 min → hourly | MWh |
| NREL turbine-models | IEC Class I/II/III normalized industry composite | power curves (BSD-3) | 0.5 m/s | 0–1 |

Period: **2024-01-01 to 2026-09-30**, limited by the start of Open-Meteo's archived forecasts. The test period runs from **2025-10-01** (12 months, all out of sample). None of the sources needs an API key.

> **About the price.** GB day-ahead auction results (EPEX, N2EX) are not free to redistribute. The target here is Elexon's Market Index Price from APX, a volume-weighted short-term traded price per settlement period, which is the standard free proxy for the GB reference price. Zero-volume periods are treated as missing.

## 3. Method

### 3.1 Information set (no look-ahead)

Delivery day *D* is forecast with information available at **D-1 09:00 Europe/London**, before the GB day-ahead auctions.

- **Elexon forecasts:** for every delivery hour, the latest vintage published before the gate is used (`timeutils.select_vintage`).
- **Weather:** `*_previous_day1` (predicted 24 h before valid time) is used only for hours where that run predates the gate. Otherwise `*_previous_day2` is used.
- **Prices:** the Market Index for D-1 afternoon is not known at 09:00 on D-1. So the textbook "same hour yesterday" lag would leak, and price features are restricted to a 48 h lag, a 168 h lag, the D-2 daily mean and the D-1 morning mean (hours ending before 08:00).
- A unit test rewrites every price after the gate to 10⁶ and checks that no feature changes. The test includes the BST changeover.

### 3.2 Generation: physical model plus ML correction

For each cell *c* and hour *t*:

| Step | Formula | Notes |
|---|---|---|
| Shear | U_hub = U₁₀₀ · (z_hub / 100)^α_t, with α_t = ln(U₁₀₀/U₁₀) / ln(10), clipped to [0, 0.5] | hub = 100 m by default, matching GA |
| Downscaling | U_c = k_c · U_hub, with k_c = GWA̅_c / mean_t(U_hub,ERA5) | anchors the long-term level to the capstone's GWA means (Staffell & Pfenninger, 2016) |
| Air density | U_eq = U_c · (ρ / 1.225)^(1/3), with ρ = p / (R_d · T) | IEC 61400-12-1 |
| Power curve | P = f_IEC(U_eq), with the class whose V_ave is nearest the cell mean | NREL composite curves |
| Energy | E_c = 6 MW · P · (1 − 0.23) | same 23% losses as the GA |

The power curve cannot see curtailment, icing, outages, wake and direction effects, or ramp timing. A **LightGBM residual model** learns these from a metered reference farm:

$$r_t = CF^{obs}_t - CF^{phys}_t,\qquad CF^{corr}_t = \mathrm{clip}\left(CF^{phys}_t + \hat r_t,\ 0,\ 1-\text{losses}\right)$$

- **Features** (all day-ahead): CF_phys, hub wind, shear exponent, wind direction (sin/cos), temperature, air density, local hour, day of year, weekday.
- **Level-matching:** the reference farm's own physical model is level-matched on training data only (k fitted so that the mean physical CF equals the mean observed CF before `test_start`). The correction therefore changes the *shape* and leaves the GWA-anchored annual energy alone.
- **Retraining:** walk-forward, refitted every 30 days on an expanding window.

**Reference farms.** Almost all English onshore wind farms are embedded in distribution networks and have no Elexon BM unit, so there is no public metered output near the portfolio. Two options are implemented:

| `reference_farms.mode` | Target | Trade-off |
|---|---|---|
| `bmu` | B1610 metered output of the BM units listed in `config.yaml` (run `make bmus` to list candidates) | real farm behaviour, but sites are mostly Scottish or offshore |
| `national` (default when no BMUs are listed) | GB wind outturn ÷ trailing 365-day 99.5th-percentile capacity proxy | always available, but spatially coarse |

### 3.3 Price models

| Model | Specification |
|---|---|
| `naive_D-2`, `naive_D-7` | price 48 h and 168 h earlier |
| **LEAR** | 24 hourly LASSO models, AIC-selected penalty (`LassoLarsIC`), asinh-median/MAD variance-stabilising transform, 365-day rolling window, recalibrated weekly (Lago et al., 2021; Uniejewski et al., 2018) |
| **LightGBM** | single model with local hour as a feature, L1 objective, expanding window, retrained every 30 days |

Features: demand forecast, wind forecast, residual demand (demand − wind), wind share, the price lags in §3.1, residual demand one week earlier, and calendar variables (hour, weekday, month, weekend, English bank holidays). Drivers are explained with **SHAP** (TreeExplainer).

### 3.4 Evaluation

- **Price:** MAE, RMSE, sMAPE, bias, and rMAE relative to `naive_D-7` (MAPE is avoided because prices go negative). The multivariate **Diebold-Mariano** test on daily mean absolute errors is one-sided at 5%.
- **Generation:** out-of-sample CF errors on the reference farm, against a climatology baseline (mean CF by month and hour).
- **Long-term check:** ERA5-driven capacity factor per farm compared with the GA capacity factor.

### 3.5 Revenue

Merchant revenue at the reference price, over the test year:

| Metric | Formula |
|---|---|
| Revenue | R = Σ E_t · P_t |
| Capture price | Σ E_t P_t / Σ E_t |
| Capture rate | capture price / mean P_t |
| Capture discount | mean P_t − capture price |
| Cannibalisation | OLS slope of price on forecast wind share (GBP/MWh per +10 pp) (Hirth, 2013) |
| Forecast value | MAE of daily expected vs realised revenue, for each pair of generation and price models |

The capture rate is reported as missing for months whose baseload is below £10/MWh, where the ratio is meaningless.

## 4. Results

_Run `make all`, then copy the tables from `results/summary.md` and the figures from `results/figures/` here._

| Figure | File |
|---|---|
| Price forecasts, most volatile test week | `results/figures/price_week.png` |
| Price MAE by hour | `results/figures/price_mae_by_hour.png` |
| SHAP drivers of price | `results/figures/price_shap.png` |
| Generation forecast vs observed (reference) | `results/figures/generation_week.png` |
| Capture price vs baseload by month | `results/figures/capture_monthly.png` |
| Expected vs realised daily revenue | `results/figures/revenue_daily.png` |

## 5. Limitations

- **The portfolio does not exist,** so its "actual" output is the physical model driven by ERA5. It contains no curtailment or outages, so portfolio-level skill numbers flatter the physical forecast and penalise the ML correction. The honest test of the correction is the reference farm.
- **The reference farm is not near the portfolio** (see §3.2), so the correction transfers behaviour learned elsewhere.
- **The price is the APX Market Index,** not the hourly day-ahead auction. There are no transaction costs, no imbalance exposure and no balancing-mechanism revenue, and the generator is a perfect price-taker.
- **Merchant only.** No CfD, PPA or REGO revenue, and no negative-price curtailment rule.
- **Single IEC class per cell, by nearest V_ave.** This should be checked against the class rule in the capstone's GA_02.
- **The data window starts in 2024** (archived forecasts), so there is only about 21 months of training before the test year.

## 6. Reproduce

```bash
git clone <this repo> && cd uk-wind-revenue-forecast
make install
make test          # 17 tests: grid, power curve, gate closure, no look-ahead, API parsing, end-to-end
make demo          # offline run on synthetic data -> results/synthetic_demo/ (not findings)

cp /path/to/england_wind.csv data/raw/
make all           # real data; first run downloads and caches (~30-60 min)
make bmus          # optional: list Elexon wind BM units for reference_farms.bmus
```

Everything is set in `config.yaml`: period, gate closure, GA option, turbine, reference farms and model hyperparameters.

```
├── config.yaml
├── data/reference/            GA portfolio export, NREL power curves (+ licence)
├── data/raw/                  england_wind.csv (git-ignored)
├── scripts/run_pipeline.py    entry point (--synthetic for offline)
├── scripts/find_wind_bmus.py  reference-farm discovery
├── src/ukwind/
│   ├── grid.py                OS grid ref -> BNG -> WGS84
│   ├── sites.py               cells, farms, GWA means, IEC class
│   ├── power.py               shear, density, power curve
│   ├── fetch_weather.py       ERA5 + archived forecasts (Open-Meteo)
│   ├── fetch_elexon.py        MID, NDF, WINDFOR, FUELHH, B1610
│   ├── timeutils.py           gate closure, forecast vintages
│   ├── generation.py          physical model, level-matching, ML correction
│   ├── price.py               naive, LEAR, LightGBM, SHAP
│   ├── evaluate.py            metrics, Diebold-Mariano
│   ├── revenue.py             capture price, forecast value
│   ├── plots.py
│   └── pipeline.py
└── tests/test_core.py
```

## 7. References

- Davis, N.N., Badger, J., Hahmann, A.N., et al. (2023). The Global Wind Atlas: A high-resolution dataset of climatologies and associated web-based application. *Bulletin of the American Meteorological Society*, 104(8), E1507–E1525.
- Diebold, F.X., Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049.
- Hirth, L. (2013). The market value of variable renewables: The effect of solar wind power variability on their relative price. *Energy Economics*, 38, 218–236.
- Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy*, 293, 116983.
- Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Staffell, I., Pfenninger, S. (2016). Using bias-corrected reanalysis to simulate current and future wind power output. *Energy*, 114, 1224–1239.
- Uniejewski, B., Weron, R., Ziel, F. (2018). Variance stabilizing transformations for electricity spot price forecasting. *IEEE Transactions on Power Systems*, 33(2), 2219–2229.
- Weron, R. (2014). Electricity price forecasting: A review of the state-of-the-art with a look into the future. *International Journal of Forecasting*, 30(4), 1030–1081.

## Acknowledgements

The site portfolio comes from the BREE 495 capstone (McGill University, Bioresource Engineering), a team project. Power curves are from NREL's [turbine-models](https://github.com/NREL/turbine-models) (BSD-3). Data are from Elexon (BMRS licence), Open-Meteo (CC BY 4.0, including ERA5 from Copernicus) and the Global Wind Atlas (CC BY 4.0).
