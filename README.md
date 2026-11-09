# UK Wind Revenue Forecast

> **Status: work in progress.** The modelling pipeline is being built out; results, figures and evaluation will be added as each stage is finished.

Day-ahead forecasting of **generation, price and revenue** for a 156 MW UK onshore wind portfolio.

## Why

A wind operator selling into the day-ahead market needs to know three things for tomorrow: how much it will generate, what the price will be, and what that means for revenue. This project builds each piece and checks how much each model actually improves on a simple baseline.

## Approach

| Stage | Method |
|---|---|
| Generation | Physical power curve from forecast wind speed, with a LightGBM model that corrects its errors |
| Price | LEAR (lasso-estimated auto-regressive) and LightGBM day-ahead price models |
| Revenue | Generation × price, with forecast error carried through |
| Explainability | SHAP, to show which inputs drive each forecast |
| Evaluation | Diebold-Mariano tests, to check whether one model's accuracy gain over another is statistically real |

## Roadmap

- [ ] Finish the generation model and compare it with the power-curve-only baseline
- [ ] Finish the price models and compare LEAR with LightGBM
- [ ] Combine them into revenue forecasts and measure revenue error
- [ ] Add figures and a short write-up of results

## Author

Lisa Maria El Khazen · [LinkedIn](https://www.linkedin.com/in/lisa-maria-el-khazen)
