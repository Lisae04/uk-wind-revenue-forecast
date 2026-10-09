# data/raw

Put `england_wind.csv` here (exported by the capstone C1 notebook, `Capstone/` Drive folder).

Required columns:

| column | unit | description |
|---|---|---|
| `cell_id` | – | OS 1 km grid reference, e.g. `TF8630` |
| `wind_speed_ms` | m/s | long-term mean wind speed at 100 m (Global Wind Atlas) |

The file is git-ignored. Only the 26 cells of the chosen GA portfolio are read.
Global Wind Atlas data are licensed CC BY 4.0 (Technical University of Denmark).
