"""Small HTTP helper with retries and an on-disk cache."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "uk-wind-revenue-forecast (research portfolio project)"


def get_json(url: str, params: dict, cache_dir: Path, retries: int = 5, pause: float = 0.3):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)

    delay = 2.0
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=90)
        except requests.RequestException as e:
            err = e
        else:
            if r.status_code == 200:
                data = r.json()
                with open(path, "w") as f:
                    json.dump(data, f)
                time.sleep(pause)
                return data
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"{r.status_code} from {url} {params}: {r.text[:300]}")
            err = RuntimeError(f"{r.status_code}: {r.text[:200]}")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"Giving up on {url} {params}: {err}")
