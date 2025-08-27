#!/usr/bin/env python3
"""Data loading utilities for FPL season planner.

Provides resilient loaders from local Vaastav CSVs or live FPL API and
helpers to locate the Fantasy-Premier-League data folder inside this repo.
"""

from __future__ import annotations

import os
import json
import re
import importlib.util
from typing import Optional, Dict, Tuple

import pandas as pd

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore


def _import_module_from_path(module_name: str, file_path: str):
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_json(path: Optional[str], url: Optional[str] = None):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if url:
        if requests is None:
            raise RuntimeError("requests not available; provide local file instead.")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        return r.json()
    raise FileNotFoundError("Provide a local path or a URL")


def _load_csv(path: Optional[str], url: Optional[str] = None) -> pd.DataFrame:
    src = path or url
    if not src:
        raise FileNotFoundError("Provide a local path or a URL")
    try:
        return pd.read_csv(src)
    except Exception:
        try:
            return pd.read_csv(src, engine="python", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(src, engine="python")


def _default_fpl_data_root() -> str:
    here = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(here, "Fantasy-Premier-League", "data")


def _latest_available_season(data_root: str) -> Optional[str]:
    try:
        candidates = [
            name
            for name in os.listdir(data_root)
            if re.fullmatch(r"\d{4}-\d{2}", name)
            and os.path.isdir(os.path.join(data_root, name))
        ]
        if not candidates:
            return None
        return sorted(candidates)[-1]
    except Exception:
        return None


def _season_paths(data_root: str, season: str) -> Dict[str, str]:
    base = os.path.join(data_root, season)
    return {
        "merged_gw": os.path.join(base, "gws", "merged_gw.csv"),
        "players_raw": os.path.join(base, "players_raw.csv"),
    }


def load_sources(args) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """Load bootstrap, fixtures, merged_gw and players_raw using args and Vaastav helpers.

    Resolution order:
    - If explicit local path or URL provided via args, use that
    - Otherwise, try Fantasy-Premier-League/getters.py for live JSON
    - For CSVs, resolve season paths under --fpl-data-root and --season
      If not found and --allow-remote, fall back to GitHub raw URLs
    """

    getters_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "Fantasy-Premier-League",
        "getters.py",
    )
    vaastav_getters = _import_module_from_path("fpl_getters", getters_path)

    # Bootstrap
    if getattr(args, "bootstrap", None) or getattr(args, "bootstrap_url", None):
        bs = _load_json(
            getattr(args, "bootstrap", None), getattr(args, "bootstrap_url", None)
        )
    else:
        if vaastav_getters and hasattr(vaastav_getters, "get_data"):
            bs = vaastav_getters.get_data()  # type: ignore[attr-defined]
        else:
            bs = _load_json(
                None, "https://fantasy.premierleague.com/api/bootstrap-static/"
            )

    # Fixtures
    if getattr(args, "fixtures", None) or getattr(args, "fixtures_url", None):
        fixtures = _load_json(
            getattr(args, "fixtures", None), getattr(args, "fixtures_url", None)
        )
    else:
        if vaastav_getters and hasattr(vaastav_getters, "get_fixtures_data"):
            fixtures = vaastav_getters.get_fixtures_data()  # type: ignore[attr-defined]
        else:
            fixtures = _load_json(
                None, "https://fantasy.premierleague.com/api/fixtures/"
            )

    data_root = getattr(args, "fpl_data_root", None) or _default_fpl_data_root()
    season = getattr(args, "season", None) or _latest_available_season(data_root)
    season_paths = _season_paths(data_root, season) if season else {"merged_gw": None, "players_raw": None}  # type: ignore[assignment]

    # merged_gw.csv
    mgw_path_or_url = getattr(args, "merged_gw", None) or season_paths.get("merged_gw")
    if mgw_path_or_url and os.path.exists(mgw_path_or_url):
        mgw = _load_csv(mgw_path_or_url)
    elif getattr(args, "merged_gw_url", None):
        mgw = _load_csv(None, getattr(args, "merged_gw_url"))
    elif getattr(args, "allow_remote", False) and season:
        mgw_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
        mgw = _load_csv(None, mgw_url)
    else:
        raise FileNotFoundError(
            "merged_gw.csv not found. Pass --merged-gw or --merged-gw-url or set --season/--fpl-data-root."
        )

    # players_raw.csv
    praw_path_or_url = getattr(args, "players_raw", None) or season_paths.get(
        "players_raw"
    )
    if praw_path_or_url and os.path.exists(praw_path_or_url):
        praw = _load_csv(praw_path_or_url)
    elif getattr(args, "players_raw_url", None):
        praw = _load_csv(None, getattr(args, "players_raw_url"))
    elif getattr(args, "allow_remote", False) and season:
        praw_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv"
        praw = _load_csv(None, praw_url)
    else:
        raise FileNotFoundError(
            "players_raw.csv not found. Pass --players-raw or --players-raw-url or set --season/--fpl-data-root."
        )

    return bs, fixtures, mgw, praw
