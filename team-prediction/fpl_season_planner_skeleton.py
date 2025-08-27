#!/usr/bin/env python3
"""Season planner and helper utilities for Fantasy Premier League (FPL).

Generates per–gameweek outputs:
- Top-5 players by position
- Captain shortlists (protect/chase modes)
- Greedy transfer suggestions over a horizon

Data can be loaded from local Vaastav CSVs or the live FPL API with fallbacks.
Sections and docstrings are provided to keep the module readable.
"""

import argparse, os, json
import re
import importlib.util
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

try:
    import requests
except Exception:
    requests = None


def _load_json(path: Optional[str], url: Optional[str] = None):
    """Load JSON from a local path if present; otherwise from the provided URL.

    Args:
        path: Local filesystem path to a JSON file.
        url: HTTP(S) URL to fetch JSON if local path is not provided.

    Raises:
        RuntimeError: If a URL is provided but the HTTP client is unavailable.
        FileNotFoundError: If neither a path nor a URL is provided.
    """
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
    """Load CSV from local path or URL with tolerant parsing fallbacks."""
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


def _import_module_from_path(module_name: str, file_path: str):
    """Import a Python module given an absolute file path."""
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _default_fpl_data_root() -> str:
    """Return default path to the Vaastav data directory within this repo."""
    here = os.path.dirname(__file__)
    return os.path.join(here, "Fantasy-Premier-League", "data")


def _latest_available_season(data_root: str) -> Optional[str]:
    """Pick the lexicographically latest season folder matching YYYY-YY under data_root."""
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


# ===== Data acquisition and resolution =====


def load_sources(args):
    """Load bootstrap, fixtures, merged_gw and players_raw using args and Vaastav helpers.

    Resolution order:
    - If explicit local path or URL provided via args, use that
    - Otherwise, try Fantasy-Premier-League/getters.py for live JSON (bootstrap, fixtures)
    - For CSVs, resolve season paths under --fpl-data-root and --season
      If not found and --allow-remote, fall back to GitHub raw URLs
    """
    # Attempt to import Vaastav getters even though the directory has a hyphen
    getters_path = os.path.join(
        os.path.dirname(__file__), "Fantasy-Premier-League", "getters.py"
    )
    vaastav_getters = _import_module_from_path("fpl_getters", getters_path)

    # Bootstrap
    if args.bootstrap or args.bootstrap_url:
        bs = _load_json(args.bootstrap, args.bootstrap_url)
    else:
        if vaastav_getters and hasattr(vaastav_getters, "get_data"):
            bs = vaastav_getters.get_data()  # type: ignore[attr-defined]
        else:
            # fall back to default URL if requests available
            bs = _load_json(
                None, "https://fantasy.premierleague.com/api/bootstrap-static/"
            )

    # Fixtures
    if args.fixtures or args.fixtures_url:
        fixtures = _load_json(args.fixtures, args.fixtures_url)
    else:
        if vaastav_getters and hasattr(vaastav_getters, "get_fixtures_data"):
            fixtures = vaastav_getters.get_fixtures_data()  # type: ignore[attr-defined]
        else:
            fixtures = _load_json(
                None, "https://fantasy.premierleague.com/api/fixtures/"
            )

    # Resolve season file paths for merged_gw and players_raw
    data_root = args.fpl_data_root or _default_fpl_data_root()
    season = args.season or _latest_available_season(data_root)
    season_paths = _season_paths(data_root, season) if season else {"merged_gw": None, "players_raw": None}  # type: ignore[assignment]

    # merged_gw.csv
    mgw_path_or_url = args.merged_gw or season_paths.get("merged_gw")
    mgw = None
    if mgw_path_or_url and os.path.exists(mgw_path_or_url):
        mgw = _load_csv(mgw_path_or_url)
    elif args.merged_gw_url:
        mgw = _load_csv(None, args.merged_gw_url)
    elif args.allow_remote and season:
        mgw_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
        mgw = _load_csv(None, mgw_url)
    else:
        raise FileNotFoundError(
            "merged_gw.csv not found. Pass --merged-gw or --merged-gw-url or set --season/--fpl-data-root."
        )

    # players_raw.csv
    praw_path_or_url = args.players_raw or season_paths.get("players_raw")
    praw = None
    if praw_path_or_url and os.path.exists(praw_path_or_url):
        praw = _load_csv(praw_path_or_url)
    elif args.players_raw_url:
        praw = _load_csv(None, args.players_raw_url)
    elif args.allow_remote and season:
        praw_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv"
        praw = _load_csv(None, praw_url)
    else:
        raise FileNotFoundError(
            "players_raw.csv not found. Pass --players-raw or --players-raw-url or set --season/--fpl-data-root."
        )

    return bs, fixtures, mgw, praw


def _minutes60_prob(df_last4: pd.DataFrame) -> float:
    if len(df_last4) == 0 or "minutes" not in df_last4.columns:
        return 0.0
    return (df_last4["minutes"] >= 60).mean()


def _returns_rate_last4(df_last4: pd.DataFrame) -> float:
    if len(df_last4) == 0 or "total_points" not in df_last4.columns:
        return 0.0
    return (df_last4["total_points"].astype(float) >= 5).mean()


def _ict_last4(df_last4: pd.DataFrame) -> float:
    if len(df_last4) == 0 or "ict_index" not in df_last4.columns:
        return 0.0
    return df_last4["ict_index"].astype(float).mean()


def _clean_sheet_rate(df_last4: pd.DataFrame) -> float:
    if len(df_last4) == 0:
        return 0.0
    if "clean_sheets" in df_last4.columns:
        return df_last4["clean_sheets"].astype(float).mean()
    if "goals_conceded" in df_last4.columns:
        return (df_last4["goals_conceded"].astype(float) == 0).mean()
    return 0.0


def _saves_per90_last4(df_last4: pd.DataFrame) -> float:
    if len(df_last4) == 0 or "saves" not in df_last4.columns:
        return 0.0
    mins = df_last4.get("minutes", pd.Series(dtype=float)).astype(float).sum()
    if mins <= 0:
        return 0.0
    saves = df_last4["saves"].astype(float).sum()
    return saves / (mins / 90.0)


def add_player_features(bootstrap: dict, merged_gw: pd.DataFrame) -> pd.DataFrame:
    """Return player DataFrame enriched with short-horizon form features.

    Computes features from last four appearances in merged_gw (form4, ict4,
    mins60, returns4, cs4, saves90_4), normalizes numeric fields, and derives
    `position` and `price`.
    """
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name"]].rename(
        columns={"id": "team_id", "name": "team_name"}
    )
    df = elements.merge(teams, left_on="team", right_on="team_id", how="left")

    mgw = merged_gw.copy()
    if "element" not in mgw.columns:
        if "id" in mgw.columns:
            mgw = mgw.rename(columns={"id": "element"})
        else:
            raise ValueError("merged_gw missing 'element' id column")

    if "round" in mgw.columns:
        mgw["round"] = pd.to_numeric(mgw["round"], errors="coerce")

    feats = []
    for pid, g in mgw.groupby("element"):
        g = g.sort_values("round")
        last4 = g.tail(4)
        feats.append(
            {
                "id": pid,
                "form4": (
                    last4["total_points"].astype(float).mean()
                    if "total_points" in last4.columns and len(last4) > 0
                    else 0.0
                ),
                "ict4": _ict_last4(last4),
                "mins60": _minutes60_prob(last4),
                "returns4": _returns_rate_last4(last4),
                "cs4": _clean_sheet_rate(last4),
                "saves90_4": _saves_per90_last4(last4),
            }
        )
    feat_df = pd.DataFrame(feats)

    df = df.merge(feat_df, on="id", how="left")
    for col in [
        "ep_next",
        "form",
        "ict_index",
        "selected_by_percent",
        "now_cost",
        "chance_of_playing_next_round",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["position"] = df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    df["price"] = (df["now_cost"] / 10.0).fillna(0.0)
    return df


def availability_penalty(status, chance) -> float:
    """Heuristic penalty for availability based on `status` and `chance`."""
    pen = 0.0
    if status in ("i", "s"):
        pen += 1.0
    if status == "d":
        pen += 0.3
    try:
        c = float(chance)
        if c < 75:
            pen += 0.4
        elif c < 100:
            pen += 0.1
    except Exception:
        pass
    return pen


def _build_team_fixture_index(fixtures_json: dict) -> dict:
    """Map team->event to opponent and difficulty (legacy helper).

    Most logic now uses the vectorized DataFrame from `_fixture_difficulty_df`.
    """
    fx = pd.DataFrame(fixtures_json)
    fx = fx[
        ["event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    ].dropna()
    fx["event"] = fx["event"].astype(int)
    team_fixt = {}
    for _, r in fx.iterrows():
        team_fixt.setdefault(int(r["team_h"]), {})[int(r["event"])] = {
            "H": int(r["team_a"]),
            "diff": float(r["team_h_difficulty"]),
        }
        team_fixt.setdefault(int(r["team_a"]), {})[int(r["event"])] = {
            "A": int(r["team_h"]),
            "diff": float(r["team_a_difficulty"]),
        }
    return team_fixt


def _fixture_difficulty_df(fixtures_json: dict) -> pd.DataFrame:
    """Return a DataFrame mapping team_id x event -> difficulty."""
    fx = pd.DataFrame(fixtures_json)
    if fx.empty:
        return pd.DataFrame(columns=["event", "team_id", "diff"]).astype(
            {"event": int, "team_id": int, "diff": float}
        )
    fx = fx[
        ["event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    ].dropna()
    fx["event"] = fx["event"].astype(int)
    home = fx[["event", "team_h", "team_h_difficulty"]].rename(
        columns={"team_h": "team_id", "team_h_difficulty": "diff"}
    )
    away = fx[["event", "team_a", "team_a_difficulty"]].rename(
        columns={"team_a": "team_id", "team_a_difficulty": "diff"}
    )
    out = pd.concat([home, away], ignore_index=True)
    out["team_id"] = out["team_id"].astype(int)
    out["diff"] = pd.to_numeric(out["diff"], errors="coerce").fillna(3.0)
    return out


def _compute_base(p: pd.Series) -> float:
    return (
        0.55 * float(p.get("form", 0))
        + 0.25 * float(p.get("form4", 0))
        + 0.15 * (float(p.get("ict_index", 0)) / 10.0)
        + 0.05 * (float(p.get("ict4", 0)) / 10.0)
    )


def _position_boost(p: pd.Series) -> float:
    pos = p["position"]
    if pos == "GK":
        return 0.25 * float(p.get("cs4", 0)) + 0.10 * min(
            float(p.get("saves90_4", 0)) / 5.0, 1.0
        )
    if pos == "DEF":
        return 0.30 * float(p.get("cs4", 0))
    if pos == "MID":
        return 0.15 * float(p.get("returns4", 0))
    if pos == "FWD":
        return 0.20 * float(p.get("returns4", 0))
    return 0.0


def _fixture_adjustment(diff: float) -> float:
    return 1.0 + (3.0 - diff) * 0.06


def ep_per_gw(df_players: pd.DataFrame, fixtures_json: dict, gw_range=range(1, 39)):
    """Vectorized expected points per GW for all players.

    Returns a dict[gw] -> DataFrame with columns id, web_name, team_id, team_name, position, price, ep
    """
    fixt_df = _fixture_difficulty_df(fixtures_json)
    out: Dict[int, pd.DataFrame] = {}
    if df_players.empty or fixt_df.empty:
        return {
            gw: pd.DataFrame(
                columns=[
                    "id",
                    "web_name",
                    "team_id",
                    "team_name",
                    "position",
                    "price",
                    "ep",
                ]
            )
            for gw in gw_range
        }

    # Pre-compute reusable series
    base_series = (
        0.55 * pd.to_numeric(df_players.get("form", 0), errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(df_players.get("form4", 0), errors="coerce").fillna(0.0)
        + 0.15
        * (
            pd.to_numeric(df_players.get("ict_index", 0), errors="coerce").fillna(0.0)
            / 10.0
        )
        + 0.05
        * (pd.to_numeric(df_players.get("ict4", 0), errors="coerce").fillna(0.0) / 10.0)
    )

    mins60 = pd.to_numeric(df_players.get("mins60", 0.5), errors="coerce").fillna(0.5)
    xmins_series = 35.0 + 55.0 * mins60

    cs4 = pd.to_numeric(df_players.get("cs4", 0.0), errors="coerce").fillna(0.0)
    saves90_4 = pd.to_numeric(df_players.get("saves90_4", 0.0), errors="coerce").fillna(
        0.0
    )
    returns4 = pd.to_numeric(df_players.get("returns4", 0.0), errors="coerce").fillna(
        0.0
    )
    position = df_players["position"].astype(str)

    pos_boost = pd.Series(0.0, index=df_players.index)
    # GK
    is_gk = position == "GK"
    pos_boost.loc[is_gk] = 0.25 * cs4.loc[is_gk] + 0.10 * np.minimum(
        saves90_4.loc[is_gk] / 5.0, 1.0
    )
    # DEF
    is_def = position == "DEF"
    pos_boost.loc[is_def] = 0.30 * cs4.loc[is_def]
    # MID
    is_mid = position == "MID"
    pos_boost.loc[is_mid] = 0.15 * returns4.loc[is_mid]
    # FWD
    is_fwd = position == "FWD"
    pos_boost.loc[is_fwd] = 0.20 * returns4.loc[is_fwd]

    # Availability penalty vectorized
    status_pen = (
        df_players.get("status", pd.Series(index=df_players.index, dtype=object))
        .map({"i": 1.0, "s": 1.0, "d": 0.3})
        .fillna(0.0)
    )
    chance = pd.to_numeric(
        df_players.get("chance_of_playing_next_round", 100), errors="coerce"
    ).fillna(100.0)
    chance_pen = np.where(chance < 75.0, 0.4, np.where(chance < 100.0, 0.1, 0.0))
    avail_pen = status_pen + chance_pen

    base_inputs = pd.DataFrame(
        {
            "id": df_players["id"].astype(int),
            "web_name": df_players["web_name"],
            "team": pd.to_numeric(df_players["team"], errors="coerce")
            .fillna(0)
            .astype(int),
            "team_name": df_players["team_name"],
            "position": position,
            "price": pd.to_numeric(df_players["price"], errors="coerce").fillna(0.0),
            "xmins": xmins_series,
            "base": base_series,
            "pos_boost": pos_boost,
            "avail_pen": avail_pen,
        }
    )

    for gw in gw_range:
        fi = fixt_df[fixt_df["event"] == int(gw)][["team_id", "diff"]]
        if fi.empty:
            out[gw] = pd.DataFrame(
                columns=[
                    "id",
                    "web_name",
                    "team_id",
                    "team_name",
                    "position",
                    "price",
                    "ep",
                ]
            )
            continue
        df = base_inputs.merge(fi, left_on="team", right_on="team_id", how="inner")
        fixture_adj = (
            1.0 + (3.0 - pd.to_numeric(df["diff"], errors="coerce").fillna(3.0)) * 0.06
        )
        ep = (df["xmins"] / 90.0) * df["base"] * fixture_adj * (
            1.0 + df["pos_boost"]
        ) - df["avail_pen"]
        df_out = pd.DataFrame(
            {
                "id": df["id"].astype(int),
                "web_name": df["web_name"],
                "team_id": df["team_id"].astype(int),
                "team_name": df["team_name"],
                "position": df["position"],
                "price": df["price"],
                "ep": ep.clip(lower=0.0),
            }
        )
        out[gw] = df_out
    return out


def _ownership_proxy(p: pd.Series) -> float:
    try:
        return float(p.get("selected_by_percent", 0.0)) / 100.0
    except Exception:
        return 0.0


def captain_shortlist(
    ep_by_gw: Dict[int, pd.DataFrame],
    players: pd.DataFrame,
    gw: int,
    risk_mode: str = "protect",
    top_k: int = 5,
) -> pd.DataFrame:
    """Shortlist captain candidates for `gw` with ownership leverage.

    Protect mode mildly penalizes low-owned picks; chase mode rewards them.
    """
    df = ep_by_gw.get(gw, pd.DataFrame()).copy()
    if df.empty:
        return df
    base = players[["id", "selected_by_percent", "position"]].copy()
    df = df.merge(base, on="id", how="left", suffixes=("", "_pl"))
    df["ownership"] = df["selected_by_percent"].astype(float) / 100.0
    df["ownership"] = df["ownership"].fillna(0.0)
    # Leverage: boost low-ownership when chasing, penalize when protecting
    if risk_mode == "chase":
        df["cap_score"] = df["ep"] * (1.0 + (0.30 * (1.0 - df["ownership"])))
    else:  # protect
        df["cap_score"] = df["ep"] * (1.0 - (0.20 * (1.0 - df["ownership"])))
    df.sort_values(["cap_score", "ep"], ascending=False, inplace=True)
    return df.head(top_k)[
        ["id", "web_name", "team_name", "position", "ep", "ownership", "cap_score"]
    ]


# ===== Transfer planning helpers =====


def _build_ep_window(
    ep_by_gw: Dict[int, pd.DataFrame], window: List[int]
) -> pd.DataFrame:
    """Build cumulative expected points for each player across a GW window.

    Returns a DataFrame with columns [id, ep_window].
    """
    ep_sum: Optional[pd.DataFrame] = None
    for gw in window:
        df = ep_by_gw.get(gw, pd.DataFrame())[["id", "ep"]].rename(
            columns={"ep": f"ep_{gw}"}
        )
        ep_sum = df if ep_sum is None else ep_sum.merge(df, on="id", how="outer")
    if ep_sum is None:
        return pd.DataFrame(columns=["id", "ep_window"])  # type: ignore[return-value]
    ep_sum = ep_sum.fillna(0.0)
    ep_cols = [c for c in ep_sum.columns if c.startswith("ep_")]
    ep_sum["ep_window"] = ep_sum[ep_cols].sum(axis=1)
    return ep_sum[["id", "ep_window"]]


def _load_current_squad(path: Optional[str], df_players: pd.DataFrame) -> List[int]:
    """Load up to 15 player ids from CSV with either `id` or `web_name` column."""
    if not path or not os.path.exists(path):
        return []
    try:
        sq = pd.read_csv(path)
        if "id" in sq.columns:
            ids = sq["id"].dropna().astype(int).tolist()
        elif "web_name" in sq.columns:
            ids = (
                sq.merge(df_players[["id", "web_name"]], on="web_name", how="left")[
                    "id"
                ]
                .dropna()
                .astype(int)
                .tolist()
            )
        else:
            return []
        return ids[:15]
    except Exception:
        return []


def _best_xi_ep_for_squad(
    squad_ids: List[int], ep_df: pd.DataFrame
) -> Tuple[float, List[int]]:
    """Select a viable XI under position minimums and return (total_ep, xi_ids)."""
    if not squad_ids:
        return 0.0, []
    team = ep_df[ep_df["id"].isin(squad_ids)].copy()
    if team.empty:
        return 0.0, []
    # Pick XI by simple formation heuristic: 1 GK, at least 3 DEF, 2 MID, 1 FWD, rest best
    _ = []
    # GK
    gk = team[team["position"] == "GK"].nlargest(1, "ep")
    xi_ids = set(gk["id"].tolist())
    # DEF, MID, FWD minimums
    def_min = team[(team["position"] == "DEF") & (~team["id"].isin(xi_ids))].nlargest(
        3, "ep"
    )
    xi_ids.update(def_min["id"].tolist())
    mid_min = team[(team["position"] == "MID") & (~team["id"].isin(xi_ids))].nlargest(
        2, "ep"
    )
    xi_ids.update(mid_min["id"].tolist())
    fwd_min = team[(team["position"] == "FWD") & (~team["id"].isin(xi_ids))].nlargest(
        1, "ep"
    )
    xi_ids.update(fwd_min["id"].tolist())
    # Fill remaining spots up to 11 with highest ep among outfield and GK if needed
    remaining = team[~team["id"].isin(xi_ids)].nlargest(11 - len(xi_ids), "ep")
    xi_ids.update(remaining["id"].tolist())
    xi_df = team[team["id"].isin(xi_ids)]
    return float(xi_df["ep"].sum()), list(xi_ids)


def _squad_team_counts(squad_ids: List[int], players: pd.DataFrame) -> Dict[str, int]:
    """Count players per real team within a squad."""
    counts: Dict[str, int] = {}
    sub = players[players["id"].isin(squad_ids)]
    for tm in sub["team_name"].dropna().tolist():
        counts[tm] = counts.get(tm, 0) + 1
    return counts


def propose_transfers(
    players: pd.DataFrame,
    ep_by_gw: Dict[int, pd.DataFrame],
    current_squad_ids: List[int],
    gw_start: int,
    horizon: int,
    bank: float,
    free_transfers: int,
) -> Dict[str, object]:
    """Suggest up to `free_transfers` greedy upgrades over a `horizon` of GWs.

    Replaces the lowest-EP-window XI member with the best affordable upgrade
    that respects the 3-per-team constraint.
    """
    if not current_squad_ids or free_transfers <= 0:
        return {"suggestions": [], "notes": "No squad provided or 0 FTs"}
    # Build horizon EP per player (sum over gw window)
    window = [gw for gw in range(gw_start, gw_start + horizon) if gw in ep_by_gw]
    if not window:
        return {"suggestions": [], "notes": "No EP window available"}
    ep_sum = _build_ep_window(ep_by_gw, window)
    pool = players.merge(ep_sum, on="id", how="left").fillna({"ep_window": 0.0})

    # Current XI for first GW in window
    first_gw = window[0]
    first_df = ep_by_gw[first_gw]
    _, current_xi_ids = _best_xi_ep_for_squad(current_squad_ids, first_df)
    if not current_xi_ids:
        return {"suggestions": [], "notes": "Could not form XI from provided squad"}

    # Greedy loop: replace lowest ep_window XI member with the best affordable upgrade
    suggestions = []
    squad_ids = current_squad_ids.copy()
    team_counts = _squad_team_counts(squad_ids, players)
    available_bank = float(bank)
    for _ in range(free_transfers):
        xi_pool = pool[pool["id"].isin(current_xi_ids)].copy()
        xi_pool = xi_pool.merge(
            players[["id", "price", "team_name"]], on="id", how="left"
        )
        out_row = xi_pool.nsmallest(1, "ep_window")
        if out_row.empty:
            break
        out_id = int(out_row.iloc[0]["id"])
        out_price = (
            float(out_row.iloc[0]["price"])
            if not pd.isna(out_row.iloc[0]["price"])
            else 0.0
        )
        # Candidates not owned, respect team limit
        cand = pool[~pool["id"].isin(squad_ids)].copy()
        cand = cand.merge(
            players[["id", "web_name", "team_name", "price"]], on="id", how="left"
        )
        # Enforce 3-per-team
        cand = cand[
            cand.apply(lambda r: team_counts.get(r["team_name"], 0) < 3, axis=1)
        ]
        # Budget check
        cand = cand[cand["price"].fillna(0.0) <= available_bank + out_price + 1e-6]
        if cand.empty:
            break
        # Score candidate by EP gain over window
        out_epw = (
            float(out_row.iloc[0]["ep_window"])
            if not pd.isna(out_row.iloc[0]["ep_window"])
            else 0.0
        )
        cand["ep_gain"] = cand["ep_window"] - out_epw
        best = cand.sort_values(["ep_gain", "ep_window"], ascending=False).head(1)
        if best.empty or float(best.iloc[0]["ep_gain"]) <= 0.0:
            break
        in_id = int(best.iloc[0]["id"])
        in_price = (
            float(best.iloc[0]["price"]) if not pd.isna(best.iloc[0]["price"]) else 0.0
        )
        suggestions.append(
            {
                "out_id": out_id,
                "in_id": in_id,
                "out_name": (
                    players.loc[players["id"] == out_id, "web_name"].values.tolist()[0]
                    if (players["id"] == out_id).any()
                    else ""
                ),
                "in_name": (
                    players.loc[players["id"] == in_id, "web_name"].values.tolist()[0]
                    if (players["id"] == in_id).any()
                    else ""
                ),
                "ep_gain_window": float(best.iloc[0]["ep_gain"]),
                "cost_in": in_price,
                "cost_out": out_price,
            }
        )
        # Apply move to state
        squad_ids = [pid for pid in squad_ids if pid != out_id] + [in_id]
        team_counts[best.iloc[0]["team_name"]] = (
            team_counts.get(best.iloc[0]["team_name"], 0) + 1
        )
        out_tm = players.loc[players["id"] == out_id, "team_name"].values.tolist()
        if out_tm:
            team_counts[out_tm[0]] = max(0, team_counts.get(out_tm[0], 0) - 1)
        available_bank = available_bank + out_price - in_price
        # Recompute current XI for next iteration
        _, current_xi_ids = _best_xi_ep_for_squad(squad_ids, first_df)

    return {
        "suggestions": suggestions,
        "bank_remaining": round(available_bank, 1),
        "notes": "Greedy 1-step upgrades over EP window",
    }


def chip_heuristics(
    gw: int, suggestions: Dict[str, object], free_transfers: int
) -> str:
    """Light-touch chip suggestion based on GW and projected gains."""
    # Simple placeholders aligned with the brief; adapt in-season with BGW/DGW detection
    if gw == 16:
        return (
            "GW16 AFCON top-up: you have up to 5 free transfers – use as mini-wildcard."
        )
    if (
        7 <= gw <= 12
        and free_transfers < 2
        and suggestions.get("suggestions")
        and sum(s.get("ep_gain_window", 0.0) for s in suggestions["suggestions"])
        >= 12.0
    ):
        return "Consider Wildcard this week to capture fixture swing (projected gain >= 12 over 4 GWs)."
    return "No chip recommended by heuristics."


def main():
    """CLI entrypoint: loads sources, computes outputs, writes per-GW files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", help="local bootstrap-static.json")
    ap.add_argument(
        "--bootstrap-url",
        help="https://fantasy.premierleague.com/api/bootstrap-static/",
    )
    ap.add_argument("--fixtures", help="local fixtures.json (from /api/fixtures/)")
    ap.add_argument(
        "--fixtures-url", help="https://fantasy.premierleague.com/api/fixtures/"
    )
    ap.add_argument("--merged-gw", help="Vaastav merged_gw.csv")
    ap.add_argument("--merged-gw-url")
    ap.add_argument("--players-raw", help="Vaastav players_raw.csv")
    ap.add_argument("--players-raw-url")
    ap.add_argument("--season", help="Season folder under Vaastav data, e.g., 2024-25")
    ap.add_argument(
        "--fpl-data-root",
        help="Path to Vaastav data directory (defaults to repo's Fantasy-Premier-League/data)",
    )
    ap.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow fallback to GitHub raw URLs when local CSVs are missing",
    )
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--risk-mode", choices=["protect", "chase"], default="protect")
    ap.add_argument(
        "--horizon", type=int, default=4, help="Transfer evaluation window in GWs"
    )
    ap.add_argument("--current-squad", help="CSV of current 15 (id or web_name column)")
    ap.add_argument(
        "--bank", type=float, default=0.0, help="Bank in millions for transfers"
    )
    ap.add_argument(
        "--free-transfers",
        type=int,
        default=1,
        help="Number of free transfers available",
    )
    ap.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "season_outputs"),
        help="Directory for per-GW outputs",
    )
    args = ap.parse_args()

    bs, fixtures, mgw, _praw = load_sources(args)
    if fixtures is None:
        raise SystemExit(
            "Fixtures required for season planning. Pass --fixtures or --fixtures-url"
        )
    players = add_player_features(bs, mgw)

    ep_by_gw = ep_per_gw(players, fixtures, gw_range=range(1, 39))

    os.makedirs(args.output_dir, exist_ok=True)

    # Optionally load current squad for transfer suggestions
    current_squad_ids = _load_current_squad(args.current_squad, players)

    for gw, df in ep_by_gw.items():
        # Top-5 by position CSV
        top = df.sort_values(["ep"], ascending=False).groupby("position").head(5)
        top.to_csv(
            os.path.join(args.output_dir, f"gw_{gw:02d}_top5_by_pos.csv"), index=False
        )

        # Captain shortlist
        caps = captain_shortlist(
            ep_by_gw, players, gw, risk_mode=args.risk_mode, top_k=5
        )

        # Transfer suggestions (evaluated from this GW over horizon)
        transfers = propose_transfers(
            players=players,
            ep_by_gw=ep_by_gw,
            current_squad_ids=current_squad_ids,
            gw_start=gw,
            horizon=max(1, int(args.horizon)),
            bank=float(args.bank),
            free_transfers=int(
                args.free_transfers if gw != 16 else max(args.free_transfers, 5)
            ),
        )

        # Chip heuristics
        chip_note = chip_heuristics(gw, transfers, int(args.free_transfers))

        # Per-GW JSON plan
        plan = {
            "gw": gw,
            "risk_mode": args.risk_mode,
            "top5_by_pos": {
                pos: (
                    top[top["position"] == pos][
                        ["id", "web_name", "team_name", "price", "ep"]
                    ]
                    .reset_index(drop=True)
                    .to_dict(orient="records")
                )
                for pos in ["GK", "DEF", "MID", "FWD"]
            },
            "captain_shortlist": (
                caps.to_dict(orient="records") if not caps.empty else []
            ),
            "transfer_suggestions": transfers,
            "chip_suggestion": chip_note,
        }
        with open(
            os.path.join(args.output_dir, f"gw_{gw:02d}_plan.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(plan, f, indent=2)

    print(f"Wrote per-GW CSVs and JSON plans to {args.output_dir}.")


if __name__ == "__main__":
    main()
