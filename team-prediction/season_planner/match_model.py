#!/usr/bin/env python3
"""Lightweight per-match goal/CS predictor using decayed team strengths.

This is a pragmatic baseline to move from scalar FDR to match-level forecasts:
- Team attack/defence strengths estimated from Vaastav merged_gw with time-decay
- Poisson assumption for goals ⇒ CS probability = exp(-lambda_against)

It can be replaced later with a full Dixon–Coles or bivariate Poisson fit.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _exp_decay_weights(round_series: pd.Series, half_life_gw: float = 8.0) -> pd.Series:
    if round_series.empty:
        return round_series
    r = pd.to_numeric(round_series, errors="coerce").fillna(round_series.max())
    # More recent rounds get higher weight
    # decay = 0.5 ** (age / half_life)
    max_r = float(np.nanmax(r)) if len(r) else 0.0
    age = max_r - r
    decay = np.power(0.5, age / float(max(half_life_gw, 1.0)))
    return decay


def team_strengths_from_mgw(mgw: pd.DataFrame, half_life_gw: float = 8.0) -> pd.DataFrame:
    """Return DataFrame with columns [team_id, att_str, def_str].

    att_str ~ decayed average goals_scored; def_str ~ decayed average goals_conceded (lower is better).
    Normalized by league averages to be comparable across seasons.
    """
    req_cols = {"team", "goals_scored", "goals_conceded", "round"}
    if not req_cols.issubset(set(mgw.columns)):
        # Fallback to neutral strengths if data missing
        return pd.DataFrame(columns=["team_id", "att_str", "def_str"])

    df = mgw[["team", "goals_scored", "goals_conceded", "round"]].copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round"])  # ensure rounds present
    df["w"] = _exp_decay_weights(df["round"], half_life_gw=half_life_gw)

    # Weighted team metrics
    g = df.groupby("team")
    att = (g.apply(lambda x: np.average(pd.to_numeric(x["goals_scored"], errors="coerce").fillna(0.0), weights=x["w"]))
           .rename("att_raw"))
    dfn = (g.apply(lambda x: np.average(pd.to_numeric(x["goals_conceded"], errors="coerce").fillna(0.0), weights=x["w"]))
           .rename("def_raw"))
    out = pd.concat([att, dfn], axis=1).reset_index().rename(columns={"team": "team_id"})

    # Normalize by league averages
    league_att = float(out["att_raw"].mean()) if not out.empty else 1.3
    league_def = float(out["def_raw"].mean()) if not out.empty else 1.3
    out["att_str"] = (out["att_raw"] / max(league_att, 1e-6)).clip(lower=0.5, upper=1.8)
    out["def_str"] = (out["def_raw"] / max(league_def, 1e-6)).clip(lower=0.5, upper=1.8)
    return out[["team_id", "att_str", "def_str"]]


def predict_fixtures(fixtures_json: dict, strengths: pd.DataFrame) -> pd.DataFrame:
    """Return per-fixture predictions with columns:
    [event, team_id, opp_id, lambda_for, lambda_against, cs_prob]
    """
    fx = pd.DataFrame(fixtures_json)
    if fx.empty:
        return pd.DataFrame(columns=["event", "team_id", "opp_id", "lambda_for", "lambda_against", "cs_prob"])
    fx = fx[["event", "team_h", "team_a"]].dropna()
    fx["event"] = fx["event"].astype(int)
    strengths = strengths.copy()
    if strengths.empty:
        # Neutral strengths → lambda_for ~ 1.3 goals baseline
        home = fx[["event", "team_h"]].rename(columns={"team_h": "team_id"})
        away = fx[["event", "team_a"]].rename(columns={"team_a": "team_id"})
        base = pd.concat([home, away], ignore_index=True)
        base["opp_id"] = np.nan
        base["lambda_for"] = 1.3
        base["lambda_against"] = 1.3
        base["cs_prob"] = np.exp(-base["lambda_against"])  # P(opponent scores 0)
        return base

    s = strengths.rename(columns={"team_id": "tid"})
    # Build rows for both home and away teams
    rows = []
    for _, r in fx.iterrows():
        h = int(r["team_h"]); a = int(r["team_a"]); ev = int(r["event"])
        sh = s[s["tid"] == h]; sa = s[s["tid"] == a]
        if sh.empty or sa.empty:
            # Skip if strengths unavailable
            continue
        # Simple rate: league baseline scaled by attack vs opponent defence
        # baseline ~ 1.35 goals per team per match; mild home edge
        base = 1.35
        home_edge = 1.08
        lam_h = base * home_edge * float(sh["att_str"].iloc[0]) / float(sa["def_str"].iloc[0])
        lam_a = base * float(sa["att_str"].iloc[0]) / float(sh["def_str"].iloc[0])
        lam_h = float(np.clip(lam_h, 0.2, 3.2))
        lam_a = float(np.clip(lam_a, 0.2, 3.2))
        rows.append({"event": ev, "team_id": h, "opp_id": a, "lambda_for": lam_h, "lambda_against": lam_a, "cs_prob": float(np.exp(-lam_a))})
        rows.append({"event": ev, "team_id": a, "opp_id": h, "lambda_for": lam_a, "lambda_against": lam_h, "cs_prob": float(np.exp(-lam_h))})
    return pd.DataFrame(rows)


