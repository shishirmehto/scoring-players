#!/usr/bin/env python3
"""Captain shortlist with EO-aware risk tilt."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def captain_shortlist(
    ep_by_gw: Dict[int, pd.DataFrame],
    players: pd.DataFrame,
    gw: int,
    risk_mode: str = "protect",
    top_k: int = 5,
    *,
    def_guardrail_margin: float = 1.5,
) -> pd.DataFrame:
    df = ep_by_gw.get(gw, pd.DataFrame()).copy()
    if df.empty:
        return df
    base = players[["id", "selected_by_percent", "position"]].copy()
    df = df.merge(base, on="id", how="left", suffixes=("", "_pl"))

    # Exclude GK from captaincy candidates
    df = df[df["position"] != "GK"].copy()

    # EO proxy from ownership; compute z-score for tilt
    df["ownership"] = (
        pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0.0) / 100.0
    )
    mean_own = float(df["ownership"].mean()) if len(df) else 0.0
    std_own = float(df["ownership"].std()) if len(df) and df["ownership"].std() > 0 else 1.0
    df["eo_z"] = (df["ownership"] - mean_own) / std_own

    # Risk tilt
    if risk_mode == "chase":
        # Downweight high-EO; upweight low-EO
        df["cap_score"] = df["ep"] * (1.0 - 0.20 * df["eo_z"].clip(lower=-2.0, upper=2.0))
    else:
        # Protect: upweight high-EO
        df["cap_score"] = df["ep"] * (1.0 + 0.20 * df["eo_z"].clip(lower=-2.0, upper=2.0))

    # DEF guardrail in protect mode: require margin over best attacker
    if risk_mode == "protect":
        if not df[df["position"].isin(["MID", "FWD"])].empty:
            best_att_ep = float(df[df["position"].isin(["MID", "FWD"])]["ep"].max())
            is_def = df["position"] == "DEF"
            guard_fail = is_def & (df["ep"] < (best_att_ep + def_guardrail_margin))
            # Strongly downweight DEFs that don't clear the margin
            df.loc[guard_fail, "cap_score"] = df.loc[guard_fail, "cap_score"] * 0.5

    df.sort_values(["cap_score", "ep"], ascending=False, inplace=True)
    return df.head(top_k)[
        ["id", "web_name", "team_name", "position", "ep", "ownership", "cap_score"]
    ]
