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
) -> pd.DataFrame:
    df = ep_by_gw.get(gw, pd.DataFrame()).copy()
    if df.empty:
        return df
    base = players[["id", "selected_by_percent", "position"]].copy()
    df = df.merge(base, on="id", how="left", suffixes=("", "_pl"))
    df["ownership"] = (
        pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0.0) / 100.0
    )
    if risk_mode == "chase":
        df["cap_score"] = df["ep"] * (1.0 + (0.30 * (1.0 - df["ownership"])))
    else:
        df["cap_score"] = df["ep"] * (1.0 - (0.20 * (1.0 - df["ownership"])))
    df.sort_values(["cap_score", "ep"], ascending=False, inplace=True)
    return df.head(top_k)[
        ["id", "web_name", "team_name", "position", "ep", "ownership", "cap_score"]
    ]
