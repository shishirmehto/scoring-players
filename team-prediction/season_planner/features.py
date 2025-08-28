#!/usr/bin/env python3
"""Player feature engineering for FPL season planner.

Computes short-horizon form features from Vaastav merged_gw and joins with
bootstrap elements and teams to produce a player table with position/price.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np


def _minutes60_prob_last_n(df_last_n: pd.DataFrame) -> float:
    if len(df_last_n) == 0 or "minutes" not in df_last_n.columns:
        return 0.0
    return (df_last_n["minutes"] >= 60).mean()


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
    elements = (
        pd.DataFrame(bootstrap["elements"])
        if isinstance(bootstrap, dict)
        else pd.DataFrame()
    )
    teams = (
        pd.DataFrame(bootstrap.get("teams", []))[["id", "name"]].rename(
            columns={"id": "team_id", "name": "team_name"}
        )
        if isinstance(bootstrap, dict)
        else pd.DataFrame(columns=["team_id", "team_name"])
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

    feats: List[dict] = []
    for pid, g in mgw.groupby("element"):
        g = g.sort_values("round")
        last6 = g.tail(6)
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
                "mins60": (
                    _minutes60_prob_last_n(last6)
                    if len(last6) > 0 and "minutes" in last6.columns
                    else _minutes60_prob_last_n(last4)
                ),
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
