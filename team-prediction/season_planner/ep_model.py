#!/usr/bin/env python3
"""Expected points model with opponent-split adjustments and AFCON handling.

This module provides a vectorized EP computation per GW, reusing fixture
difficulty as a prior and separating attacking vs defensive adjustments.
It also supports AFCON-aware availability by downweighting or zeroing minutes
for flagged players over specified GWs.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd


def _fixture_difficulty_df(fixtures_json: dict) -> pd.DataFrame:
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


def _availability_penalty(status: pd.Series, chance: pd.Series) -> np.ndarray:
    status_pen = status.map({"i": 1.0, "s": 1.0, "d": 0.3}).fillna(0.0)
    chance_num = pd.to_numeric(chance, errors="coerce").fillna(100.0)
    chance_pen = np.where(
        chance_num < 75.0, 0.4, np.where(chance_num < 100.0, 0.1, 0.0)
    )
    return (status_pen + chance_pen).to_numpy()


def _minutes_proxy(mins60: pd.Series) -> pd.Series:
    return 35.0 + 55.0 * pd.to_numeric(mins60, errors="coerce").fillna(0.5)


def _base_form(df_players: pd.DataFrame) -> pd.Series:
    return (
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


def _position_boost(df_players: pd.DataFrame) -> pd.Series:
    cs4 = pd.to_numeric(df_players.get("cs4", 0.0), errors="coerce").fillna(0.0)
    saves90_4 = pd.to_numeric(df_players.get("saves90_4", 0.0), errors="coerce").fillna(
        0.0
    )
    returns4 = pd.to_numeric(df_players.get("returns4", 0.0), errors="coerce").fillna(
        0.0
    )
    position = df_players["position"].astype(str)

    pos_boost = pd.Series(0.0, index=df_players.index)
    is_gk = position == "GK"
    pos_boost.loc[is_gk] = 0.25 * cs4.loc[is_gk] + 0.10 * np.minimum(
        saves90_4.loc[is_gk] / 5.0, 1.0
    )
    is_def = position == "DEF"
    pos_boost.loc[is_def] = 0.30 * cs4.loc[is_def]
    is_mid = position == "MID"
    pos_boost.loc[is_mid] = 0.15 * returns4.loc[is_mid]
    is_fwd = position == "FWD"
    pos_boost.loc[is_fwd] = 0.20 * returns4.loc[is_fwd]
    return pos_boost


def ep_per_gw(
    df_players: pd.DataFrame,
    fixtures_json: dict,
    gw_range: Iterable[int] = range(1, 39),
    *,
    opponent_split: bool = True,
    afcon_player_ids: Optional[Set[int]] = None,
    afcon_gws: Optional[Iterable[int]] = None,
    pre_reset_downweight_gws: Optional[Iterable[int]] = None,
) -> Dict[int, pd.DataFrame]:
    """Vectorized expected points per GW with optional opponent-split and AFCON.

    - opponent_split: applies stronger adjustment for attackers (att FDR) and
      milder one for defenders/keepers (def FDR). Using fixture difficulty as
      a proxy until dedicated team strengths are provided.
    - afcon_player_ids/afcon_gws: set EP to 0 for these players in afcon_gws.
    - pre_reset_downweight_gws: downweight AFCON players slightly in weeks
      leading up to the reset (e.g., GW15->GW16).
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

    base_series = _base_form(df_players)
    xmins_series = _minutes_proxy(df_players.get("mins60", 0.5))
    pos_boost = _position_boost(df_players)

    avail_pen = _availability_penalty(
        df_players.get("status", pd.Series(index=df_players.index, dtype=object)),
        df_players.get("chance_of_playing_next_round", 100),
    )

    base_inputs = pd.DataFrame(
        {
            "id": df_players["id"].astype(int),
            "web_name": df_players["web_name"],
            "team": pd.to_numeric(df_players["team"], errors="coerce")
            .fillna(0)
            .astype(int),
            "team_name": df_players["team_name"],
            "position": df_players["position"].astype(str),
            "price": pd.to_numeric(df_players["price"], errors="coerce").fillna(0.0),
            "xmins": xmins_series,
            "base": base_series,
            "pos_boost": pos_boost,
            "avail_pen": avail_pen,
        }
    )

    afcon_ids = set(afcon_player_ids or set())
    afcon_weeks = set(afcon_gws or set())
    pre_weeks = set(pre_reset_downweight_gws or set())

    for gw in gw_range:
        fi = fixt_df[fixt_df["event"] == int(gw)][["team_id", "diff"]]
        if fi.empty:
            out[gw] = pd.DataFrame(columns=["id", "web_name", "team_id", "team_name", "position", "price", "ep"])  # type: ignore[assignment]
            continue
        df = base_inputs.merge(fi, left_on="team", right_on="team_id", how="inner")
        diff = pd.to_numeric(df["diff"], errors="coerce").fillna(3.0)
        if opponent_split:
            # Stronger for attackers, milder for defenders
            att_adj = 1.0 + (3.0 - diff) * 0.08
            def_adj = 1.0 + (3.0 - diff) * 0.05
            is_att = df["position"].isin(["MID", "FWD"])
            fixture_adj = np.where(is_att, att_adj, def_adj)
        else:
            fixture_adj = 1.0 + (3.0 - diff) * 0.06

        ep = (df["xmins"] / 90.0) * df["base"] * fixture_adj * (
            1.0 + df["pos_boost"]
        ) - df["avail_pen"]

        # AFCON handling
        if afcon_ids and ((gw in afcon_weeks) or (gw in pre_weeks)):
            mask = df["id"].isin(list(afcon_ids))
            if gw in afcon_weeks:
                ep = ep.mask(mask, 0.0)
            elif gw in pre_weeks:
                ep = ep.where(~mask, ep * 0.8)  # slight downweight pre-reset

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
