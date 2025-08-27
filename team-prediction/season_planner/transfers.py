#!/usr/bin/env python3
"""Transfer planning including greedy 1-move and 2-move bundles."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _best_xi_ep_for_squad(
    squad_ids: List[int], ep_df: pd.DataFrame
) -> Tuple[float, List[int]]:
    if not squad_ids:
        return 0.0, []
    team = ep_df[ep_df["id"].isin(squad_ids)].copy()
    if team.empty:
        return 0.0, []
    gk = team[team["position"] == "GK"].nlargest(1, "ep")
    xi_ids = set(gk["id"].tolist())
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
    remaining = team[~team["id"].isin(xi_ids)].nlargest(11 - len(xi_ids), "ep")
    xi_ids.update(remaining["id"].tolist())
    xi_df = team[team["id"].isin(xi_ids)]
    return float(xi_df["ep"].sum()), list(xi_ids)


def _squad_team_counts(squad_ids: List[int], players: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    sub = players[players["id"].isin(squad_ids)]
    for tm in sub["team_name"].dropna().tolist():
        counts[tm] = counts.get(tm, 0) + 1
    return counts


def _build_ep_window(
    ep_by_gw: Dict[int, pd.DataFrame], window: List[int]
) -> pd.DataFrame:
    ep_sum = None
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


def propose_transfers(
    *,
    players: pd.DataFrame,
    ep_by_gw: Dict[int, pd.DataFrame],
    current_squad_ids: List[int],
    gw_start: int,
    horizon: int,
    bank: float,
    free_transfers: int,
    allow_two_move: bool = True,
) -> Dict[str, object]:
    if not current_squad_ids or free_transfers <= 0:
        return {"suggestions": [], "notes": "No squad provided or 0 FTs"}

    window = [gw for gw in range(gw_start, gw_start + horizon) if gw in ep_by_gw]
    if not window:
        return {"suggestions": [], "notes": "No EP window available"}
    ep_sum = _build_ep_window(ep_by_gw, window)
    pool = players.merge(ep_sum, on="id", how="left").fillna({"ep_window": 0.0})

    first_gw = window[0]
    first_df = ep_by_gw[first_gw]
    _, current_xi_ids = _best_xi_ep_for_squad(current_squad_ids, first_df)
    if not current_xi_ids:
        return {"suggestions": [], "notes": "Could not form XI from provided squad"}

    suggestions: List[Dict[str, object]] = []
    squad_ids = current_squad_ids.copy()
    team_counts = _squad_team_counts(squad_ids, players)
    available_bank = float(bank)

    def _one_move(squad_ids_local, team_counts_local, bank_local):
        xi_pool = pool[pool["id"].isin(current_xi_ids)].copy()
        xi_pool = xi_pool.merge(
            players[["id", "price", "team_name"]], on="id", how="left"
        )
        out_row = xi_pool.nsmallest(1, "ep_window")
        if out_row.empty:
            return None
        out_id = int(out_row.iloc[0]["id"])
        out_price = (
            float(out_row.iloc[0]["price"])
            if not pd.isna(out_row.iloc[0]["price"])
            else 0.0
        )
        cand = pool[~pool["id"].isin(squad_ids_local)].copy()
        cand = cand.merge(
            players[["id", "web_name", "team_name", "price"]], on="id", how="left"
        )
        cand = cand[
            cand.apply(lambda r: team_counts_local.get(r["team_name"], 0) < 3, axis=1)
        ]
        cand = cand[cand["price"].fillna(0.0) <= bank_local + out_price + 1e-6]
        if cand.empty:
            return None
        out_epw = (
            float(out_row.iloc[0]["ep_window"])
            if not pd.isna(out_row.iloc[0]["ep_window"])
            else 0.0
        )
        cand["ep_gain"] = cand["ep_window"] - out_epw
        best = cand.sort_values(["ep_gain", "ep_window"], ascending=False).head(1)
        if best.empty or float(best.iloc[0]["ep_gain"]) <= 0.0:
            return None
        in_id = int(best.iloc[0]["id"])
        in_price = (
            float(best.iloc[0]["price"]) if not pd.isna(best.iloc[0]["price"]) else 0.0
        )
        move = {
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
        return move

    # Greedy loop for each free transfer; optionally evaluate a 2-move bundle
    if allow_two_move and free_transfers >= 2:
        first = _one_move(squad_ids, team_counts, available_bank)
        if first is not None:
            # Simulate applying first move and compute second
            new_squad = [pid for pid in squad_ids if pid != first["out_id"]] + [
                first["in_id"]
            ]
            new_team_counts = team_counts.copy()
            new_team_counts[
                players.loc[
                    players["id"] == first["in_id"], "team_name"
                ].values.tolist()[0]
            ] = (
                new_team_counts.get(
                    players.loc[
                        players["id"] == first["in_id"], "team_name"
                    ].values.tolist()[0],
                    0,
                )
                + 1
            )
            out_tm = players.loc[
                players["id"] == first["out_id"], "team_name"
            ].values.tolist()
            if out_tm:
                new_team_counts[out_tm[0]] = max(
                    0, new_team_counts.get(out_tm[0], 0) - 1
                )
            new_bank = (
                available_bank
                + float(first["cost_out"])
                - float(first["cost_in"])
                + 0.0
            )
            second = _one_move(new_squad, new_team_counts, new_bank)
            if (
                second is not None
                and (first["ep_gain_window"] + second["ep_gain_window"])
                > first["ep_gain_window"]
            ):
                suggestions.extend([first, second])
                return {
                    "suggestions": suggestions,
                    "bank_remaining": round(
                        new_bank + float(second["cost_out"]) - float(second["cost_in"]),
                        1,
                    ),
                    "notes": "Greedy 2-move bundle over EP window",
                }

    # Fallback to up-to-N one-move greedy
    for _ in range(min(free_transfers, 3)):
        move = _one_move(squad_ids, team_counts, available_bank)
        if move is None:
            break
        suggestions.append(move)
        squad_ids = [pid for pid in squad_ids if pid != move["out_id"]] + [
            move["in_id"]
        ]
        team_counts[
            players.loc[players["id"] == move["in_id"], "team_name"].values.tolist()[0]
        ] = (
            team_counts.get(
                players.loc[
                    players["id"] == move["in_id"], "team_name"
                ].values.tolist()[0],
                0,
            )
            + 1
        )
        out_tm = players.loc[
            players["id"] == move["out_id"], "team_name"
        ].values.tolist()
        if out_tm:
            team_counts[out_tm[0]] = max(0, team_counts.get(out_tm[0], 0) - 1)
        available_bank = (
            available_bank + float(move["cost_out"]) - float(move["cost_in"])
        )
    return {
        "suggestions": suggestions,
        "bank_remaining": round(available_bank, 1),
        "notes": "Greedy 1-step upgrades over EP window",
    }
