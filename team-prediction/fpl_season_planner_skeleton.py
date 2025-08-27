#!/usr/bin/env python3
"""Season planner and helper utilities for Fantasy Premier League (FPL).

Generates per–gameweek outputs:
- Top-5 players by position
- Captain shortlists (protect/chase modes)
- Transfer suggestions (supports two-move bundles)

Data can be loaded from local Vaastav CSVs or the live FPL API with fallbacks.
Refactored into `season_planner` modules with opponent-split EP and AFCON support.
"""

import argparse, os, json
from typing import Optional, List, Dict, Tuple, Set
import pandas as pd
import numpy as np

# New modular imports
from season_planner import (
    data as sp_data,
    features as sp_features,
    ep_model as sp_ep,
    captain as sp_captain,
    transfers as sp_transfers,
    chips as sp_chips,
)


def _parse_id_list(text: Optional[str]) -> Set[int]:
    if not text:
        return set()
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    out: Set[int] = set()
    for p in parts:
        try:
            out.add(int(p))
        except Exception:
            continue
    return out


def _parse_gw_list(text: Optional[str]) -> List[int]:
    if not text:
        return []
    tokens = [t.strip() for t in str(text).replace(";", ",").split(",") if t.strip()]
    weeks: List[int] = []
    for tok in tokens:
        if "-" in tok:
            try:
                a, b = tok.split("-", 1)
                a_i, b_i = int(a), int(b)
                weeks.extend(list(range(min(a_i, b_i), max(a_i, b_i) + 1)))
            except Exception:
                continue
        else:
            try:
                weeks.append(int(tok))
            except Exception:
                continue
    # deduplicate preserving order
    seen = set()
    uniq = []
    for w in weeks:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def _load_csv(path: Optional[str], url: Optional[str] = None) -> pd.DataFrame:
    # Backwards-compat shim; now delegated to season_planner.data
    return pd.read_csv(path or url)  # type: ignore[arg-type]


# ===== Data acquisition and resolution =====


def load_sources(args):
    """Load bootstrap, fixtures, merged_gw and players_raw using args and Vaastav helpers.

    Resolution order:
    - If explicit local path or URL provided via args, use that
    - Otherwise, try Fantasy-Premier-League/getters.py for live JSON (bootstrap, fixtures)
    - For CSVs, resolve season paths under --fpl-data-root and --season
      If not found and --allow-remote, fall back to GitHub raw URLs
    """
    # Delegate to modular loader
    return sp_data.load_sources(args)


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
    # Delegate to modular feature builder
    return sp_features.add_player_features(bootstrap, merged_gw)


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
    # Delegate to modular captain selection
    return sp_captain.captain_shortlist(
        ep_by_gw, players, gw, risk_mode=risk_mode, top_k=top_k
    )


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
    *,
    allow_two_move: bool = True,
) -> Dict[str, object]:
    # Delegate to modular transfer planner
    return sp_transfers.propose_transfers(
        players=players,
        ep_by_gw=ep_by_gw,
        current_squad_ids=current_squad_ids,
        gw_start=gw_start,
        horizon=horizon,
        bank=bank,
        free_transfers=free_transfers,
        allow_two_move=allow_two_move,
    )


def chip_heuristics(
    gw: int, suggestions: Dict[str, object], free_transfers: int
) -> str:
    # Delegate to modular chip heuristics
    return sp_chips.chip_heuristics(gw, suggestions, free_transfers)


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
    ap.add_argument(
        "--opponent-split",
        action="store_true",
        help="Use opponent-split EP adjustments (attack vs defence).",
    )
    ap.add_argument(
        "--afcon-player-ids",
        help="Comma-separated list of player ids away at AFCON (GW17-22).",
    )
    ap.add_argument(
        "--afcon-gws",
        default="17-22",
        help="GW list or ranges where AFCON players are unavailable (e.g., 17-22).",
    )
    ap.add_argument(
        "--pre-reset-downweight-gws",
        default="15",
        help="GW list or ranges to slightly downweight AFCON players before reset (e.g., 15).",
    )
    ap.add_argument(
        "--allow-two-move",
        action="store_true",
        help="Enable simple 2-move bundle search for transfers.",
    )
    args = ap.parse_args()

    bs, fixtures, mgw, _praw = load_sources(args)
    if fixtures is None:
        raise SystemExit(
            "Fixtures required for season planning. Pass --fixtures or --fixtures-url"
        )
    players = add_player_features(bs, mgw)

    afcon_ids = _parse_id_list(getattr(args, "afcon_player_ids", None))
    afcon_weeks = _parse_gw_list(getattr(args, "afcon_gws", None))
    pre_weeks = _parse_gw_list(getattr(args, "pre_reset_downweight_gws", None))
    ep_by_gw = sp_ep.ep_per_gw(
        players,
        fixtures,
        gw_range=range(1, 39),
        opponent_split=bool(args.opponent_split),
        afcon_player_ids=afcon_ids,
        afcon_gws=afcon_weeks,
        pre_reset_downweight_gws=pre_weeks,
    )

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
        caps = sp_captain.captain_shortlist(
            ep_by_gw, players, gw, risk_mode=args.risk_mode, top_k=5
        )

        # Transfer suggestions (evaluated from this GW over horizon)
        transfers = sp_transfers.propose_transfers(
            players=players,
            ep_by_gw=ep_by_gw,
            current_squad_ids=current_squad_ids,
            gw_start=gw,
            horizon=max(1, int(args.horizon)),
            bank=float(args.bank),
            free_transfers=int(
                args.free_transfers if gw != 16 else max(args.free_transfers, 5)
            ),
            allow_two_move=bool(args.allow_two_move),
        )

        # Chip heuristics
        chip_note = sp_chips.chip_heuristics(gw, transfers, int(args.free_transfers))

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
