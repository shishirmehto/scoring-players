
#!/usr/bin/env python3
# (See previous message for full header docstring)

import argparse, os, json
from typing import Optional
import pandas as pd

try:
    import requests
except Exception:
    requests = None

 

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
    if path:
        return pd.read_csv(path)
    if url:
        return pd.read_csv(url)
    raise FileNotFoundError("Provide a local path or a URL")

def load_sources(args):
    bs = _load_json(args.bootstrap, args.bootstrap_url)
    fixtures = _load_json(args.fixtures, args.fixtures_url) if (args.fixtures or args.fixtures_url) else None
    mgw = _load_csv(args.merged_gw, args.merged_gw_url)
    praw = _load_csv(args.players_raw, args.players_raw_url)
    return bs, fixtures, mgw, praw

def add_player_features(bootstrap: dict, merged_gw: pd.DataFrame) -> pd.DataFrame:
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id","name"]].rename(columns={"id":"team_id","name":"team_name"})
    df = elements.merge(teams, left_on="team", right_on="team_id", how="left")

    mgw = merged_gw.copy()
    if "element" not in mgw.columns:
        if "id" in mgw.columns:
            mgw = mgw.rename(columns={"id":"element"})
        else:
            raise ValueError("merged_gw missing 'element' id column")

    if "round" in mgw.columns:
        mgw["round"] = pd.to_numeric(mgw["round"], errors="coerce")

    feats = []
    for pid, g in mgw.groupby("element"):
        g = g.sort_values("round")
        last4 = g.tail(4)
        feats.append({
            "id": pid,
            "form4": last4["total_points"].astype(float).mean() if "total_points" in last4.columns and len(last4)>0 else 0.0,
            "ict4": last4["ict_index"].astype(float).mean() if "ict_index" in last4.columns and len(last4)>0 else 0.0,
            "mins60": (last4["minutes"]>=60).mean() if "minutes" in last4.columns and len(last4)>0 else 0.0,
            "returns4": (last4["total_points"].astype(float)>=5).mean() if "total_points" in last4.columns and len(last4)>0 else 0.0,
            "cs4": (last4["goals_conceded"].astype(float)==0).mean() if "goals_conceded" in last4.columns and len(last4)>0 else 0.0,
            "saves90_4": (last4["saves"].sum()/ (last4["minutes"].sum()/90.0)) if ("saves" in last4.columns and last4["minutes"].sum()>0) else 0.0,
        })
    feat_df = pd.DataFrame(feats)

    df = df.merge(feat_df, on="id", how="left")
    for col in ["ep_next","form","ict_index","selected_by_percent","now_cost","chance_of_playing_next_round"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["position"] = df["element_type"].map({1:"GK",2:"DEF",3:"MID",4:"FWD"})
    df["price"] = (df["now_cost"]/10.0).fillna(0.0)
    return df

def availability_penalty(status, chance) -> float:
    pen = 0.0
    if status in ("i","s"): pen += 1.0
    if status == "d": pen += 0.3
    try:
        c = float(chance)
        if c < 75: pen += 0.4
        elif c < 100: pen += 0.1
    except Exception:
        pass
    return pen

def ep_per_gw(df_players: pd.DataFrame, fixtures_json: dict, gw_range=range(1,39)):
    fx = pd.DataFrame(fixtures_json)
    fx = fx[["event","team_h","team_a","team_h_difficulty","team_a_difficulty"]].dropna()
    fx["event"] = fx["event"].astype(int)
    team_fixt = {}
    for _, r in fx.iterrows():
        team_fixt.setdefault(int(r["team_h"]), {})[int(r["event"])] = {"H": int(r["team_a"]), "diff": float(r["team_h_difficulty"])}
        team_fixt.setdefault(int(r["team_a"]), {})[int(r["event"])] = {"A": int(r["team_h"]), "diff": float(r["team_a_difficulty"])}

    out = {}
    for gw in gw_range:
        rows = []
        for _, p in df_players.iterrows():
            team_id = int(p["team"])
            fi = team_fixt.get(team_id, {}).get(gw, None)
            if fi is None:
                continue

            xmins = 35 + 55 * float(p.get("mins60", 0.5))
            base = 0.55*float(p.get("form",0)) + 0.25*float(p.get("form4",0)) + 0.15*(float(p.get("ict_index",0))/10.0) + 0.05*(float(p.get("ict4",0))/10.0)
            diff = float(fi.get("diff", 3.0)) if isinstance(fi, dict) else 3.0
            fixture_adj = 1.0 + (3.0 - diff)*0.06
            pos = p["position"]
            pos_boost = 0.0
            if pos == "GK":
                pos_boost += 0.25*float(p.get("cs4",0)) + 0.10*min(float(p.get("saves90_4",0))/5.0, 1.0)
            elif pos == "DEF":
                pos_boost += 0.30*float(p.get("cs4",0))
            elif pos == "MID":
                pos_boost += 0.15*float(p.get("returns4",0))
            elif pos == "FWD":
                pos_boost += 0.20*float(p.get("returns4",0))
            ep = (xmins/90.0) * base * fixture_adj * (1.0 + pos_boost)
            ep -= availability_penalty(p.get("status"), p.get("chance_of_playing_next_round"))
            rows.append({"id": p["id"], "web_name": p["web_name"], "team_id": team_id, "team_name": p["team_name"],
                         "position": pos, "price": p["price"], "ep": max(ep,0.0)})
        out[gw] = pd.DataFrame(rows)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", help="local bootstrap-static.json")
    ap.add_argument("--bootstrap-url", help="https://fantasy.premierleague.com/api/bootstrap-static/")
    ap.add_argument("--fixtures", help="local fixtures.json (from /api/fixtures/)")
    ap.add_argument("--fixtures-url", help="https://fantasy.premierleague.com/api/fixtures/")
    ap.add_argument("--merged-gw", help="Vaastav merged_gw.csv")
    ap.add_argument("--merged-gw-url")
    ap.add_argument("--players-raw", help="Vaastav players_raw.csv")
    ap.add_argument("--players-raw-url")
    ap.add_argument("--budget", type=float, default=100.0)
    args = ap.parse_args()

    bs, fixtures, mgw, _praw = load_sources(args)
    if fixtures is None:
        raise SystemExit("Fixtures required for season planning. Pass --fixtures or --fixtures-url")
    players = add_player_features(bs, mgw)

    ep_by_gw = ep_per_gw(players, fixtures, gw_range=range(1,39))

    # Starter: just write per-GW top picks (no MILP here to stay lightweight).
    # You can combine with the optimizer from fpl_team_builder_with_vaastav.py later.
    os.makedirs("season_outputs", exist_ok=True)
    for gw, df in ep_by_gw.items():
        top = df.sort_values(["ep"], ascending=False).groupby("position").head(5)
        top.to_csv(f"season_outputs/gw_{gw:02d}_top5_by_pos.csv", index=False)
    print("Wrote season_outputs/gw_XX_top5_by_pos.csv for all 38 GWs.")

if __name__ == "__main__":
    main()
