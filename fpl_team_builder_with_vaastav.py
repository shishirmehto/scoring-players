
import os, json, math, argparse, sys
from typing import Optional, Dict, Tuple
import pandas as pd

try:
    import requests
except Exception:
    requests = None

try:
    import pulp
except Exception:
    pulp = None

def load_bootstrap_static(bootstrap_path: Optional[str], bootstrap_url: Optional[str]) -> dict:
    if bootstrap_path and os.path.exists(bootstrap_path):
        with open(bootstrap_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if bootstrap_url:
        if requests is None:
            raise RuntimeError("requests not available to fetch bootstrap url")
        r = requests.get(bootstrap_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        return r.json()
    raise FileNotFoundError("Provide --bootstrap (local file) or --bootstrap-url")

def load_csv(path: Optional[str], url: Optional[str]) -> pd.DataFrame:
    src = path or url
    if not src:
        raise FileNotFoundError("Provide either a local path or a URL for the CSV")
    # First attempt: default engine (fast C parser)
    try:
        return pd.read_csv(src)
    except Exception:
        # Fallback: more tolerant parser, skip malformed lines
        try:
            return pd.read_csv(src, engine="python", on_bad_lines="skip")
        except TypeError:
            # Older pandas without on_bad_lines
            return pd.read_csv(src, engine="python")

def element_type_to_pos(et: int) -> str:
    return {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(int(et), "UNK")

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def minutes60_prob(df_last4: pd.DataFrame) -> float:
    if len(df_last4) == 0:
        return 0.0
    return (df_last4["minutes"] >= 60).mean()

def clean_sheet_rate(df_last4: pd.DataFrame) -> float:
    if "clean_sheets" in df_last4.columns and len(df_last4) > 0:
        return df_last4["clean_sheets"].astype(float).mean()
    if "goals_conceded" in df_last4.columns and len(df_last4) > 0:
        return (df_last4["goals_conceded"].astype(float) == 0).mean()
    return 0.0

def ict_last4(df_last4: pd.DataFrame) -> float:
    if "ict_index" in df_last4.columns and len(df_last4) > 0:
        return df_last4["ict_index"].astype(float).mean()
    return 0.0

def returns_rate_last4(df_last4: pd.DataFrame) -> float:
    if "total_points" in df_last4.columns and len(df_last4) > 0:
        return (df_last4["total_points"].astype(float) >= 5).mean()
    return 0.0

def saves_per90_last4(df_last4: pd.DataFrame) -> float:
    if "saves" in df_last4.columns and len(df_last4) > 0:
        mins = df_last4["minutes"].astype(float).sum()
        saves = df_last4["saves"].astype(float).sum()
        if mins > 0:
            return (saves / (mins/90.0))
    return 0.0

def build_player_table(bootstrap: dict, merged_gw: pd.DataFrame, players_raw: pd.DataFrame) -> pd.DataFrame:
    elements = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id","name"]].rename(columns={"id":"team_id","name":"team_name"})
    elements = elements.merge(teams, left_on="team", right_on="team_id", how="left")

    mgw = merged_gw.copy()
    if "element" not in mgw.columns:
        key = "element" if "element" in mgw.columns else ("id" if "id" in mgw.columns else None)
        if key is None:
            raise ValueError("merged_gw missing 'element' or 'id' for player id mapping")
        mgw = mgw.rename(columns={key: "element"})

    if "round" in mgw.columns:
        mgw["round"] = pd.to_numeric(mgw["round"], errors="coerce")

    feats = []
    for pid, g in mgw.groupby("element"):
        g = g.sort_values("round")
        last4 = g.tail(4)
        feats.append({
            "id": pid,
            "form_last4": last4["total_points"].astype(float).mean() if "total_points" in last4.columns and len(last4)>0 else 0.0,
            "ict_last4": ict_last4(last4),
            "mins60_prob": minutes60_prob(last4),
            "returns_rate_last4": returns_rate_last4(last4),
            "cs_rate_last4": clean_sheet_rate(last4),
            "saves_p90_last4": saves_per90_last4(last4),
        })
    feat_df = pd.DataFrame(feats)

    df = elements.merge(feat_df, left_on="id", right_on="id", how="left")

    for col in ["ep_next","form","ict_index","selected_by_percent","now_cost","chance_of_playing_next_round"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    df["position"] = df["element_type"].map({1:"GK",2:"DEF",3:"MID",4:"FWD"})
    df["price"] = (df["now_cost"].astype(float) / 10.0).fillna(0.0)

    if "element_type" not in df.columns or df["position"].isnull().any():
        if "element_type" in players_raw.columns:
            pos_map = players_raw[["id","element_type"]].drop_duplicates()
            df = df.merge(pos_map, on="id", how="left", suffixes=("","_from_raw"))
            df["element_type"] = df["element_type"].fillna(df["element_type_from_raw"])
            df["position"] = df["element_type"].map({1:"GK",2:"DEF",3:"MID",4:"FWD"})

    return df

def availability_penalty(status, chance) -> float:
    penalty = 0.0
    if status in ("i","s"):
        penalty += 1.0
    if status == "d":
        penalty += 0.3
    try:
        if pd.notna(chance):
            c = float(chance)
            if c < 75: penalty += 0.4
            elif c < 100: penalty += 0.1
    except Exception:
        pass
    return penalty

def score_row(row: pd.Series) -> float:
    ep_next = 0.0
    ep_next = safe_float(row.get("ep_next"))
    form_boot = safe_float(row.get("form"))
    ict = safe_float(row.get("ict_index"))
    sel = safe_float(row.get("selected_by_percent"))
    ict_norm = ict / 10.0
    sel_norm = sel / 100.0

    form4 = safe_float(row.get("form_last4"))
    mins60 = safe_float(row.get("mins60_prob"))
    ict4 = safe_float(row.get("ict_last4")) / 10.0
    returns4 = safe_float(row.get("returns_rate_last4"))
    cs4 = safe_float(row.get("cs_rate_last4"))
    saves_p90 = safe_float(row.get("saves_p90_last4"))
    saves_comp = min(max(saves_p90, 0.0), 5.0) / 5.0

    pos = row.get("position","UNK")

    if pos == "FWD":
        base = (0.45*ep_next + 0.15*form_boot + 0.15*form4 +
                0.12*ict_norm + 0.08*ict4 + 0.05*mins60 + 0.05*returns4)
    elif pos == "MID":
        base = (0.42*ep_next + 0.15*form_boot + 0.18*form4 +
                0.12*ict_norm + 0.08*ict4 + 0.03*sel_norm + 0.02*mins60)
    elif pos == "DEF":
        base = (0.40*ep_next + 0.14*form_boot + 0.14*form4 +
                0.12*ict_norm + 0.08*ict4 + 0.08*cs4 + 0.04*sel_norm)
    elif pos == "GK":
        base = (0.42*ep_next + 0.16*form_boot + 0.12*form4 +
                0.06*ict_norm + 0.04*ict4 + 0.10*cs4 + 0.06*saves_comp + 0.04*sel_norm)
    else:
        base = (0.50*ep_next + 0.20*form_boot + 0.10*ict_norm + 0.20*form4)

    pen = availability_penalty(row.get("status"), row.get("chance_of_playing_next_round"))
    return float(base - pen)

def top5_per_position(df: pd.DataFrame):
    df = df.copy()
    df["score"] = df.apply(score_row, axis=1)
    buckets = {}
    outcols = ["web_name","team_name","price","score","ep_next","form","form_last4","ict_index","ict_last4","mins60_prob","cs_rate_last4","saves_p90_last4","selected_by_percent","status","chance_of_playing_next_round"]
    for pos in ["GK","DEF","MID","FWD"]:
        sub = df[df["position"] == pos].copy()
        sub.sort_values(["score","ep_next"], ascending=False, inplace=True)
        buckets[pos] = sub.head(5)[outcols]
    return buckets, df

def pick_squad(df_scored: pd.DataFrame, budget: float = 100.0) -> pd.DataFrame:
    pool = df_scored[["id","web_name","team_name","team","position","price","score"]].copy()
    pool["team"] = pool["team"].fillna(pool["team_name"])
    quotas = {"GK":2, "DEF":5, "MID":5, "FWD":3}

    if pulp is not None:
        prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{int(pid)}", lowBound=0, upBound=1, cat="Binary")
             for i, pid in enumerate(pool["id"])}
        prob += pulp.lpSum([pool.iloc[i]["score"] * x[i] for i in range(len(pool))])
        prob += pulp.lpSum([pool.iloc[i]["price"] * x[i] for i in range(len(pool))]) <= budget
        prob += pulp.lpSum([x[i] for i in range(len(pool))]) == 15
        for pos, q in quotas.items():
            idx = [i for i in range(len(pool)) if pool.iloc[i]["position"] == pos]
            prob += pulp.lpSum([x[i] for i in idx]) == q
        for tm in pool["team_name"].dropna().unique():
            idx = [i for i in range(len(pool)) if pool.iloc[i]["team_name"] == tm]
            prob += pulp.lpSum([x[i] for i in idx]) <= 3
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        chosen = [i for i in range(len(pool)) if pulp.value(x[i]) == 1]
        return pool.iloc[chosen].copy().sort_values("position")

    # greedy fallback
    squad = []
    counts = {"GK":0,"DEF":0,"MID":0,"FWD":0}
    team_counts = {}
    money = budget
    pool = pool.sort_values("score", ascending=False).reset_index(drop=True)
    for _, row in pool.iterrows():
        pos = row["position"]
        if counts[pos] >= quotas[pos]: continue
        tm = row["team_name"]
        if team_counts.get(tm,0) >= 3: continue
        if row["price"] > money + 1e-6: continue
        squad.append(row)
        counts[pos]+=1
        team_counts[tm]=team_counts.get(tm,0)+1
        money -= row["price"]
        if sum(counts.values())==15: break
    return pd.DataFrame(squad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", help="Path to local bootstrap-static.json")
    ap.add_argument("--bootstrap-url", help="URL to bootstrap-static endpoint")
    ap.add_argument("--merged-gw", help="Path to Vaastav merged_gw.csv")
    ap.add_argument("--merged-gw-url", help="URL to Vaastav merged_gw.csv")
    ap.add_argument("--players-raw", help="Path to Vaastav players_raw.csv")
    ap.add_argument("--players-raw-url", help="URL to Vaastav players_raw.csv")
    ap.add_argument("--budget", type=float, default=100.0)
    args = ap.parse_args()

    bootstrap = load_bootstrap_static(args.bootstrap, args.bootstrap_url)
    merged_gw = load_csv(args.merged_gw, args.merged_gw_url)
    players_raw = load_csv(args.players_raw, args.players_raw_url)

    df = build_player_table(bootstrap, merged_gw, players_raw)
    tops, df_scored = top5_per_position(df)

    mapping = {"GK":"top5_goalkeepers.csv","DEF":"top5_defenders.csv","MID":"top5_midfielders.csv","FWD":"top5_forwards.csv"}
    for pos, tbl in tops.items():
        tbl.to_csv(mapping[pos], index=False, encoding="utf-8")
        print(f"\nTop 5 {pos}:\n{tbl[['web_name','team_name','price','score']].to_string(index=False)}")

    squad = pick_squad(df_scored, budget=args.budget)
    if not squad.empty:
        squad.to_csv("recommended_squad.csv", index=False, encoding="utf-8")
        print(f"\nRecommended 15-man squad (spent £{squad['price'].sum():.1f}m of £{args.budget:.1f}m):")
        print(squad[['position','web_name','team_name','price','score']].to_string(index=False))
    else:
        print("\nCould not build a squad with the current settings. Try adjusting budget or inputs.")

if __name__ == "__main__":
    main()
