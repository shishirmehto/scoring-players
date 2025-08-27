
#!/usr/bin/env python3
import json, os, sys, math
from typing import List, Dict, Callable

try:
    import requests
except Exception:
    requests = None

def load_bootstrap_static(src_json_path: str = None) -> dict:
    """
    Load FPL bootstrap-static JSON either from a local file or from the official endpoint.
    Uses only the single endpoint as requested.
    """
    if src_json_path and os.path.exists(src_json_path):
        with open(src_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    if requests is None:
        raise RuntimeError("requests not available and no local JSON provided.")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def next_event_id(data: dict) -> int:
    for ev in data.get("events", []):
        if ev.get("is_next"):
            return ev.get("id")
    upcoming = [ev for ev in data.get("events", []) if not ev.get("finished")]
    if upcoming:
        return sorted(upcoming, key=lambda e: e.get("id"))[0]["id"]
    return max(e["id"] for e in data.get("events", []))

def build_team_map(data: dict) -> Dict[int, str]:
    return {t["id"]: t["name"] for t in data.get("teams", [])}

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def availability_penalty(el: dict) -> float:
    status = el.get("status")
    chance = el.get("chance_of_playing_next_round")
    penalty = 0.0
    if status in ("i", "s"):
        penalty += 1.0
    if status == "d":
        penalty += 0.3
    if isinstance(chance, int):
        if chance < 75:
            penalty += 0.4
        elif chance < 100:
            penalty += 0.1
    return penalty

# ---------- Position-specific scoring functions (bootstrap-static only) ----------
# Common features from bootstrap-static: ep_next (string), form (string),
# ict_index (string), selected_by_percent (string), plus minutes/saves if needed.

def score_forward(el: dict) -> float:
    ep_next = safe_float(el.get("ep_next"))
    form = safe_float(el.get("form"))
    ict = safe_float(el.get("ict_index"))
    sel = safe_float(el.get("selected_by_percent"))
    ict_norm = ict / 10.0
    sel_norm = sel / 100.0
    base = 0.65*ep_next + 0.20*form + 0.12*ict_norm + 0.03*sel_norm
    return base - availability_penalty(el)

def score_midfielder(el: dict) -> float:
    ep_next = safe_float(el.get("ep_next"))
    form = safe_float(el.get("form"))
    ict = safe_float(el.get("ict_index"))
    sel = safe_float(el.get("selected_by_percent"))
    ict_norm = ict / 10.0
    sel_norm = sel / 100.0
    base = 0.60*ep_next + 0.22*form + 0.15*ict_norm + 0.03*sel_norm
    return base - availability_penalty(el)

def score_defender(el: dict) -> float:
    ep_next = safe_float(el.get("ep_next"))
    form = safe_float(el.get("form"))
    ict = safe_float(el.get("ict_index"))
    sel = safe_float(el.get("selected_by_percent"))
    ict_norm = ict / 10.0
    sel_norm = sel / 100.0
    base = 0.60*ep_next + 0.18*form + 0.18*ict_norm + 0.04*sel_norm
    return base - availability_penalty(el)

def score_goalkeeper(el: dict) -> float:
    ep_next = safe_float(el.get("ep_next"))
    form = safe_float(el.get("form"))
    ict = safe_float(el.get("ict_index"))
    sel = safe_float(el.get("selected_by_percent"))
    minutes = safe_float(el.get("minutes"))
    saves = safe_float(el.get("saves"))
    saves_p90 = (saves / (minutes/90.0)) if minutes > 0 else 0.0
    # Cap & scale saves_p90 into 0..1
    saves_component = min(max(saves_p90, 0.0), 5.0) / 5.0
    ict_norm = ict / 10.0
    sel_norm = sel / 100.0
    base = 0.62*ep_next + 0.20*form + 0.08*ict_norm + 0.05*sel_norm + 0.05*saves_component
    return base - availability_penalty(el)

def is_position(el: dict, et: int) -> bool:
    return el.get("element_type") == et

def shortlist_by_position(data: dict, element_type: int, scorer: Callable[[dict], float]):
    teams = build_team_map(data)
    elements = [el for el in data.get("elements", []) if is_position(el, element_type)]
    scored = []
    for el in elements:
        score = scorer(el)
        scored.append({
            "id": el["id"],
            "name": f'{el.get("first_name","").strip()} {el.get("second_name","").strip()}'.strip(),
            "web_name": el.get("web_name"),
            "team": teams.get(el.get("team"), f'Team {el.get("team")}'),
            "status": el.get("status"),
            "chance": el.get("chance_of_playing_next_round"),
            "ep_next": safe_float(el.get("ep_next")),
            "form": safe_float(el.get("form")),
            "ict_index": safe_float(el.get("ict_index")),
            "selected_by_percent": safe_float(el.get("selected_by_percent")),
            "minutes": el.get("minutes"),
            "saves": el.get("saves"),
            "now_cost": el.get("now_cost"),
            "score": round(score, 4),
        })
    scored.sort(key=lambda r: (r["score"], r["ep_next"]), reverse=True)
    top5 = scored[:5]
    return top5, scored

def print_section(title: str, top5: list):
    print(title)
    for i, row in enumerate(top5, 1):
        nm = row["web_name"] or row["name"]
        print(f"{i}) {nm} — {row['team']} | score={row['score']} | ep_next={row['ep_next']} | form={row['form']} | ict={row['ict_index']} | status={row['status']} ({row['chance']})")
    print()

def write_csv(filename: str, rows: list):
    if not rows:
        return
    import csv
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_bootstrap_static(src)
    ev_id = next_event_id(data)
    print(f"Next GW (event_id): {ev_id}\n")

    # 1: GK, 2: DEF, 3: MID, 4: FWD
    gk_top5, gk_all = shortlist_by_position(data, 1, score_goalkeeper)
    def_top5, def_all = shortlist_by_position(data, 2, score_defender)
    mid_top5, mid_all = shortlist_by_position(data, 3, score_midfielder)
    fwd_top5, fwd_all = shortlist_by_position(data, 4, score_forward)

    print_section("Top 5 Goalkeepers", gk_top5)
    print_section("Top 5 Defenders", def_top5)
    print_section("Top 5 Midfielders", mid_top5)
    print_section("Top 5 Forwards", fwd_top5)

    write_csv("fpl_ranked_goalkeepers.csv", gk_all)
    write_csv("fpl_ranked_defenders.csv", def_all)
    write_csv("fpl_ranked_midfielders.csv", mid_all)
    write_csv("fpl_ranked_forwards.csv", fwd_all)
    print("CSV files written: fpl_ranked_goalkeepers.csv, fpl_ranked_defenders.csv, fpl_ranked_midfielders.csv, fpl_ranked_forwards.csv")

if __name__ == "__main__":
    main()
