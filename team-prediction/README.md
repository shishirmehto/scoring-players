# Season Planner (team-prediction)

A holistic, data-driven season planning workflow that fuses the official FPL API with the vaastav historical dataset to project Expected Points (EP) per player for each Gameweek, manage risk via ownership-aware decisions, and schedule chips intelligently.

## What this folder contains
- `fpl_season_planner_skeleton.py`: per-GW Top-5 generator using EP heuristics and fixtures.
- `season_outputs/`: per-GW CSVs of Top-5 players by position (GK/DEF/MID/FWD).

## How to run
```bash
# Activate env (if needed)
source ../activate_env.sh

# Example: local files
python team-prediction/fpl_season_planner_skeleton.py \
  --bootstrap path/to/bootstrap-static.json \
  --fixtures path/to/fixtures.json \
  --merged-gw path/to/merged_gw.csv \
  --players-raw path/to/players_raw.csv

# Or with live URLs (API + hosted CSVs)
python team-prediction/fpl_season_planner_skeleton.py \
  --bootstrap-url https://fantasy.premierleague.com/api/bootstrap-static/ \
  --fixtures-url https://fantasy.premierleague.com/api/fixtures/ \
  --merged-gw https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv \
  --players-raw https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/players_raw.csv
```

Outputs are written to `team-prediction/season_outputs/gw_XX_top5_by_pos.csv`.

---

# Strategy Charter (how we’ll try to win, not just do well)

## 1) Know the 2025/26 ground rules (so the math is right)

- Eight chips total this season (Wildcard, Free Hit, Triple Captain, Bench Boost ×2). One set must be used before the GW19 deadline on 30 Dec 2025; the other set is for GWs 20–38. No Assistant Manager chip this season. See: [Premier League][1]
- AFCON “top-up” in GW16: your free transfers are topped up to 5 to cope with absences. See: [Premier League][1]
- Official Fixture Difficulty Rating (FDR) is maintained by FPL and updated through the season; we’ll use it as one input (but not the only one). See: [Premier League][2]

## 2) Data we’ll fuse every week

- FPL API `/bootstrap-static`: prices, flags/availability, `ep_next`, ownership, element type, status. See: [fpl.readthedocs.io][3]
- vaastav repo: historical per-GW tables (`merged_gw.csv`) and season snapshots (`players_raw.csv`) to learn priors for minutes, returns, and positional scoring. Repo updates three times per season; treat as training source; use API for live. See: [GitHub][4]
- (Optional) Public xG/xA (Understat) to separate luck from repeatable chance creation/shot quality. See: [understat.com][5]

## 3) Player expectation model (what we actually optimize)

EP(GW) = xMins × [ Attack EP + Clean-Sheet EP + Save/Bonus/Defensive-contrib EP ] − Availability penalty

- xMins (expected minutes): learned from vaastav last 6–8 GWs (starts, sub patterns, ≥60 mins), FPL flags/`chance_of_playing_next_round`, and rest days.
- Attack EP (MIDs/FWDs, attacking DEFs): calibrated from historical returns rates in vaastav, adjusted by opponent defensive strength; optionally nudged by Understat xG/xA trends.
- Clean-Sheet EP (GK/DEF): opponent attacking strength × your team defensive rating → CS probability (mapped to FPL points). FDR is a baseline prior.
- Bonus/defensive-contribution EP: includes new CBIT/CBIRT points and BPS tweaks for 2025/26.
- Availability penalty: yellow flags, doubts, benchings; hard zero if ruled out.

For context on open-source forecast strength, see OpenFPL: [arXiv][7].

## 4) Fixture & Opponent Model (stronger than “form”)

- Start with official FDR (1 easy → 5 hard) as a prior, not the truth. See: [Premier League][2]
- Split difficulty into attacking vs defensive lenses. Attacking players face opponent defensive strength; defenders/GKs face opponent attacking strength. If no Understat, approximate from vaastav goals for/against and recent NP-xG proxies. See: [FPL Form][8]
- Apply home/away adjustments and congestion (Europe/cups) to lower xMins or EP when rotation risk rises.

## 5) Ownership-aware risk control (play to objective)

- Protecting a lead → prioritize high-EP and high-EO shields; only a few swords in premium slots.
- Chasing → accept more volatility: captaincy and 1–2 weekly transfers target low-EO players with strong 2–4 GW EP windows.
- EO = started% + captained% (+ triple-captained%). See: [Premier League][9], [Fantasy Football Scout][10]

## 6) Season plan (chips + transfers) that the model enforces

- Wildcard 1 (H1): GW7–12 to pivot into first big fixture swing; must be before GW19. See: [Premier League][1]
- GW16 AFCON top-up (5 FTs): use as mini-wildcard for absences/rotation or festive schedule prep. See: [Premier League][1]
- Free Hit(s): reserve for biggest Blank GW. Bench Boost for biggest DGW. Triple Captain for premium attacker strong DGW (or elite single home if doubles weak). See: [Premier League][11]

Transfers and hits:
- Bank 1 FT per GW (you can roll up to five overall); extra transfers cost −4 each. Optimizer weighs EP gain minus hit cost over rolling 4–6 GW horizon. See: [FPL Rules][12], [FFS Guide][13]
- Around DGWs/BGWs, allow hits/chips if net EP justifies it (esp. captaincy upside).
- Use AFCON GW16 to reset structurally (rebalance funds, attack fixture tiers). See: [Premier League][1]

---

# Weekly Operating Loop

- T-72h (Mon/Tue) – Data refresh: pull latest FPL API; update flags, prices, ownership, `ep_next`. Compute EP for next 6 GWs.
- T-48h (Wed/Thu) – Transfer shortlists: generate 0–2 FT move sets (hits allowed if net EP > 6–8 over horizon). Simulate captaincy with EO.
- T-24h (Fri news) – Lock: re-run xMins after pressers; avoid new flags unless EP remains compelling. Finalize captain/vice, bench order.
- Post-GW – Review: record actual points and minutes to recalibrate minutes and returns priors.

---

## What you’ll see every Gameweek

1. Top-5 per position lists (GK/DEF/MID/FWD) for the next GW based on EP.
2. A transfer recommendation (0/1/2 FT; −4 when justified) with a 4–6 GW rationale.
3. Captain/vice shortlist with EO-aware risk notes.
4. Chip prompts only when marginal EP threshold is hit (and respecting GW19 first-set deadline + GW16 AFCON top-up). See: [Premier League][1]

## Notes on realism

- Injuries/rotation cannot be fully de-risked; xMins modeling and late news matter most.
- This plan accounts for new 2025/26 scoring (defensive contributions & BPS tweaks), two sets of chips, and GW16 transfer top-ups, so forecasts/optimizations stay calibrated. See: [Premier League][1]

---

## Implementation details in this repo

- The planner script includes a robust CSV loader fallback (engine="python", on_bad_lines="skip") to tolerate malformed rows.
- You can integrate with `fpl_team_builder_with_vaastav.py` to perform full 15-man optimization and write Top-5 lists at the repo root.

---

[1]: https://www.premierleague.com/en/news/4373187/whats-new-for-202526-changes-in-fantasy-premier-league
[2]: https://www.premierleague.com/en/news/4324765/get-the-fixture-difficulty-ratings-for-2025/26-fpl-season
[3]: https://fpl.readthedocs.io/en/latest/classes/player.html
[4]: https://github.com/vaastav/Fantasy-Premier-League
[5]: https://understat.com/league/EPL
[6]: https://www.premierleague.com/en/news/68553
[7]: https://arxiv.org/abs/2508.09992
[8]: https://fplform.com/fpl-fixture-difficulty
[9]: https://www.premierleague.com/en/news/2683145
[10]: https://www.fantasyfootballscout.co.uk/2021/03/24/what-is-effective-ownership-and-why-is-it-so-widely-talked-about-in-fpl/
[11]: https://www.premierleague.com/en/news/2174900
[12]: https://fantasy.premierleague.com/help/rules
[13]: https://www.fantasyfootballscout.co.uk/2025/07/21/how-to-make-fpl-transfers-and-explaining-a-points-hit/
