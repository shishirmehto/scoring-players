#!/usr/bin/env python3
"""Chip heuristics for two chip sets (first half and second half)."""

from __future__ import annotations


def chip_heuristics(gw: int, suggestions: dict, free_transfers: int) -> str:
    """Heuristic guidance reflecting 2025/26 rules.

    - GW16: AFCON FT top-up to five; plan to arrive with 0 banked FTs by GW15.
    - First-half chip set must be used before GW19 deadline.
    - Recommend WC/BB/TC/FH opportunistically based on projected gains.
    """
    if gw == 16:
        return "GW16 AFCON top-up: free transfers are topped up to five. Arrive with 0 banked by GW15 to capture all five."

    total_gain = 0.0
    try:
        total_gain = sum(
            s.get("ep_gain_window", 0.0) for s in suggestions.get("suggestions", [])
        )
    except Exception:
        total_gain = 0.0

    if 7 <= gw <= 12 and free_transfers < 2 and total_gain >= 12.0:
        return "Consider Wildcard to capture fixture swing (projected gain >= 12 over 4 GWs)."

    if gw <= 19:
        return "First-half chips expire by GW19 deadline. Prioritize TC on a soft home fixture, BB when bench peaks, and FH for premium clashes if needed."

    return "No chip recommended by heuristics. Save for blanks/doubles in the run-in."
