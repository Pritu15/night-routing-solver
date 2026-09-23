#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep run_ml_solver_db.py hyperparameters against one fixed input and compare
`unassigned` counts, without re-loading the DB/ML bundle per run.

Everything here is read-only and in-process: it imports run_ml_solver_db.py as a
module, loads the offline fixture ONCE, and calls `solve_night(...)` once per
SolverConfig in the grid. Same fixture + same provider each run, so any change in
`unassigned` is attributable to the hyperparameter you moved.

Usage:
    python sweep_hyperparams.py                                  # default grid, haversine+ML
    python sweep_hyperparams.py --offline full_test_data.json
    python sweep_hyperparams.py --no-ml                          # isolate non-ML knobs
    python sweep_hyperparams.py --param near_tie_slack --values 1.0,1.25,1.5,2.0
    python sweep_hyperparams.py --param cluster_restarts --values 1,4,12,24

With no --param, it runs a small default grid over every exposed knob, one
parameter at a time (holding the rest at their SolverConfig() default), and
prints one row per run with the base config's counts as the baseline (first row).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, List


def _load_module():
    path = Path(__file__).resolve().parent / "run_ml_solver_db.py"
    spec = importlib.util.spec_from_file_location("run_ml_solver_db", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_ml_solver_db"] = mod  # dataclass() needs the module registered
    spec.loader.exec_module(mod)
    return mod


# param name -> (cast, candidate values) for the default sweep
DEFAULT_GRID = {
    "near_tie_slack": (float, [1.0, 1.1, 1.25, 1.5, 2.0]),
    "near_tie_km_allowance": (float, [0.0, 0.5, 1.0, 2.0, 4.0]),
    "cluster_zone_penalty_km": (float, [0.0, 2.0, 5.0, 10.0]),
    "cluster_restarts": (int, [1, 4, 12, 24]),
    "walk_limit_min": (float, [15, 20, 30, 45]),
    "boarding_buffer_min": (float, [1, 3, 5]),
    "office_buffer_min": (float, [0, 5, 10]),
    "max_route_minutes": (float, [120]),
    "dropoff_return_weight": (float, [0.0, 0.25, 0.5, 1.0]),
}


def _row(mod, cfg, provider, foot, use_ml: bool, solver_input: dict, service_date: str) -> dict:
    solved = mod.solve_night(
        service_date=service_date, provider=provider, foot=foot,
        cfg=cfg, use_ml=use_ml, **solver_input,
    )
    counts = solved.counts()
    by_reason = {}
    for u in solved.unassigned:
        key = f"{u['type']}:{u['reason']}"
        by_reason[key] = by_reason.get(key, 0) + 1
    return {"counts": counts, "by_reason": by_reason}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", default="full_test_data.json", help="fixture JSON (default: full_test_data.json)")
    ap.add_argument("--date", default="2026-09-22", help="service_date to stamp the ML query time with")
    ap.add_argument("--no-ml", action="store_true", help="raw OSRM/haversine durations, no ML correction")
    ap.add_argument("--haversine", action="store_true", default=True,
                    help="use straight-line distances (default: on, no OSRM server required)")
    ap.add_argument("--param", help="single SolverConfig field to sweep (default: sweep all of DEFAULT_GRID)")
    ap.add_argument("--values", help="comma-separated values for --param (overrides the built-in grid)")
    args = ap.parse_args()

    mod = _load_module()

    fixture = args.offline if Path(args.offline).is_absolute() else str(Path(__file__).resolve().parent / args.offline)
    solver_input, _warns = mod._load_offline(fixture)

    provider = mod.HaversineProvider()
    foot = mod.HaversineWalkProvider()
    use_ml = not args.no_ml

    base = mod.SolverConfig()
    base_row = _row(mod, base, provider, foot, use_ml, solver_input, args.date)
    print(f"fixture={fixture}  use_ml={use_ml}  engine=haversine")
    print(f"[baseline] unassigned={base_row['counts']['unassigned']}  {base_row['by_reason']}")
    print()

    if args.param:
        cast, default_values = DEFAULT_GRID.get(args.param, (float, []))
        values: List[Any]
        if args.values:
            values = [cast(v) for v in args.values.split(",")]
        elif default_values:
            values = default_values
        else:
            raise SystemExit(f"no default grid for --param {args.param!r}; pass --values")
        grid = {args.param: values}
    else:
        grid = {k: v[1] for k, v in DEFAULT_GRID.items()}

    header = f"{'param':<26}{'value':>10}  {'unassigned':>10}  by_reason"
    print(header)
    print("-" * len(header))
    for param, values in grid.items():
        for value in values:
            cfg = replace(base, **{param: value})
            row = _row(mod, cfg, provider, foot, use_ml, solver_input, args.date)
            marker = " (baseline)" if value == getattr(base, param) else ""
            print(f"{param:<26}{value!s:>10}  {row['counts']['unassigned']:>10}  {row['by_reason']}{marker}")

    close = getattr(provider, "close", None)
    if callable(close):
        close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
