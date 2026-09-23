# Night Routing Solver

A standalone, self-contained port of a backend employee pickup/drop-off routing
solver — Held-Karp exact stop ordering, capacity-constrained k-means clustering,
an XGBoost travel-time correction model, and the full Case A/B/C/D policy logic
for a whole overnight fleet solve. See the top of
[`run_ml_solver_db.py`](run_ml_solver_db.py) for the full design writeup.

## Requirements

- Python 3.10+ (developed on 3.12)
- The packages in `requirements.txt`
- Optional, for real road-network routing: two local OSRM servers (car on
  `:5000`, foot on `:5001`). Without them the solver automatically falls back
  to straight-line (haversine) distances — nothing crashes either way.
- Optional, for live-DB runs: a Supabase project with this schema's tables
  (`vehicle`, `zone`, `pickup_request`, `dropoff_request`,
  `vehicle_pickup_location`).

## Setup

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start — no DB, no OSRM, no setup

Runs entirely against the bundled fixture (`full_test_data.json`), with
straight-line distances instead of real roads:

```bash
python run_ml_solver_db.py --offline full_test_data.json --haversine --no-ml
```

Prints routes/stops/passengers/unassigned counts, a per-shift unassigned
breakdown, and fleet/fairness diagnostics. Nothing is written to disk unless
you pass `--dump-dir`/`--sql-out`.

## Running with the ML travel-time model

The XGBoost duration-correction model is bundled at `ml_model/inference_bundle.joblib`
and is picked up automatically — no setup needed. ML is on by default (`--no-ml`
turns it off), so this alone is enough:

```bash
python run_ml_solver_db.py --offline full_test_data.json --haversine
```

(`ROUTING_ML_MODEL_PATH` still works if you ever want to point at a different
bundle — set it and it overrides the colocated one.)

## Running against a live Supabase DB

```bash
cp .env.example .env
# edit .env: fill in SUPABASE_URL and SUPABASE_KEY

python run_ml_solver_db.py --date 2026-09-22
```

That's it — `.env` next to the script is read automatically (real
`SUPABASE_URL`/`SUPABASE_KEY` environment variables still take priority if
both are set). If OSRM is reachable at `localhost:5000`/`:5001` it's used
automatically for real road distances/durations; otherwise the solver logs a
warning and falls back to haversine. Nothing is ever written back to the DB
by this script — it's read-only unless you pass `--dump-dir`/`--sql-out`/`--sql-parts`.

## Useful flags

```
--date YYYY-MM-DD          single service date (default 2026-09-22)
--start / --end            solve a date range instead
--offline FILE.json        fixture instead of the live DB
--haversine                force straight-line distances (skip OSRM)
--no-ml                    raw OSRM/haversine durations (no XGBoost correction)
--model xgb|rf             which trained model to use
--json-summary             also print one machine-readable JSON line per date
--no-diagnostics           skip the accounting/fleet/fairness diagnostics block
--quiet                    suppress the routing INFO logs (k-means per-shift lines, etc.)

# solver hyperparameters (see SolverConfig in run_ml_solver_db.py for what each does)
--walk-limit-min --walk-speed-kmph --boarding-buffer-min --office-buffer-min
--max-route-minutes --dropoff-return-weight --near-tie-slack --near-tie-km
--cluster-zone-penalty-km --cluster-restarts --cluster-seed
```

Run `python run_ml_solver_db.py --help` for the full list.

## Sweeping hyperparameters

`sweep_hyperparams.py` re-solves the same offline fixture once per parameter
value, in-process (loads the ML bundle once, reuses it across runs) so you can
see how a knob moves the `unassigned` count without re-running the whole CLI
each time:

```bash
python sweep_hyperparams.py --param max_route_minutes --values 120,135,150
```

## Regenerating the analysis report

`generate_unassigned_report.py` parses a saved block of `run_ml_solver_db.py`
console output (see `report.txt` for the format) into a multi-page PDF: per-day
unassigned charts, a shift×date heatmap, fleet/fairness trends, and a
statistics write-up.

```bash
python generate_unassigned_report.py --in report.txt --out unassigned_analysis_report.pdf
```

## Repo contents

| File | What it is |
|---|---|
| `run_ml_solver_db.py` | The solver itself — vendored config/distance-providers/DB-adapter/solver, plus the CLI |
| `sweep_hyperparams.py` | In-process hyperparameter sweep harness |
| `generate_unassigned_report.py` | Parses console output into a PDF report |
| `full_test_data.json` | Offline fixture (`--offline`) — vehicles, requests, fixed stops |
| `solved_routes.json` | A shipped baseline solve, for parity/regression comparison |
| `ml_model/inference_bundle.joblib` | Trained XGBoost travel-time model + supporting lookup tables |
| `report.txt` / `unassigned_analysis_report.pdf` | Example console output and its generated report |
| `.env.example` | Template for Supabase credentials (copy to `.env`, never commit the real one) |
