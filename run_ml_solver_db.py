#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone, self-contained twin of the backend routing package.

This single file VENDORS the production code path -- the solver constants, the
`DistanceProvider`s, the Supabase adapter and the whole `NightSolver` (including
the XGBoost ML duration layer) -- instead of importing `app.services.routing`.
So it *contains* everything `backend/app/services/routing/solver.py` has, plus
the read path, and can be read/run on its own.

It fetches one service date from Supabase (read-only; `SUPABASE_URL` /
`SUPABASE_KEY` come from the environment or from `backend/.env`), solves the
night with the ML model, and prints the counts + the unassigned split by reason.

    python run_ml_solver_db.py                       # 2026-09-22, OSRM + ML
    python run_ml_solver_db.py --date 2026-09-22
    python run_ml_solver_db.py --no-ml               # raw OSRM (routing_night parity)
    python run_ml_solver_db.py --haversine           # simulate a VM with no OSRM
    python run_ml_solver_db.py --offline FILE.json   # fixture instead of the DB

Needs the backend deps (numpy, pandas, scipy, scikit-learn, xgboost, joblib,
httpx, supabase) importable, and for real road times the two local OSRM engines
up (car :5000, foot :5001); otherwise the solver falls back to haversine and the
script prints which engine actually ran. It writes nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from supabase import Client, create_client


# ============================================================================
# Section 1/4 -- solver constants (vendored backend routing/config.py)
# ============================================================================
"""Solver constants.

Lifted verbatim from `data/routing_night.py` (the standalone twin of
`data/testing.ipynb`, which implements `data/system_data/Algo_refined.md`) so
the ported solver reproduces that script's output. Changing any value here
changes the schedule, so treat them as policy, not tuning knobs.
"""
from dataclasses import dataclass

OFFICE_LOCATION = {"lat": 23.770204034678137, "lng": 90.40845882507914}

# The office is the origin for every pickup route's end and every drop-off
# route's start. Single source of truth lives in week_service.
OFFICE = (OFFICE_LOCATION["lat"], OFFICE_LOCATION["lng"])

# BDS: employees walk to a designated pick-up point only if the foot-network
# walk is within this many minutes; past it the 10 PM car comes to the door.
WALK_LIMIT_MIN = 30

# Straight-line walk speed, used ONLY when the OSRM foot engine is unavailable.
WALK_SPEED_KMPH = 4.5

# BDS: a vehicle waits at most 5 minutes per stop for boarding/alighting.
BOARDING_BUFFER_MIN = 1

# Hard cap on a single route's on-road (passenger-journey) time.
MAX_ROUTE_MINUTES = 120

# A pickup must reach the office at least this early before the shift starts.
OFFICE_BUFFER_MIN = 3

# How much a drop-off ORDER cares about where the night ENDS. A drop-off tour
# is open -- the car drops its last rider and stops -- so the leg home is
# normally never driven, and pricing it at full cost makes the search buy a
# cheap return with real driven distance. 0.0 is the true shortest driven
# route. Raise it only if the 06:15 -> 07:30 chain starts stranding riders.
DROPOFF_RETURN_WEIGHT = 0.0

# Case D (07:30 drop-off): the shared main-road drop point (Agargaon Metro).
AGARGAON_METRO = (23.775518, 90.388407)

# Every timestamp is anchored to 22:00 on the service date, so the overnight
# timeline stays monotonic across midnight.
NIGHT_ANCHOR_HOUR = 22

# The drop-off event that triggers Case D (main-road consolidation).
MAIN_ROAD_DROP_TIME = "07:30:00"

# Stop ordering is an EXACT search (Held-Karp DP), exact over the objective
# (total passenger ride time for pickups, total route time for drop-offs). That
# is only affordable because a trip can never carry more stops than the vehicle
# has seats -- the fleet's largest capacity is 11, so n never grows large.
MAX_STOPS_FOR_EXACT = 16


@dataclass(frozen=True)
class SolverConfig:
    """Per-run knobs. Defaults reproduce `routing_night.py` exactly."""

    office: tuple[float, float] = OFFICE
    walk_speed_kmph: float = WALK_SPEED_KMPH
    walk_limit_min: float = WALK_LIMIT_MIN
    boarding_buffer_min: int = BOARDING_BUFFER_MIN
    max_route_minutes: float = MAX_ROUTE_MINUTES
    office_buffer_min: int = OFFICE_BUFFER_MIN
    dropoff_return_weight: float = DROPOFF_RETURN_WEIGHT
    agargaon_metro: tuple[float, float] = AGARGAON_METRO
    night_anchor_hour: int = NIGHT_ANCHOR_HOUR

    # BDS says the Agargaon Metro consolidation does not apply on Fridays. The
    # notebook omits this check; the backend knows the real service date, so it
    # can honour the rule. Set False to reproduce the notebook byte-for-byte on
    # a Friday service date.
    apply_friday_exception: bool = True

    # Case B (23:00 door-to-door): a car within `near_tie_slack` x the nearest
    # car's distance (or within `near_tie_km_allowance` km of it, whichever is
    # more permissive) is treated as an equally-good choice; ties then prefer a
    # car already carrying riders, then the rider's own zone. Loosening these
    # trades ride-time fairness for load-balancing across cars.
    near_tie_slack: float = 1.25
    near_tie_km_allowance: float = 1.0

    # Case B-kmeans (00:00-06:00): capacity-constrained k-means over rider
    # homes. `cluster_zone_penalty_km` discourages (but does not forbid)
    # matching a cluster to a car outside its modal zone; `cluster_restarts`
    # trades solve time for better (lower-SSE) clusters; `cluster_seed` is
    # fixed so two runs over the same data agree.
    cluster_zone_penalty_km: float = 5.0
    cluster_restarts: int = 12
    cluster_seed: int = 0

# ============================================================================
# Section 2/4 -- distance providers (vendored backend routing/distance.py)
# ============================================================================
"""Distance/duration providers for the routing solver.

Two interchangeable implementations behind one protocol:

- `OsrmProvider` — real road network via a local OSRM server. Accurate, and the
  only way to get drawable road geometry.
- `HaversineProvider` — straight-line fallback. No dependencies, no server, so
  the solver still runs anywhere (e.g. Render, CI, a laptop with no OSRM).

`get_provider()` picks one from settings and degrades to haversine whenever OSRM
is unreachable, so a solve always completes.

The caches are not an optimisation, they are a requirement: `enforce_cap_*`
re-orders a route once per candidate stop per shed iteration, and each of those
re-requests the same matrix. Without caching a full-night solve does not finish.
"""

import logging
import math
from typing import Iterable, Protocol, Sequence

import httpx

class _Settings:
    """Inline stand-in for backend `app.config.Settings` (env-driven, same defaults)."""
    routing_engine = os.environ.get("ROUTING_ENGINE", "osrm")
    osrm_base_url = os.environ.get("OSRM_BASE_URL", "http://localhost:5000")
    osrm_foot_base_url = os.environ.get("OSRM_FOOT_BASE_URL", "http://localhost:5001")
    osrm_timeout_seconds = float(os.environ.get("OSRM_TIMEOUT_SECONDS", "30"))
    osrm_probe_timeout_seconds = float(os.environ.get("OSRM_PROBE_TIMEOUT_SECONDS", "2"))
    routing_average_speed_kmph = float(os.environ.get("ROUTING_AVERAGE_SPEED_KMPH", "40"))
    routing_walk_speed_kmph = float(os.environ.get("ROUTING_WALK_SPEED_KMPH", "4.5"))

    @property
    def prefers_osrm(self) -> bool:
        return self.routing_engine.strip().lower() not in {"haversine", "lightweight"}


app_settings = _Settings()

logger = logging.getLogger("uvicorn.error")

Coord = tuple[float, float]          # (lat, lng)
Matrix = list[list[float]]

# Cache keys round coordinates to ~1 m so float noise doesn't cause misses.
_COORD_PRECISION = 5


def _cache_key(coords: Sequence[Coord]) -> tuple:
    return tuple((round(lat, _COORD_PRECISION), round(lng, _COORD_PRECISION)) for lat, lng in coords)


def haversine_km(a: Coord, b: Coord) -> float:
    """Great-circle distance in km. Ported verbatim from the notebook."""
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def walk_minutes(a: Coord, b: Coord, walk_speed_kmph: float) -> float:
    """Straight-line walking time between two points (fallback only).

    `routing_night.py` reads Case A walking times from the OSRM foot network
    (real pedestrian graph, no assumed speed). This function is the haversine
    stand-in used when that foot engine is unreachable, so a solve still
    completes. Kept as a free function so `HaversineWalkProvider` can reuse it.
    """
    return haversine_km(a, b) / walk_speed_kmph * 60.0


# ──────────────────────────────────────────────────────────────────────────────
# Foot provider: real pedestrian-network walking times (OSRM foot profile).
# `routing_night.py` Case A asks "can this rider walk to this fixed stop within
# WALK_LIMIT_MIN" against the foot graph, batching one /table call per employee
# home vs. the candidate stops. `prefetch` fills the cache in chunks; `walk_minutes`
# then reads it back pair-by-pair. Cache keys round to ~1 m like the driving side.
# ──────────────────────────────────────────────────────────────────────────────


class FootDistanceProvider(Protocol):
    """Walking times between coordinates on the pedestrian network.

    A solve is still allowed to fall back to straight-line estimates when no
    foot engine answers, so the solver only ever sees this narrow surface.
    """

    name: str

    def walk_minutes(self, a: Coord, b: Coord) -> float:
        """Minutes to walk a -> b. float("inf") when the pair is unroutable."""
        ...

    def prefetch(self, home: Coord, stops: Iterable[Coord]) -> None:
        """Warm the walk cache for `home` vs. every stop (batched)."""
        ...


def _walk_key(a: Coord, b: Coord) -> tuple:
    return (round(a[0], 6), round(a[1], 6), round(b[0], 6), round(b[1], 6))


class HaversineWalkProvider:
    """Straight-line fallback for the foot engine (no server, no assumptions)."""

    name = "haversine_walk"

    def __init__(self, walk_speed_kmph: float | None = None):
        self.walk_speed_kmph = walk_speed_kmph or app_settings.routing_walk_speed_kmph

    def walk_minutes(self, a: Coord, b: Coord) -> float:
        return walk_minutes(a, b, self.walk_speed_kmph)

    def prefetch(self, home: Coord, stops: Iterable[Coord]) -> None:
        return None


class FootOsrmProvider:
    """Pedestrian-network walking times via a local OSRM foot server.

    Mirrors `routing_night.py`'s `_foot_table`/`prefetch_walk`/`walk_minutes`:
    batched `/table/v1/foot/` calls (OSRM caps a table at ~100 coordinates),
    chunked against the stop set, with every pair cached for the life of the
    instance. A null matrix entry (unroutable on foot) becomes inf.
    """

    name = "osrm_foot"

    # OSRM caps a /table call at 100 coordinates, so prefetch chunks stops as
    # home + up to 90 per call -- the same budget routing_night.py uses.
    _PREFETCH_CHUNK = 90

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or app_settings.osrm_foot_base_url).rstrip("/")
        self.timeout = timeout or app_settings.osrm_timeout_seconds
        self._client = httpx.Client(timeout=self.timeout)
        self._cache: dict[tuple, float] = {}

    @staticmethod
    def _coord_str(coords: Sequence[Coord]) -> str:
        return ";".join(f"{lng},{lat}" for lat, lng in coords)

    def _table(self, coords: Sequence[Coord]) -> Matrix:
        """One OSRM foot /table call -> durations in minutes (inf = unroutable)."""
        response = self._client.get(
            f"{self.base_url}/table/v1/foot/{self._coord_str(coords)}",
            params={"annotations": "duration"},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "Ok":
            raise RuntimeError(f"OSRM foot error: {payload.get('code')} {payload.get('message', '')}")
        return [
            [value / 60.0 if value is not None else float("inf") for value in row]
            for row in payload["durations"]
        ]

    def prefetch(self, home: Coord, stop_coords: Iterable[Coord]) -> None:
        """Warm the walk cache: one batched foot call per chunk of stops.

        Case A walks every employee home against every candidate stop, so asking
        pair-by-pair would be tens of thousands of HTTP calls. Chunking keeps it
        one call per ~90 stops; the minutes land in the same cache `walk_minutes`
        reads back.
        """
        stops = list(stop_coords)
        for i in range(0, len(stops), self._PREFETCH_CHUNK):
            chunk = stops[i:i + self._PREFETCH_CHUNK]
            durations = self._table([home] + chunk)
            for j, c in enumerate(chunk):
                self._cache[_walk_key(home, c)] = durations[0][j + 1]

    def walk_minutes(self, a: Coord, b: Coord) -> float:
        key = _walk_key(a, b)
        got = self._cache.get(key)
        if got is not None:
            return got
        mins = self._table([a, b])[0][1]
        self._cache[key] = mins
        return mins

    def healthy(self) -> bool:
        """Cheap probe against a known-good coordinate pair."""
        probe = httpx.Client(timeout=app_settings.osrm_probe_timeout_seconds)
        try:
            response = probe.get(
                f"{self.base_url}/table/v1/foot/"
                f"{self._coord_str([(23.7702, 90.4085), (23.7702, 90.4095)])}",
                params={"annotations": "duration"},
            )
            return response.status_code == 200 and response.json().get("code") == "Ok"
        except Exception:
            return False
        finally:
            probe.close()

    def close(self) -> None:
        self._client.close()


def get_foot_provider(prefer_osrm: bool | None = None) -> FootDistanceProvider:
    """Best available foot provider. Never raises -- a solve must always complete."""
    if prefer_osrm is None:
        prefer_osrm = app_settings.prefers_osrm

    if not prefer_osrm:
        return HaversineWalkProvider()

    candidate = FootOsrmProvider()
    if candidate.healthy():
        logger.info("routing: using OSRM foot at %s", candidate.base_url)
        return candidate

    candidate.close()
    logger.warning(
        "routing: OSRM foot unreachable at %s, falling back to straight-line "
        "walking times (Case A walk limits will be approximate)",
        candidate.base_url,
    )
    return HaversineWalkProvider()


class DistanceProvider(Protocol):
    """Driving distances/durations between coordinates.

    Note the return order, which mirrors the notebook: `table` yields durations
    first, `route` yields distance first. `route`'s duration is deliberately
    unused by the solver — route timing is summed from `table` legs plus
    boarding buffers, while total distance comes from `route`. Swapping either
    source changes the emitted schedule.
    """

    name: str

    def table(self, coords: Sequence[Coord]) -> tuple[Matrix, Matrix]:
        """Full square matrix: (durations_min, distances_km)."""
        ...

    def route(self, coords: Sequence[Coord]) -> tuple[float, float, list[Coord]]:
        """(distance_km, duration_min, geometry as [lat, lng] points)."""
        ...


class HaversineProvider:
    """Straight-line fallback.

    Geometry is just the waypoints joined up, so a map drawn from it shows
    direct lines rather than roads — honest about being an approximation.
    """

    name = "haversine"

    def __init__(self, average_speed_kmph: float | None = None):
        self.average_speed_kmph = average_speed_kmph or app_settings.routing_average_speed_kmph

    def _minutes(self, km: float) -> float:
        return km / self.average_speed_kmph * 60.0

    def table(self, coords: Sequence[Coord]) -> tuple[Matrix, Matrix]:
        distances = [[haversine_km(a, b) for b in coords] for a in coords]
        durations = [[self._minutes(km) for km in row] for row in distances]
        return durations, distances

    def route(self, coords: Sequence[Coord]) -> tuple[float, float, list[Coord]]:
        km = sum(haversine_km(a, b) for a, b in zip(coords, coords[1:]))
        return km, self._minutes(km), [tuple(c) for c in coords]


class OsrmProvider:
    """Road-network distances via a local OSRM server (car profile).

    Caches every matrix and route response for the lifetime of the instance —
    one instance per solve, so the cache is naturally scoped to a service date.
    """

    name = "osrm"

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or app_settings.osrm_base_url).rstrip("/")
        self.timeout = timeout or app_settings.osrm_timeout_seconds
        self._client = httpx.Client(timeout=self.timeout)
        self._table_cache: dict[tuple, tuple[Matrix, Matrix]] = {}
        self._route_cache: dict[tuple, tuple[float, float, list[Coord]]] = {}

    # OSRM speaks lng,lat — every conversion goes through here so the flip is
    # impossible to forget at a call site.
    @staticmethod
    def _coord_str(coords: Sequence[Coord]) -> str:
        return ";".join(f"{lng},{lat}" for lat, lng in coords)

    def _get(self, path: str, params: dict) -> dict:
        response = self._client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "Ok":
            raise RuntimeError(f"OSRM error at {path}: {payload.get('code')} {payload.get('message', '')}")
        return payload

    def table(self, coords: Sequence[Coord]) -> tuple[Matrix, Matrix]:
        key = _cache_key(coords)
        if key in self._table_cache:
            return self._table_cache[key]
        payload = self._get(
            f"/table/v1/driving/{self._coord_str(coords)}",
            {"annotations": "duration,distance"},
        )
        durations = [[value / 60.0 for value in row] for row in payload["durations"]]
        distances = [[value / 1000.0 for value in row] for row in payload["distances"]]
        self._table_cache[key] = (durations, distances)
        return durations, distances

    def route(self, coords: Sequence[Coord]) -> tuple[float, float, list[Coord]]:
        key = _cache_key(coords)
        if key in self._route_cache:
            return self._route_cache[key]
        payload = self._get(
            f"/route/v1/driving/{self._coord_str(coords)}",
            {"overview": "full", "geometries": "geojson"},
        )
        leg = payload["routes"][0]
        out = (
            leg["distance"] / 1000.0,
            leg["duration"] / 60.0,
            [(lat, lng) for lng, lat in leg["geometry"]["coordinates"]],
        )
        self._route_cache[key] = out
        return out

    def healthy(self) -> bool:
        """Cheap probe against a known-good coordinate pair."""
        probe = httpx.Client(timeout=app_settings.osrm_probe_timeout_seconds)
        try:
            response = probe.get(
                f"{self.base_url}/route/v1/driving/{self._coord_str([(23.77, 90.40), (23.78, 90.39)])}",
                params={"overview": "false"},
            )
            return response.status_code == 200 and response.json().get("code") == "Ok"
        except Exception:
            return False
        finally:
            probe.close()

    def close(self) -> None:
        self._client.close()


def get_provider(prefer_osrm: bool | None = None) -> DistanceProvider:
    """Best available provider. Never raises — a solve must always complete."""
    if prefer_osrm is None:
        prefer_osrm = app_settings.prefers_osrm

    if not prefer_osrm:
        logger.info("routing: using haversine engine (ROUTING_ENGINE=%s)", app_settings.routing_engine)
        return HaversineProvider()

    candidate = OsrmProvider()
    if candidate.healthy():
        logger.info("routing: using OSRM at %s", candidate.base_url)
        return candidate

    candidate.close()
    logger.warning(
        "routing: OSRM unreachable at %s, falling back to haversine "
        "(distances and geometry will be approximate)",
        candidate.base_url,
    )
    return HaversineProvider()


# ============================================================================
# Section 3/4 -- DB adapter, DB rows -> solver input (vendored routing/adapter.py)
# ============================================================================
"""DB rows → solver input, and the ID maps needed to get back again.

The solver speaks the roster's language — `employee_email`, `plate_no`,
`zone_name` — because that is what the notebook was written against and keeping
it that way is what lets `solved_routes.json` stay a usable fixture. The
database speaks in surrogate keys. This module is the only translation layer,
and it holds both directions so `writer` never has to re-query.

Two loading decisions worth stating, because they are not the obvious ones:

1. **Requests are NOT filtered on `route_id IS NULL`.** The old per-shift solver
   relied on that as its idempotency trick, but a whole-night solve cannot: if
   half the night is already routed, re-solving the remainder gives the fleet
   simulation a truncated request set and it happily produces a schedule that
   contradicts the routes already in the table. So we load the whole night every
   time and let `writer` replace the day atomically.

2. **The solve happens before anything is deleted.** Loading here is read-only;
   `writer.persist` does the clear and the insert together. A solver crash
   leaves yesterday's routes intact rather than an empty table.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from supabase import Client

logger = logging.getLogger(__name__)

# Requests in this state are not candidates for routing.
EXCLUDED_STATUSES = {"Rejected"}


def _num(value: Any) -> Optional[float]:
    """PostgREST hands back `numeric` columns inconsistently (number or string).

    The solver does arithmetic on every coordinate, so coerce once here rather
    than discovering a str/float TypeError halfway through a 148-route solve.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clock(value: Any) -> Optional[str]:
    """Normalise a TIME column to "HH:MM:SS"."""
    if value in (None, ""):
        return None
    text = str(value)
    parts = text.split(":")
    if len(parts) == 2:
        return f"{parts[0].zfill(2)}:{parts[1]}:00"
    if len(parts) >= 3:
        return f"{parts[0].zfill(2)}:{parts[1]}:{parts[2][:2]}"
    return text


@dataclass
class RoutingContext:
    """Everything the writer needs to turn solver output back into rows."""

    service_date: str
    solver_input: Dict[str, Any] = field(default_factory=dict)

    # natural key → surrogate key
    employee_id_by_email: Dict[str, int] = field(default_factory=dict)
    vehicle_id_by_plate: Dict[str, int] = field(default_factory=dict)
    driver_id_by_plate: Dict[str, Optional[int]] = field(default_factory=dict)
    zone_id_by_name: Dict[str, int] = field(default_factory=dict)
    pickup_id_by_email: Dict[str, int] = field(default_factory=dict)
    dropoff_id_by_email: Dict[str, int] = field(default_factory=dict)

    # data-quality problems that are not unassigned requests
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def request_id(self, request_type: str, email: str) -> Optional[int]:
        table = self.pickup_id_by_email if request_type == "pickup" else self.dropoff_id_by_email
        return table.get(email)

    def zone_id(self, zone_name: Optional[str]) -> Optional[int]:
        return self.zone_id_by_name.get(zone_name) if zone_name else None


def _dedupe_latest_per_employee(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Newest request per employee wins — the ad-hoc supersedes the weekly row.

    Same rule as `RoutingService._dedupe_latest_per_employee`; duplicated here so
    the adapter has no dependency on the service that calls it. An employee whose
    ad-hoc moved their 22:00 shift to 23:00 must be routed once, at 23:00 — not
    twice.
    """
    latest: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        key = row.get("employee_id")
        current = latest.get(key)
        if current is None or (row.get("created_at") or "") > (current.get("created_at") or ""):
            latest[key] = row
    return list(latest.values())


def _email_of(row: Dict[str, Any]) -> Optional[str]:
    """Reach through the embedded employee → users join for the natural key."""
    return ((row.get("employee") or {}).get("users") or {}).get("email")


def _name_of(row: Dict[str, Any]) -> Optional[str]:
    return ((row.get("employee") or {}).get("users") or {}).get("name")


class RoutingAdapter:
    """Read one night out of Supabase in five queries."""

    REQUEST_SELECT = "*, employee(employee_id, home_lat, home_lng, users(name, email)), zone(zone_name)"
    VEHICLE_SELECT = "*, zone(zone_name), driver(driver_id, user_id, users(name, email))"

    def __init__(self, db: Client):
        self.db = db
        # email → name, accumulated across both request tables so the solver can
        # label unassigned entries and passengers without a sixth query.
        self._employee_names: Dict[str, str] = {}

    def load(self, service_date: str) -> RoutingContext:
        ctx = RoutingContext(service_date=service_date)

        zones = self._load_zones(ctx)
        vehicles = self._load_vehicles(ctx)
        fixed_stops = self._load_fixed_stops(ctx)
        pickups = self._load_requests(ctx, "pickup_request", service_date)
        dropoffs = self._load_requests(ctx, "dropoff_request", service_date)

        ctx.solver_input = {
            "vehicles": vehicles,
            "pickup_requests": pickups,
            "dropoff_requests": dropoffs,
            "fixed_stops": fixed_stops,
            "employee_names": self._employee_names,
        }
        ctx.stats = {
            "zones": len(zones),
            "vehicles": len(vehicles),
            "pickup_requests": len(pickups),
            "dropoff_requests": len(dropoffs),
            "fixed_stops": len(fixed_stops),
        }
        logger.info("routing input for %s: %s", service_date, ctx.stats)
        return ctx

    # ── zones ────────────────────────────────────────────────────────────────

    def _load_zones(self, ctx: RoutingContext) -> List[Dict[str, Any]]:
        rows = (self.db.table("zone").select("zone_id, zone_name").execute().data) or []
        ctx.zone_id_by_name = {
            r["zone_name"]: r["zone_id"] for r in rows if r.get("zone_name")
        }
        if not rows:
            # Not fatal, but Case D (the 07:30 Agargaon Metro consolidation) and
            # the zone-preferred vehicle tiers both go quiet — no error, just
            # worse routes. Say so out loud.
            ctx.warnings.append(
                "zone table is empty — zone-based rules (Agargaon Metro "
                "consolidation, zone-preferred vehicles) are disabled. "
                "Apply data/migrations/001_routing_integration.sql."
            )
        return rows

    # ── vehicles ─────────────────────────────────────────────────────────────

    def _load_vehicles(self, ctx: RoutingContext) -> List[Dict[str, Any]]:
        rows = (
            self.db.table("vehicle")
            .select(self.VEHICLE_SELECT)
            .eq("status", "Active")
            .execute()
            .data
        ) or []

        fleet: List[Dict[str, Any]] = []
        no_zone: List[str] = []
        no_driver: List[str] = []
        for row in rows:
            plate = row.get("plate_no")
            if not plate:
                continue
            driver = row.get("driver") or {}
            driver_user = driver.get("users") or {}
            zone_name = (row.get("zone") or {}).get("zone_name")

            ctx.vehicle_id_by_plate[plate] = row["vehicle_id"]
            ctx.driver_id_by_plate[plate] = driver.get("driver_id")
            if not zone_name:
                no_zone.append(plate)
            if not driver.get("driver_id"):
                no_driver.append(plate)

            fleet.append({
                "plate_no": plate,
                "capacity": row.get("capacity") or 1,
                "zone_name": zone_name,
                "parking_lat": _num(row.get("parking_lat")),
                "parking_lng": _num(row.get("parking_lng")),
                "status": row.get("status"),
                "driver_email": driver_user.get("email"),
                "driver_name": driver_user.get("name"),
            })

        if not fleet:
            ctx.warnings.append("no Active vehicles — nothing can be routed.")
        if no_zone:
            ctx.warnings.append(
                f"{len(no_zone)} vehicle(s) have no zone, so they only qualify as "
                f"last-resort spare capacity: {', '.join(sorted(no_zone)[:5])}"
                + (" …" if len(no_zone) > 5 else "")
            )
        if no_driver:
            # The route still gets built; it just has nobody to drive it, which a
            # dispatcher needs to see before the night starts.
            ctx.warnings.append(
                f"{len(no_driver)} vehicle(s) have no driver assigned: "
                f"{', '.join(sorted(no_driver)[:5])}" + (" …" if len(no_driver) > 5 else "")
            )
        return fleet

    # ── Case A fixed stops + vehicle_shifts ──────────────────────────────────

    def _load_fixed_stops(self, ctx: RoutingContext) -> List[Dict[str, Any]]:
        """`vehicle_pickup_location` — never read by the backend before now.

        Load-bearing twice: it supplies the named fixed stops the 22:00/23:00
        shifts match against (Case A), and the solver derives `vehicle_shifts`
        from it — which shifts each car works — gating both the pickup
        "assigned vehicle" pool and the drop-off tier-1 pool. Empty here is not
        an error and produces no exception; every event just falls through to
        borrow-from-anywhere and Case A degrades to door-to-door.
        """
        # The deployed Supabase was created from an older schema than
        # `data/schema.sql` and can be missing this table entirely (PostgREST
        # PGRST205). A missing table must not 500 the whole solve — it is the
        # same degradation as an empty one, so report it and carry on.
        try:
            rows = (
                self.db.table("vehicle_pickup_location")
                .select("vehicle_id, pickup_lat, pickup_lng, location_name, sequence_order, shift_time")
                .execute()
                .data
            ) or []
        except Exception as exc:  # noqa: BLE001 - a solve must still complete
            logger.warning("vehicle_pickup_location unreadable: %s", exc)
            ctx.warnings.append(
                "vehicle_pickup_location could not be read — apply "
                "001_routing_integration.sql, which creates it. Case A "
                "(fixed-route matching) is disabled until then."
            )
            return []

        plate_by_id = {vid: plate for plate, vid in ctx.vehicle_id_by_plate.items()}
        stops: List[Dict[str, Any]] = []
        orphaned = 0
        for row in rows:
            plate = plate_by_id.get(row.get("vehicle_id"))
            if plate is None:
                # Belongs to an inactive or deleted vehicle — not usable this night.
                orphaned += 1
                continue
            stops.append({
                "vehicle_plate": plate,
                "pickup_lat": _num(row.get("pickup_lat")),
                "pickup_lng": _num(row.get("pickup_lng")),
                "location_name": row.get("location_name"),
                "sequence_order": row.get("sequence_order"),
                "shift_time": _clock(row.get("shift_time")),
            })

        if not stops:
            ctx.warnings.append(
                "vehicle_pickup_location is empty — fixed-route matching (Case A) "
                "is disabled and every vehicle is treated as working every shift."
            )
        if orphaned:
            logger.info("%d fixed stop(s) skipped — vehicle not Active", orphaned)
        return stops

    # ── requests ─────────────────────────────────────────────────────────────

    def _load_requests(
        self, ctx: RoutingContext, table: str, service_date: str
    ) -> List[Dict[str, Any]]:
        rows = (
            self.db.table(table)
            .select(self.REQUEST_SELECT)
            .eq("service_date", service_date)
            .execute()
            .data
        ) or []

        rows = [r for r in rows if r.get("status") not in EXCLUDED_STATUSES]
        rows = _dedupe_latest_per_employee(rows)

        is_pickup = table == "pickup_request"
        id_field = "pickup_id" if is_pickup else "dropoff_id"
        id_map = ctx.pickup_id_by_email if is_pickup else ctx.dropoff_id_by_email

        out: List[Dict[str, Any]] = []
        no_email = 0
        no_zone = 0
        for row in rows:
            email = _email_of(row)
            if not email:
                # Without the natural key the solver's output cannot be mapped
                # back to this row, so routing it would lose the assignment.
                no_email += 1
                continue

            employee_id = row.get("employee_id")
            if employee_id is not None:
                ctx.employee_id_by_email[email] = employee_id
            id_map[email] = row[id_field]
            name = _name_of(row)
            if name:
                self._employee_names[email] = name

            zone_name = (row.get("zone") or {}).get("zone_name")
            if not zone_name:
                no_zone += 1

            if is_pickup:
                out.append({
                    "employee_email": email,
                    "zone_name": zone_name,
                    "pickup_lat": _num(row.get("pickup_lat")),
                    "pickup_lng": _num(row.get("pickup_lng")),
                    "shift_start_time": _clock(row.get("shift_start_time")),
                    "service_date": row.get("service_date"),
                    "request_type": row.get("request_type"),
                    "status": row.get("status"),
                    "vehicle_plate": row.get("vehicle_plate"),
                })
            else:
                out.append({
                    "employee_email": email,
                    "zone_name": zone_name,
                    "drop_lat": _num(row.get("drop_lat")),
                    "drop_lng": _num(row.get("drop_lng")),
                    "shift_end_time": _clock(row.get("shift_end_time")),
                    "drop_time": _clock(row.get("drop_time")),
                    "service_date": row.get("service_date"),
                    "status": row.get("status"),
                    "vehicle_plate": row.get("vehicle_plate"),
                })

        if no_email:
            ctx.warnings.append(
                f"{no_email} {table} row(s) skipped — no linked employee/user record."
            )
        if no_zone:
            # This is mismatch #10: with a NULL zone the 07:30 metro rule never
            # fires and the drop-off vehicle tiers collapse, silently.
            ctx.warnings.append(
                f"{no_zone} of {len(rows)} {table} row(s) have no zone_id — "
                "zone rules will not apply to them. "
                "Run data/migrations/002_zone_backfill.sql."
            )
        return out


def load(db: Client, service_date: str) -> RoutingContext:
    return RoutingAdapter(db).load(service_date)


# ============================================================================
# Section 4/4 -- the solver (vendored backend routing/solver.py, incl. ML)
# ============================================================================
"""The routing algorithm — a faithful port of `data/routing_night.py`.

`routing_night.py` is the standalone twin of `data/testing.ipynb` and the
current authoritative algorithm. This module ports it into the backend's pure
solver shape, preserving every behavioural decision:

- **One chronological timeline** of interleaved pickup and drop-off events,
  anchored at 22:00 on the service date so ordering survives midnight.
- **Fleet state** per vehicle (`current_location`, `status`, `_free_at`)
  carries forward across the whole night, so a car starts from where it
  actually is. A drop-off tour ends at the LAST STOP, not the office.
- **Case A (22:00)**: fixed-route matching against the roster's stops, walking
  times from the OSRM foot network (≤ `walk_limit_min`), ad-hoc door stops for
  riders with no reachable stop, then `redistribute_case_a` for cap shedding.
- **Case B (23:00)**: greedy nearest-car door-to-door with a near-tie slack.
- **Case B-kmeans (00:00–06:00)**: capacity-constrained k-means over rider
  homes, exact cluster→car matching, `_spill_riders` safety net.
- **Exact fair ordering** (Held-Karp) for pickups (minimise total passenger
  ride time) and **exact shortest open tours** for drop-offs
  (price the closing leg at `dropoff_return_weight`).
- **Case C / Case D drop-offs**: door-to-door, or the 07:30 Agargaon Metro /
  main-road consolidation (Mirpur box + Uttara quad, Friday exception). The
  22:15/23:15 evening drop-offs use the least-squares (Hungarian) fit against
  each car's own fixed route.
- **Second chance**: every rider the first pass shed is offered one more car,
  in the shift's own policy order, before anything is reported.
- **Cap shedding**: enforce the 120-min passenger cap (and, for pickups, the
  car's free window) by dropping whole stops.

ML travel-time model: leg durations used for ordering and timing come from the
trained XGBoost bundle (`ml_model/inference_bundle.joblib`) instead of raw
OSRM durations, exactly as the previous port did. Distance and geometry are
unaffected — they still come from the injected `DistanceProvider`. Set
`use_ml=False` to reproduce the notebook byte-for-byte with raw OSRM durations
(this is what the parity test uses). The foot network is never ML-predicted.

Deviations from the script, all deliberate and all listed here:

1. The three crash sites are softened (a notebook may raise; a request handler
   may not): an event spanning several `shift_end_time`s is split instead of
   asserted, an unknown employee falls back to its email instead of raising
   `KeyError`, and vehicles with no parking coordinates are reported as warnings
   instead of vanishing.
2. Case D honours the "except Fridays" rule that the script's own header
   documents but its code omits. Controlled by `SolverConfig.apply_friday_exception`.
3. Route records carry an extra `zone_name` (the modal zone) so the caller can
   set `route.zone_id`, and drop-off records carry `parking_arrival`
   (= `tour_end`) so `writer`'s assignment insert keeps a real arrival time.
"""

import logging
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger("uvicorn.error")

Coord = Tuple[float, float]

# Friday is weekday() == 4. Dhaka Metro Rail does not run on Fridays, which is
# why BDS exempts that day from the Agargaon Metro consolidation.
_FRIDAY = 4

# Case B-kmeans applies to these pick-up shifts; 11 PM stays on the greedy loop.
KMEANS_PICKUP_SHIFTS = frozenset({
    "00:00:00", "01:00:00", "02:00:00", "03:00:00",
    "04:00:00", "05:00:00", "06:00:00",
})

# The evening drop-offs are fitted against each car's own fixed route by
# least-squares (Hungarian), not by the reuse/tier rule.
EVENING_FIT_EVENTS = frozenset({"22:15:00", "23:15:00"})

# k-means tuning now lives on SolverConfig (near_tie_slack, near_tie_km_allowance,
# cluster_zone_penalty_km, cluster_restarts, cluster_seed) so a sweep can override
# them per run instead of editing this file.

# Evening-fit prices: no roster curve (usable but last), and a dummy seat.
_NO_CURVE_COST = 1.0e6
_UNSEATABLE_COST = 1.0e9

# Case D (07:30) geography — the Mirpur / Uttara boundary, fixed as plain
# constants (no OSM dependency at runtime).
MIRPUR_BBOX = (23.80520, 90.35941, 23.83011, 90.38381)
UTTARA_QUAD = [(90.3725, 23.8943), (90.4022, 23.8931),
               (90.4085, 23.8512), (90.3662, 23.8585)]


# ──────────────────────────────────────────────────────────────────────────────
# ML travel-time model (GPS_TRACE_ML/trained-model/inference_bundle.joblib)
# ──────────────────────────────────────────────────────────────────────────────
#
# Leg *durations* for ordering and timing come from this trained XGBoost model
# instead of raw OSRM/haversine durations. Distance_km and route geometry are
# unaffected — they still come straight from the injected `DistanceProvider`,
# and the model itself needs that provider's own duration/distance as two of
# its input features (it is a correction layer on top of OSRM, not a
# replacement for it). Feature engineering here is a direct port of
# `GPS_TRACE_ML/test_inference_bundle.py`'s `build_feature_row`, kept
# self-contained since that project lives outside this backend package.

_ML_BUNDLE_ENV_VAR = "ROUTING_ML_MODEL_PATH"
# Two known layouts: this standalone repo ships its own copy right next to the
# script (ml_model/inference_bundle.joblib); the original monorepo checkout has
# it under the sibling backend package instead. Try the colocated one first.
_ML_BUNDLE_LOCAL_PATH = Path(__file__).resolve().parent / "ml_model" / "inference_bundle_Retrained_V1.joblib"
_ML_BUNDLE_EXTERNAL_PATH = (Path(__file__).resolve().parent.parent
    / "Data-Driven-Employee-Routing-System" / "backend" / "app"
    / "services" / "routing" / "ml_model" / "inference_bundle_Retrained_V1.joblib")
_ML_BUNDLE_DEFAULT_PATH = (_ML_BUNDLE_LOCAL_PATH if _ML_BUNDLE_LOCAL_PATH.exists()
                          else _ML_BUNDLE_EXTERNAL_PATH)

_ml_bundle_cache: Optional[Dict[str, Any]] = None

_ML_FIXED_HOLIDAYS_MD = {(2, 21), (3, 26), (4, 14), (5, 1), (8, 15), (12, 16), (12, 25)}


def _normalise_ml_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Map a retrained bundle's naming onto the original one `_ml_feature_row` reads.

    The retrained bundles (e.g. inference_bundle_Retrained_V1.joblib) store the
    historical-speed lookup tables as (speed, n_trips) and the global fallback
    as `global_speed`; the original stores (mean, count) and `global_mean`.
    Everything else -- index names, feature/categorical columns, models -- is
    identical, so renaming here lets either bundle drive the same feature code.
    """
    renames = {"speed": "mean", "n_trips": "count"}
    for key in ("lvl1", "lvl2", "lvl3"):
        table = bundle[key]
        if "mean" not in table.columns or "count" not in table.columns:
            bundle[key] = table.rename(columns=renames)
    if "global_mean" not in bundle and "global_speed" in bundle:
        bundle["global_mean"] = bundle["global_speed"]
    return bundle


def _load_ml_bundle() -> Dict[str, Any]:
    """Loads `inference_bundle.joblib` once per process."""
    global _ml_bundle_cache
    if _ml_bundle_cache is None:
        path = os.environ.get(_ML_BUNDLE_ENV_VAR, str(_ML_BUNDLE_DEFAULT_PATH))
        _ml_bundle_cache = _normalise_ml_bundle(joblib.load(path))
    return _ml_bundle_cache


def _ml_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _ml_traffic_bucket(hour: float) -> str:
    if hour < 6:
        return "night"
    elif hour < 8:
        return "morning_offpeak"
    elif hour < 10:
        return "morning_rush"
    elif hour < 17:
        return "midday"
    elif hour < 21:
        return "evening_rush"
    return "evening_winddown"


def _ml_grid_cell(lat: float, lon: float, bundle: Dict[str, Any]) -> int:
    gp = bundle["grid_params"]
    r = min(int((lat - gp["min_lat"]) / gp["lat_step"]), gp["n_rows"] - 1)
    c = min(int((lon - gp["min_lon"]) / gp["lon_step"]), gp["n_cols"] - 1)
    return r * gp["n_cols"] + c


def _ml_feature_row(
    src: Coord,
    dst: Coord,
    query_time: datetime,
    osrm_route_distance_km: float,
    osrm_free_flow_duration_sec: float,
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """One (src, dst, query_time) trip -> the exact feature row the model expects."""
    src_lat, src_lon = src
    dst_lat, dst_lon = dst
    haversine_distance_km = haversine_km(src, dst)
    bearing_degrees = _ml_bearing_deg(src_lat, src_lon, dst_lat, dst_lon)
    route_directness_ratio = (
        haversine_distance_km / osrm_route_distance_km if osrm_route_distance_km > 0 else float("nan")
    )

    src_zone_id = _ml_grid_cell(src_lat, src_lon, bundle)
    dst_zone_id = _ml_grid_cell(dst_lat, dst_lon, bundle)
    od_zone_pair_id = f"{src_zone_id}_{dst_zone_id}"

    hour = query_time.hour + query_time.minute / 60.0 + query_time.second / 3600.0
    day_of_week = query_time.strftime("%A")
    is_friday = day_of_week == "Friday"
    is_saturday = day_of_week == "Saturday"
    is_weekend = is_friday or is_saturday
    rush_hour_flag = (8 <= hour < 10) or (17 <= hour < 21)
    bucket = _ml_traffic_bucket(hour)
    is_holiday = (query_time.month, query_time.day) in _ML_FIXED_HOLIDAYS_MD

    lvl1, lvl2, lvl3 = bundle["lvl1"], bundle["lvl2"], bundle["lvl3"]
    key1 = (od_zone_pair_id, bucket)
    if key1 in lvl1.index and lvl1.loc[key1, "count"] >= bundle["min_support"]:
        historical_avg_speed_kmh = lvl1.loc[key1, "mean"]
    elif od_zone_pair_id in lvl2.index and lvl2.loc[od_zone_pair_id, "count"] >= bundle["min_support"]:
        historical_avg_speed_kmh = lvl2.loc[od_zone_pair_id, "mean"]
    elif bucket in lvl3.index and lvl3.loc[bucket, "count"] >= bundle["min_support"]:
        historical_avg_speed_kmh = lvl3.loc[bucket, "mean"]
    else:
        historical_avg_speed_kmh = bundle["global_mean"]

    return {
        "src_zone_id": src_zone_id, "dst_zone_id": dst_zone_id, "od_zone_pair_id": od_zone_pair_id,
        "day_of_week": day_of_week, "traffic_period_bucket": bucket,
        "haversine_distance_km": haversine_distance_km, "bearing_degrees": bearing_degrees,
        "osrm_route_distance_km": osrm_route_distance_km,
        "osrm_free_flow_duration_sec": osrm_free_flow_duration_sec,
        "route_directness_ratio": route_directness_ratio,
        "hour_sin": math.sin(2 * math.pi * hour / 24.0), "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "is_friday": int(is_friday), "is_saturday": int(is_saturday), "is_weekend": int(is_weekend),
        "is_holiday": int(is_holiday), "rush_hour_flag": int(rush_hour_flag),
        "historical_avg_speed_kmh": historical_avg_speed_kmh,
    }


_ML_MODEL_KIND = "xgb"


def _ml_onehot_frame(frame: pd.DataFrame, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Expand the 5 raw categorical columns into the one-hot matrix `rf_model`
    was trained on (the 146 columns in `bundle["onehot_columns"]`)."""
    cat_cols = bundle["categorical_cols"]
    data: Dict[str, Any] = {}
    for c in frame.columns:
        if c not in cat_cols and c in bundle["onehot_columns"]:
            data[c] = frame[c].astype(float)
    for c in cat_cols:
        vals = frame[c].astype(str)
        for col in bundle["onehot_columns"]:
            if col.startswith(c + "_"):
                data[col] = (vals == col[len(c) + 1:]).astype(int)
    out = pd.DataFrame(data, index=frame.index)
    return out.reindex(columns=bundle["onehot_columns"], fill_value=0)


def _ml_predict_minutes_batch(rows: List[Dict[str, Any]], bundle: Dict[str, Any]) -> List[float]:
    """Batched prediction: one `.predict()` call for every leg in a matrix.
    Uses `xgb_model` on the raw categorical frame, or `rf_model` on the
    one-hot design matrix, depending on `_ML_MODEL_KIND`."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    for c in bundle["categorical_cols"]:
        frame[c] = pd.Categorical(frame[c].astype(str), categories=bundle["cat_categories"][c])
    for c in ["is_friday", "is_saturday", "is_weekend", "is_holiday", "rush_hour_flag"]:
        frame[c] = frame[c].astype(int)
    if _ML_MODEL_KIND == "rf":
        design = _ml_onehot_frame(frame, bundle)
        pred_seconds = np.exp(bundle["rf_model"].predict(design))
    else:
        pred_seconds = np.exp(bundle["xgb_model"].predict(frame[bundle["feature_cols"]]))
    return [float(s) / 60.0 for s in pred_seconds]


class _MlDurationProvider:
    """Decorates a `DistanceProvider`: durations come from the XGBoost model,
    distance_km and route geometry pass straight through unchanged.

    `query_time` must be set by the solver before each event is processed —
    the model's prediction is time-of-day/day-of-week dependent, and `table`/
    `route` carry no such argument in the `DistanceProvider` protocol.
    """

    name = "xgboost_ml"

    def __init__(self, inner: DistanceProvider):
        self.inner = inner
        self.query_time: Optional[datetime] = None
        self.bundle = _load_ml_bundle()
        self.name = f"{_ML_MODEL_KIND}_ml"

    def table(self, coords: Sequence[Coord]):
        raw_durations, distances = self.inner.table(coords)
        if self.query_time is None:
            return raw_durations, distances
        n = len(coords)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        rows = [
            _ml_feature_row(
                coords[i], coords[j], self.query_time,
                distances[i][j], raw_durations[i][j] * 60.0, self.bundle,
            )
            for i, j in pairs
        ]
        predicted = _ml_predict_minutes_batch(rows, self.bundle)
        durations = [row[:] for row in raw_durations]
        for (i, j), minutes in zip(pairs, predicted):
            durations[i][j] = minutes
        return durations, distances

    def route(self, coords: Sequence[Coord]):
        return self.inner.route(coords)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


# ──────────────────────────────────────────────────────────────────────────────
# Result shape
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SolvedNight:
    """Mirrors `solved_routes_walk20_retw0.json` so the script output is usable
    as a fixture.

    - `routes`      → `route_summary`
    - `stops`       → `route_stops`
    - `passengers`  → `stop_passengers`
    - `unassigned`  → `unassigned`

    `warnings` is new: data-quality problems that are not unassigned requests.
    """

    routes: List[Dict[str, Any]] = field(default_factory=list)
    stops: List[Dict[str, Any]] = field(default_factory=list)
    passengers: List[Dict[str, Any]] = field(default_factory=list)
    unassigned: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        return {
            "routes": len(self.routes),
            "pickup_routes": sum(1 for r in self.routes if r["type"] == "pickup"),
            "dropoff_routes": sum(1 for r in self.routes if r["type"] == "dropoff"),
            "stops": len(self.stops),
            "passengers": len(self.passengers),
            "unassigned": len(self.unassigned),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Overnight time algebra
# ──────────────────────────────────────────────────────────────────────────────

def normalise_clock(value: Any) -> str:
    """Any clock representation → "HH:MM:SS"."""
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M:%S")
    parts = str(value).strip().split(":")
    parts = (parts + ["00", "00"])[:3]
    return ":".join(f"{int(float(p)):02d}" for p in parts)


def night_offset(t) -> timedelta:
    """Elapsed time since 10 PM, wrapping past midnight."""
    td = timedelta(hours=t.hour, minutes=t.minute)
    start = timedelta(hours=22)
    return td - start if td >= start else td + timedelta(days=1) - start


def iso(dt: datetime) -> str:
    """Lossless serialisation: keeps the date, so overnight order survives."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ──────────────────────────────────────────────────────────────────────────────
# Solver
# ──────────────────────────────────────────────────────────────────────────────

class NightSolver:
    """One whole-night solve. Construct, call `solve()`, discard.

    Instance state replaces the script's module-level globals (`fleet`,
    `events`, `pickup_vehicle_by_employee`, the four output lists), so two
    solves can never contaminate each other.
    """

    def __init__(
        self,
        *,
        service_date: str,
        vehicles: Sequence[Dict[str, Any]],
        pickup_requests: Sequence[Dict[str, Any]],
        dropoff_requests: Sequence[Dict[str, Any]],
        fixed_stops: Sequence[Dict[str, Any]],
        provider: DistanceProvider,
        foot: Optional[FootDistanceProvider] = None,
        cfg: Optional[SolverConfig] = None,
        employee_names: Optional[Dict[str, str]] = None,
        use_ml: bool = True,
    ):
        self.cfg = cfg or SolverConfig()
        self.use_ml = use_ml
        self.provider = _MlDurationProvider(provider) if use_ml else provider
        self.foot = foot or get_foot_provider()
        self.office: Coord = self.cfg.office
        self.service_date = service_date
        self.pickup_requests = list(pickup_requests)
        self.dropoff_requests = list(dropoff_requests)
        self.fixed_stops = [s for s in fixed_stops if s.get("pickup_lat") is not None]
        self._employee_names = employee_names or {}

        self.night_anchor = datetime.strptime(service_date, "%Y-%m-%d").replace(
            hour=self.cfg.night_anchor_hour
        )

        self.out = SolvedNight()

        # A vehicle's assigned shifts = the distinct shift_time of its fixed
        # pickup stops. Load-bearing well beyond Case A: it gates the pickup
        # "dedicated vehicle" pool, the drop-off tier-1 pool, the evening fit
        # and the second chance.
        self.vehicle_shifts: Dict[str, set] = {}
        for s in self.fixed_stops:
            self.vehicle_shifts.setdefault(s["vehicle_plate"], set()).add(
                normalise_clock(s["shift_time"])
            )
        if not self.vehicle_shifts:
            self.out.warnings.append(
                "No vehicle pickup locations found: every vehicle is treated as "
                "unassigned to any shift, so all routing falls back to "
                "borrow-from-anywhere and no fixed-route (Case A) stops exist."
            )

        # Every roster route, keyed (plate, shift_time) -> its stops in
        # sequence_order. Built once from the same catalog Case A reads; it is
        # the fitting curve for the evening (22:15/23:15) drop-offs.
        self._route_by_car_shift: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for s in self.fixed_stops:
            self._route_by_car_shift.setdefault(
                (s["vehicle_plate"], normalise_clock(s["shift_time"])), []
            ).append(s)
        for stops in self._route_by_car_shift.values():
            stops.sort(key=lambda s: (s["sequence_order"] is None, s["sequence_order"]))

        # Case D main-road drops: the nearest *catalog* stop ("pickup points of
        # a car, but in reverse"). No walk rule and no ad-hoc home fallback at
        # 07:30 -- the catalog stop IS the main-road drop point.
        self._main_road_stops = [
            (s["location_name"], (s["pickup_lat"], s["pickup_lng"]))
            for s in self.fixed_stops
            if s.get("pickup_lat") is not None and s.get("pickup_lng") is not None
        ]

        # Case D fires the morning AFTER the service date, so Friday means
        # service date + 1 day.
        self._is_friday_dropoff = (
            datetime.strptime(service_date, "%Y-%m-%d") + timedelta(days=1)
        ).weekday() == _FRIDAY

        self.fleet: Dict[str, Dict[str, Any]] = {}
        self._build_fleet(vehicles)

        # Which vehicle picked up which employee — drives drop-off reuse.
        self.pickup_vehicle_by_employee: Dict[str, str] = {}

        self.events: List[Dict[str, Any]] = []

    # ── setup ────────────────────────────────────────────────────────────────

    def _build_fleet(self, vehicles: Sequence[Dict[str, Any]]) -> None:
        skipped: List[str] = []
        for v in vehicles:
            # A vehicle with no parking coordinates would poison every distance
            # computation it touches (current_location = (None, None)).
            if v.get("parking_lat") is None or v.get("parking_lng") is None:
                skipped.append(str(v.get("plate_no")))
                continue
            plate = v["plate_no"]
            self.fleet[plate] = {
                "plate_no": plate,
                "capacity": int(v["capacity"]),
                "zone_name": v.get("zone_name"),      # = route_area
                "driver_email": v.get("driver_email"),
                "parking_lat": float(v["parking_lat"]),
                "parking_lng": float(v["parking_lng"]),
                "current_location": (float(v["parking_lat"]), float(v["parking_lng"])),
                "status": "AVAILABLE",
                "_trip_end_time": None,       # clock time this trip ends
                "_trip_end_location": None,
                "_free_at": None,             # earliest this car may start a NEW trip
                "_stops": {},
                "_remaining": int(v["capacity"]),
                "_used": 0,
            }
        if skipped:
            self.out.warnings.append(
                f"{len(skipped)} vehicle(s) excluded from the fleet — no parking "
                f"coordinates: {', '.join(sorted(skipped))}"
            )

    def _employee_name(self, email: Optional[str]) -> str:
        """Never raises. The script's `user_by_email[email]["name"]` would."""
        if not email:
            return "Unknown employee"
        return self._employee_names.get(email) or str(email)

    def _raw_time(self, s: Any):
        return datetime.strptime(normalise_clock(s), "%H:%M:%S").time()

    def _parse_time(self, s: Any) -> datetime:
        """Clock string → night-anchored datetime (monotonic across midnight)."""
        return self.night_anchor + night_offset(self._raw_time(s))

    # ── timeline ─────────────────────────────────────────────────────────────

    def _build_timeline(self) -> None:
        """One chronological timeline, GROUPED by shift (one event per shift).

        Requests without coordinates are excluded here and reported as
        `no_coordinates` — they must never reach the geometry.
        """
        pickup_by_shift: Dict[str, List[Dict[str, Any]]] = {}
        for pr in self.pickup_requests:
            if pr.get("pickup_lat") is None or pr.get("pickup_lng") is None:
                continue
            pickup_by_shift.setdefault(normalise_clock(pr["shift_start_time"]), []).append(pr)
        for shift_time, reqs in pickup_by_shift.items():
            self.events.append(
                {"type": "pickup", "time": shift_time, "shift_time": shift_time, "requests": reqs}
            )

        dropoff_by_shift: Dict[str, List[Dict[str, Any]]] = {}
        for d in self.dropoff_requests:
            if d.get("drop_lat") is None or d.get("drop_lng") is None:
                continue
            dropoff_by_shift.setdefault(normalise_clock(d["drop_time"]), []).append(d)

        for drop_time, reqs in dropoff_by_shift.items():
            # The script asserts one shift_end_time per drop_time. Real data
            # will eventually violate that; splitting the event is correct and
            # keeps the office-departure timing exact for each sub-group.
            by_end: Dict[str, List[Dict[str, Any]]] = {}
            for r in reqs:
                by_end.setdefault(normalise_clock(r["shift_end_time"]), []).append(r)
            if len(by_end) > 1:
                self.out.warnings.append(
                    f"drop_time {drop_time} spans {len(by_end)} shift end times "
                    f"({', '.join(sorted(by_end))}); split into separate events."
                )
            for shift_end_time, group in by_end.items():
                self.events.append(
                    {
                        "type": "dropoff",
                        "time": drop_time,             # scheduled drop time
                        "shift_time": shift_end_time,  # office departure label
                        "requests": group,
                    }
                )

        # _parse_time is night-anchored, so this sorts correctly across midnight.
        self.events.sort(key=lambda e: (self._parse_time(e["time"]), e["time"], e["shift_time"]))

    def _report_missing_coordinates(self) -> None:
        for pr in self.pickup_requests:
            if pr.get("pickup_lat") is None or pr.get("pickup_lng") is None:
                self.out.unassigned.append(
                    self._unassigned_row(pr, "pickup", pr.get("shift_start_time"), "no_coordinates")
                )
        for d in self.dropoff_requests:
            if d.get("drop_lat") is None or d.get("drop_lng") is None:
                self.out.unassigned.append(
                    self._unassigned_row(d, "dropoff", d.get("shift_end_time"), "no_coordinates")
                )

    def _unassigned_row(
        self,
        request: Dict[str, Any],
        request_type: str,
        shift_time: Any,
        reason: str,
        plate_no: Optional[str] = None,
    ) -> Dict[str, Any]:
        email = request.get("employee_email")
        return {
            "employee_email": email,
            "employee_name": self._employee_name(email),
            "type": request_type,
            "shift_time": normalise_clock(shift_time) if shift_time else None,
            "reason": reason,
            "vehicle_id": plate_no,
        }

    # ── fleet state ──────────────────────────────────────────────────────────

    def _update_fleet(self, trip_start: datetime) -> None:
        """Release cars whose previous trip has finished by `trip_start`.

        `trip_start` is when the NEXT trip actually begins, not when the event
        fires — those differ. A pickup is planned backward from `shift - 5 min`;
        a drop-off leaves the office at its `drop_time`. Comparing against the
        event clock would let a car be dispatched before its previous trip had
        ended.

        Releasing here sets `current_location` to where that trip ended, which
        is the whole cascade: a pick-up then starts from the last drop's final
        stop, not from the office it never went back to.
        """
        for v in self.fleet.values():
            if v["status"] == "IN_TRIP" and v.get("_trip_end_time") and v["_trip_end_time"] <= trip_start:
                v["status"] = "AVAILABLE"
                v["current_location"] = v["_trip_end_location"]

    @staticmethod
    def _free_seats(v: Dict[str, Any]) -> int:
        """Seats left for THIS drop-off event (never the stale pickup _remaining)."""
        return v["capacity"] - v.get("_used", 0)

    # ── ordering primitives ──────────────────────────────────────────────────

    def _pair_minutes(self, a: Coord, b: Coord) -> float:
        """Driving minutes for ONE leg (the provider caches on the coord tuple)."""
        if a == b:
            return 0.0
        durations, _ = self.provider.table([a, b])
        return durations[0][1]

    def _held_karp_order(self, durations, stop_idx, start_idx, end_idx, weights=None):
        """Cheapest path start_idx -> (every stop once, any order) -> end_idx.

        Held-Karp bitmask DP. dp[mask][k] = minimum cost of a path that leaves
        start_idx, visits exactly the stops in `mask`, and ends at stop_idx[k];
        parent[] records the move used so the winning order can be
        reconstructed. Directed durations are used as-is — no symmetry assumed.
        O(n^2 * 2^n): instant for the <= ~11 stops a trip actually carries.

        `weights` prices each leg: weights[j] multiplies the leg that arrives
        at the (j+1)-th stop, and weights[n] multiplies the closing leg to
        end_idx. All ones means "shortest total time"; anything else prices the
        legs by how much they cost the passengers rather than the fleet — see
        `_ride_weights` and `_best_stop_order`.
        """
        n = len(stop_idx)
        size = 1 << n
        INF = float("inf")
        if weights is None:
            weights = [1.0] * (n + 1)
        nbits = [0] * size
        for m in range(1, size):
            nbits[m] = nbits[m >> 1] + (m & 1)
        dp = [[INF] * n for _ in range(size)]
        parent = [[-1] * n for _ in range(size)]
        for k in range(n):
            dp[1 << k][k] = weights[0] * durations[start_idx][stop_idx[k]]
        for mask in range(1, size):
            leg_in = nbits[mask]
            for k in range(n):
                if not (mask & (1 << k)) or dp[mask][k] == INF:
                    continue
                for j in range(n):
                    if mask & (1 << j):
                        continue
                    nmask = mask | (1 << j)
                    cand = dp[mask][k] + weights[leg_in] * durations[stop_idx[k]][stop_idx[j]]
                    if cand < dp[nmask][j]:
                        dp[nmask][j] = cand
                        parent[nmask][j] = k
        full = size - 1
        best_last, best_cost = -1, INF
        for k in range(n):
            cand = dp[full][k] + weights[n] * durations[stop_idx[k]][end_idx]
            if cand < best_cost:
                best_cost, best_last = cand, k
        if best_last == -1:
            return list(stop_idx)
        order_rev, mask, k = [], full, best_last
        while k != -1:
            order_rev.append(stop_idx[k])
            prev = parent[mask][k]
            mask ^= (1 << k)
            k = prev
        return order_rev[::-1]

    @staticmethod
    def _ride_weights(n: int, kind: str) -> List[float]:
        """Leg prices: one unit per passenger aboard, but never less than one.

        Summing those prices over the legs gives the TOTAL TIME PASSENGERS
        SPEND IN THE CAR, so an order that minimises it is the order that
        minimises total riding — the fairness objective. The `max(..., 1)` floor
        keeps an empty repositioning leg from being free.

        pickups   leg 0 is the repositioning leg (empty, floor 1), leg j
                  carries j passengers, the final run carries everybody.
        drop-offs leg j carries n - j passengers, the closing run is empty (1).
        """
        if kind == "pickup":
            return [max(1.0, float(j)) for j in range(n + 1)]
        return [max(1.0, float(n - j)) for j in range(n)] + [1.0]

    def _best_stop_order(self, durations, stop_idx, start_idx, end_idx, fair=True, kind="pickup"):
        """Exact order of `stop_idx` between the two fixed anchors.

        `fair=True` minimises TOTAL PASSENGER RIDE TIME; `fair=False` minimises
        total route time. Both are exact — the DP enumerates every one of the
        n! orders implicitly either way.
        """
        n = len(stop_idx)
        if n > MAX_STOPS_FOR_EXACT:
            raise ValueError(
                "route has %d stops > MAX_STOPS_FOR_EXACT=%d: exact search would not "
                "finish; trips are capacity-bounded so this should be unreachable."
                % (n, MAX_STOPS_FOR_EXACT))
        weights = self._ride_weights(n, kind) if fair else None
        return self._held_karp_order(durations, stop_idx, start_idx, end_idx, weights)

    # ── Case A fixed-route helpers ───────────────────────────────────────────

    def _stops_for_shift(self, shift_time: str, vehicles_this_shift) -> List[Dict[str, Any]]:
        """Fixed stops for a shift: this shift's stops, on cars actually in service."""
        plates_in_service = {v["plate_no"] for v in vehicles_this_shift}
        return [
            s for s in self.fixed_stops
            if normalise_clock(s["shift_time"]) == shift_time
            and s["vehicle_plate"] in plates_in_service
        ]

    def _request_zone(self, pr: Dict[str, Any]) -> Optional[str]:
        """The zone a rider belongs to: their own label, else their car's."""
        z = pr.get("zone_name")
        if z:
            return z
        v = self.fleet.get(pr.get("vehicle_plate") or "")
        return v["zone_name"] if v else None

    # ── pickup: Case A / Case B / Case B-kmeans ──────────────────────────────

    def _place_on(self, v, pr, home, route_by_car):
        """(stop_key, stop_item) for putting `pr` on car `v` — WITHOUT mutating v.

        The nearest stop of THAT car's own fixed route within the walk limit,
        else an ad-hoc stop at the door. A stop keeps its identity across
        riders — keyed by its roster name, so everyone walking to a stop shares
        it. An ad-hoc stop carries `_rank` as the fallback order for when the
        exact placement (`_case_a_order`) cannot run.
        """
        route = route_by_car.get(v["plate_no"])
        if not route:
            return None
        best = None
        for i, s in enumerate(route):
            w = self.foot.walk_minutes(home, (s["pickup_lat"], s["pickup_lng"]))
            if w <= self.cfg.walk_limit_min and (best is None or w < best[0]):
                best = (w, i, s)
        if best is not None:
            _, i, s = best
            return s["location_name"], {
                "coord": (s["pickup_lat"], s["pickup_lng"]),
                "name": s["location_name"], "is_adhoc": False,
                "_rank": (i, 0, 0.0), "passengers": [pr]}
        anchor, anchor_km = 0, None
        for i, s in enumerate(route):
            km = haversine_km(home, (s["pickup_lat"], s["pickup_lng"]))
            if anchor_km is None or km < anchor_km:
                anchor, anchor_km = i, km
        return f"adhoc_{pr['employee_email']}", {
            "coord": home, "name": f"Ad-hoc ({self._employee_name(pr['employee_email'])})",
            "is_adhoc": True, "_rank": (anchor, 1, anchor_km), "passengers": [pr]}

    def _add_to(self, v, pr, home, route_by_car) -> bool:
        """Put `pr` on `v` in place. False if `v` has no route to put them on."""
        placed = self._place_on(v, pr, home, route_by_car)
        if placed is None:
            return False
        key, item = placed
        if key in v["_stops"]:
            v["_stops"][key]["passengers"].append(pr)
        else:
            v["_stops"][key] = item
        return True

    def _assign_pickup_event(self, event) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        shift_time = event["shift_time"]

        # Every FREE car is a candidate. The roster decides the ORDER cars are
        # tried in, not whether they may work at all.
        free_cars = [v for v in self.fleet.values() if v["status"] == "AVAILABLE"]
        if not free_cars:
            return [], list(event["requests"])   # fleet exhausted
        roster_plates = {v["plate_no"] for v in free_cars
                         if shift_time in self.vehicle_shifts.get(v["plate_no"], set())}

        for v in self.fleet.values():
            v["_stops"] = {}
            v["_remaining"] = 0        # a car that is out must never look like it has room
        for v in free_cars:
            v["_remaining"] = v["capacity"]

        requests_this_shift = event["requests"]
        unassigned: List[Dict[str, Any]] = []

        # --- Case A (10 PM only): the roster's own operation (Algorithm 1) ---
        if shift_time == "22:00:00":
            a_cars = [v for v in free_cars
                      if shift_time in self.vehicle_shifts.get(v["plate_no"], set())]

            # Rule 2: every car's designated route, in the roster's sequence_order.
            route_by_car = {}
            for s in self._stops_for_shift(shift_time, a_cars):
                route_by_car.setdefault(s["vehicle_plate"], []).append(s)
            for stops in route_by_car.values():
                stops.sort(key=lambda s: (s["sequence_order"] is None, s["sequence_order"]))
            plate_coords = {p: [(s["pickup_lat"], s["pickup_lng"]) for s in stops]
                            for p, stops in route_by_car.items()}

            # Rule 5: the cars a zone can be served by.
            cars_by_zone = {}
            for v in a_cars:
                if v["plate_no"] in route_by_car:
                    cars_by_zone.setdefault(v["zone_name"], []).append(v)
            zone_coords = {z: sorted({c for v in vs for c in plate_coords[v["plate_no"]]})
                           for z, vs in cars_by_zone.items()}

            def zone_of(pr):
                return self._request_zone(pr)

            def candidate_cars(pr):
                """The rider's designated car first, then the other 10 PM cars of
                their zone, nearest first."""
                home = (pr["pickup_lat"], pr["pickup_lng"])
                own = pr.get("vehicle_plate")
                out = []
                if own in route_by_car:
                    v = self.fleet.get(own)
                    if v is not None and v["status"] == "AVAILABLE":
                        out.append(v)
                rest = [v for v in cars_by_zone.get(zone_of(pr), [])
                        if v["plate_no"] != own]
                rest.sort(key=lambda v: haversine_km(home, v["current_location"]))
                return out + rest

            # Rule 3: walking times come from the foot network, batched up front.
            seen_homes = set()
            for pr in requests_this_shift:
                home = (pr["pickup_lat"], pr["pickup_lng"])
                if home in seen_homes:
                    continue
                seen_homes.add(home)
                self.foot.prefetch(home, set(zone_coords.get(zone_of(pr), []))
                                          | set(plate_coords.get(pr.get("vehicle_plate"), [])))

            # Rule 4: serve the most constrained employees first.
            def priority_key(pr):
                home = (pr["pickup_lat"], pr["pickup_lng"])
                walks_in_range = [w for w in
                                  (self.foot.walk_minutes(home, (s["pickup_lat"], s["pickup_lng"]))
                                   for s in route_by_car.get(pr.get("vehicle_plate"), []))
                                  if w <= self.cfg.walk_limit_min]
                return (len(walks_in_range), -min(walks_in_range, default=999))

            def serve(pr, v, home):
                self._add_to(v, pr, home, route_by_car)
                v["_remaining"] -= 1

            for pr in sorted(requests_this_shift, key=priority_key):
                home = (pr["pickup_lat"], pr["pickup_lng"])
                for v in candidate_cars(pr):
                    if v["_remaining"] <= 0:
                        continue
                    serve(pr, v, home)
                    break
                else:
                    unassigned.append(pr)
            return free_cars, unassigned

        # --- Case B-kmeans (12 AM - 6 AM): cluster the riders, then match cars ---
        if shift_time in KMEANS_PICKUP_SHIFTS:
            return self._assign_pickup_clustered(free_cars, requests_this_shift,
                                                 shift_time, roster_plates)

        # --- Case B (11 PM): door-to-door — one greedy nearest-car pass ---
        pending = sorted(requests_this_shift,
                         key=lambda pr: min(haversine_km((pr["pickup_lat"], pr["pickup_lng"]),
                                               v["current_location"]) for v in free_cars))
        for pr in pending:
            home = (pr["pickup_lat"], pr["pickup_lng"])
            zone = pr.get("zone_name")
            with_room = [x for x in free_cars if x["_remaining"] > 0]
            if not with_room:
                unassigned.append(pr)
                continue
            pool = [x for x in with_room if x["plate_no"] in roster_plates] or with_room
            dists = {x["plate_no"]: haversine_km(home, x["current_location"]) for x in pool}
            best = min(dists.values())
            near = [x for x in pool if dists[x["plate_no"]] <= max(
                best * self.cfg.near_tie_slack, best + self.cfg.near_tie_km_allowance)]
            v = min(near, key=lambda x: (0 if x["_stops"] else 1,
                                         dists[x["plate_no"]],
                                         0 if x["zone_name"] == zone else 1))
            v["_stops"].setdefault(f"home_{pr['employee_email']}", {
                "coord": home,
                "name": f"Home ({self._employee_name(pr['employee_email'])})",
                "is_adhoc": True, "passengers": [],
            })["passengers"].append(pr)
            v["_remaining"] -= 1
        return free_cars, unassigned

    def _order_stops_pickup(self, vehicle) -> List[Tuple[Any, Dict[str, Any]]]:
        """Exact FAIREST order: car.current_location -> stops -> OFFICE.

        The objective is total passenger ride time, not total route time. Case A
        is the exception: the roster's fixed stops are pinned in their own
        `sequence_order` and only the ad-hoc door stops are placed, exactly, by
        `_case_a_order`.
        """
        items = list(vehicle["_stops"].items())
        if len(items) <= 1:
            return items
        if all("_rank" in it for _, it in items):
            if (any(it["is_adhoc"] for _, it in items)
                    and len(items) <= MAX_STOPS_FOR_EXACT):
                return self._case_a_order(vehicle, items)
            return sorted(items, key=lambda kv: kv[1]["_rank"])
        coords = [vehicle["current_location"]] + [it["coord"] for _, it in items] + [self.office]
        START, END = 0, len(items) + 1
        durations, _ = self.provider.table(coords)
        stop_idx = list(range(1, len(items) + 1))
        ordered = self._best_stop_order(durations, stop_idx, START, END, kind="pickup")
        return [items[i - 1] for i in ordered]

    def _case_a_order(self, vehicle, items) -> List[Tuple[Any, Dict[str, Any]]]:
        """Case A: the roster's fixed stops in the roster's own order, with the
        ad-hoc (door) stops slotted optimally among them.

        The DP's state is (how many fixed stops are behind us, which doors are
        placed, where we are standing); the objective is the 120-min cap's own
        quantity (first pickup to office), ties broken on the full trip.
        """
        pinned = sorted(items, key=lambda kv: kv[1]["_rank"])
        fixed = [(k, it) for k, it in pinned if not it["is_adhoc"]]
        doors = [(k, it) for k, it in pinned if it["is_adhoc"]]
        if not doors:
            return pinned
        n, k = len(fixed), len(doors)
        coords = ([vehicle["current_location"]]
                  + [it["coord"] for _, it in fixed]
                  + [it["coord"] for _, it in doors]
                  + [self.office])
        durations, _ = self.provider.table(coords)
        START, END = 0, len(coords) - 1
        FIXED = list(range(1, 1 + n))
        DOOR = list(range(1 + n, 1 + n + k))
        DOMINATE = 1e4

        def leg(u, w):
            d = durations[u][w]
            return DOMINATE * (0.0 if u == START else d) + d

        size = 1 << k
        INF = float("inf")
        AT_FIXED = 0
        dp = [[[INF] * (k + 1) for _ in range(size)] for _ in range(n + 1)]
        back = [[[None] * (k + 1) for _ in range(size)] for _ in range(n + 1)]
        dp[0][0][AT_FIXED] = 0.0
        for i in range(n + 1):
            for mask in range(size):
                for j in range(k + 1):
                    cur = dp[i][mask][j]
                    if cur == INF:
                        continue
                    if j != AT_FIXED:
                        here = DOOR[j - 1]
                    else:
                        here = START if i == 0 else FIXED[i - 1]
                    if i < n:
                        nxt = cur + leg(here, FIXED[i])
                        if nxt < dp[i + 1][mask][AT_FIXED]:
                            dp[i + 1][mask][AT_FIXED] = nxt
                            back[i + 1][mask][AT_FIXED] = (i, mask, j)
                    for l in range(k):
                        if mask & (1 << l):
                            continue
                        nxt = cur + leg(here, DOOR[l])
                        if nxt < dp[i][mask | (1 << l)][l + 1]:
                            dp[i][mask | (1 << l)][l + 1] = nxt
                            back[i][mask | (1 << l)][l + 1] = (i, mask, j)

        full = size - 1
        best, best_j = INF, AT_FIXED
        for j in range(k + 1):
            if dp[n][full][j] == INF:
                continue
            if j == AT_FIXED:
                if n == 0:
                    continue
                here = FIXED[n - 1]
            else:
                here = DOOR[j - 1]
            cand = dp[n][full][j] + leg(here, END)
            if cand < best:
                best, best_j = cand, j

        seq, i, mask, j = [], n, full, best_j
        while back[i][mask][j] is not None:
            pi, pmask, pj = back[i][mask][j]
            if i != pi:
                seq.append(fixed[i - 1])
            else:
                seq.append(doors[(mask ^ pmask).bit_length() - 1])
            i, mask, j = pi, pmask, pj
        seq.reverse()
        return seq

    def _compute_timing_pickup(self, vehicle, ordered_stops, shift_time) -> Dict[str, Any]:
        deadline = self._parse_time(shift_time) - timedelta(minutes=self.cfg.office_buffer_min)
        coords = [vehicle["current_location"]] + [it["coord"] for _, it in ordered_stops] + [self.office]
        durations, distances = self.provider.table(coords)
        legs = [durations[i][i + 1] for i in range(len(coords) - 1)]
        # The full trip includes the deadhead leg from wherever the car actually
        # started; the 120-min cap measures the passenger journey (first pickup
        # -> office), i.e. legs[1:] plus one boarding buffer per stop.
        total = sum(legs) + self.cfg.boarding_buffer_min * len(ordered_stops)
        passenger_total = sum(legs[1:]) + self.cfg.boarding_buffer_min * len(ordered_stops)
        parking_departure = deadline - timedelta(minutes=total)
        timestamps = []
        t = parking_departure
        for i, (key, _item) in enumerate(ordered_stops):
            t = t + timedelta(minutes=legs[i])
            arrival = t
            t = t + timedelta(minutes=self.cfg.boarding_buffer_min)
            timestamps.append({"stop_key": key, "arrival": arrival, "departure": t})
        office_arrival = t + timedelta(minutes=legs[-1])
        return {
            "parking_departure": parking_departure,
            "office_arrival": office_arrival,
            "total_minutes": total,
            "passenger_total_minutes": passenger_total,
            "leg_minutes": legs,
            "leg_km": [distances[i][i + 1] for i in range(len(coords) - 1)],
            "stop_timestamps": timestamps,
        }

    def _pickup_window_minutes(self, vehicle, shift_time) -> float:
        """How long this car may actually spend on the road for this shift.

        A pickup is planned BACKWARD from `shift - 5 min`, so the trip really
        starts at `parking_departure` — which can precede the event clock by up
        to two hours. Availability therefore has to be checked over the whole
        window, not at the event instant.
        """
        deadline = self._parse_time(shift_time) - timedelta(minutes=self.cfg.office_buffer_min)
        free_at = vehicle.get("_free_at")
        if free_at is None:
            return float("inf")
        return (deadline - free_at).total_seconds() / 60.0

    def _enforce_cap_pickup(self, vehicle, shift_time):
        """Shed stops until the route fits BOTH the 120-min cap and the free window.

        The cap counts the passenger journey (first pickup stop -> office); the
        free-window check measures the FULL trip, deadhead included.
        """
        window = self._pickup_window_minutes(vehicle, shift_time)
        reason = ("vehicle_not_free_in_time" if window < self.cfg.max_route_minutes
                  else "dropped_for_120min_cap")
        dropped: List[Dict[str, Any]] = []
        while True:
            ordered = self._order_stops_pickup(vehicle)
            if not ordered:
                return ordered, None, dropped, reason
            timing = self._compute_timing_pickup(vehicle, ordered, shift_time)
            over_cap = timing["passenger_total_minutes"] - self.cfg.max_route_minutes
            over_free = timing["total_minutes"] - window
            if over_cap <= 0 and over_free <= 0:
                return ordered, timing, dropped, reason
            best_key, best_score = None, None
            for key, _ in ordered:
                saved = vehicle["_stops"]
                vehicle["_stops"] = {k: v for k, v in saved.items() if k != key}
                trial = self._order_stops_pickup(vehicle)
                if trial:
                    tt = self._compute_timing_pickup(vehicle, trial, shift_time)
                    score = max(tt["passenger_total_minutes"] - self.cfg.max_route_minutes,
                                tt["total_minutes"] - window)
                else:
                    score = 0
                vehicle["_stops"] = saved
                if best_score is None or score < best_score:
                    best_score, best_key = score, key
            dropped.extend(vehicle["_stops"].pop(best_key)["passengers"])

    def _redistribute_case_a(self, vehicles_this_shift, shift_time) -> List[Dict[str, Any]]:
        """Case A only: shed the stops that break the cap, then re-place their
        riders on another 22:00 car of the SAME zone.

        Returns the riders no car could take. Mutates `_stops` on the cars it uses.
        """
        a_cars = [v for v in vehicles_this_shift
                  if shift_time in self.vehicle_shifts.get(v["plate_no"], set())]
        if not a_cars:
            return []
        by_zone = {}
        for v in a_cars:
            by_zone.setdefault(v["zone_name"], []).append(v)
        route_by_car = {}
        for s in self._stops_for_shift(shift_time, a_cars):
            route_by_car.setdefault(s["vehicle_plate"], []).append(s)
        for stops in route_by_car.values():
            stops.sort(key=lambda s: (s["sequence_order"] is None, s["sequence_order"]))

        tried: Dict[str, set] = {}

        def _seats(v):
            return v["capacity"] - sum(len(it["passengers"]) for it in v["_stops"].values())

        def _time(v, stops):
            saved = v["_stops"]
            v["_stops"] = stops
            try:
                ordered = self._order_stops_pickup(v)
                if not ordered:
                    return None, None
                return ordered, self._compute_timing_pickup(v, ordered, shift_time)
            finally:
                v["_stops"] = saved

        def _fits(v, stops):
            ordered, t = _time(v, stops)
            if not ordered:
                return False
            return (t["passenger_total_minutes"] <= self.cfg.max_route_minutes
                    and t["total_minutes"] <= self._pickup_window_minutes(v, shift_time))

        def _shed():
            out = []
            for v in a_cars:
                while v["_stops"] and not _fits(v, v["_stops"]):
                    window = self._pickup_window_minutes(v, shift_time)
                    best_key, best_over = None, None
                    for key in list(v["_stops"]):
                        trial = {x: y for x, y in v["_stops"].items() if x != key}
                        if trial:
                            _, t = _time(v, trial)
                            over = max(t["passenger_total_minutes"] - self.cfg.max_route_minutes,
                                       t["total_minutes"] - window)
                        else:
                            over = 0.0
                        if best_over is None or over < best_over:
                            best_over, best_key = over, key
                    for pr in v["_stops"].pop(best_key)["passengers"]:
                        tried.setdefault(pr["employee_email"], set()).add(v["plate_no"])
                        out.append(pr)
            return out

        def _replace(pr):
            home = (pr["pickup_lat"], pr["pickup_lng"])
            done = tried.setdefault(pr["employee_email"], set())
            pool = [v for v in by_zone.get(self._request_zone(pr), [])
                    if v["plate_no"] not in done]
            pool.sort(key=lambda v: haversine_km(home, v["current_location"]))
            for v in pool:
                done.add(v["plate_no"])
                if _seats(v) <= 0:
                    continue
                placed = self._place_on(v, pr, home, route_by_car)
                if placed is None:
                    continue
                key, item = placed
                stops = dict(v["_stops"])
                if key in stops:
                    stops[key] = dict(stops[key],
                                      passengers=list(stops[key]["passengers"]) + [pr])
                else:
                    stops[key] = item
                if _fits(v, stops):
                    v["_stops"] = stops
                    return True
            return False

        pool = _shed()
        for _ in range(len(a_cars) + 2):
            if not pool:
                return []
            left = [pr for pr in pool if not _replace(pr)]
            if len(left) == len(pool):
                return left
            pool = left + _shed()
        return pool

    # ── Case B-kmeans internals ──────────────────────────────────────────────

    @staticmethod
    def _cluster_xy(homes):
        """Rider homes in local kilometres (a degree of longitude spans ~0.92
        of a degree of latitude at Dhaka's latitude)."""
        lat0 = sum(h[0] for h in homes) / len(homes)
        lng0 = sum(h[1] for h in homes) / len(homes)
        kx = 111.32 * math.cos(math.radians(lat0))
        return [((h[1] - lng0) * kx, (h[0] - lat0) * 110.57) for h in homes]

    @staticmethod
    def _kmeans_fill(xy, cents, cap):
        """Send every rider to a cluster: nearest first, never past `cap`."""
        n, k = len(xy), len(cents)
        room = [cap] * k
        who = [-1] * n
        pairs = sorted((math.hypot(x - cx, y - cy), i, c)
                       for i, (x, y) in enumerate(xy)
                       for c, (cx, cy) in enumerate(cents))
        for _d, i, c in pairs:
            if who[i] == -1 and room[c] > 0:
                who[i] = c
                room[c] -= 1
        for i in range(n):
            if who[i] == -1:
                x, y = xy[i]
                free = [c for c in range(k) if room[c] > 0]
                c = min(free, key=lambda c: math.hypot(x - cents[c][0], y - cents[c][1]))
                who[i] = c
                room[c] -= 1
        return who

    def _kmeans_riders(self, homes, k, cap):
        """Capacity-constrained k-means over the riders' homes.

        Returns the rider indices of each cluster, taken from the tightest of
        `cfg.cluster_restarts` seeded k-means++ starts. The seed is fixed on purpose.
        """
        xy = self._cluster_xy(homes)
        n = len(xy)
        if k <= 1:
            return [list(range(n))] if n else []
        best, best_sse = None, None
        for r in range(self.cfg.cluster_restarts):
            rng = random.Random(self.cfg.cluster_seed * 9973 + r)
            cents = [list(xy[rng.randrange(n)])]
            while len(cents) < k:
                d2 = [min((x - cx) ** 2 + (y - cy) ** 2 for cx, cy in cents)
                      for x, y in xy]
                tot = sum(d2)
                if tot <= 0.0:
                    cents.append(list(xy[rng.randrange(n)]))
                    continue
                t, acc, pick = rng.random() * tot, 0.0, 0
                for i, v in enumerate(d2):
                    acc += v
                    if acc >= t:
                        pick = i
                        break
                cents.append(list(xy[pick]))
            who = []
            for _ in range(40):
                who = self._kmeans_fill(xy, cents, cap)
                acc = [[0.0, 0.0, 0] for _ in range(k)]
                for i, c in enumerate(who):
                    acc[c][0] += xy[i][0]
                    acc[c][1] += xy[i][1]
                    acc[c][2] += 1
                nxt = [[acc[c][0] / acc[c][2], acc[c][1] / acc[c][2]] if acc[c][2]
                       else list(cents[c]) for c in range(k)]
                if nxt == cents:
                    break
                cents = nxt
            sse = sum((xy[i][0] - cents[c][0]) ** 2 + (xy[i][1] - cents[c][1]) ** 2
                      for i, c in enumerate(who))
            if best_sse is None or sse < best_sse - 1e-9:
                best_sse = sse
                best = [[i for i, c in enumerate(who) if c == j] for j in range(k)]
        return [cl for cl in best if cl]

    def _match_clusters_to_cars(self, clusters, homes, rzones, cars):
        """Cheapest pairing of clusters to cars, or (None, None) if there is none."""
        k = len(clusters)
        cents = [(sum(homes[i][0] for i in cl) / len(cl),
                  sum(homes[i][1] for i in cl) / len(cl)) for cl in clusters]
        czone = [Counter(rzones[i] for i in cl).most_common(1)[0][0] for cl in clusters]
        cost = {}
        for ci, cl in enumerate(clusters):
            for v in cars:
                if v["capacity"] < len(cl):
                    continue
                pen = 0.0 if v["zone_name"] == czone[ci] else self.cfg.cluster_zone_penalty_km
                cost[(ci, v["plate_no"])] = haversine_km(cents[ci], v["current_location"]) + pen
        short_plates = set()
        for ci in range(k):
            for p in sorted((p for p in cost if p[0] == ci), key=lambda p: cost[p])[:k]:
                short_plates.add(p[1])
        short = [v for v in cars if v["plate_no"] in short_plates]
        if len(short) < k:
            return None, None
        if k > 4:
            taken, left, pick = set(), set(range(k)), {}
            for (ci, plate) in sorted(cost, key=lambda p: cost[p]):
                if ci in left and plate not in taken:
                    left.discard(ci)
                    taken.add(plate)
                    pick[ci] = plate
            if left:
                return None, None
            got = [next(v for v in cars if v["plate_no"] == pick[ci]) for ci in range(k)]
            return got, sum(cost[(ci, pick[ci])] for ci in range(k))
        best, best_cost = None, None
        for perm in permutations(short, k):
            c, ok = 0.0, True
            for ci, v in enumerate(perm):
                key = (ci, v["plate_no"])
                if key not in cost:
                    ok = False
                    break
                c += cost[key]
            if ok and (best_cost is None or c < best_cost):
                best_cost, best = c, perm
        if best is None:
            return None, None
        return list(best), best_cost

    def _cluster_stops(self, cluster, riders, homes):
        """The door stops for one cluster: one ad-hoc stop per rider."""
        return {f"home_{riders[i]['employee_email']}": {
            "coord": homes[i],
            "name": f"Home ({self._employee_name(riders[i]['employee_email'])})",
            "is_adhoc": True, "passengers": [riders[i]]} for i in cluster}

    def _route_fits(self, v, stops, shift_time) -> bool:
        """Would this car's pickup route, with exactly these stops, clear both clocks?"""
        saved = v["_stops"]
        v["_stops"] = stops
        try:
            ordered = self._order_stops_pickup(v)
            if not ordered:
                return True
            t = self._compute_timing_pickup(v, ordered, shift_time)
            return (t["passenger_total_minutes"] <= self.cfg.max_route_minutes
                    and t["total_minutes"] <= self._pickup_window_minutes(v, shift_time))
        finally:
            v["_stops"] = saved

    def _place_clusters_greedy(self, clusters, homes, rzones, free_cars):
        """Last resort: biggest cluster first, onto the cheapest car that can hold it."""
        out, taken = [], set()
        for cl in sorted(clusters, key=len, reverse=True):
            cent = (sum(homes[i][0] for i in cl) / len(cl),
                    sum(homes[i][1] for i in cl) / len(cl))
            z = Counter(rzones[i] for i in cl).most_common(1)[0][0]
            best, best_key = None, None
            for v in free_cars:
                if v["plate_no"] in taken or v["capacity"] < len(cl):
                    continue
                key = (0 if v["zone_name"] == z else 1,
                       haversine_km(cent, v["current_location"]))
                if best_key is None or key < best_key:
                    best_key, best = key, v
            if best is not None:
                taken.add(best["plate_no"])
                out.append((cl, best))
        return out

    def _spill_riders(self, left, free_cars):
        """Safety net: put any rider the clustering could not place onto the
        nearest car that still has a free seat."""
        out = []
        for pr in left:
            home = (pr["pickup_lat"], pr["pickup_lng"])
            room = [v for v in free_cars if v["_remaining"] > 0]
            if not room:
                out.append(pr)
                continue
            v = min(room, key=lambda v: haversine_km(home, v["current_location"]))
            v["_stops"].update(self._cluster_stops([0], [pr], [home]))
            v["_remaining"] -= 1
        return out

    def _assign_pickup_clustered(self, free_cars, requests_this_shift, shift_time,
                                 roster_plates) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """12 AM - 6 AM pick-ups: cluster the riders, then match the clusters to cars.

        k is swept upward from the fewest cars that can physically hold
        everyone, and the FIRST k whose routes clear both clocks wins.
        """
        riders = list(requests_this_shift)
        n = len(riders)
        if not n:
            return free_cars, []
        homes = [(r["pickup_lat"], r["pickup_lng"]) for r in riders]
        rzones = [self._request_zone(r) for r in riders]
        caps = sorted((v["capacity"] for v in free_cars), reverse=True)
        rostered = [v for v in free_cars if v["plate_no"] in roster_plates]
        pools = [(label, p) for label, p in
                 ((f"rostered({len(rostered)})", rostered), ("anyone", free_cars)) if p]

        def _match(clusters):
            for label, pool in pools:
                got, cost = self._match_clusters_to_cars(clusters, homes, rzones, pool)
                if got is not None:
                    return got, cost, label
            return None, None, None

        tries, chosen = [], None
        if n <= sum(caps):
            for k in range(max(1, math.ceil(n / caps[0])), min(len(free_cars), n) + 1):
                clusters = self._kmeans_riders(homes, k, caps[k - 1])
                if not clusters:
                    continue
                got, cost, label = _match(clusters)
                if got is None:
                    tries.append(f"k={k}: no car can hold every cluster")
                    continue
                if not all(self._route_fits(v, self._cluster_stops(cl, riders, homes), shift_time)
                           for cl, v in zip(clusters, got)):
                    tries.append(f"k={k}: a route breaks the 120-min cap or the free window")
                    continue
                chosen = (clusters, got, cost, label)
                break
        if chosen is None:
            clusters = self._kmeans_riders(homes, max(1, min(len(free_cars), n)), caps[0])
            got, cost, label = _match(clusters) if clusters else (None, None, None)
            if got is not None:
                chosen = (clusters, got, cost, label)
            else:
                pairs = self._place_clusters_greedy(clusters, homes, rzones, free_cars)
                if not pairs:
                    logger.warning("[%s] %d riders: no free car to take them",
                                   shift_time, len(riders))
                    return free_cars, riders
                chosen = ([cl for cl, _ in pairs], [v for _, v in pairs], 0.0, "greedy")

        clusters, got, cost, label = chosen
        placed = set()
        for cl, v in zip(clusters, got):
            v["_stops"].update(self._cluster_stops(cl, riders, homes))
            v["_remaining"] -= len(cl)
            placed.update(cl)
        unassigned = [riders[i] for i in range(n) if i not in placed]
        if unassigned:
            unassigned = self._spill_riders(unassigned, free_cars)

        cross = sum(1 for cl, v in zip(clusters, got)
                    if Counter(rzones[i] for i in cl).most_common(1)[0][0] != v["zone_name"])
        logger.info("[%s] %d riders -> %d car(s) %s | pool=%s deadhead=%.1f km cross-zone=%d"
                    + (f" | {len(unassigned)} unassigned" if unassigned else ""),
                    shift_time, n, len(clusters),
                    ", ".join(str(len(c)) for c in clusters), label, cost, cross)
        return free_cars, unassigned

    # ── drop-off: Case C / Case D / evening fit ─────────────────────────────

    def _deadhead_to_office(self, v) -> float:
        """Minutes for this car to reach the office from where it currently is."""
        loc = v["current_location"]
        if loc == self.office:
            return 0.0
        return self._pair_minutes(loc, self.office)

    def _can_serve_dropoff(self, v, office_departure) -> Tuple[bool, float]:
        """A car may work a drop-off only if it can physically be at the office
        by the scheduled departure: free when its last trip ends, plus deadhead."""
        free_at = v.get("_free_at")
        if free_at is None:
            return True, 0.0
        dh = self._deadhead_to_office(v)
        return free_at + timedelta(minutes=dh) <= office_departure, dh

    def _fit_route(self, plate_no: str, shift_end_time: str) -> Optional[List[Coord]]:
        """The curve a car is fitted to at this shift end, as [(lat, lng), ...].

        The roster keys a route by the shift it STARTS at, so the route sharing
        this shift's end label is the natural curve. A car with no route at that
        label falls back to its 22:00 route, then its earliest route of the night.
        """
        for key in ((plate_no, shift_end_time), (plate_no, "22:00:00")):
            stops = self._route_by_car_shift.get(key)
            if stops:
                return [(s["pickup_lat"], s["pickup_lng"]) for s in stops]
        keys = [(t, v) for (p, t), v in self._route_by_car_shift.items() if p == plate_no]
        if not keys:
            return None
        return [(s["pickup_lat"], s["pickup_lng"])
                for s in min(keys, key=lambda kv: night_offset(self._raw_time(kv[0])))[1]]

    def _sse_residual(self, home: Coord, curve) -> float:
        """One rider's squared residual: (km to the nearest stop of `curve`)^2."""
        return min(haversine_km(home, s) for s in curve) ** 2

    def _sse_fit_assign(self, requests, cars, shift_end_time, allow_routeless=False):
        """Least-squares rider -> car assignment under each car's free seats.

        Exact (Hungarian), not greedy. Returns (placed, unplaced), where
        `placed` is [(car, [request, ...]), ...] — the shape
        `_assign_dropoff_event` merges by plate.
        """
        curves = {}
        pool = []
        for v in cars:
            curve = self._fit_route(v["plate_no"], shift_end_time)
            if curve is None and not allow_routeless:
                continue
            curves[v["plate_no"]] = curve
            pool.append(v)

        riders = list(requests)
        slots = []
        for v in pool:
            slots.extend([v] * self._free_seats(v))
        if not riders or not slots:
            return [], riders

        n, m = len(riders), len(slots)
        size = max(n, m)
        cost = [[0.0] * size for _ in range(size)]
        for i, d in enumerate(riders):
            home = (d["drop_lat"], d["drop_lng"])
            for j, v in enumerate(slots):
                curve = curves[v["plate_no"]]
                cost[i][j] = _NO_CURVE_COST if curve is None else self._sse_residual(home, curve)
        for j in range(m, size):
            for i in range(n):
                cost[i][j] = _UNSEATABLE_COST

        rows, cols = linear_sum_assignment(cost)
        placed, unplaced = {}, []
        for i, j in zip(rows, cols):
            if i >= n:
                continue                         # a dummy rider = an empty seat
            if j >= m or cost[i][j] >= _UNSEATABLE_COST:
                # A dummy seat is a rider no real car could take, so report them
                # rather than letting the assignment quietly drop them.
                unplaced.append(riders[i])
                continue
            v = slots[j]
            placed.setdefault(v["plate_no"], (v, []))[1].append(riders[i])

        for v, emps in placed.values():
            v["_used"] = v.get("_used", 0) + len(emps)
        return list(placed.values()), unplaced

    def _assign_evening(self, event, reachable) -> Tuple[List, List]:
        """Zone-strict least-squares fit, then a cross-zone spill only if needed."""
        shift_end_time = event["shift_time"]

        def eligible():
            return [v for v in self.fleet.values()
                    if v["plate_no"] in reachable and v["status"] == "AVAILABLE"
                    and self._free_seats(v) > 0]

        by_zone = {}
        for d in event["requests"]:
            by_zone.setdefault(d.get("zone_name"), []).append(d)

        assigned_vehicles, unassigned, spill = [], [], []
        for zone, emps in by_zone.items():
            cars = [v for v in eligible() if zone is not None and v["zone_name"] == zone]
            placed, left = self._sse_fit_assign(emps, cars, shift_end_time)
            assigned_vehicles.extend(placed)
            spill.extend(left)

        if spill:
            placed, left = self._sse_fit_assign(spill, eligible(), shift_end_time,
                                                allow_routeless=True)
            assigned_vehicles.extend(placed)
            unassigned.extend(left)

        return assigned_vehicles, unassigned

    def _assign_capacity(self, v, emps, assigned_vehicles, unassigned, allow, reachable,
                         ref_plate) -> None:
        """Place `emps` on `v`, spilling any overflow onto other eligible cars."""
        room = self._free_seats(v)

        if len(emps) <= room:
            v["_used"] = v.get("_used", 0) + len(emps)
            assigned_vehicles.append((v, emps))
            return

        keep, overflow = emps[:room], emps[room:]
        if keep:
            v["_used"] = v.get("_used", 0) + len(keep)
            assigned_vehicles.append((v, keep))

        extra = [x for x in allow
                 if x["plate_no"] in reachable and x["status"] == "AVAILABLE"
                 and x["plate_no"] != ref_plate and self._free_seats(x) > 0]

        # Fill the roomiest eligible car first, then the next, until the
        # overflow is placed (never all-or-nothing on ONE car).
        extra.sort(key=lambda x: -self._free_seats(x))
        still = list(overflow)
        for x in extra:
            if not still:
                break
            take = still[:self._free_seats(x)]
            del still[:len(take)]
            x["_used"] = x.get("_used", 0) + len(take)
            assigned_vehicles.append((x, take))
        unassigned.extend(still)

    def _point_in_ring(self, pt, ring) -> bool:
        """Ray-casting point-in-polygon over (lon, lat) pairs; any simple ring."""
        x, y = pt[0], pt[1]
        inside = False
        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _in_mirpur_uttara(self, home: Coord) -> bool:
        """True when `home` (lat, lng) lies in Mirpur or Uttara — the Case D
        metro bucket."""
        lat, lng = home
        in_mirpur = (MIRPUR_BBOX[0] <= lat <= MIRPUR_BBOX[2]
                     and MIRPUR_BBOX[1] <= lng <= MIRPUR_BBOX[3])
        return in_mirpur or self._point_in_ring((lng, lat), UTTARA_QUAD)

    def _nearest_catalog_stop(self, home: Coord):
        """(coord, name) of the nearest catalog stop to `home`, by haversine."""
        name, coord = min(self._main_road_stops, key=lambda t: haversine_km(home, t[1]))
        return coord, name

    def _dropoff_stop_for(self, d, is_0730: bool):
        """(coord, name) of the stop one drop-off rider gets off at.

        Case C is door-to-door; Case D (07:30) is main-road only. Shared by the
        stop-building loop in `_assign_dropoff_event` and by the second chance,
        so a re-placed 07:30 rider still gets the Agargaon Metro drop.
        """
        home = (d["drop_lat"], d["drop_lng"])
        if not is_0730:
            return home, f"Home ({self._employee_name(d['employee_email'])})"
        if not self._is_friday_dropoff and self._in_mirpur_uttara(home):
            return self.cfg.agargaon_metro, "Agargaon Metro Station (shared drop point)"
        return self._nearest_catalog_stop(home)

    def _assign_dropoff_event(self, event):
        shift_end_time = event["shift_time"]
        drop_time = event["time"]
        office_departure = self._parse_time(drop_time)
        is_0730 = (drop_time == "07:30:00")

        for v in self.fleet.values():
            v["_stops"] = {}
            v["_used"] = 0

        # Who can be at the office in time, and at what cost.
        reachable = {}
        for v in self.fleet.values():
            ok, dh = self._can_serve_dropoff(v, office_departure)
            if ok:
                reachable[v["plate_no"]] = dh
        allow = list(self.fleet.values())

        rostered = {v["plate_no"] for v in self.fleet.values()
                    if shift_end_time in self.vehicle_shifts.get(v["plate_no"], set())}

        def tier_pool(zone):
            elig = [v for v in self.fleet.values()
                    if v["plate_no"] in reachable and v["status"] == "AVAILABLE"
                    and self._free_seats(v) > 0]
            rostered_elig = [v for v in elig if v["plate_no"] in rostered]
            return [rostered_elig, elig]

        def pick(candidates, zone):
            return min(candidates, key=lambda x: (reachable[x["plate_no"]],
                                                  0 if x["zone_name"] == zone else 1))

        if drop_time in EVENING_FIT_EVENTS:
            assigned_vehicles, unassigned = self._assign_evening(event, reachable)
        else:
            groups: Dict[Optional[str], List[Dict[str, Any]]] = {}
            for d in event["requests"]:
                pref_plate = self.pickup_vehicle_by_employee.get(d["employee_email"])
                groups.setdefault(pref_plate, []).append(d)

            assigned_vehicles = []
            unassigned: List[Dict[str, Any]] = []

            for pref_plate, emps in groups.items():
                zone = emps[0].get("zone_name")
                ref = (emps[0]["drop_lat"], emps[0]["drop_lng"])
                if (pref_plate and pref_plate in self.fleet
                        and self.fleet[pref_plate]["plate_no"] in reachable
                        and self.fleet[pref_plate]["status"] == "AVAILABLE"
                        and self._free_seats(self.fleet[pref_plate]) > 0):
                    v = self.fleet[pref_plate]
                else:
                    candidates = next((t for t in tier_pool(zone) if t), [])
                    if not candidates:
                        unassigned.extend(emps)
                        continue
                    v = pick(candidates, zone)
                self._assign_capacity(v, emps, assigned_vehicles, unassigned, allow,
                                      reachable, v["plate_no"])

        # One vehicle can legitimately receive more than one group (reuse +
        # borrow + overflow). Merge per plate BEFORE building stops.
        merged: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]] = {}
        for v, emps in assigned_vehicles:
            merged.setdefault(v["plate_no"], (v, []))[1].extend(emps)

        for v, emps in merged.values():
            v["_stops"] = {}
            for d in emps:
                coord, name = self._dropoff_stop_for(d, is_0730)
                v["_stops"].setdefault(coord, {
                    "coord": coord, "name": name,
                    "is_shared": is_0730, "is_adhoc": not is_0730, "passengers": [],
                })["passengers"].append(d)
        return [v for v, _ in merged.values()], unassigned

    def _order_stops_dropoff(self, vehicle) -> List[Tuple[Any, Dict[str, Any]]]:
        """Exact shortest OPEN path: OFFICE -> every stop, ending at the last home.

        The tour is open because the car does not come back. The closing leg is
        priced at `cfg.dropoff_return_weight` (0.0 by default): a tie-break for
        where the night ends, not a cost.
        """
        items = list(vehicle["_stops"].items())
        if len(items) <= 1:
            return items
        coords = [self.office] + [it["coord"] for _, it in items] + [self.office]
        START, END = 0, len(items) + 1
        durations, _ = self.provider.table(coords)
        stop_idx = list(range(1, len(items) + 1))
        weights = [1.0] * len(stop_idx) + [self.cfg.dropoff_return_weight]
        ordered = self._held_karp_order(durations, stop_idx, START, END, weights)
        return [items[i - 1] for i in ordered]

    def _compute_timing_dropoff(self, vehicle, ordered_stops, office_departure) -> Dict[str, Any]:
        """Forward timing of car.current_location -> OFFICE -> stops, leaving the
        office at `office_departure` (the drop_time, not the shift end — employees
        wait 15/30 min for the car).

        The tour ENDS AT THE LAST STOP. The 120-min cap measures the PASSENGER
        journey (office -> last stop); the deadhead in from the car's previous
        position is the car's own repositioning and is reported separately.
        """
        coords = [vehicle["current_location"], self.office] + [it["coord"] for _, it in ordered_stops]
        durations, distances = self.provider.table(coords)
        legs = [durations[i][i + 1] for i in range(len(coords) - 1)]
        deadhead = legs[0]
        timestamps = []
        t = office_departure
        for i, (key, _item) in enumerate(ordered_stops):
            t = t + timedelta(minutes=legs[i + 1])
            arrival = t
            t = t + timedelta(minutes=self.cfg.boarding_buffer_min)
            timestamps.append({"stop_key": key, "arrival": arrival, "departure": t})
        tour_end = t
        return_deadhead = (self._pair_minutes(ordered_stops[-1][1]["coord"], self.office)
                           if ordered_stops else 0.0)
        passenger_total = (tour_end - office_departure).total_seconds() / 60.0
        return {
            "office_departure": office_departure,
            "tour_end": tour_end,
            "trip_start": office_departure - timedelta(minutes=deadhead),
            "deadhead_minutes": deadhead,
            "return_deadhead_minutes": return_deadhead,
            "total_minutes": passenger_total + deadhead,
            "passenger_total_minutes": passenger_total,
            "end_location": ordered_stops[-1][1]["coord"] if ordered_stops else self.office,
            "leg_minutes": legs[1:],
            "leg_km": [distances[i][i + 1] for i in range(1, len(coords) - 1)],
            "stop_timestamps": timestamps,
        }

    def _enforce_cap_dropoff(self, vehicle, office_departure):
        """A drop-off runs FORWARD from a fixed office departure, so only the
        120-min passenger cap (office -> last stop) can bite here."""
        dropped: List[Dict[str, Any]] = []
        reason = "dropped_for_120min_cap"
        while True:
            ordered = self._order_stops_dropoff(vehicle)
            if not ordered:
                return ordered, None, dropped, reason
            timing = self._compute_timing_dropoff(vehicle, ordered, office_departure)
            if timing["passenger_total_minutes"] <= self.cfg.max_route_minutes:
                return ordered, timing, dropped, reason
            best_key, best_total = None, None
            for key, _ in ordered:
                saved = vehicle["_stops"]
                vehicle["_stops"] = {k: v for k, v in saved.items() if k != key}
                trial = self._order_stops_dropoff(vehicle)
                trial_total = (self._compute_timing_dropoff(vehicle, trial, office_departure)
                               ["passenger_total_minutes"] if trial else 0)
                vehicle["_stops"] = saved
                if best_total is None or trial_total < best_total:
                    best_total, best_key = trial_total, key
            dropped.extend(vehicle["_stops"].pop(best_key)["passengers"])

    # ── second chance ────────────────────────────────────────────────────────

    @staticmethod
    def _seats_left(v: Dict[str, Any]) -> int:
        """Seats left on `v` for THIS event, counted off the stops it has."""
        return v["capacity"] - sum(len(it["passengers"]) for it in v["_stops"].values())

    def _dropoff_fits(self, v, stops, office_departure) -> bool:
        """Would this car's drop-off, with exactly these stops, clear the cap?"""
        saved = v["_stops"]
        v["_stops"] = stops
        try:
            ordered = self._order_stops_dropoff(v)
            if not ordered:
                return True
            return (self._compute_timing_dropoff(v, ordered, office_departure)
                    ["passenger_total_minutes"] <= self.cfg.max_route_minutes)
        finally:
            v["_stops"] = saved

    def _second_chance_stop(self, d, home, v, is_pickup, shift_time, is_0730, route_by_car):
        """(stop_key, stop_item) for putting rider `d` on car `v`, or None.

        The stop rule of the shift, never a new one, so a re-placed rider lands
        where the roster would have sent them in the first place.
        """
        if is_pickup:
            if shift_time == "22:00:00":
                return self._place_on(v, d, home, route_by_car)
            return f"home_{d['employee_email']}", {
                "coord": home,
                "name": f"Home ({self._employee_name(d['employee_email'])})",
                "is_adhoc": True, "passengers": [d]}
        coord, name = self._dropoff_stop_for(d, is_0730)
        return coord, {"coord": coord, "name": name,
                       "is_shared": is_0730, "is_adhoc": not is_0730,
                       "passengers": [d]}

    def _second_chance(self, event, shift_time, left):
        """Offer every rider the first pass left behind one more car.

        `left` is those riders, in the order they are to be considered. Returns
        (still_left, touched): the riders no car could take, and the plates
        whose stops changed, so the caller knows which routes to time again.
        """
        if not left:
            return [], set()
        is_pickup = event["type"] == "pickup"
        drop_time = event["time"]
        is_0730 = (not is_pickup) and drop_time == "07:30:00"
        office_departure = None if is_pickup else self._parse_time(drop_time)

        # Case A confines a rider to their own zone (rule 5); every other shift
        # treats the zone as a tie-break.
        zone_gate = is_pickup and shift_time == "22:00:00"

        route_by_car = {}
        if zone_gate:
            for s in self._stops_for_shift(shift_time, list(self.fleet.values())):
                route_by_car.setdefault(s["vehicle_plate"], []).append(s)
            for stops in route_by_car.values():
                stops.sort(key=lambda s: (s["sequence_order"] is None, s["sequence_order"]))

        rostered = {v["plate_no"] for v in self.fleet.values()
                    if shift_time in self.vehicle_shifts.get(v["plate_no"], set())}

        def candidates(home, zone):
            out = []
            for v in self.fleet.values():
                if v["status"] != "AVAILABLE" or self._seats_left(v) <= 0:
                    continue
                if zone_gate and v["zone_name"] != zone:
                    continue
                if is_pickup:
                    rank = haversine_km(home, v["current_location"])
                else:
                    ok, dh = self._can_serve_dropoff(v, office_departure)
                    if not ok:
                        continue
                    rank = dh
                out.append((0 if v["plate_no"] in rostered else 1, rank,
                            0 if v["zone_name"] == zone else 1, v))
            out.sort(key=lambda t: t[:3])
            return [t[3] for t in out]

        still, touched = [], set()
        for d in left:
            home = ((d["pickup_lat"], d["pickup_lng"]) if is_pickup
                    else (d["drop_lat"], d["drop_lng"]))
            zone = self._request_zone(d) if is_pickup else d.get("zone_name")
            for v in candidates(home, zone):
                got = self._second_chance_stop(d, home, v, is_pickup, shift_time,
                                               is_0730, route_by_car)
                if got is None:
                    continue
                key, item = got
                stops = dict(v["_stops"])
                if key in stops:
                    stops[key] = dict(stops[key],
                                      passengers=list(stops[key]["passengers"]) + [d])
                else:
                    stops[key] = item
                fits = (self._route_fits(v, stops, shift_time) if is_pickup
                        else self._dropoff_fits(v, stops, office_departure))
                if fits:
                    v["_stops"] = stops
                    touched.add(v["plate_no"])
                    break
            else:
                still.append(d)
        return still, touched

    # ── main loop ────────────────────────────────────────────────────────────

    def solve(self) -> SolvedNight:
        self._report_missing_coordinates()
        self._build_timeline()

        for event in self.events:
            if event["type"] == "pickup":
                self._run_pickup_event(event)
            else:
                self._run_dropoff_event(event)

        logger.info(
            "routing: solved %s -> %s", self.service_date, self.out.counts()
        )
        return self.out

    @staticmethod
    def _modal_zone(ordered_stops) -> Optional[str]:
        """The zone most of this route's passengers belong to → `route.zone_id`."""
        zones = Counter(
            pr.get("zone_name")
            for _k, item in ordered_stops
            for pr in item["passengers"]
            if pr.get("zone_name")
        )
        return zones.most_common(1)[0][0] if zones else None

    def _run_pickup_event(self, event) -> None:
        shift_time = event["shift_time"]
        # ML model prediction is time-of-day dependent: anchor every leg in
        # this event to the requests' own pickup/shift-start time (event["time"]).
        self.provider.query_time = self._parse_time(event["time"])
        # latest instant the trip could start (it ends at the office deadline)
        self._update_fleet(self._parse_time(shift_time) - timedelta(minutes=self.cfg.office_buffer_min))
        vehicles, unassigned = self._assign_pickup_event(event)

        # Every rider this event has left behind so far, with the reason they
        # were left. NOT reported yet: the second chance below gets them all
        # first, and only the ones it cannot place are reported.
        left = [(pr, "no_vehicle_available", None) for pr in unassigned]

        if shift_time == "22:00:00":
            left += [(pr, "dropped_for_120min_cap", None)
                     for pr in self._redistribute_case_a(vehicles, shift_time)]

        # The cap and the free window, applied BEFORE a single route is written
        # — so the riders they shed are still free agents when the second
        # chance runs. Keyed by plate.
        enforced = {}
        for v in vehicles:
            if not v["_stops"]:
                continue
            result = self._enforce_cap_pickup(v, shift_time)
            enforced[v["plate_no"]] = result
            left += [(pr, result[3], v["plate_no"]) for pr in result[2]]

        still, touched = self._second_chance(event, shift_time,
                                             [pr for pr, _, _ in left])
        reason_of = {pr["employee_email"]: (r, pl) for pr, r, pl in left}
        for pr in still:
            reason, plate = reason_of[pr["employee_email"]]
            self.out.unassigned.append(
                self._unassigned_row(pr, "pickup", shift_time, reason, plate)
            )

        # Every car carrying someone now — the first pass's, plus any car the
        # second chance filled from empty. `recording` comes from `touched`, not
        # from "every car with stops": a car with stops is not necessarily a car
        # working THIS event.
        seen = {v["plate_no"] for v in vehicles}
        recording = [v for v in vehicles if v["_stops"]]
        recording += [self.fleet[p] for p in sorted(touched) if p not in seen]

        for v in recording:
            if v["plate_no"] in touched or v["plate_no"] not in enforced:
                ordered, timing, dropped, drop_reason = self._enforce_cap_pickup(v, shift_time)
                for pr in dropped:
                    self.out.unassigned.append(
                        self._unassigned_row(pr, "pickup", shift_time, drop_reason, v["plate_no"])
                    )
            else:
                ordered, timing = enforced[v["plate_no"]][:2]

            if not ordered or timing is None:
                continue

            # Record pickup->vehicle reuse only for passengers who survived the
            # 120-min cap, so drop-off never reuses a car that never carried them.
            for _key, item in ordered:
                for pr in item["passengers"]:
                    self.pickup_vehicle_by_employee[pr["employee_email"]] = v["plate_no"]

            full = [v["current_location"]] + [it["coord"] for _, it in ordered] + [self.office]
            dist_km, _dur_min, geometry = self.provider.route(full)
            rid = f"P{shift_time}::V{v['plate_no']}"
            self.out.routes.append({
                "route_instance_id": rid,
                "type": "pickup",
                "shift_time": shift_time,
                "service_date": self.service_date,
                "zone_name": self._modal_zone(ordered),
                "vehicle_id": v["plate_no"],
                "plate_no": v["plate_no"],
                "driver_id": v.get("driver_email"),
                "capacity": v["capacity"],
                "assigned_passengers": sum(len(it["passengers"]) for _, it in ordered),
                "stop_count": len(ordered),
                "parking_lat": v["parking_lat"],
                "parking_lng": v["parking_lng"],
                "start_lat": v["current_location"][0],
                "start_lng": v["current_location"][1],
                "parking_departure": iso(timing["parking_departure"]),
                "office_arrival": iso(timing["office_arrival"]),
                "total_minutes": round(timing["total_minutes"], 1),
                "passenger_total_minutes": round(timing["passenger_total_minutes"], 1),
                "total_distance_km": round(dist_km, 2),
                "route_geometry": geometry,
            })

            for seq, ((key, item), ts) in enumerate(zip(ordered, timing["stop_timestamps"]), start=1):
                self.out.stops.append({
                    "route_instance_id": rid,
                    "type": "pickup",
                    "shift_time": shift_time,
                    "vehicle_id": v["plate_no"],
                    "sequence_order": seq,
                    "stop_name": item["name"],
                    "stop_lat": item["coord"][0],
                    "stop_lng": item["coord"][1],
                    "is_adhoc": item["is_adhoc"],
                    "is_shared": item.get("is_shared", False),
                    "arrival_time": iso(ts["arrival"]),
                    "departure_time": iso(ts["departure"]),
                    "leg_minutes_from_previous": round(timing["leg_minutes"][seq - 1], 1),
                    "leg_km_from_previous": round(timing["leg_km"][seq - 1], 2),
                    "passenger_count": len(item["passengers"]),
                })
                for pr in item["passengers"]:
                    self.out.passengers.append({
                        "route_instance_id": rid,
                        "type": "pickup",
                        "sequence_order": seq,
                        "stop_name": item["name"],
                        "employee_id": pr["employee_email"],
                        "employee_name": self._employee_name(pr["employee_email"]),
                        "board_time": iso(ts["departure"]),
                    })

            # fleet state: vehicle now IN_TRIP, ends at OFFICE
            v["status"] = "IN_TRIP"
            v["_trip_end_time"] = timing["office_arrival"]
            v["_trip_end_location"] = self.office
            v["_free_at"] = timing["office_arrival"]
            v["_used"] = 0

    def _run_dropoff_event(self, event) -> None:
        shift_end_time = event["shift_time"]
        drop_time = event["time"]
        # ML model prediction is time-of-day dependent: anchor every leg in
        # this event to the requests' own scheduled drop-off time (event["time"]).
        self.provider.query_time = self._parse_time(drop_time)
        # the car leaves the office at drop_time, not at shift end -- the
        # 15/30-min gap is the employees' wait for the car
        office_departure = self._parse_time(drop_time)
        self._update_fleet(office_departure)
        vehicles, unassigned = self._assign_dropoff_event(event)

        left = [(d, "no_vehicle_available", None) for d in unassigned]
        enforced = {}
        for v in vehicles:
            if not v["_stops"]:
                continue
            result = self._enforce_cap_dropoff(v, office_departure)
            enforced[v["plate_no"]] = result
            left += [(d, result[3], v["plate_no"]) for d in result[2]]

        still, touched = self._second_chance(event, shift_end_time,
                                             [d for d, _, _ in left])
        reason_of = {d["employee_email"]: (r, pl) for d, r, pl in left}
        for d in still:
            reason, plate = reason_of[d["employee_email"]]
            self.out.unassigned.append(
                self._unassigned_row(d, "dropoff", shift_end_time, reason, plate)
            )

        seen = {v["plate_no"] for v in vehicles}
        recording = [v for v in vehicles if v["_stops"]]
        recording += [self.fleet[p] for p in sorted(touched) if p not in seen]

        for v in recording:
            start_loc = v["current_location"]      # where the deadhead to the office begins
            if v["plate_no"] in touched or v["plate_no"] not in enforced:
                ordered, timing, dropped, drop_reason = self._enforce_cap_dropoff(v, office_departure)
                for d in dropped:
                    self.out.unassigned.append(
                        self._unassigned_row(d, "dropoff", shift_end_time, drop_reason, v["plate_no"])
                    )
            else:
                ordered, timing = enforced[v["plate_no"]][:2]

            if not ordered or timing is None:
                continue

            full = [start_loc, self.office] + [it["coord"] for _, it in ordered]
            dist_km, _dur_min, geometry = self.provider.route(full)
            rid = f"D{shift_end_time}::V{v['plate_no']}"
            self.out.routes.append({
                "route_instance_id": rid,
                "type": "dropoff",
                "shift_time": shift_end_time,
                "service_date": self.service_date,
                "zone_name": self._modal_zone(ordered),
                "vehicle_id": v["plate_no"],
                "plate_no": v["plate_no"],
                "driver_id": v.get("driver_email"),
                "capacity": v["capacity"],
                "assigned_passengers": sum(len(it["passengers"]) for _, it in ordered),
                "stop_count": len(ordered),
                "start_lat": start_loc[0],
                "start_lng": start_loc[1],
                "end_lat": timing["end_location"][0],
                "end_lng": timing["end_location"][1],
                "office_departure": iso(timing["office_departure"]),
                "parking_arrival": iso(timing["tour_end"]),
                "trip_start": iso(timing["trip_start"]),
                "tour_end": iso(timing["tour_end"]),
                "deadhead_minutes": round(timing["deadhead_minutes"], 1),
                "return_deadhead_minutes": round(timing["return_deadhead_minutes"], 1),
                "total_minutes": round(timing["total_minutes"], 1),
                "passenger_total_minutes": round(timing["passenger_total_minutes"], 1),
                "total_distance_km": round(dist_km, 2),
                "route_geometry": geometry,
            })

            for seq, ((key, item), ts) in enumerate(zip(ordered, timing["stop_timestamps"]), start=1):
                self.out.stops.append({
                    "route_instance_id": rid,
                    "type": "dropoff",
                    "shift_time": shift_end_time,
                    "vehicle_id": v["plate_no"],
                    "sequence_order": seq,
                    "stop_name": item["name"],
                    "stop_lat": item["coord"][0],
                    "stop_lng": item["coord"][1],
                    "is_adhoc": item["is_adhoc"],
                    "is_shared": item["is_shared"],
                    "arrival_time": iso(ts["arrival"]),
                    "departure_time": iso(ts["departure"]),
                    "leg_minutes_from_previous": round(timing["leg_minutes"][seq - 1], 1),
                    "leg_km_from_previous": round(timing["leg_km"][seq - 1], 2),
                    "passenger_count": len(item["passengers"]),
                })
                for d in item["passengers"]:
                    self.out.passengers.append({
                        "route_instance_id": rid,
                        "type": "dropoff",
                        "sequence_order": seq,
                        "stop_name": item["name"],
                        "employee_id": d["employee_email"],
                        "employee_name": self._employee_name(d["employee_email"]),
                        "alight_time": iso(ts["arrival"]),
                    })

            # fleet state: the tour ENDS AT THE LAST STOP. The car is left out
            # on the road; its next pick-up starts from there.
            v["status"] = "IN_TRIP"
            v["current_location"] = timing["end_location"]
            v["_trip_end_time"] = timing["tour_end"]
            v["_trip_end_location"] = timing["end_location"]
            v["_free_at"] = timing["tour_end"]
            v["_used"] = 0


def solve_night(
    *,
    service_date: str,
    vehicles: Sequence[Dict[str, Any]],
    pickup_requests: Sequence[Dict[str, Any]],
    dropoff_requests: Sequence[Dict[str, Any]],
    fixed_stops: Sequence[Dict[str, Any]],
    provider: DistanceProvider,
    foot: Optional[FootDistanceProvider] = None,
    cfg: Optional[SolverConfig] = None,
    employee_names: Optional[Dict[str, str]] = None,
    use_ml: bool = True,
) -> SolvedNight:
    """Solve one whole service night.

    Whole-night is not a convenience: fleet state (where each car is, when it
    is next free) carries across every event, so pickups and drop-offs cannot
    be solved independently without leaving the fleet's end-of-night position
    undefined.

    Both the weekly pass and the nightly ad-hoc re-route go through here: the
    ad-hoc pass is the same whole-night solve re-run after 7 PM with the day's
    ad-hoc rows folded in (newest-wins per employee), so a change to this
    function upgrades both entry points.
    """
    return NightSolver(
        service_date=service_date,
        vehicles=vehicles,
        pickup_requests=pickup_requests,
        dropoff_requests=dropoff_requests,
        fixed_stops=fixed_stops,
        provider=provider,
        foot=foot,
        cfg=cfg,
        employee_names=employee_names,
        use_ml=use_ml,
    ).solve()


# ──────────────────────────────────────────────────────────────────────────────
# Driver: fetch one service date and solve it
# ──────────────────────────────────────────────────────────────────────────────

def _read_env_file(path: Path, url: Optional[str], key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not path.exists():
        return url, key
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == "SUPABASE_URL" and not url:
                url = v.strip()
            elif k.strip() == "SUPABASE_KEY" and not key:
                key = v.strip()
    return url, key


def _db_client() -> "Client":
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    # Two known layouts: this standalone repo's own .env (colocated with the
    # script), or the original monorepo's backend/.env one level up. Try the
    # colocated one first.
    if not url or not key:
        url, key = _read_env_file(Path(__file__).resolve().parent / ".env", url, key)
    if not url or not key:
        url, key = _read_env_file(
            Path(__file__).resolve().parent.parent
            / "Data-Driven-Employee-Routing-System" / "backend" / ".env",
            url, key)
    if not url or not key:
        raise SystemExit("SUPABASE_URL / SUPABASE_KEY not found (env, .env, or backend/.env)")
    return create_client(url, key)


def _load_offline(path: str):
    d = json.loads(Path(path).read_text())
    names = {u["email"]: u.get("name") for u in d.get("users", [])}
    return {
        "vehicles": d["vehicles"],
        "pickup_requests": d["pickup_requests"],
        "dropoff_requests": d["dropoff_requests"],
        "fixed_stops": d["vehicle_pickup_locations"],
        "employee_names": names,
    }, []


def _coordinate_check(pickup_requests, dropoff_requests) -> None:
    """How many loaded requests actually have coordinates to route on.

    A request missing pickup_lat/lng (or drop_lat/lng) never reaches the
    timeline at all -- `NightSolver._report_missing_coordinates` reports it as
    `no_coordinates` before the solve proper starts (see `_build_timeline`).
    This is the same check, run up front so it doesn't have to be inferred
    from the unassigned reasons.
    """
    pu_missing = [r["employee_email"] for r in pickup_requests
                  if r.get("pickup_lat") is None or r.get("pickup_lng") is None]
    do_missing = [d["employee_email"] for d in dropoff_requests
                  if d.get("drop_lat") is None or d.get("drop_lng") is None]
    print(f"Coordinates: {len(pickup_requests) - len(pu_missing)}/{len(pickup_requests)} "
          f"pickup requests valid, {len(dropoff_requests) - len(do_missing)}/{len(dropoff_requests)} "
          f"dropoff requests valid")
    if pu_missing:
        print(f"  {len(pu_missing)} pickup request(s) missing coordinates: {sorted(pu_missing)[:8]}"
              + (" ..." if len(pu_missing) > 8 else ""))
    if do_missing:
        print(f"  {len(do_missing)} dropoff request(s) missing coordinates: {sorted(do_missing)[:8]}"
              + (" ..." if len(do_missing) > 8 else ""))


def _summary(label, solved, engine, warnings):
    print(f"\n=== {label} ===")
    print(f"engine: {engine}")
    print(f"counts: {solved.counts()}")
    by = Counter((u["type"], u["shift_time"], u["reason"]) for u in solved.unassigned)
    if by:
        print("unassigned by (type, shift_time, reason):")
        for (t, st, r), n in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] or "", kv[0][2])):
            print(f"    {t:<8} {st or '--:--':<9} {r:<26} {n}")
    else:
        print("unassigned: none")
    for w in list(warnings) + list(solved.warnings):
        print("  warn:", w)


def _route_window(r: Dict[str, Any]) -> Tuple[datetime, datetime]:
    """(start, end) datetimes for one route -- the window it occupies its vehicle.

    Pickup and dropoff routes carry differently-named timestamps (see
    `_run_pickup_event` / `_run_dropoff_event`), so this is the one place that
    knows which pair to read.
    """
    if r["type"] == "pickup":
        return datetime.fromisoformat(r["parking_departure"]), datetime.fromisoformat(r["office_arrival"])
    return datetime.fromisoformat(r["trip_start"]), datetime.fromisoformat(r["tour_end"])


def _print_fleet_diagnostics(
    solved: "SolvedNight",
    pickup_requests: Sequence[Dict[str, Any]],
    dropoff_requests: Sequence[Dict[str, Any]],
) -> None:
    """The exact diagnostics `routing_night.py` prints after solving (its
    "=== Summary ===" cell and "6b. Duty blocks" cell) -- same text, same
    order, nothing added. `run_ml_solver_db.py` never printed any of this.
    """
    routes, stops, passengers, unassigned = (
        solved.routes, solved.stops, solved.passengers, solved.unassigned)

    print(f"\n=== Summary ===")
    print(f"Routes created: {len(routes)} "
          f"({sum(1 for r in routes if r['type']=='pickup')} pickup, "
          f"{sum(1 for r in routes if r['type']=='dropoff')} drop-off)")
    print(f"Stops served: {len(stops)}")
    print(f"Passengers assigned: {len(passengers)}")
    print(f"Unassigned: {len(unassigned)}")

    # --- accounting check: every request must be routed or reported, exactly once ---
    routed = {("pickup", p["employee_id"]) for p in passengers if p["type"] == "pickup"}
    routed |= {("dropoff", p["employee_id"]) for p in passengers if p["type"] == "dropoff"}
    reported = {(u["type"], u["employee_email"]) for u in unassigned}
    expected = {("pickup", pr["employee_email"]) for pr in pickup_requests} | \
               {("dropoff", d["employee_email"]) for d in dropoff_requests}
    missing = expected - routed - reported
    print(f"Accounting: {len(expected)} requests -> {len(routed)} routed, "
          f"{len(reported)} reported, {len(missing)} unaccounted for")
    if missing:
        print(f"  !! {len(missing)} request(s) silently lost: {sorted(missing)[:5]}")

    # --- fleet-state check: no vehicle may be on two trips at once (spec sec.3) ---
    windows: Dict[str, List[Tuple[datetime, datetime, str]]] = {}
    for r in routes:
        a, b = _route_window(r)
        windows.setdefault(r["plate_no"], []).append((a, b, r["route_instance_id"]))
    overlaps = []
    for plate, xs in windows.items():
        xs.sort()
        for i in range(len(xs) - 1):
            if xs[i + 1][0] < xs[i][1]:
                overlaps.append((plate, xs[i][2], xs[i + 1][2]))
    print(f"Fleet state: {len(overlaps)} vehicle(s) double-booked on overlapping trips")
    for o in overlaps[:5]:
        print(f"  !! {o[0]}: {o[1]} overlaps {o[2]}")

    if unassigned:
        print("\nUnassigned by reason:")
        for reason, n in Counter(u["reason"] for u in unassigned).most_common():
            print(f"  {reason:<28} {n}")

    # ---------------------------------------------------------------
    # 6b. Duty blocks -- what the cascade actually produced
    # ---------------------------------------------------------------
    blocks: Dict[str, List[Dict[str, Any]]] = {}
    for r in routes:
        blocks.setdefault(r["plate_no"], []).append(r)
    for rl in blocks.values():
        rl.sort(key=lambda r: _route_window(r)[0])

    print("\n=== Duty blocks (one line per car: its trips in order) ===")
    shapes: Counter = Counter()
    for rl in blocks.values():
        label = tuple(("P" if r["type"] == "pickup" else "D") + r["shift_time"][:5] for r in rl)
        shapes[label] += 1
    for shape, n in shapes.most_common():
        print(f"  {n:3}x  {' -> '.join(shape)}")

    drive = sum((_route_window(r)[1] - _route_window(r)[0]).total_seconds() / 60.0 for r in routes)
    idle = 0.0
    handoffs: Counter = Counter()
    for rl in blocks.values():
        for a, b in zip(rl, rl[1:]):
            idle += (_route_window(b)[0] - _route_window(a)[1]).total_seconds() / 60.0
            handoffs[(a["type"], b["type"])] += 1
    print(f"\nFleet driving minutes: {drive:.0f}")
    if idle > 0:
        print(f"Fleet idle minutes between trips: {idle:.0f}   (drive:idle = {drive / idle:.2f}:1)")
    print(f"Trip-type transitions: {dict(handoffs)}")

    # --- Fairness: how long is each passenger actually IN the car? ---
    for kind in ("pickup", "dropoff"):
        starts = {r["route_instance_id"]: r["parking_departure"] if kind == "pickup" else r["office_departure"]
                  for r in routes if r["type"] == kind}
        ends = {r["route_instance_id"]: r["office_arrival"] if kind == "pickup" else r["tour_end"]
                for r in routes if r["type"] == kind}
        rides, waits = [], []
        for p in passengers:
            if p["type"] != kind or p["route_instance_id"] not in starts:
                continue
            t = datetime.fromisoformat(p["board_time"] if kind == "pickup" else p["alight_time"])
            rides.append((datetime.fromisoformat(ends[p["route_instance_id"]]) - t).total_seconds() / 60.0)
            waits.append((t - datetime.fromisoformat(starts[p["route_instance_id"]])).total_seconds() / 60.0)
        if not rides:
            continue
        rides.sort()
        print(f"  {kind:<8} ride  mean={statistics.mean(rides):5.1f}  p50={rides[len(rides)//2]:5.1f}"
              f"  p90={rides[int(len(rides)*0.9)]:5.1f}  MAX={rides[-1]:5.1f} min"
              f"  | total {sum(rides):.0f} pax-min")
        print(f"  {kind:<8} wait  mean={statistics.mean(waits):5.1f}"
              f"  (wait + ride = the route's duration, so this is the same trade)")

    dh_in = sum(r.get("deadhead_minutes", 0.0) for r in routes if r["type"] == "dropoff")
    dh_back = sum(r.get("return_deadhead_minutes", 0.0) for r in routes if r["type"] == "dropoff")
    print(f"Drop-off deadheads IN to the office:  {dh_in:.0f} min (driven)")
    print(f"Drop-off return legs to the office:   {dh_back:.0f} min (NOT driven unless the car is reused)")
    ends_at_office = sum(1 for r in routes
                         if r["type"] == "dropoff" and (r["end_lat"], r["end_lng"]) == OFFICE)
    print(f"Drop-off tours still finishing at the office: {ends_at_office}")
    stranded = [plate for plate, rl in blocks.items()
               if rl and rl[-1]["type"] == "dropoff" and _route_window(rl[-1])[1].hour < 6]
    print(f"Cars whose night ends at a rider's stop before 06:00: {len(stranded)}")


def _time_only(iso_timestamp: Optional[str]) -> Optional[str]:
    """"2026-09-20T22:16:41" -> "22:16:41" (route_stop times are bare TIME)."""
    if not iso_timestamp:
        return None
    text = str(iso_timestamp)
    return text.split("T", 1)[1][:8] if "T" in text else text[:8]


def _sql_lit(value: Any) -> str:
    """A Postgres literal. Strings are single-quoted and escaped."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _dump_payload(service_date: str, solved, ctx) -> Dict[str, Any]:
    """Everything needed to both re-read the solve and rebuild the SQL."""
    context = None
    if ctx is not None:
        context = {
            "employee_id_by_email": ctx.employee_id_by_email,
            "vehicle_id_by_plate": ctx.vehicle_id_by_plate,
            "driver_id_by_plate": ctx.driver_id_by_plate,
            "zone_id_by_name": ctx.zone_id_by_name,
            "pickup_id_by_email": ctx.pickup_id_by_email,
            "dropoff_id_by_email": ctx.dropoff_id_by_email,
            "warnings": list(ctx.warnings),
        }
    return {
        "service_date": service_date,
        "context": context,
        "routes": solved.routes,
        "stops": solved.stops,
        "passengers": solved.passengers,
        "unassigned": solved.unassigned,
        "warnings": list(solved.warnings),
    }


def _resolve_dates(args) -> List[str]:
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end or args.start, "%Y-%m-%d").date()
        if end < start:
            raise SystemExit("--end is before --start")
        out, d = [], start
        while d <= end:
            out.append(d.isoformat())
            d += timedelta(days=1)
        return out
    return [args.date]


def _build_week_sql(dumps: List[Dict[str, Any]], out_path: Path) -> None:
    """Emit one transactional SQL file that REPLACES every listed date's routes.

    Mirrors `RoutingWriter.persist`: unlink requests, delete the children then
    the routes, insert routes/stops/passengers/assignments, relink requests.
    Rows are correlated by (service_date, route_code) via a temp table -- never
    by insertion order -- because `route.route_id` is server-generated.
    """
    dates = [d["service_date"] for d in dumps]
    date_list = ", ".join(_sql_lit(x) for x in dates)
    L: List[str] = []
    L.append("-- Auto-generated by data/run_ml_solver_db.py")
    L.append(f"-- service dates: {', '.join(dates)}")
    L.append("-- Replaces the existing routes for those dates with a fresh solve.")
    L.append("BEGIN;")
    L.append("")
    L.append("CREATE TEMP TABLE _route_map (")
    L.append("  service_date date NOT NULL,")
    L.append("  route_code   text NOT NULL,")
    L.append("  route_id     bigint NOT NULL,")
    L.append("  PRIMARY KEY (service_date, route_code)")
    L.append(") ON COMMIT DROP;")
    L.append("")
    L.append("-- 1) unlink requests (Rejected rows are left untouched)")
    L.append("UPDATE public.pickup_request SET route_id = NULL, status = 'Pending'")
    L.append(f"  WHERE service_date IN ({date_list}) AND status IS DISTINCT FROM 'Rejected';")
    L.append("UPDATE public.dropoff_request SET route_id = NULL, status = 'Pending'")
    L.append(f"  WHERE service_date IN ({date_list}) AND status IS DISTINCT FROM 'Rejected';")
    L.append("")
    L.append("-- 2) delete children then routes (the schema has no ON DELETE CASCADE)")
    L.append("DELETE FROM public.stop_passenger WHERE stop_id IN (")
    L.append("  SELECT stop_id FROM public.route_stop WHERE route_id IN (")
    L.append(f"    SELECT route_id FROM public.route WHERE service_date IN ({date_list})));")
    L.append("DELETE FROM public.route_stop WHERE route_id IN (")
    L.append(f"  SELECT route_id FROM public.route WHERE service_date IN ({date_list}));")
    L.append("DELETE FROM public.route_assignment WHERE route_id IN (")
    L.append(f"  SELECT route_id FROM public.route WHERE service_date IN ({date_list}));")
    L.append(f"DELETE FROM public.route WHERE service_date IN ({date_list});")
    L.append("")

    # 3) routes
    route_rows = []
    for d in dumps:
        zones = (d.get("context") or {}).get("zone_id_by_name", {})
        for r in d["routes"]:
            geom = (_sql_lit(json.dumps(r["route_geometry"])) + "::jsonb"
                    if r.get("route_geometry") is not None else "NULL")
            route_rows.append("(" + ", ".join([
                _sql_lit(r["route_instance_id"]),
                _sql_lit(zones.get(r.get("zone_name") or "")),
                _sql_lit(r["type"]),
                _sql_lit(d["service_date"]),
                _sql_lit(r["shift_time"]),
                _sql_lit(r["total_distance_km"]),
                _sql_lit(int(round(r["total_minutes"]))),
                geom,
            ]) + ")")
    L.append("-- 3) insert routes")
    if route_rows:
        L.append("INSERT INTO public.route (route_code, zone_id, route_type, service_date, "
                 "shift_time, total_distance_km, total_travel_time_min, route_geometry) VALUES")
        L.append(",\n".join(route_rows) + ";")
    else:
        L.append("-- (no routes in the dumps)")
    L.append("")
    L.append("INSERT INTO _route_map (service_date, route_code, route_id)")
    L.append("SELECT service_date, route_code, route_id FROM public.route")
    L.append(f"  WHERE service_date IN ({date_list}) AND route_code IS NOT NULL;")
    L.append("")

    # 4) stops
    stop_rows = []
    for d in dumps:
        for s in d["stops"]:
            stop_rows.append("(" + ", ".join([
                _sql_lit(d["service_date"]),
                _sql_lit(s["route_instance_id"]),
                _sql_lit(int(s["sequence_order"])),
                _sql_lit(s.get("stop_lat")),
                _sql_lit(s.get("stop_lng")),
                _sql_lit(_time_only(s.get("arrival_time"))),
                _sql_lit(_time_only(s.get("departure_time"))),
                _sql_lit(s.get("stop_name")),
                _sql_lit(bool(s.get("is_adhoc"))),
                _sql_lit(bool(s.get("is_shared"))),
            ]) + ")")
    L.append("-- 4) insert stops")
    if stop_rows:
        L.append("INSERT INTO public.route_stop (route_id, latitude, longitude, sequence_order, "
                 "arrival_time, departure_time, stop_name, is_adhoc, is_shared)")
        L.append("SELECT m.route_id, v.latitude::numeric, v.longitude::numeric, v.sequence_order::int,")
        L.append("       v.arrival_time::time, v.departure_time::time, v.stop_name::text,")
        L.append("       v.is_adhoc::boolean, v.is_shared::boolean")
        L.append("FROM (VALUES")
        L.append(",\n".join("  " + row for row in stop_rows))
        L.append(") AS v(service_date, route_code, sequence_order, latitude, longitude, "
                 "arrival_time, departure_time, stop_name, is_adhoc, is_shared)")
        L.append("JOIN _route_map m ON m.service_date = v.service_date::date AND m.route_code = v.route_code;")
    else:
        L.append("-- (no stops in the dumps)")
    L.append("")

    # 5) stop_passenger
    pass_rows = []
    for d in dumps:
        emp_map = (d.get("context") or {}).get("employee_id_by_email", {})
        for p in d["passengers"]:
            emp = emp_map.get(p["employee_id"])
            if emp is None:
                continue
            pass_rows.append("(" + ", ".join([
                _sql_lit(d["service_date"]),
                _sql_lit(p["route_instance_id"]),
                _sql_lit(int(p["sequence_order"])),
                _sql_lit(int(emp)),
            ]) + ")")
    L.append("-- 5) insert stop_passenger")
    if pass_rows:
        L.append("INSERT INTO public.stop_passenger (stop_id, employee_id, boarded_status)")
        L.append("SELECT rs.stop_id, v.employee_id::bigint, FALSE")
        L.append("FROM (VALUES")
        L.append(",\n".join("  " + row for row in pass_rows))
        L.append(") AS v(service_date, route_code, sequence_order, employee_id)")
        L.append("JOIN _route_map m ON m.service_date = v.service_date::date AND m.route_code = v.route_code")
        L.append("JOIN public.route_stop rs ON rs.route_id = m.route_id "
                 "AND rs.sequence_order = v.sequence_order::int;")
    else:
        L.append("-- (no passengers in the dumps)")
    L.append("")

    # 6) route_assignment
    asg_rows = []
    for d in dumps:
        c = d.get("context") or {}
        veh = c.get("vehicle_id_by_plate", {})
        drv = c.get("driver_id_by_plate", {})
        for r in d["routes"]:
            plate = r.get("plate_no")
            vid = veh.get(plate)
            if vid is None:
                continue
            if r["type"] == "pickup":
                dep, arr = r.get("parking_departure"), r.get("office_arrival")
            else:
                dep, arr = r.get("office_departure"), r.get("parking_arrival")
            asg_rows.append("(" + ", ".join([
                _sql_lit(d["service_date"]),
                _sql_lit(r["route_instance_id"]),
                _sql_lit(int(vid)),
                _sql_lit(drv.get(plate)),
                _sql_lit(_time_only(dep)),
                _sql_lit(_time_only(arr)),
            ]) + ")")
    L.append("-- 6) insert route_assignment")
    if asg_rows:
        L.append("INSERT INTO public.route_assignment (route_id, vehicle_id, driver_id, "
                 "departure_time, arrival_time, status)")
        L.append("SELECT m.route_id, v.vehicle_id::bigint, v.driver_id::bigint, "
                 "v.departure_time::time, v.arrival_time::time, 'Scheduled'")
        L.append("FROM (VALUES")
        L.append(",\n".join("  " + row for row in asg_rows))
        L.append(") AS v(service_date, route_code, vehicle_id, driver_id, departure_time, arrival_time)")
        L.append("JOIN _route_map m ON m.service_date = v.service_date::date AND m.route_code = v.route_code;")
    else:
        L.append("-- (no assignments in the dumps)")
    L.append("")

    # 7) relink requests
    def _req_updates(kind: str, table: str, id_map_key: str, id_field: str) -> None:
        rows = []
        for d in dumps:
            id_map = (d.get("context") or {}).get(id_map_key, {})
            for p in d["passengers"]:
                if p["type"] != kind:
                    continue
                rid = id_map.get(p["employee_id"])
                if rid is None:
                    continue
                rows.append("(" + ", ".join([
                    _sql_lit(d["service_date"]),
                    _sql_lit(p["route_instance_id"]),
                    _sql_lit(int(rid)),
                ]) + ")")
        if not rows:
            L.append(f"-- (no {table} links in the dumps)")
            return
        L.append(f"UPDATE public.{table} r SET route_id = m.route_id, status = 'Approved'")
        L.append("FROM (VALUES")
        L.append(",\n".join("  " + row for row in rows))
        L.append(") AS v(service_date, route_code, request_id)")
        L.append("JOIN _route_map m ON m.service_date = v.service_date::date AND m.route_code = v.route_code")
        L.append(f"WHERE r.{id_field} = v.request_id::bigint;")

    L.append("-- 7) relink requests to their route")
    _req_updates("pickup", "pickup_request", "pickup_id_by_email", "pickup_id")
    L.append("")
    _req_updates("dropoff", "dropoff_request", "dropoff_id_by_email", "dropoff_id")
    L.append("")
    L.append("-- 8) sanity check + commit")
    L.append("SELECT service_date, route_type, count(*) AS routes")
    L.append(f"  FROM public.route WHERE service_date IN ({date_list}) GROUP BY 1, 2 ORDER BY 1, 2;")
    L.append("COMMIT;")
    out_path.write_text("\n".join(L) + "\n")


def _build_week_sql_parts(dumps: List[Dict[str, Any]], out_dir: Path,
                          part_bytes: int = 400_000) -> List[Path]:
    """Split the replace into many small files (one statement each) so they can be
    pasted into the Supabase SQL editor.

    No temp table is used: every child insert joins back to `public.route` on
    (service_date, route_code), so files are independent and can be run in
    separate sessions. Routes are chunked by `part_bytes` (geometry is bulky);
    everything else is one file per date.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = [d["service_date"] for d in dumps]
    date_list = ", ".join(_sql_lit(x) for x in dates)
    written: List[Path] = []

    def emit(name: str, text: str) -> None:
        p = out_dir / name
        p.write_text(text if text.endswith("\n") else text + "\n")
        written.append(p)

    reset = [
        "-- 00 reset: unlink requests, then delete the week's routes (children first)",
        "UPDATE public.pickup_request SET route_id = NULL, status = 'Pending'",
        f"  WHERE service_date IN ({date_list}) AND status IS DISTINCT FROM 'Rejected';",
        "UPDATE public.dropoff_request SET route_id = NULL, status = 'Pending'",
        f"  WHERE service_date IN ({date_list}) AND status IS DISTINCT FROM 'Rejected';",
        "DELETE FROM public.stop_passenger WHERE stop_id IN (",
        "  SELECT stop_id FROM public.route_stop WHERE route_id IN (",
        f"    SELECT route_id FROM public.route WHERE service_date IN ({date_list})));",
        "DELETE FROM public.route_stop WHERE route_id IN (",
        f"  SELECT route_id FROM public.route WHERE service_date IN ({date_list}));",
        "DELETE FROM public.route_assignment WHERE route_id IN (",
        f"  SELECT route_id FROM public.route WHERE service_date IN ({date_list}));",
        f"DELETE FROM public.route WHERE service_date IN ({date_list});",
    ]
    emit("00_reset_week.sql", "\n".join(reset))

    route_header = ("INSERT INTO public.route (route_code, zone_id, route_type, service_date, "
                    "shift_time, total_distance_km, total_travel_time_min, route_geometry) VALUES")
    for d in dumps:
        date = d["service_date"]
        zones = (d.get("context") or {}).get("zone_id_by_name", {})
        cur: List[str] = []
        cur_bytes = len(route_header) + 2
        part = 1
        for r in d["routes"]:
            geom = (_sql_lit(json.dumps(r["route_geometry"])) + "::jsonb"
                    if r.get("route_geometry") is not None else "NULL")
            row = "(" + ", ".join([
                _sql_lit(r["route_instance_id"]), _sql_lit(zones.get(r.get("zone_name") or "")),
                _sql_lit(r["type"]), _sql_lit(date), _sql_lit(r["shift_time"]),
                _sql_lit(r["total_distance_km"]), _sql_lit(int(round(r["total_minutes"]))), geom,
            ]) + ")"
            if cur and cur_bytes + len(row) + 2 > part_bytes:
                emit(f"10_routes_{date}_{part:02d}.sql", route_header + "\n" + ",\n".join(cur) + ";")
                part += 1
                cur, cur_bytes = [], len(route_header) + 2
            cur.append(row)
            cur_bytes += len(row) + 2
        if cur:
            emit(f"10_routes_{date}_{part:02d}.sql", route_header + "\n" + ",\n".join(cur) + ";")

    for d in dumps:
        date = d["service_date"]
        c = d.get("context") or {}

        stop_rows = []
        for s in d["stops"]:
            stop_rows.append("(" + ", ".join([
                _sql_lit(date), _sql_lit(s["route_instance_id"]), _sql_lit(int(s["sequence_order"])),
                _sql_lit(s.get("stop_lat")), _sql_lit(s.get("stop_lng")),
                _sql_lit(_time_only(s.get("arrival_time"))), _sql_lit(_time_only(s.get("departure_time"))),
                _sql_lit(s.get("stop_name")), _sql_lit(bool(s.get("is_adhoc"))), _sql_lit(bool(s.get("is_shared"))),
            ]) + ")")
        if stop_rows:
            emit(f"20_stops_{date}.sql", "\n".join([
                "-- 20 stops " + date,
                "INSERT INTO public.route_stop (route_id, latitude, longitude, sequence_order, arrival_time, departure_time, stop_name, is_adhoc, is_shared)",
                "SELECT r.route_id, v.latitude::numeric, v.longitude::numeric, v.sequence_order::int,",
                "       v.arrival_time::time, v.departure_time::time, v.stop_name::text,",
                "       v.is_adhoc::boolean, v.is_shared::boolean",
                "FROM (VALUES",
                ",\n".join("  " + x for x in stop_rows),
                ") AS v(service_date, route_code, sequence_order, latitude, longitude, arrival_time, departure_time, stop_name, is_adhoc, is_shared)",
                "JOIN public.route r ON r.service_date = v.service_date::date AND r.route_code = v.route_code;",
            ]))

        emp = c.get("employee_id_by_email", {})
        pass_rows = []
        for p in d["passengers"]:
            e = emp.get(p["employee_id"])
            if e is None:
                continue
            pass_rows.append("(" + ", ".join([
                _sql_lit(date), _sql_lit(p["route_instance_id"]),
                _sql_lit(int(p["sequence_order"])), _sql_lit(int(e)),
            ]) + ")")
        if pass_rows:
            emit(f"30_passengers_{date}.sql", "\n".join([
                "-- 30 stop_passenger " + date,
                "INSERT INTO public.stop_passenger (stop_id, employee_id, boarded_status)",
                "SELECT rs.stop_id, v.employee_id::bigint, FALSE",
                "FROM (VALUES",
                ",\n".join("  " + x for x in pass_rows),
                ") AS v(service_date, route_code, sequence_order, employee_id)",
                "JOIN public.route r ON r.service_date = v.service_date::date AND r.route_code = v.route_code",
                "JOIN public.route_stop rs ON rs.route_id = r.route_id AND rs.sequence_order = v.sequence_order::int;",
            ]))

        veh = c.get("vehicle_id_by_plate", {})
        drv = c.get("driver_id_by_plate", {})
        asg_rows = []
        for r in d["routes"]:
            vid = veh.get(r.get("plate_no"))
            if vid is None:
                continue
            if r["type"] == "pickup":
                dep, arr = r.get("parking_departure"), r.get("office_arrival")
            else:
                dep, arr = r.get("office_departure"), r.get("parking_arrival")
            asg_rows.append("(" + ", ".join([
                _sql_lit(date), _sql_lit(r["route_instance_id"]), _sql_lit(int(vid)),
                _sql_lit(drv.get(r.get("plate_no"))), _sql_lit(_time_only(dep)), _sql_lit(_time_only(arr)),
            ]) + ")")
        if asg_rows:
            emit(f"40_assignments_{date}.sql", "\n".join([
                "-- 40 route_assignment " + date,
                "INSERT INTO public.route_assignment (route_id, vehicle_id, driver_id, departure_time, arrival_time, status)",
                "SELECT r.route_id, v.vehicle_id::bigint, v.driver_id::bigint, v.departure_time::time, v.arrival_time::time, 'Scheduled'",
                "FROM (VALUES",
                ",\n".join("  " + x for x in asg_rows),
                ") AS v(service_date, route_code, vehicle_id, driver_id, departure_time, arrival_time)",
                "JOIN public.route r ON r.service_date = v.service_date::date AND r.route_code = v.route_code;",
            ]))

        for kind, table, key, field in (
            ("pickup", "pickup_request", "pickup_id_by_email", "pickup_id"),
            ("dropoff", "dropoff_request", "dropoff_id_by_email", "dropoff_id"),
        ):
            id_map = c.get(key, {})
            req_rows = []
            for p in d["passengers"]:
                if p["type"] != kind:
                    continue
                rid = id_map.get(p["employee_id"])
                if rid is None:
                    continue
                req_rows.append("(" + ", ".join([
                    _sql_lit(date), _sql_lit(p["route_instance_id"]), _sql_lit(int(rid)),
                ]) + ")")
            if req_rows:
                emit(f"50_{table}_{date}.sql", "\n".join([
                    f"-- 50 relink {table} " + date,
                    f"UPDATE public.{table} q SET route_id = r.route_id, status = 'Approved'",
                    "FROM (VALUES",
                    ",\n".join("  " + x for x in req_rows),
                    ") AS v(service_date, route_code, request_id)",
                    "JOIN public.route r ON r.service_date = v.service_date::date AND r.route_code = v.route_code",
                    f"WHERE q.{field} = v.request_id::bigint;",
                ]))

    emit("99_verify.sql", "\n".join([
        "-- sanity: routes per date/type after loading all parts",
        "SELECT service_date, route_type, count(*) AS routes",
        f"  FROM public.route WHERE service_date IN ({date_list}) GROUP BY 1, 2 ORDER BY 1, 2;",
    ]))
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Standalone backend solver (ML): one date or a range, with JSON "
                    "dumps and a replace-all SQL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2026-09-22", help="single service date (default 2026-09-22)")
    ap.add_argument("--start", metavar="YYYY-MM-DD", help="first date of a range")
    ap.add_argument("--end", metavar="YYYY-MM-DD", help="last date of a range (inclusive; defaults to --start)")
    ap.add_argument("--offline", metavar="FILE", help="solve a fixture JSON instead of the DB (single date only)")
    ap.add_argument("--no-ml", action="store_true", help="raw OSRM durations (routing_night parity)")
    ap.add_argument("--haversine", action="store_true", help="force straight-line fallback")
    ap.add_argument("--model", choices=["xgb", "rf"], default="xgb",
                    help="ML duration model: xgb (native categorical) or rf (one-hot)")
    ap.add_argument("--json-summary", action="store_true",
                    help="also print one JSON line per date: {service_date, engine, use_ml, counts, unassigned_by_reason}")
    ap.add_argument("--diagnostics", action="store_true", default=True,
                    help="print accounting/fleet-overlap checks, duty blocks, and ride/wait "
                         "fairness stats after each date (default: on)")
    ap.add_argument("--no-diagnostics", action="store_false", dest="diagnostics",
                    help="skip the diagnostics block (useful for a long --start/--end range)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the routing/adapter INFO logs (k-means per-shift lines, "
                         "engine selection, etc.) -- only WARNING and above")

    cfg_default = SolverConfig()
    hp = ap.add_argument_group("solver hyperparameters (default reproduces routing_night.py)")
    hp.add_argument("--walk-limit-min", type=float, default=cfg_default.walk_limit_min,
                    help="Case A: max foot-network walk to a fixed stop, minutes")
    hp.add_argument("--walk-speed-kmph", type=float, default=cfg_default.walk_speed_kmph,
                    help="straight-line walk speed, used only when OSRM foot is unreachable")
    hp.add_argument("--boarding-buffer-min", type=float, default=cfg_default.boarding_buffer_min,
                    help="minutes a vehicle waits per stop for boarding/alighting")
    hp.add_argument("--office-buffer-min", type=float, default=cfg_default.office_buffer_min,
                    help="minutes a pickup must reach the office ahead of shift start")
    hp.add_argument("--max-route-minutes", type=float, default=cfg_default.max_route_minutes,
                    help="hard cap on a single route's on-road passenger time")
    hp.add_argument("--dropoff-return-weight", type=float, default=cfg_default.dropoff_return_weight,
                    help="0..1: how much drop-off ordering prices the (undriven) leg home")
    hp.add_argument("--near-tie-slack", type=float, default=cfg_default.near_tie_slack,
                    help="Case B (23:00): treat cars within this x the nearest car's "
                         "distance as an equally-good choice")
    hp.add_argument("--near-tie-km", type=float, default=cfg_default.near_tie_km_allowance,
                    dest="near_tie_km_allowance",
                    help="Case B (23:00): ...or within this many km, whichever is more permissive")
    hp.add_argument("--cluster-zone-penalty-km", type=float, default=cfg_default.cluster_zone_penalty_km,
                    help="Case B-kmeans: penalty (km) for matching a cluster to an out-of-zone car")
    hp.add_argument("--cluster-restarts", type=int, default=cfg_default.cluster_restarts,
                    help="Case B-kmeans: seeded k-means++ restarts, best SSE wins")
    hp.add_argument("--cluster-seed", type=int, default=cfg_default.cluster_seed,
                    help="Case B-kmeans: base RNG seed (fixed so repeat runs agree)")
    ap.add_argument("--dump-dir", metavar="DIR", help="write one solved_<date>.json per date here")
    ap.add_argument("--sql-out", metavar="FILE", help="write one transactional SQL file that replaces those dates")
    ap.add_argument("--sql-parts", metavar="DIR",
                    help="write many small single-statement SQL files here (SQL-editor friendly)")
    ap.add_argument("--sql-part-bytes", type=int, default=400_000,
                    help="max bytes per routes part file (default 400000)")
    ap.add_argument("--sql-from-dir", metavar="DIR",
                    help="do NOT solve: rebuild --sql-out/--sql-parts from solved_*.json files in DIR")
    args = ap.parse_args()

    # Every routing/adapter module in this file logs through `logging`, not
    # `print` -- the k-means per-shift line, "no free car" warnings, engine
    # selection, DB-adapter warnings. Without a handler configured, INFO never
    # reaches the console and even WARNING only hits stderr's bare last-resort
    # handler, so none of that was ever visible from the CLI.
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    # httpx (and the httpcore logger it wraps) log every single HTTP request at
    # INFO -- harmless at WARNING-level root, but deafening once routing's own
    # INFO logs are turned on, since every OSRM/foot table call floods the
    # console with its request line. Silence those two specifically; our own
    # loggers ("uvicorn.error", "app.*") are untouched.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    global _ML_MODEL_KIND
    _ML_MODEL_KIND = args.model

    cfg = SolverConfig(
        walk_limit_min=args.walk_limit_min,
        walk_speed_kmph=args.walk_speed_kmph,
        boarding_buffer_min=args.boarding_buffer_min,
        office_buffer_min=args.office_buffer_min,
        max_route_minutes=args.max_route_minutes,
        dropoff_return_weight=args.dropoff_return_weight,
        near_tie_slack=args.near_tie_slack,
        near_tie_km_allowance=args.near_tie_km_allowance,
        cluster_zone_penalty_km=args.cluster_zone_penalty_km,
        cluster_restarts=args.cluster_restarts,
        cluster_seed=args.cluster_seed,
    )

    if args.sql_from_dir:
        src = Path(args.sql_from_dir)
        files = sorted(src.glob("solved_*.json"))
        if not files:
            raise SystemExit(f"no solved_*.json found in {src}")
        dumps = [json.loads(p.read_text()) for p in files]
        if args.sql_parts:
            parts = _build_week_sql_parts(dumps, Path(args.sql_parts), args.sql_part_bytes)
            print(f"[sql] wrote {len(parts)} part file(s) to {args.sql_parts} covering "
                  f"{len(dumps)} date(s): {', '.join(d['service_date'] for d in dumps)}")
        else:
            out = Path(args.sql_out) if args.sql_out else src / "insert_week.sql"
            _build_week_sql(dumps, out)
            print(f"[sql] wrote {out} covering {len(dumps)} date(s): "
                  f"{', '.join(d['service_date'] for d in dumps)}")
        return 0

    dates = _resolve_dates(args)
    if args.offline and len(dates) > 1:
        raise SystemExit("--offline supports a single date; it has no per-date DB rows")
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
    elif args.start:
        dump_dir = Path(__file__).resolve().parent / f"solved_{dates[0]}_{dates[-1]}"
    else:
        dump_dir = None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    sql_out = Path(args.sql_out) if args.sql_out else (dump_dir / "insert_week.sql" if dump_dir else None)

    db = None
    if args.offline:
        path = args.offline if os.path.isabs(args.offline) else str(Path(__file__).resolve().parent / args.offline)
        offline_input, off_warnings = _load_offline(path)
        print(f"[source] fixture: {path}")
    else:
        db = _db_client()

    if args.haversine:
        provider, foot = HaversineProvider(), HaversineWalkProvider()
    elif args.no_ml:
        provider, foot = OsrmProvider(), get_foot_provider()
    else:
        provider, foot = get_provider(), get_foot_provider()

    use_ml = not args.no_ml
    all_dumps: List[Dict[str, Any]] = []
    for service_date in dates:
        if args.offline:
            ctx = None
            solver_input, warns = offline_input, off_warnings
        else:
            ctx = load(db, service_date)
            solver_input, warns = ctx.solver_input, ctx.warnings
            print(f"[source] live DB, service_date={service_date}  stats={ctx.stats}")
        _coordinate_check(solver_input["pickup_requests"], solver_input["dropoff_requests"])
        solved = solve_night(service_date=service_date, provider=provider, foot=foot,
                             cfg=cfg, use_ml=use_ml, **solver_input)
        engine = getattr(provider, "name", "unknown")
        _summary(f"{service_date}  engine={engine}  use_ml={use_ml}", solved, engine, warns)
        if args.diagnostics:
            _print_fleet_diagnostics(solved, solver_input["pickup_requests"],
                                     solver_input["dropoff_requests"])
        if args.json_summary:
            by_reason = Counter((u["type"], u["reason"]) for u in solved.unassigned)
            print("JSON " + json.dumps({
                "service_date": service_date, "engine": engine, "use_ml": use_ml,
                "model": _ML_MODEL_KIND if use_ml else None,
                "counts": solved.counts(),
                "unassigned_by_reason": {f"{t}:{r}": n for (t, r), n in by_reason.items()},
                "hyperparams": {
                    "near_tie_slack": cfg.near_tie_slack,
                    "near_tie_km_allowance": cfg.near_tie_km_allowance,
                    "cluster_zone_penalty_km": cfg.cluster_zone_penalty_km,
                    "cluster_restarts": cfg.cluster_restarts,
                    "cluster_seed": cfg.cluster_seed,
                    "dropoff_return_weight": cfg.dropoff_return_weight,
                    "boarding_buffer_min": cfg.boarding_buffer_min,
                    "office_buffer_min": cfg.office_buffer_min,
                    "walk_limit_min": cfg.walk_limit_min,
                    "walk_speed_kmph": cfg.walk_speed_kmph,
                    "max_route_minutes": cfg.max_route_minutes,
                },
            }))
        dump = _dump_payload(service_date, solved, ctx)
        all_dumps.append(dump)
        if dump_dir:
            p = dump_dir / f"solved_{service_date}.json"
            p.write_text(json.dumps(dump, indent=2, default=str))
            print(f"[dump] {p}")

    has_ctx = bool(all_dumps) and all(d["context"] is not None for d in all_dumps)
    if args.sql_parts:
        if not has_ctx:
            print("[sql] skipped: --offline runs carry no DB id maps, so no SQL was generated")
        else:
            parts = _build_week_sql_parts(all_dumps, Path(args.sql_parts), args.sql_part_bytes)
            print(f"[sql] wrote {len(parts)} part file(s) to {args.sql_parts}")
    elif sql_out:
        if not has_ctx:
            print("[sql] skipped: --offline runs carry no DB id maps, so no SQL was generated")
        else:
            _build_week_sql(all_dumps, sql_out)
            print(f"[sql] wrote {sql_out}")

    for obj in (foot, provider):
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - best-effort shutdown
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
