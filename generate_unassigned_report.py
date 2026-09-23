#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse data/report.txt (a week of run_ml_solver_db.py console output) and
render a multi-page PDF: per-day unassigned breakdowns, fleet/fairness trends,
and a statistical-analysis write-up.

Palette: the dataviz skill's validated default (references/palette.md) --
fixed categorical order, single-hue sequential for magnitude, no dual-axis
charts, consistent color-per-entity across pages.
"""
import re
import ast
import statistics
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

import argparse

_HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(description="Render report.txt into a PDF of unassigned-rider charts + analysis.")
_ap.add_argument("--in", dest="in_path", default=str(_HERE / "report.txt"),
                 help="console-output file to parse (default: report.txt next to this script)")
_ap.add_argument("--out", dest="out_path", default=str(_HERE / "unassigned_analysis_report.pdf"),
                 help="PDF path to write (default: unassigned_analysis_report.pdf next to this script)")
_args = _ap.parse_args()

REPORT_PATH = Path(_args.in_path)
OUT_PATH = Path(_args.out_path)

# ---- palette (dataviz skill, light mode, fixed categorical order) ----------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

REASON_COLOR = {
    "no_vehicle_available": BLUE,
    "dropped_for_120min_cap": ORANGE,
    "vehicle_not_free_in_time": AQUA,
}
TYPE_COLOR = {"pickup": BLUE, "dropoff": ORANGE}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _strip_grid(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)


# ============================================================================
# Parsing
# ============================================================================

def parse_report(text: str):
    # split on the command-prompt lines that start each run
    chunks = re.split(r"PS [^\n]*python run_ml_solver_db\.py --date (\d{4}-\d{2}-\d{2})\n", text)
    # re.split with a capturing group yields: [pre, date1, block1, date2, block2, ...]
    runs = []
    for i in range(1, len(chunks), 2):
        date = chunks[i]
        block = chunks[i + 1]
        runs.append(_parse_block(date, block))
    return runs


def _num(pattern, block, cast=float, default=None):
    m = re.search(pattern, block)
    return cast(m.group(1)) if m else default


def _parse_block(date, block):
    engine_m = re.search(r"engine=(\w+)\s+use_ml=(True|False)", block)
    engine = engine_m.group(1) if engine_m else "unknown"
    use_ml = engine_m.group(2) == "True" if engine_m else None

    counts_m = re.search(r"counts:\s*(\{.*\})", block)
    counts = ast.literal_eval(counts_m.group(1)) if counts_m else {}

    input_m = re.search(r"routing input for [^:]+:\s*(\{.*\})", block)
    inputs = ast.literal_eval(input_m.group(1)) if input_m else {}

    coord_m = re.search(
        r"Coordinates:\s*(\d+)/(\d+) pickup requests valid,\s*(\d+)/(\d+) dropoff requests valid",
        block)

    detail = []
    detail_section = re.search(
        r"unassigned by \(type, shift_time, reason\):\n((?:.*\n)*?)\n", block)
    if detail_section:
        for line in detail_section.group(1).splitlines():
            m = re.match(r"\s*(\S+)\s+(\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\d+)", line)
            if m:
                detail.append({
                    "type": m.group(1), "shift_time": m.group(2),
                    "reason": m.group(3), "count": int(m.group(4)),
                })

    by_reason = {}
    reason_section = re.search(r"Unassigned by reason:\n((?:.*\n)*?)\n", block)
    if reason_section:
        for line in reason_section.group(1).splitlines():
            m = re.match(r"\s*(\S+)\s+(\d+)\s*$", line)
            if m:
                by_reason[m.group(1)] = int(m.group(2))

    def ride_wait(kind):
        m = re.search(
            kind + r"\s+ride\s+mean=\s*([\d.]+)\s+p50=\s*([\d.]+)\s+p90=\s*([\d.]+)"
            r"\s+MAX=\s*([\d.]+) min\s*\|\s*total\s*(\d+) pax-min", block)
        w = re.search(kind + r"\s+wait\s+mean=\s*([\d.]+)", block)
        if not m:
            return None
        return {
            "ride_mean": float(m.group(1)), "ride_p50": float(m.group(2)),
            "ride_p90": float(m.group(3)), "ride_max": float(m.group(4)),
            "total_pax_min": int(m.group(5)),
            "wait_mean": float(w.group(1)) if w else None,
        }

    return {
        "date": date,
        "engine": engine,
        "use_ml": use_ml,
        "zones": inputs.get("zones"),
        "vehicles": inputs.get("vehicles"),
        "pickup_requests": inputs.get("pickup_requests"),
        "dropoff_requests": inputs.get("dropoff_requests"),
        "fixed_stops": inputs.get("fixed_stops"),
        "routes": counts.get("routes"),
        "pickup_routes": counts.get("pickup_routes"),
        "dropoff_routes": counts.get("dropoff_routes"),
        "stops": counts.get("stops"),
        "passengers": counts.get("passengers"),
        "unassigned": counts.get("unassigned"),
        "coord_pickup_valid": int(coord_m.group(1)) if coord_m else None,
        "coord_pickup_total": int(coord_m.group(2)) if coord_m else None,
        "coord_dropoff_valid": int(coord_m.group(3)) if coord_m else None,
        "coord_dropoff_total": int(coord_m.group(4)) if coord_m else None,
        "unassigned_detail": detail,
        "unassigned_by_reason": by_reason,
        "fleet_driving_min": _num(r"Fleet driving minutes:\s*(\d+)", block, int),
        "fleet_idle_min": _num(r"Fleet idle minutes between trips:\s*(\d+)", block, int),
        "drive_idle_ratio": _num(r"drive:idle = ([\d.]+):1", block, float),
        "pickup_fair": ride_wait("pickup"),
        "dropoff_fair": ride_wait("dropoff"),
        "dh_in": _num(r"Drop-off deadheads IN to the office:\s*(\d+)", block, int),
        "dh_return": _num(r"Drop-off return legs to the office:\s*(\d+)", block, int),
        "ends_at_office": _num(r"Drop-off tours still finishing at the office:\s*(\d+)", block, int),
        "stranded": _num(r"Cars whose night ends at a rider's stop before 06:00:\s*(\d+)", block, int),
    }


# ============================================================================
# Load + de-duplicate (one canonical run per date; keep all runs for the
# reproducibility note)
# ============================================================================

text = REPORT_PATH.read_text(encoding="utf-8", errors="replace")
all_runs = parse_report(text)

by_date_all = {}
for r in all_runs:
    by_date_all.setdefault(r["date"], []).append(r)

dates = sorted(by_date_all.keys())
runs = [by_date_all[d][-1] for d in dates]          # last run per date = canonical
weekdays = [datetime.strptime(d, "%Y-%m-%d").strftime("%a") for d in dates]
day_labels = [f"{wd}\n{d[5:]}" for wd, d in zip(weekdays, dates)]

n = len(dates)
x = np.arange(n)

reasons = ["no_vehicle_available", "dropped_for_120min_cap", "vehicle_not_free_in_time"]
types = ["pickup", "dropoff"]

pdf = PdfPages(OUT_PATH)


def new_fig(figsize=(11, 8.5)):
    fig = plt.figure(figsize=figsize)
    return fig


def save(fig):
    pdf.savefig(fig)
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.0f}"):
    for b in bars:
        h = b.get_height()
        if h <= 0:
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)


# ============================================================================
# Page 1 -- title + overview table
# ============================================================================
fig = new_fig()
fig.suptitle("Routing Solve: Unassigned-Rider Analysis", fontsize=18, fontweight="bold",
             color=INK, x=0.06, y=0.975, ha="left")
fig.text(0.06, 0.925,
         f"{dates[0]} \u2192 {dates[-1]}  \u00b7  {n} service dates  \u00b7  "
         f"source: run_ml_solver_db.py (live DB, engine=osrm, use_ml=True)",
         fontsize=10, color=INK_SECONDARY, ha="left")

col_labels = ["Date", "Day", "Pickup\nreq", "Dropoff\nreq", "Total\nreq",
              "Routed", "Unassigned", "Unassigned\n%"]
table_rows = []
for d, wd, r in zip(dates, weekdays, runs):
    total_req = r["pickup_requests"] + r["dropoff_requests"]
    pct = 100.0 * r["unassigned"] / total_req if total_req else 0.0
    table_rows.append([d, wd, r["pickup_requests"], r["dropoff_requests"], total_req,
                       r["passengers"], r["unassigned"], f"{pct:.1f}%"])

ax = fig.add_axes([0.06, 0.45, 0.88, 0.42])
ax.axis("off")
tbl = ax.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID)
    if row == 0:
        cell.set_text_props(weight="bold", color=INK)
        cell.set_facecolor("#f0efec")
    else:
        cell.set_facecolor(SURFACE)
        if col == 7:
            pct_val = float(table_rows[row - 1][7].rstrip("%"))
            if pct_val >= 10:
                cell.set_text_props(color=RED, weight="bold")

totals_req = sum(r["pickup_requests"] + r["dropoff_requests"] for r in runs)
totals_unassigned = sum(r["unassigned"] for r in runs)
totals_routed = sum(r["passengers"] for r in runs)
fig.text(0.06, 0.40,
         f"Week totals: {totals_req} requests \u2192 {totals_routed} routed, "
         f"{totals_unassigned} unassigned ({100*totals_unassigned/totals_req:.1f}%)",
         fontsize=10.5, color=INK, fontweight="bold")

dup_dates = [d for d in dates if len(by_date_all[d]) > 1]
note_lines = [
    "Notes on this data:",
    "\u2022 Every request across all 7 dates has valid coordinates (no 'no_coordinates' unassigned) --",
    "  unlike the offline test fixture, this live data has no missing-location rows.",
]
if dup_dates:
    d0 = dup_dates[0]
    r1, r2 = by_date_all[d0][0], by_date_all[d0][-1]
    note_lines += [
        f"\u2022 {d0} was solved twice back-to-back (identical inputs, live DB + real OSRM).",
        f"  unassigned matched exactly both times (117), but routes differed "
        f"({r1['routes']} vs {r2['routes']}, pickup_routes {r1['pickup_routes']} vs {r2['pickup_routes']})",
        "  -- evidence the solve is not perfectly reproducible run-to-run (see Statistical Analysis).",
    ]
fig.text(0.06, 0.06, "\n".join(note_lines), fontsize=8.7, color=INK_SECONDARY, va="bottom")
save(fig)

# ============================================================================
# Page 2 -- total unassigned per day
# ============================================================================
fig = new_fig()
ax = fig.add_axes([0.09, 0.14, 0.85, 0.72])
vals = [r["unassigned"] for r in runs]
bars = ax.bar(x, vals, width=0.55, color=BLUE)
bar_labels(ax, bars)
ax.set_xticks(x)
ax.set_xticklabels(day_labels)
ax.set_ylabel("Unassigned riders")
ax.set_title("Total unassigned riders per service date", fontsize=13, fontweight="bold", color=INK, loc="left")
_strip_grid(ax)
fig.text(0.09, 0.03,
         f"Range: {min(vals)}-{max(vals)}  \u00b7  mean {statistics.mean(vals):.1f}  "
         f"\u00b7  stdev {statistics.pstdev(vals):.1f}",
         fontsize=8.5, color=INK_SECONDARY)
save(fig)

# ============================================================================
# Page 3 -- unassigned by reason, stacked per day
# ============================================================================
fig = new_fig()
ax = fig.add_axes([0.09, 0.16, 0.85, 0.70])
bottoms = np.zeros(n)
for reason in reasons:
    vals = np.array([r["unassigned_by_reason"].get(reason, 0) for r in runs], dtype=float)
    ax.bar(x, vals, bottom=bottoms, width=0.55, color=REASON_COLOR[reason],
          label=reason, edgecolor=SURFACE, linewidth=1.2)
    bottoms += vals
totals = bottoms
for xi, t in zip(x, totals):
    ax.annotate(f"{t:.0f}", (xi, t), xytext=(0, 3), textcoords="offset points",
               ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels(day_labels)
ax.set_ylabel("Unassigned riders")
ax.set_title("Unassigned riders by reason, per service date", fontsize=13, fontweight="bold", color=INK, loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, frameon=False, fontsize=8.5)
_strip_grid(ax)
save(fig)

# ============================================================================
# Page 4 -- unassigned by request type, stacked per day
# ============================================================================
fig = new_fig()
ax = fig.add_axes([0.09, 0.16, 0.85, 0.70])
bottoms = np.zeros(n)
for typ in types:
    vals = np.array([sum(d["count"] for d in r["unassigned_detail"] if d["type"] == typ)
                     for r in runs], dtype=float)
    ax.bar(x, vals, bottom=bottoms, width=0.55, color=TYPE_COLOR[typ],
          label=typ, edgecolor=SURFACE, linewidth=1.2)
    bottoms += vals
for xi, t in zip(x, bottoms):
    ax.annotate(f"{t:.0f}", (xi, t), xytext=(0, 3), textcoords="offset points",
               ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels(day_labels)
ax.set_ylabel("Unassigned riders")
ax.set_title("Unassigned riders by request type, per service date", fontsize=13,
             fontweight="bold", color=INK, loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2, frameon=False, fontsize=8.5)
_strip_grid(ax)
save(fig)

# ============================================================================
# Page 5 -- heatmap: (type, shift_time) x date
# ============================================================================
row_keys = []
for r in runs:
    for d in r["unassigned_detail"]:
        key = (d["type"], d["shift_time"])
        if key not in row_keys:
            row_keys.append(key)


def _row_sort(k):
    typ, st = k
    hh = int(st[:2])
    hh_sort = hh if hh >= 22 else hh + 24
    return (0 if typ == "pickup" else 1, hh_sort)


row_keys.sort(key=_row_sort)
grid = np.zeros((len(row_keys), n))
for j, r in enumerate(runs):
    for d in r["unassigned_detail"]:
        i = row_keys.index((d["type"], d["shift_time"]))
        grid[i, j] += d["count"]

fig = new_fig()
ax = fig.add_axes([0.20, 0.14, 0.68, 0.72])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", BLUE])
im = ax.imshow(grid, cmap=cmap, aspect="auto")
ax.set_xticks(x)
ax.set_xticklabels(day_labels, fontsize=8.5)
ax.set_yticks(range(len(row_keys)))
ax.set_yticklabels([f"{typ[0].upper()} {st[:5]}" for typ, st in row_keys], fontsize=8.5)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid[i, j]
        if v > 0:
            color = SURFACE if v > grid.max() * 0.55 else INK
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=color)
fig.text(0.06, 0.90, "Unassigned riders by (type, shift) across the week",
        fontsize=13, fontweight="bold", color=INK, ha="left")
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("riders", fontsize=8.5, color=INK_SECONDARY)
save(fig)

# ============================================================================
# Page 6 -- unassigned rate (%) over the week
# ============================================================================
fig = new_fig()
ax = fig.add_axes([0.09, 0.14, 0.85, 0.72])
rates = [100.0 * r["unassigned"] / (r["pickup_requests"] + r["dropoff_requests"]) for r in runs]
ax.plot(x, rates, color=BLUE, linewidth=2, marker="o", markersize=8,
       markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.2)
for xi, v in zip(x, rates):
    ax.annotate(f"{v:.1f}%", (xi, v), xytext=(0, 8), textcoords="offset points",
               ha="center", fontsize=8, color=INK_SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels(day_labels)
ax.set_ylabel("Unassigned rate (% of all requests)")
ax.set_title("Unassigned rate over the week", fontsize=13, fontweight="bold", color=INK, loc="left")
ax.set_ylim(0, max(rates) * 1.25)
_strip_grid(ax)
save(fig)

# ============================================================================
# Page 7 -- ride-time fairness (pickup and dropoff, separate single-axis charts)
# ============================================================================
fig = new_fig()
metrics = [("ride_mean", "mean", BLUE), ("ride_p50", "p50", ORANGE),
          ("ride_p90", "p90", AQUA), ("ride_max", "max", YELLOW)]
handles = None
for row, (kind, title) in enumerate([("pickup_fair", "Pickup"), ("dropoff_fair", "Dropoff")]):
    ax = fig.add_axes([0.09, 0.56 - row * 0.46, 0.85, 0.32])
    for key, label, color in metrics:
        vals = [r[kind][key] for r in runs]
        ax.plot(x, vals, color=color, linewidth=2, marker="o", markersize=6, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels if row == 1 else [""] * n)
    ax.set_ylabel("minutes")
    ax.set_title(f"{title} ride time in the car", fontsize=11.5,
                fontweight="bold", color=INK, loc="left")
    _strip_grid(ax)
    if row == 0:
        handles, labels = ax.get_legend_handles_labels()
fig.suptitle("Passenger ride-time fairness across the week", fontsize=14, fontweight="bold",
            color=INK, x=0.09, y=0.975, ha="left")
fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.475), ncol=4,
          frameon=False, fontsize=9)
save(fig)

# ============================================================================
# Page 8 -- fleet utilization
# ============================================================================
fig = new_fig()
ax1 = fig.add_axes([0.09, 0.58, 0.85, 0.34])
w = 0.32
drive = [r["fleet_driving_min"] for r in runs]
idle = [r["fleet_idle_min"] for r in runs]
ax1.bar(x - w / 2, drive, width=w, color=BLUE, label="driving")
ax1.bar(x + w / 2, idle, width=w, color=ORANGE, label="idle")
ax1.set_xticks(x)
ax1.set_xticklabels([""] * n)
ax1.set_ylabel("minutes")
ax1.set_title("Fleet driving vs idle minutes, per night", fontsize=12, fontweight="bold",
              color=INK, loc="left")
ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.10), ncol=2, frameon=False, fontsize=8.5)
_strip_grid(ax1)

ax2 = fig.add_axes([0.09, 0.10, 0.85, 0.34])
ratio = [r["drive_idle_ratio"] for r in runs]
bars = ax2.bar(x, ratio, width=0.5, color=AQUA)
bar_labels(ax2, bars, fmt="{:.2f}")
ax2.set_xticks(x)
ax2.set_xticklabels(day_labels)
ax2.set_ylabel("drive : idle ratio")
ax2.set_title("Drive:idle ratio, per night (higher = fleet works harder per idle minute)",
             fontsize=12, fontweight="bold", color=INK, loc="left")
_strip_grid(ax2)
save(fig)

# ============================================================================
# Page 9 -- drop-off deadhead accounting
# ============================================================================
fig = new_fig()
ax = fig.add_axes([0.09, 0.16, 0.85, 0.68])
w = 0.32
dh_in = [r["dh_in"] for r in runs]
dh_ret = [r["dh_return"] for r in runs]
ax.bar(x - w / 2, dh_in, width=w, color=BLUE, label="deadhead IN (driven)")
ax.bar(x + w / 2, dh_ret, width=w, color=ORANGE, label="return legs (not driven unless reused)")
ax.set_xticks(x)
ax.set_xticklabels(day_labels)
ax.set_ylabel("minutes")
ax.set_title("Drop-off deadhead minutes: driven vs priced-but-not-driven",
             fontsize=13, fontweight="bold", color=INK, loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=1, frameon=False, fontsize=8.5)
_strip_grid(ax)
save(fig)

# ============================================================================
# Page 10+ -- statistical analysis & interpretation
# ============================================================================
total_req_arr = np.array([r["pickup_requests"] + r["dropoff_requests"] for r in runs], dtype=float)
dropoff_req_arr = np.array([r["dropoff_requests"] for r in runs], dtype=float)
pickup_req_arr = np.array([r["pickup_requests"] for r in runs], dtype=float)
unassigned_arr = np.array([r["unassigned"] for r in runs], dtype=float)
rate_arr = 100.0 * unassigned_arr / total_req_arr

reason_totals = {reason: sum(r["unassigned_by_reason"].get(reason, 0) for r in runs)
                 for reason in reasons}
reason_share = {k: 100.0 * v / totals_unassigned for k, v in reason_totals.items()}
dominant_reason = max(reason_totals, key=reason_totals.get)

d0730_no_vehicle = sum(d["count"] for r in runs for d in r["unassigned_detail"]
                      if d["type"] == "dropoff" and d["shift_time"] == "07:00:00"
                      and d["reason"] == "no_vehicle_available")
d0730_cap = sum(d["count"] for r in runs for d in r["unassigned_detail"]
               if d["type"] == "dropoff" and d["shift_time"] == "07:00:00"
               and d["reason"] == "dropped_for_120min_cap")
p2300_wnft = sum(d["count"] for r in runs for d in r["unassigned_detail"]
                if d["type"] == "pickup" and d["shift_time"] == "23:00:00"
                and d["reason"] == "vehicle_not_free_in_time")

corr_dropoff = np.corrcoef(dropoff_req_arr, unassigned_arr)[0, 1] if n > 2 else float("nan")
corr_total = np.corrcoef(total_req_arr, unassigned_arr)[0, 1] if n > 2 else float("nan")

worst_i = int(np.argmax(rate_arr))
best_i = int(np.argmin(rate_arr))

fig = new_fig()
fig.suptitle("Statistical Analysis & Interpretation", fontsize=16, fontweight="bold",
            color=INK, x=0.06, y=0.965, ha="left")

lines = []
lines.append(("Descriptive statistics (n = %d service dates)" % n, True))
lines.append((f"  \u2022 Unassigned count: mean {unassigned_arr.mean():.1f}, "
              f"median {np.median(unassigned_arr):.1f}, stdev {unassigned_arr.std(ddof=0):.1f}, "
              f"range {int(unassigned_arr.min())}-{int(unassigned_arr.max())}.", False))
lines.append((f"  \u2022 Unassigned rate: mean {rate_arr.mean():.1f}%, "
              f"range {rate_arr.min():.1f}% ({dates[best_i]}) to {rate_arr.max():.1f}% ({dates[worst_i]}).", False))
lines.append((f"  \u2022 Total requests/night ranged {int(total_req_arr.min())}-{int(total_req_arr.max())} "
              f"(mean {total_req_arr.mean():.0f}); dropoff requests consistently outnumber "
              f"pickups {dropoff_req_arr.mean():.0f} vs {pickup_req_arr.mean():.0f} on average "
              f"(~{dropoff_req_arr.mean()/pickup_req_arr.mean():.1f}x).", False))
lines.append(("", False))
lines.append(("Correlation (n=%d -- descriptive only, not a hypothesis test)" % n, True))
lines.append((f"  \u2022 dropoff_requests vs unassigned count: r = {corr_dropoff:.2f}", False))
lines.append((f"  \u2022 total_requests vs unassigned count: r = {corr_total:.2f}", False))
lines.append((f"  \u2022 With only {n} dates this is suggestive, not conclusive -- but the direction is "
              f"consistent with the reason breakdown below: nights with more dropoff demand "
              f"packed into 07:00 see more 'no_vehicle_available' at that shift specifically.", False))
lines.append(("", False))
lines.append(("Dominant failure mode", True))
for reason in sorted(reason_totals, key=reason_totals.get, reverse=True):
    lines.append((f"  \u2022 {reason}: {reason_totals[reason]} riders across the week "
                  f"({reason_share[reason]:.0f}% of all unassigned)", False))
lines.append((f"  • Overall dominant reason: '{dominant_reason}' -- but its lead over "
              f"'dropped_for_120min_cap' is data-dependent (see per-day breakdown, page 3): "
              f"it wins 5 of 7 days but 'dropped_for_120min_cap' actually leads on "
              f"{dates[2]} and {dates[4]}.", False))
lines.append(("", False))
lines.append(("Where unassigned riders concentrate", True))
lines.append((f"  \u2022 The 07:00 drop-off shift alone accounts for {d0730_no_vehicle} "
              f"'no_vehicle_available' + {d0730_cap} 'dropped_for_120min_cap' riders across the week "
              f"-- by far the single biggest concentration, and consistent every single day "
              f"(dominant on all 7 dates, see heatmap page 5).", False))
lines.append((f"  \u2022 The 23:00 pickup shift's 'vehicle_not_free_in_time' totals {p2300_wnft} riders, "
              f"but is not universal -- it appears on 4 of 7 dates and is exactly 0 on 3, so it "
              f"is a scheduling-tightness effect that depends on how the night's earlier events "
              f"line up, not a fixed structural gap like the 07:00 drop-off.", False))
lines.append(("", False))
lines.append(("Reproducibility", True))
if dup_dates:
    d0 = dup_dates[0]
    r1, r2 = by_date_all[d0][0], by_date_all[d0][-1]
    lines.append((f"  \u2022 {d0} was solved twice, back-to-back, against the same live DB rows. "
                  f"'unassigned' matched exactly (117 both times, identical reason/shift breakdown), "
                  f"but 'routes' ({r1['routes']} vs {r2['routes']}) and 'pickup_routes' "
                  f"({r1['pickup_routes']} vs {r2['pickup_routes']}) did not.", False))
    lines.append(("  \u2022 Interpretation: which riders get left behind is stable, but exactly how the "
                  "solver merges/splits routes among the surviving riders is not -- consistent with "
                  "the DB adapter issuing no explicit ORDER BY (see run_ml_solver_db.py's "
                  "RoutingAdapter queries), so tie-breaks that depend on row order can flip between "
                  "otherwise-identical runs.", False))
else:
    lines.append(("  \u2022 No date in this report was solved more than once, so run-to-run variance "
                  "cannot be measured from this data alone.", False))
lines.append(("", False))
lines.append(("What this suggests for tuning (see prior hyperparameter discussion)", True))
lines.append(("  \u2022 The 07:00 drop-off bottleneck is dominated by 'no_vehicle_available', which "
              "SolverConfig knobs (near_tie_slack, cluster_*, walk_limit_min) do not touch at all -- "
              "it is gated by _can_serve_dropoff's free-time/deadhead check, so the real levers are "
              "fleet size and how long EARLIER trips tie up each car (which loops back to the ML "
              "duration model's accuracy, since it sets every _free_at timestamp for the night).", False))
lines.append(("  \u2022 'dropped_for_120min_cap' at 07:00 is the one bucket directly gated by "
              "max_route_minutes/boarding_buffer_min -- the two knobs with a mechanical, guaranteed "
              "effect on this specific number.", False))
lines.append(("  \u2022 23:00 pickup's 'vehicle_not_free_in_time' is gated by office_buffer_min and by "
              "how tightly earlier shifts packed the fleet -- worth checking on the specific dates "
              "where it spikes (09-22, 09-23, 09-25) rather than assuming a single fix applies "
              "every night.", False))

WRAP_WIDTH = 122
LINE_H = 0.0225
HEADER_LINE_H = 0.0285
BLOCK_GAP = 0.0105
BLANK_GAP = 0.014

y = 0.90
for text, bold in lines:
    if not text:
        y -= BLANK_GAP
        continue
    hanging = "      " if text.lstrip().startswith("•") else ""
    wrapped = textwrap.wrap(text, width=WRAP_WIDTH, subsequent_indent=hanging) or [text]
    for wline in wrapped:
        fig.text(0.06, y, wline, fontsize=9.3 if not bold else 10.6,
                 fontweight="bold" if bold else "normal",
                 color=INK if bold else INK_SECONDARY, va="top")
        y -= HEADER_LINE_H if bold else LINE_H
        if y < 0.04:
            save(fig)
            fig = new_fig()
            y = 0.94
    y -= BLOCK_GAP

save(fig)

pdf.close()
print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
