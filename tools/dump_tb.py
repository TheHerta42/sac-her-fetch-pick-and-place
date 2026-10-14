"""Dump TensorBoard event files into readable summaries.

Usage:
    python dump_tb.py <dir-or-file> [more dirs...]

Prints, per run: scalar tags, point counts, first/last/min/max values.
"""
import sys
import os
import glob
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_runs(paths):
    """Yield (run_name, event_file) for every event file under the given paths."""
    for p in paths:
        if os.path.isfile(p):
            yield os.path.basename(os.path.dirname(p)) or p, p
            continue
        for ev in sorted(glob.glob(os.path.join(p, "**", "events.out.tfevents.*"), recursive=True)):
            run = os.path.relpath(os.path.dirname(ev), p).replace("\\", "/")
            yield f"{os.path.basename(p.rstrip('/'))}/{run}", ev


def summarize(run_name, ev_path):
    size_mb = os.path.getsize(ev_path) / 1e6
    acc = EventAccumulator(ev_path, size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    print(f"\n=== {run_name}  ({size_mb:.1f} MB) ===")
    if not tags:
        print("   (no scalar tags)")
        return {}
    out = {}
    for tag in sorted(tags):
        evs = acc.Scalars(tag)
        steps = [e.step for e in evs]
        vals = [e.value for e in evs]
        out[tag] = (steps, vals)
        print(f"   {tag:<45} n={len(vals):<6} "
              f"step[{steps[0]}..{steps[-1]}] "
              f"first={vals[0]:>10.3f} last={vals[-1]:>10.3f} "
              f"min={min(vals):>10.3f} max={max(vals):>10.3f}")
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for run_name, ev in find_runs(sys.argv[1:]):
        try:
            summarize(run_name, ev)
        except Exception as exc:  # noqa: BLE001 - diagnostic tool
            print(f"\n=== {run_name} ===\n   ERROR: {exc}")


if __name__ == "__main__":
    main()
