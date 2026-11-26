#!/usr/bin/env python3
"""Compare all Float32 pixels and cross sections of two complete sweeps."""
import csv
import math
import struct
import sys
from pathlib import Path

def load(p):
    p = Path(p)
    log = dict(line.split("=", 1) for line in (p / "log.txt").read_text().splitlines() if "=" in line)
    assert log["status"] == "complete", p
    with (p / "orientations.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    data = (p / "holograms.raw").read_bytes()
    assert len(rows) == int(log["image_count"])
    assert len(data) == 4 * int(log["nx"]) * int(log["ny"]) * len(rows)
    return log, rows, data

def compare(a, b):
    la, ra, da = load(a)
    lb, rb, db = load(b)
    for k in ("nx", "ny", "dx_um", "dy_um", "pad", "length_um", "diameter_um",
              "n_real", "n_imag", "n_medium", "lambda_vacuum_um", "NA",
              "z_um", "x_um", "y_um", "polarization_deg", "shape", "detector",
              "NMAX", "NGAUSS", "tmatrix_precision"):
        assert la[k] == lb[k], (k, la[k], lb[k])
    assert len(ra) == len(rb)
    columns = ("Cext_um2", "Csca_um2", "Cabs_um2")
    err = {k: 0. for k in columns}
    for arow, brow in zip(ra, rb):
        assert arow["theta_deg"] == brow["theta_deg"] and arow["phi_deg"] == brow["phi_deg"]
        for k in columns:
            err[k] = max(err[k], abs(float(arow[k]) - float(brow[k])))
    count = len(da) // 4
    worst = sq = 0.
    different = 0
    for (x,), (y,) in zip(struct.iter_unpack("<f", da), struct.iter_unpack("<f", db)):
        assert math.isfinite(x) and math.isfinite(y)
        d = abs(x-y)
        worst = max(worst, d)
        sq += d*d
        different += x != y
    ta, tb = float(la["elapsed_seconds"]), float(lb["elapsed_seconds"])
    print(f"Frames={len(ra)}, pixels={count}, different={different}")
    print(f"Max intensity error={worst:.9g}, RMS={math.sqrt(sq/count):.9g}")
    print("Maximum absolute cross-section errors (um2):", err)
    print(f"CPU={ta:.8g} s CUDA={tb:.8g} s speedup={ta/tb:.5g}")
    print("Mean image times (including first image):", *(sum(float(r["elapsed_seconds"]) for r in rows)/len(rows) for rows in (ra, rb)))
    assert worst < 2e-6 and max(err.values()) < 1e-8
    print("PASS")

if __name__ == "__main__":
    compare(*sys.argv[1:])
