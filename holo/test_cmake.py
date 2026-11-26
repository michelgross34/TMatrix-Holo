"""CTest smoke and independent numerical validation (no NumPy needed)."""
import subprocess
import sys
import uuid
import csv
import math
import struct
import os
from pathlib import Path
import validate

exe = str(Path(sys.argv[1]).resolve())
validate.ROOT = Path(exe).parent
# Launch from the source directory, not the executable directory.
os.chdir(Path(__file__).resolve().parent)
tag = "cmake_" + uuid.uuid4().hex[:12]
name = tag + "_transparent"
subprocess.run([exe, "--name", name, "--n-real", "1.33",
                "--length-um", "2", "--diameter-um", ".8",
                "--nx", "32", "--ny", "24", "--pad", "1", "--z-um", "3"], check=True)
d, data = validate.load(name)
assert set(data) == {1.0}
assert all(float(d[k]) == 0 for k in ("Csca_um2", "Cext_um2", "Cabs_um2"))
name = tag + "_mie"
subprocess.run([exe, "--name", name, "--shape", "capsule", "--lambda-um", ".63",
                "--diameter-um", ".672", "--length-um", ".672", "--n-real", "3",
                "--n-imag", ".1", "--n-medium", "1", "--nx", "64", "--ny", "48",
                "--pad", "2", "--z-um", "3", "--theta", "27", "--phi", "48",
                "--tol", "1e-7"], check=True)
validate.check_mie(name)
print("PASS: transparent object and independent Mie hologram")

# Sweep uses one T-matrix; only the four aggregate files are retained.
name = tag + "_sweep"
common = ["--length-um", "2", "--diameter-um", ".8", "--nx", "17", "--ny", "15", "--pad", "4", "--z-um", "3"]
subprocess.run([exe, "--name", name, "--theta-steps", "4",
                "--theta", "27", "--phi", "48", *common], check=True)
root = validate.ROOT / "run" / ("run_" + name)
def metadata(path):
    return dict(s.split("=", 1) for s in path.read_text().splitlines() if "=" in s)
d = metadata(root / "log.txt")
assert d["status"] == "complete" and d["tmatrix_builds"] == "1"
assert float(d["lambda_vacuum_um"]) == 0.660
assert float(d["n_medium"]) == 1.33 and float(d["n_real"]) == 1.38 and float(d["n_imag"]) == 0
assert math.isclose(float(d["lambda_medium_um"]),0.660/1.33,rel_tol=1e-14)
assert d["image_count"] == d["completed_images"] == "5"
with (root / "orientations.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
assert [float(r["theta_deg"]) for r in rows] == [0, 45, 90, 135, 180]
assert all(float(r["phi_deg"]) == 0 for r in rows)
assert d["stack_file"] == "holograms.raw" and d["sections_file"] == "cross_sections.txt"
assert int(d["stack_frame_bytes"]) == 4*17*15
assert int(d["stack_bytes"]) == int(d["stack_expected_bytes"]) == 5*4*17*15
stack = (root / d["stack_file"]).read_bytes()
assert len(stack)==5*4*17*15
assert {p.name for p in root.iterdir()} == {"log.txt","holograms.raw","cross_sections.txt","orientations.csv"}
sections = [list(map(float, s.split())) for s in (root / d["sections_file"]).read_text().splitlines() if s.strip() and not s.startswith("#")]
assert sections == [[float(r[k]) for k in ("theta_deg", "Cext_um2", "Csca_um2", "Cabs_um2")] for r in rows]
assert not (root / "holograms.raw.part").exists()
assert not (root / "cross_sections.txt.part").exists()
images = []
for j,r in enumerate(rows):
    assert r["nx"] == "17" and r["ny"] == "15"
    for k in ("Cext_um2", "Csca_um2", "Cabs_um2"):
        assert math.isfinite(float(r[k]))
    assert r["raw_file"]=="holograms.raw"
    offset=int(r["offset_bytes"])
    assert offset==j*4*17*15
    raw=stack[offset:offset+4*17*15]
    assert len(raw) == 4 * 17 * 15
    image = struct.unpack("<255f", raw)
    assert all(math.isfinite(x) and x >= 0 for x in image)
    assert float(r["raw_min"])==min(image) and float(r["raw_max"])==max(image)
    assert math.isclose(float(r["raw_mean"]),sum(image)/len(image),rel_tol=1e-14)
    images.append(image)
assert max(abs(x-y) for x,y in zip(images[0],images[-1])) < 2e-6
assert max(abs(x-y) for x,y in zip(images[0],images[2])) > 1e-4
assert max(abs(images[1][j*17+i]-images[3][j*17+16-i])
           for j in range(15) for i in range(17)) < 2e-6
single = tag + "_theta45"
subprocess.run([exe, "--name", single, "--theta", "45", "--phi", "0", *common], check=True)
d, image = validate.load(single)
assert max(abs(x-y) for x,y in zip(image,images[1])) < 1e-7
for k in ("Cext_um2", "Csca_um2", "Cabs_um2"):
    assert math.isclose(float(d[k]),float(rows[1][k]),rel_tol=1e-10,abs_tol=1e-12)
bad = subprocess.run([exe, "--theta-steps", "0"], capture_output=True)
assert bad.returncode != 0
print("PASS: theta sweep, shared T-matrix, endpoints, reflection symmetry and standalone agreement")
