"""Stage a private fixed-DP source tree for each CMake build directory."""
from pathlib import Path
import shutil
import subprocess
import sys

holo = Path(__file__).resolve().parent
root = holo.parent / "mishenko"
stage = Path(sys.argv[1]).resolve()
source = stage / "src" / "fixed_dp"
source.mkdir(parents=True, exist_ok=True)
for name in ("ampld.lp.f", "lpd.f", "ampld.par.f"):
    shutil.copyfile(root / "src" / "fixed_dp" / name, source / name)
qsource = stage / "src" / "fixed_qp"
qsource.mkdir(parents=True, exist_ok=True)
for name in ("amplq.lp.f", "lpq.f", "amplq.par.f"):
    shutil.copyfile(root / "src" / "fixed_qp" / name, qsource / name)

sys.path.insert(0, str(root / "tools"))
import prepare_sources as ps
import prepare_capsule as pc
ps.ROOT, ps.INPUT, ps.OUTPUT = stage, stage / "src", stage / "build_src"
ps.SOURCES = {"fixed_dp": ["ampld.lp.f", "lpd.f", "ampld.par.f"],
              "fixed_qp": ["amplq.lp.f", "lpq.f", "amplq.par.f"]}
ps.main()
pc.ROOT = stage
pc.prepare("dp", "d", 8, "LAM")
pc.prepare("qp", "q", 16, "DLAM")
subprocess.run([sys.executable, str(holo / "prepare_holo.py"),
                str(stage / "holo_core.f"),
                str(stage / "build_src/fixed_dp/capsule.f")], check=True)
subprocess.run([sys.executable, str(holo / "prepare_qp.py"),
                str(stage / "build_src/fixed_qp"),
                str(stage / "holo_qp.f")], check=True)
