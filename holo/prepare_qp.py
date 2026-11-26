"""Generate isolated binary128 T-matrix solver sharing the binary64 TMAT block."""
from pathlib import Path
import re
import sys
from tmatrix_openmp import parallelize

src = Path(sys.argv[1])
dest = Path(sys.argv[2])
main, routines = (src / "capsule.f").read_text(encoding="ascii").split("      SUBROUTINE AMPL ", 1)
main = main.replace("      PROGRAM MISH_CAPSULE", """      SUBROUTINE SOLVE_HOLO_QP(NP,AXID,EPSD,LAMD,MRRD,MRID,
     & DDELT,NDGS,NMAX,NGAUSS)""", 1)
main = main.replace("      REAL*16 AXI", "      REAL*16 AXI\n      REAL*8 AXID,EPSD,LAMD,MRRD,MRID", 1)
a = main.index("      AXI=10D0")
b = main.index("      P=ACOS(-1Q0)", a)
main = main[:a] + """      RAT=1D0
      AXI=REAL(AXID,16)
      EPS=REAL(EPSD,16)
      LAM=REAL(LAMD,16)
      MRR=REAL(MRRD,16)
      MRI=REAL(MRID,16)
""" + main[b:]
main = main[:main.index("C     The angular extent")]
main = main.replace("STOP 3", "GO TO 999")
main += "      RETURN\n  999 NMAX=-1\n      RETURN\n      END\n"
routines = routines[routines.index("      SUBROUTINE CONST "):]
code = main + routines + "\n" + (src / "lpq.f").read_text(encoding="ascii")
names = {s.upper() for s in re.findall(r"(?mi)^ {6}[^!\n]*?\b(?:SUBROUTINE|FUNCTION)\s+([A-Z][A-Z0-9_]*)", code)} - {"SOLVE_HOLO_QP"}
names |= {s.upper() for s in re.findall(r"(?mi)^ {6}COMMON\s*/([A-Z0-9_]+)/", code)} - {"TMAT"}
pattern = re.compile(r"\b(" + "|".join(sorted(names)) + r")\b", re.I)
def rename(line):
    if not line.startswith(" "):
        return line
    parts = re.split(r"('[^']*')", line)
    return "".join(p if i % 2 else pattern.sub(lambda m: "HQ_" + m[0].upper(), p)
                   for i,p in enumerate(parts))
dest.write_text(parallelize("\n".join(map(rename, code.splitlines()))+"\n"), encoding="ascii")
