"""Prepare compilable copies of unchanged NASA Mishchenko sources.

NASA distributes its demonstration output above the Fortran source in
several files; that leading prose is not valid fixed-form Fortran.
Retain originals under src/ and remove only that prefix from build_src/.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"src"
OUTPUT=ROOT/"build_src"
SOURCES={
    "fixed_dp":["ampld.lp.f","lpd.f","ampld.par.f"],
    "fixed_qp":["amplq.lp.f","lpq.f","amplq.par.f"],
    "random_dp":["tmd.lp.f","lpd.f","tmd.par.f"],
    "random_qp":["tmq.lp.f","lpq.f","tmq.par.f"],
    "mie_dp":["spher.f"],
}

def convert_quad_intrinsics(code):
    """Modernize obsolete F77 quad intrinsics for gfortran, preserving kind."""
    # Replace the two constructors with modern generic intrinsics and
    # explicitly request IEEE binary128 to avoid silent down-conversion.
    for old,new in (("QFLOAT","REAL"),("QCMPLX","CMPLX")):
        found=list(re.finditer(r"\b"+old+r"\s*\(",code,flags=re.I))
        for match in reversed(found):
            opening=code.find("(",match.start())
            depth=0
            end=None
            for pos in range(opening,len(code)):
                if code[pos]=="(":depth+=1
                elif code[pos]==")":
                    depth-=1
                    if depth==0:
                        end=pos
                        break
            if end is None:raise RuntimeError("unbalanced quad intrinsic "+old)
            code=(code[:match.start()]+new+code[opening:end]+",KIND=16"+
                  code[end:])
    mapping={"QABS":"ABS","QACOS":"ACOS","QARCOS":"ACOS",
             "QATAN":"ATAN","QCONJG":"CONJG","QCOS":"COS",
             "QCOSH":"COSH","QIMAG":"AIMAG","QSIN":"SIN",
             "QSINH":"SINH","QSQRT":"SQRT","QMAX1":"MAX"}
    for old,new in mapping.items():
        code=re.sub(r"\b"+old+r"\b",new,code,flags=re.I)
    return code.replace("Q10.3","E10.3")

def main():
    records=[]
    for setname,names in SOURCES.items():
        outdir=OUTPUT/setname
        outdir.mkdir(parents=True,exist_ok=True)
        for filename in names:
            orig=INPUT/setname/filename
            blob=orig.read_bytes()
            lines=blob.decode("ascii").splitlines(keepends=True)
            skipped=0
            # Lines of demonstration output precede the first F77 comment.
            if lines and lines[0][:1] not in (" ","C","c","*","!"):
                matches=[i for i,line in enumerate(lines) if re.match(r"^[Cc]\s",line)]
                if not matches:raise RuntimeError(f"No fixed-form source found in {orig}")
                skipped=matches[0]
            clean="".join(lines[skipped:])
            # GNU Fortran removed the historical Q... intrinsic names.
            # Preserve 128-bit real and 256-bit complex semantics.
            if setname.endswith("_qp") and filename.endswith(".f"):
                clean=convert_quad_intrinsics(clean)
            if not clean.strip():raise RuntimeError(f"Empty clean file {orig}")
            dest=outdir/filename
            dest.write_bytes(clean.encode("ascii"))
            records.append({
                "original":str(orig.relative_to(ROOT)),
                "prepared":str(dest.relative_to(ROOT)),
                "removed_sample_lines":skipped,
                "original_sha256":hashlib.sha256(blob).hexdigest(),
                "prepared_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
            })
            print(f"{setname}/{filename}: removed {skipped} sample-output lines")
    (OUTPUT/"manifest.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
