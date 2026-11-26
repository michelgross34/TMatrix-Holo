"""Parallelize independent surface integrals, not the shared COMMON workspaces.

The NASA fixed-form DP and QP routines use the same DO 300 integrals.
Each iteration of N1 writes a different row. All quadrature sums remain
in the original I order. Calls to TMATR0/TMATR and LAPACK remain serial.
"""
import re
import textwrap

def parallelize(code):
    start = "      DO 300  N1=MM1,NMAX\n"
    end = "  300 CONTINUE"
    assert code.count(start) == 2, "Unexpected TMATR integral structure"
    for nonzero in (False, True):
        a = code.index(start)
        b = code.index(end, a) + len(end)
        loop = code[a:b]
        # All scalar temporaries assigned within this particular loop.
        private = sorted(set(re.findall(
            r"(?m)^[ \t]*(?:\d+[ \t]+)?([A-Z][A-Z0-9]*)[ \t]*=", loop)))
        # The checked input and output arrays, and loop-invariant scalars.
        shared = sorted(set(re.findall(r"\b([A-Z][A-Z0-9]*)\(", loop)))
        base = set("AN ANN D1 D2 DDR DJ DJI DJR DR DRI DRR DY J JI JR RR SIG Y".split())
        outputs = {"R"+ij for ij in ("12","21")} | {"I"+ij for ij in ("12","21")} | {"RG"+ij for ij in ("12","21")} | {"IG"+ij for ij in ("12","21")}
        if nonzero:
            base.update(("DS","DSS"))
            outputs.update(prefix+ij for prefix in ("R","I","RG","IG") for ij in ("11","22"))
        assert set(shared) == base | outputs, "Unexpected arrays in integral"
        private += ["N2", "I"]
        shared += ["MM1", "NMAX", "NCHECK", "NGSS", "FACTOR"]
        # Split the old shared termination label into two canonical loops.
        loop = loop.replace(start, "      DO N1=MM1,NMAX\n", 1)
        assert loop.count("           DO 300 N2=MM1,NMAX") == 1
        loop = loop.replace("           DO 300 N2=MM1,NMAX", "           DO N2=MM1,NMAX", 1)
        loop = loop[:-len(end)] + "           ENDDO\n      ENDDO"
        # Fixed-form OpenMP: continuation sentinel in column six.
        clauses = ("SHARED("+",".join(shared)+")",
                   "PRIVATE("+",".join(private)+")")
        directive = "C$OMP PARALLEL DO DEFAULT(NONE) SCHEDULE(STATIC)\n"
        directive += "C$OMP& IF((NMAX-MM1+1)**2*NGSS.GE.10000)\n"
        for clause in clauses:
            # Break only at commas, so identifiers remain intact.
            parts = clause.split(",")
            line = "C$OMP& "
            for i, part in enumerate(parts):
                token = part + ("," if i+1 < len(parts) else "")
                if len(line)+len(token)>70:
                    directive += line+"\n"
                    line = "C$OMP& "
                line += token
            directive += line+"\n"
        code = code[:a] + directive + loop + "\nC$OMP END PARALLEL DO" + code[b:]
    assert code.count("C$OMP PARALLEL DO DEFAULT(NONE)") == 2
    return code
