#!/usr/bin/env python3
"""Reuse the prepared NASA DP solver and capsule profile, without editing originals."""
from pathlib import Path
import sys
from tmatrix_openmp import parallelize
root = Path(__file__).resolve().parent.parent / "mishenko"
source = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "build_src/fixed_dp/capsule.f"
src = source.read_text(encoding="ascii")
main, routines = src.split("      SUBROUTINE AMPL ", 1)
main = main.replace("      PROGRAM MISH_CAPSULE", """      SUBROUTINE SOLVE_HOLO(NP,AXI,EPS,LAM,MRR,MRI,
     & DDELT,NDGS,NMAX,NGAUSS)""", 1)
a = main.index("      AXI=10D0")
b = main.index("      P=DACOS(-1D0)", a)
main = main[:a] + "      RAT=1D0\n" + main[b:]
main = main[:main.index("C     The angular extent")] 
main = main.replace("STOP 3", "GO TO 999")
main += """      RETURN
  999 NMAX=-1
      RETURN
      END
"""
routines = "      SUBROUTINE AMPL " + routines
# Avoid undefined azimuth at quadrants and roundoff in acos in the rotated frame.
for s in ("", "1"):
    old = f"      THETP{s}=DACOS(CTP{s})"
    assert old in routines
    routines = routines.replace(old, f"      THETP{s}=DACOS(MAX(-1D0,MIN(1D0,CTP{s})))", 1)
    start = routines.index(f"      PHIP{s}=DATAN(SPP{s}/CPP{s})")
    end = routines.index("\n", routines.index(f"      IF (PHIP{s}.LT.0D0) PHIP{s}=PHIP{s}+2D0*PIN",start))
    routines = routines[:start] + f"      PHIP{s}=MODULO(DATAN2(SPP{s},CPP{s}),2D0*PIN)" + routines[end:]
bridge = """
      SUBROUTINE HOLO_AMPL(NMAX,LAM,TH,PH,ALPHA,BETA,S)
      IMPLICIT NONE
      INTEGER NMAX
      REAL*8 LAM,TH,PH,ALPHA,BETA,S(8)
      COMPLEX*16 VV,VH,HV,HH
      CALL AMPL(NMAX,LAM,0D0,TH,0D0,PH,ALPHA,BETA,
     & VV,VH,HV,HH)
      S(1)=DBLE(VV)
      S(2)=DIMAG(VV)
      S(3)=DBLE(VH)
      S(4)=DIMAG(VH)
      S(5)=DBLE(HV)
      S(6)=DIMAG(HV)
      S(7)=DBLE(HH)
      S(8)=DIMAG(HH)
      END
      SUBROUTINE HOLO_PACK_TMAT(NMAX,W)
      IMPLICIT NONE
      INTEGER NPN1,NPNG1,NPNG2,NPN2,NPL,NPN3,NPN4,NPN5,NPN6
      INCLUDE 'ampld.par.f'
      INTEGER NMAX,M,N,NN
      REAL*8 W(8,NMAX,NMAX,0:NMAX)
      REAL*8 TR11(NPN6,NPN4,NPN4),TR12(NPN6,NPN4,NPN4),
     & TR21(NPN6,NPN4,NPN4),TR22(NPN6,NPN4,NPN4),
     & TI11(NPN6,NPN4,NPN4),TI12(NPN6,NPN4,NPN4),
     & TI21(NPN6,NPN4,NPN4),TI22(NPN6,NPN4,NPN4)
      COMMON /TMAT/ TR11,TR12,TR21,TR22,TI11,TI12,TI21,TI22
      W=0D0
      DO M=0,NMAX
        DO NN=MAX(M,1),NMAX
          DO N=MAX(M,1),NMAX
            W(1,N,NN,M)=TR11(M+1,N,NN)
            W(2,N,NN,M)=TI11(M+1,N,NN)
            W(3,N,NN,M)=TR12(M+1,N,NN)
            W(4,N,NN,M)=TI12(M+1,N,NN)
            W(5,N,NN,M)=TR21(M+1,N,NN)
            W(6,N,NN,M)=TI21(M+1,N,NN)
            W(7,N,NN,M)=TR22(M+1,N,NN)
            W(8,N,NN,M)=TI22(M+1,N,NN)
          ENDDO
        ENDDO
      ENDDO
      END
"""
Path(sys.argv[1]).write_text(parallelize(main+routines+bridge), encoding="ascii")
