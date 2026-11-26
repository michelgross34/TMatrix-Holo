"""Generate fixed-orientation capsule programs from the prepared NASA copies."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

INPUT = """
      IF (COMMAND_ARGUMENT_COUNT().LT.6.OR.
     &    COMMAND_ARGUMENT_COUNT().GT.8) THEN
         WRITE(*,*) 'Usage: mish_capsule_{precision} lambda_nm radius_nm',
     &      ' straight_length_nm mr mi theta_deg [tol [ndgs]]'
         STOP 2
      ENDIF
      DO IARG=1,COMMAND_ARGUMENT_COUNT()
         CALL GET_COMMAND_ARGUMENT(IARG,ARG)
         SELECT CASE (IARG)
         CASE(1)
            READ(ARG,*,IOSTAT=IOS) WAVELENGTH_NM
         CASE(2)
            READ(ARG,*,IOSTAT=IOS) RADIUS_NM
         CASE(3)
            READ(ARG,*,IOSTAT=IOS) STRAIGHT_NM
         CASE(4)
            READ(ARG,*,IOSTAT=IOS) MRR
         CASE(5)
            READ(ARG,*,IOSTAT=IOS) MRI
         CASE(6)
            READ(ARG,*,IOSTAT=IOS) THET0
         CASE(7)
            READ(ARG,*,IOSTAT=IOS) DDELT
         CASE(8)
            READ(ARG,*,IOSTAT=IOS) NDGS
         END SELECT
         IF (IOS.NE.0) STOP 2
      ENDDO
      IF (.NOT.ALL(IEEE_IS_FINITE([WAVELENGTH_NM,RADIUS_NM,
     &     STRAIGHT_NM,THET0,DDELT,DBLE(MRR),DBLE(MRI)]))) STOP 2
      IF (MIN(WAVELENGTH_NM,RADIUS_NM,DBLE(MRR),DDELT).LE.0D0
     &    .OR.MIN(STRAIGHT_NM,DBLE(MRI),THET0).LT.0D0
     &    .OR.THET0.GT.180D0.OR.NDGS.LT.2.OR.NDGS.GT.10) STOP 2
      IF (MRR.EQ.1.AND.MRI.EQ.0) THEN
         WRITE(*,9101) 'Etheta',0D0,0D0,0D0
         WRITE(*,9101) 'Ephi',0D0,0D0,0D0
         STOP
      ENDIF
C     EPS is the ratio of the straight half-length to the radius.
      NP=-4
      RAT=1D0
      EPS=REAL(STRAIGHT_NM,KIND(EPS))/(2*RADIUS_NM)
      LAM=1
      AXI=REAL(RADIUS_NM,KIND(EPS))/WAVELENGTH_NM*
     &    (1+1.5D0*EPS)**(1/REAL(3,KIND(EPS)))
"""

SECTIONS = """
C     The angular extent and azimuthal sampling follow the multipole order.
      NTH=NMAX+2
      NPH=4*NMAX+4
      CALL GAUSS(NTH,0,0,GMU,GW)
      QSCTH=0D0
      QSCPH=0D0
      DO I=1,NTH
         THET=DACOS(GMU(I))*180D0/DACOS(-1D0)
         WGHT=GW(I)*2D0*DACOS(-1D0)/NPH
         DO J=1,NPH
            PHI=(J-0.5D0)*360D0/NPH
            CALL AMPL(NMAX,{lam},THET0,THET,0D0,PHI,0D0,0D0,
     &                S11,S12,S21,S22)
            QSCTH=QSCTH+WGHT*(ABS(S11)**2+ABS(S21)**2)
            QSCPH=QSCPH+WGHT*(ABS(S12)**2+ABS(S22)**2)
         ENDDO
      ENDDO
      CALL AMPL(NMAX,{lam},THET0,THET0,0D0,0D0,0D0,0D0,
     &          S11,S12,S21,S22)
      QEXTH=2D0*AIMAG(S11)
      QEXPH=2D0*AIMAG(S22)
      SCALE=(WAVELENGTH_NM/1000D0)**2
      IF (.NOT.ALL(IEEE_IS_FINITE([QEXTH,QEXPH,QSCTH,QSCPH]))
     &    .OR.MIN(QEXTH,QEXPH,QSCTH,QSCPH).LE.0D0) STOP 3
      WRITE(*,9100) NMAX,NGAUSS,NDGS
      WRITE(*,9101) 'Etheta',QEXTH*SCALE,QSCTH*SCALE,
     &              (QEXTH-QSCTH)*SCALE
      WRITE(*,9101) 'Ephi',QEXPH*SCALE,QSCPH*SCALE,
     &              (QEXPH-QSCPH)*SCALE
 9100 FORMAT('NMAX=',I4,' NGAUSS=',I4,' NDGS=',I3)
 9101 FORMAT(A,' Cext=',ES24.15,' Csca=',ES24.15,
     &       ' Cabs=',ES24.15)
      STOP

"""

PROFILE = """
C     Capsule profile in one hemisphere. DR = d(log(radius))/d(theta).
      SUBROUTINE RSPCAP(X,NG,NGAUSS,REV,EPS,R,DR)
      IMPLICIT REAL*{size} (A-H,O-Z)
      REAL*{size} X(NG),R(NG),DR(NG)
      RAD=REV/(1+1.5D0*EPS)**(1/REAL(3,KIND(EPS)))
      H=RAD*EPS
      XJOIN=EPS/SQRT(1+EPS*EPS)
      DO I=1,NGAUSS
         CO=-X(I)
         SI=SQRT((1-CO)*(1+CO))
         IF (CO.GT.XJOIN) THEN
            ROOT=SQRT(MAX(RAD*RAD-(H*SI)**2,REAL(0,KIND(EPS))))
            RI=H*CO+ROOT
            DLOG=-H*SI/ROOT
         ELSE
            RI=RAD/SI
            DLOG=-CO/SI
         ENDIF
         R(I)=RI**2
         R(NG-I+1)=R(I)
         DR(I)=-DLOG
         DR(NG-I+1)=DLOG
      ENDDO
      END
"""

def replace_once(code, old, new):
    if code.count(old) != 1:
        raise RuntimeError(f"Unexpected NASA source at {old!r}")
    return code.replace(old, new, 1)

def prepare(precision, suffix, size, lam):
    directory = ROOT / "build_src" / f"fixed_{precision}"
    code = (directory / f"ampl{suffix}.lp.f").read_text(encoding="ascii")
    start = code.index("      IMPLICIT REAL*8 (A-H,O-Z)")
    title = ("      PROGRAM MISH_CAPSULE\n"
             "      USE, INTRINSIC :: IEEE_ARITHMETIC\n")
    code = code[:start] + title + code[start:]
    code = replace_once(code, "      OPEN (6,FILE='test')", "")
    marker = "      COMPLEX*16 S11,S12,S21,S22"
    main_end = code.index("      SUBROUTINE AMPL ")
    main, routines = code[:main_end], code[main_end:]
    main = replace_once(main, marker, marker + """
      CHARACTER*128 ARG
      REAL*8 GMU(NPN1+2),GW(NPN1+2),WAVELENGTH_NM,RADIUS_NM,
     &       STRAIGHT_NM
      REAL*{size} AXI
""".format(size=size))
    main = replace_once(main, "      NDGS=2", "      NDGS=3\n" + INPUT.format(precision=precision))
    main = replace_once(main, "      IF (NP.EQ.-1.OR.NP.EQ.-2) NCHECK=1",
                        "      IF (NP.EQ.-1.OR.NP.EQ.-2.OR.NP.EQ.-4) NCHECK=1")
    for old in ("IF (INM1.GE.NPN1) STOP", "IF (NGAUSS.GT.NPNG1) STOP", "IF (NMA.EQ.NPN1) STOP"):
        main = replace_once(main, old, old + " 3")
    main = replace_once(main, "      IF (NGAUSS.EQ.NPNG1) GO TO 155",
                        "      IF (NGAUSS.EQ.NPNG1) STOP 3")
    main = replace_once(main, "  150 CONTINUE", "  150 CONTINUE\n      STOP 3")
    output = main.index("C  COMPUTATION OF THE AMPLITUDE AND PHASE MATRICES", main.index("  155 CONTINUE"))
    main = main[:output] + SECTIONS.format(lam=lam) + "      END\n"
    routines = re.sub(r"(?mi)^      (WRITE\s*\(6,100[56]\).*|PRINT\s+110[1-4].*)$",
                      r"C     \1", routines)
    routines = replace_once(routines, "      IF (NP.EQ.-2) GO  TO 11",
                            "      IF (NP.EQ.-2.OR.(NP.EQ.-4.AND.EPS.GT.0)) GO TO 11")
    gauss_line = "      CALL " + ("QGAUSS" if precision == "qp" else "GAUSS") + "(NG1,0,0,X1,W1)"
    routines = replace_once(routines, gauss_line,
                            "      IF (NP.EQ.-4) XX=-EPS/SQRT(1+EPS*EPS)\n" + gauss_line)
    hook = "      IF (NP.EQ.-2) CALL RSP3(X,NG,NGAUSS,A,EPS,R,DR)"
    routines = replace_once(routines, hook,
                            hook + "\n      IF (NP.EQ.-4) CALL RSPCAP(X,NG,NGAUSS,A,EPS,R,DR)")
    # Store the T-matrix in double precision for both the DP and QP cores.
    main = main.replace("      REAL*4\n", "      REAL*8\n")
    routines = routines.replace("      REAL*4\n", "      REAL*8\n")
    (directory / "capsule.f").write_text(main + routines + PROFILE.format(size=size), encoding="ascii")

if __name__ == "__main__":
    prepare("dp", "d", 8, "LAM")
    prepare("qp", "q", 16, "DLAM")
