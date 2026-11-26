#!/usr/bin/env python3
"""Independent Mie amplitudes + direct angular-spectrum sum at selected pixels.
Uses only Python's standard library. Run after the documented validation case.
"""
import cmath as cm
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load(name):
    p = ROOT / "run" / ("run_" + name)
    d = dict(line.split("=", 1) for line in (p / "log.txt").read_text().splitlines() if "=" in line)
    assert d["status"] == "complete", d
    nx, ny = int(d["nx"]), int(d["ny"])
    raw = (p / "hologram.raw").read_bytes()
    assert len(raw) == 4 * nx * ny
    data = struct.unpack("<" + str(nx * ny) + "f", raw)
    assert all(math.isfinite(v) and v >= 0 for v in data)
    return d, data

def riccati(z, n):
    p = [cm.sin(z), cm.sin(z)/z-cm.cos(z)]
    q = [cm.cos(z), cm.cos(z)/z+cm.sin(z)]
    for j in range(1, n):
        p.append((2*j+1)*p[j]/z-p[j-1])
        q.append((2*j+1)*q[j]/z-q[j-1])
    return p, [a-1j*b for a,b in zip(p,q)]

def mie(x, m, n):
    p, h = riccati(x,n)
    t, _ = riccati(m*x,n)
    coeff = []
    for j in range(1,n+1):
        dp=p[j-1]-j*p[j]/x
        dh=h[j-1]-j*h[j]/x
        dt=t[j-1]-j*t[j]/(m*x)
        a=(m*t[j]*dp-p[j]*dt)/(m*t[j]*dh-h[j]*dt)
        b=(t[j]*dp-m*p[j]*dt)/(t[j]*dh-m*h[j]*dt)
        coeff.append((a,b))
    return coeff

def amplitudes(coeff, u):
    prev, now = 0., 1.
    s1=s2=0j
    for j,(a,b) in enumerate(coeff,1):
        tau=j*u*now-(j+1)*prev
        f=(2*j+1)/(j*(j+1))
        s1+=f*(a*now+b*tau)
        s2+=f*(a*tau+b*now)
        prev,now=now,((2*j+1)*u*now-(j+1)*prev)/j
    return s1,s2

def check_mie(name="test_mie"):
    d,data=load(name)
    nx,ny,fx,fy=(int(d[k]) for k in ("nx","ny","fft_nx","fft_ny"))
    dx,dy,z,lam,na,l0,r=(float(d[k]) for k in ("dx_um","dy_um","z_um","lambda_medium_um","NA","lambda_vacuum_um","diameter_um"))
    k=2*math.pi/lam
    kc=2*math.pi*na/l0
    lx,ly=fx*dx,fy*dy
    coeff=mie(k*r/2,complex(float(d["n_real"]),float(d["n_imag"]))/float(d["n_medium"]),int(d["NMAX"]))
    ext=2*math.pi/k**2*sum((2*j+1)*(a+b).real for j,(a,b) in enumerate(coeff,1))
    sca=2*math.pi/k**2*sum((2*j+1)*(abs(a)**2+abs(b)**2) for j,(a,b) in enumerate(coeff,1))
    for label,v in (("Cext_um2",ext),("Csca_um2",sca),("Cabs_um2",ext-sca)):
        rel=abs(v-float(d[label]))/v
        print(label,"Mie",v,"relative error",rel)
        assert rel<2e-6
    terms=[]
    for j in range(fy):
        ky=2*math.pi*(j if j<=(fy-1)//2 else j-fy)/ly
        for i in range(fx):
            kx=2*math.pi*(i if i<=(fx-1)//2 else i-fx)/lx
            kr=math.hypot(kx,ky)
            if kr>kc: continue
            kz=math.sqrt(k*k-kr*kr)
            u=kz/k
            p=math.atan2(ky,kx)
            cp,sp=math.cos(p),math.sin(p)
            s1,s2=amplitudes(coeff,u)
            et,ep=1j*s2*cp/k,-1j*s1*sp/k
            f=2j*math.pi/(kz*lx*ly)*cm.exp(1j*(kz-k)*z)
            terms.append((kx,ky,f*(u*cp*et-sp*ep),f*(u*sp*et+cp*ep)))
    worst=0
    for ix,iy in ((nx//2,ny//2),(nx//2+4,ny//2),(nx//2,ny//2+5),(3,7),(nx-4,ny-5)):
        x,y=(ix-nx//2)*dx,(iy-ny//2)*dy
        ex,ey=1+0j,0j
        for kx,ky,ax,ay in terms:
            f=cm.exp(1j*(kx*x+ky*y))
            ex+=ax*f
            ey+=ay*f
        ref=abs(ex)**2+abs(ey)**2
        err=abs(ref-data[iy*nx+ix])
        worst=max(worst,err)
        print("pixel",ix,iy,"Mie",ref,"T-matrix",data[iy*nx+ix],"absolute error",err)
    print("Maximum selected-pixel intensity error:",worst)
    assert worst<2e-6

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        ROOT = Path(sys.argv[1]).resolve()
    d,data=load("test_no_contrast")
    assert set(data)=={1.0}
    assert all(float(d[k])==0 for k in ("Csca_um2","Cext_um2","Cabs_um2"))
    print("No contrast: exact unit intensity, zero cross sections")
    check_mie()
    print("PASS")
