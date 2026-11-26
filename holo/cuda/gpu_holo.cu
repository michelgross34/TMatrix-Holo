#include "gpu_holo.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#define HD __host__ __device__
namespace {
constexpr double pi=3.141592653589793238462643383279502884;
constexpr double rad=pi/180.;
struct Z{
 double re,im;
 HD Z(double a=0,double b=0):re(a),im(b){}
 HD Z operator+(Z b)const{return {re+b.re,im+b.im};}
 HD Z operator-(Z b)const{return {re-b.re,im-b.im};}
 HD Z operator*(double b)const{return {re*b,im*b};}
 HD Z operator*(Z b)const{return {re*b.re-im*b.im,re*b.im+im*b.re};}
 HD Z& operator+=(Z b){re+=b.re;im+=b.im;return *this;}
 HD double norm()const{return re*re+im*im;}
};
struct V{double x,y,z;};
HD double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
struct Weight{Z a,b,c,d;};
struct Frame{double ct,phi,r00,r01,r10,r11;};
struct State{int n;double lam,alpha,ca,sa,cb,sb,inc_phi,ix,iy,px,py;};
HD V rotate(const State&s,V v){
 return {s.ca*s.cb*v.x+s.sa*s.cb*v.y-s.sb*v.z,
         -s.sa*v.x+s.ca*v.y,s.ca*s.sb*v.x+s.sa*s.sb*v.y+s.cb*v.z};
}
// Match the small angular displacements and rotated bases of NASA AMPL.
HD Frame frame(const State&s,double theta,double phi){
 double t=theta*rad,p=phi*rad;
 constexpr double eps=1e-7;
 if(t<pi/2)t+=eps;else if(t>pi/2)t-=eps;
 if(p<pi)p+=eps;else if(p>pi)p-=eps;
 double ct=cos(t),st=sin(t),cp=cos(p),sp=sin(p);
 double ca=cos(p-s.alpha),sa=sin(p-s.alpha);
 double c=fmax(-1.,fmin(1.,ct*s.cb+st*s.sb*ca));
 double ph=atan2(st*sa,s.cb*st*ca-s.sb*ct);
 if(ph<0)ph+=2*pi;
 double cp1=cos(ph),sp1=sin(ph),st1=sin(acos(c));
 V bt={c*cp1,c*sp1,-st1},bp={-sp1,cp1,0};
 V lt=rotate(s,{ct*cp,ct*sp,-st}),lp=rotate(s,{-sp,cp,0});
 return {c,ph,dot(bt,lt),dot(bt,lp),dot(bp,lt),dot(bp,lp)};
}
// Streaming equivalent of VIGAMPL: same pole limit and normalized recurrence.
struct Vig{
 double x,qs,inv,prev,cur;
 int m;bool pole;
 HD Vig(double v,int mm):x(v),m(mm){
  pole=fabs(1-fabs(x))<=1e-10;
  qs=pole?0:sqrt(fmax(0.,1-x*x));inv=pole?0:1/qs;
  prev=m?0:1;cur=m?1:x;
  if(m&&!pole)for(int j=1;j<=m;++j)cur*=sqrt(double(2*j-1)/(2*j))*qs;
 }
 HD void next(int n,double&v1,double&v2){
  if(pole){
   v1=v2=0;
   if(m==1){double d=.5*sqrt(double(n*(n+1)));
    if(x<0&&!(n&1))d=-d;
    v1=d;v2=x<0?-d:d;}
   return;
  }
  double q=n,q1=n+1,q2=2*n+1,d3,der;
  if(m==0){d3=(q2*x*cur-q*prev)/q1;der=inv*(q1*q/q2)*(-prev+d3);}
  else{double a=sqrt(q*q-m*m),b=sqrt(q1*q1-m*m);
   d3=(q2*x*cur-a*prev)/b;der=inv*(-q1*a*prev+q*b*d3)/q2;}
  v1=cur*inv;v2=der;prev=cur;cur=d3;
 }
};
HD void amplitude(State s,const Weight* ww,double th,double ph,Z&out1,Z&out2){
 Frame f=frame(s,th,ph);
 Z v,h;
 for(int m=0;m<=s.n;++m){
  Vig g(f.ct,m);
  double fc=2*cos(m*(f.phi-s.inc_phi)),fs=2*sin(m*(f.phi-s.inc_phi));
  for(int n=m>0?m:1;n<=s.n;++n){
   double d1,d2;g.next(n,d1,d2);d1*=m;
   const Weight w=ww[m*s.n+n-1];
   if(m==0){v+=w.b*(d2*s.ix);h+=w.c*(d2*s.iy);}
   else{
    v+=(w.a*d1+w.b*d2)*(fc*s.ix)+(w.c*d1+w.d*d2)*(fs*s.iy);
    h+=(w.a*d2+w.b*d1)*(-fs*s.ix)+(w.c*d2+w.d*d1)*(fc*s.iy);
   }
  }
 }
 double factor=s.lam/(2*pi)/(f.r00*f.r11-f.r01*f.r10);
 out1=(v*f.r11-h*f.r01)*factor;
 out2=(h*f.r00-v*f.r10)*factor;
}
void check(cudaError_t e,const char*where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
void fftcheck(cufftResult e,const char*where){if(e!=CUFFT_SUCCESS)throw std::runtime_error(std::string(where)+": cuFFT "+std::to_string(int(e)));}
template<class T>struct Buffer{
 T* p=nullptr;
 Buffer()=default;
 explicit Buffer(size_t n){alloc(n);}
 ~Buffer(){if(p)cudaFree(p);}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
 void alloc(size_t n){if(p){check(cudaFree(p),"cudaFree");p=nullptr;}if(n)check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"cudaMalloc");}
 void put(const T*q,size_t n){if(n)check(cudaMemcpy(p,q,n*sizeof(T),cudaMemcpyHostToDevice),"upload");}
 void get(T*q,size_t n){if(n)check(cudaMemcpy(q,p,n*sizeof(T),cudaMemcpyDeviceToHost),"download");}
};
struct Plan{cufftHandle p=0;~Plan(){if(p)cufftDestroy(p);}};
State state{};
Buffer<Weight> weights;
__global__ void section_kernel(State s,const Weight*ww,const double*mu,const double*w,int nt,int np,double*res){
 int j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j>nt*np)return;
 Z a,b;
 if(j==nt*np){amplitude(s,ww,0,0,a,b);res[j]=2*s.lam*(s.px*a.im+s.py*b.im);}
 else{int k=j/np,l=j%np;
  amplitude(s,ww,acos(fmax(-1.,fmin(1.,mu[k])))/rad,(l+.5)*360/np,a,b);
  res[j]=w[k]*(2*pi/np)*(a.norm()+b.norm());
 }
}
__global__ void pupil_kernel(State s,const Weight*ww,cufftDoubleComplex*f,int fx,int fy,
 double dx,double dy,double z,double x,double y,double cut,int nc,unsigned int*count){
 int u=blockIdx.x*blockDim.x+threadIdx.x,N=fx*fy;
 if(u>=N)return;
 int j=u/fx,i=u%fx;
 double Lx=fx*dx,Ly=fy*dy,k=2*pi/s.lam;
 double kx=2*pi*(i<=(fx-1)/2?i:i-fx)/Lx,ky=2*pi*(j<=(fy-1)/2?j:j-fy)/Ly;
 double kr=hypot(kx,ky);
 Z ex,ey,ez;
 if(s.n&&kr<=cut){
  double kz=sqrt(k*k-kr*kr),ct=kz/k,st=kr/k,p=kr==0?0:atan2(ky,kx),cp=cos(p),sp=sin(p);
  Z a,b;amplitude(s,ww,atan2(kr,kz)/rad,p<0?p/rad+360:p/rad,a,b);
  double phase=(kz-k)*z-kx*x-ky*y,scale=2*pi/(kz*Lx*Ly);
  Z v(-scale*sin(phase),scale*cos(phase));
  ex=v*(a*(ct*cp)-b*sp);ey=v*(a*(ct*sp)+b*cp);ez=v*(a*(-st));
  atomicAdd(count,1u);
 }
 f[u]=make_cuDoubleComplex(ex.re,ex.im);
 f[N+u]=make_cuDoubleComplex(ey.re,ey.im);
 if(nc==3)f[2*N+u]=make_cuDoubleComplex(ez.re,ez.im);
}
__global__ void crop_kernel(State s,const cufftDoubleComplex*f,float*out,int nx,int ny,int fx,int fy,int nc){
 int u=blockIdx.x*blockDim.x+threadIdx.x;
 if(u>=nx*ny)return;
 int x=u%nx,y=u/nx,N=fx*fy;
 int k=((y-ny/2+fy)%fy)*fx+(x-nx/2+fx)%fx;
 Z a(f[k].x+s.px,f[k].y),b(f[N+k].x+s.py,f[N+k].y);
 double I=a.norm()+b.norm();
 if(nc==3)I+=Z(f[2*N+k].x,f[2*N+k].y).norm();
 out[u]=float(I);
}
}
void gpu_prepare(int n,double lam,double alpha,double beta,double px,double py,const double*t){
 double b=beta*rad;
 if(b<=pi/2&&pi/2-b<=1e-7)b-=1e-7;
 else if(b>pi/2&&b-pi/2<=1e-7)b+=1e-7;
 state={n,lam,alpha*rad,cos(alpha*rad),sin(alpha*rad),cos(b),sin(b),0,0,0,px,py};
 if(!n)return;
 Frame f=frame(state,0,0);
 state.inc_phi=f.phi;state.ix=f.r00*px+f.r01*py;state.iy=f.r10*px+f.r11*py;
 std::vector<Weight> w(size_t(n+1)*n);
 for(int m=0;m<=n;++m){
  Vig g(f.ct,m);
  for(int nn=m?m:1;nn<=n;++nn){
   double d1,d2;g.next(nn,d1,d2);d1*=m;
   for(int k=m?m:1;k<=n;++k){
    const double*p=t+8*(size_t(m)*n*n+size_t(nn-1)*n+k-1);
    Z t11(p[0],p[1]),t12(p[2],p[3]),t21(p[4],p[5]),t22(p[6],p[7]);
    double r=sqrt(double((2*k+1)*(2*nn+1))/double(k*nn*(k+1)*(nn+1)));
    int phase=(nn-k-1)%4;if(phase<0)phase+=4;
    Z cal=phase==0?Z(r,0):phase==1?Z(0,r):phase==2?Z(-r,0):Z(0,-r);
    auto&v=w[m*n+k-1];
    if(m==0){v.b+=cal*t22*d2;v.c+=cal*t11*d2;}
    else{
     v.a+=cal*(t11*d1+t12*d2);v.b+=cal*(t21*d1+t22*d2);
     v.c+=cal*(t11*d2+t12*d1);v.d+=cal*(t21*d2+t22*d1);
    }
   }
  }
 }
 weights.alloc(w.size());weights.put(w.data(),w.size());
}
std::array<double,3> gpu_sections(const double*mu,const double*w,int nt,int np){
 if(!state.n)return {};
 int n=nt*np+1;
 Buffer<double> dmu(nt),dw(nt),dr(n);dmu.put(mu,nt);dw.put(w,nt);
 section_kernel<<<(n+127)/128,128>>>(state,weights.p,dmu.p,dw.p,nt,np,dr.p);
 check(cudaGetLastError(),"section kernel");
 std::vector<double> r(n);dr.get(r.data(),n);
 double sca=0;for(int j=0;j<n-1;++j)sca+=r[j];
 return {r.back(),sca,r.back()-sca};
}
std::vector<float> gpu_image(int nx,int ny,int pad,double dx,double dy,double z,double x,double y,double cut,int nc,std::size_t&count){
 int fx=nx*pad,fy=ny*pad,N=fx*fy;
 std::vector<float> out(size_t(nx)*ny);
 if(!state.n){std::fill(out.begin(),out.end(),1.f);count=0;return out;}
 Buffer<cufftDoubleComplex> f(size_t(nc)*N);Buffer<float> image(out.size());Buffer<unsigned int> c(1);
 check(cudaMemset(c.p,0,sizeof(unsigned int)),"counter reset");
 pupil_kernel<<<(N+127)/128,128>>>(state,weights.p,f.p,fx,fy,dx,dy,z,x,y,cut,nc,c.p);
 check(cudaGetLastError(),"pupil kernel");
 Plan plan;fftcheck(cufftPlan2d(&plan.p,fy,fx,CUFFT_Z2Z),"cuFFT plan");
 for(int j=0;j<nc;++j)fftcheck(cufftExecZ2Z(plan.p,f.p+size_t(j)*N,f.p+size_t(j)*N,CUFFT_INVERSE),"cuFFT inverse");
 crop_kernel<<<(out.size()+127)/128,128>>>(state,f.p,image.p,nx,ny,fx,fy,nc);
 check(cudaGetLastError(),"crop kernel");
 image.get(out.data(),out.size());
 unsigned int ct=0;c.get(&ct,1);count=ct;
 return out;
}
