// Fixed-orientation NASA Mishchenko T-matrix + vector angular spectrum.
// All physical lengths in micrometres; e^{-i omega t} convention.
#ifndef HOLO_CUDA
#include <fftw3.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef HOLO_CUDA
#include "cuda/gpu_holo.hpp"
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace fs=std::filesystem;
using C=std::complex<double>;
const double pi=std::acos(-1.0), rad=pi/180;
extern "C" {
void solve_holo_(int*,double*,double*,double*,double*,double*,double*,int*,int*,int*);
void solve_holo_qp_(int*,double*,double*,double*,double*,double*,double*,int*,int*,int*);
void holo_ampl_(int*,double*,double*,double*,double*,double*,double*);
void gauss_(int*,int*,int*,double*,double*);
#ifdef HOLO_OPENMP
void holo_set_threads(int,int*);
#endif
#ifdef HOLO_CUDA
void holo_pack_tmat_(int*,double*);
#endif
}
struct Args {
 std::string shape="spheroid",name="",detector="transverse",precision="auto";
 double lambda=.660,length=8,diameter=2.5,nr=1.38,ni=0,nmed=1.33,theta=27,phi=48,z=100,na=.8,dx=.1,dy=.1,x=0,y=0,pol=0,tol=1e-5;
 int nx=1024,ny=1024,pad=3,ndgs=3,theta_steps=0,threads=0;
 double theta_end=180;
};
void help(){
 std::cout<<"mish_holo [--name ID] [--shape spheroid|capsule|cylinder] [--lambda-um .660]\n"
 <<" [--length-um 8] [--diameter-um 2.5] [--n-real 1.38] [--n-imag 0] [--n-medium 1.33]\n"
 <<" [--theta 27] [--phi 48] [--pol-deg 0] [--x-um 0] [--y-um 0] [--z-um 100] [--na .8]\n"
 <<" [--theta-steps N]  Sweep theta=0..180 inclusive in N equal intervals; phi=0 (N+1 images).\n"
 <<" [--theta-end 180]  Last angle of sweep, inclusive.\n"
 <<" [--nx 1024] [--ny 1024] [--dx-um .1] [--dy-um .1] [--pad 3] [--tol 1e-5] [--ndgs 3]\n"
 <<" [--tmatrix-precision auto|dp|qp]  auto tries DP then QP if needed.\n"
 <<" [--threads N]  OpenMP threads for T-matrix; default up to 8 or OMP_NUM_THREADS.\n"
 <<" [--detector transverse|total]\n"
 <<"Angles in degrees; indices absolute; z measured from particle centre; output <executable directory>/run/run_ID.\n";
}
double number(const std::string &s){size_t n=0; double v=std::stod(s,&n);if(n!=s.size()||!std::isfinite(v))throw std::runtime_error("Invalid number: "+s);return v;}
int integer(const std::string&s){double v=number(s);if(v!=std::floor(v)||v<1||v>32768)throw std::runtime_error("Invalid integer: "+s);return int(v);}
Args parse(int argc,char**argv){
 Args a;
 std::map<std::string,double*> d={{"--lambda-um",&a.lambda},{"--length-um",&a.length},{"--diameter-um",&a.diameter},{"--n-real",&a.nr},{"--n-imag",&a.ni},{"--n-medium",&a.nmed},{"--theta",&a.theta},{"--phi",&a.phi},{"--z-um",&a.z},{"--na",&a.na},{"--dx-um",&a.dx},{"--dy-um",&a.dy},{"--x-um",&a.x},{"--y-um",&a.y},{"--pol-deg",&a.pol},{"--tol",&a.tol}};
 std::map<std::string,int*> i={{"--nx",&a.nx},{"--ny",&a.ny},{"--pad",&a.pad},{"--ndgs",&a.ndgs},{"--theta-steps",&a.theta_steps}};
 std::map<std::string,std::string*> s={{"--shape",&a.shape},{"--name",&a.name},{"--detector",&a.detector},{"--tmatrix-precision",&a.precision}};
 d["--theta-end"]=&a.theta_end;
 i["--threads"]=&a.threads;
 for(int j=1;j<argc;j++){std::string k=argv[j];if(j+1>=argc)throw std::runtime_error("Missing value for "+k);std::string v=argv[++j];if(d.count(k))*d[k]=number(v);else if(i.count(k))*i[k]=integer(v);else if(s.count(k))*s[k]=v;else throw std::runtime_error("Unknown option: "+k);}
 if(a.lambda<=0||a.length<=0||a.diameter<=0||a.nr<=0||a.ni<0||a.nmed<=0||a.na<=0||a.na>=a.nmed||a.dx<=0||a.dy<=0||a.theta<0||a.theta>180||a.phi<0||a.phi>360||a.tol<=0||a.tol>1e-3||a.ndgs<2||a.ndgs>10||a.pad>16)throw std::runtime_error("Invalid physical/grid parameters (require 0 < NA < n_medium, tol <= 1e-3).");
 if(a.shape!="spheroid"&&a.shape!="capsule"&&a.shape!="cylinder")throw std::runtime_error("Unknown shape");
 if(a.shape=="capsule"&&a.length<a.diameter)throw std::runtime_error("Capsule length must be >= diameter");
 if(a.detector!="transverse"&&a.detector!="total")throw std::runtime_error("Unknown detector");
 if(a.precision!="auto"&&a.precision!="dp"&&a.precision!="qp")throw std::runtime_error("Unknown T-matrix precision");
 if(a.z<=std::hypot(a.length,a.diameter)/2)throw std::runtime_error("Observation plane must be outside bounding sphere");
 if(a.dx>a.lambda/(2*a.na)||a.dy>a.lambda/(2*a.na))throw std::runtime_error("Grid undersamples complex field; reduce pixel pitch");
 if(1LL*a.nx*a.ny*a.pad*a.pad>16777216)throw std::runtime_error("FFT exceeds 16M cells");
 if(a.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)throw std::runtime_error("Invalid run name");
 if(a.threads>256)throw std::runtime_error("--threads must be between 1 and 256");
#ifndef HOLO_OPENMP
 if(a.threads>1)throw std::runtime_error("This executable was compiled without OpenMP");
#endif
 if(a.theta_end<=0||a.theta_end>180||(!a.theta_steps&&a.theta_end!=180))throw std::runtime_error("Invalid theta-end (requires a sweep, 0 < end <= 180)");
 if(a.theta_steps){a.theta=0;a.phi=0;}
 return a;
}
struct Solver {
 int n=0,ng=0; double lam,alpha,beta,px,py;
 std::string precision="none";
#ifdef HOLO_CUDA
 std::vector<double> packed;
#endif
 explicit Solver(const Args&a):lam(a.lambda/a.nmed),alpha(a.phi),beta(a.theta),px(std::cos(a.pol*rad)),py(std::sin(a.pol*rad)){
  if(a.nr==a.nmed&&a.ni==0)return;
  int shape=a.shape=="spheroid"?-1:(a.shape=="capsule"?-4:-2),ndgs=a.ndgs;
  double r=a.diameter/2,h=a.length/2,eps=r/h,axi=std::cbrt(r*r*h);
  if(shape==-1&&eps==1)eps=1.000001; // NASA recommended near-sphere limit
  if(shape==-2)axi=std::cbrt(1.5*r*r*h);
  if(shape==-4){eps=(h-r)/r;axi=r*std::cbrt(1+1.5*eps);}
  double mr=a.nr/a.nmed,mi=a.ni/a.nmed,tol=a.tol;
  if(a.precision!="qp"){
   solve_holo_(&shape,&axi,&eps,&lam,&mr,&mi,&tol,&ndgs,&n,&ng);
   precision="dp";
  }
  if(n<=0&&a.precision!="dp"){
   if(a.precision=="auto")std::cout<<"DP did not converge; retrying with quadruple precision."<<std::endl;
   tol=a.tol; n=ng=0;
   solve_holo_qp_(&shape,&axi,&eps,&lam,&mr,&mi,&tol,&ndgs,&n,&ng);
   precision="qp";
  }
  if(n<=0)throw std::runtime_error("T-matrix did not converge");
#ifdef HOLO_CUDA
  packed.resize(8ULL*(n+1)*n*n);
  holo_pack_tmat_(&n,packed.data());
#endif
 }
 void set_orientation(double theta,double phi){beta=theta;alpha=phi;
#ifdef HOLO_CUDA
  gpu_prepare(n,lam,alpha,beta,px,py,packed.data());
#endif
 }
 std::array<C,2> amp(double th,double ph){
  if(n==0)return {};
  double s[8]; holo_ampl_(&n,&lam,&th,&ph,&alpha,&beta,s);
  C a=C(s[0],s[1])*px+C(s[2],s[3])*py,b=C(s[4],s[5])*px+C(s[6],s[7])*py;
  if(!std::isfinite(std::norm(a)+std::norm(b)))throw std::runtime_error("Nonfinite T-matrix amplitude");
  return {a,b};
 }
 std::array<double,3> sections(){
  if(!n)return {};
  int nt=n+2,np=4*n+4,zero=0;std::vector<double> mu(nt),w(nt);
  gauss_(&nt,&zero,&zero,mu.data(),w.data());
#ifdef HOLO_CUDA
  return gpu_sections(mu.data(),w.data(),nt,np);
#else
  double sca=0;
  for(int j=0;j<nt;j++)for(int l=0;l<np;l++){
   auto s=amp(std::acos(std::clamp(mu[j],-1.,1.))/rad,(l+.5)*360/np);
   sca+=w[j]*(2*pi/np)*(std::norm(s[0])+std::norm(s[1]));
  }
  auto f=amp(0,0);
  double ext=2*lam*std::imag(px*f[0]+py*f[1]);
  return {ext,sca,ext-sca};
#endif
 }
};
struct ImageStats{double lo,hi,mean;};
ImageStats write_hologram(const Args&a,Solver&s,std::ostream&out,std::ofstream&log){
 const int fx=a.nx*a.pad,fy=a.ny*a.pad;
 const double k=2*pi/s.lam,kcut=2*pi*a.na/a.lambda,Lx=fx*a.dx,Ly=fy*a.dy;
 const double radius=a.z*kcut/std::sqrt(k*k-kcut*kcut);
 log<<"fft_nx="<<fx<<"\nfft_ny="<<fy<<"\nfft_extent_x_um="<<Lx<<"\nfft_extent_y_um="<<Ly<<"\ncone_radius_um="<<radius<<"\n";
 if(Lx/2<radius+std::abs(a.x)+a.nx*a.dx/2||Ly/2<radius+std::abs(a.y)+a.ny*a.dy/2)log<<"warning=FFT periodic wrap may contaminate crop; verify by increasing --pad\n";
 if(a.dx>a.lambda/(4*a.na)||a.dy>a.lambda/(4*a.na))log<<"warning=Intensity may be undersampled; dx,dy <= lambda/(4*NA) recommended\n";
 log.flush();
 const int nc=a.detector=="total"?3:2;
#ifdef HOLO_CUDA
 size_t count=0;
 auto image=gpu_image(a.nx,a.ny,a.pad,a.dx,a.dy,a.z,a.x,a.y,kcut,nc,count);
 log<<"pupil_samples="<<count<<"\n";log.flush();
#else
 const size_t N=size_t(fx)*fy;
 std::vector<std::vector<C>> fields(nc,std::vector<C>(N));
 size_t count=0;
 if(s.n)for(int j=0;j<fy;j++){
  const double ky=2*pi*(j<=(fy-1)/2?j:j-fy)/Ly;
  for(int i=0;i<fx;i++){
   const double kx=2*pi*(i<=(fx-1)/2?i:i-fx)/Lx,kr=std::hypot(kx,ky);
   if(kr>kcut)continue;
   const double kz=std::sqrt(k*k-kr*kr),ct=kz/k,st=kr/k;
   const double p=kr==0?0:std::atan2(ky,kx),cp=std::cos(p),sp=std::sin(p);
   auto v=s.amp(std::atan2(kr,kz)/rad,p<0?p/rad+360:p/rad);
   // Weyl expansion of outgoing e^{ikr}/r, normalized by Lx*Ly.
   const C f=C(0,2*pi/(kz*Lx*Ly))*std::exp(C(0,(kz-k)*a.z-kx*a.x-ky*a.y));
   size_t u=size_t(j)*fx+i;
   fields[0][u]=f*(ct*cp*v[0]-sp*v[1]);
   fields[1][u]=f*(ct*sp*v[0]+cp*v[1]);
   if(nc==3)fields[2][u]=-f*st*v[0];
   ++count;
  }
 }
 log<<"pupil_samples="<<count<<"\n";log.flush();
 for(auto&f:fields){
  auto*p=reinterpret_cast<fftw_complex*>(f.data());
  auto plan=fftw_plan_dft_2d(fy,fx,p,p,FFTW_BACKWARD,FFTW_ESTIMATE);
  if(!plan)throw std::runtime_error("FFTW plan failed");
  fftw_execute(plan);fftw_destroy_plan(plan);
 }
#endif
 // FFTW backward is unnormalised; the Fourier quadrature factor above includes 1/(Lx Ly).
 static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
 double lo=std::numeric_limits<double>::infinity(),hi=-lo,sum=0;
 for(int y=0;y<a.ny;y++)for(int x=0;x<a.nx;x++){
#ifdef HOLO_CUDA
  double I=image[size_t(y)*a.nx+x];
#else
  size_t u=size_t((y-a.ny/2+fy)%fy)*fx+(x-a.nx/2+fx)%fx;
  double I=std::norm(fields[0][u]+s.px)+std::norm(fields[1][u]+s.py);
  if(nc==3)I+=std::norm(fields[2][u]);
#endif
  if(!std::isfinite(I)||I>std::numeric_limits<float>::max())throw std::runtime_error("Nonfinite hologram");
  float q=float(I); uint32_t bits;std::memcpy(&bits,&q,4);
  char bytes[4];for(int b=0;b<4;b++)bytes[b]=char((bits>>(8*b))&255);
  out.write(bytes,4);lo=std::min(lo,double(q));hi=std::max(hi,double(q));sum+=q;
 }
 out.flush();
 log<<"raw_min="<<lo<<"\nraw_max="<<hi<<"\nraw_mean="<<sum/(a.nx*double(a.ny))<<"\nraw_bytes="<<4LL*a.nx*a.ny<<"\n";
 return {lo,hi,sum/(a.nx*double(a.ny))};
}
// One T-matrix is shared by all orientations. Sweep frames are streamed directly.
void write_parameters(std::ofstream&log,const Args&a,int argc,char**argv,bool series=false){
 log<<std::setprecision(17)<<"format_version=1\ncommand=";
 for(int j=0;j<argc;j++)log<<std::quoted(argv[j])<<" ";
 log<<"\nstatus=started\nshape="<<a.shape<<"\nlambda_vacuum_um="<<a.lambda<<"\nlambda_medium_um="<<a.lambda/a.nmed<<"\nlength_um="<<a.length<<"\ndiameter_um="<<a.diameter<<"\nn_real="<<a.nr<<"\nn_imag="<<a.ni<<"\nn_medium="<<a.nmed<<"\n";
#ifdef HOLO_CUDA
 log<<"backend=cuda_fp64\n";
#else
 log<<"backend=cpu_fortran_fftw\n";
#endif
 if(series)log<<"mode=theta_sweep\ntheta_start_deg=0\ntheta_end_deg="<<a.theta_end<<"\ntheta_steps="<<a.theta_steps<<"\ntheta_step_deg="<<a.theta_end/a.theta_steps<<"\nimage_count="<<a.theta_steps+1<<"\nmanifest=orientations.csv\n";
 else log<<"theta_deg="<<a.theta<<"\nraw_file=hologram.raw\n";
 log<<"phi_deg="<<a.phi<<"\npolarization_deg="<<a.pol<<"\nx_um="<<a.x<<"\ny_um="<<a.y<<"\nz_um="<<a.z<<"\nNA="<<a.na<<"\nnx="<<a.nx<<"\nny="<<a.ny<<"\ndx_um="<<a.dx<<"\ndy_um="<<a.dy<<"\npad="<<a.pad<<"\ntol="<<a.tol<<"\nndgs="<<a.ndgs<<"\ndetector="<<a.detector<<"\nraw_type=float32\nraw_endian=little\nraw_layout=row-major, x fastest\nraw_header_bytes=0\nraw_quantity=normalized intensity (incident=1)\ncoordinate_x_um=(column-floor(nx/2))*dx_um\ncoordinate_y_um=(row-floor(ny/2))*dy_um\n";
 log.flush();
}
void write_sections(std::ofstream&log,const Solver&s,const std::array<double,3>&c){
 log<<"tmatrix_precision="<<s.precision<<"\n";
 log<<"NMAX="<<s.n<<"\nNGAUSS="<<s.ng<<"\nCext_um2="<<c[0]<<"\nCsca_um2="<<c[1]<<"\nCabs_um2="<<c[2]<<"\n";
 if(c[2]<-1e-5*std::max(c[0],c[1]))log<<"warning=negative absorption; verify solver convergence\n";
 log.flush();
}
std::array<double,3> calculate_image(const Args&a,Solver&s,std::ostream&out,std::ofstream&log,ImageStats&stats){
 s.set_orientation(a.theta,a.phi);
 auto c=s.sections();
 if(!std::all_of(c.begin(),c.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("Nonfinite cross sections");
 write_sections(log,s,c);
 stats=write_hologram(a,s,out,log);
 return c;
}
double elapsed(std::chrono::steady_clock::time_point start){
 return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
fs::path executable_directory(){
#ifdef _WIN32
 std::vector<wchar_t> buffer(1024);
 for(;;){
  DWORD n=GetModuleFileNameW(nullptr,buffer.data(),static_cast<DWORD>(buffer.size()));
  if(n==0)throw std::runtime_error("Cannot determine executable path");
  if(n<buffer.size())return fs::path(std::wstring(buffer.data(),n)).parent_path();
  if(buffer.size()>=65536)throw std::runtime_error("Executable path too long");
  buffer.resize(buffer.size()*2);
 }
#else
 return fs::canonical("/proc/self/exe").parent_path();
#endif
}
int main(int argc,char**argv){
 if(argc>1&&std::string(argv[1])=="--help"){help();return 0;}
 std::ofstream log;fs::path dir;
 try{
  auto a=parse(argc,argv);
  auto start=std::chrono::steady_clock::now();
  const fs::path root=executable_directory();
  auto id=a.name.empty()?std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()):a.name;
  fs::create_directories(root/"run");dir=root/"run"/("run_"+id);
  if(!fs::create_directory(dir))throw std::runtime_error("Run directory already exists: "+dir.string());
  log.open(dir/"log.txt");if(!log)throw std::runtime_error("Cannot open log");
  write_parameters(log,a,argc,argv,a.theta_steps>0);
  int threads=1;
#ifdef HOLO_OPENMP
  holo_set_threads(a.threads,&threads);
  log<<"openmp_enabled=1\n";
#else
  log<<"openmp_enabled=0\n";
#endif
  log<<"openmp_threads_requested="<<a.threads<<"\nopenmp_threads="<<threads<<"\n";log.flush();
  const auto t_matrix_start=std::chrono::steady_clock::now();
  Solver s(a);
  log<<"tmatrix_elapsed_seconds="<<elapsed(t_matrix_start)<<"\n";
  log<<"tmatrix_precision_requested="<<a.precision<<"\ntmatrix_precision="<<s.precision<<"\n";log.flush();
  if(a.theta_steps){
   log<<"tmatrix_builds="<<(s.n?1:0)<<"\nNMAX="<<s.n<<"\nNGAUSS="<<s.ng<<"\n";
   const uintmax_t frame_bytes=4ULL*a.nx*a.ny;
   const uintmax_t stack_bytes=frame_bytes*(a.theta_steps+1ULL);
   log<<"stack_file=holograms.raw\nstack_layout=frame-major, then row-major, x fastest\nstack_frame_bytes="<<frame_bytes<<"\nstack_expected_bytes="<<stack_bytes<<"\nstack_offset_bytes=0\nstack_gap_bytes=0\nsections_file=cross_sections.txt\nsections_columns=theta_deg Cext_um2 Csca_um2 Cabs_um2\n";
   log.flush();
   std::ofstream csv(dir/"orientations.csv");csv.exceptions(std::ios::badbit|std::ios::failbit);
   csv<<std::setprecision(17)<<"index,theta_deg,phi_deg,Cext_um2,Csca_um2,Cabs_um2,nx,ny,raw_file,offset_bytes,raw_min,raw_max,raw_mean,elapsed_seconds\n";
   csv.flush();
   std::ofstream stack(dir/"holograms.raw.part",std::ios::binary);
   stack.exceptions(std::ios::badbit|std::ios::failbit);
   std::ofstream sections(dir/"cross_sections.txt.part");
   sections.exceptions(std::ios::badbit|std::ios::failbit);
   sections<<std::setprecision(17)<<"# theta_deg Cext_um2 Csca_um2 Cabs_um2\n";
   sections.flush();
   for(int j=0;j<=a.theta_steps;j++){
    const auto t=std::chrono::steady_clock::now();
    a.theta=a.theta_end*j/a.theta_steps;
    log<<"frame_index="<<j<<"\nframe_theta_deg="<<a.theta<<"\n";log.flush();
    {
     ImageStats stats{};
     auto c=calculate_image(a,s,stack,log,stats);
     if(stack.tellp()!=static_cast<std::streamoff>(frame_bytes*(j+1ULL)))throw std::runtime_error("Unexpected stack offset");
     sections<<a.theta<<' '<<c[0]<<' '<<c[1]<<' '<<c[2]<<'\n';sections.flush();
     auto dt=elapsed(t);
     csv<<j<<','<<a.theta<<",0,"<<c[0]<<','<<c[1]<<','<<c[2]<<','<<a.nx<<','<<a.ny<<",holograms.raw,"<<frame_bytes*j<<','<<stats.lo<<','<<stats.hi<<','<<stats.mean<<','<<dt<<"\n";csv.flush();
     log<<"completed_images="<<j+1<<"\n";log.flush();
     std::cout<<std::setprecision(12)<<"["<<j+1<<"/"<<a.theta_steps+1<<"] theta="<<a.theta<<" phi=0 Cext="<<c[0]<<" Csca="<<c[1]<<" Cabs="<<c[2]<<" um^2"<<std::endl;
    }
   }
   csv.close();
   stack.close();sections.close();
   if(fs::file_size(dir/"holograms.raw.part")!=stack_bytes)throw std::runtime_error("Unexpected final stack size");
   fs::rename(dir/"holograms.raw.part",dir/"holograms.raw");
   fs::rename(dir/"cross_sections.txt.part",dir/"cross_sections.txt");
   log<<"stack_bytes="<<stack_bytes<<"\n";
  }else{
   std::ofstream image(dir/"hologram.raw.part",std::ios::binary);
   image.exceptions(std::ios::badbit|std::ios::failbit);
   ImageStats stats{};
   auto c=calculate_image(a,s,image,log,stats);
   image.close();
   fs::rename(dir/"hologram.raw.part",dir/"hologram.raw");
   std::cout<<std::setprecision(12)<<"Cext="<<c[0]<<" Csca="<<c[1]<<" Cabs="<<c[2]<<" um^2\n";
  }
  log<<"elapsed_seconds="<<elapsed(start)<<"\nstatus=complete\n";log.flush();
  std::cout<<dir.string()<<"\n";
  return 0;
 }catch(const std::exception&e){if(log){log<<"status=failed\nerror="<<e.what()<<"\n";log.flush();}std::cerr<<"Error: "<<e.what()<<"\n";return 1;}
}
