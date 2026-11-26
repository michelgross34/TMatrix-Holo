#define HOLO_DLL_EXPORT
#include "gpu_holo_abi.h"
#include "gpu_holo.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
namespace {
void message(char*err,int cap,const char*s) noexcept {
 if(err&&cap>0){std::strncpy(err,s,static_cast<size_t>(cap)-1);err[cap-1]=0;}
}
template<class F> int invoke(F f,char*err,int cap) noexcept {
 try{f();message(err,cap,"");return 0;}
 catch(const std::exception&e){message(err,cap,e.what());return 1;}
 catch(...){message(err,cap,"Unknown CUDA DLL error");return 1;}
}
}
int holo_cuda_abi_version(void){return 1;}
int holo_cuda_prepare(int n,double lam,double alpha,double beta,double px,double py,const double*t,char*err,int cap){
 return invoke([&]{if(n<0||n>100||(n&&!t))throw std::runtime_error("Invalid T-matrix");gpu_prepare(n,lam,alpha,beta,px,py,t);},err,cap);
}
int holo_cuda_sections(const double*mu,const double*w,int nt,int np,double*result,char*err,int cap){
 return invoke([&]{if(!mu||!w||!result||nt<1||np<1)throw std::runtime_error("Invalid section buffers");auto s=gpu_sections(mu,w,nt,np);std::copy(s.begin(),s.end(),result);},err,cap);
}
int holo_cuda_image(int nx,int ny,int pad,double dx,double dy,double z,double x,double y,double cut,int nc,
 float*out,uint64_t capacity,uint64_t*count,char*err,int cap){
 return invoke([&]{if(!out||!count||nx<1||ny<1||pad<1||capacity!=uint64_t(nx)*ny||(nc!=2&&nc!=3))throw std::runtime_error("Invalid image buffers");
 std::size_t n=0;auto image=gpu_image(nx,ny,pad,dx,dy,z,x,y,cut,nc,n);
 if(image.size()!=capacity)throw std::runtime_error("Unexpected CUDA image size");
 std::copy(image.begin(),image.end(),out);*count=n;},err,cap);
}
