// MinGW client for the separately compiled CUDA DLL. No CUDA headers required.
#define NOMINMAX
#include <windows.h>
#include "gpu_holo_abi.h"
#include "gpu_holo.hpp"
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <string>
namespace {
struct Api{
 HMODULE dll=nullptr;
 decltype(&holo_cuda_abi_version) version=nullptr;
 decltype(&holo_cuda_prepare) prepare=nullptr;
 decltype(&holo_cuda_sections) sections=nullptr;
 decltype(&holo_cuda_image) image=nullptr;
 template<class T> T symbol(const char*name){
  auto p=GetProcAddress(dll,name);
  if(!p)throw std::runtime_error(std::string("Missing CUDA DLL export: ")+name);
  // GetProcAddress returns a generic function pointer; copy its representation.
  static_assert(sizeof(T)==sizeof(p));
  T f;std::memcpy(&f,&p,sizeof(f));return f;
 }
 Api(){
  std::vector<wchar_t> b(32768);
  DWORD n=GetModuleFileNameW(nullptr,b.data(),static_cast<DWORD>(b.size()));
  if(!n||n>=b.size())throw std::runtime_error("Cannot determine CUDA executable path");
  auto path=std::filesystem::path(std::wstring(b.data(),n)).parent_path()/L"holo_cuda.dll";
  dll=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if(!dll)throw std::runtime_error("Cannot load holo_cuda.dll or its dependencies; Windows error "+std::to_string(GetLastError()));
  try{
   version=symbol<decltype(version)>("holo_cuda_abi_version");
   if(version()!=1)throw std::runtime_error("Unsupported holo_cuda DLL ABI");
   prepare=symbol<decltype(prepare)>("holo_cuda_prepare");
   sections=symbol<decltype(sections)>("holo_cuda_sections");
   image=symbol<decltype(image)>("holo_cuda_image");
  }catch(...){FreeLibrary(dll);dll=nullptr;throw;}
 }
 // Keep the DLL resident until process exit, including CUDA runtime cleanup.
};
Api& api(){static Api a;return a;}
void check(int code,const char*err){if(code)throw std::runtime_error(std::string("CUDA DLL: ")+err);}
}
void gpu_prepare(int n,double lam,double alpha,double beta,double px,double py,const double*t){
 char e[1024]={};check(api().prepare(n,lam,alpha,beta,px,py,t,e,sizeof(e)),e);
}
std::array<double,3> gpu_sections(const double*mu,const double*w,int nt,int np){
 std::array<double,3> r{};char e[1024]={};
 check(api().sections(mu,w,nt,np,r.data(),e,sizeof(e)),e);return r;
}
std::vector<float> gpu_image(int nx,int ny,int pad,double dx,double dy,double z,double x,double y,double cut,int nc,std::size_t&count){
 std::vector<float> r(static_cast<std::size_t>(nx)*ny);uint64_t n=0;char e[1024]={};
 check(api().image(nx,ny,pad,dx,dy,z,x,y,cut,nc,r.data(),r.size(),&n,e,sizeof(e)),e);
 count=static_cast<std::size_t>(n);return r;
}
