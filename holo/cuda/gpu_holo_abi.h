#pragma once
/* Stable C ABI between MinGW and the MSVC/nvcc DLL.
   All buffers are owned by the caller; no C++ objects cross the DLL boundary. */
#include <stdint.h>
#if defined(_WIN32) && defined(HOLO_DLL_EXPORT)
#define HOLO_API __declspec(dllexport)
#else
#define HOLO_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
HOLO_API int holo_cuda_abi_version(void);
HOLO_API int holo_cuda_prepare(int n,double lam,double alpha,double beta,double px,double py,
 const double*t,char*err,int errcap);
HOLO_API int holo_cuda_sections(const double*mu,const double*w,int nt,int np,
 double*result,char*err,int errcap);
HOLO_API int holo_cuda_image(int nx,int ny,int pad,double dx,double dy,double z,double x,double y,
 double cut,int nc,float*out,uint64_t capacity,uint64_t*count,char*err,int errcap);
#ifdef __cplusplus
}
#endif
