#pragma once
#include <array>
#include <vector>
#include <cstddef>
void gpu_prepare(int n,double lam,double alpha,double beta,double px,double py,const double* t);
std::array<double,3> gpu_sections(const double* mu,const double* w,int nt,int np);
std::vector<float> gpu_image(int nx,int ny,int pad,double dx,double dy,
                            double z,double x,double y,double kcut,int nc,std::size_t& count);
