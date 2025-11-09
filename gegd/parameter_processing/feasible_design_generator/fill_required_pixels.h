#ifndef FILL_REQUIRED_PIXELS_H
#define FILL_REQUIRED_PIXELS_H

#include <Python.h>

void fill_required_pixels(int* ind_max,
                          int* touch_solid,
                          int* touch_void,
                          int* pix_solid,
                          float* score_solid,
                          int* refconv0,
                          int* refconv1,
                          int* refconv2,
                          int Nx,
                          int Ny,
                          int symmetry);

#endif