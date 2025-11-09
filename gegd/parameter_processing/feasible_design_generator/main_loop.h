#ifndef MAIN_LOOP_H
#define MAIN_LOOP_H

#include <Python.h>

int* main_loop(float* score_solid,
               int* pix_solid,
               int* touch_solid,
               int* touch_void,
               int* refconv0,
               int* refconv1,
               int* refconv2,
               int symmetry,
               int periodic,
               int brush_size,
               int dim,
               int Nx,
               int Ny);

#endif