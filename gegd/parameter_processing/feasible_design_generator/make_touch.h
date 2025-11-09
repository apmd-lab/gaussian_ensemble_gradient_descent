#ifndef MAKE_TOUCH_H
#define MAKE_TOUCH_H

#include <Python.h>

void make_touch(int solid_flag,
                int void_flag,
                int* ind_touch,
                int* touch_solid,
                int* touch_void,
                int* pix_solid,
                int* refconv0,
                int* refconv1,
                int Nx,
                int Ny,
                int symmetry);

#endif