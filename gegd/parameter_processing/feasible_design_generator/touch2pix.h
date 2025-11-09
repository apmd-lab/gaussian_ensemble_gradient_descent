#ifndef TOUCH2PIX_H
#define TOUCH2PIX_H

#include <Python.h>

void touch2pix(int dx,
               int dy,
               int* touch1,
               int* touch2,
               const int* reference_convolution1,
               int Nx,
               int Ny,
               int symmetry);

#endif