#ifndef MAKE_FEASIBLE_H
#define MAKE_FEASIBLE_H

#include <Python.h>

int* make_feasible(
    float* weight,
    int Nx,
    int Ny,
    int brush_size,
    int periodic,
    int symmetry,
    int dim,
    int upsample_ratio
);

#endif