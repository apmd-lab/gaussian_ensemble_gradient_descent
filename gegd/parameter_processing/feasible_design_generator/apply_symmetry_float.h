#ifndef APPLY_SYMMETRY_FLOAT_H
#define APPLY_SYMMETRY_FLOAT_H

#include <Python.h>

void apply_symmetry_float(
    const float* x,
    float* out,
    int N_x,
    int N_y,
    int sym_code
);

#endif