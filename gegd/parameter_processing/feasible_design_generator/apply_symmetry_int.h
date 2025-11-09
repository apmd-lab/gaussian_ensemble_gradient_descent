#ifndef APPLY_SYMMETRY_INT_H
#define APPLY_SYMMETRY_INT_H

#include <Python.h>

void apply_symmetry_int(
    const int* x,
    int* out,
    int N_x,
    int N_y,
    int sym_code
);

#endif