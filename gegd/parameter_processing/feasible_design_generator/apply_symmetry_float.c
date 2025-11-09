#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <Python.h>
#include <stdlib.h>
#include <math.h>

#include "apply_symmetry_float.h"

#define NPY_ARRAY_TYPE NPY_INT32
#define IDX(i, j, W) ((i)*(W) + (j))
#define SYM_D1 1
#define SYM_D2 2
#define SYM_D4 4

// ------------------ Helper: D1 ------------------
void compute_D1_float(const float* x, float* out, int N_x, int N_y) {
    int half = (N_x + 1) / 2;

    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < N_y; ++j) {
            int val = x[IDX(i, j, N_y)];
            int i_flip = N_x - 1 - i;
            
            out[IDX(i, j, N_y)] = val;
            out[IDX(i_flip, j, N_y)] = val;
        }
    }
}

// ------------------ Helper: D2 ------------------
void compute_D2_float(const float* x, float* out, int N_x, int N_y) {
    int half_x = (N_x + 1) / 2;
    int half_y = (N_y + 1) / 2;

    for (int i = 0; i < half_x; ++i) {
        for (int j = 0; j < half_y; ++j) {
            int val = x[IDX(i, j, N_y)];
            int i_flip = N_x - 1 - i;
            int j_flip = N_y - 1 - j;

            out[IDX(i, j, N_y)] = val;
            out[IDX(i_flip, j, N_y)] = val;
            out[IDX(i, j_flip, N_y)] = val;
            out[IDX(i_flip, j_flip, N_y)] = val;
        }
    }
}

// ------------------ Helper: D4 ------------------
void compute_D4_float(const float* x, float* out, int N) {
    int half = (N + 1) / 2;

    for (int i = 0; i < half; ++i) {
        for (int j = i; j < half; ++j) {
            int val = x[IDX(i, j, N)];

            out[IDX(i, j, N)] = val;
            out[IDX(j, i, N)] = val;
            out[IDX(N - 1 - j, i, N)] = val;
            out[IDX(i, N - 1 - j, N)] = val;
            out[IDX(N - 1 - i, N - 1 - j, N)] = val;
            out[IDX(N - 1 - j, N - 1 - i, N)] = val;
            out[IDX(N - 1 - i, j, N)] = val;
            out[IDX(j, N - 1 - i, N)] = val;
        }
    }
}

// ------------------ Unified apply_symmetry() ------------------
void apply_symmetry_float(
    const float* x,
    float* out,
    int N_x,
    int N_y,
    int sym_code
) {

    if (sym_code == SYM_D1) {
        compute_D1_float(x, out, N_x, N_y);
    } else if (sym_code == SYM_D2) {
        compute_D2_float(x, out, N_x, N_y);
    } else if (sym_code == SYM_D4) {
        if (N_x != N_y) {
            return;
        }
        compute_D4_float(x, out, N_x);
    } else {
        for (int i = 0; i < N_x * N_y; ++i)
            out[i] = x[i];
    }
}