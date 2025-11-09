#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <Python.h>
#include <stdlib.h>
#include <math.h>

#include "touch2pix.h"
#include "apply_symmetry_int.h"
#include "utils.h"

#define NPY_ARRAY_TYPE NPY_INT32

void touch2pix(int dx,
               int dy,
               int* touch1,
               int* touch2,
               const int* reference_convolution1,
               int Nx,
               int Ny,
               int symmetry) {

    int size = Nx * Ny;
    int center_idx = dx * Ny + dy;

    int* rolled = (int*)malloc(size * sizeof(int));
    if (!rolled) return;

    roll2d(reference_convolution1, rolled, dx, dy, Nx, Ny);
    touch1[center_idx] = 1;

    for (int idx = 0; idx < size; ++idx) {
        if (rolled[idx] > 0 && touch2[idx] == 0) {
            touch2[idx] = -1;
        }
    }

    free(rolled);

    if (symmetry == 1 || symmetry == 2 || symmetry == 4) {
        int* sym1 = (int*)malloc(size * sizeof(int));
        apply_symmetry_int(
            touch1,
            sym1,
            Nx,
            Ny,
            symmetry
        );
        
        int* sym2 = (int*)malloc(size * sizeof(int));
        apply_symmetry_int(
            touch2,
            sym2,
            Nx,
            Ny,
            symmetry
        );
                       
        if (sym1 && sym2) {
            for (int i = 0; i < size; ++i) {
                touch1[i] = sym1[i];
                touch2[i] = sym2[i];
            }
        }
        free(sym1);
        free(sym2);
    }
}