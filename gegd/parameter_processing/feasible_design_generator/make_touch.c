#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <Python.h>
#include <stdlib.h>
#include <math.h>

#include "make_touch.h"
#include "touch2pix.h"
#include "apply_symmetry_int.h"
#include "utils.h"

#define NPY_ARRAY_TYPE NPY_INT32

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
                int symmetry) {

    const int size = Nx * Ny;
    const int dx = ind_touch[0];
    const int dy = ind_touch[1];

    if (solid_flag) {
        touch2pix(dx, dy, touch_solid, touch_void, refconv1, Nx, Ny, symmetry);
    } else if (void_flag) {
        touch2pix(dx, dy, touch_void, touch_solid, refconv1, Nx, Ny, symmetry);
    }
    
    // Roll refconv0
    int* rolled_mask = (int*)malloc(Nx * Ny * sizeof(int));
    if (!rolled_mask) return;
    roll2d(refconv0, rolled_mask, dx, dy, Nx, Ny);

    const int update_val = solid_flag ? 1 : -1;
    for (int idx = 0; idx < size; ++idx) {
        if (rolled_mask[idx] > 0 && pix_solid[idx] == 0) {
            pix_solid[idx] = update_val;
        }
    }
    free(rolled_mask);

    // Apply symmetry to pix_solid
    if (symmetry == 1 || symmetry == 2 || symmetry == 4) {
        int* sym_pix = (int*)malloc(Nx * Ny * sizeof(int));
        apply_symmetry_int(
            pix_solid,
            sym_pix,
            Nx,
            Ny,
            symmetry
        );
        
        if (sym_pix) {
            for (int i = 0; i < Nx * Ny; ++i)
                pix_solid[i] = sym_pix[i];
            free(sym_pix);
        }
    }
}