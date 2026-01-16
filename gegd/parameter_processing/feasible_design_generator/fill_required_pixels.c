#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "fill_required_pixels.h"
#include "make_touch.h"
#include "touch2pix.h"
#include "utils.h"

#define IDX(i, j, W) ((i) * (W) + (j))

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
                          int symmetry) {

    int size = Nx * Ny;

    int* last_affected = (int*)calloc(size, sizeof(int));
    int* req_mask = (int*)malloc(size * sizeof(int));
    int* updated = (int*)malloc(size * sizeof(int));
    if (!last_affected || !req_mask || !updated) {
        free(last_affected);
        free(req_mask);
        free(updated);
        return;
    }
    
    roll2d(refconv2, last_affected, ind_max[0], ind_max[1], Nx, Ny);

    int required = 1;
    while (required) {
        required = 0;
        
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                int idx = IDX(i, j, Ny);
                if (pix_solid[idx] == 0 && last_affected[idx]) {
                    roll2d(refconv0, req_mask, i, j, Nx, Ny);

                    int all_solid = 1, all_void = 1;
                    for (int k = 0; k < size; ++k) {
                        if (req_mask[k]) {
                            if (touch_solid[k] == 0) all_solid = 0;
                            if (touch_void[k] == 0) all_void = 0;
                        }
                    }

                    if (all_solid || all_void) {
                        int* touch = (all_solid ? touch_void : touch_solid);
                        float max_val = -FLT_MAX;
                        int max_idx = -1;

                        for (int k = 0; k < size; ++k) {
                            float score = all_solid ? -score_solid[k] : score_solid[k];
                            if (req_mask[k] && touch[k] == 0) {
                                if (score > max_val) {
                                    max_val = score;
                                    max_idx = k;
                                }
                             }
                        }

                        if (max_idx >= 0) {
                            int ix = max_idx / Ny;
                            int iy = max_idx % Ny;
                            int ind_req[2] = {ix, iy};
                            make_touch(!all_solid,
                                       all_solid,
                                       ind_req,
                                       touch_solid,
                                       touch_void,
                                       pix_solid,
                                       refconv0,
                                       refconv1,
                                       Nx,
                                       Ny,
                                       symmetry);

                            roll2d(refconv2, updated, ix, iy, Nx, Ny);
                            for (int k = 0; k < size; ++k)
                                last_affected[k] |= updated[k];
                            required = 1;
                            goto LOOP_END;
                        }
                    }
                    last_affected[idx] = 0;
                }
            }
        }
        LOOP_END:;
    }

    free(last_affected);
    free(req_mask);
    free(updated);
}