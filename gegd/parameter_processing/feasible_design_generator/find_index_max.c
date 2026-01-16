#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "find_index_max.h"

#define IDX(i, j, W) ((i) * (W) + (j))

int* find_index_max(
    int symmetry,
    int periodic,
    int brush_size,
    int dim,
    int Nx,
    int Ny,
    float* score,
    int* touch,
    float max_val
) {
                    
    int* result = (int*)malloc(2 * sizeof(int));
    if (!result) return NULL;
    result[0] = -1;
    result[1] = -1;
    
    int half_x = (Nx + 1) / 2;// + 2*(1 - periodic)*brush_size;
    int half_y = (Ny + 1) / 2;// + 2*(1 - periodic)*brush_size;
    
    const float threshold = 1e-4f;

    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int idx = i * Ny + j;
            float val = score[idx];
            
            if (fabsf(val - max_val) >= threshold) continue;
        
            if (symmetry == 0) {
                if (dim == 1) {
                    if (touch[i] == 0) {
                        result[0] = i;
                        result[1] = -1;
                        return result;
                    }
                } else {
                    if (touch[idx] == 0) {
                        result[0] = i;
                        result[1] = j;
                        return result;
                    }
                }
            } else if (symmetry == 1) {
                if (i < half_x && touch[idx] == 0) {
                    result[0] = i;
                    result[1] = j;
                    return result;
                }
            } else if (symmetry == 2) {
                if (i < half_x && j < half_y && touch[idx] == 0) {
                    result[0] = i;
                    result[1] = j;
                    return result;
                }
            } else if (symmetry == 4) {
                if (i <= j && j < half_y && touch[idx] == 0) {
                    result[0] = i;
                    result[1] = j;
                    return result;
                }
            }
        }
    }

    return result;
}