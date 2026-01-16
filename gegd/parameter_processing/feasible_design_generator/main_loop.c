#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include "main_loop.h"
#include "find_index_max.h"
#include "make_touch.h"
#include "fill_required_pixels.h"
#include "utils.h"

#define IDX(i, j, W) ((i) * (W) + (j))

int* main_loop(float* score_solid,
               int* pix_solid,
               int* touch_solid,
               int* touch_void,
               int* refconv0,
               int* refconv1,
               int* refconv2,
               int symmetry,
               int periodic,
               int brush_size,
               int dim,
               int Nx,
               int Ny) {

    int* pix_result = (int*)malloc(Nx * Ny * sizeof(int));
    if (!pix_result) return NULL;

    int n_iter = 0;
    int debug_iter = INT_MAX;
    int n_empty_touch_solid_prev = Nx * Ny;
    int n_empty_touch_void_prev = Nx * Ny;
    while (1) {
        int solid_flag = 0, void_flag = 0;
        float max_solid = -FLT_MAX, max_void = -FLT_MAX;

        int half_x = (Nx + 1) / 2;
        int half_y = (Ny + 1) / 2;
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                if (symmetry == 1 && i >= half_x) continue;
                if (symmetry == 2 && (i >= half_x || j >= half_y)) continue;
                if (symmetry == 4 && (i > j || j >= half_y)) continue;

                int idx = i * Ny + j;
                if (touch_solid[idx] == 0 && score_solid[idx] > max_solid)
                    max_solid = score_solid[idx];
                if (touch_void[idx] == 0 && -score_solid[idx] > max_void)
                    max_void = -score_solid[idx];
            }
        }
        
        if (n_iter == debug_iter) {
            printf("max_solid = %f\n", max_solid);
        }

        int* ind_max;
        if (max_solid > max_void) {
            solid_flag = 1;
            ind_max = find_index_max(symmetry,
                                     periodic,
                                     brush_size,
                                     dim,
                                     Nx,
                                     Ny,
                                     score_solid,
                                     touch_solid,
                                     max_solid);
        } else if (max_void > max_solid) {
            void_flag = 1;
            float* score_void = (float*)malloc(Nx * Ny * sizeof(float));
            for (int i = 0; i < Nx * Ny; ++i) score_void[i] = -score_solid[i];
            ind_max = find_index_max(symmetry,
                                     periodic,
                                     brush_size,
                                     dim,
                                     Nx,
                                     Ny,
                                     score_void,
                                     touch_void,
                                     max_void);
            
            free(score_void);
            
        } else {
            if ((float)rand() / RAND_MAX > 0.5f) {
                solid_flag = 1;
                ind_max = find_index_max(symmetry,
                                         periodic,
                                         brush_size,
                                         dim,
                                         Nx,
                                         Ny,
                                         score_solid,
                                         touch_solid,
                                         max_solid);
            } else {
                void_flag = 1;
                float* score_void = (float*)malloc(Nx * Ny * sizeof(float));
                for (int i = 0; i < Nx * Ny; ++i) score_void[i] = -score_solid[i];
                ind_max = find_index_max(symmetry,
                                         periodic,
                                         brush_size,
                                         dim,
                                         Nx,
                                         Ny,
                                         score_void,
                                         touch_void,
                                         max_void);
                
                free(score_void);
                
            }
        }
        
        if (n_iter == debug_iter) {
            printf("solid_flag = %d\n", solid_flag);
            printf("ind_max = [%d,%d]\n", ind_max[0], ind_max[1]);
            //exit(EXIT_FAILURE);
        }

        int n_empty_before = 0;
        for (int i = 0; i < Nx * Ny; ++i)
            if (pix_solid[i] == 0) n_empty_before++;

        make_touch(solid_flag,
                   void_flag,
                   ind_max,
                   touch_solid,
                   touch_void,
                   pix_solid,
                   refconv0,
                   refconv1,
                   Nx,
                   Ny,
                   symmetry);
        
        if (n_iter == debug_iter) {
            save_int_array_to_npy("touch_solid_make_touch.npy", touch_solid, Nx, Ny);
            save_int_array_to_npy("touch_void_make_touch.npy", touch_void, Nx, Ny);
            save_int_array_to_npy("pix_solid_make_touch.npy", pix_solid, Nx, Ny);
            exit(EXIT_FAILURE);
        }

        int n_empty_after = 0;
        for (int i = 0; i < Nx * Ny; ++i)
            if (pix_solid[i] == 0) n_empty_after++;

        if (n_empty_before != n_empty_after) {
            fill_required_pixels(ind_max,
                                 touch_solid,
                                 touch_void,
                                 pix_solid,
                                 score_solid,
                                 refconv0,
                                 refconv1,
                                 refconv2,
                                 Nx,
                                 Ny,
                                 symmetry);
        }
        
        if (n_iter == debug_iter) {
            save_int_array_to_npy("touch_solid_fill_req.npy", touch_solid, Nx, Ny);
            save_int_array_to_npy("touch_void_fill_req.npy", touch_void, Nx, Ny);
            save_int_array_to_npy("pix_solid_fill_req.npy", pix_solid, Nx, Ny);
            exit(EXIT_FAILURE);
        }

        free(ind_max);
        n_iter++;
        
        int n_empty_pix_solid = 0;
        for (int i = 0; i < Nx * Ny; ++i)
            if (pix_solid[i] == 0) n_empty_pix_solid++;
        
        int n_empty_touch_solid = 0;
        for (int i = 0; i < Nx * Ny; ++i)
            if (touch_solid[i] == 0) n_empty_touch_solid++;
            
        int n_empty_touch_void = 0;
        for (int i = 0; i < Nx * Ny; ++i)
            if (touch_void[i] == 0) n_empty_touch_void++;
        
        if (n_empty_pix_solid > 0 && n_empty_touch_solid == n_empty_touch_solid_prev && n_empty_touch_void == n_empty_touch_void_prev) {
            fprintf(stderr, "Error: No progress made in brush touching (infinite loop). n_empty_pix_solid: %d, n_empty_touch_solid: %d, n_empty_touch_void: %d\n", n_empty_pix_solid, n_empty_touch_solid, n_empty_touch_void);
            free(pix_result);
            return NULL;
        }
        n_empty_touch_solid_prev = n_empty_touch_solid;
        n_empty_touch_void_prev = n_empty_touch_void;
        //if (n_iter % 100 == 0) {
        //printf("Iter %d | Unassigned Solid Pixels: %d | Unassigned Solid Touches: %d | Unassigned Void Touches: %d\n", n_iter, n_empty_pix_solid, n_empty_touch_solid, n_empty_touch_void);
        //}

        if (n_empty_pix_solid == 0)
            break;
    }

    for (int i = 0; i < Nx * Ny; ++i)
        pix_result[i] = pix_solid[i];

    return pix_result;
}