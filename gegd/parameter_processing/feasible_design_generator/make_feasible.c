#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "make_feasible.h"
#include "main_loop.h"
#include "utils.h"
#include "apply_symmetry_int.h"
#include "apply_symmetry_float.h"

#define IDX(i, j, W) ((i) * (W) + (j))

int* make_feasible(
    float* weight,
    int Nx,
    int Ny,
    int brush_size,
    int periodic,
    int symmetry,
    int dim,
    int upsample_ratio
) {

    int debug = 0;
    
    // Upsampling
    int Nx_up = Nx * upsample_ratio;
    int Ny_up = Ny * upsample_ratio;
    float* weight_upsampled = (float*)calloc(Nx_up * Ny_up, sizeof(float));
    if (upsample_ratio > 1) {
        brush_size = (int)(floor(upsample_ratio * sqrt((brush_size - 2)*(brush_size - 2) + 1) + 2));
        
        zoom_nearest(
            weight,
            weight_upsampled,
            Nx,
            Ny,
            Nx_up,
            Ny_up,
            upsample_ratio,
            periodic
        );
        
        float* sym = (float*)malloc(Nx_up * Ny_up * sizeof(float));
        apply_symmetry_float(
            weight_upsampled,
            sym,
            Nx_up,
            Ny_up,
            symmetry
        );
        
        if (sym) {
            for (int i = 0; i < Nx_up * Ny_up; ++i) {
                weight_upsampled[i] = sym[i];
            }
        }
        
        free(sym);
        
    } else {
        for (int i = 0; i < Nx_up * Ny_up; ++i) {
            weight_upsampled[i] = weight[i];
        }
    }
    
    if (debug) {
        save_float_array_to_npy("weight.npy", weight, Nx, Ny);
        save_float_array_to_npy("weight_upsampled.npy", weight_upsampled, Nx_up, Ny_up);
        //exit(EXIT_FAILURE);
    }

    // Initialize arrays
    int* touch_solid = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    int* touch_void = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    int* pix_solid = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    float* score_solid = (float*)calloc(Nx_up * Ny_up, sizeof(float));
    float* score_solid_pre_fill = (float*)calloc(Nx_up * Ny_up, sizeof(float));

    int r_c = (brush_size - 1) / 2;
    int r_c_plus4 = (brush_size + 3) / 2;
    int brush_size_conv = brush_size + 2 * r_c;
    int brush_size_conv2 = brush_size + 4 * r_c;
    
    // Define brush
    int* brush = (int*)calloc(brush_size * brush_size, sizeof(int));
    for (int i = 0; i < brush_size; ++i) {
        for (int j = 0; j < brush_size; ++j) {
            int dx = i - r_c;
            int dy = j - r_c;
            if (sqrt(dx*dx + dy*dy) <= r_c) {
                brush[IDX(i, j, brush_size)] = 1;
            }
        }
    }

    // Define brush with size + 4
    int* brush_plus4 = (int*)calloc((brush_size + 4) * (brush_size + 4), sizeof(int));
    int brush_plus4_sum = 0;
    for (int i = 0; i < brush_size + 4; ++i) {
        for (int j = 0; j < brush_size + 4; ++j) {
            int dx = i - r_c_plus4;
            int dy = j - r_c_plus4;
            if (sqrt(dx*dx + dy*dy) <= r_c_plus4) {
                brush_plus4[IDX(i, j, brush_size + 4)] = 1;
                ++brush_plus4_sum;
            }
        }
    }

    // Define brush convolved once
    int* brush_conv_shape = (int*)calloc(brush_size_conv * brush_size_conv, sizeof(int));
    for (int i = r_c; i < brush_size_conv - r_c; ++i) {
        for (int j = r_c; j < brush_size_conv - r_c; ++j) {
            brush_conv_shape[IDX(i, j, brush_size_conv)] = brush[IDX(i - r_c, j - r_c, brush_size)];
        }
    }

    int* temp = (int*)calloc(brush_size_conv * brush_size_conv, sizeof(int));
    binary_convolve(
        brush_conv_shape,
        temp,
        brush,
        brush_size_conv,
        brush_size_conv,
        brush_size,
        0
    );
    for (int i = 0; i < brush_size_conv * brush_size_conv; ++i)
        brush_conv_shape[i] = temp[i];

    free(temp);

    // Define brush convolved twice
    int* brush_conv_shape2 = (int*)calloc(brush_size_conv2 * brush_size_conv2, sizeof(int));
    for (int i = r_c; i < brush_size_conv2 - r_c; ++i) {
        for (int j = r_c; j < brush_size_conv2 - r_c; ++j) {
            brush_conv_shape2[IDX(i, j, brush_size_conv2)] = brush_conv_shape[IDX(i - r_c, j - r_c, brush_size_conv)];
        }
    }

    temp = (int*)calloc(brush_size_conv2 * brush_size_conv2, sizeof(int));
    binary_convolve(
        brush_conv_shape2,
        temp,
        brush,
        brush_size_conv2,
        brush_size_conv2,
        brush_size,
        0
    );
    for (int i = 0; i < brush_size_conv2 * brush_size_conv2; ++i)
        brush_conv_shape2[i] = temp[i];

    free(temp);
    
    if (debug) {
        save_int_array_to_npy("brush.npy", brush, brush_size, brush_size);
        save_int_array_to_npy("brush_plus4.npy", brush_plus4, brush_size + 4, brush_size + 4);
        save_int_array_to_npy("brush_conv_shape.npy", brush_conv_shape, brush_size_conv, brush_size_conv);
        save_int_array_to_npy("brush_conv_shape2.npy", brush_conv_shape2, brush_size_conv2, brush_size_conv2);
        //exit(EXIT_FAILURE);
    }

    // Compute the solid reward array
    convolve(
        weight_upsampled,
        score_solid,
        brush,
        Nx_up,
        Ny_up,
        brush_size,
        periodic
    );

    convolve(
        weight_upsampled,
        score_solid_pre_fill,
        brush_plus4,
        Nx_up,
        Ny_up,
        brush_size + 4,
        periodic
    );
             
    free(weight_upsampled);
    
    if (debug) {
        save_float_array_to_npy("score_solid.npy", score_solid, Nx_up, Ny_up);
        save_float_array_to_npy("score_solid_pre_fill.npy", score_solid_pre_fill, Nx_up, Ny_up);
        //exit(EXIT_FAILURE);
    }
    
    // Pre-fill touches
    int* touch_solid_pos = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    int* touch_solid_neg = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    
    for (int i = 0; i < Nx_up * Ny_up; ++i) {
        if (score_solid_pre_fill[i] >= brush_plus4_sum) {
            touch_solid[i] = 1;
            touch_solid_pos[i] = 1;
        } else if (score_solid_pre_fill[i] <= -brush_plus4_sum) {
            touch_void[i] = 1;
            touch_solid_neg[i] = 1;
        }
    }
    
    temp = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    binary_convolve(
        touch_solid_pos,
        temp,
        brush,
        Nx_up,
        Ny_up,
        brush_size,
        periodic
    );
    for (int i = 0; i < Nx_up * Ny_up; ++i) {
        if (temp[i]) {
            pix_solid[i] = 1;
        }
    }
    
    free(temp);
    
    temp = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    binary_convolve(
        touch_solid_neg,
        temp,
        brush,
        Nx_up,
        Ny_up,
        brush_size,
        periodic
    );
    for (int i = 0; i < Nx_up * Ny_up; ++i) {
        if (temp[i]) {
            pix_solid[i] = -1;
        }
    }
    
    free(temp);

    temp = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    binary_convolve(
        touch_solid_pos,
        temp,
        brush_conv_shape,
        Nx_up,
        Ny_up,
        brush_size_conv,
        periodic
    );
    for (int i = 0; i < Nx_up * Ny_up; ++i) {
        if (temp[i]) {
            touch_void[i] = -1;
        }
    }
    
    free(temp);

    temp = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    binary_convolve(
        touch_solid_neg,
        temp,
        brush_conv_shape,
        Nx_up,
        Ny_up,
        brush_size_conv,
        periodic
    );
    for (int i = 0; i < Nx_up * Ny_up; ++i) {
        if (temp[i]) {
            touch_solid[i] = -1;
        }
    }
    
    free(temp);
    
    if (debug) {
        save_int_array_to_npy("touch_solid_pos.npy", touch_solid_pos, Nx_up, Ny_up);
        save_int_array_to_npy("touch_solid_neg.npy", touch_solid_neg, Nx_up, Ny_up);
        save_int_array_to_npy("touch_solid.npy", touch_solid, Nx_up, Ny_up);
        save_int_array_to_npy("touch_void.npy", touch_void, Nx_up, Ny_up);
        save_int_array_to_npy("pix_solid.npy", pix_solid, Nx_up, Ny_up);
        //exit(EXIT_FAILURE);
    }
    
    free(touch_solid_pos);
    free(touch_solid_neg);
        
    // Compute reference convolutions on delta
    int* refconv0 = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    int* refconv1 = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    int* refconv2 = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    refconv0[0] = 1;
    refconv1[0] = 1;
    refconv2[0] = 1;
    
    temp = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    binary_convolve(
        refconv0,
        temp,
        brush,
        Nx_up,
        Ny_up,
        brush_size,
        1
    );
    for (int i = 0; i < Nx_up * Ny_up; ++i)
        refconv0[i] = temp[i];
    
    free(temp);
        
    temp = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    binary_convolve(
        refconv1,
        temp,
        brush_conv_shape,
        Nx_up,
        Ny_up,
        brush_size_conv,
        1
    );
    for (int i = 0; i < Nx_up * Ny_up; ++i)
        refconv1[i] = temp[i];
    
    free(temp);
    
    temp = (int*)calloc(Nx_up * Ny_up, sizeof(int));
    binary_convolve(
        refconv2,
        temp,
        brush_conv_shape2,
        Nx_up,
        Ny_up,
        brush_size_conv2,
        1
    );
    for (int i = 0; i < Nx_up * Ny_up; ++i)
        refconv2[i] = temp[i];
    
    free(temp);
    
    free(brush);
    free(brush_plus4);
    free(brush_conv_shape);
    free(brush_conv_shape2);
    
    if (debug) {
        save_int_array_to_npy("refconv0.npy", refconv0, Nx_up, Ny_up);
        save_int_array_to_npy("refconv1.npy", refconv1, Nx_up, Ny_up);
        save_int_array_to_npy("refconv2.npy", refconv2, Nx_up, Ny_up);
        //exit(EXIT_FAILURE);
    }

    int* result = main_loop(
        score_solid,
        pix_solid,
        touch_solid,
        touch_void,
        refconv0,
        refconv1,
        refconv2,
        symmetry,
        periodic,
        brush_size,
        dim,
        Nx_up,
        Ny_up
    );
    
    free(touch_solid);
    free(touch_void);
    free(pix_solid);
    free(score_solid);
    free(refconv0);
    free(refconv1);
    free(refconv2);
    
    if (result == NULL) {
        char filename[1024];
        const char* home_dir = getenv("HOME");
        if (home_dir) {
            snprintf(filename, sizeof(filename), "%s/make_feasible_debug_weight.npy", home_dir);
        } else {
            snprintf(filename, sizeof(filename), "make_feasible_debug_weight.npy");
        }
        save_float_array_to_npy(filename, weight, Nx, Ny);
        return NULL;
    }
    
    int* result01 = (int*)malloc(Nx_up * Ny_up * sizeof(int));
    for (int i = 0; i < Nx_up * Ny_up; ++i)
        result01[i] = result[i] == 1 ? 1 : 0;

    free(result);

    return result01;
}