#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <Python.h>
#include <string.h>

#include "utils.h"

#define IDX(i, j, W) ((i) * (W) + (j))

void roll2d(const int* src,
            int* dest,
            int dx,
            int dy,
            int Nx,
            int Ny) {
            
    for (int i = 0; i < Nx; ++i) {
        int ii = (i - dx + Nx) % Nx;
        for (int j = 0; j < Ny; ++j) {
            int jj = (j - dy + Ny) % Ny;
            dest[IDX(i, j, Ny)] = src[IDX(ii, jj, Ny)];
        }
    }
}

void convolve(const float* input,
              float* output,
              const int* brush,
              int Nx,
              int Ny,
              int brush_size,
              int periodic) {
                    
    int r_c = (brush_size - 1) / 2;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            float val = 0.0f;
            for (int bi = 0; bi < brush_size; ++bi) {
                for (int bj = 0; bj < brush_size; ++bj) {
                    int ii = i + bi - r_c;
                    int jj = j + bj - r_c;
                    
                    if (periodic) {
                        ii = (ii + Nx) % Nx;
                        jj = (jj + Ny) % Ny;
                    } else {
                        if (ii < 0) ii = -ii - 1;
                        if (ii >= Nx) ii = 2 * Nx - ii - 1;
                        if (jj < 0) jj = -jj - 1;
                        if (jj >= Ny) jj = 2 * Ny - jj - 1;
                    }
                    
                    if (ii >= 0 && ii < Nx && jj >= 0 && jj < Ny) {
                        val += input[IDX(ii, jj, Ny)] * brush[IDX(bi, bj, brush_size)];
                    } 
                }
            }
            output[IDX(i, j, Ny)] = val;
        }
    }
}

void binary_convolve(const int* input,
                     int* output,
                     const int* brush,
                     int Nx,
                     int Ny,
                     int brush_size,
                     int periodic) {
                    
    int r_c = (brush_size - 1) / 2;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int val = 0;
            for (int bi = 0; bi < brush_size; ++bi) {
                for (int bj = 0; bj < brush_size; ++bj) {
                    int ii = i + bi - r_c;
                    int jj = j + bj - r_c;
                    
                    if (periodic) {
                        ii = (ii + Nx) % Nx;
                        jj = (jj + Ny) % Ny;
                    }
                    
                    if (ii >= 0 && ii < Nx && jj >= 0 && jj < Ny) {
                        val |= input[IDX(ii, jj, Ny)] & brush[IDX(bi, bj, brush_size)];
                        if (val) break;
                    }
                }
                if (val) break;
            }
            output[IDX(i, j, Ny)] = val;
        }
    }
}

void zoom_nearest(const float* input,
                  float*output,
                  int Nx_in,
                  int Ny_in,
                  int Nx_out,
                  int Ny_out,
                  int upsample_ratio,
                  int periodic) {
                  
    for (int i = 0; i < Nx_out; ++i) {
        int orig_i = floor(i / upsample_ratio);
        if (periodic)
            orig_i = (orig_i + Nx_in) % Nx_in;
            
        for (int j = 0; j < Ny_out; ++j) {
            int orig_j = floor(j / upsample_ratio);
            if (periodic)
                orig_j = (orig_j + Ny_in) % Ny_in;
                
            output[IDX(i, j, Ny_out)] = input[IDX(orig_i, orig_j, Ny_in)];
        }
    }
}

void pad_2d_constant(const float* input,
                     float* output,
                     int Nx,
                     int Ny,
                     int pad_x,
                     int pad_y,
                     float pad_value) {
                       
    int Nx_padded = Nx + 2 * pad_x;
    int Ny_padded = Ny + 2 * pad_y;

    for (int i = 0; i < Nx_padded; ++i) {
        for (int j = 0; j < Ny_padded; ++j) {
            if (i >= pad_x && i < Nx + pad_x && j >= pad_y && j < Ny + pad_y) {
                output[IDX(i, j, Ny_padded)] = input[IDX(i - pad_x, j - pad_y, Ny)];
            } else {
                output[IDX(i, j, Ny_padded)] = pad_value;
            }
        }
    }
}

void crop_2d(const int* input,
             int* output,
             int Nx,
             int Ny,
             int pad_x,
             int pad_y) {
             
    int Nx_crop = Nx - 2 * pad_x;
    int Ny_crop = Ny - 2 * pad_y;

    if (Nx_crop <= 0 || Ny_crop <= 0) return;
    
    size_t row_bytes = Ny_crop * sizeof(int);

    for (int i = 0; i < Nx_crop; ++i) {
        const int* src = input + (i + pad_x) * Ny + pad_y;
        int* dst = output + i * Ny_crop;
        memcpy(dst, src, row_bytes);
    }
}

void save_int_array_to_npy(const char* filename, int* data, int Nx, int Ny) {
    npy_intp dims[2] = {Nx, Ny};
    PyObject* array = PyArray_SimpleNewFromData(2, dims, NPY_INT, data);
    PyObject* np = PyImport_ImportModule("numpy");
    PyObject* save_func = PyObject_GetAttrString(np, "save");
    PyObject* py_filename = PyUnicode_FromString(filename);
    PyObject* args = PyTuple_Pack(2, py_filename, array);
    PyObject_CallObject(save_func, args);
    Py_DECREF(array);
    Py_DECREF(save_func);
    Py_DECREF(py_filename);
    Py_DECREF(args);
    Py_DECREF(np);
}

void save_float_array_to_npy(const char* filename, float* data, int Nx, int Ny) {
    npy_intp dims[2] = {Nx, Ny};
    PyObject* array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, data);
    PyObject* np = PyImport_ImportModule("numpy");
    PyObject* save_func = PyObject_GetAttrString(np, "save");
    PyObject* py_filename = PyUnicode_FromString(filename);
    PyObject* args = PyTuple_Pack(2, py_filename, array);
    PyObject_CallObject(save_func, args);
    Py_DECREF(array);
    Py_DECREF(save_func);
    Py_DECREF(py_filename);
    Py_DECREF(args);
    Py_DECREF(np);
}