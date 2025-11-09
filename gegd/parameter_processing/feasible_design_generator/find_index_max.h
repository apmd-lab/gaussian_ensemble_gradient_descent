#ifndef FIND_INDEX_MAX_H
#define FIND_INDEX_MAX_H

#include <Python.h>

int* find_index_max(int symmetry,
                    int periodic,
                    int brush_size,
                    int dim,
                    int Nx,
                    int Ny,
                    float* score,
                    int* touch,
                    float max_val);

#endif