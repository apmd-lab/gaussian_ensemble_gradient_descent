#ifndef UTILS_H
#define UTILS_H

void roll2d(const int* src,
            int* dest,
            int dx,
            int dy,
            int Nx,
            int Ny);

void convolve(const float* input,
              float* output,
              const int* brush,
              int Nx,
              int Ny,
              int brush_size,
              int periodic);

void binary_convolve(const int* input,
                     int* output,
                     const int* brush,
                     int Nx,
                     int Ny,
                     int brush_size,
                     int periodic);

void zoom_nearest(const float* input,
                  float*output,
                  int Nx_in,
                  int Ny_in,
                  int Nx_out,
                  int Ny_out,
                  int upsample_ratio,
                  int periodic);

void pad_2d_constant(const float* input,
                     float* output,
                     int Nx,
                     int Ny,
                     int pad_x,
                     int pad_y,
                     float pad_value);

void crop_2d(const int* input,
             int* output,
             int Nx,
             int Ny,
             int pad_x,
             int pad_y);

void save_int_array_to_npy(const char* filename, int* data, int Nx, int Ny);

void save_float_array_to_npy(const char* filename, float* data, int Nx, int Ny);

#endif