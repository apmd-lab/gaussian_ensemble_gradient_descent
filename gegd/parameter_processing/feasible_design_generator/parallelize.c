#define PY_ARRAY_UNIQUE_SYMBOL MY_MODULE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <Python.h>

#include "parallelize.h"
#include "make_feasible.h"

typedef struct {
    float* input_3d;
    int* output_3d;
    int start_idx;
    int end_idx;
    int Nx;
    int Ny;
    int brush_size;
    int periodic;
    int symmetry;
    int dim;
    int upsample_ratio;
    int error_flag;
} ThreadPoolArgs;

void* thread_worker_pool(void* arg) {
    ThreadPoolArgs* args = (ThreadPoolArgs*)arg;
    
    int input_slice_size = args->Nx * args->Ny;
    int output_slice_size = args->Nx * args->Ny * args->upsample_ratio * args->upsample_ratio;
    
    for (int n = args->start_idx; n < args->end_idx; ++n) {
        int* result = make_feasible(
            &args->input_3d[n * input_slice_size],
            args->Nx,
            args->Ny,
            args->brush_size,
            args->periodic,
            args->symmetry,
            args->dim,
            args->upsample_ratio
        );

        if (result == NULL) {
            fprintf(stderr, "Thread failed during make_feasible.\n");
            args->error_flag = 1;
            pthread_exit(NULL);
        }

        int* out_slice = &args->output_3d[n * output_slice_size];
        for (int i = 0; i < output_slice_size; ++i) {
            out_slice[i] = result[i];
        }
        free(result);
    }
    pthread_exit(NULL);
}

PyObject* make_feasible_parallel(PyObject* self, PyObject* args) {
    PyArrayObject *input_obj;
    int brush_size, periodic, symmetry, dim, upsample_ratio, num_threads;

    if (!PyArg_ParseTuple(args, "Oiiiiii", &input_obj, &brush_size,
                          &periodic, &symmetry, &dim, &upsample_ratio, &num_threads)) {
        return NULL;
    }

    float* input_data = (float*)PyArray_DATA(input_obj);
    npy_intp* shape = PyArray_SHAPE(input_obj);

    int Nsample = (int)shape[0];
    int Nx = (int)shape[1];
    int Ny = (int)shape[2];

    // Output array
    npy_intp dims[3] = {Nsample, Nx * upsample_ratio, Ny * upsample_ratio};
    PyObject* output_obj = PyArray_SimpleNew(3, dims, NPY_INT);
    int* output_data = (int*)PyArray_DATA((PyArrayObject*)output_obj);
    
    if (num_threads > Nsample) num_threads = Nsample;
    
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadPoolArgs* thread_args = (ThreadPoolArgs*)malloc(num_threads * sizeof(ThreadPoolArgs));
    
    int chunk = (Nsample + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = (start + chunk > Nsample) ? Nsample : start + chunk;
        
        thread_args[t] = (ThreadPoolArgs){
            .input_3d = input_data,
            .output_3d = output_data,
            .start_idx = start,
            .end_idx = end,
            .Nx = Nx,
            .Ny = Ny,
            .brush_size = brush_size,
            .periodic = periodic,
            .symmetry = symmetry,
            .dim = dim,
            .upsample_ratio = upsample_ratio,
            .error_flag = 0,
        };
        
        pthread_create(&threads[t], NULL, thread_worker_pool, &thread_args[t]);
    }
    
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
    
    int error_occurred = 0;
    for (int t = 0; t < num_threads; ++t) {
        if (thread_args[t].error_flag) {
            error_occurred = 1;
        }
    }
    
    free(threads);
    free(thread_args);

    if (error_occurred) {
        Py_DECREF(output_obj);
        PyErr_SetString(PyExc_RuntimeError, "Error in make_feasible (infinite loop detected)");
        return NULL;
    }

    return output_obj;
}

static PyMethodDef FeasibleGeneratorMethods[] = {
    {"make_feasible_parallel", make_feasible_parallel, METH_VARARGS, "Make Feasible Parallel"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef FeasibleGeneratorModule = {
    PyModuleDef_HEAD_INIT,
    "make_feasible_parallel",
    NULL,
    -1,
    FeasibleGeneratorMethods
};

PyMODINIT_FUNC PyInit_fdg(void) {
    import_array();
    return PyModule_Create(&FeasibleGeneratorModule);
}