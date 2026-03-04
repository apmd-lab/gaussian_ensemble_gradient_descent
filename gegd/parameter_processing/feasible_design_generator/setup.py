from setuptools import setup, Extension
import numpy

setup(
    name="fdg",
    version="0.1",
    ext_modules=[
        Extension(
            "fdg",
            sources=["apply_symmetry_int.c",
                     "apply_symmetry_float.c",
                     "touch2pix.c",
                     "make_touch.c",
                     "utils.c",
                     "fill_required_pixels.c",
                     "find_index_max.c",
                     "main_loop.c",
                     "make_feasible.c",
                     "parallelize.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-pthread"],
        )
    ],
)

#gdb --args python ../test_script_parallel.py
#python setup.py build_ext --inplace --force