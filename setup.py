from setuptools import setup, find_packages, Extension
import numpy as np

fdg = Extension(
    'gegd.parameter_processing.feasible_design_generator.fdg',              # import name
    sources=[
        'gegd/parameter_processing/feasible_design_generator/make_touch.c',
        'gegd/parameter_processing/feasible_design_generator/fill_required_pixels.c',
        'gegd/parameter_processing/feasible_design_generator/apply_symmetry_float.c',
        'gegd/parameter_processing/feasible_design_generator/main_loop.c',
        'gegd/parameter_processing/feasible_design_generator/find_index_max.c',
        'gegd/parameter_processing/feasible_design_generator/make_feasible.c',
        'gegd/parameter_processing/feasible_design_generator/parallelize.c',
        'gegd/parameter_processing/feasible_design_generator/utils.c',
        'gegd/parameter_processing/feasible_design_generator/touch2pix.c',
        'gegd/parameter_processing/feasible_design_generator/apply_symmetry_int.c',
    ],  # C source files
    include_dirs=[np.get_include(), # allows inclusion of numpy headers (e.g. numpy/arrayobject.h)
                  'gegd/parameter_processing/feasible_design_generator'],       # where to find headers
    extra_compile_args=["-O3", "-pthread"],
    # libraries=[...],            # any extra .so/.a you need to link
    # library_dirs=[...],         # where to find those libs
)

setup(name='gegd',                # Package name
      version='0.5.2',                         # Package version
      author='Seokhwan Min',                      # Your name
      author_email='petermsh513@gmail.com',   # Your email address
      description='Ensemble-based global search algorithm for freeform topology optimization.',  # Short description
      long_description=open('README.md').read(),  # Reads the long description from README.md
      long_description_content_type='text/markdown',
      url='https://github.com/apmd-lab/gaussian_ensemble_gradient_descent',  # URL to the package's GitHub repo
      packages=find_packages(exclude=['runfiles']),  # Automatically find packages; exclude tests/docs directories
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', 
                   'Operating System :: Unix',
                   'Topic :: Scientific/Engineering :: Mathematics',
                   ],
      include_package_data=True,
      ext_modules=[fdg],
      zip_safe=False,
      )