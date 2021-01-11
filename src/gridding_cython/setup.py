from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

print('hi')

# include_gsl_dir = "/home/david/miniconda3/include/gsl/"
# include_gsl_dir = "/home/david/miniconda3/include/"
# include_gsl_dir = "/usr/include/gsl/"
# lib_gsl_dir = "/home/david/minconda3/lib/"

# python setup.py build_ext --inplace

# Note that I need to include gslcblas otherwise I get import errors!!!
ext = Extension("gridding", ["gridding.pyx"], include_dirs=\
    [numpy.get_include()],\
)

setup(ext_modules=[ext], cmdclass = {'build_ext': build_ext})