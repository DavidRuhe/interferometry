#!/bin/sh
rm *.c
rm *.so

ls
/home/david/miniconda3/envs/py2/bin/python setup.py build_ext --inplace

/home/david/miniconda3/envs/py2/bin/python make_image.py