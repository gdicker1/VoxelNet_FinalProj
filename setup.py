#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# setup.py
# Builds the cython file
# by G. Dylan Dickerson

from distutils.core import setup
from Cython.Build import cythonize

setup(
	name='box overlaps',
	ext_modules=cythonize('./utils/box_overlaps.pyx')
)