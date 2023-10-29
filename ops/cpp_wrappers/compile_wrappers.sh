#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
rm -r build
rm *.so
python setup.py build_ext --inplace
cd ..
