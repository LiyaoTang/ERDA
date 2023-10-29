#/bin/bash
rm *.o *.so

nvcc knnquery_cuda_kernel.cu -o knnquery_cuda_kernel.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_INC=$(python -c "import os; print(os.sep.join('`which nvcc`'.split(os.sep)[:-2] + ['include'])) ")

TF_IS_OLD=$(python -c 'import tensorflow as tf; print(int(float(".".join(tf.__version__.split(".")[:2])) < 1.15))')
USE_CXX11_ABI=$TF_IS_OLD  # avoid undefined behavior due to tf cpp API change

echo
echo CUDA_INC=$CUDA_INC
echo TF_IS_OLD=$TF_IS_OLD, USE_CXX11_ABI=$USE_CXX11_ABI  # using g++ => automatically support cxx11

g++ -std=c++11 tf_knnquery.cpp knnquery_cuda_kernel.cu.o \
    -o tf_knn.so \
    -shared -fPIC \
    -I$CUDA_INC \
    -I$TF_INC -I$TF_INC/external/nsync/public \
    -L$TF_LIB -ltensorflow_framework \
    -O2 -D_GLIBCXX_USE_CXX11_ABI=$USE_CXX11_ABI

