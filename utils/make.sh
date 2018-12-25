#!/bin/bash
# **********************************************************
# * Author        : lingteng qiu
# * Email         : 1259738366@qq.com
# * Create time   : 2018-12-22 09:59
# * Last modified : 2018-12-22 09:59
# * Filename      : make.sh
# * Description   : compile nms_kernel.cu
# **********************************************************
nvcc --gpu-architecture=compute_61 --gpu-code=compute_61  -DGPU -I/usr/local/cuda/include/ -DCUDNN --compiler-options "-Wall -Wfatal-errors -Ofast -DOPENCV -DGPU -DCUDNN -fPIC" -c nms_kernel.cu -o nms_kernel.o

gcc -shared -o libnms.so nms_kernel.o -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn
