import numpy as np
from numba import config

config.CUDA_ENABLE_PYNVJITLINK = 1

import warnings
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

import poc_pyb

N = 32  # or warpsize


@cuda.jit
def vadd(i, a, b, c):
    c[i] = a[i] + b[i]


@cuda.jit
def vector_add_gpu(a, b, c):
    vadd(cuda.threadIdx.x, a, b, c)


s = poc_pyb.FunctionSupport()
s.print()
a, b, c = s.A(), s.B(), s.C()
print(c)
vector_add_gpu[1, N](a, b, c)
print(c)
a += 11
s.print()
