import cython
cimport numpy as cnp
import numpy as np

ctypedef np.float32_t DTYPE_t

cdef extern from "convolve.h":
    double convolve_c(double *counts, const double *disk, const double *psfs,
                      const int nrows, const int ncols);

# create the wrapper code, with numpy type annotations
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_convolve(cnp.ndarray[DTYPE_t, ndim=2] disk, np.ndarray[DTYPE_t, ndim=4] psfs):
    cdef np.ndarray[DTYPE_t, ndim=2] counts = np.zeros_like(disk, dtype=DTYPE_t)

    convolve_c(<double*> counts.data, <double*> disk.data, <double*> psfs.data, <int> disk.shape[0], <int> disk.shape[1])

    return counts
