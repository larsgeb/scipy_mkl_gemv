# Tiny MKL wrapper for SciPy CSR GEMV

This package is intended as a tiny wrapper around MKL's sparse matrix-vector products. 

I noticed that SciPy's dot operator for a sparse CSR matrix together with a dense Numpy vector is not multithreaded in the SciPy implementation. Within MKL, the [operation does exist](https://software.intel.com/en-us/mkl-developer-reference-c-mkl-cspblas-csrgemv) as: 
```cpp
mkl_cspblas_?csrgemv(const char *transa , const MKL_INT *m , const float *a , const MKL_INT *ia , const MKL_INT *ja , const float *x , float *y );
```
The function itself is deprecated as off the time of writing, MKL giving the preference to Inspector-Executor type calls. However, for typical large sparse float type matrices, the implementation in this repo works fine, and actually faster than SciPy's. The implementation is based off of [this StackOverflow answer](https://stackoverflow.com/a/23294826/6848887), with some logic for performing both `float32` and `float64` general matrix-vector product.

## Usage

First make sure that the MKL libraries are properly installed and accesible from the shell where you will execute your Python script/notebook/interactive session. If you've installed MKL in `/opt/intel` (the default) this easily done using:

```bash
export INTEL_COMPILERS_AND_LIBS=/opt/intel/compilers_and_libraries/linux
source $INTEL_COMPILERS_AND_LIBS/mkl/bin/mklvars.sh intel64
```

Now simply put the `mkl_interface.py` file in the directory where you will run your program such that you can import it in the following way:

```python
from mkl_interface import sparse_gemv
```

and use the function `sparse_gemv(scipy.sparse.csr_matrix A, numpy.ndarray x)` form anywhere. Be mindful that the precision of `A` is used to perform the matrix-vector product, i.e. x is implicitly converted (its type will probably be altered outside the function scope too).

## Benchmark results

The implementaiton performace is tested for both 'tall' (more rows) and 'fat' (more columns) matrices, but perfomance for both MKL and SciPy implementations is affected neglegibly. The common denominator for perfomance is precision (float size), percentage of non-zero elements (sparsity) and total non-zero elements. A test can be seen below on randomly generated sparse matrices. 

![here](results_workstation.svg)

Results generated on a 18-core 128gb workstation. Execution times should be seen as relative, as each test is run 50 times. Error bars are due to varying shapes of matrices (tall, fat, square, but all with same number of elements (rows × columns).

Clearly, the MKL implementation should not always be favoured over SciPy's native one. The high overhead of the MKL implementation (mainly acquiring pointers and extra memory) does mean that for small problems, SciPy's implementation is more efficient. For any high dimension, MKL should be favourable.

It should be noted that the tests performed in this repo are limited in size due to the random sparse matrix generator for SciPy, which ran out of memory on a 128gb system for a matrix size of 10⁸ total elements.

The test results can be recreated using specifically `pytest`, `pytest-harvester` and all the other imported libraries at the top of the `test_sparse.py` file.
