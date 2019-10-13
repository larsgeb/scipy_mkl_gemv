import scipy
import scipy.sparse
import numpy
import time
import pandas
import pytest
import pytest_harvest
from collections import OrderedDict
import matplotlib.pyplot as plt

# Project import
from mkl_interface import sparse_gemv


# Parameter k should only linearly increase computation time, for both methods, equally,
# as it is just repeated application of the algorithms. Parameter j is varied together
# with i within the test.
@pytest.mark.parametrize("i", numpy.logspace(1, 2, 2, dtype="int"))
@pytest.mark.parametrize("j_mult", [0.01, 0.1, 1, 10, 100])
@pytest.mark.parametrize("k", [int(1)])  # numpy.linspace(1, 100, 10, dtype="int"))
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("sparsity", numpy.logspace(-5, -1, 3, dtype="float32"))
def test_gemv(
    i: int,
    j_mult: float,
    k: int,
    precision: str,
    sparsity: float,
    results_bag,
    repeat_test: int = 50,
):

    # Test for a square matrix (i=j), a fat matrix (j = 0.1 / 0.01 i) and a
    # tall matrix (j = 10 / 100 i).
    j = int(j_mult * i)

    # if j < 1 or i < 1:
    #     pytest.skip("skipping this test, matrix has a dimension 0")

    # Don't fail the test if the requested matrix doesn't fit in memory
    try:
        A = scipy.sparse.random(i, j, density=sparsity, format="csr", dtype=precision)
    except MemoryError:
        pytest.skip("skipping this test for limited memory reasons")
        return 0

    # Create the vector
    x = numpy.ones((j, k), dtype=precision)

    # Test MKL implementation
    t0 = time.time()
    for iteration in range(repeat_test):
        # Not really fair, as object creation is done within the function
        result_1 = sparse_gemv(A, x)
    t1 = time.time()
    time_mkl = t1 - t0

    # Test SciPy implementation
    t0 = time.time()
    for iteration in range(repeat_test):
        result_2 = A.dot(x)
    t1 = time.time()
    time_native = t1 - t0

    # Check that the result are close
    assert numpy.allclose(result_1, result_2)

    # Store the results for visualization
    results_bag.time_native = time_native
    results_bag.time_mkl = time_mkl
    results_bag.relative_performance = time_mkl / time_native
    results_bag.i = i
    results_bag.j = j
    results_bag.k = k
    results_bag.precision = precision
    results_bag.sparsity = sparsity

    return 0


results_bag = pytest_harvest.create_results_bag_fixture("store", name="results_bag")


@pytest.fixture(scope="session", autouse=True)
def store(request):
    # setup: init the store
    store = OrderedDict()
    yield store

    # teardown: here you can collect all test results
    print()
    results = [store["results_bag"][test_key] for test_key in store["results_bag"]]
    df = pandas.DataFrame(results)

    # We want separate lines for every sparsity
    sparsities = df.sparsity.unique()
    # Remove nans from failed tests
    sparsities = sparsities[numpy.logical_not(numpy.isnan(sparsities))]

    plt.figure(figsize=(15, 7))

    # Plot the 32 bits results

    precisions = ["float32", "float64"]
    for i_precision in range(2):
        plt.subplot(1, 2, i_precision + 1)

        precision = precisions[i_precision]

        # Iterate over sparsities
        for i_sparsity, sparsity in enumerate(sparsities):

            result_per_sparsity = df[
                (df["precision"] == precision) & (df["sparsity"] == sparsity)
            ]

            matrix_elements = result_per_sparsity["i"] * result_per_sparsity["j"]

            sorted_unique_elements = sorted(set(matrix_elements))

            # If there is a matrix with zero elements, remove it from array
            try:
                sorted_unique_elements.remove(0)
            except ValueError:
                pass

            x = numpy.array(sorted_unique_elements)
            y_mkl = numpy.empty_like(x, dtype=numpy.float64)
            y_scp = numpy.empty_like(x, dtype=numpy.float64)

            error_mkl = numpy.empty_like(x, dtype=numpy.float64)
            error_scp = numpy.empty_like(x, dtype=numpy.float64)

            # iterate over elements

            it = numpy.nditer(x, flags=["f_index"])
            while not it.finished:
                result_per_sparsity_per_elements = df[
                    (df["precision"] == precision)
                    & (df["sparsity"] == sparsity)
                    & (df["i"] * df["j"] == it[0])
                ]

                y_mkl[it.index] = numpy.mean(
                    result_per_sparsity_per_elements["time_mkl"]
                )
                y_scp[it.index] = numpy.mean(
                    result_per_sparsity_per_elements["time_native"]
                )
                error_mkl[it.index] = numpy.std(
                    result_per_sparsity_per_elements["time_mkl"]
                )
                error_scp[it.index] = numpy.std(
                    result_per_sparsity_per_elements["time_native"]
                )

                it.iternext()

            next_color = plt.gca()._get_lines.get_next_color()

            plt.loglog(
                x, y_scp, "--", color=next_color, label=f"SCP, sparsity {sparsity:.2}"
            )
            plt.errorbar(x, y_scp, yerr=error_scp, linestyle=":", color=next_color)
            plt.loglog(
                x, y_mkl, "-", color=next_color, label=f"MKL, sparsity {sparsity:.2}"
            )
            plt.errorbar(x, y_mkl, yerr=error_mkl, linestyle="-", color=next_color)

        plt.legend()
        plt.xlabel("Matrix elements")
        plt.ylabel("Runtime")
        plt.title(f"{precision} results")

    plt.tight_layout()

    plt.savefig("results.svg")

