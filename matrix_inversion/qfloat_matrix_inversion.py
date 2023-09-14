"""
This module implements a matrix inversion algorithm
for encrypted matrices of QFloats

The pure python version of the algorithm working on floats is also provided for clear understanding
and comparing, as well as testing functions when using this module as a main

author : Lcressot
"""

import numpy as np
import scipy as sc
import time
from qfloat import QFloat, SignedBinary, Zero
from concrete import fhe

Tracer = fhe.tracing.tracer.Tracer


"""
==============================================================================

Original LU matrix inversion algorithm operating on floats

==============================================================================
"""


def pivot_matrix(M):
    """
    Returns the pivoting matrix for M, used in Doolittle's method.
    """
    assert M.shape[0] == M.shape[1]
    n = M.shape[0]

    # Create an identity matrix, with floating point values
    id_mat = np.eye(n)

    # Rearrange the identity matrix such that the largest element of
    # each column of M is placed on the diagonal of M
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(M[i, j]))
        if j != row:
            # Swap the rows
            id_mat[[j, row]] = id_mat[[row, j]]

    return id_mat


def lu_decomposition(M):
    """
    Performs a LU Decomposition of M (which must be square)
    into PM = LU. The function returns P, L, and U.
    """
    assert M.shape[0] == M.shape[1]
    n = M.shape[0]

    # Create zero matrices for L and U
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Create the pivot matrix P and the multiplied matrix PM
    P = pivot_matrix(M)
    PM = P @ M

    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity
        L[j, j] = 1.0

        # u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
        for i in range(j + 1):
            s1 = np.dot(U[0:i, j], L[i, 0:i])
            U[i, j] = PM[i, j] - s1

        # l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
        for i in range(j + 1, n):  # Ljj is always 1
            s2 = np.dot(U[0:j, j], L[i, 0:j])
            L[i, j] = (PM[i, j] - s2) / U[j, j]

    # PM = LU
    P = np.transpose(P)

    # now M = PLU
    # Note that Ljj is always 1 for all j
    return P, L, U


def lu_inverse(P, L, U, debug=False):
    """
    Compute inverse using the P,L,U decomposition
    """
    n = L.shape[0]

    # Forward substitution: Solve L * Y = P * A for Y
    Y = np.zeros((n, n))
    for i in range(n):
        Y[i, 0] = P[i, 0] / L[0, 0]
        for j in range(1, n):
            Y[i, j] = P[i, j] - np.dot(L[j, :j], Y[i, :j])

    # Backward substitution: Solve U * X = Y for X
    X = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        X[i, -1] = Y[i, -1] / U[-1, -1]
        for j in range(n - 2, -1, -1):
            X[i, j] = (Y[i, j] - np.dot(U[j, j + 1 :], X[i, j + 1 :])) / U[j, j]

    if not debug:
        return np.transpose(X)
    else:
        return np.transpose(X), Y, X


def matrix_inverse(M):
    P, L, U = lu_decomposition(M)
    return lu_inverse(P, L, U)


def test_matrix_inverse(n, m=100):
    for i in range(m):
        M = np.random.uniform(0, 100, (n, n))
        M_inv = sc.linalg.inv(M)
        M_inv_2 = matrix_inverse(M)
        error = np.mean(np.abs(M_inv_2 - M_inv))
        assert error < 0.00001

    print("test_matrix_inverse OK")


"""
==============================================================================

QFloat version of LU matrix inversion algorithm, thus operating on QFloats

==============================================================================
"""


###############################################################################
#                                     UTILS
###############################################################################


def matrix_column(M, j):
    """
    Extract column from 2D list matrix
    """
    return [row[j] for row in M]


def transpose_2D_list(list2D):
    """
    Transpose a 2D-list matrix
    """
    return [list(row) for row in zip(*list2D)]


def map_2D_list(list2D, function):
    """
    Apply a function to a 2D list
    """
    return [[function(f) for f in row] for row in list2D]


def binary_list_matrix(M):
    """
    Convert a binary matrix M into a 2D-list matrix of SignedBinarys
    """
    return map_2D_list(
        [[M[i, j] for j in range(M.shape[1])] for i in range(M.shape[0])],
        lambda x: SignedBinary(x),
    )


def zero_list_matrix(n):
    """
    Creates a square list matrix of Zeros
    """
    return [[Zero() for j in range(n)] for i in range(n)]


def qfloat_list_dot_product(list1, list2):
    """
    Dot product of two QFloat lists
    """
    if len(list1) != len(list2):
        raise ValueError("Lists should have the same length.")

    result = list1[0] * list2[0]
    for i in range(1, len(list1)):
        result += list1[i] * list2[i]  # in place addition is supported

    # # Tensorized way:
    # multiplications = QFloat.multi_from_mul(
    #     list1, list2, None, None
    # )
    # result = multiplications[0]
    # for m in multiplications[1:]:
    #     result += m  # in place addition is supported

    return result


def qfloat_list_matrix_multiply(matrix1, matrix2):
    """
    Multiply two matrices of int or QFloats
    """
    # if len(matrix1[0]) != len(matrix2):
    #     raise ValueError(
    #           "Number of columns in matrix1 should match the number of rows in matrix2."
    #      )
    result = [[None] * len(matrix2[0]) for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            result[i][j] = qfloat_list_dot_product(
                matrix1[i], matrix_column(matrix2, j)
            )

    return result


def float_matrix_to_qfloat_arrays(M, qfloat_len, qfloat_ints, qfloat_base):
    """
    converts a float matrix to arrays representing qfloats
    """
    qfloat_list = [
        QFloat.from_float(f, qfloat_len, qfloat_ints, qfloat_base) for f in M.flatten()
    ]
    n = len(qfloat_list)
    qfloat_arrays = np.zeros((n, qfloat_len), dtype="int")
    qfloat_signs = np.zeros(n, dtype="int")
    for i in range(n):
        qfloat_arrays[i, :] = qfloat_list[i].to_array()
        qfloat_signs[i] = qfloat_list[i].sign

    return qfloat_arrays, qfloat_signs


def qfloat_arrays_to_qfloat_matrix(
    qfloat_arrays, qfloat_signs, qfloat_ints, qfloat_base
):
    """
    converts qfloats array and sign array to a QFloat matrix
    """
    n = int(np.sqrt(qfloat_arrays.shape[0]))
    qfloat_M = []
    index = 0
    for i in range(n):
        row = []
        for j in range(n):
            qfloat = QFloat(
                qfloat_arrays[index, :],
                qfloat_ints,
                qfloat_base,
                True,
                qfloat_signs[index],
            )
            index += 1
            row.append(qfloat)
        qfloat_M.append(row)

    return qfloat_M


def qfloat_and_signs_arrays_to_float_matrix(qfloat_arrays, qfloat_ints, qfloat_base):
    """
    converts qfloats and signs arrays to a float matrix
    """
    n = int(np.sqrt(qfloat_arrays.shape[0]))
    M = np.zeros((n, n))
    index = 0
    for i in range(n):
        for j in range(n):
            M[i, j] = QFloat(
                qfloat_arrays[index, :-1],
                qfloat_ints,
                qfloat_base,
                True,
                qfloat_arrays[index, -1],
            ).to_float()
            index += 1

    return M


def qfloat_matrix_to_arrays_and_signs(M, qfloat_len, qfloat_ints, qfloat_base):
    """
    converts a QFloat 2D-list matrix to integer arrays and signs
    """
    n = len(M)
    assert n == len(M[0])
    qfloat_arrays = fhe.zeros((n * n, qfloat_len + 1))  # add one for the sign
    index = 0
    for i in range(n):
        for j in range(n):
            if isinstance(M[i][j], QFloat):
                qfloat_arrays[index, :qfloat_len] = M[i][j].to_array()
                qfloat_arrays[index, qfloat_len] = M[i][j].sign
            elif isinstance(M[i][j], SignedBinary):
                qfloat_arrays[index, qfloat_ints - 1] = M[i][j].value
                qfloat_arrays[index, qfloat_len] = M[i][j].value
            elif isinstance(M[i][j], Zero):
                pass
            else:
                qfloat_arrays[index, qfloat_ints - 1] = M[i][j]
                qfloat_arrays[index, qfloat_len] = np.sign(M[i][j])
            index += 1

    return qfloat_arrays


###############################################################################
#                                   PIVOT MATRIX
###############################################################################


def qfloat_argmax(indices, qfloats):
    """
    Returns the index corresponding to the biggest QFloat in the list
    """
    max_qf = qfloats[0].copy()
    maxi = indices[0]
    for i in range(1, len(indices)):
        is_gt = qfloats[i] > max_qf
        max_qf._array = is_gt * qfloats[i]._array + (1 - is_gt) * max_qf._array
        maxi = is_gt * indices[i] + (1 - is_gt) * maxi

    return maxi


def qfloat_pivot_matrix(M):
    """
    Returns the pivoting matrix for M, used in Doolittle's method.
    M is a squared 2D list of QFloats
    """
    assert len(M) == len(M[0])  # assert this looks like a 2D list of a square matrix
    n = len(M)

    # Create identity matrix, with fhe integer values
    pivot_mat = fhe.zeros((n, n))
    for i in range(n):
        pivot_mat[i, i] = fhe.ones(1)[0]

    # Rearrange the identity matrix such that the largest element of
    # each column of M is placed on the diagonal of M
    temp_mat = fhe.zeros((n, n))
    for j in range(n - 1):
        # compute argmax of abs values in range(j,n)
        r = qfloat_argmax([i for i in range(j, n)], [abs(M[i][j]) for i in range(j, n)])

        # copy pivot_mat to temp_mat
        temp_mat[:] = pivot_mat[:]

        # swap rows in pivot_mat:
        # row j becomes row r:
        bsum = temp_mat[j, :] * (j == r)
        for i in range(j + 1, n):
            bsum += temp_mat[i, :] * (i == r)

        pivot_mat[j, :] = bsum[:]

        # row r becomes row j:
        for jj in range(j + 1, n):
            jj_eq_r = jj == r
            pivot_mat[jj, :] = (1 - jj_eq_r) * temp_mat[jj, :] + jj_eq_r * temp_mat[
                j, :
            ]

    return pivot_mat


###############################################################################
#                                   LU DECOMPOSITION
###############################################################################


def qfloat_lu_decomposition(M, qfloat_len, qfloat_ints, true_division=False):
    """
    Perform a LU decomposition of square (QFloats 2D-list) matrix M
    The function returns P, L, and U such that M = PLU.

    true_division: Wether to compute true divisions instead of multiplying by inverse.
    Setting it to True is more precise but slower
    """
    assert len(M) == len(M[0])
    n = len(M)

    # add __len__ function to Tracer for simplicity:
    def arrlen(self):
        return self.shape[0]

    Tracer.__len__ = arrlen

    # Initialize 2D-list matrices for L and U
    # (QFloats can be multiplied with integers, this saves computations)
    L = zero_list_matrix(n)
    U = zero_list_matrix(n)

    # Create the pivot matrix P and the multiplied matrix PM
    P = binary_list_matrix(qfloat_pivot_matrix(M))
    PM = qfloat_list_matrix_multiply(P, M)

    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity
        L[j][j] = SignedBinary(1)
        # u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
        for i in range(j + 1):
            if i > 0:
                s1 = qfloat_list_dot_product(
                    [U[k][j] for k in range(0, i)], [L[i][k] for k in range(0, i)]
                )
                U[i][j] = PM[i][j] + s1.neg()
            else:
                U[i][j] = PM[i][j].copy()

        # l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
        if not true_division:
            # compute one precise invert division and then use multiplications
            # this is faster but less precise
            inv_Ujj = U[j][j].invert(1, qfloat_len, 0)
        for i in range(j + 1, n):
            if j > 0:
                s2 = qfloat_list_dot_product(
                    [U[k][j] for k in range(0, j)], [L[i][k] for k in range(0, j)]
                )
                # L[i][j] = (PM[i][j] - s2) * inv_Ujj
                if true_division:
                    L[i][j] = (PM[i][j] + s2.neg()) / U[j][j]
                else:
                    L[i][j] = QFloat.from_mul(
                        (PM[i][j] + s2.neg()), inv_Ujj, qfloat_len, qfloat_ints
                    )

            else:
                # L[i][j] = PM[i][j] / U[j][j]
                if true_division:
                    L[i][j] = PM[i][j] / U[j][j]
                else:
                    L[i][j] = QFloat.from_mul(
                        PM[i][j], inv_Ujj, qfloat_len, qfloat_ints
                    )

    # for now, PM = LU
    P = transpose_2D_list(P)
    # now M = PLU (Note that Ljj is always 1 for all j)
    return P, L, U


###############################################################################
#                                   LU INVERSE
###############################################################################


def qfloat_lu_inverse(
    P, L, U, qfloat_len, qfloat_ints, true_division=False, debug=False
):
    """
    Compute inverse using the P,L,U decomposition (operating on QFloats 2D-lists)

    true_division: Wether to compute true divisions instead of multiplying by inverse.
    Setting it to True is more precise but slower
    """
    n = len(L)

    # Forward substitution: Solve L * Y = P * A for Y
    Y = zero_list_matrix(n)
    for i in range(n):
        # Y[i, 0] = P[i, 0] / L[0, 0]
        Y[i][0] = P[i][
            0
        ].copy()  # dividing by diagonal values of L is useless because they are equal to 1
        for j in range(1, n):
            # Y[i, j] = (P[i, j] - np.dot(L[j, :j], Y[i, :j]))
            Y[i][j] = P[i][j] - qfloat_list_dot_product(
                [L[j][k] for k in range(j)], [Y[i][k] for k in range(j)]
            )

    # Backward substitution: Solve U * X = Y for X
    X = zero_list_matrix(n)

    # precompute inverse of U to make less divisions (simplify U before as we know its range)
    if not true_division:
        Ujj_inv = [U[j][j].invert(1, qfloat_len, 0) for j in range(n)]
        # Ujj_inv = QFloat.multi_invert(
        #     [U[j][j] for j in range(n)], 1, qfloat_len, 0
        # )
    for i in range(n - 1, -1, -1):
        # X[i, -1] = Y[i, -1] / U[-1, -1]
        if true_division:
            X[i][-1] = Y[i][-1] / U[-1][-1]
        else:
            X[i][-1] = QFloat.from_mul(Y[i][-1], Ujj_inv[-1], qfloat_len, qfloat_ints)

        for j in range(n - 2, -1, -1):
            # X[i, j] = (Y[i, j] - np.dot(U[j, j+1:], X[i, j+1:])) / U[j, j]
            temp = Y[i][j] - qfloat_list_dot_product(
                [U[j][k] for k in range(j + 1, n)], [X[i][k] for k in range(j + 1, n)]
            )
            if true_division:
                X[i][j] = temp / U[j][j]
            else:
                X[i][j] = QFloat.from_mul(temp, Ujj_inv[j], qfloat_len, qfloat_ints)

    if not debug:
        return transpose_2D_list(X)
    else:
        return transpose_2D_list(X), Y, X  # return Y and X for debug purpose


###############################################################################
#                               2x2 SHORCUT FORMULA
###############################################################################


def qfloat_inverse_2x2(qfloat_M, qfloat_len, qfloat_ints):
    """
    The inverse of a 2x2 matrix has a simple formula: M_inv = 1/det(M) * [[d,-b],[-c,a]]
    WARNING : The function is optimized for inputs in range [0,2^8] (see samplers)
    """
    [a, b] = qfloat_M[0]
    [c, d] = qfloat_M[1]

    ad = QFloat.from_mul(
        a, d, 2 * qfloat_ints + 3, 2 * qfloat_ints
    )  # produces a longer integer part
    bc = QFloat.from_mul(
        b, c, 2 * qfloat_ints + 3, 2 * qfloat_ints
    )  # produces a longer integer part

    det = ad + bc.neg()  # determinant of M in 17 bits

    det_inv = det.invert(
        1, qfloat_len, 0
    )  # computes invert with a lot of decimals but no integer part

    mul = lambda x, y: QFloat.from_mul(
        x, y, qfloat_len, qfloat_ints
    )  # multiply to output format
    M_inv = [
        [mul(d, det_inv), mul(b, det_inv).neg()],
        [mul(c, det_inv).neg(), mul(a, det_inv)],
    ]

    return M_inv


def qfloat_inverse_2x2_multi(qfloat_M, qfloat_len, qfloat_ints):
    """
    The inverse of a 2x2 matrix has a simple formula: M_inv = 1/det(M) * [[d,-b],[-c,a]]
    WARNING : The function is optimized for inputs in range [0,2^8] (see samplers)
    This function uses multi_from_mul for tensorization
    """
    [a, b] = qfloat_M[0]
    [c, d] = qfloat_M[1]

    [ad, bc] = QFloat.multi_from_mul(
        [a, b], [d, c], 2 * qfloat_ints + 3, 2 * qfloat_ints
    )  # produces a longer integer part

    det = ad + bc.neg()  # determinant of M in 17 bits

    det_inv = det.invert(
        1, qfloat_len, 0
    )  # computes invert with a lot of decimals but no integer part

    [mula, mulb, mulc, muld] = QFloat.multi_from_mul(
        [a, b, c, d], [det_inv] * 4, qfloat_len, qfloat_ints
    )  # produces a longer integer part

    M_inv = [
        [muld, mulb.neg()],
        [mulc.neg(), mula],
    ]

    return M_inv


###############################################################################
#                                   COMPLETE FUNCTION
###############################################################################


def qfloat_pivot(qfloat_arrays, qfloat_signs, params):
    """
    computes only the pivot matrix
    """
    [n, qfloat_len, qfloat_ints, qfloat_base, _] = params

    assert n * n == qfloat_arrays.shape[0]
    assert qfloat_len == qfloat_arrays.shape[1]

    # reconstruct the matrix of QFloats with encrypted values:
    qfloat_M = qfloat_arrays_to_qfloat_matrix(
        qfloat_arrays, qfloat_signs, qfloat_ints, qfloat_base
    )

    # compute the pivot matrix
    P = qfloat_pivot_matrix(qfloat_M)

    return P


def qfloat_lu_L(qfloat_arrays, qfloat_signs, params):
    """
    compute only PLU and returns L
    """
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params

    assert n * n == qfloat_arrays.shape[0]
    assert qfloat_len == qfloat_arrays.shape[1]

    # reconstruct the matrix of QFloats with encrypted values:
    qfloat_M = qfloat_arrays_to_qfloat_matrix(
        qfloat_arrays, qfloat_signs, qfloat_ints, qfloat_base
    )

    # compute the LU decomposition
    bin_P, qfloat_L, qfloat_U = qfloat_lu_decomposition(
        qfloat_M, qfloat_len, qfloat_ints, true_division
    )

    # break the resulting QFloats into arrays:
    qfloat_inv_arrays_L = qfloat_matrix_to_arrays_and_signs(
        qfloat_L, qfloat_len, qfloat_ints, qfloat_base
    )
    # qfloat_inv_arrays_U = qfloat_matrix_to_arrays_and_signs(
    #     qfloat_U, qfloat_len, qfloat_ints, qfloat_base
    # )

    return qfloat_inv_arrays_L


def qfloat_lu_U(qfloat_arrays, qfloat_signs, params):
    """
    compute only PLU and returns U
    """
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params

    assert n * n == qfloat_arrays.shape[0]
    assert qfloat_len == qfloat_arrays.shape[1]

    # reconstruct the matrix of QFloats with encrypted values:
    qfloat_M = qfloat_arrays_to_qfloat_matrix(
        qfloat_arrays, qfloat_signs, qfloat_ints, qfloat_base
    )

    # compute the LU decomposition
    bin_P, qfloat_L, qfloat_U = qfloat_lu_decomposition(
        qfloat_M, qfloat_len, qfloat_ints, true_division
    )

    # break the resulting QFloats into arrays:
    # qfloat_inv_arrays_L = qfloat_matrix_to_arrays_and_signs(
    #     qfloat_L, qfloat_len, qfloat_ints, qfloat_base
    # )
    qfloat_inv_arrays_U = qfloat_matrix_to_arrays_and_signs(
        qfloat_U, qfloat_len, qfloat_ints, qfloat_base
    )

    return qfloat_inv_arrays_U


def qfloat_matrix_inverse(
    qfloat_arrays, qfloat_signs, n, qfloat_len, qfloat_ints, qfloat_base, true_division
):
    """
    Compute the whole inverse
    """
    assert n * n == qfloat_arrays.shape[0]
    assert qfloat_len == qfloat_arrays.shape[1]

    # reconstruct the matrix of QFloats with encrypted values:
    qfloat_M = qfloat_arrays_to_qfloat_matrix(
        qfloat_arrays, qfloat_signs, qfloat_ints, qfloat_base
    )

    if n == 2:
        # use shortcut formula
        qfloat_Minv = qfloat_inverse_2x2_multi(qfloat_M, qfloat_len, qfloat_ints)

    else:
        # compute the LU decomposition
        bin_P, qfloat_L, qfloat_U = qfloat_lu_decomposition(
            qfloat_M, qfloat_len, qfloat_ints, true_division
        )

        # compute inverse from P L U
        qfloat_Minv = qfloat_lu_inverse(
            bin_P, qfloat_L, qfloat_U, qfloat_len, qfloat_ints, true_division
        )

    # break the resulting QFloats into arrays:
    qfloat_inv_arrays = qfloat_matrix_to_arrays_and_signs(
        qfloat_Minv, qfloat_len, qfloat_ints, qfloat_base
    )

    return qfloat_inv_arrays


"""
==============================================================================

Tests for functions in this file.
The funcitons above can be imported from another file

==============================================================================
"""


#######################################################################################

# ████████ ███████ ███████ ████████ ███████
#    ██    ██      ██         ██    ██
#    ██    █████   ███████    ██    ███████
#    ██    ██           ██    ██         ██
#    ██    ███████ ███████    ██    ███████

# tests for functions in this file.
# The above the functions can be imported from another file

#######################################################################################


def measure_time(function, descripton, verbose, *inputs):
    # Compute a function on inputs and return output along with duration
    print(descripton + " ...", end="", flush=True)
    print("\r", end="")
    start = time.time()
    output = function(*inputs)
    end = time.time()
    print(f"|  {descripton} : {end-start:.2f} s  |")
    return output, end - start


###############################################################################
#                                    PYTHON
###############################################################################


def test_qfloat_PLU_python(M, params):
    """
    Test the LU decomposition using QFloats, in python
    """
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params

    # convert it to QFloat arrays
    qfloat_arrays, qfloat_signs = float_matrix_to_qfloat_arrays(
        M, qfloat_len, qfloat_ints, qfloat_base
    )

    # reconstruct the matrix of QFloats
    qfloat_M = qfloat_arrays_to_qfloat_matrix(
        qfloat_arrays, qfloat_signs, qfloat_ints, qfloat_base
    )

    # compute the LU decomposition
    bin_P, qfloat_L, qfloat_U = qfloat_lu_decomposition(
        qfloat_M, qfloat_len, qfloat_ints, true_division
    )

    # convert bin_P from binaryValue to floats
    P = np.array(map_2D_list(bin_P, lambda x: x.value))

    # break the resulting QFloats into arrays:
    qfloat_arrays_L = qfloat_matrix_to_arrays_and_signs(
        qfloat_L, qfloat_len, qfloat_ints, qfloat_base
    )
    qfloat_arrays_U = qfloat_matrix_to_arrays_and_signs(
        qfloat_U, qfloat_len, qfloat_ints, qfloat_base
    )

    L = qfloat_and_signs_arrays_to_float_matrix(
        qfloat_arrays_L, qfloat_ints, qfloat_base
    )
    U = qfloat_and_signs_arrays_to_float_matrix(
        qfloat_arrays_U, qfloat_ints, qfloat_base
    )

    print(" PIVOT MATRIX\n============")
    print("QFloat P :")
    print(P)
    print(" ")

    P_, L_, U_ = lu_decomposition(M)
    print("PLU P :")
    print(P_)
    print("")

    print(" L MATRIX\n============")
    print("QFloat L :")
    print(L)
    print(" ")

    print("PLU L :")
    print(L_)
    print(" ")

    print(" U MATRIX\n============")
    print("QFloat U :")
    print(U)
    print(" ")

    print("PLU U :")
    print(U_)
    print(" ")


def run_qfloat_inverse_python(M, params):
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params

    # convert it to QFloat arrays
    qfloat_arrays, qfloat_signs = float_matrix_to_qfloat_arrays(
        M, qfloat_len, qfloat_ints, qfloat_base
    )

    output = qfloat_matrix_inverse(qfloat_arrays, qfloat_signs, *params)

    qfloat_res = qfloat_and_signs_arrays_to_float_matrix(
        output, qfloat_ints, qfloat_base
    )

    return qfloat_res


def test_qfloat_inverse_python(M, params):
    """
    Test the matrix inverse using QFloats, in python
    """
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params

    # convert it to QFloat arrays
    qfloat_arrays, qfloat_signs = float_matrix_to_qfloat_arrays(
        M, qfloat_len, qfloat_ints, qfloat_base
    )

    start = time.time()
    qfloat_res = run_qfloat_inverse_python(M, params)
    end = time.time()

    print("\n==== Test python ====")
    print("\nQFloat inverse :")
    print(qfloat_res)

    print("\nLU inverse :")
    print(matrix_inverse(M))

    print("\nScipy inv :")
    scres = sc.linalg.inv(M)
    print(scres)

    error = np.abs(qfloat_res - scres)
    print(f"\nAverage Error: {np.mean(error):.6f}")
    print(f"    Max Error: {np.max(error):.6f}")
    print(f"    Min Error: {np.min(error):.6f}")
    print(f"  Total Error: {np.sum(error):.6f}")

    print(f"\n|  Took : {end-start:.4f} s  |\n")


def debug_qfloat_inverse_python(sampler, params, verbose=False, N=100):
    """
    Debug the algorithm to look for big errors
    """
    errors = np.zeros(N)
    nerrors = 0
    for i in range(N):
        # print(i)
        # gen random matrix
        M = sampler()

        [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params

        # convert it to QFloat arrays
        qfloat_arrays, qfloat_signs = float_matrix_to_qfloat_arrays(
            M, qfloat_len, qfloat_ints, qfloat_base
        )

        qfloat_res = run_qfloat_inverse_python(M, params)

        scres = sc.linalg.inv(M)

        error = np.mean(np.abs(qfloat_res - scres))

        errors[i] = error

        if error > 1:
            nerrors += 1

            if not verbose:
                continue

            err = np.abs(qfloat_res - scres)
            print(f"\nAverage Error: {np.mean(err):.6f}")
            print(f"    Max Error: {np.max(err):.6f}")
            print(f"    Min Error: {np.min(err):.6f}")
            print(f"  Total Error: {np.sum(err):.6f}")

            # reconstruct the matrix of QFloats
            qfloat_M = qfloat_arrays_to_qfloat_matrix(
                qfloat_arrays, qfloat_signs, qfloat_ints, qfloat_base
            )

            # compute the LU decomposition
            bin_P, qfloat_L, qfloat_U = qfloat_lu_decomposition(
                qfloat_M, qfloat_len, qfloat_ints, true_division
            )

            # compute the inverse
            qfloat_Minv, qfloat_Y, qfloat_X = qfloat_lu_inverse(
                bin_P,
                qfloat_L,
                qfloat_U,
                qfloat_len,
                qfloat_ints,
                true_division,
                debug=True,
            )

            # convert QFloat mattrices to floats
            L = map_2D_list(qfloat_L, lambda x: x.to_float())
            U = map_2D_list(qfloat_U, lambda x: x.to_float())

            P_, L_, U_ = lu_decomposition(M)
            Minv_, Y_, X_ = lu_inverse(P_, L_, U_, debug=True)

            print("\nL")
            print(L)
            print(L_)
            print("\nU")
            print(U)
            print(U_)
            print("\nInv")
            print(qfloat_res)
            print(scres)

            Y = map_2D_list(qfloat_Y, lambda x: x.to_float())
            X = map_2D_list(qfloat_X, lambda x: x.to_float())

            print("\nX")
            print(X)
            print(X_)
            print("\nY")
            print(Y)
            print(Y_)

    print("mean error :", np.mean(errors))
    print("big error rate :" + str(100 * nerrors / N) + " %")


###############################################################################
#                                           FHE
###############################################################################


def compile_circuit(params, sampler, circuit_function, verbose=True):
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params

    inputset = []
    for i in range(100):
        M = sampler()
        qfloat_arrays, qfloat_signs = float_matrix_to_qfloat_arrays(
            M, qfloat_len, qfloat_ints, qfloat_base
        )
        inputset.append((qfloat_arrays, qfloat_signs))

    compiler = fhe.Compiler(
        lambda x, y: circuit_function(x, y, *params),
        {"x": "encrypted", "y": "encrypted"},
    )
    make_circuit = lambda: compiler.compile(
        inputset=inputset,
        configuration=fhe.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location=".keys",
            single_precision=False,
            show_graph=False,
            dataflow_parallelize=True,
        ),
        verbose=False,
    )

    circuit, time = measure_time(make_circuit, "Compiling", verbose)
    return circuit, time


def run_qfloat_circuit_fhe(
    circuit, M, qfloat_len, qfloat_ints, qfloat_base, simulate=False, raw_output=False, verbose=True
):
    # convert it to QFloat arrays
    qfloat_arrays, qfloat_signs = float_matrix_to_qfloat_arrays(
        M, qfloat_len, qfloat_ints, qfloat_base
    )

    simulation_time = None
    encryption_time = None
    running_time = None

    # Run FHE
    if not simulate:
        encrypted, encryption_time = measure_time(
            circuit.encrypt, "Encrypting", verbose, qfloat_arrays, qfloat_signs
        )
        run, running_time = measure_time(circuit.run, "Running", verbose, encrypted)
        decrypted = circuit.decrypt(run)

    else:
        decrypted, simulation_time = measure_time(
            circuit.simulate, "Simulating", verbose, qfloat_arrays, qfloat_signs
        )

    if raw_output:
        return decrypted, (simulation_time, encryption_time, running_time)

    result = qfloat_and_signs_arrays_to_float_matrix(
        decrypted, qfloat_ints, qfloat_base
    )

    if not simulate:
        return result, (encryption_time, running_time)
    else:
        return result, simulation_time


def test_qfloat_pivot_fhe(circuit, M, params, simulate=False):
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params
    qfloat_res, _ = run_qfloat_circuit_fhe(
        circuit, M, qfloat_len, qfloat_ints, qfloat_base, simulate, True
    )
    if simulate:
        print("SIMULATING\n")

    print("QFloat pivot :")
    print(qfloat_res)
    print(" ")

    print("LU pivot :")
    print(pivot_matrix(M))
    print(" ")


def test_qfloat_LU_U_fhe(circuit, M, params, simulate=False):
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params
    qfloat_res, _ = run_qfloat_circuit_fhe(
        circuit, M, qfloat_len, qfloat_ints, qfloat_base, simulate
    )

    if simulate:
        print("SIMULATING")

    print("QFloat LU U :")
    print(qfloat_res)
    print(" ")

    print("LU U :")
    P, L, U = lu_decomposition(M)
    print(U)
    print(" ")


def test_qfloat_LU_L_fhe(circuit, M, params, simulate=False):
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params
    qfloat_res, _ = run_qfloat_circuit_fhe(
        circuit, M, qfloat_len, qfloat_ints, qfloat_base, simulate
    )
    if simulate:
        print("SIMULATING")

    print("QFloat LU L :")
    print(qfloat_res)
    print(" ")

    print("LU L :")
    P, L, U = lu_decomposition(M)
    print(L)
    print(" ")


def test_qfloat_inverse_fhe(circuit, M, params, simulate=False, verbose=True):
    [n, qfloat_len, qfloat_ints, qfloat_base, true_division] = params

    print("\nScipy inv :")
    scres = sc.linalg.inv(M)
    print(scres)

    print("\nQFloat inverse (Python):")
    print(run_qfloat_inverse_python(M, params))

    if simulate:
        qfloat_res, simulation_time = run_qfloat_circuit_fhe(
            circuit, M, qfloat_len, qfloat_ints, qfloat_base, True
        )

        print("\nQFloat inverse (simulating):")
        print(qfloat_res)
        print(" ")

    qfloat_res, (compilation_time, running_time) = run_qfloat_circuit_fhe(
        circuit, M, qfloat_len, qfloat_ints, qfloat_base, False
    )

    print("\nQFloat inverse (FHE) :")
    print(qfloat_res)
    print(" ")

    error = np.abs(qfloat_res - scres)
    print(f"\nAverage Error: {np.mean(error):.6f}")
    print(f"    Max Error: {np.max(error):.6f}")
    print(f"    Min Error: {np.min(error):.6f}")
    print(f"  Total Error: {np.sum(error):.6f}")

    if simulate:
        return simulation_time

    return (compilation_time, running_time)


# TODO: test with higher bases and smaller arrays
# (should be faster to compile and run, slower to encrypt),
# currently their is a bug in concrete.

if __name__ == "__main__":
    # n=2; qfloat_base = 2; qfloat_len = 23; qfloat_ints = 9;
    # n=2; qfloat_base = 2; qfloat_len = 44; qfloat_ints = 14;
    # n=2;  qfloat_base = 16; qfloat_len = 5; qfloat_ints = 3; # CONCRETE BUG
    # n=2;  qfloat_base = 32; qfloat_len = 5; qfloat_ints = 2; # CONCRETE BUG

    # low precision
    true_division = False
    n = 2
    qfloat_base = 2
    qfloat_len = 23
    qfloat_ints = 9
    # n=5; qfloat_base = 2; qfloat_len = 23; qfloat_ints = 9;

    # # intermediate precision
    # true_division = False
    # n=2;
    # qfloat_base = 2;
    # qfloat_len = 31;
    # qfloat_ints = 16;
    # n=4; qfloat_base = 2; qfloat_len = 31; qfloat_ints = 16;
    # n=3; qfloat_base = 16; qfloat_len = 5; qfloat_ints = 3; # CONCRETE BUG

    # higher precision
    # true_division = True
    # n=2;
    # qfloat_base = 2;
    # qfloat_len = 31;
    # qfloat_ints = 16;

    # high precision
    # true_division = True
    # n=10;
    # qfloat_base = 2;
    # qfloat_len = 40;
    # qfloat_ints = 20;

    normal_sampler = ("Normal", lambda: np.random.randn(n, n) * 100)
    uniform_sampler = ("Uniform", lambda: np.random.uniform(0, 100, (n, n)))

    sampler = normal_sampler[1]
    # sampler = uniform_sampler[1]

    params = [n, qfloat_len, qfloat_ints, qfloat_base, true_division]

    # # debug
    # debug_qfloat_inverse_python(sampler, params, verbose=False, N=1000)
    # exit()

    # # gen random matrix
    # M = sampler()

    # print("Test for Matrix :\n==================")
    # print(M)
    # print("")

    # test PLU qf python
    # ----------------------
    # test_qfloat_PLU_python(sampler, params)

    # test inverse qf python
    # ----------------------
    # QFloat.reset_stats()
    # test_qfloat_inverse_python(M, params)
    # QFloat.show_stats()

    # test pivot in fhe:
    # ------------------
    # QFloat.reset_stats()
    # circuit, compilation_time = compile_circuit(params, sampler, qfloat_pivot)
    # QFloat.show_stats()
    # test_qfloat_pivot_fhe(circuit, M, params, False)

    # test LU U decomposition in fhe:
    # -----------------------------
    # QFloat.reset_stats()
    # circuit, compilation_time = compile_circuit(params, sampler, qfloat_lu_U)
    # QFloat.show_stats()
    # test_qfloat_LU_U_fhe(circuit, M, params, False)

    # test LU L decomposition in fhe:
    # -----------------------------
    # QFloat.reset_stats()
    # circuit, compilation_time = compile_circuit(params, sampler, qfloat_lu_L)
    # QFloat.show_stats()
    # test_qfloat_LU_L_fhe(circuit, M, params, simulate=True)

    # test inversion in fhe:
    # -----------------------------
    # start = time.time()

    # QFloat.reset_stats()
    # circuit, compilation_time = compile_circuit(params, sampler, qfloat_matrix_inverse)
    # QFloat.show_stats()
    # test_qfloat_inverse_fhe(circuit, M, params)

    # end = time.time()
    # print(f"|  Total time : {end-start:.2f} s  |")

    ## benchmarking
    def write_file(text, erase=False):
        with open("./log-nothing.txt", "a") as file:
            if erase:
                file.truncate(0)
            file.write(text)

    write_file('', True)

    for n in [2, 3, 5]:
        params[0] = n
        times = []
        write_file("Benchmark for n = " + str(n) + "\n")

        for k in range(3):
            M = sampler()

            start = time.time()

            circuit, compilation_time = compile_circuit(
                params, sampler, qfloat_matrix_inverse, False
            )

            # write the time
            write_file(str(k+1)+"\n")
            write_file("compilation :" + str(compilation_time) + "\n")

            (compilation_time, running_time) = test_qfloat_inverse_fhe(circuit, M, params, False)

            end = time.time()
            current_time = end - start

            times.append(current_time)

            write_file("running     :" + str(running_time) + "\n")
            write_file("total       :" + str(current_time) + "\n")

        # write the mean time
        write_file("\nmean :" + str(np.mean(np.array(times))) + "\n\n\n")
