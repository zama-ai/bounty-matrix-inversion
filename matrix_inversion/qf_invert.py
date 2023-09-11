import numpy as np
import scipy as sc
import time
from QFloat import QFloat, SignedBinary, Zero
from concrete import fhe

Tracer = fhe.tracing.tracer.Tracer


# ============================================================================================================================================

#  ██████  ██████  ██  ██████  ██ ███    ██  █████  ██          ██      ██    ██     ██ ███    ██ ██    ██ ███████ ██████  ███████ ███████
# ██    ██ ██   ██ ██ ██       ██ ████   ██ ██   ██ ██          ██      ██    ██     ██ ████   ██ ██    ██ ██      ██   ██ ██      ██
# ██    ██ ██████  ██ ██   ███ ██ ██ ██  ██ ███████ ██          ██      ██    ██     ██ ██ ██  ██ ██    ██ █████   ██████  ███████ █████
# ██    ██ ██   ██ ██ ██    ██ ██ ██  ██ ██ ██   ██ ██          ██      ██    ██     ██ ██  ██ ██  ██  ██  ██      ██   ██      ██ ██
#  ██████  ██   ██ ██  ██████  ██ ██   ████ ██   ██ ███████     ███████  ██████      ██ ██   ████   ████   ███████ ██   ██ ███████ ███████

# the python algorithm operating of floats for inverting matrices

# ============================================================================================================================================


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


# ============================================================================================================================================

#  ██████  ███████ ██       ██████   █████  ████████     ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
# ██    ██ ██      ██      ██    ██ ██   ██    ██        ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
# ██    ██ █████   ██      ██    ██ ███████    ██        █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
# ██ ▄▄ ██ ██      ██      ██    ██ ██   ██    ██        ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
#  ██████  ██      ███████  ██████  ██   ██    ██        ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████
#     ▀▀

# functions working on QFloats instead of floats

# ============================================================================================================================================


########################################################################################
#                                     UTILS
########################################################################################


def matrix_column(M, j):
    """
    Extract column from 2D list matrix
    """
    return [row[j] for row in M]


def transpose_2DList(list2D):
    """
    Transpose a 2D-list matrix
    """
    return [list(row) for row in zip(*list2D)]


def map_2D_list(list2D, function):
    """
    Apply a function to a 2D list
    """
    return [[function(f) for f in row] for row in list2D]


def binaryListMatrix(M):
    """
    Convert a binary matrix M into a 2D-list matrix of SignedBinarys
    """
    return map_2D_list(
        [[M[i, j] for j in range(M.shape[1])] for i in range(M.shape[0])],
        lambda x: SignedBinary(x),
    )


def zeroListMatrix(n):
    """
    Creates a square list matrix of Zeros
    """
    return [[Zero() for j in range(n)] for i in range(n)]


def qf_list_dot_product(list1, list2):
    """
    Dot product of two QFloat lists
    """
    if len(list1) != len(list2):
        raise ValueError("Lists should have the same length.")

    result = list1[0] * list2[0]
    for i in range(1, len(list1)):
        result += list1[i] * list2[i]  # in place addition is supported

    return result


def qf_list_matrix_multiply(matrix1, matrix2):
    """
    Multiply two matrices of int or QFloats
    """
    # if len(matrix1[0]) != len(matrix2):
    #     raise ValueError("Number of columns in matrix1 should match the number of rows in matrix2.")
    result = [[None] * len(matrix2[0]) for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            result[i][j] = qf_list_dot_product(matrix1[i], matrix_column(matrix2, j))

    return result


def float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base):
    """
    converts a float matrix to arrays representing qfloats
    """
    qf_list = [QFloat.fromFloat(f, qf_len, qf_ints, qf_base) for f in M.flatten()]
    n = len(qf_list)
    qf_arrays = np.zeros((n, qf_len), dtype="int")
    qf_signs = np.zeros(n, dtype="int")
    for i in range(n):
        qf_arrays[i, :] = qf_list[i].toArray()
        qf_signs[i] = qf_list[i].getSign()

    return qf_arrays, qf_signs


def qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base):
    """
    converts qfloats array and sign array to a QFloat matrix
    """
    n = int(np.sqrt(qf_arrays.shape[0]))
    qf_M = []
    index = 0
    for i in range(n):
        row = []
        for j in range(n):
            qf = QFloat(qf_arrays[index, :], qf_ints, qf_base, True, qf_signs[index])
            index += 1
            row.append(qf)
        qf_M.append(row)

    return qf_M


def qfloatNsigns_arrays_to_float_matrix(qf_arrays, qf_ints, qf_base):
    """
    converts qfloats and signs arrays to a float matrix
    """
    n = int(np.sqrt(qf_arrays.shape[0]))
    M = np.zeros((n, n))
    index = 0
    for i in range(n):
        for j in range(n):
            M[i, j] = QFloat(
                qf_arrays[index, :-1], qf_ints, qf_base, True, qf_arrays[index, -1]
            ).toFloat()
            index += 1

    return M


def qfloat_matrix_to_arraysNsigns(M, qf_len, qf_ints, qf_base):
    """
    converts a QFloat 2D-list matrix to integer arrays and signs
    """
    n = len(M)
    assert n == len(M[0])
    qf_arrays = fhe.zeros((n * n, qf_len + 1))  # add one for the sign
    index = 0
    for i in range(n):
        for j in range(n):
            if isinstance(M[i][j], QFloat):
                qf_arrays[index, :qf_len] = M[i][j].toArray()
                qf_arrays[index, qf_len] = M[i][j].getSign()
            elif isinstance(M[i][j], SignedBinary):
                qf_arrays[index, qf_ints - 1] = M[i][j].value
                qf_arrays[index, qf_len] = M[i][j].value
            elif isinstance(M[i][j], Zero):
                pass
            else:
                qf_arrays[index, qf_ints - 1] = M[i][j]
                qf_arrays[index, qf_len] = np.sign(M[i][j])
            index += 1

    return qf_arrays


########################################################################################
#                                   PIVOT MATRIX
########################################################################################


def qf_argmax(indices, qfloats):
    """
    Returns the index corresponding to the biggest QFloat in the list
    """
    maxQf = qfloats[0].copy()
    maxi = indices[0]
    for i in range(1, len(indices)):
        is_gt = qfloats[i] > maxQf
        maxQf._array = is_gt * qfloats[i]._array + (1 - is_gt) * maxQf._array
        maxi = is_gt * indices[i] + (1 - is_gt) * maxi

    return maxi


def qf_pivot_matrix(M):
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
        r = qf_argmax([i for i in range(j, n)], [abs(M[i][j]) for i in range(j, n)])

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


########################################################################################
#                                   LU DECOMPOSITION
########################################################################################


def qf_lu_decomposition(M, qf_len, qf_ints, trueDivision=False):
    """
    Perform a LU decomposition of square (QFloats 2D-list) matrix M
    The function returns P, L, and U such that M = PLU.

    trueDivision: Wether to compute true divisions instead of multiplying by inverse. Setting it to True is more precise but slower
    """
    assert len(M) == len(M[0])
    n = len(M)

    # add __len__ function to Tracer for simplicity:
    def arrlen(self):
        return self.shape[0]

    Tracer.__len__ = arrlen

    # Initialize 2D-list matrices for L and U
    # (QFloats can be multiplied with integers, this saves computations)
    L = zeroListMatrix(n)
    U = zeroListMatrix(n)

    # Create the pivot matrix P and the multiplied matrix PM
    P = binaryListMatrix(qf_pivot_matrix(M))
    PM = qf_list_matrix_multiply(P, M)

    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity
        L[j][j] = SignedBinary(1)
        # u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
        for i in range(j + 1):
            if i > 0:
                s1 = qf_list_dot_product(
                    [U[k][j] for k in range(0, i)], [L[i][k] for k in range(0, i)]
                )
                U[i][j] = PM[i][j] + s1.neg()
            else:
                U[i][j] = PM[i][j].copy()

        # l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
        if not trueDivision:
            inv_Ujj = U[j][j].invert(
                1, qf_len, 0
            )  # compute one precise invert division and then use multiplications, faster but less precise
        for i in range(j + 1, n):
            if j > 0:
                s2 = qf_list_dot_product(
                    [U[k][j] for k in range(0, j)], [L[i][k] for k in range(0, j)]
                )
                # L[i][j] = (PM[i][j] - s2) * inv_Ujj
                if trueDivision:
                    L[i][j] = (PM[i][j] + s2.neg()) / U[j][j]
                else:
                    L[i][j] = QFloat.fromMul(
                        (PM[i][j] + s2.neg()), inv_Ujj, qf_len, qf_ints
                    )

            else:
                # L[i][j] = PM[i][j] / U[j][j]
                if trueDivision:
                    L[i][j] = PM[i][j] / U[j][j]
                else:
                    L[i][j] = QFloat.fromMul(PM[i][j], inv_Ujj, qf_len, qf_ints)

    # for now, PM = LU
    P = transpose_2DList(P)
    # now M = PLU (Note that Ljj is always 1 for all j)
    return P, L, U


########################################################################################
#                                   LU INVERSE
########################################################################################


def qf_lu_inverse(P, L, U, qf_len, qf_ints, trueDivision=False, debug=False):
    """
    Compute inverse using the P,L,U decomposition (operating on QFloats 2D-lists)

    trueDivision: Wether to compute true divisions instead of multiplying by inverse. Setting it to True is more precise but slower
    """
    n = len(L)

    # Forward substitution: Solve L * Y = P * A for Y
    Y = zeroListMatrix(n)
    for i in range(n):
        # Y[i, 0] = P[i, 0] / L[0, 0]
        Y[i][0] = P[i][
            0
        ].copy()  # dividing by diagonal values of L is useless because they are equal to 1
        for j in range(1, n):
            # Y[i, j] = (P[i, j] - np.dot(L[j, :j], Y[i, :j]))
            Y[i][j] = P[i][j] - qf_list_dot_product(
                [L[j][k] for k in range(j)], [Y[i][k] for k in range(j)]
            )

    # Backward substitution: Solve U * X = Y for X
    X = zeroListMatrix(n)

    # precompute inverse of U to make less divisions (simplify U before as we know its range)
    if not trueDivision:
        Ujj_inv = [U[j][j].invert(1, qf_len, 0) for j in range(n)]
    for i in range(n - 1, -1, -1):
        # X[i, -1] = Y[i, -1] / U[-1, -1]
        if trueDivision:
            X[i][-1] = Y[i][-1] / U[-1][-1]
        else:
            X[i][-1] = QFloat.fromMul(Y[i][-1], Ujj_inv[-1], qf_len, qf_ints)

        for j in range(n - 2, -1, -1):
            # X[i, j] = (Y[i, j] - np.dot(U[j, j+1:], X[i, j+1:])) / U[j, j]
            temp = Y[i][j] - qf_list_dot_product(
                [U[j][k] for k in range(j + 1, n)], [X[i][k] for k in range(j + 1, n)]
            )
            if trueDivision:
                X[i][j] = temp / U[j][j]
            else:
                X[i][j] = QFloat.fromMul(temp, Ujj_inv[j], qf_len, qf_ints)

    if not debug:
        return transpose_2DList(X)
    else:
        return transpose_2DList(X), Y, X  # return Y and X for debug purpose


########################################################################################
#                               2x2 SHORCUT FORMULA
########################################################################################


def qf_inverse_2x2(qf_M, qf_len, qf_ints):
    """
    The inverse of a 2x2 matrix has a simple formula: M_inv = 1/det(M) * [[d,-b],[-c,a]]
    WARNING : The function is optimized for inputs in range [0,2^8] (see samplers)
    """
    [a, b] = qf_M[0]
    [c, d] = qf_M[1]

    ad = QFloat.fromMul(
        a, d, 2 * qf_ints + 3, 2 * qf_ints
    )  # produces a longer integer part
    bc = QFloat.fromMul(
        b, c, 2 * qf_ints + 3, 2 * qf_ints
    )  # produces a longer integer part

    det = ad - bc  # determinant of M in 17 bits

    det_inv = det.invert(
        1, qf_len, 0
    )  # computes invert with a lot of decimals but no integer part

    mul = lambda x, y: QFloat.fromMul(
        x, y, qf_len, qf_ints
    )  # multiply to output format
    M_inv = [
        [mul(d, det_inv), mul(b, det_inv).neg()],
        [mul(c, det_inv).neg(), mul(a, det_inv)],
    ]

    return M_inv


########################################################################################
#                                   COMPLETE FUNCTION
########################################################################################


def qf_pivot(qf_arrays, qf_signs, params):
    """
    computes only the pivot matrix
    """
    [n, qf_len, qf_ints, qf_base, _] = params

    assert n * n == qf_arrays.shape[0]
    assert qf_len == qf_arrays.shape[1]

    # reconstruct the matrix of QFloats with encrypted values:
    qf_M = qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # compute the pivot matrix
    P = qf_pivot_matrix(qf_M)

    return P


def qf_lu_L(qf_arrays, qf_signs, params):
    """
    compute only PLU and returns L
    """
    [n, qf_len, qf_ints, qf_base, trueDivision] = params

    assert n * n == qf_arrays.shape[0]
    assert qf_len == qf_arrays.shape[1]

    # reconstruct the matrix of QFloats with encrypted values:
    qf_M = qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # compute the LU decomposition
    bin_P, qf_L, qf_U = qf_lu_decomposition(qf_M, qf_len, qf_ints, trueDivision)

    # break the resulting QFloats into arrays:
    qf_inv_arrays_L = qfloat_matrix_to_arraysNsigns(qf_L, qf_len, qf_ints, qf_base)
    # qf_inv_arrays_U = qfloat_matrix_to_arraysNsigns(qf_U, qf_len, qf_ints, qf_base)

    return qf_inv_arrays_L


def qf_lu_U(qf_arrays, qf_signs, params):
    """
    compute only PLU and returns U
    """
    [n, qf_len, qf_ints, qf_base, trueDivision] = params

    assert n * n == qf_arrays.shape[0]
    assert qf_len == qf_arrays.shape[1]

    # reconstruct the matrix of QFloats with encrypted values:
    qf_M = qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # compute the LU decomposition
    bin_P, qf_L, qf_U = qf_lu_decomposition(qf_M, qf_len, qf_ints, trueDivision)

    # break the resulting QFloats into arrays:
    # qf_inv_arrays_L = qfloat_matrix_to_arraysNsigns(qf_L, qf_len, qf_ints, qf_base)
    qf_inv_arrays_U = qfloat_matrix_to_arraysNsigns(qf_U, qf_len, qf_ints, qf_base)

    return qf_inv_arrays_U


def qf_matrix_inverse(qf_arrays, qf_signs, params):
    """
    Compute the whole inverse
    """
    [n, qf_len, qf_ints, qf_base, trueDivision] = params

    assert n * n == qf_arrays.shape[0]
    assert qf_len == qf_arrays.shape[1]

    # reconstruct the matrix of QFloats with encrypted values:
    qf_M = qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    if n == 2:
        # use shortcut formula
        qf_Minv = qf_inverse_2x2(qf_M, qf_len, qf_ints)

    else:
        # compute the LU decomposition
        bin_P, qf_L, qf_U = qf_lu_decomposition(qf_M, qf_len, qf_ints, trueDivision)

        # compute inverse from P L U
        qf_Minv = qf_lu_inverse(bin_P, qf_L, qf_U, qf_len, qf_ints, trueDivision)

    # break the resulting QFloats into arrays:
    qf_inv_arrays = qfloat_matrix_to_arraysNsigns(qf_Minv, qf_len, qf_ints, qf_base)

    return qf_inv_arrays


#############################################################################################################

# ████████ ███████ ███████ ████████ ███████
#    ██    ██      ██         ██    ██
#    ██    █████   ███████    ██    ███████
#    ██    ██           ██    ██         ██
#    ██    ███████ ███████    ██    ███████

# tests for functions in this file. The above the functions can be imported from another file

#############################################################################################################


def measure_time(function, descripton, *inputs):
    # Compute a function on inputs and return output along with duration
    print(descripton + " ...", end="", flush=True)
    print("\r", end="")
    start = time.time()
    output = function(*inputs)
    end = time.time()
    print(f"|  {descripton} : {end-start:.2f} s  |")
    return output


########################################################################################
#                                    PYTHON
########################################################################################


def test_qf_PLU_python(M, params):
    """
    Test the LU decomposition using QFloats, in python
    """
    [n, qf_len, qf_ints, qf_base, trueDivision] = params

    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    # reconstruct the matrix of QFloats
    qf_M = qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # compute the LU decomposition
    bin_P, qf_L, qf_U = qf_lu_decomposition(qf_M, qf_len, qf_ints, trueDivision)

    # convert bin_P from binaryValue to floats
    P = np.array(map_2D_list(bin_P, lambda x: x.value))

    # break the resulting QFloats into arrays:
    qf_arrays_L = qfloat_matrix_to_arraysNsigns(qf_L, qf_len, qf_ints, qf_base)
    qf_arrays_U = qfloat_matrix_to_arraysNsigns(qf_U, qf_len, qf_ints, qf_base)

    L = qfloatNsigns_arrays_to_float_matrix(qf_arrays_L, qf_ints, qf_base)
    U = qfloatNsigns_arrays_to_float_matrix(qf_arrays_U, qf_ints, qf_base)

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


def run_qf_inverse_python(M, qf_len, qf_ints, qf_base):
    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    output = qf_matrix_inverse(qf_arrays, qf_signs, params)

    qf_Res = qfloatNsigns_arrays_to_float_matrix(output, qf_ints, qf_base)

    return qf_Res


def test_qf_inverse_python(M, params):
    """
    Test the matrix inverse using QFloats, in python
    """
    [n, qf_len, qf_ints, qf_base, trueDivision] = params

    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    start = time.time()
    qf_Res = run_qf_inverse_python(M, qf_len, qf_ints, qf_base)
    end = time.time()

    print("\n==== Test python ====")
    print("\nQFloat inverse :")
    print(qf_Res)

    print("\nLU inverse :")
    print(matrix_inverse(M))

    print("\nScipy inv :")
    scres = sc.linalg.inv(M)
    print(scres)

    error = np.abs(qf_Res - scres)
    print(f"\nAverage Error: {np.mean(error):.6f}")
    print(f"    Max Error: {np.max(error):.6f}")
    print(f"    Min Error: {np.min(error):.6f}")
    print(f"  Total Error: {np.sum(error):.6f}")

    print(f"\n|  Took : {end-start:.4f} s  |\n")


def debug_qf_inverse_python(sampler, params, verbose=False, N=100):
    """
    Debug the algorithm to look for big errors
    """
    errors = np.zeros(N)
    nerrors = 0
    for i in range(N):
        # print(i)
        # gen random matrix
        M = sampler()

        [n, qf_len, qf_ints, qf_base, trueDivision] = params

        # convert it to QFloat arrays
        qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

        qf_Res = run_qf_inverse_python(M, qf_len, qf_ints, qf_base)

        scres = sc.linalg.inv(M)

        error = np.mean(np.abs(qf_Res - scres))

        errors[i] = error

        if error > 1:
            nerrors += 1

            if not verbose:
                continue

            err = np.abs(qf_Res - scres)
            print(f"\nAverage Error: {np.mean(err):.6f}")
            print(f"    Max Error: {np.max(err):.6f}")
            print(f"    Min Error: {np.min(err):.6f}")
            print(f"  Total Error: {np.sum(err):.6f}")

            # reconstruct the matrix of QFloats
            qf_M = qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

            # compute the LU decomposition
            bin_P, qf_L, qf_U = qf_lu_decomposition(qf_M, qf_len, qf_ints, trueDivision)

            # compute the inverse
            qf_Minv, qf_Y, qf_X = qf_lu_inverse(
                bin_P, qf_L, qf_U, qf_len, qf_ints, trueDivision, debug=True
            )

            # convert QFloat mattrices to floats
            L = map_2D_list(qf_L, lambda x: x.toFloat())
            U = map_2D_list(qf_U, lambda x: x.toFloat())

            P_, L_, U_ = lu_decomposition(M)
            Minv_, Y_, X_ = lu_inverse(P_, L_, U_, debug=True)

            print("\nL")
            print(L)
            print(L_)
            print("\nU")
            print(U)
            print(U_)
            print("\nInv")
            print(qf_Res)
            print(scres)

            Y = map_2D_list(qf_Y, lambda x: x.toFloat())
            X = map_2D_list(qf_X, lambda x: x.toFloat())

            print("\nX")
            print(X)
            print(X_)
            print("\nY")
            print(Y)
            print(Y_)

    print("mean error :", np.mean(errors))
    print("big error rate :" + str(100 * nerrors / N) + " %")


########################################################################################
#                                           FHE
########################################################################################


def compile_circuit(params, sampler, circuit_function):
    [n, qf_len, qf_ints, qf_base, trueDivision] = params

    inputset = []
    for i in range(100):
        M = sampler()
        qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)
        inputset.append((qf_arrays, qf_signs))

    compiler = fhe.Compiler(
        lambda x, y: circuit_function(x, y, params),
        {"x": "encrypted", "y": "encrypted"},
    )
    make_circuit = lambda: compiler.compile(
        inputset=inputset,
        configuration=fhe.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location=".keys",
            single_precision=False,
            # dataflow_parallelize=True,
        ),
        verbose=False,
    )

    circuit = measure_time(make_circuit, "Compiling")
    return circuit


def run_qf_circuit_fhe(
    circuit, M, qf_len, qf_ints, qf_base, simulate=False, raw_output=False
):
    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    # Run FHE
    if not simulate:
        encrypted = measure_time(circuit.encrypt, "Encrypting", qf_arrays, qf_signs)
        run = measure_time(circuit.run, "Running", encrypted)
        decrypted = circuit.decrypt(run)
    else:
        decrypted = measure_time(circuit.simulate, "Simulating", qf_arrays, qf_signs)

    if raw_output:
        return decrypted

    return qfloatNsigns_arrays_to_float_matrix(decrypted, qf_ints, qf_base)


def test_qf_pivot_fhe(circuit, M, params, simulate=False):
    [n, qf_len, qf_ints, qf_base, trueDivision] = params
    qf_Res = run_qf_circuit_fhe(circuit, M, qf_len, qf_ints, qf_base, simulate, True)
    if simulate:
        print("SIMULATING\n")

    print("QFloat pivot :")
    print(qf_Res)
    print(" ")

    print("LU pivot :")
    print(pivot_matrix(M))
    print(" ")


def test_qf_LU_U_fhe(circuit, M, params, simulate=False):
    [n, qf_len, qf_ints, qf_base, trueDivision] = params
    qf_Res = run_qf_circuit_fhe(circuit, M, qf_len, qf_ints, qf_base, simulate)

    if simulate:
        print("SIMULATING")

    print("QFloat LU U :")
    print(qf_Res)
    print(" ")

    print("LU U :")
    P, L, U = lu_decomposition(M)
    print(U)
    print(" ")


def test_qf_LU_L_fhe(circuit, M, params, simulate=False):
    [n, qf_len, qf_ints, qf_base, trueDivision] = params
    qf_Res = run_qf_circuit_fhe(circuit, M, qf_len, qf_ints, qf_base, simulate)
    if simulate:
        print("SIMULATING")

    print("QFloat LU L :")
    print(qf_Res)
    print(" ")

    print("LU L :")
    P, L, U = lu_decomposition(M)
    print(L)
    print(" ")


def test_qf_inverse_fhe(circuit, M, params):
    [n, qf_len, qf_ints, qf_base, trueDivision] = params

    print("\nScipy inv :")
    scres = sc.linalg.inv(M)
    print(scres)

    print("\nQFloat inverse (Python):")
    print(run_qf_inverse_python(M, qf_len, qf_ints, qf_base))

    qf_Res = run_qf_circuit_fhe(circuit, M, qf_len, qf_ints, qf_base, True)

    print("\nQFloat inverse (simulating):")
    print(qf_Res)
    print(" ")

    qf_Res = run_qf_circuit_fhe(circuit, M, qf_len, qf_ints, qf_base, False)

    print("\nQFloat inverse (Running) :")
    print(qf_Res)
    print(" ")

    error = np.abs(qf_Res - scres)
    print(f"\nAverage Error: {np.mean(error):.6f}")
    print(f"    Max Error: {np.max(error):.6f}")
    print(f"    Min Error: {np.min(error):.6f}")
    print(f"  Total Error: {np.sum(error):.6f}")


# TODO: test with higher bases and smaller arrays (should be faster to compile and run, slower to encrypt), currently their is a bug in concrete.

if __name__ == "__main__":
    # n=2; qf_base = 2; qf_len = 23; qf_ints = 9;
    # n=2; qf_base = 2; qf_len = 44; qf_ints = 14;
    # n=2;  qf_base = 16; qf_len = 5; qf_ints = 3; # CONCRETE BUG
    # n=2;  qf_base = 32; qf_len = 5; qf_ints = 2; # CONCRETE BUG

    # low precision
    trueDivision = False
    n = 3
    qf_base = 2
    qf_len = 23
    qf_ints = 9
    # n=5; qf_base = 2; qf_len = 23; qf_ints = 9;

    # intermediate precision
    trueDivision = False
    # n=3; qf_base = 2; qf_len = 31; qf_ints = 16;
    # n=4; qf_base = 2; qf_len = 31; qf_ints = 16;
    # n=3; qf_base = 16; qf_len = 5; qf_ints = 3; # CONCRETE BUG

    # higher precision
    # trueDivision = True
    # n=3; qf_base = 2; qf_len = 31; qf_ints = 16;

    normal_sampler = ("Normal", lambda: np.random.randn(n, n) * 100)
    uniform_sampler = ("Uniform", lambda: np.random.uniform(0, 100, (n, n)))

    sampler = normal_sampler[1]
    # sampler = uniform_sampler[1]

    # gen random matrix
    M = sampler()

    print("Test for Matrix :\n==================")
    print(M)
    print("")

    params = [n, qf_len, qf_ints, qf_base, trueDivision]

    # debug
    # debug_qf_inverse_python(sampler, params, verbose=False, N=1000)

    # test PLU qf python
    # ----------------------
    # test_qf_PLU_python(sampler, params)

    # test inverse qf python
    # ----------------------
    QFloat.resetStats()
    test_qf_inverse_python(M, params)
    QFloat.showStats()

    # test pivot in fhe:
    # ------------------
    # QFloat.resetStats()
    # circuit = compile_circuit(params, sampler, qf_pivot)
    # QFloat.showStats()
    # test_qf_pivot_fhe(circuit, M, params, False)

    # test LU U decomposition in fhe:
    # -----------------------------
    # QFloat.resetStats()
    # circuit = compile_circuit(params, sampler, qf_lu_U)
    # QFloat.showStats()
    # test_qf_LU_U_fhe(circuit, M, params, False)

    # test LU L decomposition in fhe:
    # -----------------------------
    # QFloat.resetStats()
    # circuit = compile_circuit(params, sampler, qf_lu_L)
    # QFloat.showStats()
    # test_qf_LU_L_fhe(circuit, M, params, simulate=True)

    # test inversion in fhe:
    # -----------------------------
    QFloat.resetStats()
    circuit = compile_circuit(params, sampler, qf_matrix_inverse)
    QFloat.showStats()
    test_qf_inverse_fhe(circuit, M, params)
