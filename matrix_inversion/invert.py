import numpy as np
import scipy as sc

def pivot_matrix(M):
    """Returns the pivoting matrix for M, used in Doolittle's method."""
    assert(M.shape[0]==M.shape[1])
    n = M.shape[0]

    # Create an identity matrix, with floating point values
    id_mat = np.eye(n)

    # Rearrange the identity matrix such that the largest element of
    # each column of M is placed on the diagonal of M
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(M[i,j]))
        if j != row:
            # Swap the rows
            id_mat[[j, row]] = id_mat[[row, j]]

    return id_mat


def lu_decomposition(M):
    """Performs an LU Decomposition of M (which must be square)
    into PM = LU. The function returns P, L, and U."""
    assert(M.shape[0]==M.shape[1])
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
        for i in range(j+1):
            s1 = np.dot(U[0:i, j], L[i,0:i])
            U[i, j] = PM[i, j] - s1

        # l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
        for i in range(j, n):
            s2 = np.dot(U[0:j, j], L[i,0:j])
            L[i, j] = (PM[i, j] - s2) / U[j, j]

    # PM = LU
    P= np.transpose(P)

    # now M = PLU
    return P, L, U


def lu_inverse(P, L, U):
    n = L.shape[0]

    # Forward substitution: Solve L * Y = P * A for Y
    Y = np.zeros((n, n))
    for i in range(n):
        Y[i, 0] = P[i, 0] / L[0, 0]
        for j in range(1, n):
            Y[i, j] = (P[i, j] - np.dot(L[j, :j], Y[i, :j])) / L[j, j]

    # Backward substitution: Solve U * X = Y for X
    X = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        X[i, -1] = Y[i, -1] / U[-1, -1]
        for j in range(n - 2, -1, -1):
            X[i, j] = (Y[i, j] - np.dot(U[j, j+1:], X[i, j+1:])) / U[j, j]

    return np.transpose(X)

def matrix_inverse(M):
    P, L, U = lu_decomposition(M)
    return lu_inverse(P, L, U)    


def test_matrix_inverse(n, m=100):
    for i in range(m):
        M = np.random.uniform(0, 100, (n,n))
        M_inv = sc.linalg.inv(M)
        M_inv_2 = matrix_inverse(M)
        error = np.mean(np.abs(M_inv_2 - M_inv))
        assert(error < 0.00001)
    print('test_matrix_inverse OK')


#test_lu_decomposition(3)
test_matrix_inverse(3)



# #does not provide eigenvectors

# def gram_schmidt_qr(A):
#     # Get the shape of A
#     n = A.shape[0]

#     # Create empty Q and R matrices
#     Q = np.zeros((n, n))
#     R = np.zeros((n, n))

#     # The Gram-Schmidt process
#     for i in range(n):
#         # Start with the i-th column of A
#         v = A[:, i]

#         # Subtract the projections of v onto each column of Q
#         for j in range(i):
#             q = Q[:, j]
#             R[j, i] = q @ v
#             v = v - R[j, i] * q

#         # Normalize v
#         norm = np.linalg.norm(v)
#         Q[:, i] = v / norm
#         R[i, i] = norm

#     return Q, R

# def qr(A, k):
#     for i in range(k):
#         Q,R = gram_schmidt_qr(A);
#         A = R @ Q
#     return A, Q

# def qr_diag(A, k):
#     A_triangular, Q = qr(A, k)
#     n=A.shape[0]
#     D = np.zeros((n,n))
#     for i in range(n):
#         D[i,i] = A_triangular[i,i]
#     return D, Q