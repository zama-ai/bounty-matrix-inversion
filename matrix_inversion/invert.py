import numpy as np
import scipy as sc

def pivot_matrix(M):
    """Returns the pivoting matrix for M, used in Doolittle's method."""
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


def lu_decomposition(A):
    """Performs an LU Decomposition of A (which must be square)
    into PA = LU. The function returns P, L, and U."""
    n = A.shape[0]

    # Create zero matrices for L and U
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Create the pivot matrix P and the multiplied matrix PA
    P = pivot_matrix(A)
    PA = P @ A

    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity
        L[j, j] = 1.0

        # u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
        for i in range(j+1):
            s1 = np.dot(U[0:i, j], L[i,0:i])
            U[i, j] = PA[i, j] - s1

        # l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
        for i in range(j, n):
            s2 = np.dot(U[0:j, j], L[i,0:j])
            L[i, j] = (PA[i, j] - s2) / U[j, j]

    # PA = LU
    P= np.transpose(P)

    # now A = PLU
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

# Example usage:

Aarr= [[4, 3, 1],
       [6, 3, 2],
       [9, 5, 7]]

A = np.array(Aarr)

# print("Scipy :")
P_, L_, U_ = sc.linalg.lu(A)
print(" === SCIPY : ===")
print("Pt:",P_)
print("L:",L_)
print("U:",U_)
print("A:",P_ @ L_ @ U_)
print("")

A_inv_ = sc.linalg.inv (A)
print("A inv:", A_inv_)
print("A * A inv:", A @ A_inv_)
print("")

print(" === ME : ===")
print("Det(A):", sc.linalg.det(A))

P, L, U = lu_decomposition(A)
print("P:",P)
print("L:",L)
print("U:",U)
print("A:", P @ L @ U)

A_inv = lu_inverse(P, L, U)
print("A inv:", A_inv)
print("A * A inv:", A @ A_inv)
print("")
