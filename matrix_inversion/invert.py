import numpy as np
import scipy as sc

def pivot_matrix(M):
    """Returns the pivoting matrix for M, used in Doolittle's method."""
    m = M.shape[0]

    # Create an identity matrix, with floating point values
    id_mat = np.eye(m)

    # Rearrange the identity matrix such that the largest element of
    # each column of M is placed on the diagonal of M
    for j in range(m):
        row = max(range(j, m), key=lambda i: abs(M[i,j]))
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

    return np.transpose(P), L, U



# Example usage:

Aarr= [[4, 3, 1],
       [6, 3, 2],
       [9, 5, 7]]

A = np.array(Aarr)

# print("Scipy :")
P_, L_, U_ = sc.linalg.lu(A)
print(" === SCIPY : ===")
print("P:",P_)
print("L:",L_)
print("U:",U_)
print("A:",P_ @ L_ @ U_)
print("")

A_inv_ = sc.linalg.inv (A)
print("A inv:", A_inv_)
print("")

print(" === ME : ===")
print("Det(A):", sc.linalg.det(A))

P, L, U = lu_decomposition(A)
print("P:",P)
print("L:",L)
print("U:",U)
print("A:", np.array(P) @ np.array(L) @ np.array(U))
