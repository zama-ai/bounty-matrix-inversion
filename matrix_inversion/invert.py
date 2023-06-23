import numpy as np
import scipy as sc

import numpy as np

def pivot_matrix_2(M):
    """Returns the pivoting matrix for M, used in Doolittle's method."""
    m = len(M)

    # Create an identity matrix, with floating point values                                                                                                                                                                                            
    id_mat = [[float(i ==j) for i in range(m)] for j in range(m)]

    # Rearrange the identity matrix such that the largest element of                                                                                                                                                                                   
    # each column of M is placed on the diagonal of of M                                                                                                                                                                                               
    for j in range(m):
        row = max(range(j, m), key=lambda i: abs(M[i][j]))
        if j != row:
            # Swap the rows                                                                                                                                                                                                                            
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]

    return id_mat

def lu_decomposition_2(A):
    """Performs an LU Decomposition of A (which must be square)                                                                                                                                                                                        
    into PA = LU. The function returns P, L and U."""
    n = len(A)

    # Create zero matrices for L and U                                                                                                                                                                                                                 
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]

    # Create the pivot matrix P and the multipled matrix PA                                                                                                                                                                                            
    P = pivot_matrix_2(A)
    PA = (np.array(P) @ np.array(A)).tolist()

    # Perform the LU Decomposition                                                                                                                                                                                                                     
    for j in range(n):
        # All diagonal entries of L are set to unity                                                                                                                                                                                                   
        L[j][j] = 1.0

        # LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}                                                                                                                                                                                      
        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = PA[i][j] - s1

        # LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik} )                                                                                                                                                                  
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]

    return (P, L, U)


# Example usage:

Aarr= [[4, 3, 1],
       [6, 3, 2],
       [9, 5, 7]]

A = np.array(Aarr)

# print("Scipy :")
P_, L_, U_ = sc.linalg.lu(A)
print(" === SCIPY : ===")
print("L:",L_)
print("U:",U_)
print("A:",P_ @ L_ @ U_)
print("")

A_inv_ = sc.linalg.inv(A)
print("A inv:", A_inv_)
print("")

print(" === ME : ===")
print("Det(A):", sc.linalg.det(A))

P, L, U = lu_decomposition_2(Aarr)
print("L:",L)
print("U:",U)
print("A:", np.array(P) @ np.array(L) @ np.array(U))
