import numpy as np
import scipy as sc
import time
from QFloat import QFloat
from concrete import fhe

#transpose = [list(row) for row in zip(*matrix)]



########################################################################################
#                                   PIVOT MATRIX
########################################################################################


#### INITIAL FUNCTION  ####################################################################
def pivot_matrix(M):
    """Returns the pivoting matrix for M, used in Doolittle's method."""
    assert(M.shape[0]==M.shape[1])
    n = M.shape[0]

    # Create an identity matrix, with floating point values
    id_mat = np.eye(n)

    # Rearrange the identity matrix such that the largest element of
    # each column of M is placed on the diagonal of M
    #for j in range(n): # last step j=n-1 is unusefull cause row will be equal to j
    for j in range(n-1):
        row = max(range(j, n), key=lambda i: abs(M[i,j]))
        if j != row:
            # Swap the rows
            id_mat[[j, row]] = id_mat[[row, j]]

    return id_mat


def map_2D_list(list2D, function):
    return [[function(f) for f in row] for row in list2D]

def qf_argmax(indices, qfloats):
    """
    Returns the index corresponding to the biggest QFloat in the list
    """
    maxQf = qfloats[0].copy()
    maxi = indices[0] 
    for i in range(1,len(indices)):
        is_gt = qfloats[i] > maxQf
        maxQf._array = is_gt*qfloats[i]._array + (1-is_gt)*maxQf._array
        maxi = is_gt*indices[i] + (1-is_gt)*maxi

    return maxi

def boolean_sum(sumlist, conditions):
    """
    Sum elements in a list with boolean conditioning
    """
    bsum = sumlist[0]*conditions[0]
    for i in range(1,len(sumlist)):
        bsum += sumlist[i]*conditions[i]

    return bsum

def qf_pivot_matrix(M):
    """
    Returns the pivoting matrix for M, used in Doolittle's method.
    M is a squared 2D list of QFloats
    """
    assert(len(M)==len(M[0])) # assert this looks like a 2D list of a square matrix
    n = len(M)

    # Create a fhe identity matrix, with fhe integer values
    pivot_mat = fhe.zeros((n,n))
    for j in range(n):
        pivot_mat[j,j] = fhe.ones(1)[0]

    # Rearrange the identity matrix such that the largest element of
    # each column of M is placed on the diagonal of M
    temp_mat = fhe.zeros((n,n))
    for j in range(n-1):
        # compute argmax of abs values in range(j,n)
        r = qf_argmax( [i for i in range(j, n)], [ abs(M[i][j]) for i in range(j, n)] )

        # copy pivot_mat to temp_mat
        temp_mat[:] = pivot_mat[:]

        # swap rows in pivot_mat:
        # row j becomes row r:
        #pivot_mat[j,:] = boolean_sum( [ temp_mat[k,:] for k in range(j,n) ], np.arange(j,n)==r )
        
        bsum = temp_mat[j,:]*(j==r)
        for i in range(j+1,n):
            bsum += temp_mat[i,:]*(i==r)

        pivot_mat[j,:] = bsum[:]

        # row r becomes row j:
        for jj in range(j+1,n):
          jj_eq_r = jj==r
          pivot_mat[jj,:] = (1-jj_eq_r)*temp_mat[jj,:] + jj_eq_r*temp_mat[j,:]

    return pivot_mat    






########################################################################################
#                                   LU DECOMPOSITION
########################################################################################

#### INITIAL FUNCTION  ####################################################################
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






########################################################################################
#                                   LU INVERSE
########################################################################################


#### INITIAL FUNCTION  ####################################################################
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





########################################################################################
#                                   INVERSE FUNCTION
########################################################################################


#### INITIAL FUNCTION  ####################################################################
def matrix_inverse(M):
    P, L, U = lu_decomposition(M)
    return lu_inverse(P, L, U)


def float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base):
    """
    converts a float matrix to arrays representing qfloats
    """
    qf_list = [ QFloat.fromFloat(f, qf_len, qf_ints, qf_base) for f in M.flatten()]
    n=len(qf_list)
    qf_arrays = np.zeros((n, qf_len), dtype='int')
    qf_signs = np.zeros(n, dtype='int')
    for i in range(n):
        qf_arrays[i,:] = qf_list[i].toArray()
        qf_signs[i] = qf_list[i].getSign()

    return qf_arrays, qf_signs

def qfloat_arrays_to_float_matrix(qf_arrays, qf_signs, qf_ints, qf_base):
    """
    converts qfloats arrays to a float matrix
    """
    n = int(np.sqrt(qf_arrays.shape[0]))
    qf_M = []
    index=0
    for i in range(n):
        row=[]
        for j in range(n):
            qf = QFloat(qf_arrays[index,:], qf_ints, qf_base, True, qf_signs[index])
            index+=1
            row.append(qf)
        qf_M.append(row)

    return qf_M


def fhematrix(qf_arrays, qf_signs, params):

    n, qf_len, qf_ints, qf_base = params

    assert( n*n == qf_arrays.shape[0])
    assert( qf_len == qf_arrays.shape[1])

    # reconstruct the matrix of QFloats with encrypted values:
    qf_M = qfloat_arrays_to_float_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # compute the pivot matrix
    qf_P = qf_pivot_matrix(qf_M)

    return qf_P

########################################################################################
#                                   TESTS
########################################################################################


#### INITIAL FUNCTION  ####################################################################
def test_matrix_inverse(n, m=100):
    for i in range(m):
        M = np.random.uniform(0, 100, (n,n))
        M_inv = sc.linalg.inv(M)
        M_inv_2 = matrix_inverse(M)
        error = np.mean(np.abs(M_inv_2 - M_inv))
        assert(error < 0.00001)

    print('test_matrix_inverse OK')



def test_qf_matrix_inverse(n, m=100):
    for i in range(m):
        M = np.random.uniform(0, 100, (n,n))
        P = pivot_matrix(M)

        # with qfloats
        qf_M = map_2D_list(M.tolist(), lambda x:QFloat.fromFloat(x,30,8,2))
        P2 = qf_pivot_matrix(qf_M)

        print(P-P2)
        
    print('test_matrix_inverse OK')


def measure_time(function, descripton, *inputs):
    #Compute a function on inputs and return output along with duration
    start = time.time()
    output = function(*inputs)
    end = time.time()
    print(f"|  {descripton} : {end-start:.2f} s  |")
    return output


def test_qf_fhe(n, simulate=False):
    # gen random matrix
    M = np.random.uniform(0, 100, (n,n))
    qf_base=2
    qf_len = 30
    qf_ints = 8

    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    # set params
    params = [n, qf_len, qf_ints, qf_base]

    compiler = fhe.Compiler(lambda x,y: fhematrix(x,y,params), {"x": "encrypted", "y": "encrypted"})
    circuit = compiler.compile(
        inputset=[
                (np.random.randint(0, qf_base, size=(n*n, qf_len)),
                np.random.randint(0, qf_base, size=(n*n,)))
                for _ in range(100)
            ],
        configuration=fhe.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location=".keys",
            #dataflow_parallelize=True,
        ),
        verbose=False,
    )

    # First print the description and a waiting message
    print("Matrix inversion")
    print("Computing ...", end="", flush=True)
    print("\r", end="")

    # Run FHE
    if not simulate:
        encrypted = measure_time(circuit.encrypt, 'Encrypting', qf_arrays, qf_signs)
        run = measure_time(circuit.run,'Running', encrypted)
        decrypted = circuit.decrypt(run)
    else:
        decrypted = measure_time(circuit.simulate,'Simulating', qf_arrays, qf_signs)

    print(decrypted)
    print(pivot_matrix(M))

    # Convert output to floats
    # output_mat = qfloat_arrays_to_float_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # print(output_mat) 
    # print(pivot_matrix(M))

#test_lu_decomposition(2)
#test_matrix_inverse(3,100)
#test_qf_matrix_inverse(4,4)
test_qf_fhe(2, True)