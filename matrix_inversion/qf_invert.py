import numpy as np
import scipy as sc
import time
from QFloat import QFloat
from concrete import fhe




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
    """
    Performs an LU Decomposition of M (which must be square)
    into PM = LU. The function returns P, L, and U.
    """
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
    return PM, L, U


def matrix_column(M,j):
    """
    Extract column from 2D list matrix
    """
    return [row[j] for row in M] 

def qf_list_dot_product(list1, list2, qf0):
    """
    Dot product of two QFloat lists
    """
    if len(list1) != len(list2):
        raise ValueError("Lists should have the same length.")
    
    result = qf0.copy() # copy the zero QFloat for initialization

    for i in range(len(list1)):
        if isinstance(list1[i], fhe.tracing.tracer.Tracer): # multiply a QFloat with a Tracer, not the opposite
            result += (list2[i] * list1[i]) # in place addition is supported    
        else:
            result += (list1[i] * list2[i]) # in place addition is supported
    
    return result

def qf_list_matrix_multiply(matrix1, matrix2, qf0):
    """
    Multiply two matrices of int or QFloats
    """
    # if len(matrix1[0]) != len(matrix2):
    #     raise ValueError("Number of columns in matrix1 should match the number of rows in matrix2.")
    result = [[None] * len(matrix2[0]) for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            result[i][j] = qf_list_dot_product( matrix1[i], matrix_column(matrix2,j) , qf0)
    
    return result

def transpose_2DList(list2D):
    """
    Transpose a 2D-list matrix
    """
    return [list(row) for row in zip(*list2D)]

def qf_lu_decomposition(M, qf0):
    """
    Performs an LU Decomposition of square QFloats 2D-list matrix M
    The function returns P, L, and U such that M = PLU. 
    """
    assert(len(M)==len(M[0]))
    n = len(M)

    # add __len__ function to fhe.tracing.tracer.Tracer for simplicity:
    def arrlen(self):
        return self.shape[0]
    fhe.tracing.tracer.Tracer.__len__ = arrlen

    # Initialize 2D-list matrices for L and U
    # (QFloats can be multiplied with integers, this saves computations)
    L = np.zeros((n,n), dtype='int').tolist()
    U = np.zeros((n,n), dtype='int').tolist()

    # Create the pivot matrix P and the multiplied matrix PM
    P = qf_pivot_matrix(M)
    PM = qf_list_matrix_multiply(P, M, qf0)

    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity
        L[j][j] = 1
        # u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
        for i in range(j+1):
            if i>0:
                s1 = qf_list_dot_product([U[k][j] for k in range(0,i)], [ L[i][k] for k in range(0,i) ], qf0)
                U[i][j] = PM[i][j] - s1
            else:
                U[i][j] = PM[i][j].copy()

        # l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
        for i in range(j, n):
            if j>0:
                s2 = qf_list_dot_product([U[k][j] for k in range(0,j)], [L[i][k] for k in range(0,j)], qf0)
                L[i][j] = (PM[i][j] - s2) / U[j][j]
            else:
                L[i][j] = PM[i][j] / U[j][j]

    # PM = LU
    P= transpose_2DList(P)

    # now M = PLU
    return PM, L, U





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
            Y[i][j] = (P[i][j] - np.dot(L[j][:j], Y[i][:j])) / L[j][j]

    # Backward substitution: Solve U * X = Y for X
    X = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        X[i][-1] = Y[i][-1] / U[-1][-1]
        for j in range(n - 2, -1, -1):
            X[i][j] = (Y[i][j] - np.dot(U[j][j+1:], X[i][j+1:])) / U[j][j]

    return np.transpose(X)


def qf_lu_inverse(P, L, U, qf0):
    n = len(L)

    # Forward substitution: Solve L * Y = P * A for Y
    Y = np.zeros((n,n), dtype='int').tolist()
    for i in range(n):
        Y[i][0] = P[i][0] / L[0][0]
        for j in range(1, n):
            Y[i][j] = (P[i][j] - qf_list_dot_product([ L[j][k] for k in range(j) ], [ Y[i][k] for k in range(j)], qf0)) / L[j][j]

    # Backward substitution: Solve U * X = Y for X
    X = np.zeros((n,n), dtype='int').tolist()
    for i in range(n - 1, -1, -1):
        X[i][-1] = Y[i][-1] / U[-1][-1]
        for j in range(n - 2, -1, -1):
            X[i][j] = (Y[i][j] - qf_list_dot_product( [ U[j][k] for k in range(j+1,n)], [ X[i][k] for k in range(j+1,n) ], qf0)) / U[j][j]

    return transpose_2DList(X)




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

def qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base):
    """
    converts qfloats arrays to a QFloat matrix
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

def qfloat_arrays_to_float_matrix(qf_arrays, qf_ints, qf_base):
    """
    converts qfloats arrays to a float matrix
    """
    n = int(np.sqrt(qf_arrays.shape[0]))
    M = np.zeros((n,n))
    index=0
    for i in range(n):
        for j in range(n):
            M[i,j] = QFloat(qf_arrays[index,:], qf_ints, qf_base).toFloat()
            index+=1

    return M    

def qfloat_matrix_to_arrays(M, qf_len, qf_ints, qf_base):
    """
    converts a QFloat 2D-list matrix to integer arrays 
    """
    n=len(M)
    assert(n==len(M[0]))
    qf_arrays = fhe.zeros((n*n, qf_len))
    index=0
    for i in range(n):
        for j in range(n):
            if isinstance(M[i][j], QFloat):
                qf_arrays[index,:] = M[i][j].toArray()
            else:
                qf_arrays[index,qf_ints-1] = M[i][j]                
            index += 1

    return qf_arrays


def qf_matrix_inverse(qf_arrays, qf_signs, params):
    """
    Main function
    """
    n, qf_len, qf_ints, qf_base = params

    assert( n*n == qf_arrays.shape[0])
    assert( qf_len == qf_arrays.shape[1])

    # reconstruct the matrix of QFloats with encrypted values:
    qf_M = qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # a zero QFloat with good format
    qf0 = QFloat.zero(qf_len, qf_ints, qf_base)

    # compute the LU decomposition
    qf_P, qf_L, qf_U = qf_lu_decomposition(qf_M, qf0)

    # compute inverse from P L U
    qf_Minv = qf_lu_inverse(qf_P, qf_L, qf_U, qf0)

    # break the resulting QFloats into arrays:
    qf_arrays_out = qfloat_matrix_to_arrays(qf_Minv, qf_len, qf_ints, qf_base)

    return qf_arrays_out

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
    print(descripton+' ...', end="", flush=True)
    print("\r", end="")
    start = time.time()
    output = function(*inputs)
    end = time.time()
    print(f"|  {descripton} : {end-start:.2f} s  |")
    return output


def test_qf_fhe(n, simulate=False):
    # gen random matrix
    M = np.random.uniform(0, 100, (n,n))
    qf_base=2
    qf_len = 12
    qf_ints = 8

    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    # set params
    params = [n, qf_len, qf_ints, qf_base]

    QFloat.KEEP_TIDY=False

    compiler = fhe.Compiler(lambda x,y: fhematrix(x,y,params), {"x": "encrypted", "y": "encrypted"})
    make_circuit = lambda : compiler.compile(
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
    print("Matrix inversion\n")
 
    circuit = measure_time( make_circuit, 'Compiling')

    # Run FHE
    if not simulate:
        encrypted = measure_time(circuit.encrypt, 'Encrypting', qf_arrays, qf_signs)
        run = measure_time(circuit.run,'Running', encrypted)
        decrypted = circuit.decrypt(run)
    else:
        decrypted = measure_time(circuit.simulate,'Simulating', qf_arrays, qf_signs)

    QFloat.KEEP_TIDY=True

    qf_Res = qfloat_arrays_to_float_matrix(decrypted, qf_ints, qf_base)

    print(qf_Res)

    P, L, U = lu_decomposition(M)
    print(U)

    # Convert output to floats
    # output_mat = qfloat_arrays_to_float_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # print(output_mat) 
    # print(pivot_matrix(M))


def test_qf_python(n):
    # gen random matrix
    M = np.random.uniform(0, 100, (n,n))
    N = np.random.uniform(0, 100, (n,n))
    qf_base= 2
    qf_len = 30
    qf_ints = 8

    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    # set params
    params = [n, qf_len, qf_ints, qf_base]

    #QFloat.KEEP_TIDY=False

    output = qf_matrix_inverse(qf_arrays, qf_signs, params)

    QFloat.KEEP_TIDY=True

    qf_Res = qfloat_arrays_to_float_matrix(output, qf_ints, qf_base)

    print(qf_Res)
    print(' ')

    Minv = matrix_inverse(M)
    print(Minv)    

#test_lu_decomposition(2)
#test_matrix_inverse(3,100)
#test_qf_matrix_inverse(4,4)
#test_qf_fhe(2, True)
test_qf_python(2)