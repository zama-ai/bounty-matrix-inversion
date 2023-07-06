import numpy as np
import scipy as sc
import time
from QFloat import QFloat, BinaryValue
from concrete import fhe

Tracer = fhe.tracing.tracer.Tracer

#############################################################################################################

#                       ORIGINAL FUNCTIONS FOR LU MATRIX INVERSE (for comparison)

#############################################################################################################


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











#############################################################################################################

#                       FUNCTIONS FOR LU MATRIX INVERSE ADAPTED TO QFLOATS

#############################################################################################################



########################################################################################
#                                     UTILS
########################################################################################

def matrix_column(M,j):
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
    Convert a binary matrix M into a 2D-list matrix of BinaryValues
    """
    return map_2D_list( [[M[i,j] for j in range(M.shape[1])] for i in range(M.shape[0]) ], lambda x: BinaryValue(x) )    

def qf_list_dot_product(list1, list2):
    """
    Dot product of two QFloat lists
    """
    if len(list1) != len(list2):
        raise ValueError("Lists should have the same length.")
    
    result = list1[0] * list2[0]
    for i in range(1,len(list1)):
        result += (list1[i] * list2[i]) # in place addition is supported
    
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
            result[i][j] = qf_list_dot_product( matrix1[i], matrix_column(matrix2,j))
    
    return result


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
            elif isinstance(M[i][j], BinaryValue):
                qf_arrays[index,qf_ints-1] = M[i][j].value
            else:
                qf_arrays[index,qf_ints-1] = M[i][j]
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

    # Create identity matrix, with fhe integer values
    pivot_mat = fhe.zeros((n,n))
    for i in range(n):
        pivot_mat[i,i] = fhe.ones(1)[0]

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

def qf_lu_decomposition(M):
    """
    Performs an LU Decomposition of square QFloats 2D-list matrix M
    The function returns P, L, and U such that M = PLU. 
    """
    assert(len(M)==len(M[0]))
    n = len(M)

    # add __len__ function to Tracer for simplicity:
    def arrlen(self):
        return self.shape[0]
    Tracer.__len__ = arrlen

    # Initialize 2D-list matrices for L and U
    # (QFloats can be multiplied with integers, this saves computations)
    L = binaryListMatrix(np.zeros((n,n), dtype='int'))
    U = binaryListMatrix(np.zeros((n,n), dtype='int'))

    # Create the pivot matrix P and the multiplied matrix PM
    P = binaryListMatrix(qf_pivot_matrix(M))
    PM = qf_list_matrix_multiply(P, M)

    # Perform the LU Decomposition
    for j in range(n):
        # All diagonal entries of L are set to unity
        L[j][j] = BinaryValue(1)
        # u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
        for i in range(j+1):
            if i>0:
                s1 = qf_list_dot_product([U[k][j] for k in range(0,i)], [ L[i][k] for k in range(0,i) ])
                U[i][j] = PM[i][j] - s1
            else:
                U[i][j] = PM[i][j].copy()

        # l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik})
        for i in range(j, n):
            if j>0:
                s2 = qf_list_dot_product([U[k][j] for k in range(0,j)], [L[i][k] for k in range(0,j)])
                L[i][j] = (PM[i][j] - s2) / U[j][j]
            else:
                L[i][j] = PM[i][j] / U[j][j]

    # PM = LU
    P= transpose_2DList(P)

    # now M = PLU
    return P, L, U


########################################################################################
#                                   LU INVERSE
########################################################################################

def qf_lu_inverse(P, L, U):
    n = len(L)

    # Forward substitution: Solve L * Y = P * A for Y
    Y = np.zeros((n,n), dtype='int').tolist()
    for i in range(n):
        Y[i][0] = P[i][0] / L[0][0]
        for j in range(1, n):
            Y[i][j] = (P[i][j] - qf_list_dot_product([ L[j][k] for k in range(j) ], [ Y[i][k] for k in range(j)])) / L[j][j]

    # Backward substitution: Solve U * X = Y for X
    X = np.zeros((n,n), dtype='int').tolist()
    for i in range(n - 1, -1, -1):
        X[i][-1] = Y[i][-1] / U[-1][-1]
        for j in range(n - 2, -1, -1):
            X[i][j] = (Y[i][j] - qf_list_dot_product( [ U[j][k] for k in range(j+1,n)], [ X[i][k] for k in range(j+1,n) ])) / U[j][j]

    return transpose_2DList(X)



########################################################################################
#                                   INVERSE FUNCTION
########################################################################################

def qf_matrix_inverse(qf_arrays, qf_signs, params):
    """
    Main function
    """
    n, qf_len, qf_ints, qf_base = params

    assert( n*n == qf_arrays.shape[0])
    assert( qf_len == qf_arrays.shape[1])

    # reconstruct the matrix of QFloats with encrypted values:
    qf_M = qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # compute the LU decomposition
    bin_P, qf_L, qf_U = qf_lu_decomposition(qf_M)

    # compute inverse from P L U
    qf_Minv = qf_lu_inverse(bin_P, qf_L, qf_U)

    # break the resulting QFloats into arrays:
    qf_inv_arrays = qfloat_matrix_to_arrays(qf_Minv, qf_len, qf_ints, qf_base)

    return qf_inv_arrays






#############################################################################################################

#                                                  TESTS

#############################################################################################################



def measure_time(function, descripton, *inputs):
    #Compute a function on inputs and return output along with duration
    print(descripton+' ...', end="", flush=True)
    print("\r", end="")
    start = time.time()
    output = function(*inputs)
    end = time.time()
    print(f"|  {descripton} : {end-start:.2f} s  |")
    return output


def test_qf_PLU_python(n, qf_len, qf_ints, qf_base):
    """
    Test the PLU decomposition using QFloats, in python
    """
    # gen random matrix
    M = np.random.uniform(0, 100, (n,n))

    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    # reconstruct the matrix of QFloats 
    qf_M = qfloat_arrays_to_QFloat_matrix(qf_arrays, qf_signs, qf_ints, qf_base)

    # compute the LU decomposition
    bin_P, qf_L, qf_U = qf_lu_decomposition(qf_M)

    # convert bin_P from binaryValue to floats
    P = np.array( map_2D_list(bin_P, lambda x: x.value) )

    # break the resulting QFloats into arrays:
    qf_arrays_L = qfloat_matrix_to_arrays(qf_L, qf_len, qf_ints, qf_base)
    qf_arrays_U = qfloat_matrix_to_arrays(qf_U, qf_len, qf_ints, qf_base)

    L = qfloat_arrays_to_float_matrix(qf_arrays_L, qf_ints, qf_base)
    U = qfloat_arrays_to_float_matrix(qf_arrays_U, qf_ints, qf_base)

    print(' PIVOT MATRIX\n============')
    print('QFloat P :')
    print(P)
    print(' ')

    P_, L_, U_ = lu_decomposition(M)
    print('PLU P :')
    print(P_)
    print('') 
   
    print(' L MATRIX\n============')
    print('QFloat L :')
    print(L)
    print(' ')

    print('QFloat L :')
    print(L_)
    print(' ')

    print(' U MATRIX\n============')
    print('QFloat U :')
    print(U)
    print(' ')

    print('PLU U :')
    print(U_)
    print(' ') 


def test_qf_inverse_python(n, qf_len, qf_ints, qf_base):
    """
    Test the matrix inverse using QFloats, in python
    """
    # gen random matrix
    M = np.random.uniform(0, 100, (n,n))

    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    # set params
    params = [n, qf_len, qf_ints, qf_base]

    QFloat.KEEP_TIDY=True
    output = qf_matrix_inverse(qf_arrays, qf_signs, params)

    qf_Res = qfloat_arrays_to_float_matrix(output, qf_ints, qf_base)

    print('QFloat inverse :')
    print(qf_Res)
    print(' ')

    print('LU inveres :')
    print(matrix_inverse(M))
    print(' ') 

    print('Scipy inv :')    
    print(sc.linalg.inv(M))    


def compile_circuit(n, qf_len, qf_ints, qf_base, keep_tidy=True):

    # set params
    params = [n, qf_len, qf_ints, qf_base]

    QFloat.KEEP_TIDY=keep_tidy

    compiler = fhe.Compiler(lambda x,y: qf_matrix_inverse(x,y,params), {"x": "encrypted", "y": "encrypted"})
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

    QFloat.KEEP_TIDY=True

    circuit = measure_time( make_circuit, 'Compiling')
    return circuit


def test_qf_inverse_fhe(n, circuit, qf_len, qf_ints, qf_base, simulate=False):
    # gen random matrix
    M = np.random.uniform(0, 100, (n,n))

    # convert it to QFloat arrays
    qf_arrays, qf_signs = float_matrix_to_qfloat_arrays(M, qf_len, qf_ints, qf_base)

    # Run FHE
    if not simulate:
        encrypted = measure_time(circuit.encrypt, 'Encrypting', qf_arrays, qf_signs)
        run = measure_time(circuit.run,'Running', encrypted)
        decrypted = circuit.decrypt(run)
    else:
        decrypted = measure_time(circuit.simulate,'Simulating', qf_arrays, qf_signs)

    qf_Res = qfloat_arrays_to_float_matrix(decrypted, qf_ints, qf_base)

    print(qf_Res)

    Minv = matrix_inverse(M)
    print(Minv)


if __name__ == '__main__':

    n=2
    qf_len = 20
    qf_ints = 9
    qf_base = 2

    # circuit_n = compile_circuit(n, qf_len, qf_ints, qf_base)
    # test_qf_inverse_fhe(n, circuit, qf_len, qf_ints, qf_base, False, False)

    #results for 16, 9, 2

    #test_qf_PLU_python(n, qf_len, qf_ints, qf_base)
    test_qf_inverse_python(n, qf_len, qf_ints, qf_base)

