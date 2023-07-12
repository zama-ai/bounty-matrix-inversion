import sys, os, time

import unittest
import numpy as np
from concrete import fhe

sys.path.append(os.getcwd())

from matrix_inversion.QFloat import QFloat, SignedBinary


SIMULATE=False
base=2
floatLenght = 8
QFloat.KEEP_TIDY=False


def print_red(text):
    # ANSI escape sequence for red color
    red_color = "\033[91m"
    # ANSI escape sequence to reset color back to default
    reset_color = "\033[0m"
    # Print text in red
    print(red_color + text + reset_color)    

def measure_time(function, descripton, *inputs):
    #Compute a function on inputs and return output along with duration
    print(descripton+' ...', end="", flush=True)
    print("\r", end="")
    start = time.time()
    output = function(*inputs)
    end = time.time()
    print(f"|  {descripton} : {end-start:.2f} s  |")
    return output

def float_array_to_qfloat_arrays_fhe(arr, qf_len, qf_ints, qf_base):
    """
    converts a float list to arrays representing qfloats
    """
    qf_array = [ QFloat.fromFloat(f, qf_len, qf_ints, qf_base) for f in arr]
    n=len(qf_array)
    qf_arrays = fhe.zeros((n, qf_len))
    qf_signs = fhe.zeros(n)
    for i in range(n):
        qf_arrays[i,:] = qf_array[i].toArray()
        qf_signs[i] = qf_array[i].getSign()

    return qf_arrays, qf_signs

def float_array_to_qfloat_arrays_python(arr, qf_len, qf_ints, qf_base):
    """
    converts a float list to arrays representing qfloats
    """
    qf_array = [ QFloat.fromFloat(f, qf_len, qf_ints, qf_base) for f in arr]
    n=len(qf_array)
    qf_arrays = np.zeros((n, qf_len), dtype='int')
    qf_signs = np.zeros(n, dtype='int')
    for i in range(n):
        qf_arrays[i,:] = qf_array[i].toArray()
        qf_signs[i] = qf_array[i].getSign()

    return qf_arrays, qf_signs


def qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base):
    """
    converts qfloats arrays to a QFloat matrix
    """
    n = int(qf_arrays.shape[0])
    qf_L = []
    for i in range(n):
        qf = QFloat(qf_arrays[i,:], qf_ints, qf_base, True, qf_signs[i])
        qf_L.append(qf)

    return qf_L

def qfloat_list_to_qfloat_arrays(L, qf_len, qf_ints, qf_base):
    """
    converts a QFloat 2D-list matrix to integer arrays 
    """
    if not isinstance(L, list):
        raise TypeError('L must be list')
    n=len(L)
    qf_arrays = fhe.zeros((n, qf_len))
    for i in range(n):
        if isinstance(L[i], QFloat):
            qf_arrays[i,:] = L[i].toArray()
        elif isinstance(L[i], SignedBinary):
            qf_arrays[i,qf_ints-1] = L[i].value
        elif isinstance(L[i], Zero):
            qf_arrays[i,qf_ints-1] = 0
        else:
            qf_arrays[i,qf_ints-1] = L[i]

    return qf_arrays

def qfloat_arrays_to_float_array(qf_arrays, qf_ints, qf_base):
    """
    converts qfloats arrays to a float matrix
    """
    n = int(qf_arrays.shape[0])
    arr=np.zeros(n)
    for i in range(n):
        arr[i] = QFloat(qf_arrays[i,:], qf_ints, qf_base).toFloat()

    return arr



class QFloatCircuit:
   
    """
    Circuit factory class for testing FheSeq on 2 sequences input
    """
    def __init__(self, n, circuit_function, qf_len, qf_ints, qf_base, verbose=False):
        inputset=[]
        for i in range(100):
            floatList = [np.random.uniform(0,100,1)[0] for i in range(n)]
            qf_arrays, qf_signs = float_array_to_qfloat_arrays_python(floatList, qf_len, qf_ints, qf_base)
            inputset.append((qf_arrays, qf_signs))

        params = [qf_len, qf_ints, qf_base]
        compiler = fhe.Compiler(lambda x,y: circuit_function(x,y,params), {"x": "encrypted", "y": "encrypted"})
        make_circuit = lambda : compiler.compile(
            inputset=inputset,
            configuration=fhe.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
                #dataflow_parallelize=True,
            ),
            verbose=False,
        )
        self.qf_len = qf_len
        self.qf_ints = qf_ints
        self.qf_base = qf_base

        self.circuit = measure_time( make_circuit, 'Compiling')

    
    def run(self, floatList, simulate=False, raw_output=False):
        if not self.circuit: raise Error('circuit was not set')
        qf_arrays, qf_signs = float_array_to_qfloat_arrays_fhe(floatList, self.qf_len, self.qf_ints, self.qf_base)

        # Run FHE
        if not simulate:
            encrypted = measure_time(self.circuit.encrypt, 'Encrypting', qf_arrays, qf_signs)
            run = measure_time(self.circuit.run,'Running', encrypted)
            decrypted = self.circuit.decrypt(run)
        else:
            decrypted = measure_time(self.circuit.simulate,'Simulating', qf_arrays, qf_signs)

        if not raw_output:
            qf_Res = qfloat_arrays_to_float_array(decrypted, self.qf_ints, self.qf_base)
        else:
            qf_Res = decrypted

        return qf_Res



class TestQFloat(unittest.TestCase):

 ##################################################  NON FHE TESTS ##################################################

    def test_conversion_np(self):

        # test conversion from float to QFloat and vice-versa
        for i in range(100):
            base = np.random.randint(2,10)
            size = np.random.randint(20,30)
            ints = np.random.randint(8,12)
            f = (np.random.randint(0,20000)-10000)/100 # float of type (+/-)xx.xx
            qf = QFloat.fromFloat(f, size, ints, base)
            if not ( (qf.toFloat()-f) < 0.1 ):
                raise ValueError( 'Wrong QFloat: ' + str(qf) + ' for float: ' + str(f) + ' and conversion: ' + str(qf.toFloat()))

        # mixed values
        qf = QFloat(np.array([0,-1,1,0,1,0]), 3, 2)
        assert(qf.toFloat()==-0.75)

    def test_str(self):
        # positive value:
        qf = QFloat.fromFloat(13.75, 10, 5, 2)
        assert(str(qf)=='01101.11000')

        # negative value
        qf = QFloat.fromFloat(-13.75, 10, 5, 2)
        assert(str(qf)=='-01101.11000')

        #zero
        qf = QFloat.fromFloat(0, 10, 5, 2)
        assert(str(qf)=='00000.00000')


    def test_getSign_np(self):
        # zero
        f = 0
        qf = QFloat.fromFloat(f, 10, 5, 2)
        if not ( qf.getSign() == 1 ): # sign of 0 is 1
            print(qf.getSign())
            raise ValueError( 'Wrong sign for QFloat: ' + str(qf) + ' for float: ' + str(f))        

        # non zero
        for i in range(100):
            base = np.random.randint(2,10)
            size = np.random.randint(20,30)
            ints = np.random.randint(8,12)
            f = (np.random.randint(0,20000)-10000)/100 # float of type (+/-)xx.xx
            if f==0:
                f+=1
            qf = QFloat.fromFloat(f, size, ints, base)
            if not ( qf.getSign() == np.sign(f)  ):
                raise ValueError( 'Wrong sign for QFloat: ' + str(qf) + ' for float: ' + str(f))


    def test_add_sub_np(self):
        # test add and sub
        for i in range(100):
            base = np.random.randint(2,10)
            size = np.random.randint(20,30)
            ints = np.random.randint(8,12)
            f1 = (np.random.randint(0,20000)-10000)/100 # float of type (+/-)xx.xx
            f2 = (np.random.randint(0,20000)-10000)/100 # float of type (+/-)xx.xx
            qf1 = QFloat.fromFloat(f1, size, ints, base)
            assert( (2+qf1).toFloat() - (2+f1) < 0.1)
            assert( (qf1+2).toFloat() - (2+f1) < 0.1)
            assert( (SignedBinary(1)+qf1).toFloat() - (1+f1) < 0.1)

            assert( (2-qf1).toFloat() - (2-f1) < 0.1)
            assert( (qf1-2).toFloat() - (f1-2) < 0.1)
            assert( (SignedBinary(1)-qf1).toFloat() - (1-f1) < 0.1)            

            qf2 = QFloat.fromFloat(f2, size, ints, base)
            assert( (qf1+qf2).toFloat()-(f1+f2) < 0.1 )
            assert( (qf1-qf2).toFloat()-(f1-f2) < 0.1 )
            qf1 += qf2
            assert( qf1.toFloat()-(f1+f2) < 0.1 ) # iadd

    def test_mul_np(self):
        # test multiplication by QFloat and integer
        for i in range(100):
            base = np.random.randint(2,3)
            size = np.random.randint(30,40)
            ints = np.random.randint(10,13)
            f1 = (np.random.randint(0,200)-100)/10 # float of type (+/-)x.x
            f2 = (np.random.randint(0,200)-100)/10 # float of type (+/-)x.x
            integer = np.random.randint(-2,3)
            qf1 = QFloat.fromFloat(f1, size, ints, base)
            assert( (2*qf1).toFloat() - 2*f1 < 0.1)
            assert( (qf1*2).toFloat() - 2*f1 < 0.1)
            assert( (SignedBinary(1)*qf1).toFloat() - f1 < 0.1)
            qf2 = QFloat.fromFloat(f2, size, ints, base)
            prod = qf1*qf2

            prod2 = integer*qf2
            assert( prod.toFloat()-(f1*f2) < 0.1 ) # mul
            assert( prod2.toFloat()-(integer*f2) < 0.1 ) # mul

            # from mul
            assert( QFloat.fromMul(qf1, qf2).toFloat()-(f1*f2) < 0.1 )

            qf1 *= qf2
            assert( qf1.toFloat()-(f1*f2) < 0.1 ) # imul


    def test_div_np(self):
        # test division
        for i in range(100):
            base = np.random.randint(2,3)
            size = np.random.randint(30,40)
            ints = np.random.randint(10,13)
            f1 = (np.random.randint(0,200)-100)/10 # float of type (+/-)x.x
            f2 = (np.random.randint(0,200)-100)/10 # float of type (+/-)x.x
            if f2==0:
                f2+=1.0
            if f1==0:
                f1+=1.0                
            qf1 = QFloat.fromFloat(f1, size, ints, base)

            assert( (2/qf1).toFloat() - 2/f1 < 0.1)
            assert( (qf1/2).toFloat() - f1/2.0 < 0.1)

            assert( (SignedBinary(1)/qf1).toFloat() - 1.0/f1 < 0.1)
            assert( (SignedBinary(-1)/qf1).toFloat() - (-1.0/f1) < 0.1)
            assert( (qf1/SignedBinary(0)).toFloat() > 1000) #overflow

            newlen = np.random.randint(30,40)
            newints = np.random.randint(10,13)
            assert( qf1.invert(1, newlen, newints).toFloat() - 1.0/f1 < 0.1)
            assert( (SignedBinary(-1)/qf1).toFloat() - (-1.0/f1) < 0.1)            

            qf2 = QFloat.fromFloat(f2, size, ints, base)
            div = qf1/qf2
            if not( div.toFloat()-(f1/f2) < 0.1 ):
                raise Exception('Wrong division for f1:' + str(f1) +' f2: ' + str(f2) + ' f1/f2: ' + str(f1/f2) + ' and div : '+str(div.toFloat()))

    def test_abs_np(self):
        for i in range(100):
            base = np.random.randint(2,3)
            size = np.random.randint(30,40)
            ints = np.random.randint(10,13)
            f1 = (np.random.randint(0,200)-100)/10 # float of type (+/-)x.x
            qf1 = QFloat.fromFloat(f1, size, ints, base)
            assert( abs(qf1).toFloat()-(abs(f1)) < 0.1 )


    def test_tidy_np(self):
        # mixed signs
        for i in range(100):
            base = np.random.randint(2,10)
            size = np.random.randint(20,30)
            ints = np.random.randint(size//2-2,size//2+2)
            arr = np.zeros(size)
            i1 = len(arr)//4
            i2 = 3*i1
            arr[i1:i2] = np.random.randint(-4*base,4*base, i2-i1)
            qf = QFloat(arr, ints, base, False)            
            f = qf.toFloat()
            qf.tidy()
            if not ( (f-qf.toFloat()) <= 0.0001 ):
                raise ValueError( 'Wrong tidy value for QFloat: ' + str(qf) + ' toFloat : ' + str(qf.toFloat()) + ' for actual float : ', f)
            assert(qf.getSign() == (np.sign(f) or 1)) # check computed sign as well



##################################################  FHE TESTS ##################################################

    def test_add_sub_fhe(self):
        # test add and sub

        def add_qfloats(qf_arrays, qf_signs, params):
            qf_len, qf_ints, qf_base = params
            a,b = qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base)
            res = [a+b]
            return qfloat_list_to_qfloat_arrays(res, qf_len, qf_ints, qf_base)

        for i in range(10):
            #base = np.random.randint(2,10)
            base = 2
            size = np.random.randint(20,30)
            ints = np.random.randint(12, 16)
            f1 = np.random.uniform(0,100,1)[0]
            f2 = np.random.uniform(0,100,1)[0]

            circuit = QFloatCircuit(2, add_qfloats, size, ints, base)
            addition = circuit.run(np.array([f1,f2]), False)[0]
            assert(addition  - (f1+f2) < 0.01)


    def test_mul_fhe(self):
        # test add and sub

        def mul_qfloats(qf_arrays, qf_signs, params):
            qf_len, qf_ints, qf_base = params
            a,b = qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base)
            res = [a*b]
            return qfloat_list_to_qfloat_arrays(res, qf_len, qf_ints, qf_base)

        for i in range(10):
            #base = np.random.randint(2,10)
            base = 2
            size = np.random.randint(20,30)
            ints = np.random.randint(12, 16)
            f1 = np.random.uniform(0,100,1)[0]
            f2 = np.random.uniform(0,100,1)[0]

            circuit = QFloatCircuit(2, mul_qfloats, size, ints, base)
            multiplication = circuit.run(np.array([f1,f2]), False)[0]
            assert(multiplication  - (f1*f2) < 0.01)

    def test_div_fhe(self):
        # test add and sub

        def div_qfloats(qf_arrays, qf_signs, params):
            qf_len, qf_ints, qf_base = params
            a,b = qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base)
            res = [a/b]
            return qfloat_list_to_qfloat_arrays(res, qf_len, qf_ints, qf_base)

        for i in range(10):
            #base = np.random.randint(2,10)
            base = 2
            size = np.random.randint(20,30)
            ints = np.random.randint(12, 16)
            f1 = np.random.uniform(0,100,1)[0]
            f2 = np.random.uniform(0,100,1)[0]

            circuit = QFloatCircuit(2, div_qfloats, size, ints, base)
            division = circuit.run(np.array([f1,f2]), False)[0]
            assert(division  - (f1/f2) < 0.01)  


    # def test_multi_fhe(self):
    #     # test multi operations to count time

    #     def multi_qfloats(qf_arrays, qf_signs, params):
    #         qf_len, qf_ints, qf_base = params
    #         a,b = qfloat_arrays_to_qfloat_list(qf_arrays, qf_signs, qf_ints, qf_base)
    #         res = a+a+a+a-b
    #         res = res*a*a
            
    #         return qfloat_list_to_qfloat_arrays([res], qf_len, qf_ints, qf_base)

    #     for i in range(10):
    #         #base = np.random.randint(2,10)
    #         base = 2
    #         size = np.random.randint(20,30)
    #         ints = np.random.randint(12, 16)
    #         f1 = np.random.uniform(0,100,1)[0]
    #         f2 = np.random.uniform(0,100,1)[0]

    #         QFloat.KEEP_TIDY=False
    #         circuit = QFloatCircuit(2, multi_qfloats, size, ints, base)
    #         multi = circuit.run(np.array([f1,f2]), False)[0]
    #         QFloat.KEEP_TIDY=True
    #         #assert(division  - (f1/f2) < 0.01)  


unittest.main()

# suite = unittest.TestLoader().loadTestsFromName('test_QFloat.TestQFloat.test_multi_fhe')
# unittest.TextTestRunner(verbosity=1).run(suite)