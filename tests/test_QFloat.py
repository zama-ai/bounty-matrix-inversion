import sys, os

import unittest
import numpy as np
from concrete import fhe

sys.path.append(os.getcwd())

from matrix_inversion.QFloat import QFloat


class QFloatCircuit:
   
    """
    Circuit factory class for testing FheSeq on 2 sequences input
    """
    def __init__(self, length, base=2, simulate=False):
        self.length=length
        self.inputset=[
            (np.random.randint(-base, base, size=(length,)),
            np.random.randint(-base, base, size=(length,)))
            for _ in range(100)
        ]
        self.circuit = None
        self.simulate = simulate

    def set(self, circuitFunction, verbose=False):
        compiler = fhe.Compiler(lambda data1,data2: circuitFunction(data1, data2), {"data1": "encrypted", "data2": "encrypted"})
        self.circuit = compiler.compile(
            inputset=self.inputset,
            configuration=fhe.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
                #dataflow_parallelize=True,
            ),
            verbose=verbose,
        )
    
    def run(self, array1, array2):
        if not self.circuit: raise Error('circuit was not set')
        assert len(array1) == self.seq_length, f"Sequence 1 length is not correct, should be {self.seq_length} characters"
        assert len(array2) == self.seq_length, f"Sequence 2 length is not correct, should be {self.seq_length} characters"

        return self.circuit.simulate(array1, array2) if self.simulate else self.circuit.encrypt_run_decrypt(array1, array2)


SIMULATE=False
base=2
floatLenght = 8

class TestQFloat(unittest.TestCase):


    def test_conversion_np(self):

        # test conversion from float to QFloat and vice-versa

        # positive value:
        qf = QFloat.fromFloat(13.75, 10, 5, 2)
        assert(str(qf)=='01101.11000')
        assert(qf.toFloat()==13.75)

        # negative value
        qf = QFloat.fromFloat(-13.75, 10, 5, 2)
        assert(str(qf)=='0-1-10-1.-1-1000')
        assert(qf.toFloat()==-13.75)

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
        print(qf.toFloat())
        assert(qf.toFloat()==-0.75)


    def test_operands_np(self):
        
        # test add and sub
        for i in range(100):
            base = np.random.randint(2,10)
            size = np.random.randint(20,30)
            ints = np.random.randint(8,12)
            f1 = (np.random.randint(0,20000)-10000)/100 # float of type (+/-)xx.xx
            f2 = (np.random.randint(0,20000)-10000)/100 # float of type (+/-)xx.xx
            qf1 = QFloat.fromFloat(f1, size, ints, base)
            qf2 = QFloat.fromFloat(f2, size, ints, base)
            assert( (qf1+qf2).toFloat()-(f1+f2) < 0.1 )
            assert( (qf1-qf2).toFloat()-(f1-f2) < 0.1 )


    # def test_operands(self):
    #     circuit = QFloatCircuit(6, base, SIMULATE)

    #     circuit.set(lambda x,y: FheSeq(x)==FheSeq(y) , True)
    #     assert( circuit.run(seq1, seq1) )


#unittest.main()

suite = unittest.TestLoader().loadTestsFromName('test_QFloat.TestQFloat.test_operands_np')
unittest.TextTestRunner(verbosity=1).run(suite)