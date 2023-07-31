import sys, os, time

import unittest
import numpy as np
from concrete import fhe

sys.path.append(os.getcwd())

from matrix_inversion.QFloat import QFloat, SignedBinary


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

            # from mul with specific values
            f1 = (np.random.randint(1,100)/1.0) # float of type (+/-)xx.
            f2 = (np.random.randint(1,10000))/10000000 # float of type (+/-)0.000xxx
            qf1 = QFloat.fromFloat(f1, 18, 18, 2)
            qf2 = QFloat.fromFloat(f2, 25, 0, 2)
            # from mul
            assert( QFloat.fromMul(qf1, qf2, 18, 1).toFloat()-(f1*f2) < 0.1 )            


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




#unittest.main()

suite = unittest.TestLoader().loadTestsFromName('test_QFloat.TestQFloat.test_mul_np')
unittest.TextTestRunner(verbosity=1).run(suite)