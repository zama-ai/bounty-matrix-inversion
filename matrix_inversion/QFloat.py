import numpy as np
import numbers
from concrete import fhe

Tracer = fhe.tracing.tracer.Tracer

#=======================================================================================================================
#                            Functions operating on arrays in base p (p=2 means binary)
#=======================================================================================================================

def base_p_to_int(arr, p):
    """
    Convert base p array to int
    Can be signed (+/- values)
    """
    # Flip the array to have least significant digit at the start
    arr = np.flip(arr)
    # Compute the place values (p^0, p^1, p^2, ...)
    place_values = p**np.arange(arr.size)
    # Return the sum of arr*place_value
    return np.sum(arr * place_values)

def int_to_base_p(integer, n, p):
    """
    Convert a base-p array to a float of the form 0.xxx..
    Can be signed (+/- values)
    """    
    if n==0:
        return np.array([])
    # Convert the integer to a base p string
    base_p_string= np.base_repr(np.abs(integer), p)
    # Prepend zeros to the binary representation until it reaches the desired size
    base_p_string = base_p_string.zfill(n)
    if len(base_p_string) > n:
        raise Exception(f"integer: {integer} cannot be represented with {n} values in base {p}")
    # Convert the base_p string to a NumPy array of integers
    base_p_array = np.sign(integer)*np.array([int(digit) for digit in base_p_string])
    return base_p_array

def base_p_to_float(arr, p):
    """
    Convert a base-p array to a float of the form 0.xxx..
    Can be signed (+/- values)
    """
    f = 0.0
    for i in range(len(arr)):
        f += arr[i] * (p ** -(i+1))
    return f

def float_to_base_p(f, precision, p):
    """
    Convert a float of type 0.xxx.. to a base-p array with a given precision.
    Can be signed (+/- values)
    """
    sgn = np.sign(f)
    f=np.abs(f)
    assert(0 <= f < 1), 'Input should be a float between 0 and 1 (exclusive)'
    basep = []
    while f and len(basep) < precision:
        f *= p
        digit = int(f)
        if digit > 0:
            f -= digit
            basep.append(digit)
        else:
            basep.append(0)
    while len(basep) < precision:
        basep.append(0)
    return sgn*np.array(basep)

def base_p_addition(a, b, p, inplace=False):
    """
    Add arrays in base p. (a and b must be positive and tidy)
    If a and b have different sizes, the longer array is considered to have extra zeros to the left
    if inplace is True, the result is written is a
    """
    carry = 0
    if inplace:
        result = a
    else:
        result = fhe.zeros(a.size)

    # Loop through both arrays and perform binary addition
    for i in range(min(a.size, b.size)):
        bit_sum = a[-i-1] + b[-i-1] + carry
        result[-i-1] = bit_sum % 2
        carry = bit_sum // 2

    return result

def base_p_subtraction_overflow(a, b, p):
    """
    Subtract arrays in base p. (a and b must be tidy)
    If a and b have different sizes, the longer array is considered to have extra zeros to the left
    If a < b, the result is wrong and full of ones on the left part, so we can return in addition wether a < b
    """
    difference = fhe.zeros(a.size)
    borrow = 0
    for i in range(min(a.size,b.size)):
        # Perform subtraction for each bit
        temp = a[-i-1] - b[-i-1] - borrow
        borrow = temp < 0
        difference[-i-1] = temp + p*borrow
    return difference, borrow    

def base_p_subtraction(a, b, p):
    """
    Subtract arrays in base p. (a and b must be tidy, postive with a >= b)
    Also, if they have different sizes, the longer array is considered to have extra zeros to the left
    """
    return base_p_subtraction_overflow(a,b,p)[0]

def base_p_division(dividend, divisor, p):
    """
    Divide arrays in base p. (dividend and divisor must be tidy and positive)
    """    
    # Initialize the quotient array
    quotient = fhe.zeros(dividend.size)
    # Initialize the remainder
    remainder = dividend[0].reshape(1)

    for i in range(dividend.size):
        if i>0:
            # Left-roll the remainder and bring down the next bit from the dividend
            # also cut remainder if its size is bigger than divisor's, cause there are extra zeros
            d=1*(remainder.size > divisor.size)
            remainder = np.concatenate((remainder[d:], dividend[i].reshape(1)), axis=0)
        # If the remainder is larger than or equal to the divisor
        for j in range(p-1):
            is_ge = is_greater_or_equal_base_p(remainder, divisor)
            # Subtract the divisor from the remainder
            remainder = is_ge*base_p_subtraction(remainder, divisor, p) + (1-is_ge)*remainder
            # Set the current quotient bit to 1
            quotient[i] += is_ge

    return quotient   

# def base_p_division_fp(dividend, divisor, p, fp):
#     """
#     Divide arrays in base p. (dividend and divisor must be tidy and positive)
#     Optimized for fp (float prevision fp) where we consider that the fp first 
#     digits of the result should be zeros, (else it would be an overflow) and
#     so start with a biger remainder of size fp
#     """    
#     # Initialize the quotient array
#     quotient = fhe.zeros(dividend.size)
#     # Initialize the remainder
#     remainder = dividend[0:fp]

#     for i in range(fp,dividend.size):
#         if i>0:
#             # Left-roll the remainder and bring down the next bit from the dividend
#             # also cut remainder if its size is bigger than divisor's, cause there are extra zeros
#             d=1*(remainder.size > divisor.size)
#             remainder = np.concatenate((remainder[d:], dividend[i].reshape(1)), axis=0)
#         # If the remainder is larger than or equal to the divisor
#         for j in range(p-1): 
#             is_ge = is_greater_or_equal_base_p(remainder, divisor)
#             # Subtract the divisor from the remainder
#             remainder = is_ge*base_p_subtraction(remainder, divisor, p) + (1-is_ge)*remainder
#             # Set the current quotient bit to 1
#             quotient[i] += is_ge

#     return quotient      

def is_greater_or_equal(a, b):
    """
    Fast computation of wether an array number a is greater or equal to an array number b
    Both arrays must be base tidy, in which case the subtraction of a-b will work if a>=b and overflow if a<b
    The overflow is a fast way to compute wether a>=b <=> not a<b
    """
    borrow = 0
    for i in range(min(a.size,b.size)):
        # report borrow
        borrow = a[-i-1] - b[-i-1] - borrow < 0
    return 1-borrow  

def is_equal(a, b):
    """
    Computes wether an array is equal to another
    """
    return (a.size - np.sum(a == b))==0

def is_positive(a):
    """
    Fast computation of wether an array number a is positive (or zero)
    a must be base tidy (returns the first non zero sign)
    """
    borrow = 0
    for i in range(a.size):
        # report borrow
        borrow = a[-i-1] - borrow < 0
    return 1-borrow  

def is_greater_or_equal_base_p(a, b):
    """
    Computes wether a base-p number (little endian) is greater or equal than another, works for different sizes
    """    
    diff=b.size-a.size
    if diff==0:
        return is_greater_or_equal(a, b)
    elif diff>0:
        return is_greater_or_equal(a, b[diff:]) & (np.sum(b[0:diff])==0)
    else:
        return is_greater_or_equal(a[-diff:], b) | (np.sum(a[0:-diff])>0)


def insert_array_at_index(a, B, i, j):
    """
    Insert elements of array a into array B[i,:] starting at index j
    """
    # Case when i is negative
    if j < 0:
        # We will take elements of a starting at index -j
        a = a[-j:]
        j = 0
    # Compute the number of elements we can insert from a into b
    n = min(B[i].size - j, a.size)
    # Insert elements from a into B[i]
    B[i,j:j+n] = a[:n]


#=======================================================================================================================
#                                                       Zero
#=======================================================================================================================

class Zero():
    """
    A simple class to differentiate values that we know are zero from others
    It is usefull to save computations for operations with QFloats
    """
    def __init__(self):
        pass

    def copy(self):
        return Zero()

    def toFloat(self):
        return float(0)

    def __add__(self, other):
        if isinstance(other, Zero):
            return self
        else:
            return other

    def __radd__(self, other):
        if isinstance(other, Zero):
            return self
        else:
            return other            

    def __sub__(self, other):
        if isinstance(other, Zero):
            return self        
        else:
            return -other

    def __rsub__(self, other):
        return other

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self        

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        raise Exception('division by Zero')

    def __neg__(self):
        return self

    def neg(self):
        return self

#=======================================================================================================================
#                                                       SignedBinary
#=======================================================================================================================

class SignedBinary():
    """
    A simple class to differentiate values that we know are binary (+1, -1 or 0) from others
    It is usefull to save computations for operations with QFloats
    The value can be encrypted or not, as long as it is binary
    """
    def __init__(self, value):
        # init with a signed binary value (+1, -1 or 0)
        self._value = value
        self._encrypted = isinstance(value, Tracer)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, newvalue):
        self._value = value
        self._encrypted = isinstance(value, Tracer)

    def copy(self):
        return SignedBinary(self._value)

    def toFloat(self):
        return float(self._value)

    def __add__(self, other):
        if isinstance(other, SignedBinary):
            # potentially no more a binary
            return self._value + other._value
        elif isinstance(other, QFloat):
            return other.__add__(self)
        else:
            return self._value + other

    def __sub__(self, other):
        if isinstance(other, SignedBinary):
            # potentially no more a binary
            return self._value - other._value
        elif isinstance(other, QFloat):
            return other.__rsub__(self)            
        else:
            return self._value - other

    def __mul__(self, other):
        if isinstance(other, SignedBinary):
            # stays binary
            return SignedBinary(self._value * other._value)
        elif isinstance(other, QFloat):
            return other.__mul__(self)            
        else:
            return self._value * other    

    def __truediv__(self, other):
        if isinstance(other, SignedBinary):
            # stays binary
            return SignedBinary(self._value // other._value)
        elif isinstance(other, QFloat):
            return other.__rtruediv__(self)
        else:
            return self._value / other

    def __neg__(self):
        return SignedBinary(-1*self._value)

    def neg(self):
        self._value *= -1
        return self

    def __abs__(self):
        return SignedBinary(np.abs(self._value))
      

#=======================================================================================================================
#                                                       QFfloats
#=======================================================================================================================

class QFloat():
    """
    A class for quantizing floats
    Floats are encoded as encrypted or unencrypted arrays of integers in a specified base (the fastest for fhe is base 2)
    Encrypted QFloat can be summed, multiplied etc. with unencrypted QFloats, and vice-versa

    TODO: replace length + ints with length + dot position where dot position can be negative to encode efficiently
    for very low numbers. For isntance: 0.000000001110 encoded as -.--------1110 with dot position = -8
    TODO: fast inversion algorithm using dichotomia search
    """

    # Statistics for counting operations between QFloats within a circuit
    ADDITIONS=0
    MULTIPLICATION=0
    DIVISION=0

    def __init__(self, array, ints=None, base=2, isBaseTidy=True, sign=1):
        """
        - array: an encrypted or unencrypted array representing a number in base p (little endian)
        - ints: gives the number of digits before the dot, so ints = 1 will encode a number like x.xxxx...
        - isBaseTidy: wether the array is already basetidy
        - sign: provide the sign: 1 or -1
        """
        if not (isinstance(array, np.ndarray) or isinstance(array, Tracer)):
            raise ValueError('array must be np.ndarray or Tracer')

        if isinstance(array, np.ndarray):
            array=array.astype('int')

        self._encrypted = isinstance(array, Tracer)

        if len(array.shape) > 1:
            raise ValueError('array must be one dimension')
        self._array = array[:]

        if not self._encrypted:
            self._array = self._array.astype(int)

        if not (isinstance(base, int) and base > 1):
            raise ValueError('base must be a int >1')
        self._base = base       
        if ints is None:
            ints = array.size//2
        else:
            if not (isinstance(ints, int) and 0<=ints and array.size>=ints):
                raise ValueError('ints must be in range [0,array.size]')
        self._ints = int(ints)
        self._sign = sign # a sign of 0 makes the QFloat null
        if isinstance(self._sign, float):
            self._sign=int(self._sign)

        self._isBaseTidy = isBaseTidy
        if not self._isBaseTidy:
            self.baseTidy()

    def resetStats():
        QFloat.ADDITIONS=0
        QFloat.MULTIPLICATION=0
        QFloat.DIVISION=0

    def showStats():
        print('\nQFloat statistics :')
        print('======================')
        print('Additions       : ' +  str(QFloat.ADDITIONS))
        print('Multiplications : ' +  str(QFloat.MULTIPLICATION))
        print('Divisions       : ' +  str(QFloat.DIVISION))
        print('\n')

    def _checkUnencrypted(self):
        if self._encrypted:
            raise Exception("This function does not work on encrypted QFloats")

    #=============================================================================================
    #                        Functions for unencrypted QFloats only
    #=============================================================================================

    def toStr(self, tidy=True):
        """
        Convert the QFloat to a string

        WARNING : does not work on encrypted QFLoat
        """
        self._checkUnencrypted()

        if tidy: # tidy before return the representation
            self.baseTidy()

        sgn = self.getSign()

        integerPart = (self._array[0:self._ints]*(sgn!=0)).astype('int') # 0 is sign is 0
        floatPart = (self._array[self._ints:]*(sgn!=0)).astype('int') # 0 is sign is 0

        if self._base <= 10:
            integerPart = ''.join([str(i) for i in integerPart])
            floatPart = ''.join([str(i) for i in floatPart])
        else:
            integerPart = str(integerPart)
            floatPart = str(floatPart)

        sgnstr = '' if sgn >=0 else '-'

        return sgnstr+integerPart+'.'+floatPart

    def __str__(self):
        """
        Convert the QFloat to a tidy string

        WARNING : does not work on encrypted QFLoat
        """
        return self.toStr(True)   


    def fromFloat(f, length=10, ints=None, base=2):
        """
        Create a QFloat from an unencrypted float

        WARNING : will return an unencrypted QFLoat
        """
        if ints is None:
            ints = length//2

        integerPart = int(f)
        floatPart = f-integerPart

        intArray = int_to_base_p(integerPart, ints, base)
        floatArray = float_to_base_p(floatPart, length-ints, base)

        array = fhe.zeros(length)
        array[0:ints] = intArray
        array[ints:] = floatArray
        sign = np.sign(f) or 1 # zero has sign 1

        return QFloat(array*sign, ints, base, True, sign)

    def toFloat(self):
        """
        Create an unencrypted float from the QFloat

        WARNING : does not work on encrypted QFLoat
        """
        self._checkUnencrypted()

        integerPart = base_p_to_int(self._array[0:self._ints], self._base)
        floatPart = base_p_to_float(self._array[self._ints:], self._base)

        return (integerPart + floatPart)*self._sign # will yield a 0 if sign is 0


    #=============================================================================================
    #                      Functions for both encrypted or unencrypted QFloats
    #
    #       Functions with (self,other) can work for mixed encrypted and unencrypted arrays
    #=============================================================================================

    def zero(length, ints, base, encrypted=True):
        """
        Create a QFloat of 0
        """
        if not (isinstance(length, int) and length > 0):
            raise ValueError('length must be a positive int')
        if encrypted:
            return QFloat(fhe.zeros(length), ints, base, True, fhe.ones(1)[0])
        else:
            return QFloat(np.zeros(length), ints, base, True, 1)

    def zero_like(other):
        """
        Create a QFloat of 0 with same shape as other
        """
        if not isinstance(other, QFloat):
            raise ValueError('Object must be a QFloat')

        return QFloat.zero(len(other), other._ints, other._base, other._encrypted)

    def one(length, ints, base, encrypted=True):
        """
        Create a QFloat of 1
        """        
        if encrypted:
            array=fhe.zeros(length)
            array[ints-1] = fhe.ones(1)[0]
            return QFloat(array, ints, base, True, fhe.ones(1)[0])
        else:
            array=np.zeros(length)
            array[ints-1] = 1
            return QFloat(array, ints, base, True, 1)

    def one_like(other):
        """
        Create a QFloat of 1 with same shape as other
        """
        if not isinstance(other, QFloat):
            raise ValueError('Object must be a QFloat')

        return QFloat.one(len(other), other._ints, other._base, other._encrypted)

    def copy(self):
        """
        Create a QFloat copy
        """
        return QFloat(self.toArray(), self._ints, self._base, self._isBaseTidy, self._sign)

    def toArray(self):
        """
        Rerturn copy of array
        """
        if not self._encrypted:
            return np.copy(self._array)
        else:
            return self._array[:]

    def set_len_ints(self, newlen, newints):
        """
        Set new length and new ints
        WARNING : This operation may troncate the int or float part
        """
        zeros = fhe.zeros if self._encrypted else lambda x: np.zeros(x, dtype='int')
        # set ints
        if self._ints != newints:
            if newints > self._ints:
                self._array = np.concatenate( (zeros(int(newints - self._ints)), self._array), axis=0)
            else:
                self._array = self._array[self._ints - newints:]
            self._ints = int(newints)

        # set length
        difflen = int(newlen - len(self))
        if difflen!=0:
            if difflen >0:
                self._array = np.concatenate((self._array, zeros(difflen)),axis=0)
            else:
                self._array = self._array[:-difflen]

    def checkCompatibility(self, other):
        """
        Check wether other has equal encoding
        """
        if not isinstance(other, QFloat):
            raise ValueError('Object must also be a ' + str(QFloat))

        if self._base != other._base:
            raise ValueError( str(QFloat)+'s bases are different')

        if len(self) != len(other):
            raise ValueError( str(QFloat)+'s have different length')

        if self._ints != other._ints:
            raise ValueError( str(QFloat)+'s have different dot index')                

    def getSign(self):
        """
        Return the sign of the QFloat
        """
        return self._sign

    def baseTidy(self):
        """
        Tidy array so that values are in range [-base, base] as they should be, but signs can be mixed
        Keeping arrays untidy when possible saves computation time
        """
        if self._isBaseTidy:
            return

        dividend = 0
        for i in reversed(range(len(self))):
            curr = self._array[i]+dividend
            dividend = (np.abs(curr) // self._base)*np.sign(curr)
            curr -= dividend*self._base
            self._array[i] = curr

        self._isBaseTidy = True

    def tidy(self):
        """
        Tidy array so that values are in range [0, base]
        """

        # first of all, make all values (negative or positive) fall under the base:
        if not self._isBaseTidy:
            self.baseTidy()

        # then, make all value the same sign. To do this, we consider that: 
        # - a Qfloat F is the sum of its positive and negative parts: F = P + N
        # - as N is negative, we also have F = P - |N|, where both P and |N| are positive,
        #   so we can use the regular subtraction algorithm
        # - if P < |N|, we can use F = -1 * (|N| - P) so the subtraction algorithm works

        P =  self._array*(self._array >= 0)
        absN = -1*(self._array*(self._array < 0))

        P_minus_absN, isNegative = base_p_subtraction_overflow(P, absN, self._base)
        isPositiveOr0 = (1-isNegative)
        self._array = isPositiveOr0*P_minus_absN + isNegative*base_p_subtraction(absN,P,self._base)

        self._sign = 2*isPositiveOr0-1

    def __len__(self):
        """
        Return length of array
        """
        return self._array.size

    def __eq__(self, other):
        """
        Return wether self is identical to other
        """
        self.checkCompatibility(other)

        if not (self._isBaseTidy and other._isBaseTidy):
            raise Exception('cannot compare QFloats that are not tidy')        

        return is_equal(self._array, other._array) & (self._sign==other._sign)

    def __lt__(self, other):
        """ Computes wether an array is lower than another

        The proposition "A < B" is equivalent to " B > A " (see self.__gt__)

        If array have different length, the shorter one is considered to have extra -np.inf values
        """
        return other > self

    def __le__(self, other):
        """ Computes wether an array is lower or equal than another

        The proposition "A <= B" is the converse of "A > B" (see self.__gt__)

        If array have different length, the shorter one is considered to have extra extra -np.inf values
        """       
        return 1-(self > other)

    def __gt__(self, other):
        """ Computes wether an array is greater than another

        An qfloat A is greater than a qfloat B if and only if:
        - they have different signs and sign(A) > sign(B)
        - or they have identical signs, are not equal, and (arr(A) > arr(B) and the signs are positive),
            else (arr(A) < arr(B) and the signs are negative)

        If array have different length, the shorter one is considered to have extra extra -np.inf values
        """
        self.checkCompatibility(other)

        self.baseTidy()
        other.baseTidy()

        sgn_eq = self._sign == other._sign

        self_gt_other = 1-is_greater_or_equal(other._array, self._array) # not b>=a <=> not not a>b  <=> a>b 
        inverse = (self._sign < 0) & (1-is_equal(self._array, other._array)) # inverse if negative sign but no equality

        return sgn_eq*(self_gt_other ^ inverse) + (1-sgn_eq)*(self._sign > other._sign)

    def __ge__(self, other):
        """ Computes wether an array is greater or equal than another, in alphabetical order

        The proposition "A >= B" is the mathematical converse of "B > A" (see self.__gt__)
        """     
        return 1-(other > self)

    def __abs__(self):
        """
        Returns the absolute value
        """
        absval = self.copy()
        absval._sign *= absval._sign # stays 0 if 0
        return absval

    def abs(self):
        """
        In place absolute value
        """
        self._sign *= self._sign # stays 0 if 0
        return self

    def __add__(self, other):
        """
        Sum with another QFLoat or single integer
        """
        addition = self.copy()
        addition += other
        return addition

    def __radd__(self, other):
        """
        Add other object on the right
        """
        return self.__add__(other)

    def checkConvertFHE(qf, condition):
        """
        If qf is not encrypted and condition is verified, convert qf array to fhe array
        """
        if (not qf._encrypted) and condition:
            fhe_array = fhe.ones(len(qf)) * qf._array
            qf._array = fhe_array
            qf._encrypted = True

    def selfCheckConvertFHE(self, condition):
        """
        If self is not encrypted and condition is verified, convert self to fhe array
        """
        QFloat.checkConvertFHE(self, condition)

    def __iadd__(self, other):
        """
        Sum with another QFLoat or single integer, in place
        """

        if isinstance(other, Zero):
            return # no change if adding a Zero
        
        QFloat.ADDITIONS+=1 # count addition in all other cases, cause we have to tidy

        # multiply array by sign first (becomes 0 if sign is 0)
        self._array *= self._sign

        if isinstance(other, Tracer) or isinstance(other, numbers.Integral):
            self.selfCheckConvertFHE(isinstance(other, Tracer))
            # Add a single integer
            self._array[self._ints-1]+=other
        elif isinstance(other, SignedBinary):
            self.selfCheckConvertFHE(other._encrypted)
            self._array[self._ints-1]+=other.value        
        else:
            self.selfCheckConvertFHE(other._encrypted)

            self.checkCompatibility(other)
            self._array = self._array+ other._array*other._sign

        self._isBaseTidy=False
        self._sign=None

        # tidy the qflloat to make the integers positive and know the sign
        self.tidy()

        return self

    def __sub__(self, other):
        """
        Subtract another QFLoat
        """
        res = -other
        res += self
        return res

    def __rsub__(self, other):
        """
        Add other object on the right
        """
        res = -self
        res += other
        return res

    def __imul__(self, other):
        """
        Multiply with another QFLoat or integer, in place

        WARNING: precision of multiplication does not increase, so it may overflow if not enough
        """
        if isinstance(other, Tracer) or isinstance(other, numbers.Integral):
            self.selfCheckConvertFHE(isinstance(other, Tracer))
            # multiply everything by a single integer
            sign=np.sign(other)
            self._array *= (other*sign)            
            self._sign *= sign
            self._isBaseTidy=False
            self.baseTidy()

        elif isinstance(other, SignedBinary):
            self.selfCheckConvertFHE(other._encrypted)
            # multiplying by a binary value is the same as multiplying the sign, aka the value
            # if the vale is zero, the sign becomes zero which will yield a zero QFloat
            self._sign *= other.value
        else:
            QFloat.MULTIPLICATION+=1 # count only multiplications with other Qfloat
            # multiply with another compatible QFloat 

            # always base tidy before a multiplication between to QFloats prevent multiplying big bitwidths
            self.baseTidy()
            other.baseTidy()

            self.selfCheckConvertFHE(other._encrypted)
            self.checkCompatibility(other)

            # A QFloat array is made of 2 parts, integer part and float part
            # The multiplication array will be the sum of integer * other + float * other
            n=len(self)
            mularray = fhe.zeros((n, n))
            # integer part, shift  to the left
            for i in range(0,self._ints):
                mularray[i, 0:n-(self._ints-1-i)] = self._array[i]*other._array[self._ints-1-i:]
            # float part, shift to the right
            for i in range(self._ints,n):
                mularray[i, 1+i-self._ints:] = self._array[i]*other._array[0:n-(i-self._ints)-1]

            # the multiplication array is made from the sum of the muarray rows
            self._array = np.sum(mularray, axis=0)

            self._sign = self._sign*other._sign

            self._isBaseTidy=False

            # base tidy to keep bitwidth low
            self.baseTidy()

        return self

    def __mul__(self, other):
        """
        Multiply with another QFLoat or number, see __imul__
        """
        # special case when multiplying by unencrypted 0, the result is an unencrypted 0
        if isinstance(other, Zero):
            return Zero()

        multiplication = self.copy()
        if isinstance(other, Tracer) or (isinstance(other, SignedBinary) and other._encrypted):
            multiplication.selfCheckConvertFHE(True) # convert to encrypted array if needed

        multiplication *= other

        return multiplication

    def __rmul__(self, other):
        """
        Multiply with other object on the right
        """
        return self.__mul__(other)

    def __neg__(self):
        """
        Negative
        """    
        neg=self.copy()
        neg._sign *= -1
        return neg

    def neg(self):
        """
        In place negative
        Use it for faster subtraction if you don't need to keep the second QFloat: a - b = a + b.neg()
        """    
        self._sign *= -1
        return self

    def fromMul(a, b, newlength=None, newints=None):
        """
        Compute the multiplication of QFloats a and b, inside a new QFloat of given length and ints
        Warning: if the new length and ints are too low, the result might be cropped
        """
        if newlength is None:
            newlength = len(a) + len(b)
        if newints is None:
            newints = a._ints + b._ints

        # special case when multiplying by unencrypted 0, the result is an unencrypted 0
        if isinstance(a, Zero) or isinstance(b, Zero):
            return Zero()

        elif isinstance(a, SignedBinary) or isinstance(b, SignedBinary):
            if (isinstance(a, SignedBinary) and isinstance(b, SignedBinary)):
                return a*b
            # simple multiplication and size setting        
            multiplication = a*b
            multiplication.set_len_ints(newlength, newints)

        else:
            QFloat.MULTIPLICATION+=1 # count only multiplications with other Qfloat
            
            # always base tidy before a multiplication between to QFloats prevent multiplying big bitwidths
            a.baseTidy()
            b.baseTidy()

            # convert a to encrypted if needed
            QFloat.checkConvertFHE(a, b._encrypted)

            # check base compatibility
            if not a._base == b._base:
                raise ValueError('bases are different')

            # A QFloat array is made of 2 parts, integer part and float part
            # The multiplication array will be the sum of a.integer * b + a.float * b
            mularray = fhe.zeros((len(a), newlength))
            for i in range(0,len(a)):
                # index from b where the array mul should be inserted in mularray 
                indb = newints-a._ints+i+1-b._ints
                # compute only needed multiplication of b._array*a._array[i], accounting for crops
                ind1 = 0 if indb >=0 else -indb
                ind2 = min(len(b),newlength-indb)
                if ind2>ind1:
                    mul = b._array[ind1:ind2]*a._array[i]
                    if ind2-ind1==1:
                        mul = mul.reshape(1)
                    insert_array_at_index(mul, mularray, i, indb+ind1)

            # the multiplication array is made from the sum of the muarray rows with product of signs
            multiplication = QFloat(np.sum(mularray, axis=0), newints, a._base, False, a._sign*b._sign)

            # base tidy to keep bitwidth low
            multiplication.baseTidy()    

        return multiplication             

    def __itruediv__(self, other):
        """
        Divide by another QFLoat, in place
        Dividing requires arrays to be tidy and will return a tidy array

        Consider two integers a and b, that we want to divide with float precision fp:
        We have: (a / b) = (a * fp) / b / fp
        Where a * fp / b is an integer division, and ((a * fp) / b / fp) is a float number with fp precision

        WARNING: dividing by zero will give zero
        WARNING: precision of division does not increase
        """
        if isinstance(other, Zero):
            raise Exception('division by Zero')

        if isinstance(other, SignedBinary):
            self.selfCheckConvertFHE(other._encrypted)

            # In this case, the array is either unchanged (just signed), or overflowed (dividing by 0 causes overflow)
            is_zero = other.value==0
            sign = other.value # the value is also its sign
            self._array = (1-is_zero)*self._array + is_zero*fhe.ones(len(self))*(self._base-1)
            self._sign = (1-is_zero)*sign + is_zero*self._sign
            return self

        other.baseTidy()

        QFloat.DIVISION+=1 # count only divisions with other Qfloat
        self.checkCompatibility(other)
        self.baseTidy()

        # The float precision is the number of digits after the dot:
        fp = len(self)-self._ints

        # We consider each array as representing integers a and b here
        # Let's left shit the first array which corresponds by multiplying a by 2^fp:
        shift_arr = np.concatenate((self._array, fhe.zeros(fp)), axis=0)
        # Make the integer division (a*fp)/b with our long division algorithm:
        div_array = base_p_division(shift_arr, other._array, self._base)
        # The result array encodes for a QFloat with fp precision, which is equivalent to divide the result by fp
        # giving as expected the number (a * fp) / b / fp :
        self._sign = self.getSign()*other.getSign()
        self._array = div_array[fp:]

        return self

    def __truediv__(self, other):
        """
        Divide by another QFLoat, see __itruediv__
        """
        division = self.copy()
        division /= other
        return division

    def __rtruediv__(self, other):
        """
        Return other divided by self
        """
        if isinstance(other, Zero):
            # special case when other is unencrypted 0, the result is an unencrypted 0
            return Zero()
        
        elif isinstance(other, Tracer) or isinstance(other, numbers.Integral):
            # create a QFloat if other is a number:
            qf = QFloat.one(len(self), self._ints, self._base, encrypted=isinstance(other, Tracer))
            sign = np.sign(other)
            qf._array[self._ints-1]*=(other*sign)
            qf._sign *= sign
            qf.baseTidy()
            return qf / self

        elif isinstance(other, QFloat):
            return other / self

        elif isinstance(other, SignedBinary):
            # get sign, then invert ith this sign (can be 0)
            return self.invert(other.value, len(self), self._ints) # the value is also its sign
            
        else:
            raise ValueError('Unknown class for other')

    def invert(self, sign=1, newlength=None, newints=None):
        """
        Compute the signed invert of the QFloat, with different length and ints values if requested
        """
        if not (isinstance(sign, SignedBinary) or (isinstance(sign,numbers.Integral) and abs(sign)==1) ):
            raise ValueError("sign must be a SignedBinary or a signed binary scalar")

        QFloat.DIVISION+=1 # this division is counted because it is heavy

        # tidy before dividing
        self.baseTidy()

        if newlength is None:
            newlength = len(self)
        if newints is None:
            newints = self._ints

        a = fhe.ones(1) # an array with one value to save computations
        b = self._array 

        # The float precision is the number of digits after the dot:
        fp = newlength-newints # new float precision
        fpself = len(self)-self._ints # self float precision

        # We consider each array as representing integers a and b here
        # Let's left shit the first array which corresponds by multiplying a by 2^(fpself + fp) (decimals of old+new precision):
        shift_arr = np.concatenate((a, fhe.zeros(fpself+fp)), axis=0)
        # Make the integer division (a*fp)/b with our long division algorithm:
        div_array = base_p_division(shift_arr, b, self._base)

        # Correct size of div_array
        diff=newlength-div_array.size
        if diff>0:
            div_array = np.concatenate( (fhe.zeros(diff), div_array), axis=0)
        else:
            div_array = div_array[-diff:]
        # The result array encodes for a QFloat with fp precision, which is equivalent to divide the result by fp
        # giving as expected the number (a * fp) / b / fp :
        newsign=sign*self.getSign()
        invert_div = QFloat(div_array, newints, self._base, True, newsign)

        return invert_div
