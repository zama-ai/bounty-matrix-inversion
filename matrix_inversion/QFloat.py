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
    # Convert the integer to a base p string
    base_p_string= np.base_repr(np.abs(integer), p)
    # Prepend zeros to the binary representation until it reaches the desired size
    base_p_string = base_p_string.zfill(n)
    if len(base_p_string) > n:
        raise Exception(f"integer: {integer} cannot be rerpresented with {n} values in base {p}")
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


def insert_array_at_index(a, b, i):
    # Case when i is negative
    if i < 0:
        # We will take elements of a starting at index -i
        a = a[-i:]
        i = 0
    # Compute the number of elements we can insert from a into b
    n = min(len(b) - i, len(a))
    # Insert elements from a into b
    b[i:i+n] = a[:n]


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
    """

    # keepTidy: wether to keep arrays tidy at all time. Setting to False can save time in FHE.
    # Use it with caution because it can increase bitwidth too much in FHE:
    # to prevent this, you may need to manually tidy some QFloats in your algorithm to prevent
    KEEP_TIDY=True

    # Statistics for counting operations between QFloats within a circuit
    ADDITIONS=0
    SUBTRACTION=0
    MULTIPLICATION=0
    DIVISION=0

    def __init__(self, array, ints=None, base=2, isTidy=True, sign=None):
        """
        - array: an encrypted or unencrypted array representing a number in base p (little endian)
        - ints: gives the number of digits before the dot, so ints = 1 will encode a number like x.xxxx...
        - isTidy: wether the array is already tidy
        - sign: provide the sign if it is already known (usefull in FHE to save computations)
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
        if not ints:
            ints = array.size//2
        else:
            if not (isinstance(ints, int) and 0<ints and array.size>=ints):
                raise ValueError('ints must be in range [1,array.size]')
        self._ints = ints
        self._isTidy = isTidy # wether array is tidy (with mixed signs or with abs values >= base)
        self._isBaseTidy = isTidy # wether array is tidy for values (abs values >= base) but signs can be mixed
        self._sign = sign # sign if known

        if not isTidy and QFloat.KEEP_TIDY:
            self.tidy()

    def resetStats():
        QFloat.ADDITIONS=0
        QFloat.SUBTRACTION=0
        QFloat.MULTIPLICATION=0
        QFloat.DIVISION=0

    def showStats():
        print('\nQFloat statistics :')
        print('======================')
        print('Additions       : ' +  str(QFloat.ADDITIONS))
        print('Subtractions    : ' +  str(QFloat.SUBTRACTION))
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
            self.tidy()
            sgn = self.getSign()

            integerPart = (sgn*self._array[0:self._ints]).astype('int')
            floatPart = (sgn*self._array[self._ints:]).astype('int')

            if self._base <= 10:
                integerPart = ''.join([str(i) for i in integerPart])
                floatPart = ''.join([str(i) for i in floatPart])
            else:
                integerPart = str(integerPart)
                floatPart = str(floatPart)

            sgnstr = '' if sgn >=0 else '-'

            return sgnstr+integerPart+'.'+floatPart

        else: # return the untidy representation

            integerPart = self._array[0:self._ints].astype('int')
            floatPart = self._array[self._ints:].astype('int')

            integerPart = ''.join([str(i) for i in integerPart])
            floatPart = ''.join([str(i) for i in floatPart])

            return integerPart+'.'+floatPart

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
        if not ints:
            ints = length//2

        integerPart = int(f)
        floatPart = f-integerPart

        intArray = int_to_base_p(integerPart, ints, base)
        floatArray = float_to_base_p(floatPart, length-ints, base)

        array = fhe.zeros(length)
        array[0:ints] = intArray
        array[ints:] = floatArray
        sign = np.sign(f) or 1 # zero has sign 1

        return QFloat(array, ints, base, True, sign)

    def toFloat(self):
        """
        Create an unencrypted float from the QFloat

        WARNING : does not work on encrypted QFLoat
        """
        self._checkUnencrypted()

        integerPart = base_p_to_int(self._array[0:self._ints], self._base)
        floatPart = base_p_to_float(self._array[self._ints:], self._base)

        return integerPart + floatPart


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
        copy = QFloat(self.toArray(), self._ints, self._base, self._isTidy, self._sign)
        copy._isBaseTidy = self._isBaseTidy
        return copy       

    def toArray(self):
        """
        Rerturn copy of array
        """
        if not self._encrypted:
            return np.copy(self._array)
        else:
            return self._array[:]

    def set_len_ints(newlen, newints):
        """
        Set new length and new ints
        WARNING : This operation may troncate the int or float part
        """
        # set ints
        if newints > self.ints:
            lib = fhe if self._encrypted else np
            self._array = np.concatenate( lib.zeros(newints - self.ints), self._array)
        else:
            self._array = self._array[self.ints - newints:]
        self._ints = newints

        # set length
        difflen = newlen - len(self)
        if diff >0:
            lib = fhe if self._encrypted else np
            self._array = np.concatenate(self._array, lib.zeros(diff))
        else:
            self._array = self._array[:-diff]

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
        WARNING: 0 is considered positive for faster computations
        but it can happen than a sign is set to zero in other function (see __mul__)
        """
        if self._sign is None:
            # base tidy so that sign can be computed from the first non zero number:
            self.baseTidy()

            # most efficient way to compute the sign of the first non zero number:
            borrow = 0
            for i in range(self._array.size):
                borrow = self._array[-i-1] - borrow < 0

            self._sign = 2*(1-borrow)-1

        return self._sign

    def baseTidy(self):
        """
        Tidy array so that values are in range [-base, base] as they should be, but signs can be mixed
        Keeping arrays untidy when possible saves computation time
        """
        if self._isBaseTidy:
            return

        dividend = fhe.zeros(1)[0]
        for i in reversed(range(len(self))):
            curr = self._array[i]+dividend
            dividend = (np.abs(curr) // self._base)*np.sign(curr)
            curr -= dividend*self._base
            self._array[i] = curr

        self._isBaseTidy = True

    def tidy(self):
        """
        Tidy array so that values are in range [-base, base] and signs are all the same
        This gives the standard reprensentation of the number
        Keeping arrays untidy when possible saves computation time
        """
        if self._isTidy:
            return

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
        self._array = isPositiveOr0*P_minus_absN - isNegative*base_p_subtraction(absN,P,self._base)

        self._sign = 2*isPositiveOr0-1
        self._isTidy=True

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

        if not (self._isTidy and other._isTidy):
            raise Exception('cannot compare QFloats that are not tidy')        

        return (len(self) - np.sum(self._array == other._array))==0

    def __lt__(self, other):
        """ Computes wether an array is lower than another

        An array A is lower than a array B if and only if:
        there exist an index i such that: A[i] < B[i]  AND  for all k<i, A[k] <= B[k]

        This is the mathematical converse of A >= B (see self.__ge__) which is easier to do in fhe

        If array have different length, the shorter one is considered to have extra -np.inf values
        """
        return 1-(self >= other)

    def __le__(self, other):
        """ Computes wether an array is lower or equal than another

        The proposition "A <= B" is equivalent to "B >= A" (see self.__ge__)

        If array have different length, the shorter one is considered to have extra extra -np.inf values
        """       
        return other >= self

    def __gt__(self, other):
        """ Computes wether an array is greater than another

        The proposition "A > B" is the mathematical converse of "B >= A" (see self.__ge__)

        If array have different length, the shorter one is considered to have extra extra -np.inf values
        """
        return 1-(other >= self)

    def __ge__(self, other):
        """ Computes wether an array is greater or equal than another, in alphabetical order

        An array A is greater or equal than an array B if and only if:
        for all index i:  either  A[i] >= B[i]  or  there is an index k<i where A[k] > B[k]
        """     
        self.checkCompatibility(other)

        self.tidy()
        other.tidy()

        return is_greater_or_equal(self._array, other._array)

    def __abs__(self):
        """
        Returns the absolute value
        """
        absval = self.copy()
        absval.tidy() # need to tidy before computing abs
        absval._array = np.abs(absval._array)
        absval._sign = fhe.ones(1)[0]
        return absval

    def __add__(self, other):
        """
        Sum with another QFLoat or single integer
        Summing will potentially make values in the sum array be greater than the base and not tidy, so isTidy becomes False
        Hence we need to tidy the sum if requested
        """
        if isinstance(other, Tracer) or isinstance(other, numbers.Integral):
            # Add a single integer
            addition = QFloat( self._array, self._ints, self._base, False)
            addition._array[self._ints-1]+=other
        elif isinstance(other, SignedBinary):
            addition = QFloat( self._array, self._ints, self._base, False)
            addition._array[self._ints-1]+=other.value
        else:
            QFloat.ADDITIONS+=1 # count only addition with other Qfloat
            self.checkCompatibility(other)
            addition = QFloat(self._array + other._array, self._ints, self._base, False)

        if QFloat.KEEP_TIDY:
            addition.tidy()

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
        if not qf._encrypted and condition:
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
        Summing will potentially make values in the sum array be greater than the base and not tidy, so isTidy becomes False
        Hence we need to tidy if requested
        """
        if isinstance(other, Tracer) or isinstance(other, numbers.Integral):
            self.selfCheckConvertFHE(isinstance(other, Tracer))
            # Add a single integer
            self._array[self._ints-1]+=other
        elif isinstance(other, SignedBinary):
            self.selfCheckConvertFHE(other._encrypted)
            self._array[self._ints-1]+=other.value        
        else:
            QFloat.ADDITIONS+=1 # count only addition with other Qfloat
            self.selfCheckConvertFHE(other._encrypted)

            self.checkCompatibility(other)
            self._array += other._array

        self._isTidy=False
        self._isBaseTidy=False
        self._sign=None

        if QFloat.KEEP_TIDY:
            self.tidy()

        return self

    def __sub__(self, other):
        """
        Subtract another QFLoat
        Subtracting will potentially make values in the sum array be greater than the base and not tidy, so isTidy becomes False
        Hence we need to tidy the subtraction if requested
        """
        if isinstance(other, Tracer) or isinstance(other, numbers.Integral):
            subtraction = QFloat( self._array, self._ints, self._base, False)
            subtraction._array[self._ints-1]-=other
        elif isinstance(other, SignedBinary):
            subtraction = QFloat( self._array, self._ints, self._base, False)
            subtraction._array[self._ints-1]-=other.value
        else:
            QFloat.SUBTRACTION+=1 # count only subtractions with other Qfloat
            self.checkCompatibility(other)
            subtraction = QFloat(self._array - other._array, self._ints, self._base, False)
        
        if QFloat.KEEP_TIDY:
            subtraction.tidy()

        return subtraction

    def __rsub__(self, other):
        """
        Add other object on the right
        """
        return -self + other

    def __imul__(self, other):
        """
        Multiply with another QFLoat or integer, in place
        Multiplying will potentially make values in the sum array be greater than the base and not tidy, so isTidy becomes False
        Hence we need to tidy if requested

        WARNING: precision of multiplication does not increase, so it may overflow if not enough
        """
        if isinstance(other, Tracer) or isinstance(other, numbers.Integral):
            self.selfCheckConvertFHE(isinstance(other, Tracer))
            # multiply everything by a single integer
            self._array *= other
            self._isTidy=False
            self._isBaseTidy=False
            if self._sign is not None:
                self._sign = self._sign*np.sign(other)
        elif isinstance(other, SignedBinary):
            self.selfCheckConvertFHE(other._encrypted)
            # multiply everything by a binary value, which keeps the array tidy
            self._array *= other.value
            #self._isTidy and self._isBaseTidy are not impacted here
            if self._sign is not None:
                self._sign = self._sign*np.sign(other.value)            
        else:
            QFloat.MULTIPLICATION+=1 # count only multiplications with other Qfloat
            # multiply with another compatible QFloat 
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

            if self._sign is not None and other._sign is not None: # avoid computing sign of the sign if we already know it
                self._sign = self._sign*other._sign
            else:
                self._sign=None

            self._isTidy=False
            self._isBaseTidy=False

        if QFloat.KEEP_TIDY:
            self.tidy()

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
        return self*SignedBinary(-1)

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
            
            # convert a to encrypted if needed
            QFloat.checkConvertFHE(a, b._encrypted)

            # check base compatibility
            if not a._base == b._base:
                raise ValueError('bases are different')

            # crop b to save computations if necessary:
            diffints = newints - b._ints
            diffdec = (newlength - newints) - (len(b) - b._ints)
            if diffints < 0 or diffdec < 0:
                b=b.copy()

                if diffints < 0:
                    b._array = b._array[-diffints:]
                    b._ints = newints

                if diffdec < 0:
                    b._array = b._array[:-diffdec]

            # A QFloat array is made of 2 parts, integer part and float part
            # The multiplication array will be the sum of a.integer * b + a.float * b
            mularray = fhe.zeros((len(a), newlength))
            # integer part, shift  to the left
            for i in range(0,len(a)):
                mul = b._array*a._array[i]
                # index from b where the array mul should be inserted in mularray
                indb = newints-a._ints+i+1-b._ints if i<a._ints else newints-a._ints+i+1-b._ints
                insert_array_at_index(mul, mularray[i], indb)

            # the multiplication array is made from the sum of the muarray rows
            multiplication = QFloat(np.sum(mularray, axis=0), newints, a._base, False)

            if a._sign is not None and b._sign is not None: # avoid computing sign of the sign if we already know it
                multiplication._sign = a._sign*b._sign
            else:
                multiplication._sign=None

            if QFloat.KEEP_TIDY:
                multiplication.tidy()

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
            self._array = (1-is_zero)*sign*self._array + is_zero*fhe.ones(len(self))*(self._base-1)
            self._sign = sign
            return self

        elif isinstance(other, Tracer) or isinstance(other, numbers.Integral):
            # create a QFloat from this number
            qf = QFloat.one(len(self), self._ints, self._base, encrypted=isinstance(other, Tracer))
            qf._array[self._ints-1]*=other
            qf.tidy()
            other=qf
        else:
            other.tidy()

        QFloat.DIVISION+=1 # count only divisions with other Qfloat
        self.checkCompatibility(other)
        self.tidy()

        # get signs and make arrays positive
        signa = self.getSign()
        a = signa*(self._array)

        signb = other.getSign()
        b = signb*(other._array)        

        # The float precision is the number of digits after the dot:
        fp = len(self)-self._ints

        # We consider each array as representing integers a and b here
        # Let's left shit the first array which corresponds by multiplying a by 2^fp:
        shift_arr = np.concatenate((a, fhe.zeros(fp)), axis=0)
        # Make the integer division (a*fp)/b with our long division algorithm:
        div_array = base_p_division(shift_arr, b, self._base)
        # The result array encodes for a QFloat with fp precision, which is equivalent to divide the result by fp
        # giving as expected the number (a * fp) / b / fp :
        self._sign = signa*signb
        self._array = div_array[fp:]*self._sign  # result is tidy and signed

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
            qf._array[self._ints-1]*=other
            qf._sign *= np.sign(other)
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
        QFloat.DIVISION+=1 # this division is counted because it is heavy

        # tidy before dividing
        self.tidy()

        if not newlength:
            newlength = len(self)
        if not newints:
            newints = self._ints

        # get signs of dividend and divisor
        signa = sign
        a = fhe.ones(1) # an array with one value to save computations

        signb = self.getSign()
        b = signb*(self._array) # make array positive for division

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
        newsign=signa*signb
        invert_div = QFloat(newsign*div_array, newints, self._base, True, newsign)

        return invert_div            
