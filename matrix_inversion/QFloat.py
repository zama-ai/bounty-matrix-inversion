import numpy as np
import numbers
from concrete import fhe


#=======================================================================================================================
#                                    Functions operating on arrays in base p
#=======================================================================================================================

def base_p_to_int(arr, p):
    """
    Convert base p array to int
    Can be signed (+/- values)
    """
    # Flip the array to have least significant bit at the start
    arr = np.flip(arr)
    # Compute the place values (p^0, p^1, p^2, ...)
    place_values = p**np.arange(arr.size)
    # Return the sum of bit*place_value
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
    # Convert the base_p string to a NumPy array of integers
    base_p_array = np.sign(integer)*np.array([int(bit) for bit in base_p_string])
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

def base_p_subtraction(a, b, p):
    """
    Subtract arrays in base p. (a and b must be tidy, postive with a > b)
    """
    difference = fhe.zeros(a.size)
    borrow = 0
    for i in reversed(range(len(a))):
        # Perform subtraction for each bit
        temp = a[i] - b[i] - borrow
        borrow = temp < 0
        temp += p*borrow
        difference[i] = temp
    return difference

def base_p_division(dividend, divisor, p):
    """
    Divide arrays in base p. (dividend and divisor must be tidy and positive)
    """    
    # Initialize the quotient array
    quotient = fhe.zeros(dividend.size)
    # Initialize the remainder
    remainder = fhe.zeros(divisor.size)

    for i in range(len(dividend)):
        # Left-roll the remainder
        remainder = np.concatenate((remainder[1:], remainder[0].reshape(1)), axis=0)
        # Bring down the next bit from the dividend
        remainder[-1] = dividend[i]
        # If the remainder is larger than or equal to the divisor
        for j in range(p-1):
            if is_greater_or_equal(remainder, divisor):
                # Subtract the divisor from the remainder
                remainder = base_p_subtraction(remainder, divisor, p)
                # Set the current quotient bit to 1
                quotient[i] += 1

    return quotient

def is_greater_or_equal(A, B):
    """
    Computes wether an array is greater or equal than another, in alphabetical order

    An array A is greater or equal than an array B if and only if:
    for all index i:  either  A[i] >= B[i]  or  there is an index k<i where A[k] > B[k]
    """    
    n = A.size

    ## compute A[i] >= B[i] array
    ge_array = A >= B
    
    ## compute A[i] > B[i] array
    gt_array = A > B 

    ## compute "there is an index k<i where A[i] > B[i]" array in cum_gt_array:
    # if gt_array[k] is 1 for some k, cum_gt_array[i] will be 1 for all i>k
    cum_gt_array = fhe.zeros(n)
    if n >1:
        cum_gt_array[1] = gt_array[0]
    else:
        return ge_array[0] # special case if array has size one

    for i in range(2,n):
        cum_gt_array[i] = cum_gt_array[i-1] | gt_array[i-1]

    ## now compute " A[i] >= B[i]  or  there is an index k<i where A[i] > B[i] " array in or_array:
    or_array = ge_array | cum_gt_array

    ## return wether or_array is true for all indices
    return (n - np.sum(or_array))==0



#=======================================================================================================================
#                                                       QFfloats
#=======================================================================================================================

class QFloat():
    """
    A class for quantizing floats
    """

    # keepTidy: wether to keep arrays tidy at all time. Setting to False can save time
    KEEP_TIDY=True

    def __init__(self, array, ints=None, base=2, isTidy=True, sign=None):
        """
        - array can be encrypted or not
        - ints gives the number of digits before the dot, so ints = 1 will encode a number like x.xxxx...
        - isTidy: wether the array is already tidy
        - sign: provide the sign if it is already known (usefull in FHE to save computations)
        """
        if not (isinstance(array, np.ndarray) or isinstance(array, fhe.tracing.tracer.Tracer)):
            raise ValueError('array must be np.ndarray or fhe.tracing.tracer.Tracer')

        self._encrypted = isinstance(array, fhe.tracing.tracer.Tracer)

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

        return QFloat(array, ints, base, True, np.sign(f))

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
    #                                 Functions for both encrypted or unencrypted QFloats
    #
    #       Function with (self,other) can work for mixed encrypted and unencrypted arrays
    #=============================================================================================

    def zero(length, ints, base):
        """
        Create a QFloat of 0
        """
        if not (isinstance(length, int) and length > 0):
            raise ValueError('length must be a positive int')
        return QFloat(fhe.zeros(length), ints, base, True, fhe.zeros(1)[0])

    def zero_like(other):
        """
        Create a QFloat of 0 with same shape as other
        """
        if not isinstance(other, QFloat):
            raise ValueError('Object must be a QFloat')

        return QFloat.zero(len(other), other._ints, other._base, )

    def one(length, ints, base):
        """
        Create a QFloat of 1
        """        
        array=fhe.zeros(length)
        array[ints-1] = fhe.ones(1)[0]
        return QFloat(array, ints, base, True, fhe.ones(1)[0])

    def one_like(other):
        """
        Create a QFloat of 1 with same shape as other
        """
        if not isinstance(other, QFloat):
            raise ValueError('Object must be a QFloat')

        return QFloat.one(len(other), other._ints, other._base)

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
        Note: 0 is considered positive for faster computations

        When the QFloat is base tidy, its sign is the the same as the sign of first non zero integer
        in its array, or 0 if the array is null
        """
        if self._sign is not None:
            return self._sign

        self.baseTidy()

        # compute array signs and zeros
        signs = np.sign(self._array)
        is_zero = signs==0
        is_not_zero = signs!=0

        sign = signs[0]
        notfound = is_zero[0] 
        for i in range(1, signs.size):
            change = is_not_zero[i] & notfound
            not_change = (1-change)
            sign = change*signs[i] + not_change*sign
            notfound = notfound & not_change

        self._sign = sign # avoid computing is several times
        return sign

    def baseTidy(self):
        """
        Tidy array so that values are in range [-base, base] as they should be, but signs can be mixed
        Keeping arrays untidy when possible saves computation time
        """
        if self._isBaseTidy:
            return

        dividend = fhe.zeros(1)[0]
        for i in reversed(range(len(self))):
            self._array[i] += dividend
            dividend = (np.abs(self._array[i]) // self._base)*np.sign(self._array[i])
            self._array[i] -= dividend*self._base

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

        isPositive = self.getSign() >=0
        self._array = isPositive*base_p_subtraction(P,absN,self._base) - (1-isPositive)*base_p_subtraction(absN,P,self._base)

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
        if isinstance(other, fhe.tracing.tracer.Tracer) or isinstance(other, numbers.Integral):
            # Add a single integer
            addition = QFloat( self._array, self._ints, self._base, False)
            addition._array[self._int-1]+=other
        else:
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

    def __iadd__(self, other):
        """
        Sum with another QFLoat or single integer, in place
        Summing will potentially make values in the sum array be greater than the base and not tidy, so isTidy becomes False
        Hence we need to tidy if requested
        """
        if isinstance(other, fhe.tracing.tracer.Tracer) or isinstance(other, numbers.Integral):
            # Add a single integer
            self._array[self._int-1]+=other
        else:
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
        self.checkCompatibility(other)
        subtraction = QFloat(self._array - other._array, self._ints, self._base, False)
        if QFloat.KEEP_TIDY:
            subtraction.tidy()
        return subtraction

    def __mul__(self, other):
        """
        Multiply with another QFLoat or integer
        Multiplying will potentially make values in the sum array be greater than the base and not tidy, so isTidy becomes False
        Hence we need to tidy the multiplication if requested

        WARNING: precision of multiplication does not increase, so it may overflow if not enough
        """

        if isinstance(other, fhe.tracing.tracer.Tracer) or isinstance(other, numbers.Integral):
            # multiply everything by a single integer
            multiplication = QFloat( other*self._array, self._ints, self._base, False)        
        else:
            # multiply with another compatible QFloat 
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
            multiplication = QFloat(np.sum(mularray, axis=0), self._ints, self._base, False)

            if self._sign is not None and other._sign is not None: # avoid computing sign of the sign if we already know it
                multiplication._sign = self._sign*other._sign

        if QFloat.KEEP_TIDY:
            multiplication.tidy()
        return multiplication

    def __rmul__(self, other):
        """
        Multiply with other object on the right
        """
        return self.__mul__(other)

    def __imul__(self, other):
        """
        Multiply with another QFLoat or integer, in place
        Multiplying will potentially make values in the sum array be greater than the base and not tidy, so isTidy becomes False
        Hence we need to tidy if requested

        WARNING: precision of multiplication does not increase, so it may overflow if not enough
        """

        if isinstance(other, fhe.tracing.tracer.Tracer) or isinstance(other, numbers.Integral):
            # multiply everything by a single integer
            self._array *= other
        else:
            # multiply with another compatible QFloat 
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

            if self._sign and other._sign: # avoid computing sign of the sign if we already know it
                self._sign = self._sign*other._sign
            else:
                self._sign=None

        self._isTidy=False
        self._isBaseTidy=False

        if QFloat.KEEP_TIDY:
            self.tidy()

        return self

    def __truediv__(self, other):
        """
        Divide by another QFLoat
        Dividing requires arrays to be tidy and will return a tidy array

        Consider two integers a and b, that we want to divide with float precision fp:
        We have: (a / b) = (a * fp) / b / fp
        Where a * fp / b is an integer division, and ((a * fp) / b / fp) is a float number with fp precision

        WARNING: dividing by zero will give zero
        WARNING: precision of division does not increase
        """
        self.checkCompatibility(other)
        self.tidy()
        other.tidy()

        # get signs and make arrays positive
        signa = self.getSign()
        a = signa*(self._array)

        signb = other.getSign()
        b = signb*(other._array)        

        # The float precision is the number of digits after the dot:
        fp = len(self)-self._ints

        # We consider each array as representing integers a and b here
        # Let's left shit the first array which corresponds by multiplying a by fp:
        shift_arr = np.concatenate((a, fhe.zeros(fp)), axis=0)
        # Make the integer division (a*fp)/b with our long division algorithm:
        div_array = base_p_division(shift_arr, b, self._base)
        # The result array encodes for a QFloat with fp precision, which is equivalent to divide the result by fp
        # giving as expected the number (a * fp) / b / fp :
        division = QFloat(div_array[fp:]*signa*signb, self._ints, self._base, True, signa*signb) # result is tidy and signed

        return division


# class PreciseQFloat()

#     def __mul__(self, other):
#         #return a result with bigger length thatn self and other to preven overflow

