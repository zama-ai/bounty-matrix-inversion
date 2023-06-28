import numpy as np
from concrete import fhe


def base_p_to_int(arr, p):
    """
    Convert base p array to int
    """
    # Flip the array to have least significant bit at the start
    arr = np.flip(arr)
    # Compute the place values (p^0, p^1, p^2, ...)
    place_values = p**np.arange(arr.size)
    # Return the sum of bit*place_value
    return np.sum(arr * place_values)

def int_to_base_p(integer, n, p):
    # Convert the integer to a base p string
    base_p_string= np.base_repr(np.abs(integer), p)
    # Prepend zeros to the binary representation until it reaches the desired size
    base_p_string = base_p_string.zfill(n)
    # Convert the base_p string to a NumPy array of integers
    base_p_array = np.sign(integer)*np.array([int(bit) for bit in base_p_string])
    return base_p_array

def base_p_to_float(arr, p):
    """Convert a base-p array to a float."""
    f = 0.0
    for i in range(len(arr)):
        f += arr[i] * (p ** -(i+1))
    return f

def float_to_base_p(f, precision, p):
    """
    Convert a float to a base-p array with a given precision.
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




class QFloat():
    """
    A class of quantized floats
    """

    def __init__(self, array, ints=None, base=2):
        """
        ints gives the number of digits before the dot.
        ints = 1 will encode number like x.xxxx...
        """
        # if not lib in [fhe, np]:
        #   raise ValueError('lib must be numpy or concrete.fhe')
        # self._lib = lib

        # if lib is fhe and not array.isinstance(fhe.tracing.tracer.Tracer):
        #   raise ValueError('array must be fhe.tracing.tracer.Tracer')
        # elif lib is np and not array.isinstance(np.ndarray):
        #   raise ValueError('array must be fhe.tracing.tracer.Tracer')

        if not (isinstance(array, np.ndarray) or isinstance(array, fhe.tracing.tracer.Tracer)):
            raise ValueError('array must be np.ndarray or fhe.tracing.tracer.Tracer')

        if len(array.shape) > 1:
            raise ValueError('array must be one dimension')
        self._array = array[:]

        if not (isinstance(base, int) and base > 1):
            raise ValueError('base must be a int >1')
        self._base = base       
        if not ints:
            ints = array.size//2
        else:
            if not (isinstance(ints, int) and 0<ints and array.size>=ints):
                raise ValueError('ints must be in range [1,array.size]')
        self._ints = ints
        self._isTidy = True # keep track of tidyness or the array
        self._QFZero = None # keep a QFloat of zero for getting sign

    def __str__(self):
        """
        Convert the QFloat to a string

        WARNING : does not work in FHE
        """
        self.tidy() # tidy array before converting it to string

        sgn = self.getSign()

        integerPart = sgn*self._array[0:self._ints]
        floatPart = sgn*self._array[self._ints:]

        if self._base <= 10:
            integerPart = ''.join([str(i) for i in integerPart])
            floatPart = ''.join([str(i) for i in floatPart])
        else:
            integerPart = str(integerPart)
            floatPart = str(floatPart)

        sgnstr = '' if sgn else '-'

        return sgnstr+integerPart+'.'+floatPart

    def fromFloat(f, length=10, ints=None, base=2):
        """
        Create a QFloat from a float

        WARNING : does not work in FHE
        """
        qf = QFloat.zero(length, ints, base)

        integerPart = int(f)
        floatPart = f-integerPart

        intArray = int_to_base_p(integerPart, ints, base)
        floatArray = float_to_base_p(floatPart, length-ints, base)

        qf._array[0:ints] = intArray
        qf._array[ints:] = floatArray

        return qf

    def toFloat(self):
        """
        Create a float from the QFloat

        WARNING : does not work in FHE
        """
        integerPart = base_p_to_int(self._array[0:self._ints], self._base)
        floatPart = base_p_to_float(self._array[self._ints:], self._base)

        return integerPart + floatPart

    def zero(length, ints, base):
        """
        Create a QFloat of 0
        """
        if not (isinstance(length, int) and length > 0):
            raise ValueError('length must be a positive int')
        return QFloat(fhe.zeros(length), ints, base)

    def zero_like(other):
        """
        Create a QFloat of 0 with same shape as other
        """
        if not isinstance(other, QFloat):
            raise ValueError('Object must be a QFloat')

        return QFloat.zero(len(other), other._ints, other._base)

    def one(length, ints, base):
        """
        Create a QFloat of 1
        """
        array=fhe.zeros(length)
        array[ints-1] = fhe.ones(1)[0]
        return QFloat(array, ints, base)

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
        return self.__class__(self._array[:], self._ints, self._base)       

    def toArray(self):
        """
        Rerturn copy of array
        """
        return self._array[:]

    def checkCompatibility(self, other):
        """
        Check wether other has equal encoding
        """
        if not isinstance(other, QFloat):
            raise ValueError('Object must be a QFloat')

        if self._base != other._base:
            raise ValueError('QFLoat bases are different')

        if len(self) != len(other):
            raise ValueError('QFLoat have different length')

        if self._ints != other._ints:
            raise ValueError('QFLoat have different dot index')        

    def getSign(self):
        """
        Return the sign of the QFloat
        Note: 0 is considered positive for faster computations
        """
        if not self._QFZero:
            self._QFZero = QFloat.zero_like(self)

        return self >= self._QFZero

    # def tidy(self):
    #     """
    #     Tidy array so that values are in range [-base, base] as they should be
    #     Keeping arrays untidy when possible saves computation time
    #     """
    #     abs_ge_base = np.abs(self._array) >= base

    #     for i in reversed(range(len(self))):
    #         abs_ge_base[i]

    #     self._isTidy=True

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

        If array have different length, the shorter one is considered to have extra extra -np.inf values
        """     
        self.checkCompatibility(other)

        if not (self._isTidy and other._isTidy):
            raise Exception('cannot compare QFloats that are not tidy')

        n = len(self)

        A=self._array  # doesn't copy array, just for naming clarity
        B=other._array # doesn't copy array, just for naming clarity

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

    def __getitem__(self, index):
        """Return a subsequence as a single integer or as a sequence object.
        """
        if isinstance(index, numbers.Integral):
            # Return a single integer
            return self._array[index]
        else:
            # Return the (sub)sequence as another Seq/MutableSeq object
            return self.__class__(self._array[index])

    def __add__(self, other):
        """
        Sum with another QFLoat
        Summing will potentially make values in the sum array be greater than the base, so tidy becomes False
        """
        self.checkCompatibility(other)
        addition = QFloat(self._array + other._array, self._ints, self._base)
        addition._isTidy = False
        return addition

    def __sub__(self, other):
        """
        Subtract another QFLoat
        Subtracting will potentially make values in the sum array be greater than the base, so tidy becomes False
        """
        self.checkCompatibility(other)
        subtraction = QFloat(self._array - other._array, self._ints, self._base)
        subtraction._isTidy = False
        return subtraction

