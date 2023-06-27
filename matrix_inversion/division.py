import numpy as np
import scipy as sc

def base2_to_int(arr):
    """
    Convert base 2 array to int
    """
    # Flip the array to have least significant bit at the start
    arr = np.flip(arr)
    # Compute the place values (2^0, 2^1, 2^2, ...)
    place_values = 2**np.arange(arr.size)
    # Return the sum of bit*place_value
    return np.sum(arr * place_values)

def int_to_base2_array(integer, n):
    # Convert the integer to a base2 string
    base2_string = np.binary_repr(integer)
    # Prepend zeros to the binary representation until it reaches the desired size
    base2_string = base2_string.zfill(n)
    # Convert the base2 string to a NumPy array of integers
    base2_array = np.array([int(bit) for bit in base2_string])
    return base2_array

def base2_subtraction(a, b):
    difference = np.zeros_like(a)
    borrow = 0
    for i in reversed(range(len(a))):
        # Perform subtraction for each bit
        temp = a[i] - b[i] - borrow
        borrow = temp < 0
        temp += 2*borrow
        difference[i] = temp
    return difference

def base2_subtraction(a, b):
    difference = np.zeros_like(a)
    borrow = 0
    for i in reversed(range(len(a))):
        # Perform subtraction for each bit
        temp = a[i] - b[i] - borrow
        if temp < 0:
            temp += 2
            borrow = 1
        else:
            borrow = 0
        difference[i] = temp
    return difference

def is_greater_or_equal(a, b):
    if a.size==0:
        return True
    
    return (a[0]>b[0]) or ( (a[0]==b[0]) and is_greater_or_equal(a[1:], b[1:]))

def base2_division(dividend, divisor):
    # Initialize the quotient array
    quotient = np.zeros_like(dividend)
    # Initialize the remainder
    remainder = np.zeros_like(divisor)

    for i in range(len(dividend)):
        # Left-roll the remainder
        remainder = np.roll(remainder, -1)
        # Bring down the next bit from the dividend
        remainder[-1] = dividend[i]
        # If the remainder is larger than or equal to the divisor
        if is_greater_or_equal(remainder, divisor):
            # Subtract the divisor from the remainder
            remainder = base2_subtraction(remainder, divisor)
            # Set the current quotient bit to 1
            quotient[i] = 1
        else:
            quotient[i] = 0

    return quotient


def test_base2_subtraction(n=20, k=100):
    for i in range(k):
        b = np.random.randint(0,10000)
        a = np.random.randint(b,10000) # a >= b
        base2_a = int_to_base2_array(a, n)
        base2_b = int_to_base2_array(b, n)
        sub = base2_subtraction(base2_a, base2_b)
        assert( base2_to_int(sub) == (a-b) )
    print("test_base2_subtraction OK")

def test_base2_division(n=20, k=100):
    for i in range(k):
        b = np.random.randint(0,10000)
        a = np.random.randint(b,10000) # a >= b
        base2_a = int_to_base2_array(a, n)
        base2_b = int_to_base2_array(b, n)
        div = base2_division(base2_a, base2_b)
        assert( base2_to_int(div) == a//b )
    print("test_base2_division OK")    

test_base2_subtraction(20, 100)
test_base2_division(20, 100)

# _45 = int_to_base2_array(45)
# _8 = int_to_base2_array(8)
# _27= int_to_base2_array(27)
# _23 = int_to_base2_array(23)
# _621 = int_to_base2_array(621)

# print(base2_division(_621, _27))
# print(_23)
# print(_27)
# print(_621)

#print(int_to_base2_array(45, 20))
# print(int_to_base2_array(8))
# print(int_to_base2_array(5))

