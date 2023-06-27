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

def int_to_base2_array(integer):
    # Convert the integer to a base2 string
    base2_string = np.binary_repr(integer)
    # Convert the base2 string to a NumPy array of integers
    base2_array = np.array([int(bit) for bit in base2_string])
    return base2_array

def base2_division(dividend, divisor):
    # Initialize the quotient array
    quotient = np.zeros_like(dividend)

    # Initialize the remainder
    remainder = np.zeros_like(divisor)

    for i in range(len(dividend)):
        # Left-shift the remainder
        remainder = np.roll(remainder, -1)
        # Bring down the next bit from the dividend
        remainder[-1] = dividend[i]

        # If the remainder is larger than or equal to the divisor
        if (remainder >= divisor).all():
            # Subtract the divisor from the remainder
            remainder = base2_subtraction(remainder, divisor)
            # Set the current quotient bit to 1
            quotient[i] = 1
        else:
            quotient[i] = 0

    return quotient

def base2_subtraction(minuend, subtrahend):
    difference = np.zeros_like(minuend)
    borrow = 0
    for i in reversed(range(len(minuend))):
        # Perform subtraction for each bit
        temp = minuend[i] - subtrahend[i] - borrow
        if temp < 0:
            temp += 2
            borrow = 1
        else:
            borrow = 0
        difference[i] = temp
    return difference

_45 = int_to_base2_array(45)
_8 = int_to_base2_array(8)

print(base2_division(_45, _8))
print(int_to_base2_array(5))