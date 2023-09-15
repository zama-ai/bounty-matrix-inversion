"""
This module implements functions operating on arrays reprenseting
integers in little endian base p (p=2 means binary)
author : Lcressot
"""

import numpy as np
from concrete import fhe


def base_p_to_int(arr, p):
    """
    Convert base p array to int
    Can be signed (+/- values)
    """
    # Flip the array to have least significant digit at the start
    arr = np.flip(arr)
    # Compute the place values (p^0, p^1, p^2, ...)
    place_values = p ** np.arange(arr.size)
    # Return the sum of arr*place_value
    return np.sum(arr * place_values)


def int_to_base_p(integer, n, p):
    """
    Convert an integer into base-p array
    Can be signed (+/- values)
    """
    if n == 0:
        return np.array([], dtype=int)

    sgn = int(np.sign(integer))
    integer = int(np.abs(integer))

    if not isinstance(integer, int) or not isinstance(n, int) or not isinstance(p, int):
        raise ValueError("All inputs must be integers")
    if integer < 0 or n <= 0 or p <= 1:
        raise ValueError("Invalid input values")

    result = np.zeros(n, dtype=int)

    for i in reversed(range(n)):
        power = pow(p, i)
        div = integer // power
        integer -= div * power
        result[n - 1 - i] = div

    return result * sgn


def base_p_to_float(arr, p):
    """
    Convert a base-p array to a float of the form 0.xxx..
    Can be signed (+/- values)
    """
    f = 0.0
    for i in range(len(arr)):
        f += arr[i] * (p ** -(i + 1))
    return f


def float_to_base_p(f, precision, p):
    """
    Convert a float of type 0.xxx.. to a base-p array with a given precision.
    Can be signed (+/- values)
    """
    sgn = np.sign(f)
    f = np.abs(f)
    assert 0 <= f < 1, "Input should be a float between 0 and 1 (exclusive)"
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
    return sgn * np.array(basep)


def base_p_addition(a, b, p, inplace=False):
    """
    Add arrays in base p. (a and b must be positive and tidy)

    WARNING: If a and b have different sizes, the longer array is
    considered to have zeros on its left side

    if inplace is True, the result is written is a
    """
    carry = 0
    if inplace:
        result = a
    else:
        result = fhe.zeros(a.size)

    # Loop through both arrays and perform base p addition
    for i in range(min(a.size, b.size)):
        sum_ = a[-i - 1] + b[-i - 1] + carry
        result[-i - 1] = sum_ % p
        carry = sum_ // p

    return result


def base_p_subtraction(a, b, p, overflow=False):
    """
    Subtract arrays in base p. (a and b must be tidy)

    If a < b, the difference is wrong and full of ones on the left part,
    so we can return in addition wether a < b
    """
    min_size = min(a.size, b.size)
    difference = fhe.zeros(a.size)
    borrow = 0
    a_minus_b = a[-min_size:] - b[-min_size:]
    for i in range(min_size):
        # Perform subtraction for each bit
        temp = a_minus_b[-i - 1] - borrow
        borrow = temp < 0
        difference[-i - 1] = temp + p * borrow

    if not overflow:
        return difference

    else:
        # now lets compute if there was an overflow (which also tell us if a < b)
        diff = b.size - a.size
        if diff == 0:
            a_lt_b = borrow
        elif diff < 0:
            a_lt_b = borrow & (np.sum(a[0:-diff]) == 0)
            difference[0:-diff] = a[0:-diff]
        else:
            a_lt_b = borrow | (np.sum(b[0:diff]) > 0)

        return difference, a_lt_b


def multi_base_p_subtraction(a_arrays, b_arrays, p, overflow=False):
    """
    Tensorized version of base_p_subtraction
    """
    min_size = min(a_arrays.shape[1], b_arrays.shape[1])
    difference = fhe.zeros(a_arrays.shape)
    borrow = fhe.zeros(a_arrays.shape[0])
    a_minus_b = a_arrays[:, -min_size:] - b_arrays[:, -min_size:]
    for i in range(min_size):
        # Perform subtraction for each bit
        temp = a_minus_b[:, -i - 1] - borrow
        borrow = temp < 0
        difference[:, -i - 1] = temp + p * borrow

    if not overflow:
        return difference

    else:
        # now lets compute if there was an overflow (which also tell us if a < b)
        diff = b_arrays.shape[1] - a_arrays.shape[1]
        if diff == 0:
            a_lt_b = borrow
        elif diff < 0:
            a_lt_b = borrow & (np.sum(a_arrays[:, 0:-diff], axis=1) == 0)
            difference[:, 0:-diff] = a_arrays[:, 0:-diff]
        else:
            a_lt_b = borrow | (np.sum(b_arrays[:, 0:diff], axis=1) > 0)

        return difference, a_lt_b


def base_p_division(dividend, divisor, p):
    """
    Divide arrays in base p. (dividend and divisor must be tidy and positive)
    """
    # Initialize the quotient array
    quotient = fhe.zeros(dividend.size)
    # Initialize the remainder
    remainder = dividend[0].reshape(1)

    for i in range(dividend.size):
        if i > 0:
            # Left-roll the remainder and bring down the next digit from the dividend
            # also cut remainder if its size is bigger than divisor's, cause there are extra zeros
            d = 1 * (remainder.size > divisor.size)
            remainder = np.concatenate((remainder[d:], dividend[i].reshape(1)), axis=0)
        # If the remainder is larger than or equal to the divisor
        for _ in range(p - 1):
            # let's compute both (remainder - divisor) and (remainded < divisor) at once
            difference, is_lt = base_p_subtraction(remainder, divisor, p, True)
            is_ge = 1 - is_lt

            remainder = (
                # tensor_fast_boolean_mul(difference, is_ge)
                # + tensor_fast_boolean_mul(remainder, is_lt)
                difference * is_ge
                + remainder * is_lt
            )
            # Update the current quotient digit
            quotient[i] += is_ge

    return quotient


def multi_base_p_division(dividends, divisors, p):
    """
    Tensorized version of base_p_division
    """
    n_arrays = dividends.shape[0]
    # Initialize the quotients arrays
    quotients = fhe.zeros(dividends.shape)
    # Initialize the remainders
    remainders = dividends[:, 0].reshape((n_arrays, 1))

    for i in range(dividends.shape[1]):
        if i > 0:
            # Left-roll the remainders and bring down the next bit from the dividends
            # also cut remainders if its size is bigger than divisors's, cause there are extra zeros
            d = 1 * (remainders.shape[1] > divisors.shape[1])
            remainders = np.concatenate(
                (remainders[:, d:], dividends[:, i].reshape((n_arrays, 1))), axis=1
            )
        # If the remainders is larger than or equal to the divisors
        for _ in range(p - 1):
            # let's compute both (remainders - divisors) and (remainders < divisors) at once
            differences, are_lt = multi_base_p_subtraction(
                remainders, divisors, p, True
            )
            are_lt = are_lt.reshape((n_arrays, 1))
            are_ge = 1 - are_lt

            remainders = (
                # tensor_fast_boolean_mul(differences, are_ge)
                # + tensor_fast_boolean_mul(remainders, are_lt)
                differences * are_ge
                + remainders * are_lt
            )
            # Update the current quotient digit
            quotients[:, i] += are_ge.flatten()

    return quotients


def is_greater_or_equal(a, b):
    """
    Fast computation of wether an array number a is greater or equal to an array number b

    Both arrays must be base tidy, in which case the subtraction
    of a-b will work if a>=b and overflow if a<b

    The overflow is a fast way to compute wether a>=b <=> not a<b
    """
    min_size = min(a.size, b.size)
    a_minus_b = a[-min_size:] - b[-min_size:]
    borrow = 0
    for i in range(min_size):
        # report borrow
        borrow = a_minus_b[-i - 1] - borrow < 0
    return 1 - borrow


def multi_is_greater_or_equal(a_arrays, b_arrays):
    """
    Tensorized version of is_greater_or_equal
    """
    min_size = min(a_arrays.shape[1], b_arrays.shape[1])
    a_minus_b = a_arrays[:, -min_size:] - b_arrays[:, -min_size:]
    borrow = fhe.zeros(a_arrays.shape[0])
    for i in range(min_size):
        # report borrow
        borrow = a_minus_b[:, -i - 1] - borrow < 0
    return 1 - borrow


def is_equal(a, b):
    """
    Computes wether an array is equal to another
    """
    return (a.size - np.sum(a == b)) == 0


def is_positive(a):
    """
    Fast computation of wether an array number a is positive (or zero)
    a must be base tidy (returns the first non zero sign)
    """
    borrow = 0
    for i in range(a.size):
        # report borrow
        borrow = a[-i - 1] - borrow < 0
    return 1 - borrow


def is_greater_or_equal_base_p(a, b):
    """
    Computes wether a base-p number (little endian) is greater
    or equal than another, works for different sizes
    """
    diff = b.size - a.size
    if diff == 0:
        return is_greater_or_equal(a, b)
    if diff > 0:
        return is_greater_or_equal(a, b[diff:]) & (np.sum(b[0:diff]) == 0)

    return is_greater_or_equal(a[-diff:], b) | (np.sum(a[0:-diff]) > 0)


def multi_is_greater_or_equal_base_p(a_arrays, b_arrays):
    """
    Tensorized version of is_greater_or_equal_base_p
    """
    diff = b_arrays.shape[1] - a_arrays.shape[1]
    if diff == 0:
        return multi_is_greater_or_equal(a_arrays, b_arrays)
    if diff > 0:
        return multi_is_greater_or_equal(a_arrays, b_arrays[:, diff:]) & (
            np.sum(b_arrays[:, 0:diff]) == 0
        )

    return multi_is_greater_or_equal(a_arrays[:, -diff:], b_arrays) | (
        np.sum(a_arrays[:, 0:-diff]) > 0
    )


def insert_array_at_index(a, B, i, j):
    """
    Insert elements of array a into array B[i,:] starting at index j
    """
    # Case when j is negative
    if j < 0:
        # We will take elements of a starting at index -j
        a = a[-j:]
        j = 0
    # Compute the number of elements we can insert from a into b
    n = min(B.shape[1] - j, a.size)
    # Insert elements from a into B[i]
    B[i, j : j + n] = a[:n]


def insert_array_at_index_3D(A, B, i, j):
    """
    Tensorized version of insert_array_at_index
    Insert elements of 2D array A into 3D array B[:,i,j:]
    """
    # Case when j is negative
    if j < 0:
        # We will take elements of a starting at index -j
        A = A[:, -j:]
        j = 0
    # Compute the number of elements we can insert from a into b
    n = min(B[0, i].size - j, A[0].size)
    # Insert elements from a into B[i]
    B[:, i, j : j + n] = A[:, :n]


def tensor_fast_boolean_mul(x, boolean):
    """
    Fast multiplication of a tensor with a boolean
    This is sometimes faster than usual multiplications
    """
    pack = (x * 2) + boolean
    return fhe.univariate(lambda pack: np.where(pack & 1 == 0, 0, pack >> 1))(pack)
