import numpy as np
import scipy as sc

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

def int_to_base_p_array(integer, n, p):
    # Convert the integer to a base p string
    base_p_string= np.base_repr(integer, p)
    # Prepend zeros to the binary representation until it reaches the desired size
    base_p_string = base_p_string.zfill(n)
    # Convert the base_p string to a NumPy array of integers
    base_p_array = np.array([int(bit) for bit in base_p_string])
    return base_p_array

def base_p_subtraction(a, b, p):
    difference = np.zeros_like(a)
    borrow = 0
    for i in reversed(range(len(a))):
        # Perform subtraction for each bit
        temp = a[i] - b[i] - borrow
        borrow = temp < 0
        temp += p*borrow
        difference[i] = temp
    return difference

# def base_p_subtraction(a, b):
#     return a-b

def is_greater_or_equal(a, b, i=0):
    if a.size==i:
        return True
    return (a[i]>b[i]) or ( (a[i]==b[i]) and is_greater_or_equal(a,b, i+1))


def base_p_division(dividend, divisor, p):
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
        for j in range(p-1):
            if is_greater_or_equal(remainder, divisor):
                # Subtract the divisor from the remainder
                remainder = base_p_subtraction(remainder, divisor, p)
                # Set the current quotient bit to 1
                quotient[i] += 1

    return quotient


def test_base_p_subtraction(n=20, k=100, p1=2, p2=5):
    for i in range(k):
        p = np.random.randint(p1,p2+1)
        b = np.random.randint(0,10000)
        a = np.random.randint(b,10000) # a >= b
        base_p_a = int_to_base_p_array(a, n, p)
        base_p_b = int_to_base_p_array(b, n, p)
        sub = base_p_subtraction(base_p_a, base_p_b, p)
        assert( base_p_to_int(sub, p) == (a-b) )
    print("test_base_p_subtraction OK")

def test_base_p_division(n=20, k=100, p1=2, p2=5):
    for i in range(k):
        p = np.random.randint(p1,p2+1)
        b = np.random.randint(0,10000)
        a = np.random.randint(b,10000) # a >= b
        base_p_a = int_to_base_p_array(a, n, p)
        base_p_b = int_to_base_p_array(b, n, p)
        div = base_p_division(base_p_a, base_p_b, p)
        assert( base_p_to_int(div, p) == a//b )
    print("test_base_p_division OK")    

import time


test_base_p_subtraction(20, 100)

start = time.time()
test_base_p_division(20, 100, 2,2)
end = time.time()
print('took: ' + str(end - start) + ' seconds')

start = time.time()
test_base_p_division(20, 100, 4,4)
end = time.time()
print('took: ' + str(end - start) + ' seconds')


# _45 = int_to_base_p_array(45)
# _8 = int_to_base_p_array(8)
# _27= int_to_base_p_array(27)
# _23 = int_to_base_p_array(23)
# _621 = int_to_base_p_array(621)

# print(base_p_division(_621, _27))
# print(_23)
# print(_27)
# print(_621)

#print(int_to_base_p_array(45, 20))
# print(int_to_base_p_array(8))
# print(int_to_base_p_array(5))



# def base_p_subtraction(a, b):
#     difference = np.zeros_like(a)
#     borrow = 0
#     for i in reversed(range(len(a))):
#         # Perform subtraction for each bit
#         temp = a[i] - b[i] - borrow
#         if temp < 0:
#             temp += 2
#             borrow = 1
#         else:
#             borrow = 0
#         difference[i] = temp
#     return difference

# def base_p_division(dividend, divisor):
#     # Initialize the quotient array
#     quotient = np.zeros_like(dividend)
#     # Initialize the remainder
#     remainder = np.zeros_like(divisor)

#     for i in range(len(dividend)):
#         # Left-roll the remainder
#         remainder = np.roll(remainder, -1)
#         # Bring down the next bit from the dividend
#         remainder[-1] = dividend[i]
#         # If the remainder is larger than or equal to the divisor
#         if is_greater_or_equal(remainder, divisor):
#             # Subtract the divisor from the remainder
#             remainder = base_p_subtraction(remainder, divisor)
#             # Set the current quotient bit to 1
#             quotient[i] = 1
#         else:
#             quotient[i] = 0

#     return quotient