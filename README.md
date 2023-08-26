# Matrix inversion for TFHE ZAMA concrete

## Installation
```bash
poetry install
```

## Run
```bash
poetry shell
python matrix_inversion/matrix_invert.py
```

## Dev

### Running tests
```bash
poetry shell
python tests/test_QFloat.py
python tests/test_QFloat_FHE.py
python matrix_inversion/qf_invert.py
```

### Contributing

see TODO in `QFLoat.py` and `qf_invert.py`

## About

### QFloats to quantize floats in FHE
- The **QFloat** class allows to quantize float numbers as FHE compatible arrays of integers, and to make operations between them like additions, multiplications, etc.
The array can be made in any given base (for instance in base 2 which is binary) and with any precision (the longer the array and the greater the base, the more precise the **QFloat** representation).  

- The array is split between an integer part and a pure floating part. For instance 34.42 = 34 + 0.32. The length of the integer part must be provided along with the total length of the array to create a QFloat. The longer the integer part is, the higher the **QFloat** can go to encode a number, and the longer the total length is (hence the floating part is) the more precise it can get.  
This representation allows to make any operations on the QFloats in FHE, but has a limited range of values. The usual unencrypted float representation that uses an exponent (like 3.442 e-1), cannot be made in FHE to run operations (mostly additions) without disproportionally huge computations.

- Along with the **QFloat** class, the **SignedBinary** and **Zero** classes can be used to limit the computations when we know values are binary numbers or are equal to zero. They are compatible with QFloats.


### Why using QFloats is required for matrix inversion
- To compute the inverse of a float matrix in FHE, one must inevitably run computations on representations of floats in FHE, with a data structure keeping a minimal precision of the initial floats which are 32 to 64 bits. Hence, a class like **QFloat** is absolutely required here, with enough precision provided.

### Why the LU decomposition algorithm was chosen
- The algorithm used for inverting the **QFloat** matrix is called **inversion from LU decomposition**. It is a perfect algorithm, not an approximate one. Approximate algorithms were investigated (ones that use convergence to estimate the inverse) but they all require constraints on the input matrix, which cannot be verified or guessed within FHE. The **LU decomposition** algorithm is perfect and works for all invertible matrices.  
Furthermore, these algorithms require a lot of matrix products which are expensive to compute, and it is not clear wether they would be faster than a perfect algorithm in FHE (provided they could be done somehow).

### How to set QFloats for the algorithm
- QFloats are created with 3 custom parameters:
	- `qf_base`: The base (base 2 is binary).
	- `qf_len`: The total length of the integer array. The greater it is, the more precise the qfloats get.
	- `qf_ints`: The length of the integer part of the float. The greater it is, the higher the **QFloat** can go.
- The `qf_base` needs to be 2 for the current version of concrete-python (2.1.0), because there is currently a bug preventing the use of higher bases. Higher bases would reduce greatly the size of the array for a similar precision, so that the compilation and running times are expected to diminish, while the encryption time is expected to increase.
- Any desired precision can be obtained if setting the QFloats with more precision (longer length), but the computation will be significantly slower.

### Further notice
- Running the computations in pure python can be useful to assess the correcteness of the output, see `qf_invert.py` in the test main part.
- The inversion algorithm could be optimized to be faster for the specific task of inverting matrices of floats in range ~0-100, but it was kept more general here.

