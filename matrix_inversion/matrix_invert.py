import time
from typing import Tuple

import numpy as np
from concrete import fhe

from qf_invert import qf_matrix_inverse, float_matrix_to_qfloat_arrays, qfloatNsigns_arrays_to_float_matrix

# def invert_matrix(x):
#     return x

class EncryptedMatrixInversion:
    shape: Tuple[int, int]
    circuit: fhe.Circuit

    def __init__(self, n, sampler, qf_base = 2, qf_len = 32, qf_ints = 16, trueDivision=False):
        self.shape = (n, n)

        # custom quantization parameters
        self.qf_base = qf_base; # QFloats base (2=binary)
        self.qf_len = qf_len; # QFloats length
        self.qf_ints = qf_ints; # QFloats integer part length

        params = [n, self.qf_len, self.qf_ints, self.qf_base, trueDivision]

        inputset = [sampler() for _ in range(100)]
        for sample in inputset:
            assert isinstance(sample, np.ndarray)
            assert np.issubdtype(sample.dtype, np.floating)
            assert sample.shape == self.shape

        quantized_inputset = [self.quantize(sample) for sample in inputset]
        # for quantized_sample in quantized_inputset:
        #     assert isinstance(quantized_sample, np.ndarray)
        #     assert np.issubdtype(quantized_sample.dtype, np.integer)
        #     assert quantized_sample.shape == self.shape

        compiler = fhe.Compiler(lambda x,y: qf_matrix_inverse(x,y,params), {"x": "encrypted", "y": "encrypted"})
        self.circuit = compiler.compile(quantized_inputset)

    def quantize(self, matrix: np.ndarray) -> np.ndarray:
        return float_matrix_to_qfloat_arrays(matrix, self.qf_len, self.qf_ints, self.qf_base)

    def encrypt(self, quantized_matrix: np.ndarray, qfloats_signs: np.ndarray) -> fhe.PublicArguments:
        return self.circuit.encrypt(quantized_matrix, qfloats_signs)

    def evaluate(self, encrypted_quantized_matrix: fhe.PublicArguments) -> fhe.PublicResult:
        return self.circuit.run(encrypted_quantized_matrix)

    def decrypt(self, encrypted_quantized_inverted_matrix: fhe.PublicResult) -> np.ndarray:
        return self.circuit.decrypt(encrypted_quantized_inverted_matrix)

    def dequantize(self, quantized_inverted_matrix: np.ndarray) -> np.ndarray:
        return qfloatNsigns_arrays_to_float_matrix(quantized_inverted_matrix, self.qf_ints, self.qf_base)

    def run(self, matrix: np.ndarray, simulate=False) -> np.ndarray:
        assert np.issubdtype(matrix.dtype, np.floating)
        assert matrix.shape == self.shape

        quantized_matrix, qfloats_signs = self.quantize(matrix)
        if not simulate:    
            encrypted_quantized_matrix = self.encrypt(quantized_matrix, qfloats_signs)
            encrypted_quantized_inverted_matrix = self.evaluate(encrypted_quantized_matrix)
            quantized_inverted_matrix = self.decrypt(encrypted_quantized_inverted_matrix)
        else:
            quantized_inverted_matrix = self.circuit.simulate(quantized_matrix, qfloats_signs)

        inverted_matrix = self.dequantize(quantized_inverted_matrix)

        assert np.issubdtype(inverted_matrix.dtype, np.floating)
        assert inverted_matrix.shape == self.shape

        return inverted_matrix


normal_sampler = ("Normal", lambda: np.random.randn(n, n) * 100)
uniform_sampler = ("Uniform", lambda: np.random.uniform(0, 100, (n, n)))


"""
Custom quantization parameters :

qf_len = QFloats length
qf_ints = QFloats integer part length
qf_base = QFloats base (2=binary)

trueDivision: wether to perform true divisions in the inversion algorithm (more precise but slower)
"""

# less precise, more prone to errors, but faster
trueDivision = False
qf_base = 2; qf_len = 23; qf_ints = 9;

# intermediate precision
#trueDivision = False
#qf_base = 2; qf_len = 28; qf_ints = 13;  # intermediate

# more precise, less prone to errors, but slower
#trueDivision = False
#qf_base = 2; qf_len = 32; qf_ints = 16;

# even more precise, even less prone to errors, but even slower
#trueDivision = True
#qf_base = 2; qf_len = 32; qf_ints = 16; # more precise, less prone to errors, but slower

#qf_base = 16; qf_len = 8; qf_ints = 4; # to test with the new release (concrete bug for now), could be faster

for name, sampler in {normal_sampler, uniform_sampler}:
    for n in [2, 3, 5, 10]:
        print()

        title = f"Sampler={name}, N={n}"
        print(title)
        print("-" * len(title))

        print(f"Compiling...")
        start = time.time()
        encrypted_matrix_inversion = EncryptedMatrixInversion(n, sampler, qf_base, qf_len, qf_ints, trueDivision)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print(f"Generating Keys...")
        start = time.time()
        encrypted_matrix_inversion.circuit.keygen()
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        sample_input = sampler()
        expected_output = np.linalg.inv(sample_input)

        print(f"Running...")
        start = time.time()
        actual_output = encrypted_matrix_inversion.run(sample_input)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        error = np.abs(expected_output - actual_output)

        print(f"Average Error: {np.mean(error):.6f}")
        print(f"    Max Error: {np.max(error):.6f}")
        print(f"    Min Error: {np.min(error):.6f}")
        print(f"  Total Error: {np.sum(error):.6f}")
        
        print()