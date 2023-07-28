import time
from typing import Tuple

import numpy as np
from concrete import fhe

from qf_invert import qf_matrix_inverse, float_matrix_to_qfloat_arrays, qfloat_arrays_to_float_matrix

# def invert_matrix(x):
#     return x

class EncryptedMatrixInversion:
    shape: Tuple[int, int]
    circuit: fhe.Circuit

    def __init__(self, n, sampler):
        self.shape = (n, n)

        # custom quantization parameters
        self.qf_len = 12; # QFloats length
        self.qf_ints = 9; # QFloats integer part length
        self.qf_base = 2; # QFloats base (2=binary)
        self.qf_len_out=20; # output QFloats length
        self.qf_ints_out = 8; # output QFloats integer part length
        
        params = [n, self.qf_len, self.qf_ints, self.qf_base, self.qf_len_out, self.qf_ints_out]

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
        return qfloat_arrays_to_float_matrix(quantized_inverted_matrix, self.qf_ints_out, self.qf_base)

    def run(self, matrix: np.ndarray) -> np.ndarray:
        assert np.issubdtype(matrix.dtype, np.floating)
        assert matrix.shape == self.shape

        quantized_matrix, qfloats_signs = self.quantize(matrix)
        encrypted_quantized_matrix = self.encrypt(quantized_matrix, qfloats_signs)
        encrypted_quantized_inverted_matrix = self.evaluate(encrypted_quantized_matrix)
        quantized_inverted_matrix = self.decrypt(encrypted_quantized_inverted_matrix)
        inverted_matrix = self.dequantize(quantized_inverted_matrix)

        assert np.issubdtype(inverted_matrix.dtype, np.floating)
        assert inverted_matrix.shape == self.shape

        return inverted_matrix


normal_sampler = ("Normal", lambda: np.random.randn(n, n) * 100)
uniform_sampler = ("Uniform", lambda: np.random.uniform(0, 100, (n, n)))

for name, sampler in {normal_sampler, uniform_sampler}:
    for n in {3, 5, 10}:        
        print()

        title = f"Sampler={name}, N={n}"
        print(title)
        print("-" * len(title))

        print(f"Compiling...")
        start = time.time()
        encrypted_matrix_inversion = EncryptedMatrixInversion(n, sampler)
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