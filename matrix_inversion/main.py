import time
from typing import Tuple

import numpy as np
from concrete import fhe

from qfloat_matrix_inversion import (
    qfloat_matrix_inverse,
    float_matrix_to_qfloat_arrays,
    qfloat_and_signs_arrays_to_float_matrix,
)

# def invert_matrix(x):
#     return x


class EncryptedMatrixInversion:
    shape: Tuple[int, int]
    circuit: fhe.Circuit

    def __init__(
        self, n, sampler, qfloat_base=2, qfloat_len=32, qfloat_ints=16, true_division=False, tensorize=True
    ):
        self.shape = (n, n)

        # custom quantization parameters
        self.qfloat_base = qfloat_base
        # QFloats base (2=binary)
        self.qfloat_len = qfloat_len
        # QFloats length
        self.qfloat_ints = qfloat_ints
        # QFloats integer part length

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

        compiler = fhe.Compiler(
            lambda x, y: qfloat_matrix_inverse(
                x, y, n, self.qfloat_len, self.qfloat_ints, self.qfloat_base, true_division, tensorize
            ),
            {"x": "encrypted", "y": "encrypted"},
        )
        self.circuit = compiler.compile(quantized_inputset)

    def quantize(self, matrix: np.ndarray) -> np.ndarray:
        return float_matrix_to_qfloat_arrays(
            matrix, self.qfloat_len, self.qfloat_ints, self.qfloat_base
        )

    def encrypt(
        self, quantized_matrix: np.ndarray, qfloats_signs: np.ndarray
    ) -> fhe.PublicArguments:
        return self.circuit.encrypt(quantized_matrix, qfloats_signs)

    def evaluate(
        self, encrypted_quantized_matrix: fhe.PublicArguments
    ) -> fhe.PublicResult:
        return self.circuit.run(encrypted_quantized_matrix)

    def decrypt(
        self, encrypted_quantized_inverted_matrix: fhe.PublicResult
    ) -> np.ndarray:
        return self.circuit.decrypt(encrypted_quantized_inverted_matrix)

    def dequantize(self, quantized_inverted_matrix: np.ndarray) -> np.ndarray:
        return qfloat_and_signs_arrays_to_float_matrix(
            quantized_inverted_matrix, self.qfloat_ints, self.qfloat_base
        )

    def run(self, matrix: np.ndarray, simulate=False) -> np.ndarray:
        assert np.issubdtype(matrix.dtype, np.floating)
        assert matrix.shape == self.shape

        quantized_matrix, qfloats_signs = self.quantize(matrix)
        if not simulate:
            encrypted_quantized_matrix = self.encrypt(quantized_matrix, qfloats_signs)
            encrypted_quantized_inverted_matrix = self.evaluate(
                encrypted_quantized_matrix
            )
            quantized_inverted_matrix = self.decrypt(
                encrypted_quantized_inverted_matrix
            )
        else:
            quantized_inverted_matrix = self.circuit.simulate(
                quantized_matrix, qfloats_signs
            )

        inverted_matrix = self.dequantize(quantized_inverted_matrix)

        assert np.issubdtype(inverted_matrix.dtype, np.floating)
        assert inverted_matrix.shape == self.shape

        return inverted_matrix


normal_sampler = ("Normal", lambda: np.random.randn(n, n) * 100)
uniform_sampler = ("Uniform", lambda: np.random.uniform(0, 100, (n, n)))


"""
Custom quantization parameters :

qfloat_len = QFloats length
qfloat_ints = QFloats integer part length
qfloat_base = QFloats base (2=binary)

true_division: wether to perform true divisions in the inversion algorithm (more precise but slower)
"""

tensorize = True # seems to be better if dataflow_parallelize=True

# less precise, more prone to errors, but faster
true_division = False
qfloat_base = 2
qfloat_len = 23
qfloat_ints = 9

# intermediate precision
# true_division = False
# qfloat_base = 2; qfloat_len = 28; qfloat_ints = 13;  # intermediate

# more precise, less prone to errors, but slower
# true_division = False
# qfloat_base = 2; qfloat_len = 32; qfloat_ints = 16;

# even more precise, even less prone to errors, but even slower
# true_division = True
# qfloat_base = 2; qfloat_len = 32; qfloat_ints = 16; # more precise, less prone to errors, but slower

# qfloat_base = 16; qfloat_len = 8; qfloat_ints = 4; # to test with the new release (concrete bug for now), could be faster

for name, sampler in {normal_sampler, uniform_sampler}:
    for n in [2, 3, 5, 10]:
        print()

        title = f"Sampler={name}, N={n}"
        print(title)
        print("-" * len(title))

        print(f"Compiling...")
        start = time.time()
        encrypted_matrix_inversion = EncryptedMatrixInversion(
            n, sampler, qfloat_base, qfloat_len, qfloat_ints, true_division, tensorize
        )
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
