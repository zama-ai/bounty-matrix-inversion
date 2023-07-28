# Matrix inversion for TFHE ZAMA concrete

### installation
```bash
poetry install
```

### run
```bash
poetry shell
python matrix_inversion/matrix_invert.py
```

### dev test
```bash
poetry shell
python tests/test_QFloat.py
python tests/test_FHE.py
python matrix_inversion/qf_invert.py
```