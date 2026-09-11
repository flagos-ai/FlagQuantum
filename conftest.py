# flagquantum/conftest.py
import os

# Must be set before any imports
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if os.environ.get("PYTEST_DEBUG_CONFIG"):
    print("=" * 60)
    print("Pytest configuration loaded")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
    print("=" * 60)
