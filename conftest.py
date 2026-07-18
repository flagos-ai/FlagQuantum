# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
