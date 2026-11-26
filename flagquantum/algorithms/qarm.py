"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.algorithms.quantum_ml.qarm import (
    _DEFAULT_SHOTS as _DEFAULT_SHOTS,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _HALF_TURN as _HALF_TURN,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _MAX_ITEM_QUBITS as _MAX_ITEM_QUBITS,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _MAX_SUPPORT_QUBITS as _MAX_SUPPORT_QUBITS,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    FrequentItemsetOperator as FrequentItemsetOperator,
)
from flagquantum.algorithms.quantum_ml.qarm import __all__ as __all__
from flagquantum.algorithms.quantum_ml.qarm import (
    _append_pattern_bracket as _append_pattern_bracket,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _append_support_match as _append_support_match,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _append_support_step as _append_support_step,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _ladder_ancillas as _ladder_ancillas,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _validated_incidence as _validated_incidence,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _validated_item_count as _validated_item_count,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _validated_matrix as _validated_matrix,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _validated_support_width as _validated_support_width,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    _validated_threshold as _validated_threshold,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    frequent_itemset_operator as frequent_itemset_operator,
)
from flagquantum.algorithms.quantum_ml.qarm import (
    run_frequent_itemset as run_frequent_itemset,
)

_implementation = _import_module("flagquantum.algorithms.quantum_ml.qarm")
_sys.modules[__name__] = _implementation
