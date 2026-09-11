# Circuit Drawing

This package renders circuits as terminal text or Matplotlib figures. It owns
layout, labels, gate symbols, and presentation styles. It does not execute circuits,
compile gates, choose devices, or submit jobs.

`ir_adapter.py` accepts Circuit/CircuitIR inputs and existing device operation
histories. Keep compatibility handling there. The text renderer depends only on
the standard library and Core; Matplotlib is optional and belongs only to the
figure renderer and style handling.

The public entry is `draw`; `draw_text` and `draw_mpl` select a renderer explicitly.
Drawing options belong to the caller. Copy mutable options before changing them,
and let callers close figures they receive.

```python
import flagquantum as fq
from flagquantum.drawer import draw

circuit = fq.Circuit(2)
circuit.h(0).cx(0, 1)
print(draw(circuit))
```

For a typical change, adjust a gate's renderer and check the diagram through
`tests/test_drawer_ir.py`. Changes to option handling also run
`tests/unit/test_drawer_input_ownership.py`. Use Matplotlib's `Agg` backend for
headless rendering tests; text drawing must still work without Matplotlib.
