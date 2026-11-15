# Measurement tutorial

Q-0245 is the complete `qualtran/bloqs/basic_gates/measurement.ipynb` at
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`6ef102fbe51210ad742e23dd6b0833a4d90d5a44`. The source remains unchanged.

| Original topic | Native evidence |
|---|---|
| `MeasureZ`, concrete example and graphical signature | Same single-bit logical signature, `q` (`QBit`, LEFT) and `c` (`CBit`, RIGHT). The runtime example binds physical wire 0 and classical slot 0. |
| `MeasureZ` call graph and leaf count | The locked source and native declaration each have one atomic measurement leaf and no edges. |
| `MeasureX`, concrete example and graphical signature | The same destructive logical signature, with X-basis projection. Explicit H before Z measurement implements the basis change in native lowering. |
| `MeasureX` call graph and leaf count | The locked source and native declaration each have one atomic measurement leaf and no edges. The declaration treats the basis as part of the measurement. |

Both logical inputs are scalar, so neither signature has a register bit-order
choice. Outcome 0 corresponds to `|0>` or `|+>` and outcome 1 corresponds to
`|1>` or `|->`. Surviving wires keep their original ordering. The existing
exact branch tests use the original effect tensors as bras, compare the
unnormalized complex amplitudes for both outcomes, and verify the consumed
physical slot is zero after clear. Existing owned-register and tensor tests
also check classical output ownership, spectator density and retained phase.

The added `circuit.append(op)` example uses the same phased Bell state for
both measurement bases. It computes each branch probability and checks the
full normalized complex states from the dynamic runtime for both labels.
The fixed-width execution retains the physical slot, clears it using feedback
from the same outcome, and exposes one classical result. This physical-slot
policy is separate from logical consumption and from dropping a result.

## Emission and resource accounting

| Basis | Emitted instructions for the measurement and clear | Source model |
|---|---|---|
| Z | One measurement and one X conditioned on outcome 1 | One measurement, zero modeled Clifford or T, peak one qubit |
| X | One H, one measurement and one X conditioned on outcome 1 | One measurement, zero modeled Clifford or T, peak one qubit |

The emitters reuse the original physical wire and require no extra work wire.
The example prints static instruction counts, including the conditional X.
The source model applies to the logical instrument alone.

## Notebook verification

The native notebook retains all 18 original cells' topics and adds two cells
for the append example. The native execution checks both 0.5 branch weights
and both normalized complex branches in each basis.

Both notebooks ran in independent kernels. The native kernel rejected
Qualtran, Quimb and Cirq imports, and all four rendered SVGs were inspected.
Five original measurement cases, 608 concentrated native regression cases
and 109 integration cases passed. Strict typing, formatting, team scope,
architecture, the multilevel IR contract and the public API baseline passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_measurement.ipynb
```

The companion X- and Z-basis implementation files remain under their own
inventory items Q-0268 and Q-0273.
