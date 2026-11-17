# Fourier package entry points

Q-0635 covers the locked `qualtran/bloqs/qft/__init__.py` at Qualtran
v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`cf1da7994d7f70e90ef04b7f1a63f99c1e464902`. The initializer is
Copyright 2023 Google LLC, Apache-2.0. Its entire executable scope is four
relative re-exports.

| Locked entry | FlagQuantum owner and circuit entry | Existing consumer and evidence |
| --- | --- | --- |
| `ApproximateQFT` | `algorithms.primitives.approximate_qft.approximate_qft_ir` | [Approximate QFT notebook](../../examples/notebooks/approximate_qft.ipynb) and [source audit](REGISTER_APPROXIMATE_QFT.md) |
| `QFTPhaseGradient` | `algorithms.primitives.qft_phase_gradient.qft_phase_gradient_ir` | [Phase-gradient QFT notebook](../../examples/notebooks/phase_gradient_qft.ipynb) and [source audit](REGISTER_PHASE_GRADIENT_QFT.md) |
| `QFTTextBook` | `algorithms.primitives.qft.qft` and `append_qft` | `MeasureQFT`, [textbook QFT notebook](../../examples/notebooks/textbook_qft.ipynb) and [source audit](TEXTBOOK_QFT_FILE_AUDIT.md) |
| `TwoBitFFFT` | `algorithms.primitives.fermionic_fourier.two_bit_fermionic_fourier_ir` | Hubbard plaquette, [fermionic Fourier notebook](../../examples/notebooks/two_bit_fermionic_fourier.ipynb) and [source audit](REGISTER_TWO_BIT_FERMIONIC_FOURIER.md) |

The first two routes use caller-bound gradient and clean work; the
textbook and two-bit circuits require neither. Each owner retains its
separate signature, ordered execution and resource evidence. The
existing public `qft` and `append_qft` are the same callables exposed
by the native primitive package. The gradient and fermionic routes
remain in their owned modules, with existing append bindings.
The frozen reference test checks all four original export identities,
their exact target modules, and the four native entry points.
The four migrated notebooks exercise the native routes through
`Circuit.append` in isolated kernels.

Q-0635 acceptance is for package correspondence. The source behavior
and model gaps recorded in Q-0637, Q-0638, Q-0640, Q-0641, Q-0644, Q-0646
and Q-0647 remain independently tracked.

The two focused package checks, four notebook kernel regressions and 5,566
concentrated tests passed. The concentrated run includes all four Fourier
suites, MeasureQFT and both Hubbard consumers. The only warning is the
known upstream symbolic T-count helper deprecation. Formatting, strict
typing, team scope, architecture, internal IR and public API checks passed.
