# Training State, Precision, And Determinism

`fq.Module.save_checkpoint()` writes a versioned, atomic PyTorch training-state
envelope containing module and optimizer state, RuntimePolicy, runtime plan,
FlagQuantum IR version/hash, seed and RNG state, precision policy, step, and
hybrid-parallel topology. `load_checkpoint()` rejects unknown versions,
different IR versions, precision mismatches, and non-equivalent data/state/model
group layouts before restoring state.

`flagquantum.training.PrecisionPolicy` supports complex64/float32 and
complex128/float64 full
precision, plus an explicit mixed policy with a named accumulator dtype and
tolerances. Full precision rejects mismatched circuit, parameter, or
accumulator dtypes; differing accumulator precision therefore requires
`mode="mixed"` rather than becoming an accidental intermediate cast. Reducing
trainable parameter precision raises unless
`allow_parameter_downcast=True`; circuit gate construction follows the declared
complex dtype and cannot silently return complex64 for a complex128 circuit.

`flagquantum.training.seed_everything(seed)` seeds Python and Torch CPU/CUDA,
enables deterministic
algorithms by default, and returns independent sampling and trajectory seeds
plus explicit JAX PRNG key words. JAX has no hidden global RNG ownership; a JAX
adapter must consume the reported key explicitly.

Set `RuntimePolicy(correctness_debug=True)` to check module parameters and
results for NaN/Inf and install a gradient hook that fails at the first
non-finite backward boundary. Low-level manual checks after optimizer or
distributed synchronization phases remain an internal/experimental concern;
they are not additional stable root APIs.

Determinism means bitwise replay for supported deterministic PyTorch CPU
operations on equivalent software and topology. Accelerator/vendor kernels may
provide tolerance-bounded reproducibility instead; the active precision policy
records `atol` and `rtol` and must accompany such evidence.
