# Logical cat qubits

Run the complete original experiment locally:

```bash
python -m examples.synthesis.community.basic_examples.logical_qubits.logical_qubits_by_alice_and_bob.overview.main
```

The 17-stage Notebook keeps `RX(0.1)`, fresh `RandomState(12)` amplitudes for two identical three-bit registers, the seven-bit SWAP test, the original distance-31 configuration and all 105 combinations of five photon numbers, seven odd distances and three two-photon loss scales. Each executed case makes 10,000 actual finite draws: 1,070,000 in the full workflow. The same physical discrete program is reused across the grid; only the analytical processor parameters change.

The example-local Solovay–Kitaev implementation returns actual H/T/Tdg words and measures their operator errors. It uses an explicit two-iteration default/grid and three-iteration accuracy demonstration. The source places the grid's SK setting in execution preferences; its hosted compiler's exact default, words and depths are not available. Native words, approximation errors and depths are reported on their own basis. Only uncontrolled single-qubit rotations are approximated; an unsupported gate is refused. Exact input preparation is independently checked before approximation. Approximation is never presented as exact source-state fidelity.

The physical noise formulas follow the public Apache-2.0 Alice & Bob processor at commit `9ad0893cf9d70555f68cd71a7d1773b743811177`:
[logical-cat analytical model](https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider/blob/9ad0893cf9d70555f68cd71a7d1773b743811177/qiskit_alice_bob_provider/processor/logical_cat.py).
The pinned formulas are independently audited at all 106 original physical configurations. Bit and phase rates are exponentiated into a valid Pauli channel, including saturation in the original distance-31 case. They are not replaced with a depolarizing model.

Native execution uses an explicit all-to-all ASAP unit-cycle schedule. Every gate cycle lasts `5 * distance / kappa_2` seconds. Preparation, readout and idle-wire correction are included. The original seven-bit grid program contains 13,339 physical instructions, 6,144 correction clocks and 43,008 single-qubit noise locations. Between entanglers, every physical gate and correction is composed into bounded CPTP channels. The 301 merged execution channels are reported separately from physical resources; no physical gate, idle location, input bit or sample is omitted. All 16,384 entries of one complete original density matrix are independently checked against an unmerged gate-by-gate and error-location oracle.

The fresh native ASAP model and its measured outcomes do not claim equivalence to an unavailable hosted scheduling implementation or historical vendor samples. Actual finite observations above 0.5 remain in the results and labels; the heatmap keeps the source display limits of 0 to 0.5. Very small predicted probabilities and zero observed counts remain distinct in the full JSON report. Local random generators leave application-global RNG state untouched. No vendor SDK, credentials, cloud job, production dependency or protected API change is required.
