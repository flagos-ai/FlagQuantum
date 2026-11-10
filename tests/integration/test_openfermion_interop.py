"""The optional migration adapter leaves the native environment independent."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_native_import_and_measurement_with_upstream_blocked() -> None:
    script = """
        import importlib.abc
        import sys
        class NoUpstream(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'openfermion', 'cirq'}:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, NoUpstream())
        import flagquantum as fq
        from flagquantum.ecosystem.openfermion import (
            from_openfermion_fermion, from_openfermion_qubit,
            to_openfermion_fermion, to_openfermion_qubit,
        )
        from flagquantum.observables.fermion import FermionOperator, number, jordan_wigner
        from flagquantum.core.algebra.pauli import PauliOperator
        circuit = fq.Circuit(2).x(0)
        value = fq.run(circuit, outputs=[fq.expectation(jordan_wigner(number(0), n_modes=2))])
        assert float(value.expectation().reshape(-1)[0]) == 1
        for call, source in (
            (from_openfermion_fermion, None), (from_openfermion_qubit, None),
            (to_openfermion_fermion, FermionOperator()),
            (to_openfermion_qubit, PauliOperator()),
        ):
            try:
                call(source)
            except ImportError as error:
                assert 'interoperability environment' in str(error)
            else:
                raise AssertionError('missing upstream dependency accepted')
        assert not {'openfermion', 'cirq'}.intersection(
            name.split('.')[0] for name in sys.modules
        )
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
