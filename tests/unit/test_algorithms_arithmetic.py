"""The reversible adder is the sum, and its cost is the compiler's own number.

Three properties are load-bearing here, and each is checked against an artifact
rather than against a second copy of the adder.

**It computes the sum.** ``flagquantum.algorithms.arithmetic`` builds a circuit,
and the assertion is over that circuit's own unitary: the matrix comes from
``flagquantum.simulation.unitary.get_unitary``, which runs the program's exact
statevector kernel once per basis state, so a change to the construction that
stopped summing shows up as a matrix column in the wrong place. Nothing here
re-derives the gadget sequence.

**It is inside its stated domain.** The adder needs two wires to enter holding
``|0>``: the working carry wire, which it restores, and the carry-out wire, which
it leaves holding the carry out of the top position. Only those inputs are
promised a sum. ``test_the_map_is_a_permutation_of_the_whole_space`` checks the
other half of that honesty: where the ancillas enter dirty the circuit is still a
permutation -- no information is lost and the map is reversible -- but it is not
the sum, and the test says so rather than quietly extending the claim.

**Its T-count is not a second opinion.** The compiler already owns the Toffoli
rule (``flagquantum/compiler/basis_translation.py``, the fifteen-gate identity),
so the tests lower the adder through
``flagquantum.compiler.basis_conversion.convert_basis`` and read the cost from
``flagquantum.compiler.resource_estimation.estimate_resources``, then assert that
``flagquantum.algorithms.logical_resources`` reports the same tally. A seven-T
form written into the arithmetic module would fail here, which is the drift this
file exists to catch.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from flagquantum.algorithms.arithmetic import (
    AdderWires,
    adder_circuit,
    adder_wires,
)
from flagquantum.algorithms.logical_resources import estimate_logical_resources
from flagquantum.circuit import Circuit
from flagquantum.compiler.basis_conversion import BasisConversionError, convert_basis
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

#: The Clifford+T basis the Toffoli rule lands in. ``tdg`` is named because the
#: identity uses it three times: a basis of ``h``/``t``/``cx`` alone is refused,
#: which `test_the_clifford_t_basis_the_identity_needs_is_named` pins.
CLIFFORD_T_BASIS = ("cx", "h", "s", "sdg", "t", "tdg")

#: Widths whose dense unitary the tests can hold: ``2 ** (2 n + 2)`` entries per
#: column, so ``n = 4`` is 1024 columns of 1024 and the run stays under a second.
CHECKED_WIDTHS = (1, 2, 3, 4)

#: One Toffoli is seven T through the compiler's own identity, each bit costs two
#: of them -- one in ``MAJ``, one in its inverse -- and six ``cx``: two in each
#: gadget and the two the sum injection adds. The identity replaces one operation
#: with fifteen, so it adds fourteen per Toffoli.
_TOFFOLI_T_COUNT = 7
_TOFFOLI_T_FORM_GATES = 15
_TOFFOLIS_PER_BIT = 2
_CX_PER_BIT = 6


def _bits(label: int, width: int) -> list[int]:
    """Return ``label`` as ``width`` bits, wire 0 first."""
    return [(label >> (width - 1 - index)) & 1 for index in range(width)]


def _label(bits: list[int]) -> int:
    """Return the basis-state label of a wire-first bit list."""
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def _registers(bits: list[int], n_bits: int) -> tuple[int, int, int]:
    """Return the two addends and the carry-in bit of a basis state."""
    a = _label(bits[:n_bits])
    b = _label(bits[n_bits : 2 * n_bits])
    return a, b, bits[2 * n_bits]


def _sum_bits(bits: list[int], n_bits: int) -> list[int]:
    """Return the basis state an ``n_bits``-wide addition sends ``bits`` to.

    The contract the module documents: the first addend comes back unchanged, the
    second holds the sum modulo ``2 ** n_bits``, the working carry wire is
    restored to ``|0>``, and the carry-out wire holds the carry out of the top
    position. Wire ``n_bits + index`` carries bit weight
    ``2 ** (n_bits - 1 - index)``, because wire 0 is the most significant bit of
    its register. The carry-in wire must enter at zero; the caller checks that
    before calling this.
    """
    a, b, _ = _registers(bits, n_bits)
    total = a + b
    result = list(bits)
    for index in range(n_bits):
        result[n_bits + index] = (total >> (n_bits - 1 - index)) & 1
    result[2 * n_bits] = 0
    result[2 * n_bits + 1] = (total >> n_bits) & 1
    return result


def _unitary(n_bits: int) -> torch.Tensor:
    """Return the adder's unitary at ``n_bits``, in complex128.

    The dense matrix is exact to the kernel's precision, so a permutation entry is
    compared against one rather than against a tolerance that would also accept a
    small amplitude leaking into a second column.
    """
    return get_unitary(adder_circuit(n_bits), dtype=torch.complex128)


def _image(unitary: torch.Tensor, basis: int, width: int) -> list[int]:
    """Return the basis state ``basis`` reaches, and assert it reaches exactly one.

    A column with no unit entry, or with its weight split across two entries,
    would make every property below meaningless, so the caller gets the image only
    once that has been checked here.
    """
    column = unitary[:, basis].abs()
    target = int(column.argmax())
    assert complex(unitary[target, basis]) == pytest.approx(
        1.0, abs=1e-12
    ), f"basis {basis} does not reach a single basis state: {complex(column[target])}"
    assert float(column.sum()) == pytest.approx(1.0, abs=1e-12)
    return _bits(target, width)


def _inside_domain(bits: list[int], n_bits: int) -> bool:
    """Whether a basis state meets the adder's ancilla contract on entry."""
    return not bits[2 * n_bits] and not bits[2 * n_bits + 1]


@pytest.mark.parametrize("n_bits", CHECKED_WIDTHS)
def test_every_input_inside_the_domain_sums_exactly(n_bits: int) -> None:
    """Each ancilla-clean basis state goes to the basis state holding the sum.

    The three claims a caller composes an adder for are asserted separately and on
    the same actual image, so a regression fails naming the wire rather than as a
    wrong number: the first addend is unchanged, the working wire is clean again,
    and the carry-out wire is the top bit of the exact integer sum.
    """
    width = 2 * n_bits + 2
    unitary = _unitary(n_bits)

    checked = 0
    for basis in range(2**width):
        bits = _bits(basis, width)
        if not _inside_domain(bits, n_bits):
            continue
        a, b, _ = _registers(bits, n_bits)
        image = _image(unitary, basis, width)
        expected = _sum_bits(bits, n_bits)

        assert image == expected, (
            f"n_bits={n_bits}: {a} + {b} on basis {basis} ({bits}) reached "
            f"{image} ({_label(image)}), expected {expected} ({_label(expected)})"
        )
        assert (
            image[:n_bits] == bits[:n_bits]
        ), f"n_bits={n_bits}: the first addend moved on basis {basis}"
        assert (
            image[2 * n_bits] == 0
        ), f"n_bits={n_bits}: the working carry wire was not restored on {basis}"
        assert image[2 * n_bits + 1] == (
            1 if a + b >= 2**n_bits else 0
        ), f"n_bits={n_bits}: {a} + {b} reported the wrong carry out"
        checked += 1
    # One basis state per pair of addends, over all 2 ** (2 n_bits) pairs.
    assert checked == 2 ** (2 * n_bits)


@pytest.mark.parametrize("n_bits", CHECKED_WIDTHS)
def test_the_map_is_a_permutation_of_the_whole_space(n_bits: int) -> None:
    """The circuit is reversible even on inputs it does not promise to sum.

    Every column of the unitary has exactly one unit entry and so does every row,
    so the map is a permutation of all ``2 ** (2 n_bits + 2)`` basis states rather
    than only of the ancilla-clean ones. What it does not do off that domain is
    compute the sum: the carry wire is a genuine input there, and the last
    assertion says the map is reversible rather than that it still adds. Without
    it the domain restriction in the docstring would read as a remark.
    """
    width = 2 * n_bits + 2
    unitary = _unitary(n_bits)
    magnitudes = unitary.abs()
    ones = torch.ones(2**width, dtype=magnitudes.dtype)

    assert torch.allclose(magnitudes.max(dim=0).values, ones)
    assert torch.allclose(magnitudes.max(dim=1).values, ones)
    assert int((magnitudes > 1e-9).sum()) == 2**width

    off_domain = [
        basis
        for basis in range(2**width)
        if not _inside_domain(_bits(basis, width), n_bits)
        and _image(unitary, basis, width) != _sum_bits(_bits(basis, width), n_bits)
    ]
    assert off_domain, (
        "no dirty-ancilla input leaves the sum, so the domain restriction is "
        "either unnecessary or untested"
    )


@pytest.mark.parametrize("n_bits", CHECKED_WIDTHS)
def test_the_construction_costs_eight_operations_per_bit(n_bits: int) -> None:
    """``2 n`` Toffolis and ``6 n + 1`` ``cx``, which is ``8 n + 1`` operations.

    The count is a property of the gadget sequence, so it is read from the program
    the module built rather than from a table beside it: a change to the
    construction that left the sum right and the count different would be a
    different circuit at the same cost claim, and the module states the count. The
    opcodes are checked as a set as well, because the construction carries no
    angle and the presence of one would be a phase the module does not document.
    """
    circuit = adder_circuit(n_bits)
    names = [instruction.name for instruction in circuit.to_ir().instructions]

    assert circuit.n_wires == 2 * n_bits + 2
    assert names.count("ccx") == _TOFFOLIS_PER_BIT * n_bits
    assert names.count("cx") == _CX_PER_BIT * n_bits + 1
    assert len(names) == (_TOFFOLIS_PER_BIT + _CX_PER_BIT) * n_bits + 1
    assert set(names) == {"ccx", "cx"}


@pytest.mark.parametrize("n_bits", CHECKED_WIDTHS)
def test_the_register_map_covers_every_wire_exactly_once(n_bits: int) -> None:
    """The map is a partition of the circuit's wires, so no wire is unaccounted."""
    wires = adder_wires(n_bits)
    circuit_wires = set(range(adder_circuit(n_bits).n_wires))

    assert isinstance(wires, AdderWires)
    assert wires.n_bits == n_bits
    assert wires.n_wires == 2 * n_bits + 2
    assert len(wires.a) == len(wires.b) == n_bits
    assert set(wires.a) | set(wires.b) | {wires.carry, wires.carry_out} == circuit_wires
    # Most significant bit first in both registers, with the two ancillas above
    # them: the ordering the module documents rather than the one it happens to
    # have.
    assert wires.a == tuple(range(n_bits))
    assert wires.b == tuple(range(n_bits, 2 * n_bits))
    assert wires.carry == 2 * n_bits
    assert wires.carry_out == 2 * n_bits + 1


@pytest.mark.parametrize("n_bits", CHECKED_WIDTHS)
def test_the_ripple_runs_on_the_working_wire(n_bits: int) -> None:
    """How much work each wire does is what separates the two ancilla roles.

    The map names wire ``2 n`` the working carry and wire ``2 n + 1`` the carry
    out, and the ancilla contract by itself does not tell them apart: a circuit
    that rippled on the carry-out wire and never touched the working one would
    satisfy both sentences, because an untouched wire is trivially returned to
    ``|0>``. What separates them is the count. Per bit the working wire is reached
    by three operations in the forward gadget, three in the inverse one, and one
    sum injection, plus the single ``cx`` that hands the carry out of the top
    position across -- ``7 n + 1`` appearances, of which ``2 n`` are the gadget's
    own Toffoli targets -- while the carry-out wire is written exactly once. The
    same asymmetry puts the sum in the second register: an ``a`` wire is reached
    five times and a ``b`` wire six, twice and four times as a target.
    """
    circuit = adder_circuit(n_bits)
    wires = adder_wires(n_bits)
    reached: Counter[int] = Counter()
    targeted: Counter[int] = Counter()
    for instruction in circuit.to_ir().instructions:
        for wire in instruction.wires:
            reached[wire] += 1
        targeted[instruction.wires[-1]] += 1

    assert reached[wires.carry] == 7 * n_bits + 1
    assert targeted[wires.carry] == 2 * n_bits
    assert reached[wires.carry_out] == 1
    assert targeted[wires.carry_out] == 1
    for a_wire in wires.a:
        assert reached[a_wire] == 5
        assert targeted[a_wire] == 2
    for b_wire in wires.b:
        assert reached[b_wire] == 6
        assert targeted[b_wire] == 4


@pytest.mark.parametrize(
    "width, message, error",
    [
        (0, "at least 1", ValueError),
        (-1, "at least 1", ValueError),
        (2.0, "must be an integer", TypeError),
        ("2", "must be an integer", TypeError),
        (True, "must be an integer", TypeError),
        (None, "must be an integer", TypeError),
    ],
)
def test_a_width_that_is_not_a_positive_integer_is_refused(
    width: object, message: str, error: type[Exception]
) -> None:
    """Both entry points refuse a width that is not a positive integer."""
    for entry in (adder_circuit, adder_wires):
        with pytest.raises(error, match=message):
            entry(width)


def test_the_clifford_t_basis_the_identity_needs_is_named() -> None:
    """A basis without ``tdg`` is refused by name rather than silently widened.

    The Toffoli identity uses ``tdg`` three times, so a caller who asks for
    ``h``/``t``/``cx`` is asking for a basis this identity cannot land in. The
    refusal has to name the missing gate, because a conversion that quietly added
    it would report a program in a basis the caller did not name.
    """
    with pytest.raises(BasisConversionError, match="tdg"):
        convert_basis(adder_circuit(1), gates=("cx", "h", "t"))


def test_the_t_cost_is_the_compilers_own_toffoli_rule() -> None:
    """The adder's T-count is ``14 n``, and it comes from the compiler.

    One Toffoli is seven T through ``flagquantum/compiler/basis_translation.py``'s
    fifteen-gate identity and the adder contains ``2 n`` of them, so the cost is
    ``14 n`` T and nothing else contributes a T. Reading the seven from the
    compiler's own lowering rather than repeating the identity here is what keeps
    this number from drifting away from the one a user gets who converts a Toffoli
    by hand.
    """
    one_toffoli = estimate_resources(
        convert_basis(Circuit(3).ccx(0, 1, 2), gates=CLIFFORD_T_BASIS).program
    )
    assert one_toffoli.t_count == _TOFFOLI_T_COUNT
    assert one_toffoli.operation_counts == {"h": 2, "cx": 6, "tdg": 3, "t": 4}

    for n_bits in (1, 2, 3, 4, 5):
        lowered = convert_basis(adder_circuit(n_bits), gates=CLIFFORD_T_BASIS)
        estimate = estimate_resources(lowered.program)
        assert estimate.t_count == _TOFFOLI_T_COUNT * _TOFFOLIS_PER_BIT * n_bits
        # The tally is read off this program and not off the adder's gate list:
        # the compiler's ``t`` and ``tdg`` counts are the ones the identity wrote,
        # and neither appears in the adder.
        assert (
            estimate.operation_counts["t"] + estimate.operation_counts["tdg"]
            == estimate.t_count
        )


def test_the_logical_report_repeats_the_compilers_tally() -> None:
    """The surface-code report prices the adder without a second opinion.

    ``algorithms.logical_resources`` is defined to read
    ``compiler.resource_estimation`` rather than recount, and this is that
    agreement on a program whose gate count comes from somewhere else.
    """
    n_bits = 2
    distance = 5
    lowered = convert_basis(adder_circuit(n_bits), gates=CLIFFORD_T_BASIS)
    estimate = estimate_resources(lowered.program)
    report = estimate_logical_resources(lowered.program, distance=distance)

    assert report.t_count == estimate.t_count
    assert report.clifford_count == estimate.n_operations - estimate.t_count
    assert report.logical_depth == estimate.depth
    assert report.n_qubits == 2 * n_bits + 2
    # 2 d^2 - 1 physical qubits per patch, one patch per wire, and no other code
    # model in play: the footprint is the product the unit documents.
    assert report.physical_qubits == (2 * distance**2 - 1) * (2 * n_bits + 2)
    assert report.surface_code_cycles == distance * report.logical_depth


def test_the_default_operation_bound_is_reported_rather_than_hidden() -> None:
    """The conversion's added-operation bound stops the adder at nine bits.

    ``convert_basis`` bounds how many operations a lowering may add and the default
    is 256, which a ``28 n`` expansion reaches at ``n = 9``. The refusal is a bound
    the caller can raise rather than a limit of the construction, so the test shows
    both halves: ten bits is refused by name at the default, and ten bits converts
    once the bound is raised.
    """
    added_per_bit = _TOFFOLIS_PER_BIT * (_TOFFOLI_T_FORM_GATES - 1)
    widened = convert_basis(adder_circuit(9), gates=CLIFFORD_T_BASIS)
    assert len(widened.program.instructions) == 8 * 9 + 1 + added_per_bit * 9

    with pytest.raises(BasisConversionError, match="max_added_operations"):
        convert_basis(adder_circuit(10), gates=CLIFFORD_T_BASIS)

    raised = convert_basis(
        adder_circuit(10), gates=CLIFFORD_T_BASIS, max_added_operations=1024
    )
    assert estimate_resources(raised.program).t_count == _TOFFOLI_T_COUNT * 2 * 10
