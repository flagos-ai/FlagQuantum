"""Lexical binding scopes preserve source ownership, bit order and bounded gates."""

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.machine_learning.autoencoder.comparisons.circuit import (
    run_example,
)
from flagquantum import synth
from tools.validate_qmod_binding_scopes import validate

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


@pytest.mark.parametrize("case", [0, 1])
def test_unchanged_autoencoder_full_vector(case: int) -> None:
    assert validate(case)["validation"] == "passed"


def test_native_qfunc_original_and_circuit_example() -> None:
    run_example()


def scoped(
    actions: str,
    *,
    before: str = "allocate(a);allocate(b);",
    after: str = "",
    preparation: str = "{a,b}->x;",
    width: str = "",
) -> str:
    return f"qfunc main(output a:qbit[2],output b:qbit[1]) {{x:qbit[{width}]; {before} within {{{preparation}}} apply {{{actions}}} {after}}}"


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_static_loops_preserve_concat_order_and_restore_sources(steps: int) -> None:
    text = scoped(
        f"power({steps}) {{RY(0.31,x[2]);CX(x[0],x[2]);}}",
        before="allocate(a);allocate(b);H(a[0]);X(b[0]);",
        after="X(a[0]);",
    )
    p = synth.synthesize(text, source_format="qmod")
    c = fq.Circuit(3, dtype=DTYPE).h(1).x(2)
    for _ in range(steps):
        c.ry(2, theta=0.31).cx(1, 2)
    c.x(1)
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )
    assert p.report["workspace_qubits"] == 0


def test_nested_binding_scopes_restore_outer_alias() -> None:
    text = "qfunc main(output a:qbit[2],output b:qbit[1]) {x:qbit[];u:qbit[1];v:qbit[2];allocate(a);allocate(b);H(a[0]);within {{a,b}->x;} apply {within {x->{u,v};} apply {X(u[0]);CX(v[0],v[1]);} X(x[1]);} H(a[1]);}"
    p = synth.synthesize(text, source_format="qmod")
    c = fq.Circuit(3, dtype=DTYPE).h(1).x(1).cx(0, 2).x(0).h(0)
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("after", ["X(x[0]);", "hadamard_transform(x);"])
def test_temporary_alias_cannot_escape(after: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(scoped("X(x[0]);", after=after), source_format="qmod")


def test_source_name_is_suspended_inside_scope() -> None:
    with pytest.raises(ValueError, match="suspended"):
        synth.synthesize(scoped("X(a[0]);"), source_format="qmod")


@pytest.mark.parametrize("preparation", ["{a,a}->x;", "{a,b}->a;", "a->missing;"])
def test_invalid_binding_names_and_aliases_refuse(preparation: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            scoped("X(x[0]);", preparation=preparation), source_format="qmod"
        )


@pytest.mark.parametrize("width", ["2", "4"])
def test_bound_view_width_must_cover_source_exactly(width: str) -> None:
    with pytest.raises(ValueError, match="binding widths"):
        synth.synthesize(scoped("X(x[0]);", width=width), source_format="qmod")


@pytest.mark.parametrize("before", ["allocate(a);", "allocate(b);", ""])
def test_input_storage_requires_initialization_even_for_zero_iterations(
    before: str,
) -> None:
    with pytest.raises(ValueError, match="initializ"):
        synth.synthesize(
            scoped("power(0) {X(x[0]);}", before=before), source_format="qmod"
        )


@pytest.mark.parametrize(
    "action",
    [
        "allocate(fresh);",
        "prepare_amplitudes([1,0],0,fresh);",
        "swap_test(x[0:1],x[1:2],fresh);",
    ],
)
def test_apply_cannot_create_output_storage(action: str) -> None:
    text = f"qfunc main(output a:qbit[2],output b:qbit[1],output fresh:qbit) {{x:qbit[];allocate(a);allocate(b);within {{{{a,b}}->x;}} apply {{{action}}}}}"
    with pytest.raises(ValueError, match="binding.scope"):
        synth.synthesize(text, source_format="qmod")


def test_const_storage_is_not_writable_through_alias() -> None:
    text = "qfunc act(const a:qbit[2]) {x:qbit[];within {a->x;} apply {X(x[0]);}} qfunc main(output a:qbit[2]) {allocate(a);act(a);}"
    with pytest.raises(ValueError):
        synth.synthesize(text, source_format="qmod")


@pytest.mark.parametrize("count", [-1, 257])
def test_loop_bounds_still_refuse(count: int) -> None:
    with pytest.raises(ValueError, match="nonnegative|0..256"):
        synth.synthesize(scoped(f"power({count}) {{X(x[0]);}}"), source_format="qmod")


def test_nested_loop_index_cannot_shadow_outer_index() -> None:
    with pytest.raises(ValueError, match="shadow"):
        synth.synthesize(
            scoped("repeat(i:2) {repeat(i:2) {X(x[i]);}}"), source_format="qmod"
        )


def test_nested_loops_keep_effect_expansion_budget() -> None:
    with pytest.raises(ValueError, match="4096"):
        synth.synthesize(
            scoped("repeat(i:256) {repeat(j:256) {X(x[0]);}}"), source_format="qmod"
        )


def test_bound_helper_array_concat_includes_whole_words_and_static_bits() -> None:
    text = "qfunc act(q:qbit[]) {X(q[0]);CX(q[1],q[2]);} qfunc main(output a:qbit[2],output b:qbit[1]) {allocate(a);allocate(b);H(a[1]);act({b[0],a});}"
    p = synth.synthesize(text, source_format="qmod")
    c = fq.Circuit(3, dtype=DTYPE).h(0).x(2).cx(1, 0)
    torch.testing.assert_close(
        fq.run(p).statevector(), fq.run(c).statevector(), atol=1e-12, rtol=0
    )


def test_concat_does_not_hide_repeated_storage_in_helper_arguments() -> None:
    text = "qfunc act(q:qbit[]) {X(q[0]);} qfunc main(output a:qbit[2]) {allocate(a);act({a,a[0]});}"
    with pytest.raises(ValueError, match="disjoint"):
        synth.synthesize(text, source_format="qmod")


@pytest.mark.parametrize(
    "options",
    [
        {"max_qubits": 2},
        {"max_operations": 1},
        {"max_depth": 1},
        {"memory_limit_bytes": 16},
    ],
)
def test_physical_resource_limits_are_preserved(options: dict[str, int]) -> None:
    text = scoped(
        "RY(0.31,x[0]);CX(x[0],x[2]);", before="allocate(a);allocate(b);H(a[0]);"
    )
    with pytest.raises(ValueError):
        synth.synthesize(text, source_format="qmod", **options)


def test_static_loop_nesting_keeps_scope_depth_budget() -> None:
    body = "X(x[0]);"
    for index in range(33):
        body = f"repeat(i{index}:0) {{{body}}}"
    with pytest.raises(ValueError, match="scope nesting exceeds depth 32"):
        synth.synthesize(scoped(body), source_format="qmod")
