"""Check flat record storage, composition and named field readout contracts."""

from dataclasses import replace
from math import sin

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


def test_record_fields_share_storage_and_preserve_named_readout():
    source = """qstruct State {flags:qbit[2]; tag:qbit;}
      qfunc flip(s:State) {X(s.flags[0]); S(s.tag);}
      qfunc main(output s:State) {
        allocate(s.size,s); H(s.flags[1]); X(s.tag); flip(s);
      }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(3, dtype=DTYPE).h(1).x(0).x(2).s(0)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert dict(program.registers) == {"s.flags": (1, 2), "s.tag": (0,)}
    assert dict(
        program.probabilities(result, registers=("s.flags", "s.tag"))
    ) == pytest.approx({(1, 1): 0.5, (3, 1): 0.5})
    assert program.report["qmod_records"] == (
        ("s", "State", (("flags", "qbit[]", 2), ("tag", "qbit", 1))),
    )
    assert program.report["workspace_qubits"] == 0


def test_record_output_helper_and_whole_field_hadamard():
    source = """qstruct State {flags:qbit[2];}
      qfunc prepare(output s:State) {allocate(s); hadamard_transform(s.flags);}
      qfunc main(output s:State) {prepare(s);}"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(2, dtype=DTYPE).h(0).h(1)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_numeric_record_words_preserve_shape_storage_and_readout():
    source = """qstruct State {words:qnum<2>[2]; tag:qbit;}
      qfunc prepare(output s:State) {
        allocate(s.size,s);
        repeat(i:s.words.len) {hadamard_transform(s.words[i]);}
        X(s.tag);
      }
      qfunc kick(s:State) {phase(s.words[0]+2*s.words[1]+s.tag,0.37);}
      qfunc main(output s:State) {prepare(s);kick(s);hadamard_transform(s.words);}"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(5, dtype=DTYPE).x(0)
    for wire, weight in ((4, 1), (3, 2), (2, 2), (1, 4)):
        reference.h(wire).phase(wire, theta=0.37 * weight).h(wire)
    reference.phase(0, theta=0.37)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert dict(program.registers) == {
        "s.words[0]": (3, 4),
        "s.words[1]": (1, 2),
        "s.tag": (0,),
    }
    expected = {}
    for first in range(4):
        for second in range(4):
            probability = 1.0
            for value, angles in ((first, (0.37, 0.74)), (second, (0.74, 1.48))):
                for bit, angle in enumerate(angles):
                    one = sin(angle / 2) ** 2
                    probability *= one if value & (1 << bit) else 1 - one
            expected[first, second, 1] = probability
    assert dict(
        program.probabilities(result, registers=tuple(program.registers))
    ) == pytest.approx(expected)
    assert program.report["qmod_records"] == (
        ("s", "State", (("words", "qnum[]", 2, 2), ("tag", "qbit", 1))),
    )
    assert program.report["resources"].n_qubits == 5
    assert program.report["workspace_qubits"] == 0
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(source, source_format="qmod", max_qubits=4)


def test_matrix_record_cells_preserve_row_order_and_bit_significance():
    source = """qstruct State {tag:qbit; grid:qbit[3][2]; tail:qnum<2>;}
      qfunc prepare(output s:State) {
        allocate(s); X(s.tag); H(s.grid[0][0]);
        X(s.grid[0][2]); X(s.grid[1][1]);
      }
      qfunc kick(s:State) {phase(s.grid[0][0]+2*s.grid[1][1]+s.tag,0.37);}
      qfunc main(output s:State) {prepare(s);kick(s);H(s.grid[0][0]);}"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(9, dtype=DTYPE).x(8).h(7).x(5).x(3)
    reference.phase(7, theta=0.37).phase(3, theta=0.74).phase(8, theta=0.37).h(7)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert dict(program.registers) == {
        "s.tag": (8,),
        "s.grid[0]": (5, 6, 7),
        "s.grid[1]": (2, 3, 4),
        "s.tail": (0, 1),
    }
    one = sin(0.37 / 2) ** 2
    assert dict(
        program.probabilities(result, registers=tuple(program.registers))
    ) == pytest.approx({(1, 4, 2, 0): 1 - one, (1, 5, 2, 0): one})
    assert program.report["qmod_records"] == (
        (
            "s",
            "State",
            (("tag", "qbit", 1), ("grid", "qbit[][]", 3, 2), ("tail", "qnum", 2)),
        ),
    )
    assert program.report["resources"].n_qubits == 9
    assert program.report["workspace_qubits"] == 0
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(source, source_format="qmod", max_qubits=8)


def test_matrix_row_views_lengths_and_writable_helper_share_storage():
    source = """qstruct State {grid:qbit[3][2];}
      qfunc flip(row:qbit[]) {repeat(j:row.len) {X(row[j]);}}
      qfunc main(output s:State) {
        allocate(s.size,s);
        repeat(i:s.grid.len) {X(s.grid[i][0]);}
        repeat(j:s.grid[0].len) {X(s.grid[0][j]);}
        flip(s.grid[1]);
        hadamard_transform(s.grid[0]); hadamard_transform(s.grid[0]);
      }"""
    program = synth.synthesize(source, source_format="qmod")
    result = fq.run(program)
    assert dict(
        program.probabilities(result, registers=tuple(program.registers))
    ) == pytest.approx({(6, 6): 1.0})
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        ("qstruct R {} qfunc main(output r:R) {allocate(r);}", "at least one field"),
        ("qstruct R {a:qbit;} qfunc helper(", "expected an Output"),
        ("qstruct R {a:qbit;a:qbit;} qfunc main(output r:R) {}", "unique"),
        ("qstruct R {size:qbit;} qfunc main(output r:R) {}", "shadow size"),
        ("qstruct R {a:qbit[256];b:qbit;} qfunc main(output r:R) {}", "exceeds 256"),
        ("qstruct R {a:qnum;} qfunc main(output r:R) {}", "fixed width"),
        (
            "qstruct R {a:qnum<2,True,0>;} qfunc main(output r:R) {}",
            "return every initialized register",
        ),
        ("qstruct R {a:qnum<2,False,1>;} qfunc main(output r:R) {}", "fractional"),
        ("qstruct R {a:qnum<8>[33];} qfunc main(output r:R) {}", "exceeds 256"),
        (
            "qstruct R {a:qnum<2>[2][2];} qfunc main(output r:R) {}",
            "one array dimension",
        ),
        (
            "qstruct R {a:qnum<2>[2];} qfunc main(output r:R) {allocate(r);phase(r.a[2],1.0);}",
            "field index",
        ),
        (
            "qstruct R {a:qnum<2>[2];} qfunc main(output r:R) {allocate(r);phase(r.a,1.0);}",
            "element index",
        ),
        (
            "qstruct R {a:qnum<1>;} qfunc main(output r:R) {allocate(r);X(r.a);}",
            "scalar field",
        ),
        (
            "qstruct R {a:qnum<2>[2];} qfunc main(output r:R) {allocate(r);X(r.a[0]);}",
            "array field",
        ),
        (
            "qstruct R {a:qbit[2][2][2];} qfunc main(output r:R) {}",
            "two array dimensions",
        ),
        ("qstruct R {a:qbit[16][17];} qfunc main(output r:R) {}", "exceeds 256"),
        (
            "qstruct R {a:qbit[3][2];} qfunc main(output r:R) {allocate(r);X(r.a[2][0]);}",
            "row index",
        ),
        (
            "qstruct R {a:qbit[3][2];} qfunc main(output r:R) {allocate(r);X(r.a[0][3]);}",
            "column index",
        ),
        (
            "qstruct R {a:qbit[3][2];} qfunc main(output r:R) {allocate(r);phase(r.a[2][0],1.0);}",
            "row index",
        ),
        (
            "qstruct R {a:qbit[3][2];} qfunc main(output r:R) {allocate(r);phase(r.a[0][3],1.0);}",
            "column index",
        ),
        (
            "qstruct R {a:qbit[3][2];} qfunc main(output r:R) {allocate(r);phase(r.a[0],1.0);}",
            "row and column",
        ),
        (
            "qstruct R {a:qbit[3][2];} qfunc main(output r:R) {allocate(r);hadamard_transform(r.a[2]);}",
            "field index",
        ),
        (
            "qstruct R {a:qnum<2>[2];} qfunc main(output r:R) {allocate(r);X(r.a[0][0]);}",
            "two-dimensional bit",
        ),
        (
            "qstruct R {a:qbit;} qfunc main(output r:R) {allocate(r);X(r.b);}",
            "unknown record field",
        ),
        (
            "qstruct R {a:qbit[2];} qfunc main(output r:R) {allocate(r);X(r.a[2]);}",
            "field index",
        ),
        (
            "qstruct R {a:qbit;} qfunc main(output r:R) {X(r.a);allocate(r);}",
            "uninitialized",
        ),
        (
            "qstruct R {a:qbit;} qfunc main(output r:R) {allocate(2,r);}",
            "declared width",
        ),
    ],
)
def test_record_admission_rejections(source, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


def test_equal_width_record_types_are_not_interchangeable():
    source = """qstruct R {a:qbit;} qstruct S {b:qbit;}
      qfunc flip(r:R) {X(r.a);}
      qfunc main(output s:S) {allocate(s);flip(s);}"""
    with pytest.raises(InteropConversionError, match="matching record type"):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize(
    "fields",
    [
        (("a", "qbit", 2),),
        (("a", "qbit[]", 1),),
        (("a", "qbit", 1), ("a", "qbit", 1)),
        (("a", "qbit", 1), ("a.b", "qbit", 1)),
        (("a..b", "qbit", 1), ("c", "qbit", 1)),
        (("a.size", "qbit", 1), ("c", "qbit", 1)),
        (("a", "qbit", True), ("b", "qbit", 1)),
        (("a", "qnum[]", 1, True), ("b", "qnum", 1)),
        (("a", "qnum[]", 2),),
        (("a", "qnum", 1, 2),),
        (("a", "qnum[]", 2, 257),),
        (("a", "qbit[][]", 2),),
        (("a", "qbit[][]", 1, False),),
        (("a", "qbit[][]", 1, 0),),
    ],
)
def test_compiler_rechecks_owned_record_layout(fields):
    imported = parse_qmod(
        "qstruct R {a:qbit;b:qbit;} qfunc main(output r:R) {allocate(r);}",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    block = imported.program.body.blocks[0]
    declaration = block.operations[0]
    broken = replace(
        declaration, attributes={**declaration.attributes, "fields": fields}
    )
    program = replace(
        imported.program,
        body=replace(
            imported.program.body,
            blocks=(replace(block, operations=(broken, *block.operations[1:])),),
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.record"):
        verify_register_program(program)


@pytest.mark.parametrize(
    "name",
    [
        "ising_model",
        "kidney_exchange_problem",
        "portfolio_optimization",
        "integer_linear_programming",
        "learning_optimization",
        "combi_workshop_Inequality_constriants_PO",
        "combi_workshop_equality_constriants_PO",
    ],
)
@pytest.mark.parametrize("scale", [0.003, -0.005])
def test_small_original_records_against_independent_cost(name, scale):
    import hashlib

    from tools.validate_qmod_records import BLOBS, FIXTURES, RecordCost
    from tools.validate_qmod_sparse_phase import independent_terms, reference_circuit

    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == BLOBS[name]
    oracle = RecordCost(raw.decode())
    params = [
        scale * (index + 1) * (-1 if index % 2 else 1) for index in range(oracle.count)
    ]
    program = synth.synthesize(source, classical_inputs={"params": params})
    terms = independent_terms(oracle, oracle.degree, oracle.width)
    midpoint = oracle.count // 2
    interleaved = (
        params
        if oracle.interleaved
        else [
            value
            for pair in zip(params[:midpoint], params[midpoint:], strict=True)
            for value in pair
        ]
    )
    reference = reference_circuit(interleaved, terms, oracle.width)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    distribution = program.probabilities(result, registers=tuple(program.registers))
    assert sum(distribution.values()) == pytest.approx(1)


@pytest.mark.parametrize(
    ("name", "columns", "rows", "width"),
    [
        ("electric_grid_optimization", 4, 3, 18),
        ("max_induced_k_color_subgraph", 6, 2, 12),
        ("min_graph_coloring", 5, 3, 15),
    ],
)
def test_original_matrix_records_keep_shape_and_resource_budget(
    name, columns, rows, width
):
    import hashlib

    from tools.validate_qmod_records import BLOBS, FIXTURES, RecordCost

    source = FIXTURES / f"{name}.qmod"
    raw = source.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == BLOBS[name]
    count = RecordCost(raw.decode()).count
    bindings = {"params": [0.003 * (i + 1) for i in range(count)]}
    program = synth.synthesize(source, classical_inputs=bindings)
    assert program.report["resources"].n_qubits == width
    assert program.report["workspace_qubits"] == 0
    assert program.report["qmod_records"][0][2][0] == ("x", "qbit[][]", columns, rows)
    assert sorted(
        wire for ports in program.registers.values() for wire in ports
    ) == list(range(width))
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(source, classical_inputs=bindings, max_qubits=width - 1)


def test_nested_nominal_views_and_qualified_leaves_match_circuit():
    source = """qstruct Packet {data:Payload; ready:qbit;}
    qstruct Payload {prefix:qnum<2>; body:Body;}
    qstruct Body {flags:qbit[3]; tag:qbit;}
    qfunc prepare(output p:Packet) {allocate(p.size,p);}
    qfunc inner(b:Body) {H(b.flags[0]); CX(b.flags[0],b.tag);}
    qfunc flip(q:qbit[]) {repeat(i:q.len) {X(q[i]);}}
    qfunc main(output p:Packet) {
      prepare(p); inner(p.data.body); flip(p.data.body.flags[1:3]);
      if (p.data.body.size == 4) {X(p.ready);}
    }"""
    program = synth.synthesize(source, source_format="qmod")
    reference = fq.Circuit(7, dtype=DTYPE).h(4).cnot(4, 1).x(3).x(2).x(0)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert dict(program.registers) == {
        "p.data.prefix": (5, 6),
        "p.data.body.flags": (2, 3, 4),
        "p.data.body.tag": (1,),
        "p.ready": (0,),
    }
    assert dict(program.probabilities(result, registers=tuple(program.registers))) == (
        pytest.approx({(0, 6, 0, 1): 0.5, (0, 7, 1, 1): 0.5})
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    ("definitions", "body", "diagnostic"),
    [
        ("qstruct R {r:R;}", "allocate(r);", "acyclic"),
        ("qstruct R {v:Other;} qstruct Other {r:R;}", "allocate(r);", "acyclic"),
        ("qstruct R {v:Missing;}", "allocate(r);", "unknown field record"),
        (
            "qstruct R {v:Wide; tag:qbit;} qstruct Wide {bits:qbit[256];}",
            "allocate(r);",
            "exceeds 256 bits",
        ),
        (
            "qstruct R {v:Left;} qstruct Left {b:qbit;} qstruct Right {b:qbit;}"
            " qfunc flip(x:Right) {X(x.b);}",
            "allocate(r); flip(r.v);",
            "matching record type",
        ),
        (
            "qstruct R {v:Left;} qstruct Left {b:qbit[2];}"
            " qfunc flip(x:qbit[]) {H(x[0]);}",
            "allocate(r); flip(r.v.b[1:3]);",
            "out of bounds",
        ),
        (
            "qstruct R {v:Left;} qstruct Left {b:qbit;}"
            " qfunc flip(x:Left,y:qbit) {X(x.b); H(y);}",
            "allocate(r); flip(r.v,r.v.b);",
            "must be disjoint",
        ),
    ],
)
def test_nested_record_rejects_invalid_layout_and_views(definitions, body, diagnostic):
    source = definitions + f" qfunc main(output r:R) {{{body}}}"
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(source, source_format="qmod")


@pytest.mark.parametrize("lower,excluded", [(0, 2), (1, 3)])
def test_nested_field_increment_and_bound_predicate_x(
    lower: int, excluded: int
) -> None:
    source = """qstruct State {velocity:qnum<2>; direction:qbit; pos:qnum<2>;}
      qfunc stream(s:State) {
        control(s.velocity == 2) {
          control(s.direction) {s.pos += 1;} else {s.pos += -1;}
        }
      }
      qfunc reflect(velocity:qnum<2>, pos:qnum<2>, direction:qbit, limits:real[]) {
        control(velocity == 2 and pos >= limits[0] and not (pos == limits[1])) {
          X(direction);
        }
      }
      qfunc main(output s:State) {
        allocate(s); hadamard_transform(s.velocity); H(s.direction);
        hadamard_transform(s.pos); phase(s.pos, 0.31); S(s.direction);
        stream(s); reflect(s.velocity, s.pos, s.direction, [LOWER, EXCLUDED]);
      }""".replace("LOWER", str(lower)).replace("EXCLUDED", str(excluded))
    program = synth.synthesize(source, source_format="qmod")
    assert dict(program.registers) == {
        "s.velocity": (3, 4),
        "s.direction": (2,),
        "s.pos": (0, 1),
    }

    permutation = torch.zeros((32, 32), dtype=DTYPE)
    for index in range(32):
        position, remainder = divmod(index, 8)
        direction, velocity = divmod(remainder, 4)
        if velocity == 2:
            position = (position + (1 if direction else -1)) % 4
            if position >= lower and position != excluded:
                direction ^= 1
        permutation[8 * position + 4 * direction + velocity, index] = 1
    reference = fq.Circuit(5, dtype=DTYPE)
    for qubit in range(5):
        reference.h(qubit)
    reference.phase(0, theta=0.62).phase(1, theta=0.31).s(2)
    reference.any(0, 1, 2, 3, 4, matrix=permutation)
    actual = fq.run(program).statevector()
    expected = fq.run(reference).statevector()
    workspace = program.report["workspace_qubits"]
    assert tuple(actual.shape) == (1, 1 << (5 + workspace))
    actual = actual.reshape(1, 32, 1 << workspace)
    torch.testing.assert_close(actual[:, :, 0], expected, atol=1e-12, rtol=0)
    assert actual[:, :, 1:].abs().sum().item() < 1e-12


def test_nested_field_increment_and_bound_predicate_reject_control_aliases() -> None:
    nested = """qstruct State {value:qnum<2>; flag:qbit;}
      qfunc apply_update(s:State) {
        control(s.value == 1) {control(s.flag) {s.value += 1;}}
      }
      qfunc main(output s:State) {allocate(s); apply_update(s);}"""
    with pytest.raises(
        InteropConversionError, match="controls and target must be disjoint"
    ):
        synth.synthesize(nested, source_format="qmod")

    alias = """qstruct State {value:qnum<2>; flag:qbit;}
      qfunc toggle(value:qnum<2>, flag:qbit) {
        control(value <= 2 and flag) {X(flag);}
      }
      qfunc main(output s:State) {allocate(s); toggle(s.value, s.flag);}"""
    with pytest.raises(
        InteropConversionError, match="predicate cannot read its XOR target"
    ):
        synth.synthesize(alias, source_format="qmod")


def test_fully_fractional_record_field_keeps_bit_order_and_scale() -> None:
    source = """qstruct Coordinate {x:qnum<3, UNSIGNED, 3>; flag:qbit;}
      qstruct State {coord:Coordinate; tail:qbit;}
      qfunc mark(bits:qbit[3], flag:qbit) {
        X(bits[0]); CX(bits[0], flag); CRY(0.41, bits[0], bits[2]);
      }
      qfunc main(output s:State) {
        allocate(s); mark(s.coord.x, s.coord.flag); X(s.tail);
      }"""
    program = synth.synthesize(source, source_format="qmod", max_workspace=0)
    assert program.report["qmod_fractional_bits"] == {"s.coord.x": 3}
    assert dict(program.registers) == {
        "s.coord.x": (2, 3, 4),
        "s.coord.flag": (1,),
        "s.tail": (0,),
    }
    expected = fq.Circuit(5, dtype=DTYPE).x(4).cx(4, 1).cry(4, 2, theta=0.41).x(0)
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(expected).statevector(), atol=1e-12, rtol=0
    )
    assert dict(
        program.probabilities(result, registers=("s.coord.x",))
    ) == pytest.approx({(1,): 1 - sin(0.41 / 2) ** 2, (5,): sin(0.41 / 2) ** 2})


@pytest.mark.parametrize("numeric_type", ["qnum", "qnum<3, UNSIGNED, 3>"])
def test_partition_preserves_complete_fractional_field(numeric_type: str) -> None:
    source = """qstruct Coordinate {x:qnum<3, UNSIGNED, 3>; flag:qbit;}
      qfunc main(output x:OUTPUT_TYPE, output flag:qbit) {
        s:Coordinate;
        allocate(s); hadamard_transform(s.x); X(s.flag); s -> {x, flag};
      }""".replace("OUTPUT_TYPE", numeric_type)
    program = synth.synthesize(source, source_format="qmod")
    assert program.report["qmod_fractional_bits"] == {"x": 3}
    assert dict(program.registers) == {"x": (0, 1, 2), "flag": (3,)}
    expected = fq.Circuit(4, dtype=DTYPE).h(0).h(1).h(2).x(3)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(expected).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "signature,publication",
    [
        ("output x:qnum<3>, output flag:qbit", "s -> {x, flag};"),
        ("output x:qnum<2>, output rest:qnum<2>", "s -> {x, rest};"),
        ("output bits:qbit[]", "s -> bits;"),
    ],
)
def test_fractional_record_publication_refuses_reinterpretation(
    signature: str, publication: str
) -> None:
    source = f"""qstruct Coordinate {{x:qnum<3, UNSIGNED, 3>; flag:qbit;}}
      qfunc main({signature}) {{s:Coordinate; allocate(s); {publication}}}"""
    with pytest.raises(InteropConversionError, match="publish fractional fields whole"):
        synth.synthesize(source, source_format="qmod")


def test_fractional_record_borrowing_cannot_write_through_const() -> None:
    source = """qstruct Coordinate {x:qnum<3, UNSIGNED, 3>; flag:qbit;}
      qfunc flip(bits:qbit[3]) {X(bits[0]);}
      qfunc violate(const s:Coordinate) {flip(s.x);}
      qfunc main(output s:Coordinate) {allocate(s); violate(s);}"""
    with pytest.raises(InteropConversionError, match="no writes or Output escape"):
        synth.synthesize(source, source_format="qmod")
