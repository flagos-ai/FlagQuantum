import flagquantum as fq


def test_draw_accepts_native_circuit_and_ir():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    circuit_text = fq.draw(circuit)
    ir_text = fq.draw(circuit.to_ir())

    assert circuit_text == ir_text
    assert "H" in circuit_text
    assert "X" in circuit_text


def test_circuit_draw_uses_unified_drawer():
    circuit = fq.Circuit(2)
    circuit.rx(0, theta=0.25).cz(0, 1)

    text = circuit.draw(decimals=2)

    assert text == fq.draw(circuit.to_ir(), decimals=2)
    assert "RX(0.25)" in text
    assert "Z" in text


def test_drawer_keeps_legacy_qdev_compatibility():
    class LegacyDevice:
        n_wires = 1
        op_history = [{"name_or_mat": "h", "wires": [0], "params": []}]

    assert "H" in fq.draw(LegacyDevice())
