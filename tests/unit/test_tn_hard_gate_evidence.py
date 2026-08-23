from benchmarks.development.build_tn_hard_gate_evidence import build_payload


def test_tn_hard_gate_builder_rejects_missing_scaling_world_size(tmp_path):
    paths = []
    for world_size in (1, 2, 4, 4):
        path = tmp_path / f"{len(paths)}.json"
        path.write_text(
            (
                '{"world_size": %d, "max_execution_seconds": %f}'
                % (world_size, 100.0 / world_size)
            ),
            encoding="utf-8",
        )
        paths.append(path)
    multinode = tmp_path / "multinode.json"
    multinode.write_text('{"world_size": 16, "node_count": 2}', encoding="utf-8")
    rdma = tmp_path / "rdma.log"
    rdma.write_text(
        '{"collective_correctness_passed": true}\n',
        encoding="utf-8",
    )

    try:
        build_payload(
            single_node_paths=tuple(paths),
            multinode_path=multinode,
            rdma_log_path=rdma,
        )
    except ValueError as exc:
        assert "world sizes 1/2/4/8" in str(exc)
    else:
        raise AssertionError("invalid scaling evidence was accepted")
