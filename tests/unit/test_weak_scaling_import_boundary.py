def test_weak_scaling_report_is_package_importable():
    from flagquantum.benchmarking import statevector_weak_scaling_report as report

    assert report.SCHEMA.endswith("weak_scaling_report.v1")
