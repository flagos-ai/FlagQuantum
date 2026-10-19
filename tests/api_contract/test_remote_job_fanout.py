"""One program across several targets: independent streams, no shared claim."""

import pytest

import flagquantum as fq
from flagquantum.remote import jobs

pytestmark = pytest.mark.unit

TARGETS = ("quafu:Baihua", "quafu:Dongling")

# Named in an order that is not the sorted one, so a report that sorts its own
# targets is a different report from one that preserves the caller's order.
CALLER_ORDER = ("quafu:Dongling", "quafu:Baihua")


class Transport:
    """One target's own transport, with the state that target is in."""

    def __init__(self):
        self.posts = []
        self.gets = []
        self.state = "Transpiled"
        self.counts = {"01": 1000}

    def post_json(self, url, payload, headers, timeout):
        self.posts.append(payload)
        return {"task_id": "job-123"}

    def get_json(self, url, headers, timeout):
        self.gets.append(url)
        if "/status/" in url:
            return self.state
        if "/result/" in url:
            return {"count": dict(self.counts), "transpiled": "physical circuit"}
        if "/cancel/" in url:
            self.state = "Cancelled"
            return {}
        raise AssertionError(url)


def _install(monkeypatch, refusal=None):
    """Install a Quafu client with one transport per submission.

    The transport is keyed by backend name rather than by arrival order, because
    submissions run concurrently and their arrival order is not the caller's.
    """

    registry = {}
    refuse = refusal if refusal is not None else lambda name: None

    class Client(jobs.QuafuProvider):
        def __init__(self, **kwargs):
            self._stream = Transport()
            super().__init__(
                token="test-private-token", transport=self._stream, **kwargs
            )

        def submit(self, package):
            registry[package.backend.name] = self._stream
            error = refuse(package.backend.name)
            if error is not None:
                raise error
            return super().submit(package)

    monkeypatch.setattr(jobs, "QuafuProvider", Client)
    return registry


@pytest.fixture
def streams(monkeypatch):
    return _install(monkeypatch)


@pytest.fixture
def dongling_down(monkeypatch):
    return _install(
        monkeypatch,
        lambda name: TimeoutError("response lost") if name == "Dongling" else None,
    )


@pytest.fixture
def all_down(monkeypatch):
    return _install(monkeypatch, lambda name: TimeoutError("response lost"))


def test_every_named_target_is_submitted_once_with_its_own_stream(streams):
    report = jobs.submit_to_targets(
        fq.Circuit(2).x(0), targets=CALLER_ORDER, shots=1024
    )
    assert report.targets == CALLER_ORDER
    assert report.submitted_count == 2
    assert report.failed_targets == ()
    assert sorted(report.jobs) == sorted(TARGETS)
    assert [len(streams[name].posts) for name in ("Baihua", "Dongling")] == [1, 1]
    assert streams["Baihua"] is not streams["Dongling"]
    assert len({id(job) for job in report.jobs.values()}) == 2


def test_a_failing_target_is_recorded_and_the_others_still_submit(dongling_down):
    report = jobs.submit_to_targets(fq.Circuit(2), targets=TARGETS, shots=1024)
    assert report.failed_targets == ("quafu:Dongling",)
    assert report.submitted_count == 1
    assert report.jobs["quafu:Baihua"].id == "job-123"
    broken = next(item for item in report.submissions if item.job is None)
    assert broken.submitted is False
    assert broken.error_kind == "TimeoutError"
    assert broken.error_message == "response lost"
    assert dongling_down["Dongling"].posts == []
    assert report.evidence()["failed_targets"] == ("quafu:Dongling",)


def test_a_failed_submission_is_never_retried(all_down):
    report = jobs.submit_to_targets(fq.Circuit(2), targets=TARGETS, shots=1024)
    assert report.submitted_count == 0
    assert report.failed_targets == TARGETS
    assert [len(all_down[name].posts) for name in ("Baihua", "Dongling")] == [0, 0]
    assert report.jobs == {}


@pytest.mark.parametrize(
    "targets, message",
    [
        # A bare string is a Sequence of characters, so it is refused by name
        # rather than splintered into twelve single-character targets.
        ("quafu:Baihua", "non-empty sequence"),
        ((), "non-empty sequence"),
        ([], "non-empty sequence"),
        (["quafu:Baihua", "quafu:Baihua"], "must be distinct"),
        ([1, 2], "non-empty sequence"),
        (["quafu:Baihua", None], "non-empty sequence"),
    ],
)
def test_an_invalid_target_list_is_refused_before_any_contact(
    streams, targets, message
):
    with pytest.raises(ValueError, match=message):
        jobs.submit_to_targets(fq.Circuit(2), targets=targets, shots=1024)
    assert streams == {}


def test_a_shared_name_is_refused_for_several_targets(streams):
    with pytest.raises(ValueError, match="one submission"):
        jobs.submit_to_targets(fq.Circuit(2), targets=TARGETS, shots=1024, name="bell")
    assert streams == {}
    one = jobs.submit_to_targets(
        fq.Circuit(2), targets=["quafu:Baihua"], shots=1024, name="bell"
    )
    assert one.submitted_count == 1


def test_an_unreadable_program_is_refused_before_any_target(streams):
    with pytest.raises(TypeError, match="CircuitIR"):
        jobs.submit_to_targets(object(), targets=TARGETS, shots=1024)
    assert streams == {}


def test_a_shared_option_refused_by_every_target_contacts_none(streams):
    report = jobs.submit_to_targets(
        fq.Circuit(2),
        targets=TARGETS,
        shots=1024,
        outputs=fq.expectation(fq.Z(0)),
    )
    assert report.submitted_count == 0
    assert {item.error_kind for item in report.submissions} == {"CapabilityError"}
    assert streams == {}


def test_a_mixed_target_list_records_the_provider_specific_refusal(streams):
    report = jobs.submit_to_targets(
        fq.Circuit(2), targets=["quafu:Baihua", "jiuding:cpu"], shots=1024
    )
    assert report.submitted_count == 1
    assert report.failed_targets == ("jiuding:cpu",)
    broken = next(item for item in report.submissions if item.job is None)
    assert broken.error_kind == "ValueError"
    assert "requires image" in (broken.error_message or "")
    assert len(streams["Baihua"].posts) == 1


def test_results_are_read_from_each_targets_own_stream(streams):
    report = jobs.submit_to_targets(fq.Circuit(2), targets=TARGETS, shots=1024)
    for name in ("Baihua", "Dongling"):
        streams[name].state = "Finished"
    streams["Baihua"].counts = {"01": 1000}
    streams["Dongling"].counts = {"10": 1000}
    first = report.jobs["quafu:Baihua"].result()
    second = report.jobs["quafu:Dongling"].result()
    assert list(first.counts[0].values()) == [1000]
    assert set(first.counts[0]) != set(second.counts[0])
    assert len(streams["Baihua"].posts) == 1
    assert len(streams["Dongling"].posts) == 1


def test_targets_are_contacted_concurrently_not_one_after_another(monkeypatch):
    """Each submission waits for its peers, so a sequential loop cannot pass.

    Both submissions block until the other has arrived. Only a fan-out that
    hands both to the pool before waiting on either can satisfy the barrier; a
    sequential loop raises BrokenBarrierError after the barrier's own timeout,
    which surfaces as a recorded failure rather than as a hang.
    """

    import threading

    barrier = threading.Barrier(2, timeout=2.0)
    arrivals = []

    class Client(jobs.QuafuProvider):
        def __init__(self, **kwargs):
            self._stream = Transport()
            super().__init__(
                token="test-private-token", transport=self._stream, **kwargs
            )

        def submit(self, package):
            arrivals.append(package.backend.name)
            barrier.wait()
            return super().submit(package)

    monkeypatch.setattr(jobs, "QuafuProvider", Client)
    report = jobs.submit_to_targets(fq.Circuit(2), targets=TARGETS, shots=1024)
    assert report.submitted_count == 2, report.submissions
    assert sorted(arrivals) == ["Baihua", "Dongling"]


def test_the_report_refuses_a_scalability_claim(streams):
    report = jobs.submit_to_targets(fq.Circuit(2), targets=CALLER_ORDER, shots=1024)
    evidence = report.evidence()
    assert evidence["distribution_semantics"] == "replicated_independent_targets"
    assert evidence["scalability_claim_allowed"] is False
    assert evidence["targets"] == CALLER_ORDER
    assert evidence["submitted"] == 2
    assert len(evidence["blockers"]) == 3
    assert "world_size" not in evidence
    assert "node_count" not in evidence
    with pytest.raises(TypeError):
        report.jobs["quafu:Other"] = report.jobs["quafu:Baihua"]


def test_the_fanout_is_not_re_exported(streams):
    import flagquantum.remote as remote

    assert not hasattr(remote, "submit_to_targets")
    assert not hasattr(fq, "submit_to_targets")
    assert callable(jobs.submit_to_targets)
