"""Jiuding workspace task adapter. Experimental; no Stable Core API changes.

Scripts, receipts and results must be on storage shared with the submitted job.
The adapter manages external tasks; it does not select numerical backends.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import select
import shlex
import socket
import struct
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ._jiuding_credentials import load_jiuding_credentials
from ._job_results import read_job_result, save_receipt
from ._program_submission import _ProgramSubmissionMixin, decode_job_result
from ._workspace_results import decode_tensor, measurement_result

_DEFAULT_CLIENTS: dict[str, "JiudingClient"] = {}
_MAX_BATCH_ITEMS = 256
_SUPPORTED_RESIDENT_MEASUREMENTS = {
    "counts",
    "counts_ps",
    "expectation_identity",
    "expectation_ps",
    "probabilities",
    "sample",
    "sample_ps",
}


def _measurement_program(program, *, outputs, shots):
    from dataclasses import replace

    from ...core.ir import ensure_circuit_ir
    from ...observables import lower_outputs

    ir = ensure_circuit_ir(program)
    measurements = lower_outputs(outputs, n_wires=ir.n_wires, shots=shots)
    assert measurements is not None
    if any(item.kind not in _SUPPORTED_RESIDENT_MEASUREMENTS for item in measurements):
        raise ValueError(
            "Jiuding resident execution received an unsupported output request"
        )
    return replace(ir, measurements=measurements)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class JiudingClient(_ProgramSubmissionMixin):
    """Use injected credentials and discover the current workspace's queue.

    ``workspace`` is optional inside a platform pod; outside it, specify a
    unique workspace name. Credentials come from JIUDING_AK/JIUDING_SK or
    /etc/accesskey/user-{ak,sk}. No credentials are stored in receipts.
    """

    def __init__(
        self,
        *,
        workspace: str | None = None,
        endpoint: str = "https://platform-multi.baai.ac.cn",
    ):
        url = urlsplit(endpoint)
        if (
            url.scheme != "https"
            or not url.netloc
            or url.username
            or url.password
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            raise ValueError("endpoint must be an HTTPS origin")
        self.endpoint = endpoint.rstrip("/")
        self.workspace_name = workspace
        self._token = ""
        self._expires = 0.0
        self._workspace: dict | None = None
        self._ssh_command: tuple[str, ...] | None = None
        self._executor_channels: dict[int, subprocess.Popen] = {}
        self._executor_health: dict[int, dict] = {}

    def _request(
        self, path: str, body: dict | None, headers: dict, *, method: str = "POST"
    ) -> dict:
        if method not in {"GET", "POST", "PUT"}:
            raise ValueError("Jiuding request method must be GET, POST or PUT")
        request = Request(
            self.endpoint + path,
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with build_opener(_NoRedirect).open(request, timeout=20) as response:
                result = json.load(response)
        except HTTPError as exc:
            message = ""
            try:
                error = json.loads(exc.read(8192))
                if isinstance(error, dict):
                    message = str(error.get("message", ""))[:1000]
                    for value in headers.values():
                        if value:
                            message = message.replace(value, "[REDACTED]")
            except (ValueError, OSError):
                pass
            raise RuntimeError(f"Jiuding {path}: HTTP {exc.code} {message}") from None
        except (URLError, OSError, ValueError):
            raise RuntimeError(
                f"Jiuding {path}: transport/JSON error; request not retried"
            ) from None
        if not isinstance(result, dict):
            raise RuntimeError(f"Jiuding {path}: expected JSON object")
        return result

    def _auth(self) -> dict:
        if time.monotonic() >= self._expires:
            ak, sk = load_jiuding_credentials()
            path = "/api/v1/users/token/exchange"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            digest = hashlib.sha256(f"POST\n{path}\n{stamp}".encode()).hexdigest()
            signature = hmac.new(
                sk.encode(),
                f"AIRS-HMAC-SHA256\n{stamp}\n{digest}".encode(),
                hashlib.sha256,
            ).hexdigest()
            result = self._request(
                path,
                None,
                {
                    "Authorization": f"AIRS-HMAC-SHA256 Credential={ak}, Signature={signature}",
                    "X-Airs-Date": stamp,
                },
            )
            token = result.get("token")
            if not isinstance(token, str) or not token:
                raise RuntimeError("Jiuding authentication returned no token")
            lifetime = int(result.get("expiresIn", 0))
            if lifetime <= 0:
                raise RuntimeError("Jiuding authentication returned invalid expiry")
            self._token = token
            self._expires = time.monotonic() + max(1, lifetime - min(60, lifetime / 2))
        return {"AIRS-Token": self._token}

    def _pages(self, path: str, body: dict, field: str, headers: dict):
        for page in range(1, 101):
            result = self._request(
                path, {**body, "paging": {"page": page, "pageSize": 100}}, headers
            )
            rows = result.get(field)
            if not isinstance(rows, list):
                raise RuntimeError(f"Jiuding response missing {field}")
            yield from rows
            if len(rows) < 100:
                return
        raise RuntimeError(
            "Jiuding pagination limit reached; refusing incomplete discovery"
        )

    def workspace(self) -> dict:
        """Return non-secret context for one unambiguous visible workspace."""
        if self._workspace is None:
            matches = [
                w
                for w in self._pages(
                    "/api/v1/workspaces/select", {}, "items", self._auth()
                )
                if (
                    w.get("name") == self.workspace_name
                    if self.workspace_name
                    else w.get("podName") == socket.gethostname()
                )
            ]
            if len(matches) != 1:
                raise RuntimeError("Specify a unique Jiuding workspace name")
            w = matches[0]
            fields = (
                "id",
                "name",
                "projId",
                "projsetId",
                "queueId",
                "queueName",
                "clusterId",
                "zoneId",
                "queueStatus",
                "resourceRegion",
                "creatorId",
                "creatorName",
            )
            self._workspace = {k: w[k] for k in fields}
            self._workspace["storageInfo"] = w.get("advanceConfig", {}).get(
                "storageInfo", []
            )
            self._workspace["acceleratorModel"] = (
                w.get("quotaDetail", {})
                .get("resourceDetail", {})
                .get("acceleratorModel", "")
            )
            self._workspace["clusterName"] = w.get("quotaDetail", {}).get(
                "clusterName", ""
            )
        return json.loads(json.dumps(self._workspace))

    def _headers(self) -> dict:
        w = self.workspace()
        return {
            **self._auth(),
            "Airs-Proj-id": w["projId"],
            "Airs-Projset-id": w["projsetId"],
            "AIRS-Cluster-ID": w["clusterId"],
            "AIRS-Zone-ID": w["zoneId"],
        }

    def _resolve_compute_target(self, target: str) -> tuple[str, str]:
        if not isinstance(target, str):
            raise TypeError("target must be a string")
        provider, separator, resource = target.partition(":")
        if separator != ":" or provider.lower() != "jiuding" or not resource:
            raise ValueError(
                "target must use 'jiuding:<chip-type>' or "
                "'jiuding:<chip-type>/<model>'"
            )
        chip_type, model_separator, model = resource.partition("/")
        chip_type = chip_type.lower()
        if chip_type not in {"cpu", "gpu", "mlu", "npu", "xpu"}:
            raise ValueError("Jiuding chip type must be cpu, gpu, mlu, npu or xpu")
        if model_separator and not model:
            raise ValueError("target model must not be empty")
        if chip_type == "cpu":
            if model_separator:
                raise ValueError("jiuding:cpu does not accept a model")
            return chip_type, ""
        if chip_type != "gpu":
            raise NotImplementedError(
                f"{target!r} is reserved but not implemented by JiudingClient"
            )
        available_model = self.workspace().get("acceleratorModel", "")
        if not available_model:
            raise RuntimeError("The selected Jiuding queue exposes no GPU model")
        if model and model != available_model:
            raise ValueError(
                f"target model {model!r} does not match queue model "
                f"{available_model!r}"
            )
        return chip_type, available_model

    def _workspace_record(self, reference: str) -> dict:
        context = self.workspace()
        matches = [
            item
            for item in self._pages(
                "/api/v1/workspaces/select", {}, "items", self._auth()
            )
            if item.get("id") == reference or item.get("name") == reference
            if item.get("projId") == context["projId"]
            and item.get("projsetId") == context["projsetId"]
        ]
        if len(matches) != 1:
            raise RuntimeError("Workspace was not uniquely resolved in this project")
        return matches[0]

    def _ssh_base(self) -> list[str]:
        """Build a constrained SSH command from the platform workspace record."""

        if self._ssh_command is not None:
            return list(self._ssh_command)
        record = self._workspace_record(self.workspace()["id"])
        login = record.get("SSHLogin")
        if not isinstance(login, str):
            raise RuntimeError("Jiuding workspace exposes no SSH login")
        tokens = shlex.split(login)
        if not tokens or tokens[0] != "ssh":
            raise RuntimeError("Jiuding workspace returned an invalid SSH login")
        endpoint = next(
            (item for item in tokens[1:] if re.fullmatch(r"[\w.-]+@[\w.-]+", item)),
            None,
        )
        ports = [
            tokens[index + 1] for index, item in enumerate(tokens[:-1]) if item == "-p"
        ]
        if endpoint is None or len(ports) != 1 or not ports[0].isdigit():
            raise RuntimeError("Jiuding workspace returned an invalid SSH endpoint")
        identity = hashlib.sha256(
            f"{self.endpoint}\0{endpoint}\0{ports[0]}".encode()
        ).hexdigest()[:16]
        command = (
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=300",
            "-o",
            f"ControlPath=/tmp/fq-jiuding-{identity}",
            "-p",
            ports[0],
            endpoint,
        )
        self._ssh_command = command
        return list(command)

    def _executor_request(self, payload: dict, *, port: int, timeout: float) -> dict:
        from ._workspace_executor import MAX_MESSAGE_BYTES

        if not 1024 <= port <= 65535:
            raise ValueError("port must be between 1024 and 65535")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise ValueError("workspace executor request exceeds the size limit")
        frame = struct.pack("!I", len(encoded)) + encoded
        process = self._executor_channels.get(port)
        if process is None or process.poll() is not None:
            command = [*self._ssh_base(), "-W", f"127.0.0.1:{port}"]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._executor_channels[port] = process
        assert process.stdin is not None and process.stdout is not None
        deadline = time.monotonic() + timeout
        try:
            process.stdin.write(frame)
            process.stdin.flush()

            def read_exact(size: int) -> bytes:
                chunks = bytearray()
                while len(chunks) < size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    ready, _, _ = select.select([process.stdout], [], [], remaining)
                    if not ready:
                        raise TimeoutError
                    chunk = os.read(process.stdout.fileno(), size - len(chunks))
                    if not chunk:
                        raise EOFError
                    chunks.extend(chunk)
                return bytes(chunks)

            size = struct.unpack("!I", read_exact(4))[0]
            if size <= 0 or size > MAX_MESSAGE_BYTES:
                raise ValueError("invalid workspace executor response size")
            stdout = read_exact(size)
        except TimeoutError:
            self._close_executor_channel(port)
            raise TimeoutError("Jiuding workspace executor request timed out") from None
        except (BrokenPipeError, EOFError, OSError, ValueError):
            self._close_executor_channel(port)
            raise RuntimeError("Jiuding workspace executor is unavailable") from None
        try:
            response = json.loads(stdout)
            if not isinstance(response, dict):
                raise TypeError
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Jiuding workspace executor returned invalid data"
            ) from exc
        if not response.get("ok"):
            raise RuntimeError(
                f"Jiuding workspace execution failed: {response.get('message', '')}"
            )
        return response

    def _close_executor_channel(self, port: int) -> None:
        process = self._executor_channels.pop(port, None)
        self._executor_health.pop(port, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _reset_workspace_connection(self) -> None:
        """Discard transport facts that can change when a workspace restarts."""

        self.close()
        self._ssh_command = None
        self._workspace = None

    def close(self) -> None:
        """Close persistent SSH channels owned by this client."""

        for port in tuple(self._executor_channels):
            self._close_executor_channel(port)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def start_executor(
        self,
        *,
        target: str,
        port: int = 57621,
        timeout: float = 30,
        python: str = "/opt/conda/bin/python",
    ) -> dict:
        """Ensure one loopback-only resident executor is ready in the workspace."""

        chip_type, model = self._resolve_compute_target(target)
        device = "cuda:0" if chip_type == "gpu" else "cpu"
        effective_target = f"jiuding:gpu/{model}" if model else "jiuding:cpu"
        cached = self._executor_health.get(port)
        if (
            cached is not None
            and cached.get("target") == effective_target
            and cached.get("device") == device
        ):
            return dict(cached)
        health = {
            "schema": "flagquantum.jiuding.workspace_executor",
            "version": "1.3",
            "operation": "health",
        }
        try:
            response = self._executor_request(
                health, port=port, timeout=min(timeout, 5)
            )
        except (RuntimeError, TimeoutError):
            self._reset_workspace_connection()
            chip_type, model = self._resolve_compute_target(target)
            device = "cuda:0" if chip_type == "gpu" else "cpu"
            effective_target = f"jiuding:gpu/{model}" if model else "jiuding:cpu"
            command = shlex.join(
                [
                    python,
                    "-m",
                    "flagquantum.remote.compute._workspace_executor",
                    "--port",
                    str(port),
                    "--target",
                    effective_target,
                    "--device",
                    device,
                ]
            )
            remote = (
                "nohup "
                + command
                + " </dev/null >/tmp/flagquantum-workspace-executor.log 2>&1 &"
            )
            started = subprocess.run(
                [*self._ssh_base(), remote],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            if started.returncode:
                raise RuntimeError("Failed to start Jiuding workspace executor")
            deadline = time.monotonic() + timeout
            while True:
                try:
                    response = self._executor_request(
                        health,
                        port=port,
                        timeout=min(5, max(0.1, deadline - time.monotonic())),
                    )
                    break
                except (RuntimeError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Jiuding workspace executor did not become ready"
                        ) from None
                    time.sleep(0.2)
        if (
            response.get("target") != effective_target
            or response.get("device") != device
        ):
            raise RuntimeError(
                "Resident executor target differs from the requested target"
            )
        if chip_type == "gpu" and not response.get("cuda_available"):
            raise RuntimeError("Resident executor cannot access CUDA")
        self._executor_health[port] = dict(response)
        return dict(response)

    def run_statevector(
        self,
        program,
        *,
        target: str,
        port: int = 57621,
        timeout: float = 30,
    ):
        """Run a circuit on the warm workspace and return a normal ExecutionResult."""

        response, effective_target = self._execute(
            program,
            target=target,
            port=port,
            timeout=timeout,
        )
        state = decode_tensor(response["state"])
        from ...runtime.result import ExecutionResult

        evidence = dict(response.get("evidence", {}))
        return ExecutionResult(
            state=state,
            runtime={
                "execution_path": "jiuding_workspace_executor",
                "device": evidence.get("device"),
                "elapsed_seconds": evidence.get("elapsed_seconds"),
            },
            provenance={
                "requested_target": target,
                "selected_target": effective_target,
                "accelerator": evidence.get("accelerator"),
                "cpu_fallback_used": evidence.get("cpu_fallback_used"),
            },
        )

    def run(
        self,
        program,
        *,
        target: str,
        outputs=None,
        shots: int | None = None,
        port: int = 57621,
        timeout: float = 30,
    ):
        """Run statevector or requested measurements in a resident workspace."""

        if outputs is None:
            if shots is not None:
                raise TypeError(
                    "shots requires fq.samples(...) or fq.counts(...) output"
                )
            return self.run_statevector(
                program,
                target=target,
                port=port,
                timeout=timeout,
            )
        ir = _measurement_program(program, outputs=outputs, shots=shots)
        response, effective_target = self._execute(
            ir,
            target=target,
            measurements=ir.measurements,
            port=port,
            timeout=timeout,
        )
        return measurement_result(
            response,
            requested_target=target,
            effective_target=effective_target,
        )

    def run_batch(
        self,
        programs,
        *,
        target: str,
        outputs,
        shots: int | None = None,
        port: int = 57621,
        timeout: float = 30,
    ):
        """Execute a bounded measurement batch in one workspace request."""

        if not isinstance(programs, (list, tuple)):
            raise TypeError("programs must be a list or tuple of circuits")
        if not programs:
            raise ValueError("programs must not be empty")
        if len(programs) > _MAX_BATCH_ITEMS:
            raise ValueError(
                f"programs exceeds the {_MAX_BATCH_ITEMS}-circuit batch limit"
            )
        if outputs is None:
            raise TypeError("batch execution requires measurement outputs")
        prepared = tuple(
            _measurement_program(program, outputs=outputs, shots=shots)
            for program in programs
        )
        self.start_executor(target=target, port=port, timeout=timeout)
        chip_type, model = self._resolve_compute_target(target)
        effective_target = f"jiuding:gpu/{model}" if model else "jiuding:cpu"
        response = self._executor_request(
            {
                "schema": "flagquantum.jiuding.workspace_executor",
                "version": "1.3",
                "operation": "measurement_batch",
                "request_id": uuid.uuid4().hex,
                "target": effective_target,
                "programs": [program.to_dict() for program in prepared],
            },
            port=port,
            timeout=timeout,
        )
        responses = response.get("results")
        if not isinstance(responses, list) or len(responses) != len(prepared):
            raise RuntimeError("Jiuding workspace returned an invalid batch result")
        batch_evidence = response.get("evidence", {})
        return tuple(
            measurement_result(
                item,
                requested_target=target,
                effective_target=effective_target,
                batch_index=index,
                batch_size=len(responses),
                batch_elapsed_seconds=batch_evidence.get("elapsed_seconds"),
            )
            for index, item in enumerate(responses)
        )

    def _execute(
        self,
        program,
        *,
        target: str,
        measurements=(),
        port: int,
        timeout: float,
    ) -> tuple[dict, str]:
        from dataclasses import replace

        from ...core.ir import ensure_circuit_ir

        ir = ensure_circuit_ir(program)
        if measurements:
            ir = replace(ir, measurements=tuple(measurements))
        else:
            ir = replace(ir, measurements=())
        self.start_executor(target=target, port=port, timeout=timeout)
        chip_type, model = self._resolve_compute_target(target)
        effective_target = f"jiuding:gpu/{model}" if model else "jiuding:cpu"
        response = self._executor_request(
            {
                "schema": "flagquantum.jiuding.workspace_executor",
                "version": "1.3",
                "operation": "measurements" if measurements else "statevector",
                "request_id": uuid.uuid4().hex,
                "target": effective_target,
                "program": ir.to_dict(),
            },
            port=port,
            timeout=timeout,
        )
        return response, effective_target

    def workspace_state(self, reference: str) -> dict:
        """Return the current non-secret state of one project workspace."""
        item = self._workspace_record(reference)
        fields = (
            "id",
            "name",
            "status",
            "queueId",
            "queueName",
            "imageId",
            "imageName",
            "imageRegion",
            "lastStartTime",
            "lastStopTime",
            "failReason",
        )
        return {field: item.get(field) for field in fields}

    def create_workspace(
        self,
        name: str,
        *,
        target: str,
        image: str,
        image_region: str = "PRIVATE",
        accelerator_count: int | None = None,
        cpus: int = 4,
        memory_gib: int = 16,
        description: str = "",
    ) -> dict:
        """Create and start one workspace in the current project and queue.

        This explicit infrastructure operation is separate from ``run()``.
        ``target`` names CPU or one GPU model. The operation does not enable
        privileged mode, Jupyter, or automatic snapshots.
        """
        if not re.fullmatch(r"[a-z][a-z0-9-]{1,30}[a-z0-9]", name):
            raise ValueError("name must be 3-32 lowercase letters, digits or hyphens")
        if type(cpus) is not int or cpus < 1:
            raise ValueError("cpus must be a positive integer")
        if type(memory_gib) is not int or memory_gib < 2:
            raise ValueError("memory_gib must be an integer of at least 2")
        context = self.workspace()
        chip_type, model = self._resolve_compute_target(target)
        if accelerator_count is None:
            accelerator_count = int(chip_type == "gpu")
        if type(accelerator_count) is not int or accelerator_count < 0:
            raise ValueError("accelerator_count must be a non-negative integer")
        if chip_type == "cpu" and accelerator_count:
            raise ValueError("CPU workspaces cannot request accelerators")
        if chip_type == "gpu" and accelerator_count < 1:
            raise ValueError("GPU workspaces must request at least one accelerator")
        catalog_image = self._catalog_image(image, image_region)
        body = {
            "name": name,
            "imageId": catalog_image["id"],
            "imageRegion": image_region,
            "dataset": {"types": 1, "items": []},
            "modelType": "PRIVATE",
            "description": description,
            "store": 50,
            "creatorName": context["creatorName"],
            "isAutoSaveSnap": False,
            "autoSaveFrequency": 7,
            "autoSaveTime": "03",
            "resourceRegion": context["resourceRegion"],
            "queueId": context["queueId"],
            "queueName": context["queueName"],
            "clusterId": context["clusterId"],
            "zoneId": context["zoneId"],
            "advanceConfig": {"storageInfo": context["storageInfo"]},
            "quotaDetail": {
                "clusterName": context.get("clusterName", ""),
                "priority": "",
                "resourceDetail": {
                    "acceleratorModel": model,
                    "acceleratorCount": accelerator_count,
                    "cpuCores": cpus,
                    "memGib": memory_gib,
                    "quotaItemId": 0,
                    "sharedMemGib": memory_gib // 2,
                },
            },
            "modelVersionIdItems": [],
            "isJupyterCloudIde": False,
            "isPrivileged": False,
        }
        created = self._request("/api/v1/workspaces/create", body, self._headers())
        if not created.get("id"):
            raise RuntimeError("Workspace creation returned no ID; outcome uncertain")
        return {
            "id": created["id"],
            "name": name,
            "image": f"{catalog_image['name']}:{catalog_image['tag']}",
            "queueId": context["queueId"],
            "resources": {
                "target": target,
                "cpus": cpus,
                "memory_gib": memory_gib,
                "gpus": accelerator_count,
                "accelerator_model": model,
            },
        }

    def start_workspace(self, reference: str) -> dict:
        """Request startup of one stopped workspace in the current project."""
        item = self._workspace_record(reference)
        return self._request(
            f"/api/v1/workspaces/{item['id']}/restart",
            {
                "id": item["id"],
                "userId": item["creatorId"],
                "isPrivileged": bool(item.get("isPrivileged", False)),
            },
            self._headers(),
            method="PUT",
        )

    def stop_workspace(self, reference: str) -> dict:
        """Stop one workspace without creating a container snapshot."""
        item = self._workspace_record(reference)
        return self._request(
            f"/api/v1/workspaces/{item['id']}/stop",
            {"id": item["id"], "userId": item["creatorId"], "saveSnapshot": False},
            self._headers(),
            method="PUT",
        )

    def _jobs(self):
        w = self.workspace()
        return self._pages(
            "/api/v1/job/select",
            {"queueId": w["queueId"], "experimentType": 1},
            "jobInfos",
            self._headers(),
        )

    def images(self) -> list[str]:
        """Images observed in this queue's existing jobs, not an image catalog."""
        w = self.workspace()
        return sorted(
            {
                j["basicImage"]
                for j in self._jobs()
                if j.get("queueId") == w["queueId"] and j.get("basicImage")
            }
        )

    def _catalog_image(self, image: str, image_region: str = "PUBLIC") -> dict:
        """Resolve one exact image through Jiuding's image catalog."""
        if image_region not in ("PUBLIC", "PRIVATE"):
            raise ValueError("image_region must be PUBLIC or PRIVATE")
        leaf = image.rsplit("/", 1)[-1]
        name, separator, tag = leaf.partition(":")
        if not separator or not name or not tag:
            raise ValueError("Use an exact image name and tag, such as name:v1")
        result = self._request(
            "/api/v1/images/select",
            {
                "imageStatus": [10],
                "repoTypes": image_region,
                "clusterId": self.workspace()["clusterId"],
                "name": name,
                "paging": {"page": 1, "pageSize": 100},
            },
            self._headers(),
        )
        items = result.get("items")
        if not isinstance(items, list):
            raise RuntimeError("Jiuding image catalog response missing items")
        matches = {
            item["id"]: item
            for item in items
            if isinstance(item, dict)
            and item.get("id")
            and item.get("status") == 10
            and item.get("clusterId") == self.workspace()["clusterId"]
            and (
                f"{item.get('name')}:{item.get('tag')}" in (image, leaf)
                or item.get("baseUrl") == image
                or item.get("registryUrl") == image
            )
        }
        if len(matches) != 1:
            raise ValueError("Image was not uniquely resolved in the Jiuding catalog")
        return next(iter(matches.values()))

    def submit(
        self,
        script: str | Path,
        *,
        target: str,
        image: str,
        receipt: str | Path,
        image_region: str = "PUBLIC",
        python: str = sys.executable,
        pythonpath: str | Path | None = None,
        cpus: int = 2,
        memory_gib: int = 2,
    ) -> dict:
        """Run a shared Python file's main() and persist its JSON return value.

        ``target`` explicitly selects CPU or one GPU. Receipt path must be new
        and on shared storage. Reusing it is rejected, including after an
        ambiguous timeout.
        """
        if (
            type(cpus) is not int
            or cpus < 1
            or type(memory_gib) is not int
            or memory_gib < 2
        ):
            raise ValueError("cpus must be positive and memory_gib >= 2")
        chip_type, target_model = self._resolve_compute_target(target)
        gpus = int(chip_type == "gpu")
        script, receipt = Path(script).resolve(), Path(receipt).resolve()
        if not script.is_file() or not receipt.parent.is_dir():
            raise ValueError(
                "script and receipt directory must exist on shared storage"
            )
        w = self.workspace()
        if w["queueStatus"] != "QUEUE_STATUS_ACTIVE":
            raise RuntimeError("Workspace queue is not active")
        if image_region not in ("PUBLIC", "PRIVATE"):
            raise ValueError("image_region must be PUBLIC or PRIVATE")
        catalog_image = (
            self._catalog_image(image, image_region)
            if ":" in image and "/" not in image
            else None
        )
        examples = (
            []
            if catalog_image
            else [
                j
                for j in self._jobs()
                if j.get("queueId") == w["queueId"] and j.get("basicImage") == image
            ]
        )
        if catalog_image:
            executable_image = catalog_image.get(
                "registryUrl" if image_region == "PRIVATE" else "baseUrl"
            )
            if not executable_image:
                raise RuntimeError(
                    "Jiuding image catalog returned no executable image URL"
                )
            models = [w["acceleratorModel"]] if w.get("acceleratorModel") else []
        elif examples:
            executable_image = image
            image_region = examples[0]["imageRegion"]
            models = sorted(
                {
                    role.get("resourceRequestDetail", {}).get("acceleratorModel")
                    for job in examples
                    for role in job.get("roleInfos", [])
                    if role.get("resourceRequestDetail", {}).get("acceleratorModel")
                }
            )
        else:
            catalog_image = self._catalog_image(image, image_region)
            executable_image = catalog_image.get(
                "registryUrl" if image_region == "PRIVATE" else "baseUrl"
            )
            if not executable_image:
                raise RuntimeError(
                    "Jiuding image catalog returned no executable image URL"
                )
            models = [w["acceleratorModel"]] if w.get("acceleratorModel") else []
        if gpus and target_model not in models:
            raise ValueError(
                "target model must match a model observed with this image and queue"
            )
        model = target_model or (models[0] if models else "")
        run_id = uuid.uuid4().hex
        result_path = receipt.parent / (run_id + ".result.json")
        command = shlex.join(
            [
                python,
                "-m",
                "flagquantum.remote.compute._worker",
                str(script),
                str(result_path),
                run_id,
                "--gpus",
                str(gpus),
            ]
        )
        if pythonpath is not None:
            root = Path(pythonpath).resolve()
            if not root.is_dir():
                raise ValueError("pythonpath must be a shared source directory")
            command = "env " + shlex.quote("PYTHONPATH=" + str(root)) + " " + command
        resource = {
            "queueId": w["queueId"],
            "priority": "high",
            "basicImage": executable_image,
            "imageRegion": image_region,
            "clusterId": w["clusterId"],
            "zoneId": w["zoneId"],
            "podRestartPolicy": "Never",
            "restartScope": "FailedInstanceOnly",
            "roleInfoList": [
                {
                    "name": "Master",
                    "replicas": 1,
                    "resourceRegion": w["resourceRegion"],
                    "resourceRequestDetail": {
                        "acceleratorModel": model,
                        "acceleratorCount": gpus,
                        "cpuCores": cpus,
                        "memGib": memory_gib,
                        "sharedMemGib": 1,
                        "rdmaSharedCount": 0,
                    },
                }
            ],
        }
        body = {
            "projId": w["projId"],
            "projsetId": w["projsetId"],
            "name": "flagquantum-" + run_id[:12],
            "experimentType": "1",
            "trainFrame": "PyTorch",
            "creatorId": w["creatorId"],
            "creator": w["creatorName"],
            "storageInfo": w["storageInfo"],
            "heteroType": 1,
            "advanceConfigInfos": [
                {
                    "configName": "config1",
                    "codeConfig": "0",
                    "command": command,
                    "hyperParameter": {},
                    "slotsPerWorker": 0,
                    "resourceConfigList": [resource],
                }
            ],
        }
        body.update(
            description="",
            timeType=60000,
            duration=0,
            duration1=0,
            codeConfig="0",
            createdTime="",
            delivery=False,
            mirrorType="",
            nativeCluster="",
            restartPolicy="",
            experimentId="",
            nameSpace="",
            profilerInfo={"enabled": False, "level": "typical"},
            modelStorageInfo={},
            flag=False,
        )
        record = {
            "run_id": run_id,
            "experimentName": body["name"],
            "entrypoint": str(script),
            "endpoint": self.endpoint,
            "queueId": w["queueId"],
            "workspace": w["name"],
            "receipt": str(receipt),
            "result_path": str(result_path),
            "submission": "unknown",
            "resources": {
                "target": target,
                "cpus": cpus,
                "memory_gib": memory_gib,
                "gpus": gpus,
                "accelerator_model": model,
            },
        }
        with receipt.open("x") as file:
            json.dump(record, file)
        created = self._request("/api/v1/experiment", body, self._headers())
        if not created.get("experimentId"):
            raise RuntimeError(
                "Creation outcome uncertain; reconcile receipt before retrying"
            )
        record.update(experimentId=created["experimentId"], submission="created")
        save_receipt(receipt, record)
        details = self._request(
            "/api/v1/experiment/select",
            {
                "experimentId": record["experimentId"],
                "experimentType": 1,
                "userId": "-1",
            },
            self._headers(),
        )
        matches = [
            e
            for e in details.get("experimentSummaryInfos", [])
            if e.get("experimentId") == record["experimentId"]
        ]
        if len(matches) != 1:
            raise RuntimeError("Created experiment could not be uniquely resolved")
        detail = matches[0]
        configs = detail.get("advanceConfigInfos", [])
        if len(configs) != 1 or not configs[0].get("confId"):
            raise RuntimeError("Expected one executable experiment configuration")
        config = configs[0]
        if gpus:
            resources = config.get("resourceConfigList", [])
            roles = resources[0].get("roleInfoList", []) if len(resources) == 1 else []
            request = (
                roles[0].get("resourceRequestDetail", {}) if len(roles) == 1 else {}
            )
            if (
                len(roles) != 1
                or roles[0].get("replicas") != 1
                or resources[0].get("queueId") != w["queueId"]
                or request.get("acceleratorCount") != gpus
                or request.get("acceleratorModel") != model
                or request.get("cpuCores") != cpus
                or request.get("memGib") != memory_gib
            ):
                raise RuntimeError(
                    "Saved GPU resource configuration differs from request; Job was not launched"
                )
        # Request shape verified against the platform's bundled airsctl using a
        # loopback fake gateway, then exercised directly on the real platform.
        launch = {
            "advanceConfigInfos": [config],
            "confId": config["confId"],
            "confName": config["configName"],
            "creatorId": str(detail["creatorId"]),
            "experimentOwnerId": str(detail["creatorId"]),
            "experimentId": record["experimentId"],
            "experimentType": 1,
            "jobOrigin": "JOB_ORIGIN_CLI",
            "name": record["experimentName"],
            "nameSpace": "airs",
            "nativeCluster": detail["nativeCluster"],
            "projId": w["projId"],
            "projsetId": w["projsetId"],
            "restartPolicy": "Never",
            "trainFrame": "PyTorch",
            "storageInfo": detail.get("storageInfo", []),
            "modelStorageInfo": {},
        }
        record["submission"] = "launch_unknown"
        save_receipt(receipt, record)
        started = self._request("/api/v1/job", launch, self._headers())
        if not started.get("jobId"):
            raise RuntimeError(
                "Launch returned no Job ID; reconcile saved experiment before retrying"
            )
        record.update(submission="submitted", jobId=started["jobId"])
        save_receipt(receipt, record)
        return record

    def status(self, receipt: dict) -> list[dict]:
        """Query this experiment only; an empty list is not success."""
        w = self.workspace()
        if (
            receipt.get("endpoint") != self.endpoint
            or receipt.get("queueId") != w["queueId"]
        ):
            raise ValueError("Receipt belongs to a different endpoint or queue")
        jobs = self._pages(
            "/api/v1/job/select",
            {
                "queueId": w["queueId"],
                "experimentType": 1,
                "experimentId": receipt["experimentId"],
            },
            "jobInfos",
            self._headers(),
        )
        return [
            {k: j.get(k) for k in ("id", "status", "createdTime", "endTime")}
            for j in jobs
            if j.get("experimentId") == receipt["experimentId"]
            and j.get("queueId") == w["queueId"]
        ]

    def result(self, receipt: dict, *, timeout: float = 120, poll_interval: float = 3):
        """Wait for platform success and return the matching shared JSON result.

        Timeout leaves the job running; use cancel() explicitly if appropriate.
        """
        if timeout < 0 or poll_interval <= 0:
            raise ValueError("timeout must be nonnegative and poll_interval positive")
        deadline = time.monotonic() + timeout
        while True:
            jobs = self.status(receipt)
            if jobs and all(j["status"] == "Succeed" for j in jobs):
                if receipt.get("artifact_transport") == "workspace_ssh":
                    from ._managed_program import read_managed_result

                    value = read_managed_result(self, receipt)
                else:
                    value = read_job_result(
                        Path(receipt["result_path"]), receipt["run_id"]
                    )
                return decode_job_result(value, receipt)
            if any(
                j["status"]
                not in ("Pending", "Scheduling", "Starting", "Running", "Succeed")
                for j in jobs
            ):
                raise RuntimeError(f"Jiuding job did not succeed: {jobs}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Job not complete; receipt retained, no resubmission or cancellation"
                )
            time.sleep(min(poll_interval, remaining))

    def cancel(self, receipt: dict) -> None:
        """Stop active jobs for this receipt; never archive/delete experiments."""
        for job in self.status(receipt):
            if job["status"] in ("Pending", "Scheduling", "Starting", "Running"):
                self._request(
                    "/api/v1/job/cancel", {"jobIds": [job["id"]]}, self._headers()
                )


def run(program, *, target: str, outputs=None, shots: int | None = None):
    """Execute through the process-local resident Jiuding workspace client."""

    workspace = os.environ.get("JIUDING_WORKSPACE", "").strip()
    client = _DEFAULT_CLIENTS.get(workspace)
    if client is None:
        client = JiudingClient(workspace=workspace or None)
        _DEFAULT_CLIENTS[workspace] = client
    return client.run(program, target=target, outputs=outputs, shots=shots)
