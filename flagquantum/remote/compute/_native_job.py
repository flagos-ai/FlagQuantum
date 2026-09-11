"""Jiuding project jobs using HTTP commands and bounded structured log results."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ...core.ir import ensure_circuit_ir
from ...observables import lower_outputs
from ...runtime.result import ExecutionResult
from ._program_job import decode_program_result
from .jiuding import JiudingClient

_MAX_RESULT = 8 * 1024 * 1024
_MAX_COMMAND = 64 * 1024


def _one(rows: list[dict[str, Any]], name: str, label: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"{label} {name!r} did not resolve uniquely")
    return matches[0]


def _command(request: dict[str, Any], run_id: str) -> str:
    """Encode data as an argument; never interpolate user data into Python code."""
    encoded = base64.b64encode(json.dumps(request, allow_nan=False).encode()).decode()
    code = """import base64,hashlib,json,sys
from flagquantum.remote.compute._workspace_executor import execute
p=json.loads(base64.b64decode(sys.argv[1]))
r=execute(p,target=p["target"],device="cuda:0" if p["target"].startswith("jiuding:gpu/") else "cpu")
if r.get("ok") is not True:
    raise RuntimeError(r.get("message","FlagQuantum execution failed"))
b=json.dumps(r,allow_nan=False,separators=(",",":")).encode()
if len(b)>8388608: raise RuntimeError("Result exceeds the 8 MiB job log transport limit")
s=base64.b64encode(b).decode(); chunks=[s[i:i+3072] for i in range(0,len(s),3072)]
h=hashlib.sha256(b).hexdigest()
for i,chunk in enumerate(chunks):
    print("FQ_RESULT_"+sys.argv[2]+" "+str(i)+"/"+str(len(chunks))+" "+h+" "+chunk,flush=True)
"""
    command = shlex.join(["python", "-u", "-c", code, encoded, run_id])
    if len(command.encode()) > _MAX_COMMAND:
        raise ValueError("Circuit exceeds the 64 KiB inline job command limit")
    return command


def _decode_lines(lines: list[str], run_id: str) -> dict[str, Any]:
    prefix = f"FQ_RESULT_{run_id} "
    chunks: dict[int, str] = {}
    identity: tuple[int, str] | None = None
    for line in lines:
        if not line.startswith(prefix):
            continue
        try:
            index_count, digest, content = line[len(prefix) :].split(" ", 2)
            index, count = map(int, index_count.split("/"))
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Malformed job result chunk") from exc
        if not 0 <= index < count <= 4096 or len(content) > 3072:
            raise RuntimeError("Job result chunk exceeds transport bounds")
        if identity is not None and identity != (count, digest):
            raise RuntimeError("Conflicting job result identities")
        identity = (count, digest)
        if index in chunks and chunks[index] != content:
            raise RuntimeError("Conflicting job result chunks")
        chunks[index] = content
    if identity is None or len(chunks) != identity[0]:
        raise RuntimeError("Jiuding job has no result yet; log chunks are incomplete")
    encoded = "".join(chunks[index] for index in range(identity[0]))
    data = base64.b64decode(encoded, validate=True)
    if len(data) > _MAX_RESULT or hashlib.sha256(data).hexdigest() != identity[1]:
        raise RuntimeError("Jiuding result size or digest mismatch")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise RuntimeError("Jiuding result must be a JSON object")
    return value


class NativeJiudingJobClient(JiudingClient):
    """Reuse the job HTTP transport, resolving context from projects, not pods."""

    def __init__(self, *, project: str, queue: str | None = None) -> None:
        super().__init__()
        parts = project.split(".")
        if len(parts) != 2 or not all(parts):
            raise ValueError("project must use 'project-set.project' notation")
        self.project = project
        self.queue = queue
        self._context: dict[str, Any] | None = None

    def workspace(self) -> dict[str, Any]:
        """Supply project/queue context to existing HTTP helpers; never access a pod."""
        if self._context is not None:
            return dict(self._context)
        auth = self._auth()
        user = self._request("/api/v1/users/token/userinfo", None, auth, method="GET")[
            "data"
        ]
        sets = self._request("/api/v1/projsets/select-joined", {}, auth)["items"]
        set_name, project_name = self.project.split(".")
        project_set = _one([x["projsetInfo"] for x in sets], set_name, "Project set")
        projects = self._request(
            "/api/v1/projects/select-joined", {"projsetId": project_set["id"]}, auth
        )["items"]
        project = _one([x["projInfo"] for x in projects], project_name, "Project")
        headers = {
            **auth,
            "x-user-id": str(user["id"]),
            "AIRS-Proj-ID": project["id"],
            "AIRS-Projset-ID": project_set["id"],
        }
        queues = list(
            self._pages(
                "/api/v1/queue/select",
                {
                    "projId": project["id"],
                    "projsetId": project_set["id"],
                    "as_user": True,
                    "userId": str(user["id"]),
                },
                "queueSummaryInfos",
                headers,
            )
        )
        active = [q for q in queues if q.get("status") == "QUEUE_STATUS_ACTIVE"]
        if self.queue is None:
            if len(active) != 1:
                raise ValueError(
                    "Specify queue; active choices: "
                    + ", ".join(q["name"] for q in active)
                )
            selected = active[0]
        else:
            selected = _one(active, self.queue, "Queue")
        configs = [
            (cluster, zone, quota)
            for cluster in selected["quotaInfoList"]
            for zone in cluster["zoneQuotaInfoList"]
            for quota in zone["quotaMetaList"]
            if quota.get("priority") == "high"
        ]
        if len(configs) != 1:
            raise ValueError(
                "Queue must resolve one high-priority resource configuration"
            )
        cluster, zone, quota = configs[0]
        self._context = {
            "name": self.project,
            "projId": project["id"],
            "projsetId": project_set["id"],
            "queueId": selected["id"],
            "queueName": selected["name"],
            "queueStatus": selected["status"],
            "clusterId": cluster["clusterId"],
            "zoneId": zone["zoneId"],
            "clusterName": cluster["clusterName"],
            "resourceRegion": "ACCELERATOR_SUITE",
            "creatorId": str(user["id"]),
            "creatorName": user.get("alias", ""),
            "storageInfo": [],
            "acceleratorModel": quota["resourceDetail"].get("acceleratorModel", ""),
        }
        return dict(self._context)

    def submit_program(
        self,
        program: Any,
        *,
        target: str,
        image: str,
        outputs: Any = None,
        shots: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if kwargs:
            raise TypeError("Unsupported native job options: " + ", ".join(kwargs))
        ir = ensure_circuit_ir(program)
        if outputs is None and shots is not None:
            raise TypeError("shots requires an explicit sampling output")
        measurements = (
            lower_outputs(outputs, n_wires=ir.n_wires, shots=shots)
            if outputs is not None
            else ()
        )
        ir = replace(ir, measurements=measurements or ())
        w = self.workspace()
        chip, model = self._resolve_compute_target(target)
        selected = f"jiuding:gpu/{model}" if chip == "gpu" else "jiuding:cpu"
        from ._workspace_executor import SCHEMA, VERSION

        run_id = uuid.uuid4().hex
        command = _command(
            {
                "schema": SCHEMA,
                "version": VERSION,
                "operation": "measurements" if outputs is not None else "statevector",
                "request_id": "batch-program",
                "target": selected,
                "program": ir.to_dict(),
            },
            run_id,
        )
        catalog = self._catalog_image(image, "PRIVATE")
        executable = catalog.get("registryUrl")
        if not isinstance(executable, str) or not executable:
            raise ValueError("Image has no executable registry URL")
        # Preflight the same authenticated log service used for result retrieval.
        datasource = self._request(
            "/grafana/api/datasources/name/"
            + quote("log." + w["clusterName"], safe=""),
            None,
            self._auth(),
            method="GET",
        )
        if datasource.get("type") != "loki" or not isinstance(
            datasource.get("uid"), str
        ):
            raise RuntimeError("Expected the platform Loki job log datasource")
        receipt = Path(tempfile.mkdtemp(prefix="flagquantum-job-")) / "receipt.json"
        record = {
            "run_id": run_id,
            "experimentName": "flagquantum-" + run_id[:12],
            "endpoint": self.endpoint,
            "queueId": w["queueId"],
            "submission": "unknown",
            "project": self.project,
            "queue": w["queueName"],
            "receipt": str(receipt),
            "requested_target": target,
            "selected_target": selected,
            "artifact_transport": "job_logs",
            "datasource_uid": datasource["uid"],
            "created_ms": int(time.time() * 1000),
        }
        try:
            return self._launch_command(
                command=command,
                w=w,
                executable_image=executable,
                image_region="PRIVATE",
                model=model,
                gpus=int(chip == "gpu"),
                cpus=2,
                memory_gib=2,
                run_id=run_id,
                receipt=receipt,
                record=record,
            )
        except Exception as exc:
            # The journal records whether creation/launch may already have happened.
            raise RuntimeError(
                f"Native job submission stopped; inspect {receipt} before retrying"
            ) from exc

    def read_result(self, receipt: dict[str, Any]) -> ExecutionResult:
        if (
            receipt.get("project") != self.project
            or receipt.get("endpoint") != self.endpoint
        ):
            raise ValueError("Job receipt belongs to another project or endpoint")
        job_id = str(uuid.UUID(receipt["jobId"]))
        run_id = uuid.UUID(hex=receipt["run_id"]).hex
        if self.workspace()["queueId"] != receipt["queueId"]:
            raise ValueError("Job receipt belongs to another queue")
        response = self._request(
            "/grafana/api/ds/query",
            {
                "from": str(receipt["created_ms"] - 60000),
                "to": str(int(time.time() * 1000)),
                "queries": [
                    {
                        "refId": "A",
                        "datasource": {
                            "type": "loki",
                            "uid": receipt["datasource_uid"],
                        },
                        "expr": '{namespace="airs", app=~"job-'
                        + job_id
                        + '.*"} |= "FQ_RESULT_'
                        + run_id
                        + ' "',
                        "queryType": "range",
                        "maxLines": 4096,
                        "direction": "forward",
                    }
                ],
            },
            self._auth(),
        )
        result = response.get("results", {}).get("A", {})
        if result.get("error") or result.get("status") != 200:
            raise RuntimeError("Jiuding job log query did not succeed")
        lines = []
        for frame in result.get("frames", []):
            fields = frame["schema"]["fields"]
            for index, field in enumerate(fields):
                if field.get("name") == "Line":
                    lines.extend(frame["data"]["values"][index])
        value = _decode_lines(lines, run_id)
        return decode_program_result(
            value,
            requested_target=receipt["requested_target"],
            selected_target=receipt["selected_target"],
        )
