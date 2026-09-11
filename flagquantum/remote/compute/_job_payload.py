"""Build the platform experiment payload shared by native and workspace jobs."""

from typing import Any


def experiment_body(
    *,
    command: str,
    w: dict[str, Any],
    executable_image: str,
    image_region: str,
    model: str,
    gpus: int,
    cpus: int,
    memory_gib: int,
    run_id: str,
) -> dict[str, Any]:
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
    return body
