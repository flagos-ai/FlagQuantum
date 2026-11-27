"""Non-secret Jiuding resource discovery for native jobs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol


class _Request(Protocol):
    def __call__(
        self,
        path: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        *,
        method: str = "POST",
    ) -> dict[str, Any]: ...


Pages = Callable[[str, dict[str, Any], str, dict[str, str]], Iterable[dict[str, Any]]]


def _objects(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise RuntimeError(f"Jiuding response missing {label}")
    return value


def _named_record(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Jiuding response missing {label}")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise RuntimeError(f"Jiuding {label} has no id")
    if not isinstance(value.get("name"), str) or not value["name"]:
        raise RuntimeError(f"Jiuding {label} has no name")
    return value


def _high_priority_configurations(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []
    for cluster in _objects(queue.get("quotaInfoList"), "queue quotaInfoList"):
        for zone in _objects(cluster.get("zoneQuotaInfoList"), "zoneQuotaInfoList"):
            for quota in _objects(zone.get("quotaMetaList"), "quotaMetaList"):
                if quota.get("priority") != "high":
                    continue
                detail = quota.get("resourceDetail")
                if not isinstance(detail, dict):
                    raise RuntimeError("Jiuding quota has no resourceDetail")
                cluster_id = cluster.get("clusterId")
                zone_id = zone.get("zoneId")
                if not isinstance(cluster_id, str) or not cluster_id:
                    raise RuntimeError("Jiuding queue configuration has no clusterId")
                if not isinstance(zone_id, str) or not zone_id:
                    raise RuntimeError("Jiuding queue configuration has no zoneId")
                configurations.append(
                    {
                        "cluster_id": cluster_id,
                        "zone_id": zone_id,
                        "accelerator_model": str(detail.get("acceleratorModel", "")),
                    }
                )
    return configurations


def _private_images(
    *,
    request_pages: Pages,
    headers: dict[str, str],
    cluster_id: str,
) -> list[str]:
    images: set[str] = set()
    for item in request_pages(
        "/api/v1/images/select",
        {
            "imageStatus": [10],
            "repoTypes": "PRIVATE",
            "clusterId": cluster_id,
        },
        "items",
        headers,
    ):
        name, tag = item.get("name"), item.get("tag")
        if (
            item.get("status") == 10
            and item.get("clusterId") == cluster_id
            and isinstance(name, str)
            and name
            and isinstance(tag, str)
            and tag
        ):
            images.add(f"{name}:{tag}")
    return sorted(images)


def discover_resources(
    *,
    auth: dict[str, str],
    request: _Request,
    request_pages: Pages,
) -> list[dict[str, Any]]:
    """Return native-job resource names visible to the authenticated account."""

    user = request("/api/v1/users/token/userinfo", None, auth, method="GET").get("data")
    if not isinstance(user, dict) or not isinstance(user.get("id"), (str, int)):
        raise RuntimeError("Jiuding user information has no id")
    user_id = str(user["id"])

    joined_sets = _objects(
        request("/api/v1/projsets/select-joined", {}, auth).get("items"),
        "project-set items",
    )
    resources: list[dict[str, Any]] = []
    for joined_set in joined_sets:
        project_set = _named_record(joined_set.get("projsetInfo"), "project set")
        joined_projects = _objects(
            request(
                "/api/v1/projects/select-joined",
                {"projsetId": project_set["id"]},
                auth,
            ).get("items"),
            "project items",
        )
        for joined_project in joined_projects:
            project = _named_record(joined_project.get("projInfo"), "project")
            project_name = f"{project_set['name']}.{project['name']}"
            project_headers = {
                **auth,
                "x-user-id": user_id,
                "AIRS-Proj-ID": project["id"],
                "AIRS-Projset-ID": project_set["id"],
            }
            queues = request_pages(
                "/api/v1/queue/select",
                {
                    "projId": project["id"],
                    "projsetId": project_set["id"],
                    "as_user": True,
                    "userId": user_id,
                },
                "queueSummaryInfos",
                project_headers,
            )
            for queue in queues:
                if queue.get("status") != "QUEUE_STATUS_ACTIVE":
                    continue
                queue_name = queue.get("name")
                if not isinstance(queue_name, str) or not queue_name:
                    raise RuntimeError("Jiuding active queue has no name")
                configurations = _high_priority_configurations(queue)
                if len(configurations) != 1:
                    resources.append(
                        {
                            "project": project_name,
                            "queue": queue_name,
                            "accelerator_model": None,
                            "images": [],
                            "submission_supported": False,
                            "reason": (
                                "queue must expose exactly one high-priority "
                                "resource configuration"
                            ),
                        }
                    )
                    continue
                configuration = configurations[0]
                images = _private_images(
                    request_pages=request_pages,
                    headers={
                        **project_headers,
                        "AIRS-Cluster-ID": configuration["cluster_id"],
                        "AIRS-Zone-ID": configuration["zone_id"],
                    },
                    cluster_id=configuration["cluster_id"],
                )
                resources.append(
                    {
                        "project": project_name,
                        "queue": queue_name,
                        "accelerator_model": configuration["accelerator_model"],
                        "images": images,
                        "submission_supported": True,
                    }
                )
    return sorted(resources, key=lambda item: (item["project"], item["queue"]))
