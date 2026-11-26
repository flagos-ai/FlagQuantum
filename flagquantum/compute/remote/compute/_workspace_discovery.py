"""Pure workspace discovery rules for the Jiuding adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def summarize_workspaces(
    workspaces: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove credentials and transport details from workspace records."""

    summaries = []
    for workspace in workspaces:
        summary = {
            key: workspace[key]
            for key in (
                "id",
                "name",
                "projName",
                "projsetName",
                "queueName",
                "queueStatus",
            )
            if key in workspace
        }
        quota = workspace.get("quotaDetail") or {}
        resources = quota.get("resourceDetail") or {}
        model = resources.get("acceleratorModel", "")
        if model:
            summary["acceleratorModel"] = model
        summaries.append(summary)
    return summaries


def select_workspace(
    workspaces: Iterable[Mapping[str, Any]],
    *,
    requested_name: str | None,
    pod_name: str,
) -> dict[str, Any]:
    """Select an explicit, current-pod, or sole visible workspace."""

    visible = list(workspaces)
    if requested_name:
        matches = [item for item in visible if item.get("name") == requested_name]
    else:
        matches = [item for item in visible if item.get("podName") == pod_name]
        if not matches and len(visible) == 1:
            matches = visible
    if len(matches) == 1:
        return dict(matches[0])

    names = sorted(
        name for item in visible if isinstance(name := item.get("name"), str)
    )
    if not names:
        raise RuntimeError(
            "No Jiuding workspace is visible. Ask a project administrator for "
            "project and queue access, then create a development workspace."
        )
    raise RuntimeError(
        "Specify a unique Jiuding workspace name. "
        f"Available workspaces: {', '.join(names)}"
    )


__all__: tuple[str, ...] = ()
