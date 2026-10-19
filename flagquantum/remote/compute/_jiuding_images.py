"""Exact image selection across Jiuding's authenticated catalog regions."""

from collections.abc import Callable
from typing import Any


def resolve_catalog_image(
    image: str,
    image_region: str,
    find_matches: Callable[[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Reject ambiguity while recognizing aliases for one executable image."""
    if image_region not in ("PUBLIC", "PRIVATE", "PROJSET", "AUTO"):
        raise ValueError("image_region must be PUBLIC, PRIVATE or PROJSET")
    name, separator, tag = image.rsplit("/", 1)[-1].partition(":")
    if not separator or not name or not tag:
        raise ValueError("Use an exact image name and tag, such as name:v1")
    matches: dict[str | tuple[str, str], dict[str, Any]] = {}
    if image_region == "AUTO":
        for region in ("PROJSET", "PRIVATE", "PUBLIC"):
            for item in find_matches(image, region):
                executable = item.get(
                    "baseUrl" if region == "PUBLIC" else "registryUrl"
                )
                key = executable or (region, item["id"])
                matches.setdefault(key, {**item, "imageRegion": region})
    else:
        matches = {item["id"]: item for item in find_matches(image, image_region)}
    if len(matches) != 1:
        raise ValueError("Image was not uniquely resolved in the Jiuding catalog")
    return next(iter(matches.values()))
