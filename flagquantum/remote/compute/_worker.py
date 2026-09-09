"""Execute one shared Python entrypoint and atomically write its JSON result."""

import json
import runpy
import sys
from pathlib import Path


def main():
    script, output, run_id = sys.argv[1:]
    sys.path.insert(0, str(Path(script).parent))
    entrypoint = runpy.run_path(script)["main"]
    value = entrypoint()
    result = Path(output)
    temporary = result.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"run_id": run_id, "value": value}, allow_nan=False) + "\n"
    )
    temporary.replace(result)


if __name__ == "__main__":
    main()
