import json
from pathlib import Path


def load_run(path):
    """
    opens and parses json file
    validates top-level structure is dictionary
    "meta" and "results" MUST exist in dictionary (implement in worker later)
    """
    path = Path(path)

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

    except FileNotFoundError as err:
        raise FileNotFoundError(f"File not found: {path}") from err

    except json.JSONDecodeError as err:
        raise ValueError(
            f"Bad JSON in {path}, line {err.lineno}, "
            f"column {err.colno}: {err.msg}"
        ) from err

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary in {path}")

    if "meta" not in data or "results" not in data:
        missing = []

        if "meta" not in data:
            missing.append("meta")

        if "results" not in data:
            missing.append("results")

        raise ValueError(f"Missing keys {missing} in {path}")

    return data

#todo: write check meta function
