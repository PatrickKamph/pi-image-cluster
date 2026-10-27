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


def check_meta(reference, candidate):
    """
    compares metadata between reference and candidate runs
    returns errors for critical mismatches and warnings for library differences
    """
    ref_meta = reference.get("meta", {})
    cand_meta = candidate.get("meta", {})

    errors = []
    warnings = []

    # keys that must match between runs
    match_keys = ["model", "conf", "iou"]

    for key in match_keys:
        if key not in ref_meta:
            errors.append(f"'{key}' missing from reference")

        if key not in cand_meta:
            errors.append(f"'{key}' missing from candidate")

        if key in ref_meta and key in cand_meta:
            if ref_meta[key] != cand_meta[key]:
                errors.append(
                    f"'{key}' mismatch: reference has {ref_meta[key]}, candidate has {cand_meta[key]}"
                )

    # check library versions (warnings only)
    if "library_versions" not in ref_meta:
        warnings.append("'library_versions' missing from reference")

    if "library_versions" not in cand_meta:
        warnings.append("'library_versions' missing from candidate")

    if "library_versions" in ref_meta and "library_versions" in cand_meta:
        ref_libs = ref_meta["library_versions"]
        cand_libs = cand_meta["library_versions"]

        if isinstance(ref_libs, dict) and isinstance(cand_libs, dict):
            all_packages = sorted(set(ref_libs.keys()) | set(cand_libs.keys()))

            for pkg in all_packages:
                if pkg not in ref_libs:
                    warnings.append(f"package '{pkg}' missing from reference")

                if pkg not in cand_libs:
                    warnings.append(f"package '{pkg}' missing from candidate")

                if pkg in ref_libs and pkg in cand_libs:
                    if ref_libs[pkg] != cand_libs[pkg]:
                        warnings.append(
                            f"package '{pkg}' mismatch: reference has {ref_libs[pkg]}, candidate has {cand_libs[pkg]}"
                        )

    return errors, warnings

