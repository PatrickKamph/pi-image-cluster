import json
from pathlib import Path
import math

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

    if not isinstance(ref_meta, dict):
        return ["reference 'meta' is not a dictionary"], []
    
    if not isinstance(cand_meta, dict):
        return ["candidate 'meta' is not a dictionary"], []

    errors = []
    warnings = []

    # keys that must match between runs
    match_keys = ["model", "conf", "iou"]

    for key in match_keys:
        if key not in ref_meta:
            errors.append(f"'{key}' missing from reference")

        elif key not in cand_meta:
            errors.append(f"'{key}' missing from candidate")

        elif key in ref_meta and key in cand_meta:
            if ref_meta[key] != cand_meta[key]:
                errors.append(f"'{key}' mismatch: reference has {ref_meta[key]}, candidate has {cand_meta[key]}")

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
                        warnings.append(f"package '{pkg}' mismatch: reference has {ref_libs[pkg]}, candidate has {cand_libs[pkg]}")

        else:
            warnings.append("'library_versions' is not a dictionary in one of the runs")

    return errors, warnings


def compare_detections(serial_results: dict, parallel_results: dict, image_list: list[str]) -> tuple[
    list[str], list[str]]:
    serial_keys = set(serial_results.keys())
    parallel_keys = set(parallel_results.keys())

    warnings = [image for image in image_list if image not in serial_keys]  #
    errors = [image for image in image_list if image not in parallel_keys]  # candidate

    return errors, warnings

    """ Returns tuple of total mismatch and avg confidence discrepancy. Temp code if we wanna use in future

        if serial_keys != parallel_keys:
        missing = serial_keys - parallel_keys
        extra = parallel_keys - serial_keys

        return False, f"Cannot iterate keys, they are mismatched. Missing {len(missing)} files, Have {len(extra)} extra files."


    diff_dict = {}

    all_files = set(serial_keys.union(parallel_keys))

    for file_name in all_files:
        s_boxes, p_boxes = serial_results.get(file_name), parallel_results.get(file_name)


        # If the file is missing in one or the other, set the total discrepancies to the size and the avg difference to 100%.
        if s_boxes is None or p_boxes is None:
            count = len(s_boxes) if s_boxes else len(p_boxes)

            diff_dict[file_name] = (count, 1.0)
            continue

        # If they completely match then we keep going so it doesnt accidentally set the file as having a discrepancy.
        if s_boxes == p_boxes:
            continue

        box_count_diff = abs(len(s_boxes) - len(p_boxes))


        # The avg for serial
        if len(s_boxes) > 0:
            s_avg = sum(item['confidence'] for item in s_boxes) / len(s_boxes)
        else:
            s_avg = 0.0

        # The avg for parallel
        if len(p_boxes) > 0:
            p_avg = sum(item['confidence'] for item in p_boxes) / len(p_boxes)
        else:
            p_avg = 0.0

        diff = abs(p_avg - s_avg)

        # Just to make sure any super small discrepancies dont flag as bad
        if diff > conf_tolerance:
            diff_dict[file_name] = round(diff, 4)

    return diff_dict
    """
