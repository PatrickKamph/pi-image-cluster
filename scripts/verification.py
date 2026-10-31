import json
from pathlib import Path
from collections import Counter
import math
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

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

def record_run():

    pass

def record_final(errors: list[str], warnings: list[str], data: list[dict[str, Any]]) -> None:

    structured_output = {
        'meta': [
            {'errors' : errors},
            {'warnings' : warnings}
        ],

        'results': [
            data
        ]

    }

    with open(RESULTS_DIR/"test1.json", "w", encoding="utf-8") as json_file:

        json.dump(structured_output, json_file, indent=4)

    pass


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


def compare_detections(serial_results: dict, parallel_results: dict, image_list: list[str]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """
    Compares the results of run_serial() and run_parallel() and outputs statistics about them.

    args:
        serial_results - The output from run_serial()
        parallel_results - The output from run_parallel()
        image_list - A list of strings which contain the file name/suffix of every image in /data/

    returns:
        warnings - A list of all images in /data/ that have not been processed by run_serial()
        errors - A list of all images in /data/ that have not been processed by run_parallel()
        output_list - A list of dictionaries containing comparison statistics of serial and parallel

    """



    serial_keys = set(serial_results.keys())
    parallel_keys = set(parallel_results.keys())

    warnings = [image for image in image_list if image not in serial_keys]  # reference
    errors = [image for image in image_list if image not in parallel_keys]  # candidate



    output_list = []

    # If we check by image list instead, if there is any missing image we skipped then it will break which would be bad probably
    all_files = set(serial_keys.union(parallel_keys))

    for file_name in all_files:
        diff_dict = {}


        s_boxes, p_boxes = serial_results.get(file_name), parallel_results.get(file_name)

        def freeze(boxes):
            return [tuple(sorted(b.items())) for b in boxes]

        s_frozen = freeze(s_boxes)
        p_frozen = freeze(p_boxes)

        counts_s = Counter(s_frozen)
        counts_p = Counter(p_frozen)

        remain = counts_s - counts_p # Any missing
        remain_b = counts_p - counts_s # Any additions

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


        # If they completely match then we keep going so it doesnt accidentally set the file as having a discrepancy.
        if s_boxes == p_boxes:
            continue
        else:
            diff_dict['file'] = file_name
            diff_dict['match'] = False
            diff_dict['avg_conf_difference'] = diff
            diff_dict['missing'] = list(remain.elements())
            diff_dict['extra'] = list(remain_b.elements())

        output_list.append(diff_dict)








    return errors, warnings, output_list



""" Getting the avg confidence if needed in future
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
"""

""" Would output any missing detections and any additional detections
counts_s = Counter(s_boxes)
counts_p = Counter(p_boxes)

# serial = [1, 2, 2, 3]
# parallel = [1, 2, 3, 4]

# parallel is missing a 2 but has an extra number 4
# [missing: [2], additional : [4]


remain = counts_s - counts_p # Any missing
remain_b = counts_p - counts_s # Any additions

combined = remain + remain_b

diff_dict['missing'] = list(remain.elements())
diff_dict['extra'] = list(remain_b.elements())

diff_dict[file_name] = list(combined.elements())

"""