import json
import os
from pathlib import Path
from collections import Counter
from typing import Any
import platform
import sys
import torch
import ultralytics
from datetime import datetime

from cluster.worker import MODEL_PATH, CONF_THRESHOLD



PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "json_results"

def build_meta() -> dict[str, Any]:
    """
    Builds the meta portion of all JSON files.

    Returns:
    A dictionary with the specification on the left and the info on the right

    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        'run_timestamp': timestamp,
        'os': platform.system(),
        'cpu_count': os.cpu_count() or 1,
        'runtime': {
            'python_version': sys.version.split()[0],
            'torch_version': torch.__version__,
            'ultralytics_version': ultralytics.__version__
        },
        'model_parameters': {
            'model_name': MODEL_PATH.name,
            'confidence_threshold': CONF_THRESHOLD
        }
    }

def load_run(path):
    """
    opens and parses JSON file, validates top-level structure is dictionary
    checks to see if meta & results are in dictionary.

    Args:
        path: The file path of the JSON file we are loading

    Returns:
        A dictionary of the JSON file contents that have been verified as working.
        If it is missing required keys, then it will raise an error.


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

def record_run(run_type: str, results: dict[str, Any]) -> Path:
    """
    Saves serial/parallel as a JSON file for outside readability.

    Args:
        run_type: If its serial or parallel
        results: The results of either serial or parallel

    Returns:
        Path to the saved JSON file of serial or parallel containing their results

    """

    if run_type not in {'serial', 'parallel'}:
        raise ValueError("Invalid run_type! It must be either serial or parallel!")

    output = {
        'meta': build_meta(),
        'results': results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"{run_type}_run.json"

    with open(out_file, 'w', encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    return out_file


def check_meta(reference: dict, candidate: dict) -> tuple[list[str], list[str]]:
    """
    Compares metadata between reference(serial) and candidate(parallel) runs.

    Args:
        reference: A dictionary containing the dataset of reference (serial baseline)
        candidate: A dictionary containing the dataset of candidate (parallel baseline)

    Returns:
        returns errors for critical mismatches and warnings for library differences.

    """
    ref_meta = reference.get("meta", {})
    cand_meta = candidate.get("meta", {})

    if not isinstance(ref_meta, dict):
        return ["reference 'meta' is not a dictionary"], []
    
    if not isinstance(cand_meta, dict):
        return ["candidate 'meta' is not a dictionary"], []

    errors = []
    warnings = []

    ref_params = ref_meta.get("model_parameters", {}) # serial
    cand_params = cand_meta.get("model_parameters", {}) # parallel

    # keys that must match between runs
    model_keys = ["model_name", "confidence_threshold"]

    for key in model_keys:
        if key not in ref_params:
            errors.append(f"'{key}' missing from reference")

        elif key not in cand_params:
            errors.append(f"'{key}' missing from candidate")

        elif ref_params[key] != cand_params[key]:
            errors.append(f"'{key}' mismatch: reference has {ref_params[key]}, candidate has {cand_params[key]}")

    ref_runtime = ref_meta.get("runtime", {})
    cand_runtime = cand_meta.get("runtime", {})

    for package in ["torch_version", "ultralytics_version", "python_version"]:
        if ref_runtime.get(package) != cand_runtime.get(package):
            warnings.append(f"{package} mismatch: reference has {ref_runtime.get(package)}, candidate has {cand_runtime.get(package)}")

    return errors, warnings


def compare_detections(serial_file: Path | str, parallel_file: Path | str, image_list: list[str]) -> dict[str, Any]:
    """
    Compares the JSON files of serial & parallel and outputs statistics about them.

    args:
        serial_file: Path to serial_run.json
        parallel_file: Path to parallel_run.json
        image_list: A list of strings which contain the file name/suffix of every image in /data/

    returns:
        A dictionary containing:
            - 'meta_errors': Critical metadata mismatches between serial and parallel runs.
            - 'meta_warnings': Library version differences between runs.
            - 'missing_in_serial': Images in image_list not processed by serial run.
            - 'missing_in_parallel': Images in image_list not processed by parallel run.
            - 'detection_discrepancies': List of dictionaries containing comparison statistics and detection differences.
            - 'serial_length': Total number of images processed in the serial baseline.

    """



    serial_path = Path(serial_file)
    parallel_path = Path(parallel_file)

    if not serial_path.is_file():
        raise FileNotFoundError(f"Serial run file not found at {serial_path.resolve()}")
    if not parallel_path.is_file():
        raise FileNotFoundError(f"Parallel run file not found at {parallel_path.resolve()}")

    serial_data = load_run(serial_path)
    parallel_data = load_run(parallel_path)

    meta_errors, meta_warnings = check_meta(serial_data, parallel_data)

    serial_results = serial_data.get("results", {})
    parallel_results = parallel_data.get("results", {})

    serial_keys = set(serial_results.keys())
    parallel_keys = set(parallel_results.keys())

    warnings = [image for image in image_list if image not in serial_keys]  # reference
    errors = [image for image in image_list if image not in parallel_keys]  # candidate

    detection_discrepancies = []

    # If we check by image list instead, if there is any missing image we skipped then it will break.
    all_files = set(serial_keys.union(parallel_keys))

    def freeze(boxes):
        if not boxes:
            return []
        return [tuple(sorted(b.items())) for b in boxes]

    for file_name in all_files:


        s_boxes, p_boxes = serial_results.get(file_name), parallel_results.get(file_name)

        # If they completely match then we keep going so it doesn't accidentally set the file as having a discrepancy.
        if s_boxes == p_boxes:
            continue



        s_boxes_list = s_boxes if s_boxes else []
        p_boxes_list = p_boxes if p_boxes else []

        s_frozen = freeze(s_boxes_list)
        p_frozen = freeze(p_boxes_list)

        counts_s = Counter(s_frozen)
        counts_p = Counter(p_frozen)

        remain = counts_s - counts_p # Any missing
        remain_b = counts_p - counts_s # Any additions

        # The avg for serial
        if s_boxes_list:
            s_avg = sum(item['confidence'] for item in s_boxes) / len(s_boxes)
        else:
            s_avg = 0.0

        # The avg for parallel
        if p_boxes_list:
            p_avg = sum(item['confidence'] for item in p_boxes) / len(p_boxes)
        else:
            p_avg = 0.0

        diff = abs(p_avg - s_avg)

        detection_discrepancies.append(
            {
                "file": file_name,
                "avg_conf_difference": round(diff, 4),
                'missing': [dict(item) for item in remain.elements()],
                'extra': [dict(item) for item in remain_b.elements()]
            }

        )

    return {
        "meta_errors": meta_errors,
        "meta_warnings": meta_warnings,
        "missing_in_serial": warnings,
        "missing_in_parallel": errors,
        "detection_discrepancies": detection_discrepancies,
        "serial_length": len(serial_results)
    }

def record_final(summary_data: dict[str, Any]) -> str:
    """
    Formats & prints the comparison summary

    Args:
        summary_data: The data from compare_detections()

    Returns:
        A string containing all summary comparison data


    """

    percentage_flag = False

    lines = [
        "=====================================================",
        "  Final Report  ",
        "=====================================================",
        "1. Metadata Comparison:"
    ]

    # Checks to see if meta has any differences

    meta_errors = summary_data.get("meta_errors", [])
    meta_warnings = summary_data.get("meta_warnings", [])

    if not meta_errors and not meta_warnings:
        lines.append("    -Perfect match between serial & parallel meta.")
    else:
        percentage_flag = True

        for error in meta_errors:
            lines.append(f"    Meta Error: {error}")
        for warning in meta_warnings:
            lines.append(f"    Meta Warning: {warning}")


    # Checks to see if file is missing

    lines.append("\n2. File Completion Check:  ")
    missing_serial = summary_data.get("missing_in_serial") or []
    missing_parallel = summary_data.get("missing_in_parallel") or []

    if not missing_serial and not missing_parallel:
        lines.append("    -All images ran in both serial and parallel.")
    else:
        if missing_serial:
            lines.append(f"    Missing in Serial Baseline: {missing_serial}")
        if missing_parallel:
            lines.append(f"    Missing in Parallel Baseline: {missing_parallel}")

    total_expected = summary_data.get("serial_length", 0) + len(missing_serial)

    if total_expected > 0:
        unique_missing = len(set(missing_serial) | set(missing_parallel))
        file_percentage = (max(0, total_expected - unique_missing) / total_expected) * 100
    else:
        file_percentage = 100.0

    lines.append(f"    Serial/Parallel File Match: {file_percentage:.2f}%")




    # Gives all object detection mismatches

    lines.append("\n3. Object Detection Discrepancies:  ")
    discrepancies = summary_data.get("detection_discrepancies", [])
    total_evaluated = summary_data.get("serial_length", 0)

    if not discrepancies:
        lines.append("    No object or confidence differences across all images.")
        object_percentage = 100.0
    else:
        lines.append(f"    Found {len(discrepancies)} file(s) with different detections!")

        for item in discrepancies:
            lines.append(f"        - File: {item['file']}")
            lines.append(f"          Avg Confidence Difference: {item['avg_conf_difference']}")

            if item["missing"]:
                lines.append(f"          Missing from Parallel: {item['missing']}")
            if item['extra']:
                lines.append(f"          Missing from Serial: {item['extra']}")

        matching_evaluated = max(0, total_evaluated - len(discrepancies))
        object_percentage = (matching_evaluated / total_evaluated) * 100 if total_evaluated > 0 else 100.0

    lines.append(f"    Serial/Parallel Object Match: {object_percentage:.2f}%")



    lines.append(f"")
    lines.append(f"Summary:")
    lines.append(f"    Metadata Match across Serial/Parallel")
    lines.append(f"    File Match: {file_percentage:.2f}%")
    lines.append(f"    Object Match: {object_percentage:.2f}%")

    lines.append("")
    lines.append("=====================================================")

    output_str = "\n".join(lines)

    print(output_str)

    return output_str

