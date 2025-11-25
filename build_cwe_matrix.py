#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


def load_codeql_cwe_map(allqueries_path: Path) -> Dict[str, Set[str]]:
    with allqueries_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    queries = data.get("queries", []) if isinstance(data, dict) else []
    id_to_cwes: Dict[str, Set[str]] = {}
    def normalize_cwe_code(raw: str) -> str:
        digits = "".join(ch for ch in (raw or "") if ch.isdigit())
        if not digits:
            return ""
        return f"CWE{int(digits)}"

    for q in queries:
        if not isinstance(q, dict):
            continue
        meta = q.get("metadata") or {}
        analysis_id = meta.get("id")
        if not analysis_id:
            continue
        tags_field = meta.get("tags")
        cwes: Set[str] = set()
        if isinstance(tags_field, str):
            # multi-line string: split by whitespace/newlines
            parts = [p.strip() for p in tags_field.split() if p.strip()]
        elif isinstance(tags_field, list):
            parts = []
            for t in tags_field:
                if isinstance(t, str):
                    parts.extend(t.split())
        else:
            parts = []

        for t in parts:
            t_lower = t.lower()
            # tags often like external/cwe/cwe-190 or external/cwe/cwe-327
            if t_lower.startswith("external/cwe/cwe-"):
                num = t_lower.split("external/cwe/cwe-", 1)[1]
                norm = normalize_cwe_code(num)
                if norm:
                    cwes.add(norm)
        id_to_cwes[analysis_id] = cwes

    return id_to_cwes


def load_results(results_csv_path: Path) -> Tuple[List[Dict[str, str]], Set[str]]:
    rows: List[Dict[str, str]] = []
    dataset_cwes: Set[str] = set()
    def normalize_cwe_code(raw: str) -> str:
        digits = "".join(ch for ch in (raw or "") if ch.isdigit())
        if not digits:
            return ""
        return f"CWE{int(digits)}"

    with results_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expected columns: testcase_name,testcase_CWE,result1,result2,...
        for r in reader:
            rows.append(r)
            testcase_cwe = (r.get("testcase_CWE") or "").strip()
            if testcase_cwe:
                dataset_cwes.add(normalize_cwe_code(testcase_cwe))
            # Also collect any detected CodeQL result ids as CWE categories later
    return rows, dataset_cwes


def collect_codeql_analysis_ids(result_rows: List[Dict[str, str]]) -> Set[str]:
    analysis_ids: Set[str] = set()
    for r in result_rows:
        for k, v in r.items():
            if k in ("testcase_name", "testcase_CWE"):
                continue
            value = (v or "").strip()
            if value:
                # These are CodeQL analysis ids like java/error-message-exposure, java/xss, etc.
                for entry in value.split(";"):
                    entry = entry.strip()
                    if entry:
                        analysis_ids.add(entry)
    return analysis_ids


def build_cwe_universe(dataset_cwes: Set[str], analysis_ids: Set[str], id_to_cwes: Dict[str, Set[str]]) -> List[str]:
    cwe_set: Set[str] = set(dataset_cwes)
    for aid in analysis_ids:
        cwe_set.update(id_to_cwes.get(aid, set()))
    # Sort numerically by CWE number
    def cwe_key(c: str) -> Tuple[int]:
        digits = "".join(ch for ch in c if ch.isdigit())
        return (int(digits or 0),)

    ordered = sorted(cwe_set, key=cwe_key)
    return ordered


def make_header(cwes: List[str]) -> List[str]:
    def cwe_number_label(cwe: str) -> str:
        return "".join(ch for ch in cwe if ch.isdigit())
    return [
        "type",  # summary-codeql, summary-dataset, analysis, testcase
        "name",  # analysis id or testcase name
        "testcase_CWE",  # only for testcase rows
        "SUCCESS",  # moved next to testcase_CWE
        "detectable_by_CodeQL",
        *[cwe_number_label(c) for c in cwes],
    ]


def row_for_summary(label: str, present_cwes: Set[str], cwes: List[str]) -> List[str]:
    row = [label, label, "", "", ""]
    for c in cwes:
        row.append("X" if c in present_cwes else "")
    return row


def row_for_analysis(aid: str, supported: Set[str], cwes: List[str]) -> List[str]:
    row = ["analysis", aid, "", "", ""]
    for c in cwes:
        row.append("X" if c in supported else "")
    return row


def row_for_intersection(label: str, a: Set[str], b: Set[str], cwes: List[str]) -> List[str]:
    both = a.intersection(b)
    row = [label, label, "", "", ""]
    for c in cwes:
        row.append("X" if c in both else "")
    return row


def evaluate_testcase_row(r: Dict[str, str], cwes: List[str], id_to_cwes: Dict[str, Set[str]], codeql_supported_cwes: Set[str]) -> List[str]:
    testcase_name = (r.get("testcase_name") or "").strip()
    # Normalize testcase CWE
    def normalize_cwe_code(raw: str) -> str:
        digits = "".join(ch for ch in (raw or "") if ch.isdigit())
        if not digits:
            return ""
        return f"CWE{int(digits)}"
    testcase_cwe = normalize_cwe_code((r.get("testcase_CWE") or "").strip())
    is_good = testcase_name.endswith("_good")

    # Aggregate all detections across analyses in the row
    detected_cwes: Set[str] = set()
    has_detection_for_test_cwe = False

    for k, v in r.items():
        if k in ("testcase_name", "testcase_CWE"):
            continue
        value = (v or "").strip()
        if not value:
            continue
        for entry in value.split(";"):
            aid = entry.strip()
            if not aid:
                continue
            cwes_for_a = id_to_cwes.get(aid, set())
            detected_cwes.update(cwes_for_a)
            if testcase_cwe and testcase_cwe in cwes_for_a:
                has_detection_for_test_cwe = True

    # For each CWE column, mark FP/FN/TP cell semantics don't apply; columns are X marks only
    # SUCCESS/FAIL semantics:
    # - bad testcase: SUCCESS if detected; FAIL if missed
    # - good testcase: blank if not detected (true negative), FAIL if detected
    if is_good:
        success = "FAIL" if has_detection_for_test_cwe else ""
    else:
        success = "SUCCESS" if has_detection_for_test_cwe else "FAIL"

    detectable_flag = "X" if (testcase_cwe and testcase_cwe in codeql_supported_cwes) else ""
    row_prefix = ["testcase", testcase_name, testcase_cwe, success, detectable_flag]
    mark_cols: List[str] = []
    for c in cwes:
        if c == testcase_cwe:
            if is_good:
                # For good cases, detection of the testcase CWE is a false positive; otherwise blank (true negative)
                mark_cols.append("FP" if has_detection_for_test_cwe else "")
            else:
                # For bad cases, detection of the testcase CWE is a true positive; otherwise false negative
                mark_cols.append("TP" if has_detection_for_test_cwe else "FN")
        else:
            # Any detection of a different CWE is still an FP regardless of good/bad
            mark_cols.append("FP" if c in detected_cwes else "")

    return row_prefix + mark_cols


def build_matrix(allqueries_path: Path, results_csv_path: Path, output_csv_path: Path) -> None:
    id_to_cwes = load_codeql_cwe_map(allqueries_path)
    result_rows, dataset_cwes = load_results(results_csv_path)
    analysis_ids_in_results = collect_codeql_analysis_ids(result_rows)
    cwes = build_cwe_universe(dataset_cwes, analysis_ids_in_results, id_to_cwes)

    header = make_header(cwes)

    # Summary rows
    codeql_supported_cwes: Set[str] = set()
    for aid in analysis_ids_in_results:
        codeql_supported_cwes.update(id_to_cwes.get(aid, set()))

    rows_out: List[List[str]] = []
    rows_out.append(row_for_summary("detectable by CodeQL", codeql_supported_cwes, cwes))
    rows_out.append(row_for_summary("contained in Dataset", dataset_cwes, cwes))
    rows_out.append(row_for_intersection("in Dataset and detectable by CodeQL", dataset_cwes, codeql_supported_cwes, cwes))

    # Analysis rows
    for aid in sorted(analysis_ids_in_results):
        rows_out.append(row_for_analysis(aid, id_to_cwes.get(aid, set()), cwes))

    # Testcase rows sorted by: CWE number asc, bad before good, then name
    def tc_sort_key(x: Dict[str, str]):
        name = (x.get("testcase_name") or "")
        cwe_raw = (x.get("testcase_CWE") or "")
        cwe_num = int("".join(ch for ch in cwe_raw if ch.isdigit()) or 0)
        is_good = 1 if name.endswith("_good") else 0  # bad(0) before good(1)
        return (cwe_num, is_good, name)

    for r in sorted(result_rows, key=tc_sort_key):
        rows_out.append(evaluate_testcase_row(r, cwes, id_to_cwes, codeql_supported_cwes))

    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_out)


def build_cwe_evaluation(
    allqueries_path: Path,
    results_csv_path: Path,
    output_eval_csv_path: Path,
) -> None:
    id_to_cwes = load_codeql_cwe_map(allqueries_path)
    result_rows, dataset_cwes = load_results(results_csv_path)
    analysis_ids_in_results = collect_codeql_analysis_ids(result_rows)
    codeql_supported_cwes: Set[str] = set()
    for aid in analysis_ids_in_results:
        codeql_supported_cwes.update(id_to_cwes.get(aid, set()))

    # Precompute detections per row
    prepped: List[Tuple[str, str, bool, Set[str]]] = []
    def normalize_cwe_code(raw: str) -> str:
        digits = "".join(ch for ch in (raw or "") if ch.isdigit())
        if not digits:
            return ""
        return f"CWE{int(digits)}"

    for r in result_rows:
        name = (r.get("testcase_name") or "").strip()
        testcase_cwe = normalize_cwe_code((r.get("testcase_CWE") or "").strip())
        is_good = name.endswith("_good")
        detected_cwes: Set[str] = set()
        for k, v in r.items():
            if k in ("testcase_name", "testcase_CWE"):
                continue
            value = (v or "").strip()
            if not value:
                continue
            for entry in value.split(";"):
                aid = entry.strip()
                if not aid:
                    continue
                detected_cwes.update(id_to_cwes.get(aid, set()))
        prepped.append((name, testcase_cwe, is_good, detected_cwes))

    # Aggregate per CWE
    rows_out: List[List[str]] = []
    header = ["CWE", "detectable_by_CodeQL", "tp", "fn", "fp", "bad_cases", "good_cases", "judgement"]

    for cwe in sorted(dataset_cwes, key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or 0)):
        tp = 0
        fn = 0
        fp = 0
        bad_cases = 0
        good_cases = 0
        for name, tc_cwe, is_good, detected in prepped:
            if tc_cwe == cwe:
                if is_good:
                    good_cases += 1
                    if cwe in detected:
                        fp += 1  # detecting CWE in good testcase is FP
                else:
                    bad_cases += 1
                    if cwe in detected:
                        tp += 1
                    else:
                        fn += 1

        # Judgement
        if tp == 0:
            judgement = "unsupported"
        elif fn == 0 and fp == 0:
            judgement = "perfect support"
        elif fn > 0 and fp > 0:
            judgement = "partial support (imprecise and unsound)"
        elif fn > 0:
            judgement = "partial support (unsound)"
        elif fp > 0:
            judgement = "partial support (imprecise)"
        else:
            judgement = "partial support"

        detectable_flag = "X" if cwe in codeql_supported_cwes else ""
        rows_out.append([cwe, detectable_flag, str(tp), str(fn), str(fp), str(bad_cases), str(good_cases), judgement])

    with output_eval_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_out)


def main(argv: List[str]) -> int:
    base = Path(__file__).resolve().parent
    allqueries_path = base / "allqueries.yml"
    results_csv_path = base / "results.csv"
    output_csv_path = base / "results_smart.csv"
    output_eval_csv_path = base / "results_cwe_eval.csv"

    if len(argv) >= 2:
        allqueries_path = Path(argv[1])
    if len(argv) >= 3:
        results_csv_path = Path(argv[2])
    if len(argv) >= 4:
        output_csv_path = Path(argv[3])
    if len(argv) >= 5:
        output_eval_csv_path = Path(argv[4])

    build_matrix(allqueries_path, results_csv_path, output_csv_path)
    build_cwe_evaluation(allqueries_path, results_csv_path, output_eval_csv_path)
    print(f"Wrote {output_csv_path}")
    print(f"Wrote {output_eval_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


