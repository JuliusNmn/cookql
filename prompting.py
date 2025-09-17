"""Prompt construction utilities for CookQLAgent."""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import zipfile
import json

def validation_error_repair_prompt(error_message: Any, files: Dict[str, str]) -> str:
    """
    Build a prompt instructing the agent to fix a CodeQL validation error in the
    provided in-memory file set.

    Args:
        error_message: Validation error string or structured message.
        files: Mapping of filename to file contents for all current files.

    Returns:
        A single instruction string suitable for passing as the agent input.
    """
    if isinstance(error_message, list):
        error_text = "\n".join(str(m) for m in error_message)
    else:
        error_text = str(error_message)

    # Provide a concise directory view for context
    file_list_preview: List[str] = []
    for name in sorted(files.keys()):
        file_list_preview.append(f"- {name}")
    file_overview = "\n".join(file_list_preview)

    instructions = (
        "You are assisting with repairing CodeQL query validation errors.\n"
        "Goal: Fix the validation/compilation errors so queries compile successfully.\n\n"
        "Constraints and guidelines:\n"
        "- Make minimal, targeted edits necessary to fix the errors.\n"
        "- Preserve existing intent and structure where possible.\n"
        "- Only modify .ql, .qll, or related config files as needed.\n"
        "- Ensure imports/library references are correct for CodeQL Java queries.\n"
        "- If missing dependencies are referenced, correct the import paths or adjust the code accordingly.\n"
        "- When creating or editing files, write complete, compilable content.\n\n"
        "Current validation error message(s):\n"
        f"{error_text}\n\n"
        "Files currently in the workspace (in-memory):\n"
        f"{file_overview}\n\n"
        "Task: Apply the necessary changes to resolve the validation errors."
        "Please output the patched files using the tools provided."
    )

    return instructions


def describe_failures(databases: List[Tuple[str, Any]], results: List[Dict[str, Any]]) -> str:
    """
    Build a large text blob describing false negatives and false positives.
    For each false negative (bad variant with 0 results) and false positive
    (good variant with >0 results), extract the corresponding testcase files
    from the database's src.zip and include their contents. For false positives,
    also include the query results emitted by CodeQL.

    Args:
        databases: List of (project_name, database_path) tuples.
        results: List of QueryResult objects or dicts.

    Returns:
        A concatenated string with sections for each failure case.
    """
    # Map project name to its database Path for locating src.zip
    name_to_db = {name: Path(path) for name, path in databases}

    sections: List[str] = []

    def _safe_read_from_zip(zf: zipfile.ZipFile, member: str) -> str:
        candidates = [member.lstrip("/\\"), member.replace("\\", "/").lstrip("/\\")]
        for cand in candidates:
            try:
                with zf.open(cand) as f:
                    return f.read().decode("utf-8", errors="ignore")
            except KeyError:
                continue
            except Exception:
                break
        return "<unreadable>"

    def _collect_uris_from_result_item(item: Dict[str, Any]) -> List[str]:
        uris: List[str] = []
        try:
            locs = item.get("locations", []) if isinstance(item, dict) else []
            for loc in locs:
                phys = (loc or {}).get("physicalLocation", {})
                art = phys.get("artifactLocation", {})
                uri = art.get("uri")
                if uri:
                    uris.append(uri)
            # Optionally include relatedLocations
            related = item.get("relatedLocations", []) if isinstance(item, dict) else []
            for rl in related:
                phys = (rl or {}).get("physicalLocation", {})
                art = phys.get("artifactLocation", {})
                uri = art.get("uri")
                if uri:
                    uris.append(uri)
        except Exception:
            pass
        # Deduplicate while preserving order
        seen = set()
        ordered: List[str] = []
        for u in uris:
            if u not in seen:
                seen.add(u)
                ordered.append(u)
        return ordered
    
    for r in results:
        project_name = getattr(r, "project_name", None) or (r.get("project_name") if isinstance(r, dict) else None)
        variant = getattr(r, "variant", None) or (r.get("variant") if isinstance(r, dict) else None)
        result_count = getattr(r, "result_count", None) or (r.get("result_count", 0) if isinstance(r, dict) else 0)
        query_results = getattr(r, "results", None) or (r.get("results", []) if isinstance(r, dict) else [])
        db_path_str = getattr(r, "database_path", None) or (r.get("database_path") if isinstance(r, dict) else None)

        try:
            rc_int = int(result_count)
        except Exception:
            rc_int = 0

        is_false_negative = variant == 'bad' and rc_int == 0
        is_false_positive = variant == 'good' and rc_int > 0

        if not (is_false_negative or is_false_positive):
            continue

        header = "False Negative" if is_false_negative else "False Positive"
        section_lines: List[str] = [f"=== {header}: {project_name} ==="]

        # Determine src.zip path
        db_path = name_to_db.get(project_name)
        if db_path is None and db_path_str:
            db_path = Path(db_path_str)
        src_zip = (db_path / "src.zip") if db_path is not None else None

        if src_zip and src_zip.exists():
            try:
                with zipfile.ZipFile(src_zip, 'r') as zf:
                    # Select files to include
                    members: List[str] = []
                    
                    # Additionally include any archive entries whose name contains the project name
                    # or the project base name (with _bad/_good suffix removed)
                    project_base = project_name or ""
                    if project_base.endswith('_bad'):
                        project_base = project_base[:-4]
                    elif project_base.endswith('_good'):
                        project_base = project_base[:-5]

                    related = [
                        n for n in zf.namelist()
                        if (project_name and project_name in n.split("/")[-1]) or (project_base and project_base in n.split("/")[-1])
                    ]
                    if related:
                        existing = set(members)
                        for n in related:
                            if n not in existing:
                                members.append(n)
                                existing.add(n)

                    if not members:
                        section_lines.append("[info] No source files found in src.zip.")
                    else:
                        for name in members:
                            content = _safe_read_from_zip(zf, name)
                            section_lines.append(f"\n--- FILE: {name} ---\n{content}")
            except Exception as e:
                section_lines.append(f"[warn] Could not read src.zip: {e}")
        else:
            section_lines.append("[info] src.zip not found for this database.")

        if is_false_positive and query_results:
            section_lines.append("\n--- Query Results (False Positive) ---")
            section_lines.append(json.dumps(query_results, indent=2))
            for idx, item in enumerate(query_results, start=1):
                try:
                    if isinstance(item, dict):
                        msg = item.get('message', {})
                        msg_text = msg.get('text') if isinstance(msg, dict) else str(msg)
                        rule_id = item.get('ruleId') or (item.get('rule', {}) or {}).get('id')
                        if rule_id or msg_text:
                            prefix = f"[{rule_id}] " if rule_id else ""
                            section_lines.append(f"{idx}. {prefix}{msg_text or ''}")
                        else:
                            section_lines.append(f"{idx}. {item}")
                    else:
                        section_lines.append(f"{idx}. {str(item)}")
                except Exception:
                    section_lines.append(f"{idx}. <unprintable result>")

        sections.append("\n".join(section_lines))

    return "\n\n".join(sections) if sections else ""


def build_refinement_prompt(error_analysis: str, files: Dict[str, str]) -> str:
    """
    Build a prompt instructing the agent to refine the queries based on the
    provided false positive/false negative analysis.

    Args:
        error_analysis: Large text analysis created by describe_failures.
        files: Mapping of filename to content for current workspace files.

    Returns:
        Instruction text to guide refinement edits.
    """
    file_list_preview: List[str] = []
    for name in sorted(files.keys()):
        file_list_preview.append(f"- {name}")
    file_overview = "\n".join(file_list_preview)

    instructions = (
        "You are refining CodeQL queries to reduce false positives and false negatives.\n"
        "Use the analysis below to make targeted improvements, without overfitting.\n\n"
        "Guidelines:\n"
        "- For false negatives (bad variants undetected): broaden or correct the query to catch the intended patterns.\n"
        "- For false positives (good variants flagged): tighten conditions, improve taint steps/sources/sinks, or add guards.\n"
        "- Keep changes minimal and maintain readability and performance.\n"
        "- Ensure queries compile and leverage standard CodeQL libraries where appropriate.\n\n"
        "Failure analysis:\n"
        f"{error_analysis}\n\n"
        "Files currently in the workspace:\n"
        f"{file_overview}\n\n"
        "Task: Update the relevant .ql/.qll files to address the analysis using the tools provided."
    )

    return instructions

