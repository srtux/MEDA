"""Normalize raw ``failed_constraints`` strings into stable error signatures.

A *signature* is a coarse, deterministic class label (no LLM involved) used to
group similar failures so the learning store can deduplicate lessons and match
a current failure against past ones. Keep these stable: changing the mapping
invalidates dedup grouping for existing stored lessons.
"""

import re

# Ordered (regex, signature) rules — first match wins.
_RULES = [
    # --- compile / runtime errors -------------------------------------------
    (re.compile(r"NameError", re.I), "compile:name_error"),
    (re.compile(r"SyntaxError", re.I), "compile:syntax_error"),
    (re.compile(r"AttributeError", re.I), "compile:attribute_error"),
    (re.compile(r"TypeError", re.I), "compile:type_error"),
    (re.compile(r"ValueError", re.I), "compile:value_error"),
    (re.compile(r"IndexError", re.I), "compile:index_error"),
    (re.compile(r"ImportError|ModuleNotFound", re.I), "compile:import_error"),
    (re.compile(r"StdFail|OCP|BRep|Standard_|topo", re.I), "compile:kernel_error"),
    (re.compile(r"\[EXPORT_ERROR\]", re.I), "export:error"),
    (re.compile(r"\[METRICS_ERROR\]", re.I), "compile:no_model"),
    (re.compile(r"Timeout", re.I), "compile:timeout"),
    # --- geometric constraint misses ----------------------------------------
    (re.compile(r"\bvolume\b", re.I), "topology:volume_mismatch"),
    (re.compile(r"num_faces", re.I), "topology:num_faces_mismatch"),
    (re.compile(r"num_edges", re.I), "topology:num_edges_mismatch"),
    (re.compile(r"num_vertices", re.I), "topology:num_vertices_mismatch"),
    (re.compile(r"center_of_mass", re.I), "topology:com_mismatch"),
    (re.compile(r"chamfer|hausdorff|distance", re.I), "topology:distance_mismatch"),
    # --- visual critique -----------------------------------------------------
    (re.compile(r"Visual validation failure", re.I), "visual:mismatch"),
    (re.compile(r"Visual critique error", re.I), "visual:critique_error"),
]

_GENERIC_COMPILE = re.compile(r"Compile error|Compilation failure", re.I)


def to_signature(failed_constraint: str) -> str:
    """Map a single ``failed_constraints`` entry to a stable signature label."""
    if not failed_constraint:
        return "unknown"
    for pattern, sig in _RULES:
        if pattern.search(failed_constraint):
            return sig
    if _GENERIC_COMPILE.search(failed_constraint):
        return "compile:other"
    return "unknown"
