"""Tests for the sandbox AST allow-list validator (no cadquery needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sandbox import validate_code
from core.geometry_introspection import INTROSPECTION_SNIPPET


def test_allows_safe_cadquery_code():
    code = (
        "import cadquery as cq\n"
        "import math\n"
        "width = 10\n"
        "model = cq.Workplane('XY').box(width, width, width)\n"
        "model = model.faces('>Z').workplane().hole(2)\n"
        "model = model.edges('|Z').fillet(1)\n"
    )
    assert validate_code(code) is None


def test_blocks_os_import():
    assert validate_code("import os\nos.system('rm -rf /')") is not None


def test_blocks_from_subprocess_import():
    assert validate_code("from subprocess import run\nrun(['ls'])") is not None


def test_blocks_eval():
    assert validate_code("x = eval('1+1')") is not None


def test_blocks_open():
    assert validate_code("open('/etc/passwd').read()") is not None


def test_blocks_dunder_escape():
    assert validate_code("().__class__.__bases__[0].__subclasses__()") is not None


def test_blocks_getattr():
    assert validate_code("getattr(obj, '__globals__')") is not None


def test_syntax_error_is_not_rejected_here():
    # We defer real SyntaxErrors to the interpreter's own error path.
    assert validate_code("model = (((") is None


def test_introspection_snippet_passes_validation():
    # The introspection snippet is appended to user code and must not trip the
    # validator, otherwise inspect_current_model would always be blocked.
    assert validate_code(INTROSPECTION_SNIPPET) is None


def test_numpy_allowed():
    assert validate_code("import numpy as np\nx = np.array([1,2,3])") is None


# --- escape vectors found by the adversarial review --------------------------

def test_blocks_indirect_eval_reference():
    assert validate_code("_run = eval\n_run('__import__(\"os\")')") is not None


def test_blocks_eval_in_list_literal():
    assert validate_code("[eval][0]('1+1')") is not None


def test_blocks_map_exec():
    assert validate_code("list(map(exec, ['import os']))") is not None


def test_blocks_default_arg_eval():
    assert validate_code("def f(a=eval):\n    return a") is not None


def test_blocks_operator_import():
    # operator.attrgetter('__globals__') is a string-based escape vector.
    assert validate_code("import operator\nx = operator.attrgetter('__globals__')") is not None


def test_blocks_traceback_frame_walk():
    payload = (
        "try:\n"
        "    raise ValueError()\n"
        "except ValueError as _e:\n"
        "    _bi = _e.__traceback__.tb_frame.f_builtins\n"
    )
    assert validate_code(payload) is not None


def test_blocks_format_string_dunder():
    assert validate_code("evil = '{0.__globals__}'.format(open)") is not None


def test_allows_future_import():
    code = "from __future__ import annotations\nimport cadquery as cq\nmodel = cq.Workplane('XY').box(1,1,1)"
    assert validate_code(code) is None


def test_store_context_to_banned_name_is_allowed():
    # Assigning to a name that shadows a banned builtin (Store ctx) is fine as
    # long as it is never referenced (Load); only Load references are an escape.
    assert validate_code("vars = [1, 2, 3]\nmodel = cq.Workplane('XY').box(1,1,1)") is None


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run()
