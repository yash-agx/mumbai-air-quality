"""Every project file must parse on the Python runtime.txt pins.

runtime.txt and CI now both say 3.12, raised because numpy 2.5 and scipy 1.18
do not build below it. Nothing imports app.py (it is a Streamlit script), so
compile() here is the only thing that would catch a file that does not parse.

The second check dates from the 3.11 floor: PEP 701 lifted the ban on reusing
the enclosing quote inside a replacement field, so `f"{d["k"]}"` parses on 3.12
and is a SyntaxError on 3.11, and app.py had exactly that. On 3.11 compile()
rejected it outright; on 3.12 it parses, so the token scan was what found it
there. That syntax is legal on the floor this project now pins, so the scan
guards a constraint that no longer binds and could be dropped.
"""

import io
import sys
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PROJECT_FILES = sorted(
    p for p in ROOT.rglob("*.py")
    if ".venv" not in p.parts and "__pycache__" not in p.parts
    and not p.name.startswith("_"))


def _delimiter(token):
    """The quote sequence that closes this f-string, e.g. '"' or '\"\"\"'."""
    text = token.string
    return text[-3:] if text[-3:] in ('"""', "'''") else text[-1]


def _clashes(inner, outer):
    """Could a string opened with `inner` terminate an f-string using `outer`?

    Inside f\"\"\"...\"\"\" a single " is fine on 3.11; only the full triple
    clashes. Inside f"..." any " clashes.
    """
    return inner == outer if len(outer) == 3 else inner[0] == outer[0]


def pep701_only(path):
    """Replacement fields that 3.11 rejects: reused quotes, backslash, comment."""
    found, stack = [], []
    with open(path, encoding="utf-8") as fh:
        for tok in tokenize.generate_tokens(io.StringIO(fh.read()).readline):
            if tok.type == tokenize.FSTRING_START:
                if stack and _clashes(_delimiter(tok), stack[-1]):
                    found.append((tok.start[0], "nested f-string reuses the quote"))
                stack.append(_delimiter(tok))
            elif tok.type == tokenize.FSTRING_END:
                stack.pop()
            elif stack and tok.type != tokenize.FSTRING_MIDDLE:
                if tok.type == tokenize.STRING:
                    quote = tok.string.lstrip("rRbBuUfF")
                    seq = quote[:3] if quote[:3] in ('"""', "'''") else quote[:1]
                    if _clashes(seq, stack[-1]):
                        found.append((tok.start[0], f"{tok.string} reuses the "
                                                    f"enclosing {stack[-1]}"))
                elif tok.type == tokenize.COMMENT:
                    found.append((tok.start[0], "comment in a replacement field"))
                elif "\\" in tok.string:
                    found.append((tok.start[0], "backslash in a replacement field"))
    return found


def test_the_scan_covers_the_files_that_matter():
    names = {p.name for p in PROJECT_FILES}
    assert {"app.py", "clean.py", "live.py", "interpolate.py"} <= names


@pytest.mark.parametrize("path", PROJECT_FILES,
                         ids=lambda p: p.relative_to(ROOT).as_posix())
def test_file_compiles_on_this_interpreter(path):
    """The parse check. On the old 3.11 floor this also caught PEP 701 syntax."""
    compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason="f-strings are one token before 3.12; compile() "
                           "already rejects the 3.12-only syntax there")
@pytest.mark.parametrize("path", PROJECT_FILES,
                         ids=lambda p: p.relative_to(ROOT).as_posix())
def test_file_has_no_312_only_fstrings(path):
    hits = pep701_only(path)
    assert not hits, "\n".join(f"line {line}: {why}" for line, why in hits)


def test_the_scan_detects_a_known_offender(tmp_path):
    """The check has to fail on the thing it is guarding against."""
    bad = tmp_path / "bad.py"
    bad.write_text('d = {"k": 1}\nprint(f"{d["k"]}")\n', encoding="utf-8")
    assert pep701_only(bad)

    ok = tmp_path / "ok.py"
    ok.write_text('d = {"k": 1}\nprint(f"{d[\'k\']}")\n', encoding="utf-8")
    assert not pep701_only(ok)


def test_triple_quoted_fstrings_may_use_single_quotes_inside(tmp_path):
    """Not everything that looks like reuse is: only the full triple clashes."""
    fine = tmp_path / "fine.py"
    fine.write_text('d = {"k": 1}\nprint(f"""{d["k"]}""")\n', encoding="utf-8")
    assert not pep701_only(fine)
