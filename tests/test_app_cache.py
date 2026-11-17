"""Every cached function in app.py must be keyed on a version argument.

Streamlit's caches outlive the code that filled them: a process still running
after a deploy keeps the values it already holds, so new page code can be
handed an object built by the previous version. That is not hypothetical here
- the app crashed with an AttributeError when the cached model module predated
a function the page had started calling.

The guard is a `version` argument on every cached function, built from the
files its result depends on. Two properties of Streamlit's key make the exact
spelling matter, and both are asserted by behaviour below rather than trusted:

  - the key is built from the arguments actually PASSED, so a default value is
    invisible and the version has to be handed over by the caller;
  - an argument whose name starts with "_" is deliberately not hashed, so
    `_version` would be silently ignored.

app.py is a Streamlit script - importing it calls st.set_page_config and draws
a page - so this reads it as source rather than importing it.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app.py"

CACHE_DECORATORS = {"cache_data", "cache_resource"}


@pytest.fixture(scope="module")
def tree():
    return ast.parse(APP_PATH.read_text(encoding="utf-8"), str(APP_PATH))


def _decorator_names(node):
    """Names like "st.cache_data" for every decorator on a function."""
    out = []
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute):
            out.append(target.attr)
        elif isinstance(target, ast.Name):
            out.append(target.id)
    return out


def _cached_functions(tree):
    return [n for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and CACHE_DECORATORS & set(_decorator_names(n))]


def _arg_names(node):
    a = node.args
    return [arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


def test_the_scan_finds_the_cached_functions(tree):
    """If this fails the rest proves nothing: the scan found no caches."""
    names = {n.name for n in _cached_functions(tree)}
    assert len(names) >= 8, names
    assert {"_engine", "_panel", "_surface", "_live_snapshot"} <= names


@pytest.mark.parametrize("attr", sorted(CACHE_DECORATORS))
def test_cache_decorators_are_only_used_on_versioned_functions(tree, attr):
    for node in _cached_functions(tree):
        if attr in _decorator_names(node):
            assert "version" in _arg_names(node), (
                f"{node.name} is @st.{attr} but takes no version argument, so "
                f"it will keep serving values built by superseded code")


def test_no_cached_function_hides_the_version_behind_an_underscore(tree):
    """Streamlit skips arguments named with a leading underscore."""
    for node in _cached_functions(tree):
        assert "_version" not in _arg_names(node), node.name


def test_no_cached_function_defaults_its_version(tree):
    """A default is never part of the key, so it would never invalidate."""
    for node in _cached_functions(tree):
        names = _arg_names(node)
        if "version" not in names:
            continue
        a = node.args
        positional = [arg.arg for arg in (*a.posonlyargs, *a.args)]
        if "version" in positional:
            # Defaults bind to the tail of the positional list.
            defaulted = positional[len(positional) - len(a.defaults):]
            assert "version" not in defaulted, node.name
        else:
            i = [arg.arg for arg in a.kwonlyargs].index("version")
            assert a.kw_defaults[i] is None, node.name


def test_every_cached_function_is_reached_through_a_wrapper(tree):
    """The private cached function is called in exactly one place, by the
    public wrapper of the same name, which passes a version constant.

    Anything else calling it directly would have to repeat the version, and
    sooner or later one caller would not.
    """
    cached = {n.name for n in _cached_functions(tree)}
    wrappers = {n.name: n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name not in cached}

    for name in sorted(cached):
        public = name.lstrip("_")
        assert public in wrappers, f"{name} has no {public}() wrapper"

        calls = [c for c in ast.walk(wrappers[public])
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                 and c.func.id == name]
        assert len(calls) == 1, f"{public}() should call {name} exactly once"

        passed = [a.id for a in calls[0].args if isinstance(a, ast.Name)]
        assert any("VERSION" in p for p in passed), (
            f"{public}() calls {name} without passing a version constant")

    # And nothing outside its own wrapper calls one of them.
    for other in tree.body:
        if not isinstance(other, ast.FunctionDef) or other.name in cached:
            continue
        for call in ast.walk(other):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id in cached):
                assert other.name == call.func.id.lstrip("_"), (
                    f"{other.name}() calls {call.func.id} directly instead of "
                    f"its wrapper")


def test_the_version_covers_the_model_source(tree):
    """The crash was a stale model module, so that file must be in the key."""
    src = APP_PATH.read_text(encoding="utf-8")
    assert 'MODEL_SOURCE = ROOT / "model" / "interpolate.py"' in src
    assert "MODEL_INPUTS = (MODEL_SOURCE," in src
    assert "VERSION = fingerprint(*MODEL_INPUTS)" in src
