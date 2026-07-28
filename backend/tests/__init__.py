"""Backend test package.

This `__init__.py` looks empty but it is FUNCTIONAL and must not be deleted.

In pytest's default (`prepend`) import mode, a test file's package root is found
by walking the `__init__.py` chain upwards, and the PARENT directory of that root
is added to `sys.path`. Because `backend/tests/__init__.py` exists, the `backend/`
directory ends up on `sys.path` and the tests can say `from app.main import app`.

The alternative was manipulating `sys.path` by hand in `conftest.py`; this route
is less magical and relies on pytest's own rule.
"""
