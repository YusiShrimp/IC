"""pytest configuration for the tests in this folder.

    pytest tests/       tests automata.py (the base version)
    pytest -U tests/    tests automata_updated.py (the AI-updated version)
"""
import importlib
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_addoption(parser):
    parser.addoption("-U", "--updated", action="store_true", default=False,
                     help="test automata_updated.py instead of automata.py")


def module_name(config):
    """The file under test: automata, or automata_updated with -U."""
    return "automata_updated" if config.getoption("--updated") else "automata"


def pytest_configure(config):
    name = module_name(config)
    if not os.path.exists(os.path.join(ROOT, name + ".py")):
        raise pytest.UsageError(f"{name}.py not found in {ROOT}")
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise pytest.UsageError(
            f"importing {name}.py failed: {type(exc).__name__}: {exc}")
    # The tests `import automata`: make that the file under test.
    sys.modules["automata"] = module


def pytest_report_header(config):
    return f"testing: {module_name(config)}.py"
