"""Environment check — written from scratch, not copied.

Prints PASS/FAIL for the tools + libs this course actually needs on THIS machine,
and reports the real torch device string (mps on Apple Silicon, else cpu).

Exit 0 iff every core check passes.
"""

# your code below
from __future__ import annotations

import shutil
import subprocess
import sys
from functools import partial
from typing import Callable

CheckResult = tuple[bool, str]


def check_python() -> CheckResult:
    """Check that Python 3.12+ is installed."""
    return (
        sys.version_info >= (3, 12),
        sys.version.split("\n")[0],
    )


def check_package(name: str) -> CheckResult:
    """Check whether a Python package is installed."""
    try:
        module = __import__(name)
        return True, getattr(module, "__version__", "installed")
    except ImportError:
        return False, "not installed"


def check_command(command: str) -> CheckResult:
    """Check whether a command-line executable exists."""
    if shutil.which(command) is None:
        return False, "not installed"

    result = subprocess.run(
        [command, "--version"],
        capture_output=True,
        text=True,
    )

    output = (result.stdout or result.stderr).splitlines()[0]
    return True, output


def check_device() -> CheckResult:
    """Return the PyTorch execution device."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return True, "mps"

        if torch.cuda.is_available():
            return True, "cuda"

        return True, "cpu"

    except ImportError:
        return False, "PyTorch not installed"


def print_result(name: str, result: CheckResult) -> bool:
    """Print a formatted check result and return its status."""
    ok, details = result
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({details})")
    return ok


def main() -> int:
    """Run environment checks.

    Returns:
        0 if every required dependency is present, otherwise 1.
    """
    required: list[tuple[str, Callable[[], CheckResult]]] = [
        ("Python 3.12+", check_python),
        ("NumPy", partial(check_package, "numpy")),
        ("Matplotlib", partial(check_package, "matplotlib")),
        ("Git", partial(check_command, "git")),
        ("Node.js", partial(check_command, "node")),
        ("Rust (cargo)", partial(check_command, "cargo")),
    ]

    passed = True

    for name, check in required:
        passed &= print_result(name, check())

    print("\nPyTorch:")
    print_result("PyTorch", check_package("torch"))
    print_result("Device", check_device())

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())