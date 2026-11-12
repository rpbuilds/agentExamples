"""Create the project's virtual environment and install requirements.txt into it.

Uses only the standard library, so it runs with any Python 3.10+ before anything else
is installed.

Run from this folder:
    python setup_env.py
"""

import subprocess
import sys
import venv
from pathlib import Path

PROJECT_DIR: Path = Path(__file__).resolve().parent
VENV_DIR: Path = PROJECT_DIR / ".venv"
REQUIREMENTS_FILE: Path = PROJECT_DIR / "requirements.txt"


def _venv_python(venv_dir: Path) -> Path:
    """Return the interpreter inside the venv, whose location differs by OS.

    Args:
        venv_dir: Root folder of the virtual environment.

    Returns:
        Path to the venv's python executable.
    """
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> None:
    """Create .venv if it is missing, then install the pinned requirements into it."""
    # Create the venv only once; re-running the script just refreshes packages
    if not VENV_DIR.exists():
        print(f"Creating virtual environment in {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print(f"Using existing virtual environment in {VENV_DIR}")

    python_path: Path = _venv_python(VENV_DIR)

    # Install into the venv's interpreter, never the global one
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True
    )
    subprocess.run(
        [str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], check=True
    )

    activate_hint: str = (
        r".venv\Scripts\Activate.ps1" if sys.platform == "win32" else "source .venv/bin/activate"
    )
    print(f"\nDone. Activate the environment with:\n    {activate_hint}")


if __name__ == "__main__":
    main()
