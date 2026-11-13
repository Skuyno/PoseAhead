"""Compatibility entry point for local development.

The installed command is ``poseahead``. This file remains as a convenient
``python main.py`` entry point while the repository is being bootstrapped.
"""

from poseahead.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
