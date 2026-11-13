"""Compatibility launcher; live orchestration belongs to the installed package."""

from poseahead.runtime.webcam import main

if __name__ == "__main__":
    raise SystemExit(main())
