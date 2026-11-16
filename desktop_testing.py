import multiprocessing
import os
import sys
from pathlib import Path

import cv2
from streamlit_desktop_app import start_desktop_app


def getResourcePath(relativePath: str) -> Path:
    """Resolve a file during development or inside PyInstaller."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        basePath = Path(sys._MEIPASS)
    else:
        basePath = Path(__file__).resolve().parent

    return basePath / relativePath


def main() -> None:
    basePath = getResourcePath(".")

    # Allow relative st.Page("filename.py") paths to work.
    os.chdir(basePath)

    streamlitScript = getResourcePath("streamlit_control.py")

    if not streamlitScript.exists():
        raise FileNotFoundError(
            f"Streamlit entry file was not found: {streamlitScript}"
        )

    start_desktop_app(
        str(streamlitScript),
        title="Root Hair Analyzer",
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()