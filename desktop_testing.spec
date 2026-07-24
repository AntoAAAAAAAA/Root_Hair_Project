# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, copy_metadata


# ---------------------------------------------------------
# Collect Streamlit package files
# Equivalent to:
#   --collect-all streamlit
# ---------------------------------------------------------
streamlitDatas, streamlitBinaries, streamlitHiddenImports = collect_all(
    "streamlit"
)


# ---------------------------------------------------------
# Collect OpenCV package files
# Equivalent to:
#   --collect-all cv2
# ---------------------------------------------------------
cv2Datas, cv2Binaries, cv2HiddenImports = collect_all(
    "cv2"
)

# ---------------------------------------------------------
# Collect Ultralytics package files
# Equivalent to:
#   --collect-all ultralytics
# ---------------------------------------------------------
ultralyticsDatas, ultralyticsBinaries, ultralyticsHiddenImports = collect_all(
    "ultralytics"
)

# ---------------------------------------------------------
# Data files copied into the packaged application
#
# Each tuple is:
#   ("source location", "destination inside app")
#
# This replaces your --add-data options.
# ---------------------------------------------------------
projectDatas = [
    ("streamlit_control.py", "."),
    ("streamlit_main.py", "."),
    ("streamlit_instr.py", "."),
    ("streamlit_diagnostics.py", "."),
    ("streamlit_main_testing.py", "."),
    ("src", "src"),
    (".streamlit", ".streamlit"),
    ("sam2_l.pt", "."),
    ("sam2_b.pt", "."),
]


# ---------------------------------------------------------
# Package metadata
# Equivalent to:
#   --copy-metadata streamlit
#   --copy-metadata streamlit-desktop-app
# ---------------------------------------------------------
metadataDatas = (
    copy_metadata("streamlit")
    + copy_metadata("streamlit-desktop-app")
    + copy_metadata("ultralytics")
)


# ---------------------------------------------------------
# Combine all collected data
# ---------------------------------------------------------
allDatas = (
    streamlitDatas
    + cv2Datas
    + ultralyticsDatas
    + metadataDatas
    + projectDatas
)


# ---------------------------------------------------------
# Combine package binaries
# ---------------------------------------------------------
allBinaries = (
    streamlitBinaries
    + cv2Binaries
    + ultralyticsBinaries
)


# ---------------------------------------------------------
# Combine hidden imports
#
# "cv2" is included explicitly to match:
#   --hidden-import cv2
# ---------------------------------------------------------
allHiddenImports = list(
    dict.fromkeys(
        streamlitHiddenImports
        + cv2HiddenImports
        + ultralyticsHiddenImports
        + [
            "cv2",
            "ultralytics",
            "ultralytics.models",
            "ultralytics.models.sam",
        ]
    )
)


# ---------------------------------------------------------
# Analyze the launcher and its dependencies
# ---------------------------------------------------------
a = Analysis(
    ["desktop_testing.py"],
    pathex=[],
    binaries=allBinaries,
    datas=allDatas,
    hiddenimports=allHiddenImports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)


# ---------------------------------------------------------
# Store pure Python modules in PyInstaller's PYZ archive
# ---------------------------------------------------------
pyz = PYZ(a.pure)


# ---------------------------------------------------------
# Create the executable
#
# console=False is the spec-file equivalent of --windowed.
# ---------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="desktop_testing",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


# ---------------------------------------------------------
# Build the one-folder application contents
# ---------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="desktop_testing",
)


# ---------------------------------------------------------
# Create the macOS .app bundle
# ---------------------------------------------------------
app = BUNDLE(
    coll,
    name="desktop_testing.app",
    icon=None,
    bundle_identifier="com.antoantony.roothairanalyzer",
)