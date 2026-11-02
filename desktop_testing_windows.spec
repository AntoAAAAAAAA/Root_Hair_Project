
# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, copy_metadata


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
projectRoot = Path(SPEC).resolve().parent

# Locate custom trainer modules if stored in either folder.
trainerPaths = [
    projectRoot,
    projectRoot / "Compressed_Unet2",
    projectRoot / "Compressed_Unet",
]

trainerPaths = [
    str(path) for path in trainerPaths if path.is_dir()
]


# ---------------------------------------------------------
# Collect Streamlit
# ---------------------------------------------------------
streamlitDatas, streamlitBinaries, streamlitHiddenImports = collect_all(
    "streamlit"
)


# ---------------------------------------------------------
# Collect OpenCV
# ---------------------------------------------------------
cv2Datas, cv2Binaries, cv2HiddenImports = collect_all(
    "cv2"
)


# ---------------------------------------------------------
# Collect Ultralytics / SAM
# ---------------------------------------------------------
ultralyticsDatas, ultralyticsBinaries, ultralyticsHiddenImports = collect_all(
    "ultralytics"
)


# ---------------------------------------------------------
# Collect nnU-Net v2
# ---------------------------------------------------------
nnunetDatas, nnunetBinaries, nnunetHiddenImports = collect_all(
    "nnunetv2"
)


# ---------------------------------------------------------
# Application data and model weights
# ---------------------------------------------------------
projectDatas = [
    ("streamlit_control.py", "."),
    ("streamlit_main.py", "."),
    ("streamlit_instr.py", "."),
    ("streamlit_diagnostics.py", "."),
    ("streamlit_main_testing.py", "."),

    ("src", "src"),
    (".streamlit", ".streamlit"),

    # SAM model weights
    ("sam2_l.pt", "."),

    # nnU-Net model weights and configuration
    ("Compressed_Unet2", "Compressed_Unet2"),
]


# ---------------------------------------------------------
# Package metadata
# ---------------------------------------------------------
metadataDatas = (
    copy_metadata("streamlit")
    + copy_metadata("streamlit-desktop-app")
    + copy_metadata("ultralytics")
    + copy_metadata("nnunetv2")
)


# ---------------------------------------------------------
# Combine data
# ---------------------------------------------------------
allDatas = (
    streamlitDatas
    + cv2Datas
    + ultralyticsDatas
    + nnunetDatas
    + metadataDatas
    + projectDatas
)


# ---------------------------------------------------------
# Combine binaries
# ---------------------------------------------------------
allBinaries = (
    streamlitBinaries
    + cv2Binaries
    + ultralyticsBinaries
    + nnunetBinaries
)


# ---------------------------------------------------------
# Combine hidden imports
# ---------------------------------------------------------
allHiddenImports = list(
    dict.fromkeys(
        streamlitHiddenImports
        + cv2HiddenImports
        + ultralyticsHiddenImports
        + nnunetHiddenImports
        + [
            "cv2",
            "ultralytics",
            "ultralytics.models",
            "ultralytics.models.sam",

            "nnunetv2",
            "nnunetv2.inference.predict_from_raw_data",

            "custom_trainers.nnUNetTrainer_RootHair",

            "platformdirs",
        ]
    )
)


# ---------------------------------------------------------
# Analyze launcher
# ---------------------------------------------------------
a = Analysis(
    ["desktop_testing.py"],

    pathex=trainerPaths,

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
# PYZ archive
# ---------------------------------------------------------
pyz = PYZ(a.pure)


# ---------------------------------------------------------
# Windows executable
# ---------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],

    exclude_binaries=True,

    name="RootHairAnalyzer",

    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,

    console=False,

    disable_windowed_traceback=False,
    argv_emulation=False,
)


# ---------------------------------------------------------
# One-folder distribution
# ---------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,

    strip=False,
    upx=False,
    upx_exclude=[],

    name="RootHairAnalyzer",
)
