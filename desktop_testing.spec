
# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, copy_metadata


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

    ("sam2_l.pt", "."),

    # Model weights, configuration and custom trainer source
    ("Compressed_Unet", "Compressed_Unet"),
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

    pathex=[".", "Compressed_Unet"],

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
# Executable
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
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


# ---------------------------------------------------------
# One-folder distribution
# ---------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RootHairAnalyzer",
)


# ---------------------------------------------------------
# macOS app bundle
# ---------------------------------------------------------
app = BUNDLE(
    coll,
    name="RootHairAnalyzer.app",
    icon=None,
    bundle_identifier="com.antoantony.roothairanalyzer",
)