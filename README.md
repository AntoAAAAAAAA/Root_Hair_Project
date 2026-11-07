
# Root Hair Analyzer

**Project Website:** [Root Hair Analyzer](https://roothairwebsite.streamlit.app/)

A Streamlit-based desktop application for semi-automated root hair detection, visualization, and length measurement from microscopy images.
This project was developed to reduce the manual work required when identifying and measuring root hairs in plant microscopy images. The current version (v2.0) uses a machine learning segmentation model trained with nnU-Net, followed by graph-based post-processing to identify and measure individual root hairs, including hairs that overlap.
The application is available for macOS and Windows. Although the current version represents a significant improvement over earlier iterations, the project is still under development, with further improvements to segmentation accuracy and root hair identification planned.



## Features

- Automated root hair detection and length measurement
- Side-by-side T0 and T1 image analysis
- Interactive visualization and individual root hair review
- Adjustable length filters and microscope calibration
- Editable measurement table for transferring results to Excel
- Persistent measurement storage across application sessions
- Graph-based reconstruction of overlapping root hairs



## Download and Installation

Root Hair Analyzer can be downloaded through the [project website](https://roothairwebsite.streamlit.app/). Follow the instructions provided on the website to download and install the appropriate version for your operating system.

Alternatively, installers and additional release information are available under [GitHub Releases](https://github.com/AntoAAAAAAAA/Root_Hair_Project/releases).

### Installation Instructions

**macOS:**
1. Download `RHA_Installer.dmg`.
2. Open the disk image and drag Root Hair Analyzer into the Applications folder.
3. Launch the application from the Applications folder.

**Windows:**
1. Download `RHA_Installer_Windows.exe`.
2. Run the installer and follow the on-screen instructions.
3. Launch Root Hair Analyzer from the Start Menu or desktop shortcut.

### Security Warnings

Root Hair Analyzer is not currently digitally signed, so Windows and macOS may display security warnings during installation.

**Windows:**
1. If Microsoft Defender SmartScreen displays a "Windows protected your PC" warning, click **More info**.
2. Select **Run anyway**, if available.
3. If prompted to allow the installer to make changes to your device, click **Yes**.

**macOS:**
1. Attempt to open Root Hair Analyzer from the Applications folder.
2. If macOS blocks the application because it cannot verify the developer, navigate to **System Settings → Privacy & Security**.
3. Scroll to the Security section and select **Open Anyway** for Root Hair Analyzer.
4. Confirm by clicking **Open** and entering your Mac password if requested.

**Important:** Only proceed after confirming that the installer was downloaded from the official project website or GitHub release and is legitimate. Do not bypass warnings that identify malware or report that the application is damaged.


## Repository Structure

```text
Root_Hair_Project/
├── .devcontainer/
├── .streamlit/
│
├── src/
│   ├── __init__.py
│   ├── ML.py
│   ├── final_visualize.py
│   ├── hybrid_ML.py
│   ├── main.py
│   ├── main_root_mask.py
│   ├── main_v2.py
│   ├── main_v3.py
│   └── root_hairs_mask.py
│
├── streamlit_main.py
├── streamlit_control.py
├── streamlit_instr.py
├── data.json
├── requirements.txt
├── .gitattributes
├── .gitignore
└── README.md
```

### Folder and File Descriptions

| File/Folder | Purpose |
|---|---|
| `src/` | Main source code for segmentation, image processing, graph-based root hair reconstruction, and visualization |
| `src/ML.py` | nnU-Net model loading and inference |
| `src/main_v3.py` | Current image analysis pipeline integrating segmentation and root hair reconstruction |
| `src/root_hairs_mask.py` | Root hair processing, skeleton graph construction, traversal, and filtering |
| `src/final_visualize.py` | Visualization of detected root hairs and their measurements |
| `src/main.py`, `src/main_v2.py` | Previous iterations of the image analysis pipeline |
| `streamlit_main.py` | Main Streamlit app interface |
| `streamlit_control.py` | Streamlit app navigation and page management |
| `streamlit_instr.py` | User instructions for operating the application |
| `data.json` | JSON-based storage for retaining root hair measurements across app sessions |
| `requirements.txt` | Python dependencies |


## Dependencies

Root Hair Analyzer uses several Python libraries for image processing, machine learning, graph analysis, and desktop application development.

**Core libraries:**
- [Streamlit](https://github.com/streamlit/streamlit)
- [OpenCV](https://github.com/opencv/opencv)
- [NumPy](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [scikit-image](https://github.com/scikit-image/scikit-image)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [Plotly](https://github.com/plotly/plotly)

**Machine learning and graph processing:**
- [PyTorch](https://github.com/pytorch/pytorch)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) — Semantic segmentation framework used to train the root hair identification model.
- [sknw](https://github.com/Image-Py/sknw) — Converts skeletonized images into graph representations.
- [NetworkX](https://github.com/networkx/networkx) — Used for graph processing, traversal, and maximum-weight matching of root hair branches.

**Desktop packaging:**
- [streamlit-desktop-app](https://github.com/ohtaman/streamlit-desktop-app)
- [PyInstaller](https://github.com/pyinstaller/pyinstaller)

See `requirements.txt` for all of the project's Python dependencies.


## Development Notes

Root Hair Analyzer began as a traditional computer vision project that used thresholding, connected component analysis, morphological operations, and skeletonization to identify root hairs. The first major version (v1.0) used a hybrid approach combining these techniques with SAM 2 segmentation to isolate the main root and identify individual root hairs. Although this approach worked on *some* microscopy images, inconsistent lighting, image noise, and overlapping root hairs created significant difficulties in obtaining reliable measurements.

Version 2.0 introduced a new machine learning pipeline using a custom-trained nnU-Net model developed from 250 manually labeled microscopy images. The model produces semantic segmentation masks identifying the background, main root, and root hairs. These masks are then skeletonized and converted into graph representations using `sknw`, which relies on NetworkX for graph operations. A custom post-processing algorithm uses branch direction calculations, maximum-weight matching, and graph traversal to reconstruct individual root hairs, including hairs that overlap or intersect.
This new approach has significantly improved the identification of overlapping root hairs compared with Version 1.0. However, the reliability of the final measurements still depends on segmentation quality, and further improvements remain necessary.


## Limitations

Although Version 2.0 improves on several of the issues present in Version 1.0, the analyzer may still struggle with:
- Dense clusters containing large numbers of overlapping root hairs
- Root hairs that blend into the background due to poor contrast or inconsistent lighting
- Segmentation errors resulting in missing, fragmented, or incorrectly connected root hairs
- Noisy backgrounds and other image artifacts
- Root hairs that blend into the main root

**Important:** The current pipeline requires microscopy images in which the main root runs ***vertically***. Images with a horizontally oriented main root are ***not supported*** and will not produce reliable results. This limitation applies to both Versions 1.0 and 2.0.

Because of the limitations listed above, our Root Hair Analyzer should be considered a **semi-automated supplementary measurement tool** rather than a replacement for manual verification.
