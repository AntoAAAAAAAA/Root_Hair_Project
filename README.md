# Root Hair Analyzer

A Streamlit-based image analysis tool for semi-automated root hair detection, masking, visualization, and measurement from microscopy images.

This project was developed to reduce the amount of manual work required when measuring root hairs in plant microscopy images. The current version is primarily a **computer vision / visual processing pipeline** that uses thresholding, connected components, morphology, skeletonization, and segmentation-assisted processing to detect the main root, isolate candidate root hairs, and estimate root hair lengths.

This repository is still a clear work in progress. The current goal is to build a reliable semi-automated workflow first, while continuing to improve the pipeline and eventually expand toward a custom machine learning model.

---

## Project Status

This project is actively being developed and refined. The current version focuses on building a practical Streamlit application and a reliable visual processing workflow rather than a fully autonomous detector.

Difficult images may still require user review or correction, especially when:

- root hairs overlap heavily,
- the main root is angled or nearly horizontal,
- lighting/shadow artifacts interfere with thresholding,
- the main root boundary is not cleanly separated from root hairs,
- the segmentation model produces an incomplete mask.

This README will continue to be updated as the project develops.

---

## Current Version: Visual Processing Pipeline

The current version of the Root Hair Analyzer is based mainly on traditional image processing and computer vision methods.

The pipeline currently uses approaches such as:

- grayscale image processing,
- standard and adaptive thresholding,
- connected component analysis,
- contour detection and filling,
- morphological operations,
- distance transform-based root core extraction,
- SAM/SAM-2 assisted segmentation,
- root hair mask subtraction,
- skeletonization,
- pixel-based length estimation.

This approach is intentionally interpretable. Since the project is still being tested and refined, it is helpful to see each processing step and understand why a mask or measurement succeeds or fails.

---

## Machine Learning Model

**Work in progress.**

A future goal of this project is to develop a more complete machine learning-based model for root hair detection and measurement. Ideally, this would involve creating and training a custom model from scratch or fine-tuning a model specifically for root microscopy images.

Potential future machine learning directions include:

- building a labeled dataset of root and root hair images,
- training an instance segmentation or semantic segmentation model,
- comparing custom model performance against the current visual processing pipeline,
- improving robustness for difficult images with overlapping hairs or unusual root orientations,
- reducing the amount of manual correction needed from the user.

This section will be updated as machine learning development progresses.

---

## Main Features

- Upload and process root microscopy images
- Detect and mask the main root
- Separate candidate root hairs from the main root
- Skeletonize root hair structures for centerline-based measurement
- Filter candidate hairs using image-processing logic
- Estimate root hair lengths using a microscope conversion factor
- Visualize masks, skeletons, and processed results
- Support continued testing of thresholding, SAM-based segmentation, and hybrid pipelines

---

## Current Processing Pipeline

The general workflow is:

1. **Image loading**
   - Read the uploaded microscopy image.
   - Convert the image to grayscale when needed.

2. **Main root detection**
   - Apply thresholding to separate the root structure from the background.
   - Use connected component analysis to identify the largest root-associated region.
   - Refine the root mask using morphology, distance transforms, or SAM-assisted segmentation.

3. **Root hair isolation**
   - Use thresholding and mask subtraction to remove the main root.
   - Preserve thin root hair-like structures around the root boundary.

4. **Skeletonization**
   - Convert candidate hair masks into one-pixel-wide skeletons.
   - Use skeleton structure to estimate centerline length.

5. **Filtering and measurement**
   - Remove obvious noise and invalid candidate components.
   - Estimate length using pixel distance and a microscope conversion factor.

6. **Visualization**
   - Display intermediate and final masks.
   - Show detected structures and measurements in the Streamlit interface.

---

## Repository Structure

```text
ROOT_HAIR/
├── archive/
│   ├── archived.py
│   ├── old_code.py
│   └── test.ipynb
│
├── src/
│   ├── __init__.py
│   ├── final_visualize.py
│   ├── hybrid_ML.py
│   ├── main.py
│   ├── main_root_mask.py
│   ├── main_v2.py
│   └── root_hairs_mask.py
│
├── streamlit_main.py
├── streamlit_control.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── .gitattributes
└── README.md
```

### Folder and file descriptions

| File/Folder | Purpose |
|---|---|
| `src/` | Main source code for image processing, masking, segmentation, and visualization |
| `archive/` | Older experiments, backup code, and testing notebooks |
| `streamlit_main.py` | Main Streamlit application entry point |
| `streamlit_control.py` | Streamlit testing/control script or alternate app workflow |
| `requirements.txt` | Python dependencies for local use and deployment |
| `packages.txt` | System-level packages for Streamlit Cloud deployment |
| `runtime.txt` | Python runtime version for deployment |
| `.gitattributes` | Git/Git LFS configuration |
| `README.md` | Project overview and setup instructions |

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd ROOT_HAIR
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App

Run the main Streamlit app:

```bash
streamlit run streamlit_main.py
```

If you are currently testing a different entry-point file, use:

```bash
streamlit run streamlit_control.py
```

---

## Model Weights and Git LFS

This project uses a SAM/SAM-2 model weights file, currently:

```text
sam2_l.pt
```

Because model weight files can be large, Git LFS has been configured for this repository and is being used to track the `.pt` model file. After cloning the repository, make sure Git LFS is installed and pull the LFS files if needed:

```bash
git lfs install
git lfs pull
```

If the model file is missing after cloning, confirm that Git LFS is installed correctly and that the `.pt` file has been downloaded.

---

## Dependencies

The project uses Python image-processing and app-development libraries such as:

- Streamlit
- OpenCV
- NumPy
- SciPy
- scikit-image
- Matplotlib
- Plotly
- Ultralytics / SAM

See `requirements.txt` for the full dependency list.

---

## Development Notes

This project has gone through multiple experimental approaches, including:

- standard thresholding,
- adaptive thresholding,
- connected component filtering,
- contour filling,
- distance transform-based root core extraction,
- morphological opening and closing,
- SAM/SAM-2 segmentation with bounding box prompts,
- skeleton-based root hair length estimation.

The current goal is not only to produce a final measurement, but also to keep the pipeline interpretable and adjustable for difficult microscopy images.

---

## Suggested `.gitignore`

A basic `.gitignore` for this project:

```gitignore
# Python
__pycache__/
*.pyc

# VS Code
.vscode/

# macOS
.DS_Store

# Jupyter
.ipynb_checkpoints/
```

These files and folders are generated automatically and do not need to be tracked in Git.

---

## Limitations

The analyzer may struggle with:

- dense clusters of overlapping hairs,
- poor image contrast,
- root hairs that blend into the main root,
- unusual root orientations,
- segmentation artifacts,
- noisy backgrounds,
- inconsistent lighting across images.

Because of these limitations, this project is best understood as a **semi-automated analysis tool** rather than a fully automated biological measurement system.

---

## Future Improvements

Potential future improvements include:

- adding manual correction tools in the Streamlit interface,
- improving root boundary detection,
- testing RGB/LAB color channels for difficult lighting cases,
- comparing standard and adaptive thresholding when root detection is weak,
- improving sideways-root robustness,
- adding clearer per-hair measurement overlays,
- saving results to CSV,
- supporting batch image processing,
- building a custom machine learning model for root hair detection,
- creating a polished user guide for students or lab members.

---

## Purpose

The long-term goal of this project is to make root hair measurement faster, more consistent, and more accessible for students and researchers working with microscopy images.

Instead of manually measuring every hair, users should be able to upload an image, inspect the automatically generated masks, make corrections when needed, and export useful measurements for downstream analysis.
