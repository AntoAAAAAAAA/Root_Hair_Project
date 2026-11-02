# Root Hair Analyzer

Link to project: [Click Here](https://roothairanalyzer.streamlit.app/)

A Streamlit-based image analysis tool for semi-automated root hair detection, masking, visualization, and measurement from microscopy images.

This project was developed to reduce the amount of manual work required when measuring root hairs in plant microscopy images. The current version is primarily a **computer vision / visual processing pipeline** that uses thresholding, connected components, morphology, skeletonization, and segmentation-assisted processing to detect the main root, isolate candidate root hairs, and estimate root hair lengths.

This repository is still a clear work in progress. The current goal is to build a reliable semi-automated workflow first, while continuing to improve the pipeline and eventually expand toward a custom machine learning model.

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
streamlit run streamlit_control.py
```

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
- streamlit-desktop-app

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
