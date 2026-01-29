<<<<<<< HEAD
# Photometric Stereo Toolkit

A Python implementation of a Photometric Stereo pipeline for reconstructing surface normals, albedo (diffuse reflectance), and depth (height maps) from multiple images captured under varying lighting directions.

## Features

- Recover per-pixel surface normals and albedo from multiple images
- Integrate normals into a smooth height map using a DCT-based Poisson solver
- Support for high-resolution inputs and color restoration
- Simple GUI for virtual relighting and interactive inspection

## Repository Structure

``text
Photometric Stereo/
├── src/
│   ├── utils.py               # Data parsing and image loading helpers
│   ├── step1_calibration.py   # Light source calibration using a chrome ball
│   ├── step2_photometric.py   # Photometric stereo: solve normals & albedo
│   └── step3_integration.py   # DCT-based integration to recover height map
├── psmImages/                 # Input datasets (buddha, cat, chrome, ...)
├── output/                    # Generated normals, albedo maps, height maps
├── gui_app.py                 # Simple GUI for relighting and visualization
└── README.md                  # This file
```

## Quick Start

1. Create a Python virtual environment (recommended) and activate it.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install the main libraries:

```bash
pip install numpy opencv-python matplotlib scipy Pillow
```

3. Run the GUI (for interactive relighting and inspection):

```bash
python gui_app.py
```

4. Typical workflow (command-line or via GUI):

- Step 1: Calibrate light directions using the chrome ball dataset (e.g., `chrome.txt`).
- Step 2: Run photometric stereo to compute normals and albedo (use calibrated light directions).
- Step 3: Integrate normals into a height map using the DCT Poisson solver.

## Implementation Notes

- Calibration (src/step1_calibration.py): locate the chrome ball, detect highlights, compute per-image light directions.
- Photometric stereo (src/step2_photometric.py): solve a linear system per pixel (least squares) to obtain the product of albedo and normal; separate magnitude (albedo) and direction (normal).
- Integration (src/step3_integration.py): use a frequency-domain solver (DCT) for robust and fast height recovery on large images.

## Data

Place image lists (e.g., `buddha.txt`, `cat.txt`, `chrome.txt`) and their corresponding images under `psmImages/`.

Outputs (normals, albedo, height maps) are written to `output/` by the scripts.

## Notes & Tips

- For best results, calibrate light directions using the chrome ball set before running photometric stereo on target objects.
- If you encounter shadows or specularities, use robust masking or channel-wise color processing to reduce artifacts.

## License

This repository is provided as-is for research and educational use. Add a license file if you plan to publish or redistribute.

---
