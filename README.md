# Land-Cover Map Intercomparison (LC Intercomparison)

## Project Overview

LC Intercomparison is a research-oriented Python framework developed as part of a PhD-level geospatial processing course. The project focuses on the pixel-wise intercomparison of land cover (LC) raster products and the quantitative evaluation of their agreement using standard accuracy assessment metrics widely adopted in remote sensing literature.

The framework is designed to support reproducible, transparent, and scientifically rigorous land-cover validation, with particular relevance to large-scale environmental and hydrological studies.

## Scientific Motivation
Land cover datasets derived from different sensors, classification algorithms, and temporal baselines often exhibit inconsistencies that can propagate uncertainty into downstream analyses (e.g., hydrological modeling, climate impact assessments, ecosystem monitoring).

This project aims to:
- Quantify agreement between two categorical land cover products
- Identify systematic classification differences
- Provide statistically sound accuracy metrics for comparative analysis

The methodology follows best practices recommended in the remote sensing accuracy assessment literature (e.g., confusion matrix–based evaluation).

## Code Overview
1. Imports and Dependencies

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

These libraries are used for:
-numpy: Array manipulation and numerical calculations
-pandas: Creating tables and exporting CSVs
-rasterio: Reading raster files (GeoTIFFs)
-matplotlib & seaborn: Visualization of confusion matrices
-os: Managing paths and directories

2. Loading Raster Data
   
def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)

- Reads a raster file and returns a 2D NumPy array of pixel values.
- Used for both the reference raster (ESA CCI) and comparison raster (GLC).

3. Computing the Confusion Matrix
   
def compute_confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    for t, p in zip(y_true, y_pred):
        cm[label_to_index[t], label_to_index[p]] += 1

    return pd.DataFrame(cm, index=labels, columns=labels)

- Flattens rasters into 1D arrays for pixel-wise comparison
- Determines unique land cover classes if labels not provided
- Counts the number of pixels classified into each category (rows = true, columns = predicted)
- Returns a pandas DataFrame representing the confusion matrix

4. Accuracy Metrics
   
a) Overall Accuracy (OA)

def overall_accuracy(confusion_matrix):
    cm = confusion_matrix.values
    return np.trace(cm) / cm.sum()

- Percentage of correctly classified pixels (sum of diagonal / total pixels)

b) Kappa Coefficient

def kappa_coefficient(confusion_matrix):
    cm = confusion_matrix.values
    total = cm.sum()
    po = np.trace(cm) / total
    row_marginals = cm.sum(axis=1)
    col_marginals = cm.sum(axis=0)
    pe = np.sum(row_marginals * col_marginals) / (total ** 2)
    return (po - pe) / (1 - pe)

- Measures agreement accounting for chance
- Values range from -1 to 1, with 1 indicating perfect agreement

c) Producer’s Accuracy (PA)

def producers_accuracy(confusion_matrix):
    cm = confusion_matrix.values
    correct = np.diag(cm)
    reference_total = cm.sum(axis=1)
    return correct / reference_total

- Probability that a reference pixel is correctly classified (omission error)

d) User’s Accuracy (UA)

def users_accuracy(confusion_matrix):
    cm = confusion_matrix.values
    correct = np.diag(cm)
    classified_total = cm.sum(axis=0)
    return correct / classified_total

- Probability that a pixel classified into a category actually belongs to that category (commission error)

5. Main Workflow

- Input Paths: Paths to ESA CCI and GLC rasters are defined.
- Output Directory: An output/ folder is created automatically.
- Load Rasters: load_raster() reads both datasets.
- Flatten Rasters: Arrays are flattened for pixel-wise comparison.
- Compute Confusion Matrix: Using compute_confusion_matrix().
- Compute Accuracy Metrics: OA, Kappa, PA, UA, mean PA, and mean UA.
- Save Outputs: Confusion matrix and accuracy summary saved as CSV.
- Visualize Confusion Matrix: Heatmap generated with seaborn and saved as PNG.

6. Output Files

| File                   | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| `cfm.csv`              | Confusion matrix (row = reference, column = predicted) |
| `accuracy_summary.csv` | Metrics: OA, Kappa, mean PA, mean UA                   |
| `cfm_htmap.png`        | Heatmap of confusion matrix                            |


## Repository Structure

```
lc_intercomparison/
|-- data/
|   |-- reference_lc.tif
|   `-- comparison_lc.tif
|
|-- lc_accuracy/
|   |-- confusion_matrix.py
|   |-- accuracy_metrics.py
|   |-- raster_io.py
|   |-- main.py
|   |-- test.py
|   `-- __init__.py
|
|-- output/
|   |-- accuracy_summary.csv
|   |-- cfm.csv
|   `-- cfm_htmap.png
|
|-- .gitignore
|-- LICENSE
|-- README.md
`-- github_installation.py
```

## Software Requirements
- Python ≥ 3.8
- Numpy
- Pandas
- Rasterio
- Matplotlib
- Seaborn

## Data Notes
- Rasters must have identical projection, resolution, and extent
- No-data values must be handled prior to analysis
- Classes should be harmonized across datasets

data/
├── reference_lc.tif  (Reference land cover raster)
└── comparison_lc.tif  (Comparison land cover raster)

Ensure that the input rasters:
- Are in GeoTIFF (`.tif`) format
- Have identical projection, spatial resolution, and spatial extent
- Use harmonized class definitions
- Have no-data values handled prior to analysis

The paths to these raster files must be updated manually in `lc_intercomparison_full.py` before running the analysis.

## Installation
Clone the repository:
git clone https://github.com/keerthana-150397/lc_intercomparison.git
cd lc_intercomparison

(Optional but recommended) Create a Conda environment:
conda create -n lc_intercomparison python=3.9 -y
conda activate lc_intercomparison

Install dependencies:
pip install numpy pandas rasterio matplotlib seaborn

## Usage
1. Edit the input paths in lc_intercomparison_full.py to point to your land cover raster datasets:
- Reference raster (e.g., ESA CCI)
- Comparison raster (e.g., GLC_FCS30)
2. Run the script:
python lc_accuracy/lc_intercomparison_full.py
3. All outputs will be saved automatically in the output/ directory.

## Outputs
The framework produces:
- Confusion matrix (CSV) – class-by-class comparison
- Accuracy summary (CSV) – OA, Kappa, mean PA, mean UA
- Confusion matrix heatmap (PNG) – visual representation of agreement
These outputs are suitable for inclusion in reports, course submissions, and scientific manuscripts.

## Data Assumptions & Notes
1. Input rasters must have identical:
  - Projection
  - Spatial resolution
  - Extent
2. No-data values should be handled prior to analysis
3. Class harmonization (if needed) should be performed before running the script

## Academic Context
This project was developed as part of a PhD-level course in Geospatial Processing, with direct relevance to ongoing doctoral research on land cover change in Mediterranean basins.

