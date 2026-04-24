import os
import numpy as np
import pandas as pd
import seaborn as sns
from raster_io import load_raster
from confusion_matrix import compute_confusion_matrix

import matplotlib.pyplot as plt

from accuracy_metrics import (
    overall_accuracy,
    kappa_coefficient,
    producers_accuracy,
    users_accuracy
)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    # ----------------------------
    # Input paths
    # ----------------------------
    cci_path = r"C:\Users\user\projects\lc_intercomparison\data\Tevere_ESA_2000.tif"
    glc_path = r"C:\Users\user\projects\lc_intercomparison\data\Tevere_GLC_2000_1.tif"

    # ----------------------------
    # Output folder
    # ----------------------------
    output_dir = r"C:\Users\user\projects\lc_intercomparison\data\output"
    os.makedirs(output_dir, exist_ok=True)

    cm_csv = os.path.join(output_dir, "tevere_cfm_2000_1.csv")
    acc_csv = os.path.join(output_dir, "tevere_accuracy_summary_2000_1.csv")
    cm_png = os.path.join(output_dir, "tevere_cfm_heatmap_2000_1.png")

    # ----------------------------
    # Load rasters
    # ----------------------------
    print("Loading rasters...")
    cci = load_raster(cci_path)
    glc = load_raster(glc_path)

    # ----------------------------
    # Confusion matrix
    # ----------------------------
    print("Computing confusion matrix...")
    cm = compute_confusion_matrix(cci, glc)
    cm.to_csv(cm_csv)

    # ----------------------------
    # Accuracy metrics
    # ----------------------------
    oa = overall_accuracy(cm)
    kappa = kappa_coefficient(cm)
    pa = producers_accuracy(cm)
    ua = users_accuracy(cm)

    accuracy_summary = pd.DataFrame({
        "Metric": [
            "Overall Accuracy",
            "Kappa Coefficient",
            "Mean Producer's Accuracy",
            "Mean User's Accuracy"
        ],
        "Value": [
            oa,
            kappa,
            np.nanmean(pa),
            np.nanmean(ua)
        ]
    })

    accuracy_summary.to_csv(acc_csv, index=False)

    # ----------------------------
    # Plot confusion matrix
    # ----------------------------
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted (GLC)")
    plt.ylabel("Reference (ESA CCI)")
    plt.title("Confusion Matrix ESA CCI vs GLC")
    plt.savefig(cm_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ All outputs saved in:", output_dir)