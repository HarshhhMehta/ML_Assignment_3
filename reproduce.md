# Reproduction Guide — Machine Learning Assignment 3

This document provides exact, deterministic commands to reproduce every experiment, quantitative metric, figure, and report table in this repository.

---

## 1. Environment Setup

The codebase has been designed and tested on standard Python 3.12 (CPU execution, 8 threads).

```bash
# Clone the repository
git clone https://github.com/HarshhhMehta/ML_Assignment_3.git
cd mlassignment3

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Setup

The code uses the official CIFAR-10 Python binary distribution. The dataset will be automatically downloaded into `data/raw/` on first execution if not already present.

* **Deterministic Seed:** Derived from roll number `M26CSA013` $\to$ **`26013`**.
* **Pre-Generated Split Indices:** The repository includes `data/split_indices/train_indices.npy` ($45{,}000$ train) and `val_indices.npy` ($5{,}000$ val), ensuring zero data leakage and 100% exact numerical reproducibility across all questions.
* **Q4 SVM Subset Indices:** `data/subset_indices/q4_svm_subset_indices.npy` ($5{,}000$ stratified samples, exactly $500$/class).

---

## 3. Running Individual Experiments

### Question 1: MLP Bottlenecks & Memorization Dynamics
```bash
# Trains Standard MLP (112,330 params) & Patch-LCN (113,690 params)
# Runs the 25-epoch 45% label noise memorization experiment
# Generates figures/q1a_learning_curves.* and figures/q1b_memorization_trajectory.*
python scripts/run_q1.py

# Runs the comprehensive multi-subset noise audit on the saved checkpoint
python scripts/audit_q1b.py
```
* **Key Outputs:**
  * Standard MLP Test Accuracy: `50.77%`
  * Patch-LCN Test Accuracy: `57.94%` ($+7.17\%$ architectural gain)
  * Peak Clean Validation Accuracy: `44.30%` at Epoch 5
  * Generalization Gap at Epoch 25: `+46.38%`

---

### Question 2: CNN vs. MLP Under Spatial Disruption
```bash
# Evaluates Clean, Patch-Permuted (8x8), and Pixel-Permuted conditions
# for FairCNN (111,906 params) vs. Standard MLP (112,330 params)
# Generates figures/q2_spatial_disturbance.* and figures/q2c_confidence_analysis.*
python scripts/run_q2.py
```
* **Key Outputs:**
  * Clean: FairCNN `72.05%` vs. Standard MLP `50.77%` ($+21.28\%$)
  * Patch-Permuted: FairCNN `65.56%` vs. Standard MLP `50.17%`
  * Pixel-Permuted: FairCNN `53.89%` vs. Standard MLP `50.55%` (MLP is completely permutation-invariant)
  * ECE Calibration: Clean CNN `0.1141`, Patch CNN `0.1371`, Pixel CNN `0.0612`

---

### Question 3: Autoencoders & Representation Learning
```bash
# Trains symmetrical convolutional autoencoders across bottlenecks (64, 256, 1024)
# Trains Denoising Autoencoder (sigma=0.2 Gaussian noise, zero labels)
# Evaluates downstream recovery on fixed FairCNN classifier from Q2
# Trains frozen linear probes on bottleneck representations
# Generates reconstruction grids, denoising comparisons, and trade-off curves
python scripts/run_q3.py
```
* **Key Outputs:**
  * Clean AE Selection: Bottleneck $d=1024$ selected strictly on validation MSE (`0.005855` val, `0.005888` test)
  * Downstream Recovery: Raw corrupted `20.07%` $\to$ Clean AE Recon `27.65%` ($+7.58\%$), DAE Recon `27.57%` ($+7.50\%$)
  * Linear Probes: $d=64 \to 36.66\%$, $d=256 \to 38.63\%$, $d=1024 \to 40.16\%$

---

### Question 4 (Bonus): Kernel SVMs, Margin Geometry & Clustering
```bash
# Performs the 15-configuration SVM sweep (Linear, RBF, Poly x 5 C values)
# Performs top-3 hard pair margin analysis using raw OVO boundary scores
# Evaluates train-only label-free clustering selection (K=6,8,10,12,14)
# Fits K-Means and GMM (K=10) and verifies exact 10,000-sample contingency matrix
python scripts/run_q4.py
```
* **Key Outputs:**
  * Best SVM: RBF with $C=0.1$ selected on validation (`72.88%` val, `71.64%` test, SV fraction `62.48%`)
  * Hard Pairs: Cat vs. Dog (391 errors), Auto vs. Truck (181 errors), Bird vs. Deer (171 errors)
  * Unsupervised Clustering ($K=10$): K-Means Test ARI `0.4537`, NMI `0.5206`; GMM Test ARI `0.4262`, NMI `0.5056`
  * Verified Contingency Table sum: Exactly `10,000` test samples

---

## 4. Re-Compiling the CVPR Report

To compile the LaTeX source into `report/main.pdf`:

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

To verify that the report strictly satisfies the $\le 10$ pages excluding references requirement:
```bash
python scripts/check_pdf_pages.py
```
Expected output:
* Total pages: **11**
* Pages 1–10: All body content (Abstract, Sections 1–7)
* Page 11: References only