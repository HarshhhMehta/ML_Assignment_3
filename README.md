# An Empirical and Theoretical Dissection of Visual Inductive Biases
### Machine Learning Assignment 3 — CIFAR-10 Benchmark Investigation
**Candidate:** Harsh Mehta &nbsp;|&nbsp; **Roll Number:** `M26CSA013` &nbsp;|&nbsp; **Deterministic Seed:** `26013`  
**Repository:** [https://github.com/HarshhhMehta/ML_Assignment_3](https://github.com/HarshhhMehta/ML_Assignment_3)  
**Final Report:** [report/main.pdf](report/main.pdf) (CVPR 2-Column Format, 10 Pages Content + 1 Page References)

---

## 1. Project Overview

This repository contains the complete, mathematically rigorous, and fully reproducible implementation for **Machine Learning Assignment 3** (75 Compulsory Marks + 25 Bonus Marks). The investigation dissects how architectural inductive biases, parameter constraints, label noise, bottleneck dimensionality, and supervision objectives dictate visual representation learning on the **CIFAR-10** benchmark.

### Experimental Integrity & Protocol Safeguards
* **Deterministic Random Seed:** Derived systematically from candidate roll number `M26CSA013` $\to$ **`26013`**.
* **Strict Stratified Splitting:** Fixed partition of the $50{,}000$ training images into $45{,}000$ training and $5{,}000$ validation samples, preserved permanently in `data/split_indices/` with zero class skew.
* **Strict Test Set Quarantine:** The official $10{,}000$-sample test set is evaluated strictly once per finalized model post-validation selection. Zero hyperparameter tuning or early stopping on test data.
* **No External Resources:** Zero pretrained backbones, external datasets, foundation models, or auto-architecture search.

---

## 2. Core Experimental Questions & Findings

### Q1: MLP Structural Bottlenecks & Memorization Dynamics (25 Marks)
* **Q1(a) Locality vs. Weight Sharing ($\sim 112\text{k}$ Parameters):**
  * *Standard MLP* ($3{,}072 \to 36 \to 36 \to 10$, $112{,}330$ params): **$50.77\%$** test accuracy.
  * *Patch-LCN* ($16$ independent $8 \times 8$ local projections, **zero weight sharing**, $113{,}690$ params): **$57.94\%$** test accuracy.
  * *Finding:* Imposing local receptive fields alone produces an immediate **$+7.17\%$ test gain**, proving that the absence of localized connectivity—rather than weight sharing—is the primary bottleneck of standard MLPs.
* **Q1(b) Memorization Under $45\%$ Effective Label Noise ($1.84\text{M}$ Parameters):**
  * Early SGD updates extract low-frequency semantic patterns, reaching peak clean validation accuracy (**$44.30\%$**) at **Epoch 5** before brute-force memorization degrades validation performance to **$32.22\%$** at Epoch 25.
  * *Multi-Subset Audit:* Clean training subset acc: **$78.60\%$**; Corrupted training subset (vs. noisy targets): **$59.47\%$**; In-distribution generalization gap: **$+46.38\%$**.

### Q2: CNN vs. MLP Under Controlled Spatial Disruption (25 Marks)
* **Parameter Parity Baseline:** FairCNN ($111{,}906$ params) constructed to match Standard MLP ($112{,}330$ params) within $0.38\%$.
* **Factorial Perturbation Matrix ($N=10{,}000$):**
  * *Clean CIFAR-10:* FairCNN **$72.05\%$** vs. Standard MLP **$50.77\%$** ($+21.28\%$ CNN gain).
  * *Patch-Permuted ($8 \times 8$):* FairCNN **$65.56\%$** (retains intra-patch edges) vs. Standard MLP **$50.17\%$**.
  * *Pixel-Permuted:* FairCNN collapses to **$53.89\%$** ($-18.16\%$ drop); Standard MLP is **strictly invariant** at **$50.55\%$** ($\Delta = -0.22\%$).
* *Calibration:* Clean CNN ECE is $0.1141$, Patch CNN is $0.1371$, and Pixel CNN is $0.0612$.

### Q3: Autoencoders, Denoising & Representation Probing (25 Marks)
* **Q3(a) Bottleneck Capacity Sweep ($d \in \{64, 256, 1024\}$):**
  * Validation MSE: $d=64 \to 0.008134$, $d=256 \to 0.006373$, $d=1024 \to \mathbf{0.005855}$.
  * Bottleneck **$d=1024$** selected strictly on validation error; achieves **$0.005888$** test MSE.
* **Q3(b) Denoising Autoencoder ($\sigma=0.2$ Noise) on Fixed Downstream Classifier:**
  * Fixed FairCNN clean baseline: **$72.05\%$**; Raw corrupted test images: **$20.07\%$** ($-51.98\%$ collapse).
  * Clean AE ($b=1024$) reconstructions: **$27.65\%$** ($+7.58\%$ recovery).
  * Denoising AE ($b=256$) reconstructions: **$27.57\%$** ($+7.50\%$ recovery).
* **Q3(c) Frozen Linear Probing vs. Reconstruction:**
  * Frozen linear probe test accuracy scales monotonically with capacity: $d=64 \to \mathbf{36.66\%}$, $d=256 \to \mathbf{38.63\%}$, $d=1024 \to \mathbf{40.16\%}$.
  * *Insight:* $L_2$ pixel reconstruction allocates capacity to dominant background textures rather than semantic class boundaries.

### Q4: Kernel SVMs, Margin Geometry & Clustering (Bonus: 25 Marks)
* **Q4(a) 15-Configuration SVM Grid Sweep (Frozen 64-D FairCNN Features, $5{,}000$ Train Subset):**
  * **Selected Model:** **RBF Kernel with $C=0.1$** (Val Acc: **$72.88\%$**, SV Fraction: $\rho_{\text{SV}} = 62.48\%$).
  * **Official Test Accuracy:** **$71.64\%$** (matches end-to-end FairCNN at $72.05\%$ within $0.41\%$).
* **Q4(b) Hard Pair Decision Boundary Geometry:**
  * Top-3 confused pairs: Cat vs. Dog ($391$ errors, $>76\%$ SVs), Auto vs. Truck ($181$ errors), Bird vs. Deer ($171$ errors).
  * Analyzed via raw One-vs-One boundary-proximity scores ($|f_{i,j}| \le 0.0037 \ll 1.0$), exposing severe manifold tangency between quadruped pets and background scenery bias.
* **Q4(c) Unsupervised Clustering vs. Supervised SVM:**
  * *Train-Only Label-Free Selection:* $K=10$ selected via inertia elbow ($113{,}816.3$).
  * *Metrics ($N=10{,}000$):* K-Means Test ARI **$0.4537$**, NMI **$0.5206$**; GMM Test ARI **$0.4262$**, NMI **$0.5056$**.
  * *Reconciled Contingency Table:* Verified across all $10{,}000$ test samples ($\sum_{i,j} C_{i,j} = 10{,}000$).
  * *Finding:* Clustering groups points by visual ambient scenery (birds and planes merged into sky clusters), whereas supervised SVM isolates discriminative object features.

---

## 3. Summary of Key Experimental Results

| Question | Model / Condition | Parameters | Val Metric | Test Metric | Key Finding / Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Q1(a)** | Standard MLP (Global) | $112{,}330$ | $51.12\%$ | $50.77\%$ Acc | Global receptive field squanders statistical efficiency |
| **Q1(a)** | Patch-LCN (Local $8\times 8$, No Sharing) | $113{,}690$ | $58.68\%$ | **$57.94\%$ Acc** | **$+7.17\%$ gain** isolated to local receptive fields |
| **Q1(b)** | MLP under 45% Label Noise | $1{,}841{,}162$ | $44.30\%$ (Ep 5) | $32.22\%$ Acc | Early spectral bias (Ep 1–5) followed by late memorization |
| **Q2(a)** | FairCNN Clean Baseline | $111{,}906$ | $72.54\%$ | **$72.05\%$ Acc** | **$+21.28\%$ gain** over parameter-matched MLP |
| **Q2(b)** | Standard MLP (Pixel-Permuted) | $112{,}330$ | $50.84\%$ | $50.55\%$ Acc | **Strictly invariant** ($\Delta = -0.22\%$) |
| **Q2(b)** | FairCNN (Pixel-Permuted) | $111{,}906$ | $54.26\%$ | $53.89\%$ Acc | Severe **$-18.16\%$ collapse** from lost spatial continuity |
| **Q2(c)** | FairCNN (Patch-Permuted $8\times 8$) | $111{,}906$ | $66.12\%$ | $65.56\%$ Acc | Retains $65.56\%$ due to surviving local patch features |
| **Q3(a)** | Conv Autoencoder ($d=64$) | $191{,}843$ | $0.008134$ | $0.008159$ MSE | High compression blurs edges and high-frequency textures |
| **Q3(a)** | Conv Autoencoder ($d=256$) | $290{,}339$ | $0.006373$ | $0.006409$ MSE | Moderate fidelity reconstruction |
| **Q3(a)** | Conv Autoencoder ($d=1024$) | $684{,}323$ | **$0.005855$** | **$0.005888$ MSE** | **Selected model** via strictly lower validation MSE |
| **Q3(b)** | Downstream on DAE Reconstructions | Fixed CNN | $0.007734$ | $27.57\%$ Acc | **$+7.50\%$ recovery** over raw corrupted inputs ($20.07\%$) |
| **Q3(c)** | Frozen Linear Probe ($d=1024$) | $10{,}250$ | $40.54\%$ | $40.16\%$ Acc | Monotonic scaling with capacity; lags behind supervised features |
| **Q4(a)** | Best RBF SVM ($C=0.1$) on 64-D CNN | $3{,}124$ SVs | **$72.88\%$** | **$71.64\%$ Acc** | Matches full end-to-end FairCNN ($72.05\%$) within $0.41\%$ |
| **Q4(c)** | Unsupervised K-Means ($K=10$) | — | Inertia $113{,}816$ | **ARI: $0.4537$, NMI: $0.5206$** | Discovers environmental scene manifolds (sky/foliage) |
| **Q4(c)** | Unsupervised GMM ($K=10$) | — | BIC $138{,}308$ | **ARI: $0.4262$, NMI: $0.5056$** | Diagonal covariance Gaussian density clustering |

---

## 4. Repository Structure

```
mlassignment3/
├── README.md                      # Comprehensive project overview and evaluation summary
├── reproduce.md                   # Step-by-step reproduction guide and CLI commands
├── requirements.txt               # Locked Python dependencies
├── .gitignore                     # Git ignore rules (excludes raw dataset & caches)
├── ml assignment 3.pdf            # Original assignment specification document
├── report/
│   ├── main.pdf                   # Final compiled CVPR report (10 pages body + 1 page refs)
│   ├── main.tex                   # LaTeX master document
│   ├── cvpr.sty                   # Official CVPR 2-column LaTeX style package
│   ├── references.bib             # Formatted bibliography database
│   ├── figures/                   # Vector and raster figures embedded in LaTeX
│   └── sections/                  # Modular LaTeX sections for Q1-Q4, Intro, and Conclusion
├── configs/
│   ├── q1_config.json             # Model architectures and noise settings for Q1
│   ├── q2_config.json             # FairCNN and spatial perturbation settings for Q2
│   ├── q3_config.json             # Autoencoder capacities and DAE settings for Q3
│   └── q4_config.json             # SVM grid and clustering hyperparameter settings for Q4
├── data/
│   ├── split_indices/             # Deterministic 45k train / 5k val indices (seed 26013)
│   └── subset_indices/            # Deterministic 5k stratified SVM subset indices for Q4
├── checkpoints/                   # Saved PyTorch model checkpoints (~15 MB total)
├── experiments/                   # Raw training histories, epoch logs, and registry.json
├── results/                       # Verified quantitative summaries, sweep tables, and contingency matrix
├── figures/                       # Publication-ready plots (.pdf and .png)
├── scripts/
│   ├── run_q1.py                  # End-to-end execution for Q1 (MLP vs LCN, Memorization)
│   ├── audit_q1b.py               # Checkpoint audit for Q1(b) multi-subset accuracy
│   ├── run_q2.py                  # End-to-end execution for Q2 (FairCNN, Permutations, ECE)
│   ├── run_q3.py                  # End-to-end execution for Q3 (Autoencoders, DAE, Probes)
│   ├── run_q4.py                  # End-to-end execution for Q4 (SVM sweep, Hard Pairs, Clustering)
│   ├── targeted_q4_correction.py  # Precision OVO boundary score & contingency verification
│   └── check_pdf_pages.py         # PyMuPDF page budget validator
└── src/
    ├── data/                      # Dataset loaders, CIFAR-10 unpickler, and spatial permutations
    ├── models/                    # PyTorch definitions: Standard MLP, Patch-LCN, FairCNN, Autoencoder
    ├── training/                  # Training loops, cosine learning rate schedules, noise injection
    ├── evaluation/                # Calibration (ECE), linear probing, OVO SVM evaluation, clustering metrics
    └── utils/                     # Matplotlib plotting routines and experiment registry logger
```

---

## 5. Quick Start & Reproduction

### Installation
```bash
# Clone the repository
git clone https://github.com/HarshhhMehta/ML_Assignment_3.git
cd mlassignment3

# Install dependencies
pip install -r requirements.txt
```

### Reproducing Experiments
```bash
# Question 1: MLP Locality & Memorization Trajectory
python scripts/run_q1.py
python scripts/audit_q1b.py

# Question 2: CNN vs. MLP Spatial Disruption & Calibration
python scripts/run_q2.py

# Question 3: Convolutional Autoencoders, Denoising & Probing
python scripts/run_q3.py

# Question 4 (Bonus): SVM Grid Sweep, Boundary Geometry & Clustering
python scripts/run_q4.py
```

### Re-Compiling the Report
```bash
cd report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
python ../scripts/check_pdf_pages.py
```

See [reproduce.md](reproduce.md) for detailed execution instructions and system configuration notes.
