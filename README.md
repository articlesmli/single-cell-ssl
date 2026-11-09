# single-cell-ssl: Self-Supervised Foundation Model for Single-Cell Transcriptomics

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade deep learning research codebase implementing a **Masked Autoencoder (MAE)** Transformer architecture for self-supervised representation learning (SSL) on single-cell RNA sequencing (scRNA-seq) data.

---

##  Project Overview & Research Context

High-dimensional single-cell transcriptomics generates massive matrices of gene expression profiles, but acquiring manually labeled cell types or perturbation outcomes is expensive and slow. 

`single-cell-ssl` addresses this by implementing a **Self-Supervised Biological Foundation Model**. By pretraining a Transformer encoder on large, unlabeled cell atlases (e.g., Tabula Sapiens), the model learns the foundational "syntax" of gene co-expression networks, cellular states, and developmental trajectories without human supervision.

### Core Capabilities Unlocked:
1. **Zero-Shot & Fine-Tuned Cell Typing:** Extracts dense cellular embeddings to map complex populations and classify cell types with minimal labeled data.
2. **In Silico Perturbation Screening:** Simulates cellular responses to gene knockouts or drug treatments.
3. **Expression Imputation & Denoising:** Reconstructs technical dropout noise inherent in single-cell sequencing.

---

##  Technical Architecture & Design

The model follows a custom **Masked Autoencoder (MAE)** paradigm tailored for continuous gene expression values:
* **Gene Tokenization & Embedding:** Continuous gene expression values are projected into dense vector spaces via custom linear embedding layers.
* **Masking Strategy:** A configurable percentage (e.g., 30–50%) of gene expression values are randomly masked and replaced with learnable `[MASK]` tokens.
* **Transformer Encoder-Decoder:** Visible tokens pass through multi-head self-attention Transformer encoder blocks, while a lightweight decoder reconstructs the masked gene expression profiles via Mean Squared Error (MSE) optimization.


```

[ Raw scRNA-seq Data (.h5ad) ]
│
▼
(Random Masking: 40%) ──► [ Transformer Encoder ] ──► [ Transformer Decoder ]
│                                                              │
└───────────────────── (MSE Reconstruction Loss) ──────────────┘

```

---

##  Repository File Tree

```text
single-cell-ssl/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── configs/
│   ├── base_config.yaml
│   └── pretrain_mae.yaml
├── data/
│   ├── raw/                  # (Git ignored) Original .h5ad files
│   └── processed/            # (Git ignored) Preprocessed expression matrices
├── notebooks/
│   ├── 01_eda_scanpy.ipynb   # Exploratory data analysis & QC
│   └── 02_embedding_eval.ipynb # Zero-shot visualization & clustering
├── src/
│   └── scgenemae/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py    # PyTorch Dataset class & AnnData loader
│       │   └── preprocessing.py # Scanpy normalization & HVG selection
│       ├── models/
│       │   ├── __init__.py
│       │   ├── embeddings.py # Gene tokenization & continuous value embedding
│       │   └── mae.py        # Masked Autoencoder Transformer architecture
│       ├── training/
│       │   ├── __init__.py
│       │   ├── evaluate.py   # Downstream cell type extraction & metrics
│       │   └── train.py      # Self-supervised pretraining loop with PyTorch
│       └── utils/
│           ├── __init__.py
│           └── logger.py     # Checkpointing & state management helpers
└── tests/
    ├── __init__.py
    ├── test_data.py          # Pytest for dataset shapes and masking logic
    └── test_model.py         # Pytest for Transformer forward/backward passes

```

---

##  Installation & Environment Setup

Ensure you have **Python 3.10+** and a CUDA-compatible GPU installed.

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/single-cell-ssl.git](https://github.com/your-username/single-cell-ssl.git)
cd single-cell-ssl

# 2. Create and activate an isolated virtual environment
python3 -m venv ssl
source ssl/bin/activate

# 3. Upgrade pip and install the package with development dependencies
pip install --upgrade pip
pip install -e ".[dev]"

```

---

##  Usage Guide

### Step 1: Preprocess Single-Cell Data

Run the scanpy preprocessing script to filter cells, normalize counts, log-transform, and select Highly Variable Genes (HVGs):

```python
from scgenemae.data.preprocessing import preprocess_adata

preprocess_adata(
    adata_path="data/raw/sample_cells.h5ad",
    save_path="data/processed/processed_cells.h5ad",
    n_top_genes=2000
)

```

### Step 2: Run Self-Supervised Pretraining

Execute the pretraining loop to train the Transformer MAE using your configuration file or command-line arguments:

```bash
python src/scgenemae/training/train.py \
    --data_path data/processed/processed_cells.h5ad \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4

```

---

##  Testing & Quality Assurance

To fulfill rigorous software engineering and reproducibility standards, the codebase includes a comprehensive `pytest` suite verifying data loader dimensions, masking ratios, and model tensor shapes:

```bash
pytest tests/

```

---

## License

Distributed under the **MIT License**. See the `LICENSE` file for more details.
