<p align="center">
  <h1 align="center">🏥 MedWatchPro</h1>
  <p align="center">
    <strong>AI-Powered Drug Safety Intelligence using Heterogeneous Graph Transformers</strong>
  </p>
  <p align="center">
    <em>Predict adverse drug reactions, biomarker interactions, and polypharmacy risks with explainable AI</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-2.6-3C2179?logo=pyg" alt="PyG">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/AUC-0.904-brightgreen" alt="AUC">
  <img src="https://img.shields.io/badge/Bio_Acc-90.6%25-brightgreen" alt="Bio Acc">
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset Details](#dataset-details)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Installation & Setup](#installation--setup)
- [Training on Google Colab](#training-on-google-colab)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Research Contributions](#research-contributions)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Overview

**MedWatchPro** is an end-to-end drug safety intelligence platform that combines heterogeneous graph neural networks with pretrained biomedical language models to predict:

1. **Adverse Drug Reactions (ADRs)** — Multi-label classification across 100 side effects
2. **Biomarker Interactions** — Whether genetic variants cause adverse/efficacy responses
3. **Polypharmacy Risks** — Drug combination safety with synergy scoring
4. **Explainable Predictions** — Attention-based explanations showing *why* the model predicts specific risks

The system constructs a large-scale biomedical knowledge graph from DrugBank and SIDER, encodes molecular & protein structures using ChemBERTa and ProtBERT, and processes them through a Heterogeneous Graph Transformer (HGT) for multi-task prediction.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Side-Effect Prediction** | Multi-label prediction across 100 ADRs with calibrated probabilities |
| 🧬 **Biomarker Analysis** | Predict adverse/efficacy/other response for drug-biomarker pairs |
| 💊 **Polypharmacy Safety** | Analyze 2-3 drug combinations with synergy scoring & DDI warnings |
| 🧠 **Explainable AI (XAI)** | HGT attention weights reveal which proteins/drugs drive predictions |
| 🧪 **Novel Compound Analysis** | Predict side effects for new drugs using SMILES strings (via ChemBERTa) |
| ⏱️ **Temporal Onset Prediction** | Predict *when* side effects appear (acute, early, delayed, late, chronic) using real-world FDA FAERS data |
| 🌐 **Interactive Dashboard** | Dark-themed web UI with real-time search, risk bars, temporal badges, and explanation cards |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedWatchPro                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ DrugBank │   │  SIDER   │   │ChemBERTa │   │ ProtBERT │    │
│  │  (XML)   │   │  (TSV)   │   │  (384d)  │   │ (1024d)  │    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘    │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│  ┌──────────────────────────────────────────────────────┐      │
│  │           Heterogeneous Knowledge Graph              │      │
│  │  14,616 Drugs  ──targets──▸  5,237 Proteins          │      │
│  │       │                          │                    │      │
│  │  interacts(2.3M)           targeted_by                │      │
│  │       │                          │                    │      │
│  │  causes(45K)──▸ 100 Side Effects                     │      │
│  │  associated(317)──▸ 144 Biomarkers                   │      │
│  └──────────────────────┬───────────────────────────────┘      │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │          HGT Model (891K params)                      │      │
│  │  Input Projections → 2× HGTConvWithAttn → Decoders   │      │
│  │                                                       │      │
│  │  SE Decoder (128→64→100)  Bio Decoder (256→128→3)    │      │
│  └──────────────────────┬───────────────────────────────┘      │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              FastAPI REST Server                      │      │
│  │  /predict  /predict_smiles  /predict_combination     │      │
│  └──────────────────────┬───────────────────────────────┘      │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │          Frontend Dashboard (HTML/CSS/JS)            │      │
│  │  Single Drug │ Combination │ XAI Explanations        │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dataset Details

### Data Sources

| Source | Description | Size |
|--------|-------------|------|
| **[DrugBank](https://go.drugbank.com/)** | Comprehensive drug database with targets, interactions, biomarkers | ~1.9 GB XML |
| **[SIDER](http://sideeffects.embl.de/)** | Side effects of marketed medicines (MedDRA coded) | ~2.3 MB |
| **[FAERS](https://github.com/kimkimjh/temp-FAERS)** | FDA Adverse Event Reporting System (2012-2021) for temporal onset mapping | ~2 GB CSVs |

### Required DrugBank Files

You need a DrugBank account (academic license is free) to download:

| File | Content | Used For |
|------|---------|----------|
| `drugs_features.csv` | Drug properties, SMILES strings | Drug nodes, ChemBERTa input |
| `proteins.csv` | Target protein sequences | Protein nodes, ProtBERT input |
| `biomarkers.csv` | Gene-variant-drug associations | Biomarker nodes |
| `dti.csv` | Drug-Target Interactions | DTI edges (30,872) |
| `ddi.csv` | Drug-Drug Interactions | DDI edges (2,363,496) |

### Required SIDER Files

Download from [SIDER 4.1](http://sideeffects.embl.de/download/):

| File | Content | Used For |
|------|---------|----------|
| `meddra_all_se.tsv.gz` | Drug-side effect associations | Side effect edges, labels |
| `drug_names.tsv` | SIDER drug name mappings | Linking SIDER ↔ DrugBank |

### Processed Graph Statistics

| Component | Count |
|-----------|-------|
| **Drug nodes** | 14,616 |
| **Protein nodes** | 5,237 |
| **Biomarker nodes** | 144 |
| **Side effect nodes** | 100 |
| Drug–targets–Protein edges | 30,872 |
| Drug–interacts–Drug edges | 2,363,496 |
| Drug–causes–Side Effect edges | 45,742 |
| Drug–associated_with–Biomarker edges | 317 |
| **Total parameters** | 891,095 |

### Embeddings

| Type | Model | Dimensions | Generated On |
|------|-------|------------|-------------|
| Drug | `DeepChem/ChemBERTa-77M-MLM` | 384 | Local GPU / CPU |
| Protein | `Rostlab/prot_bert` | 1024 | Google Colab T4 |
| Biomarker | Random initialized | 128 | — |
| Side Effect | Random initialized | 128 | — |

---

## Model Architecture

### Heterogeneous Graph Transformer (HGT)

```
Input: x_dict (node features per type)
  │
  ├── Input Projections (per node type → 128-dim)
  │     Drug:      Linear(384  → 128)
  │     Protein:   Linear(1024 → 128)
  │     Biomarker: Linear(128  → 128)
  │     SideEffect:Linear(128  → 128)
  │
  ├── HGTConvWithAttn Layer 1 (128→128, 4 heads) + LayerNorm + Residual
  ├── HGTConvWithAttn Layer 2 (128→128, 4 heads) + LayerNorm + Residual
  │
  ├── SE Decoder: Linear(128→128) → BN → ReLU → Drop(0.4)
  │                → Linear(128→64) → BN → ReLU → Drop(0.4)
  │                → Linear(64→100) → Sigmoid
  │
  └── Bio Decoder: Linear(256→128) → BN → ReLU → Drop(0.4)
                    → Linear(128→3) → Softmax
```

### XAI: Attention Weight Capture

We created `HGTConvWithAttn`, a custom subclass of PyG's `HGTConv` that hooks into the `message()` function to capture attention weights during inference — **without retraining**:

```python
class HGTConvWithAttn(HGTConv):
    def message(self, k_j, q_i, v_j, edge_attr, index, ptr, size_i):
        alpha = softmax((q_i * k_j).sum(-1) * edge_attr / sqrt(D))
        self._attn_weights = alpha.detach().mean(dim=-1)  # Captured!
        return (v_j * alpha).view(-1, self.out_channels)
```

### Polypharmacy: Embedding Aggregation

For drug combinations, we aggregate individual HGT embeddings:

```python
combined = (max_pool(embeddings) + mean_pool(embeddings)) / 2
synergy = P(combined) - max(P(individual))  # per side effect
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Side-Effect AUC-ROC** | 0.904 |
| **Biomarker Accuracy** | 90.6% |
| **Biomarker F1-Score** | 0.898 |
| **Training Epochs** | 94 (early stopped at patience=20) |
| **Training Time** | ~94 seconds (Colab T4 GPU) |

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, CPU works for inference)
- ~4 GB disk space for processed data

### 1. Clone & Setup

```bash
git clone https://github.com/<your-username>/MedWatchPro.git
cd MedWatchPro

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

1. **DrugBank** — Register at [drugbank.com](https://go.drugbank.com/) (free academic license)
   - Download: `drugs_features.csv`, `proteins.csv`, `biomarkers.csv`, `dti.csv`, `ddi.csv`
   - Place in: `drugbank_all_full_database.xml/`

2. **SIDER 4.1** — Download from [sideeffects.embl.de](http://sideeffects.embl.de/download/)
   - Download: `meddra_all_se.tsv.gz`, `drug_names.tsv`
   - Place in project root

### 3. Run Data Preprocessing

```bash
python scripts/01_prepare_nodes_and_edges.py
python scripts/02_generate_embeddings.py
```

> **Note**: ProtBERT embedding generation requires ~8GB VRAM. If your GPU is smaller, use Google Colab (see below).

---

## Training on Google Colab

The HGT model should be trained on a GPU with ≥8GB VRAM. We recommend Google Colab (free T4 GPU).

### Step-by-Step Colab Instructions

1. **Upload files to Colab** (or mount Google Drive):
   ```
   data/processed/graph_data.pt         (~89 MB)
   data/processed/nodes_drugs.csv
   data/processed/nodes_proteins.csv
   data/processed/nodes_biomarkers.csv
   data/processed/nodes_side_effects.csv
   colab_train.py
   ```

2. **Install dependencies** in a Colab cell:
   ```python
   !pip install torch-geometric transformers
   ```

3. **Run training**:
   ```python
   !python colab_train.py
   ```
   This will:
   - Load the heterogeneous graph
   - Train the HGT model for up to 100 epochs (early stopping at patience=20)
   - Save `best_hgt_model.pt`, `results.json`, and `training_history.json`

4. **Download trained files** back to your local machine:
   ```
   best_hgt_model.pt     → models/
   results.json           → models/
   training_history.json  → models/
   ```

### Expected Training Output

```
Device: cuda | GPU: Tesla T4 | VRAM: 14.6 GB
Params: 891,095

Training 100 epochs (patience=20)
  Ep  1 | Loss 3.9947 | AUC:0.515  F1:0.060 | BioAcc:0.69
  Ep 20 | Loss 0.6422 | AUC:0.878  F1:0.199 | BioAcc:0.91
  Ep 50 | Loss 0.4190 | AUC:0.901  F1:0.225 | BioAcc:0.91
  Ep 94 | Loss 0.3564 | AUC:0.904  F1:0.227 | BioAcc:0.91
Early stopping at epoch 94
```

### ProtBERT on Colab (if needed)

If you cannot run ProtBERT locally:

```python
# In a Colab notebook:
!pip install transformers torch

from transformers import AutoTokenizer, AutoModel
import torch, pandas as pd

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
model = AutoModel.from_pretrained("Rostlab/prot_bert").to("cuda")
model.eval()

# Upload nodes_proteins.csv to Colab first
proteins = pd.read_csv("nodes_proteins.csv")
embeddings = []

for seq in proteins["clean_sequence"]:
    spaced = " ".join(list(seq[:1000]))  # Truncate if needed
    inputs = tokenizer(spaced, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    embeddings.append(out.last_hidden_state[:, 0, :].cpu())

protein_embs = torch.cat(embeddings, dim=0)
torch.save(protein_embs, "protein_embeddings.pt")
# Download protein_embeddings.pt → data/processed/
```

---

## Running the Application

### Start the Server

```bash
# Activate virtual environment
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

# Start FastAPI server
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Access the Dashboard

Open **http://localhost:8000** in your browser.

### Features Available

- **Single Drug Analysis** — Search by name or DrugBank ID
- **Novel Compound Analysis** — Enter a SMILES string for predictions on unknown drugs
- **Drug Combination Analysis** — Enter 2-3 drugs to assess polypharmacy risk
- **Model Explanation** — See which proteins/drugs influenced each prediction

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stats` | System statistics (drug count, model metrics) |
| `POST` | `/api/search` | Search drugs by name or ID |
| `POST` | `/api/predict` | Predict side effects & biomarkers for a known drug |
| `POST` | `/api/predict_smiles` | Predict for a novel compound via SMILES string |
| `POST` | `/api/predict_combination` | Polypharmacy analysis for 2-3 drugs |

### Example API Call

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"drug_name": "aspirin", "top_k": 10, "threshold": 0.3}'
```

### Example Response

```json
{
  "drug": {"id": "DB00945", "name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(O)=O"},
  "side_effects": [
    {"name": "nausea", "probability": 0.8521, "risk_level": "high"},
    {"name": "headache", "probability": 0.8467, "risk_level": "high"}
  ],
  "biomarkers": [...],
  "explanations": [
    {"type": "protein", "name": "Prostaglandin G/H synthase 2", "attention": 0.082},
    {"type": "protein", "name": "Prostaglandin G/H synthase 1", "attention": 0.061}
  ]
}
```

---

## Project Structure

```
MedWatchPro/
├── server.py                          # FastAPI server (XAI, polypharmacy)
├── colab_train.py                     # Google Colab training script
├── requirements.txt                   # Python dependencies
├── .gitignore
├── README.md
├── FINAL_DOCUMENTATION.md             # Detailed research documentation
├── PROGRESS_LOG.md                    # Development progress log
│
├── frontend/
│   └── index.html                     # Dashboard UI
│
├── scripts/
│   ├── 01_prepare_nodes_and_edges.py  # Data preprocessing pipeline
│   ├── 02_generate_embeddings.py      # ChemBERTa + ProtBERT embeddings
│   └── 04_hgt_model.py               # HGT model class definition
│
├── models/
│   ├── best_hgt_model.pt              # Trained model weights (3.6 MB)
│   ├── results.json                   # Final evaluation metrics
│   └── training_history.json          # Per-epoch training log
│
└── data/
    └── processed/
        ├── graph_data.pt              # PyG HeteroData graph (89 MB)
        ├── drug_embeddings.pt         # ChemBERTa embeddings (22 MB)
        ├── protein_embeddings.pt      # ProtBERT embeddings (21 MB)
        ├── side_effect_labels.npy     # Binary label matrix
        ├── nodes_drugs.csv            # 14,616 drug nodes
        ├── nodes_proteins.csv         # 5,237 protein nodes
        ├── nodes_biomarkers.csv       # 144 biomarker nodes
        ├── nodes_side_effects.csv     # 100 side effect nodes
        ├── edges_dti.csv              # Drug-Target edges (30K)
        ├── edges_ddi.csv              # Drug-Drug edges (2.3M)
        ├── edges_drug_side_effect.csv # Drug-SE edges (45K)
        └── edges_biomarkers.csv       # Drug-Biomarker edges (317)
```

---

## Research Contributions

### Novel Aspects

1. **Multi-Modal HGT**: Combines ChemBERTa (molecular), ProtBERT (protein sequence), and knowledge graph embeddings in a unified heterogeneous graph transformer — most existing work uses only one modality.

2. **Attention-Based XAI without Retraining**: Custom `HGTConvWithAttn` subclass captures attention weights from the HGT message-passing mechanism, providing interpretable explanations without model modification or retraining.

3. **Polypharmacy Synergy Scoring**: Novel embedding aggregation (max + mean pooling) for drug combination risk prediction, with synergy scores that quantify interaction-amplified adverse effects.

4. **Dual-Objective Training**: Joint optimization for side-effect prediction (multi-label) and biomarker interaction classification (multi-class) in a single forward pass.

5. **Zero-Shot Novel Compound Prediction**: ChemBERTa generates embeddings for unseen SMILES strings, which are dynamically integrated into the HGT graph for inference.

6. **Real-World Temporal Mapping**: Processed 1.14M FDA FAERS adverse event reports to calculate median time-to-onset for drug-reaction pairs, providing clinical context (e.g., "within 24 hours" vs "1-6 months") alongside static risk probabilities.

---

## Limitations & Future Work

### Current Limitations

- Biomarker edge count is small (317 edges for 144 biomarkers)
- Side-effect labels are binary (no severity or temporal information)
- Polypharmacy uses simple embedding aggregation rather than learned interaction functions
- XAI uses graph-level attention statistics; per-edge attribution mapping is a future goal

### Planned Improvements

- **Counterfactual Explanations**: Edge-masking to show causal reasoning ("if drug X didn't target Y...")
- **Patient-Specific Risk**: Incorporate demographics for personalized predictions
- **Severity Classification**: Predict mild/moderate/severe/life-threatening risk levels

---

## Citation

If you use MedWatchPro in your research, please cite:

```bibtex
@software{medwatchpro2026,
  title={MedWatchPro: AI-Powered Drug Safety Intelligence using HGT},
  year={2026},
  url={https://github.com/<your-username>/MedWatchPro}
}
```

---

## License

This project is for academic and research purposes. DrugBank data requires a separate license from [DrugBank](https://go.drugbank.com/).
