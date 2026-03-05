# MedWatchPro: AI-Powered Drug Safety & Biomarker Analysis
**Comprehensive Project Documentation**
**Date**: March 2026

---

## 1. Project Overview
**MedWatchPro** is an end-to-end artificial intelligence pipeline designed to predict adverse drug reactions (ADRs) and classify drug-biomarker interactions. By leveraging state-of-the-art **Heterogeneous Graph Transformer (HGT)** architectures combined with advanced Deep Learning language models (ChemBERTa and ProtBERT), the system processes complex multi-relational graphs of drugs, proteins, and genetic variants to provide high-accuracy safety intelligence.

### 1.1 Core Objectives
1. **Side-Effect Prediction**: Predict the probability of 100 high-impact adverse events for any given drug compound.
2. **Biomarker Classification**: Determine whether a specific genetic variant (biomarker) leads to an *adverse* reaction or an *efficacy* response when exposed to a specific drug.
3. **Novel Compound Analysis**: Enable real-time prediction for completely new, out-of-database molecules using only their SMILES strings.
4. **Temporal Onset Prediction**: Predict *when* side effects are likely to occur (acute to chronic) using real-world FDA FAERS data integration.

---

## 2. System Architecture & Methodology

### 2.1 Graph Construction Structure
The core of the system is a heterogeneous property graph built from DrugBank and SIDER databases.
- **Nodes (4 Types)**: Drugs (14,616), Proteins (5,237), Biomarkers (144), Side-Effects (100).
- **Edges (5 Types)**:
  - `(drug, targets, protein)`: 30,872 edges representing pharmacological targets.
  - `(protein, targeted_by, drug)`: 30,872 reverse edges.
  - `(drug, interacts, drug)`: 2,363,496 edges representing known DDI (Drug-Drug Interactions).
  - `(drug, causes, side_effect)`: 45,742 edges.
  - `(drug, associated_with, biomarker)`: 317 edges with specialized labels (Efficacy, Adverse, Other) for link-prediction tasks.

### 2.2 Molecular Translation (Embeddings)
To feed raw biological data into the graph network, pre-trained transformer models were utilized to extract dense feature representations:
- **Drugs**: `DeepChem/ChemBERTa-77M-MLM` was used to tokenize and embed SMILES strings into **384-dimensional** continuous vectors.
- **Proteins**: `Rostlab/prot_bert` was used to translate amino-acid sequences into **1,024-dimensional** continuous vectors.

### 2.3 The HGT Model
A PyTorch Geometric **Heterogeneous Graph Transformer (HGT)** was implemented with:
- **2 HGTConv Layers**: Allows information to propagate across a 2-hop neighborhood, effectively combining target information, drug-drug interactions, and structural embeddings.
- **Hidden Dimension**: 128
- **Attention Heads**: 4
- **Parameters**: 891,095
- **Dual-Task Decoder**:
  1. A Multi-Layer Perceptron (MLP) mapping the final 128-dim drug embeddings to 100 independent sigmoid outputs (Multi-label classification of Side-Effects).
  2. A secondary MLP taking the concatenated embeddings of a (Drug, Biomarker) pair to predict a 3-class categorical output (Adverse, Efficacy, Other).
- **Temporal Mapping**: Real-world FDA FAERS data (1.14M records) mapped to predicted side-effects to provide median onset times and risk categories (e.g., Acute <24h).

---

## 3. Results and Performance

The model was trained on a Tesla T4 GPU (Google Colab) for 94 epochs using early stopping.

### 3.1 Quantitative Metrics
| Task | Metric | Test Set Score |
|------|--------|----------------|
| **Side-Effect Prediction** | **ROC-AUC** | **0.904** |
| Side-Effect Prediction | F1-Score | 0.227 (High precision threshold tuning required) |
| **Biomarker Link Prediction** | **Accuracy** | **90.6%** |
| Biomarker Link Prediction | F1-Score | 0.898 |

*Note: Achieving >0.90 AUC on multi-label adverse event prediction significantly outperforms traditional baseline models (which typically hover around ~0.75 - 0.80 AUC).*

### 3.2 System Capabilities
- **Real-Time Inference**: The FastAPI backend pre-computes the final 128-dim graph embeddings at startup. End-user queries resolve in <50ms.
- **Live SMILES Processing**: Novel compounds dynamically pass through the ChemBERTa model, get injected into the live graph, undergo forward-pass message passing, and return predictions instantly.

---

## 4. Research Gaps & Limitations

While the system performs exceptionally well, several critical limitations and opportunities for future research remain:

### 4.1 Data & Graph Limitations
1. **Sparse Biomarker Data**: The graph contains 14,000+ drugs but only 317 highly-curated biomarker interaction edges. This severe class imbalance limits the model's ability to generalize complex pharmacogenomic interactions for rarer genetic variants.
2. **Missing Edge Attributes**: Currently, `(drug, interacts, drug)` edges are unweighted and unlabelled. In reality, a DDI might represent a *synergistic* effect or an *antagonistic* clearance inhibition. The model is forced to infer this latent relationship implicitly.
3. **Sequence Length Truncation**: Protein sequences often exceed 1,000 amino acids. Due to memory constraints (local 4GB VRAM and Colab 16GB VRAM), sequences were truncated to 512 tokens during ProtBERT embedding extraction, resulting in a loss of structural information at the C-terminus of large proteins.

### 4.2 Architectural Limitations
1. **Static Graph for Inference**: When evaluating novel SMILES strings, the system creates "dummy" edges to existing biomarkers to force the message-passing step. However, it cannot dynamically predict new `(drug, targets, protein)` edges on the fly. Therefore, the novel drug's HGT embedding relies heavily on its structural ChemBERTa embedding rather than its *actual* biological neighborhood.
2. **Lack of Dose Dependency**: Adverse reactions are highly dose-dependent, but the current graph schema treats side effects as binary (Has_Effect / No_Effect). A toxic prediction does not indicate the threshold required to trigger it.

### 4.3 Evaluation Gaps
1. **Low F1-Score on Side Effects**: The ROC-AUC is excellent (0.904), but the standard F1-score is low (0.227). This occurs because side-effects are a heavily imbalanced problem (most drugs do not cause most side effects). The model requires rigorous threshold tuning (moving away from standard >0.5 threshold) using Precision-Recall Area Under Curve (PR-AUC) metrics down the line.

---

## 5. Future Scope & Enhancements

1. **Integration of 3D Conformational Data**: Upgrading from ChemBERTa (1D sequence representation) to 3D Graph Neural Networks (like SchNet or DimeNet) that process the actual spatial coordinates and bond angles of the compounds.
2. **DDI Edge Classification**: Adding a third objective to the model to predict the specific *type* of drug-drug interaction (e.g., "Drug A decreases the metabolism of Drug B").
3. **Dynamic Edge Prediction**: Allowing novel SMILES inputs to first predict their protein targets, update the graph structure, and *then* run the HGT message-passing algorithm for highly accurate zero-shot predictions.
4. **Attention Interpretability**: Utilizing the attention weights derived from the HGT layers to highlight *which* specific neighboring node (e.g., which target protein) was most responsible for a predicted adverse reaction, adding "Explainable AI" (XAI) to the pipeline.

---
*Developed as an advanced Graph AI research implementation.*
