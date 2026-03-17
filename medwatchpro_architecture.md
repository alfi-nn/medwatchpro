# 🏥 MedWatchPro — System Architecture

> AI-Powered Drug Safety Intelligence using Heterogeneous Graph Transformers
> **AUC-ROC: 0.904 · Biomarker Acc: 90.6% · 891K params · <50ms inference**

---

## 1. End-to-End Pipeline Overview

```mermaid
flowchart TD
    subgraph SOURCES["📦 Raw Data Sources"]
        DB["DrugBank XML\n(~1.9 GB)\ndrugs · proteins · biomarkers\ndti · ddi"]
        SD["SIDER 4.1\nmeddra_all_se.tsv.gz\ndrug_names.tsv"]
        FA["FDA FAERS\n1.14M adverse event reports\n(2012-2021)"]
    end

    subgraph STEP1["🔧 Step 1 — Preprocessing\n01_prepare_nodes_and_edges.py"]
        N1["Drug Nodes\n14,616 drugs w/ SMILES"]
        N2["Protein Nodes\n5,237 proteins (FASTA→clean seq)"]
        N3["Biomarker Nodes\n144 gene-variant pairs"]
        N4["Side-Effect Nodes\nTop 100 from SIDER"]
        E1["edges_dti.csv (30,872)"]
        E2["edges_dti_reverse.csv (30,872)"]
        E3["edges_ddi.csv (2,363,496)"]
        E4["edges_drug_side_effect.csv (45,742)"]
        E5["edges_biomarkers.csv (317)\n+ labels: adverse/efficacy/other"]
        LM["side_effect_labels.npy\n[14616 × 100] binary matrix"]
    end

    subgraph STEP2["🤗 Step 2 — Embeddings\n02_generate_embeddings.py"]
        CB["ChemBERTa-77M-MLM\n(SMILES → CLS token)\n→ drug_embeddings.pt [14616, 384]"]
        PB["ProtBERT\n(AA seq → mean pool)\n→ protein_embeddings.pt [5237, 1024]"]
        RI["Random Init 128-dim\n(biomarkers + side-effects)"]
    end

    subgraph STEP3["🕸️ Step 3 — Graph Construction\n03_build_graph.py"]
        HG["PyG HeteroData\ngraph_data.pt (89 MB)\n80/10/10 train/val/test split"]
    end

    subgraph STEP4["🧠 Step 4+5 — HGT Training\n04_hgt_model.py + 05_train.py"]
        HGT["HGTModel\n891,095 parameters\nTrained on Colab T4 GPU"]
        PT["best_hgt_model.pt (3.6 MB)"]
        TH["training_history.json\nresults.json"]
    end

    subgraph STEP6["⏱️ Step 6 — Temporal Enrichment\n06_temporal_adr.py"]
        FAP["Parse FAERS reports\nCompute time-to-onset per SE"]
        TL["temporal_labels.json\ntemporal_categories.npy"]
    end

    subgraph SERVER["🌐 FastAPI Server — server.py"]
        BOOT["Startup:\nLoad graph + weights\nPre-compute 128-dim embeddings\nBuild XAI neighbor lookup"]
        API["REST API Endpoints"]
    end

    subgraph UI["🖥️ Frontend\nfrontend/index.html + landing.html"]
        DASH["Interactive Dashboard\n(HTML/CSS/JS, no build step)"]
    end

    DB --> STEP1
    SD --> STEP1
    FA --> STEP6

    STEP1 --> STEP2
    STEP2 --> STEP3
    STEP3 --> STEP4
    STEP4 --> PT
    STEP4 --> TH
    STEP6 --> TL

    PT --> SERVER
    HG --> SERVER
    TL --> SERVER

    SERVER --> UI
```

---

## 2. Knowledge Graph Structure

```mermaid
graph LR
    D(["💊 Drug\n14,616 nodes\n[384-dim ChemBERTa]"])
    P(["🧬 Protein\n5,237 nodes\n[1024-dim ProtBERT]"])
    B(["🔬 Biomarker\n144 nodes\n[128-dim trainable]"])
    SE(["⚠️ Side Effect\n100 nodes\n[128-dim trainable]"])

    D -->|"targets (30,872)"| P
    P -->|"targeted_by (30,872)"| D
    D -->|"interacts (2,363,496)"| D
    D -->|"causes (45,742)"| SE
    D -->|"associated_with (317)\nlabeled: adverse/efficacy/other"| B
```

---

## 3. HGT Model Architecture

```mermaid
flowchart TD
    subgraph INPUT["Input Projections (per node type → 128-dim)"]
        ID["Drug: Linear(384→128)"]
        IP["Protein: Linear(1024→128)"]
        IB["Biomarker: Linear(128→128)"]
        IS["Side Effect: Linear(128→128)"]
    end

    subgraph HGT1["HGTConvWithAttn Layer 1\n(128→128, 4 heads)"]
        MP1["Multi-head attention\nover all edge types\n→ captures attention weights α"]
        RN1["+ Residual  + LayerNorm  + Dropout(0.4)"]
    end

    subgraph HGT2["HGTConvWithAttn Layer 2\n(128→128, 4 heads)"]
        MP2["2-hop neighborhood\nmessage aggregation"]
        RN2["+ Residual  + LayerNorm  + Dropout(0.4)"]
    end

    subgraph DECODE["Task Decoders"]
        SE_D["SE Decoder\n128→128→64→100\n+ BN + ReLU + Dropout\n→ Sigmoid\n[Multi-label BCEWithLogits]"]
        BI_D["Bio Decoder\ncat(drug‖bio) 256→128→3\n+ BN + ReLU + Dropout\n→ Softmax\n[CrossEntropy: adverse/efficacy/other]"]
    end

    INPUT --> HGT1 --> HGT2 --> DECODE
```

**Combined loss**: `L = 1.0 × L_SE + 2.0 × L_Bio` — biomarker up-weighted due to sparse data (317 edges).

**Optimizer**: AdamW, lr=1e-3, weight_decay=1e-4, CosineAnnealingLR, gradient clipping (max_norm=1.0), patience=15–20.

---

## 4. Inference & API Layer

```mermaid
flowchart TD
    subgraph STARTUP["Server Startup (once)"]
        L1["Load: graph_data.pt + best_hgt_model.pt"]
        L2["Load: temporal_labels.json (FAERS)"]
        L3["Pre-compute: full forward pass → drug_embeddings[14616,128]\nbio_embeddings[144,128]"]
        L4["Build XAI neighbor tables\n(drug→proteins, drug→drugs)"]
    end

    subgraph ENDPOINTS["REST API Endpoints"]
        E1["GET /api/stats\nModel metrics + graph counts"]
        E2["POST /api/search\nFuzzy drug name lookup"]
        E3["POST /api/predict\nSingle known drug\n→ SE probs + biomarkers + temporal + XAI"]
        E4["POST /api/predict_smiles\nNovel compound (SMILES)\n→ Live ChemBERTa → HGT injection\n→ SE probs + biomarkers"]
        E5["POST /api/predict_combination\n2-3 drugs (polypharmacy)\n→ max+mean pooling → synergy score\n→ known DDI edge check"]
        E6["POST /api/counterfactual\nEdge masking XAI\n→ causal impact per protein target"]
    end

    subgraph XAI["XAI — HGTConvWithAttn"]
        AT["Capture: α = softmax(q·k / √d)\nduring message passing\n→ per-edge attention weights"]
        EX["get_explanations():\nTop neighbor proteins/drugs\nscaled by mean ± σ attention"]
    end

    subgraph TEMPORAL["Temporal Enrichment"]
        TC["5 onset categories:\nAcute (<24h) · Early (1-7d)\nDelayed (1-4w) · Late (1-6m)\nChronic (6m+)"]
        TM["median_days from FAERS\nattached to each SE prediction"]
    end

    STARTUP --> ENDPOINTS
    ENDPOINTS --> XAI
    ENDPOINTS --> TEMPORAL
```

---

## 5. Frontend Routes

| Route | File | Description |
|-------|------|-------------|
| `GET /` | [landing.html](file:///d:/medwatchpro/frontend/landing.html) | Marketing landing page with DNA background |
| `GET /app` | [index.html](file:///d:/medwatchpro/frontend/index.html) | Full interactive dashboard |
| `GET /{file_path}` | Static fallback | Serves images/assets from `frontend/` |

**Dashboard tabs (index.html):**
- **Single Drug** — name/ID search → SE risk bars + temporal badges + biomarker cards + XAI explanation cards
- **Novel Compound** — SMILES input → live ChemBERTa → predictions
- **Combination** — 2-3 drug polypharmacy → synergy scores + known DDI warnings

---

## 6. Key Metrics & File Inventory

### Performance

| Task | Metric | Value |
|------|--------|-------|
| Side-Effect Prediction | ROC-AUC | **0.904** |
| Side-Effect Prediction | F1 (threshold 0.5) | 0.227 |
| Biomarker Classification | Accuracy | **90.6%** |
| Biomarker Classification | F1-Score | 0.898 |
| Training | Epochs (early stop) | 94 |
| Inference latency | Pre-embedded drug | **<50 ms** |

### Critical Files

| File | Role | Size |
|------|------|------|
| [data/processed/graph_data.pt](file:///d:/medwatchpro/data/processed/graph_data.pt) | PyG HeteroData object | 89 MB |
| [data/processed/drug_embeddings.pt](file:///d:/medwatchpro/data/processed/drug_embeddings.pt) | ChemBERTa [14616,384] | 22 MB |
| [data/processed/protein_embeddings.pt](file:///d:/medwatchpro/data/processed/protein_embeddings.pt) | ProtBERT [5237,1024] | 21 MB |
| [data/processed/side_effect_labels.npy](file:///d:/medwatchpro/data/processed/side_effect_labels.npy) | Binary label matrix | 5.8 MB |
| [data/processed/temporal_labels.json](file:///d:/medwatchpro/data/processed/temporal_labels.json) | FAERS onset data | 17 KB |
| [models/best_hgt_model.pt](file:///d:/medwatchpro/models/best_hgt_model.pt) | Trained weights | 3.6 MB |
| [models/training_history.json](file:///d:/medwatchpro/models/training_history.json) | Per-epoch metrics | 9.7 KB |
| [server.py](file:///d:/medwatchpro/server.py) | FastAPI app (788 lines) | 32 KB |
| [frontend/index.html](file:///d:/medwatchpro/frontend/index.html) | Dashboard UI | 61 KB |

---

## 7. Novel Compound (SMILES) Flow

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI /predict_smiles
    participant CB as ChemBERTa
    participant HGT as HGT Model
    participant Decoder

    User->>API: POST {smiles: "CC(=O)Oc1..."}
    API->>CB: tokenize(SMILES)
    CB-->>API: drug_emb [1, 384] (CLS token)
    API->>HGT: Append to x_dict["drug"], run forward()
    HGT-->>API: novel_emb [1, 128] (post-message-passing)
    API->>Decoder: predict_side_effects(novel_emb)
    Decoder-->>API: probs[100] → top-K filtered
    API->>Decoder: predict_biomarker_type(all 144 biomarkers)
    Decoder-->>API: class probs → filter conf > 0.60
    API-->>User: side_effects + biomarkers + temporal + XAI
```

---

## 8. Polypharmacy Synergy Scoring

```
Individual:  P(drug_A)[100]  P(drug_B)[100]
             ↓               ↓
             emb_A [128]     emb_B [128]
                    ↓
             max_pool + mean_pool / 2  →  combined_emb [128]
                    ↓
             P(combined)[100]
                    ↓
Synergy[i] = P(combined)[i] - max(P(drug_A)[i], P(drug_B)[i])
             > 0.05 → "interaction_amplified": true
```
