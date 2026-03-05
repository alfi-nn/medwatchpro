# MedWatchPro - Project Progress Log
> **AI-Powered Adverse Drug Reaction & Biomarker Prediction using HGT**
> Last Updated: 2026-03-05 09:55 IST
> **Project Location**: `D:\medwatchpro` (moved from C: for disk space)

---

## Project Overview
Build a Heterogeneous Graph Transformer (HGT) to predict:
1. **General ADRs**: Drug side effects (from SIDER)
2. **Personalized Biomarker Reactions**: Whether a patient's genetic variant causes adverse or efficacy effects with a drug (from DrugBank biomarkers)
3. **Explainable AI**: Attention-based model explanations showing which neighbors influence predictions
4. **Polypharmacy Safety**: Drug combination risk analysis with synergy scoring

**Data Sources**: DrugBank XML/CSVs, SIDER database
**Embedding Models**: ChemBERTa (drugs), ProtBERT (proteins)
**Framework**: PyTorch + PyTorch Geometric

---

## Environment
- **OS**: Windows
- **Python**: 3.11.9 (venv at `medwatchpro/venv/`)
- **GPU**: NVIDIA RTX 3050 Laptop (4GB VRAM)
- **Key Packages**: PyTorch 2.6.0+cu124, Transformers, pandas, numpy, scikit-learn

---

## Progress Timeline

### Session 1 (2026-03-04)

#### Phase 1: Planning [COMPLETED]
- Analyzed DrugBank XML structure and all CSV files
- Designed heterogeneous graph schema: 4 node types, 5+ edge types
- Planned feature extraction using ChemBERTa and ProtBERT
- Created and approved `implementation_plan.md`

#### Phase 2: Data Preprocessing [COMPLETED]
**Script**: `scripts/01_prepare_nodes_and_edges.py`

| Output File | Description | Count/Size |
|---|---|---|
| `nodes_drugs.csv` | Drugs with SMILES | 14,616 nodes |
| `nodes_proteins.csv` | Proteins with sequences | 5,237 nodes |
| `nodes_biomarkers.csv` | Gene + variant pairs | 144 nodes |
| `nodes_side_effects.csv` | Top 100 from SIDER | 100 nodes |
| `edges_dti.csv` | Drug-Target Interactions | 30,872 edges |
| `edges_dti_reverse.csv` | Reverse DTI | 30,872 edges |
| `edges_ddi.csv` | Drug-Drug Interactions | 2,363,496 edges |
| `edges_drug_side_effect.csv` | Drug-SideEffect | 45,742 edges |
| `edges_biomarkers.csv` | Drug-Biomarker (target) | 317 edges |
| `side_effect_labels.npy` | Binary matrix 14616x100 | 5.6 MB |

**Key Fix**: DTI ID mismatch (dti.csv uses `BE*` IDs, proteins.csv uses UniProt IDs). Resolved by joining on protein name. 922 rows unmatched.

**Label Stats**: Side-effect label mean positive rate = 3.13% (range 1.96%-6.27%)
**Biomarker Labels**: 187 efficacy, 130 adverse

#### Phase 3: Embedding Generation [COMPLETED]
**Script**: `scripts/02_generate_embeddings.py`

| Embedding | Model | Dim | Status |
|---|---|---|---|
| Drug (ChemBERTa) | DeepChem/ChemBERTa-77M-MLM | 384 | DONE (GPU) |
| Protein (ProtBERT) | Rostlab/prot_bert | 1024 | DONE (Colab T4) |

**Issue**: ProtBERT (~1.7GB model) causes CUDA OOM on 4GB VRAM. Successfully completed on Google Colab with T4 GPU.

#### Phase 4: HGT Model & Training [COMPLETED]
**Scripts**: `scripts/04_hgt_model.py`, `colab_train.py`

- **HGT Architecture**: 2 HGTConv layers, 128 hidden dim, 4 attention heads, 891K params
- **Dual objectives**: Side-effect multi-label prediction + biomarker edge classification
- **Trained on Colab T4**: 94 epochs (~94 sec), early stopped
- **Results**: Side-Effect AUC: 0.904, Biomarker Accuracy: 90.6%, Biomarker F1: 0.898

#### Phase 5: API & Frontend [COMPLETED]
**Scripts**: `server.py`, `frontend/index.html`

- **FastAPI REST API**: Drug search, side-effect prediction, biomarker classification
- **Modern dashboard**: Dark-themed UI with search, risk bars, biomarker cards
- **Novel SMILES prediction**: ChemBERTa on-the-fly embedding for unknown compounds
- **Running at**: `http://localhost:8000`

---

### Session 2 (2026-03-05)

#### Phase 6: Explainable AI (XAI) via Attention Weights [COMPLETED]

**Problem**: HGTConv computes attention weights internally but discards them — no way to explain predictions.

**Solution**: Created `HGTConvWithAttn` subclass that hooks into `message()` to capture attention weights:
```python
class HGTConvWithAttn(HGTConv):
    def message(self, k_j, q_i, v_j, edge_attr, index, ptr, size_i):
        alpha = softmax((q_i * k_j).sum(-1) * edge_attr / sqrt(D))
        self._attn_weights = alpha.detach().mean(dim=-1)  # capture!
        return (v_j * alpha).view(-1, self.out_channels)
```

**Key Design Decision**: No retraining needed — the subclass uses identical parameter names, so existing model weights load directly.

**Changes**:
- `server.py`: Added `HGTConvWithAttn`, `get_last_layer_attention()`, `get_explanations()` function
- `frontend/index.html`: New "Model Explanation (XAI)" card with type badges (purple=protein, blue=drug), attention bars, and scores

**Verification**: Aspirin correctly showed Prostaglandin G/H synthase 1 & 2 (COX-1, COX-2) as top influences — its real biological targets.

#### Phase 7: Polypharmacy Drug Combination Safety [COMPLETED]

**Problem**: No existing system predicts side effects for drug *combinations* — critical for elderly patients on multiple medications.

**Solution**: New `POST /api/predict_combination` endpoint:
1. Resolve 2-3 drug names to their HGT node embeddings
2. Aggregate via **max + mean pooling**: `combined = (max_pool + mean_pool) / 2`
3. Run through `se_decoder` for combined side-effect prediction
4. Compute **synergy scores**: `P(combined) - max(P(individual))` per side effect
5. Check for direct DDI edges in the 2.3M drug-drug interaction graph
6. Flag side effects where synergy > 5% as "interaction-amplified"

**Changes**:
- `server.py`: New `CombinationRequest` model, `/api/predict_combination` endpoint
- `frontend/index.html`: Mode toggle (Single Drug / Drug Combination), multi-drug input fields, combo results panel with DDI warnings, summary cards, synergy badges

**Features**:
- ⚠️ Known DDI warnings (checks 2.3M interaction edges)
- 🔴 "Amplified" badges on synergy-boosted side effects
- 📊 Per-drug individual scores for comparison
- Summary: drugs combined, interaction-amplified count, known DDI pairs

---

## All Phases Complete ✅

| Phase | Description | Status |
|---|---|---|
| 1 | Planning & Design | ✅ COMPLETED |
| 2 | Data Preprocessing | ✅ COMPLETED |
| 3 | Embedding Generation | ✅ COMPLETED |
| 4 | HGT Model & Training | ✅ COMPLETED |
| 5 | API & Frontend | ✅ COMPLETED |
| 6 | Explainable AI (XAI) | ✅ COMPLETED |
| 7 | Polypharmacy Safety | ✅ COMPLETED |

---

## File Structure
```
medwatchpro/
  venv/                          # Python 3.11 virtual environment
  requirements.txt               # All dependencies
  server.py                      # FastAPI server (HGTConvWithAttn, XAI, polypharmacy)
  colab_train.py                 # Colab training script
  FINAL_DOCUMENTATION.md         # Comprehensive project documentation
  PROGRESS_LOG.md                # This file
  frontend/
    index.html                   # Dashboard (single drug, combo, XAI)
  scripts/
    01_prepare_nodes_and_edges.py # Data preprocessing
    02_generate_embeddings.py    # ChemBERTa + ProtBERT
    04_hgt_model.py              # HGT model definition
  models/
    best_hgt_model.pt            # Trained model weights
    results.json                 # Training metrics
    training_curves.png          # Loss/AUC plots
  data/
    processed/
      graph_data.pt              # PyG HeteroData graph
      nodes_drugs.csv            # 14,616 drugs
      nodes_proteins.csv         # 5,237 proteins
      nodes_biomarkers.csv       # 144 biomarkers
      nodes_side_effects.csv     # 100 side effects
      edges_*.csv                # All edge files
      drug_embeddings.pt         # ChemBERTa [14616 x 384]
      protein_embeddings.pt      # ProtBERT [5237 x 1024]
```

---

## Known Issues & Decisions
1. **4GB VRAM limit**: ProtBERT and training run on Colab; inference runs fine on CPU
2. **Label imbalance**: Side-effect labels at 3.13% positive rate — handled with pos_weight in loss
3. **Biomarker edge count (317)**: Small but achieves 90.6% accuracy
4. **XAI attention**: Uses graph-level attention statistics for now; per-edge extraction requires bipartite index mapping (future improvement)
5. **Polypharmacy aggregation**: Max+mean pooling is a simple but effective baseline; a dedicated interaction-aware decoder would be more principled (future improvement)
