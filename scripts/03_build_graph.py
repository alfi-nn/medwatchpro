"""
============================================================
 STEP 3: Build PyG HeteroData Graph
 Purpose: Combine all node features and edge indices into a
          single torch_geometric.data.HeteroData object,
          ready for HGT training.
============================================================
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_edge_index(filename, data_dir=DATA_DIR):
    """Load a CSV file with 'source' and 'target' columns into a [2, E] tensor."""
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    src = torch.tensor(df["source"].values, dtype=torch.long)
    dst = torch.tensor(df["target"].values, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def build_hetero_graph():
    print("=" * 60)
    print(" MedWatchPro -- Step 3: Build HeteroData Graph")
    print("=" * 60)

    data = HeteroData()

    # ── 1. NODE FEATURES ─────────────────────────────────────

    # Drug nodes
    drug_embeds = torch.load(os.path.join(DATA_DIR, "drug_embeddings.pt"), weights_only=True)
    data["drug"].x = drug_embeds
    print(f"  Drug nodes:    {drug_embeds.shape}")

    # Protein nodes
    prot_embeds = torch.load(os.path.join(DATA_DIR, "protein_embeddings.pt"), weights_only=True)
    data["protein"].x = prot_embeds
    print(f"  Protein nodes: {prot_embeds.shape}")

    # Biomarker nodes (trainable embeddings, initialized randomly)
    bio_df = pd.read_csv(os.path.join(DATA_DIR, "nodes_biomarkers.csv"))
    num_biomarkers = len(bio_df)
    # Use a 128-dim trainable embedding for biomarkers
    bio_embed_dim = 128
    data["biomarker"].x = torch.randn(num_biomarkers, bio_embed_dim)
    print(f"  Biomarker nodes: [{num_biomarkers}, {bio_embed_dim}] (trainable)")

    # Side-effect nodes (trainable embeddings)
    se_df = pd.read_csv(os.path.join(DATA_DIR, "nodes_side_effects.csv"))
    num_se = len(se_df)
    se_embed_dim = 128
    data["side_effect"].x = torch.randn(num_se, se_embed_dim)
    print(f"  SideEffect nodes: [{num_se}, {se_embed_dim}] (trainable)")

    # ── 2. EDGE INDICES ──────────────────────────────────────

    # Drug-Target Interaction (DTI)
    dti_edges = load_edge_index("edges_dti.csv")
    data["drug", "targets", "protein"].edge_index = dti_edges
    print(f"  DTI edges:     {dti_edges.shape[1]}")

    # Reverse DTI (Protein -> Drug)
    dti_rev = load_edge_index("edges_dti_reverse.csv")
    data["protein", "targeted_by", "drug"].edge_index = dti_rev
    print(f"  Rev DTI edges: {dti_rev.shape[1]}")

    # Drug-Drug Interaction (DDI)
    ddi_edges = load_edge_index("edges_ddi.csv")
    data["drug", "interacts", "drug"].edge_index = ddi_edges
    print(f"  DDI edges:     {ddi_edges.shape[1]}")

    # Drug -> SideEffect
    dse_edges = load_edge_index("edges_drug_side_effect.csv")
    data["drug", "causes", "side_effect"].edge_index = dse_edges
    print(f"  Drug->SE edges: {dse_edges.shape[1]}")

    # Drug -> Biomarker (prediction target edges)
    bio_edge_df = pd.read_csv(os.path.join(DATA_DIR, "edges_biomarkers.csv"))
    bio_src = torch.tensor(bio_edge_df["source"].values, dtype=torch.long)
    bio_dst = torch.tensor(bio_edge_df["target"].values, dtype=torch.long)
    data["drug", "associated_with", "biomarker"].edge_index = torch.stack([bio_src, bio_dst], dim=0)
    print(f"  Drug->Bio edges: {len(bio_edge_df)}")

    # Store biomarker edge labels for training
    label_map = {"adverse": 0, "efficacy": 1, "other": 2}
    bio_labels = torch.tensor(
        bio_edge_df["label"].map(label_map).values, dtype=torch.long
    )
    data["drug", "associated_with", "biomarker"].edge_label = bio_labels
    print(f"  Biomarker labels: {dict(bio_edge_df['label'].value_counts())}")

    # ── 3. SIDE-EFFECT LABEL MATRIX ──────────────────────────

    # Load the full label matrix for ADR prediction
    label_path = os.path.join(DATA_DIR, "side_effect_labels.npy")
    if os.path.exists(label_path):
        labels = torch.tensor(np.load(label_path), dtype=torch.float32)
        data["drug"].side_effect_labels = labels
        print(f"  SE label matrix: {labels.shape}")

    # ── 4. TRAIN/VAL/TEST SPLIT ──────────────────────────────

    # Split drugs for ADR prediction (80/10/10)
    num_drugs = data["drug"].x.shape[0]
    perm = torch.randperm(num_drugs)
    train_end = int(0.8 * num_drugs)
    val_end = int(0.9 * num_drugs)

    train_mask = torch.zeros(num_drugs, dtype=torch.bool)
    val_mask = torch.zeros(num_drugs, dtype=torch.bool)
    test_mask = torch.zeros(num_drugs, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    data["drug"].train_mask = train_mask
    data["drug"].val_mask = val_mask
    data["drug"].test_mask = test_mask

    print(f"\n  Train/Val/Test split:")
    print(f"    Train: {train_mask.sum().item()} drugs")
    print(f"    Val:   {val_mask.sum().item()} drugs")
    print(f"    Test:  {test_mask.sum().item()} drugs")

    # Split biomarker edges for edge classification (80/10/10)
    num_bio_edges = bio_labels.shape[0]
    bio_perm = torch.randperm(num_bio_edges)
    bio_train_end = int(0.8 * num_bio_edges)
    bio_val_end = int(0.9 * num_bio_edges)

    bio_train_mask = torch.zeros(num_bio_edges, dtype=torch.bool)
    bio_val_mask = torch.zeros(num_bio_edges, dtype=torch.bool)
    bio_test_mask = torch.zeros(num_bio_edges, dtype=torch.bool)

    bio_train_mask[bio_perm[:bio_train_end]] = True
    bio_val_mask[bio_perm[bio_train_end:bio_val_end]] = True
    bio_test_mask[bio_perm[bio_val_end:]] = True

    data["drug", "associated_with", "biomarker"].train_mask = bio_train_mask
    data["drug", "associated_with", "biomarker"].val_mask = bio_val_mask
    data["drug", "associated_with", "biomarker"].test_mask = bio_test_mask

    print(f"\n  Biomarker edge split:")
    print(f"    Train: {bio_train_mask.sum().item()} edges")
    print(f"    Val:   {bio_val_mask.sum().item()} edges")
    print(f"    Test:  {bio_test_mask.sum().item()} edges")

    # ── 5. SAVE ──────────────────────────────────────────────

    out_path = os.path.join(DATA_DIR, "graph_data.pt")
    torch.save(data, out_path)
    print(f"\n  [OK] HeteroData saved: {out_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f" GRAPH SUMMARY")
    print(f"{'=' * 60}")
    print(data)
    print(f"{'=' * 60}")

    return data


if __name__ == "__main__":
    graph = build_hetero_graph()
