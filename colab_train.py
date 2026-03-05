# MedWatchPro - Google Colab Training Notebook
# =============================================
# This script is designed to be copy-pasted into Google Colab cells.
# Upload data/processed/graph_data.pt from your local machine before running.
#
# Colab Setup:
# 1. Runtime -> Change runtime type -> T4 GPU
# 2. Upload graph_data.pt to Colab
# 3. Run each cell in order

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
# !pip install torch-geometric transformers -q

# ============================================================
# CELL 2: Model Definition + Training
# ============================================================

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score, f1_score, classification_report
# from google.colab import files  # Uncomment in Colab

# ── HGT Model ───────────────────────────────────────────────
class HGTModel(nn.Module):
    def __init__(self, node_types, metadata, in_channels_dict,
                 hidden_channels=128, num_heads=4, num_layers=2,
                 num_se_classes=100, num_bio_classes=3, dropout=0.4):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Per-type input projection
        self.input_projections = nn.ModuleDict()
        for ntype in node_types:
            self.input_projections[ntype] = Linear(in_channels_dict[ntype], hidden_channels)

        # HGT layers with LayerNorm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels,
                                       metadata=metadata, heads=num_heads))
            norm_dict = nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
            self.norms.append(norm_dict)

        # Side-Effect decoder (multi-label)
        self.se_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_se_classes),
        )

        # Biomarker edge classifier
        self.bio_decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_bio_classes),
        )

    def forward(self, x_dict, edge_index_dict):
        # IMPORTANT: copy to avoid mutating the original input
        x_dict = {k: v.clone() for k, v in x_dict.items()}

        for ntype in x_dict:
            if ntype in self.input_projections:
                x_dict[ntype] = self.input_projections[ntype](x_dict[ntype])

        for i, conv in enumerate(self.convs):
            x_new = conv(x_dict, edge_index_dict)
            for nt in x_new:
                x_new[nt] = self.norms[i][nt](x_new[nt] + x_dict[nt])
                x_new[nt] = F.dropout(x_new[nt], p=0.4, training=self.training)
            x_dict = x_new
        return x_dict

    def predict_side_effects(self, drug_emb):
        return self.se_decoder(drug_emb)

    def predict_biomarker_type(self, drug_emb, bio_emb, edge_index):
        src = drug_emb[edge_index[0]]
        dst = bio_emb[edge_index[1]]
        return self.bio_decoder(torch.cat([src, dst], dim=-1))


# ── Helpers ──────────────────────────────────────────────────
def compute_pos_weight(labels, mask):
    m = labels[mask]
    pos = m.sum(dim=0)
    neg = mask.sum() - pos
    pw = neg / (pos + 1e-6)
    return pw.clamp(max=50.0)

def eval_se(model, x_dict, eid, labels, mask):
    model.eval()
    with torch.no_grad():
        out = model(x_dict, eid)
        logits = model.predict_side_effects(out["drug"][mask])
        probs = torch.sigmoid(logits).cpu().numpy()
        tgt = labels[mask].cpu().numpy()
    aucs = []
    for i in range(tgt.shape[1]):
        if 0 < tgt[:, i].sum() < len(tgt):
            aucs.append(roc_auc_score(tgt[:, i], probs[:, i]))
    auc = np.mean(aucs) if aucs else 0.0
    f1 = f1_score(tgt, (probs > 0.5).astype(int), average="micro", zero_division=0)
    return auc, f1

def eval_bio(model, x_dict, eid, bio_ei, bio_labels, mask):
    model.eval()
    with torch.no_grad():
        out = model(x_dict, eid)
        logits = model.predict_biomarker_type(out["drug"], out["biomarker"], bio_ei[:, mask])
        preds = logits.argmax(dim=-1).cpu().numpy()
        tgt = bio_labels[mask].cpu().numpy()
    return (preds == tgt).mean(), f1_score(tgt, preds, average="macro", zero_division=0)


# ── Main Training ────────────────────────────────────────────
def train_hgt(graph_path="graph_data.pt", epochs=100, lr=1e-3, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # Load graph
    print("\nLoading graph...")
    data = torch.load(graph_path, weights_only=False).to(device)
    print(data)

    x_dict = {nt: data[nt].x for nt in data.node_types}
    eid = {et: data[et].edge_index for et in data.edge_types}

    se_labels = data["drug"].side_effect_labels
    train_m = data["drug"].train_mask
    val_m = data["drug"].val_mask
    test_m = data["drug"].test_mask

    bio_et = ("drug", "associated_with", "biomarker")
    bio_ei = data[bio_et].edge_index
    bio_labels = data[bio_et].edge_label
    bio_train_m = data[bio_et].train_mask
    bio_val_m = data[bio_et].val_mask
    bio_test_m = data[bio_et].test_mask

    # Model
    in_ch = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = HGTModel(
        node_types=data.node_types, metadata=data.metadata(),
        in_channels_dict=in_ch, hidden_channels=128, num_heads=4,
        num_layers=2, num_se_classes=se_labels.shape[1],
        num_bio_classes=3, dropout=0.4
    ).to(device)
    print(f"\nParams: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    pw = compute_pos_weight(se_labels, train_m).to(device)
    se_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    bio_counts = torch.bincount(bio_labels[bio_train_m], minlength=3).float()
    bio_wt = (1.0 / (bio_counts + 1.0))
    bio_wt = (bio_wt / bio_wt.sum() * 3.0).to(device)
    bio_loss_fn = nn.CrossEntropyLoss(weight=bio_wt)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    # Training loop
    best_auc = 0.0
    wait = 0
    hist = {"loss": [], "val_auc": [], "val_f1": [], "bio_f1": []}

    print(f"\nTraining {epochs} epochs (patience={patience})")
    print("-" * 70)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        out = model(x_dict, eid)
        se_l = se_loss_fn(model.predict_side_effects(out["drug"][train_m]), se_labels[train_m])
        bio_l = bio_loss_fn(model.predict_biomarker_type(
            out["drug"], out["biomarker"], bio_ei[:, bio_train_m]
        ), bio_labels[bio_train_m])
        loss = se_l + 2.0 * bio_l

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        va, vf = eval_se(model, x_dict, eid, se_labels, val_m)
        ba, bf = eval_bio(model, x_dict, eid, bio_ei, bio_labels, bio_val_m)

        hist["loss"].append(loss.item())
        hist["val_auc"].append(va)
        hist["val_f1"].append(vf)
        hist["bio_f1"].append(bf)

        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | Loss {loss:.4f} (SE:{se_l:.4f} Bio:{bio_l:.4f}) | "
                  f"AUC:{va:.4f} F1:{vf:.4f} | BioAcc:{ba:.2f} BioF1:{bf:.4f} | "
                  f"{time.time()-t0:.1f}s")

        if va > best_auc:
            best_auc = va
            wait = 0
            torch.save({"state": model.state_dict(), "epoch": ep, "auc": va},
                       "best_hgt_model.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"\n  Early stop at epoch {ep} (best AUC: {best_auc:.4f})")
                break

    # ── Test ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    ckpt = torch.load("best_hgt_model.pt", weights_only=False)
    model.load_state_dict(ckpt["state"])
    print(f"  Best model from epoch {ckpt['epoch']}")

    ta, tf = eval_se(model, x_dict, eid, se_labels, test_m)
    bta, btf = eval_bio(model, x_dict, eid, bio_ei, bio_labels, bio_test_m)

    print(f"\n  Side-Effect:  AUC={ta:.4f}  F1={tf:.4f}")
    print(f"  Biomarker:    Acc={bta:.2f}  F1={btf:.4f}")

    # Detailed report
    model.eval()
    with torch.no_grad():
        out = model(x_dict, eid)
        bl = model.predict_biomarker_type(out["drug"], out["biomarker"], bio_ei[:, bio_test_m])
        bp = bl.argmax(-1).cpu().numpy()
        bt = bio_labels[bio_test_m].cpu().numpy()
    print("\n" + classification_report(bt, bp, labels=[0, 1, 2],
          target_names=["adverse", "efficacy", "other"], zero_division=0))

    # Save history + results
    with open("training_history.json", "w") as f:
        json.dump(hist, f, indent=2)
    results = {"test_auc": ta, "test_f1": tf, "bio_acc": float(bta),
               "bio_f1": btf, "best_epoch": ckpt["epoch"]}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: best_hgt_model.pt, training_history.json, results.json")

    # Download files (uncomment in Colab)
    # files.download("best_hgt_model.pt")
    # files.download("training_history.json")
    # files.download("results.json")

    return model, hist, results


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model, hist, results = train_hgt(
        graph_path="graph_data.pt",
        epochs=100,
        lr=1e-3,
        patience=20
    )
