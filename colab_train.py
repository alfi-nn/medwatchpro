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
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab/server
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
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
    hist = {"loss": [], "se_loss": [], "bio_loss": [], "val_auc": [], "val_f1": [], "bio_acc": [], "bio_f1": []}

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
        hist["se_loss"].append(se_l.item())
        hist["bio_loss"].append(bio_l.item())
        hist["val_auc"].append(va)
        hist["val_f1"].append(vf)
        hist["bio_acc"].append(float(ba))
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

    # ── Generate report figures ──────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING REPORT FIGURES")
    print("=" * 70)
    generate_report_figures(hist, bt, bp, ckpt["epoch"])

    # Download files (uncomment in Colab)
    # files.download("best_hgt_model.pt")
    # files.download("training_history.json")
    # files.download("results.json")
    # files.download("training_curves.png")
    # files.download("biomarker_confusion_matrix.png")

    return model, hist, results


def generate_report_figures(hist, bio_targets, bio_preds, best_epoch):
    """Generate publication-quality figures for the project report."""

    sns.set_theme(style="whitegrid", font_scale=1.1)
    epochs = list(range(1, len(hist["loss"]) + 1))

    # ── Figure 1: Training Curves (Loss + Val AUC) ───────────
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Left axis: Losses
    color_total = "#e63946"
    color_se = "#457b9d"
    color_bio = "#2a9d8f"
    ax1.plot(epochs, hist["loss"], color=color_total, linewidth=2, label="Total Loss", alpha=0.9)
    ax1.plot(epochs, hist["se_loss"], color=color_se, linewidth=1.2, label="SE Loss", linestyle="--", alpha=0.7)
    ax1.plot(epochs, hist["bio_loss"], color=color_bio, linewidth=1.2, label="Bio Loss", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=13, fontweight="bold", color=color_total)
    ax1.tick_params(axis="y", labelcolor=color_total)
    ax1.set_xlim(1, len(epochs))

    # Right axis: Validation AUC
    ax2 = ax1.twinx()
    color_auc = "#4361ee"
    ax2.plot(epochs, hist["val_auc"], color=color_auc, linewidth=2.5, label="Val ROC-AUC", alpha=0.95)
    ax2.set_ylabel("Validation ROC-AUC", fontsize=13, fontweight="bold", color=color_auc)
    ax2.tick_params(axis="y", labelcolor=color_auc)

    # Mark best epoch
    if best_epoch <= len(epochs):
        best_auc = hist["val_auc"][best_epoch - 1]
        ax2.axvline(x=best_epoch, color="#aaa", linestyle=":", linewidth=1.5, alpha=0.7)
        ax2.scatter([best_epoch], [best_auc], color=color_auc, s=120, zorder=5,
                    edgecolors="white", linewidths=2)
        ax2.annotate(f"Best: {best_auc:.4f}\n(Epoch {best_epoch})",
                     xy=(best_epoch, best_auc),
                     xytext=(best_epoch + len(epochs)*0.05, best_auc - 0.02),
                     fontsize=10, fontweight="bold", color=color_auc,
                     arrowprops=dict(arrowstyle="->", color=color_auc, lw=1.5))

    # Early stop marker
    ax2.axvline(x=len(epochs), color="#e63946", linestyle="--", linewidth=1, alpha=0.5)
    ax1.annotate("Early Stop", xy=(len(epochs), hist["loss"][-1]),
                 xytext=(len(epochs) - len(epochs)*0.12, max(hist["loss"]) * 0.85),
                 fontsize=9, color="#e63946", alpha=0.7,
                 arrowprops=dict(arrowstyle="->", color="#e63946", lw=1, alpha=0.5))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10,
               framealpha=0.9, edgecolor="#ddd")

    ax1.set_title("MedWatchPro HGT Training: Loss Convergence & Validation ROC-AUC",
                  fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: training_curves.png")

    # ── Figure 2: Biomarker Confusion Matrix ─────────────────
    cm = confusion_matrix(bio_targets, bio_preds, labels=[0, 1, 2])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    class_names = ["Adverse", "Efficacy", "Other"]

    # Annotation: count + percentage
    annot = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)")
        annot.append(row)

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor="white", square=True,
                cbar_kws={"label": "Count", "shrink": 0.8}, ax=ax)

    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title("Biomarker Interaction Classification\nConfusion Matrix (Test Set)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    fig.savefig("biomarker_confusion_matrix.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: biomarker_confusion_matrix.png")

    # ── Figure 3: Validation Metrics Over Epochs ─────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Side-Effect metrics
    ax1.plot(epochs, hist["val_auc"], color="#4361ee", linewidth=2, label="ROC-AUC")
    ax1.plot(epochs, hist["val_f1"], color="#f72585", linewidth=2, label="F1-Score")
    if best_epoch <= len(epochs):
        ax1.axvline(x=best_epoch, color="#aaa", linestyle=":", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Score", fontweight="bold")
    ax1.set_title("Side-Effect Prediction (Validation)", fontweight="bold")
    ax1.legend(framealpha=0.9)
    ax1.set_xlim(1, len(epochs))
    ax1.set_ylim(0, 1.05)

    # Biomarker metrics
    ax2.plot(epochs, hist["bio_acc"], color="#4cc9f0", linewidth=2, label="Accuracy")
    ax2.plot(epochs, hist["bio_f1"], color="#7209b7", linewidth=2, label="Macro F1")
    if best_epoch <= len(epochs):
        ax2.axvline(x=best_epoch, color="#aaa", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Score", fontweight="bold")
    ax2.set_title("Biomarker Classification (Validation)", fontweight="bold")
    ax2.legend(framealpha=0.9)
    ax2.set_xlim(1, len(epochs))
    ax2.set_ylim(0, 1.05)

    fig.suptitle("MedWatchPro HGT — Validation Metrics Over Training",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("validation_metrics.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: validation_metrics.png")

    print("\n  All 3 report figures generated successfully!")


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model, hist, results = train_hgt(
        graph_path="graph_data.pt",
        epochs=100,
        lr=1e-3,
        patience=20
    )
