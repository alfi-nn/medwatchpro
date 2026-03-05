"""
============================================================
 STEP 5: Train HGT Model
 Purpose: Train the Heterogeneous Graph Transformer for:
   Task A: Side-effect prediction (multi-label, BCEWithLogitsLoss)
   Task B: Biomarker edge classification (CrossEntropyLoss)
============================================================

 Usage (local):
   D:\\medwatchpro\\venv\\Scripts\\python.exe scripts/05_train.py

 Usage (Colab):
   Upload scripts/ and data/processed/ to Colab, then:
   !python scripts/05_train.py --device cuda --epochs 100
============================================================
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import importlib
hgt_module = importlib.import_module("scripts.04_hgt_model")
HGTModel = hgt_module.HGTModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train HGT for ADR prediction")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--se_weight", type=float, default=1.0,
                        help="Loss weight for side-effect task")
    parser.add_argument("--bio_weight", type=float, default=2.0,
                        help="Loss weight for biomarker task (upweighted due to small size)")
    return parser.parse_args()


def setup_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  VRAM: {vram:.1f} GB")
        else:
            device = torch.device("cpu")
            print("  Device: CPU")
    else:
        device = torch.device(device_arg)
    return device


def compute_pos_weight(labels, mask):
    """Compute positive class weights for imbalanced multi-label classification."""
    masked_labels = labels[mask]
    pos_count = masked_labels.sum(dim=0)
    neg_count = mask.sum() - pos_count
    # Avoid division by zero
    pos_weight = neg_count / (pos_count + 1e-6)
    pos_weight = pos_weight.clamp(max=50.0)  # cap to prevent extreme weights
    return pos_weight


def evaluate_se(model, x_dict, edge_index_dict, labels, mask, device):
    """Evaluate side-effect prediction."""
    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        logits = model.predict_side_effects(out["drug"][mask])
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = labels[mask].cpu().numpy()

    # AUC (per-class, then average)
    aucs = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets[:, i]):
            aucs.append(roc_auc_score(targets[:, i], probs[:, i]))
    mean_auc = np.mean(aucs) if aucs else 0.0

    # F1 (threshold = 0.5)
    preds = (probs > 0.5).astype(int)
    f1 = f1_score(targets, preds, average="micro", zero_division=0)

    return mean_auc, f1


def evaluate_bio(model, x_dict, edge_index_dict, bio_edge_index, bio_labels, mask, device):
    """Evaluate biomarker edge classification."""
    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        logits = model.predict_biomarker_type(
            out["drug"], out["biomarker"], bio_edge_index[:, mask]
        )
        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = bio_labels[mask].cpu().numpy()

    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    acc = (preds == targets).mean()
    return acc, f1


def train():
    args = parse_args()

    print("=" * 60)
    print(" MedWatchPro -- Step 5: HGT Training")
    print("=" * 60)

    # ── Setup ────────────────────────────────────────────────
    device = setup_device(args.device)
    data_dir = os.path.join(BASE_DIR, "data", "processed")
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # ── Load graph ───────────────────────────────────────────
    print("\n  Loading graph data...")
    data = torch.load(os.path.join(data_dir, "graph_data.pt"), weights_only=False)
    print(f"  {data}")

    # Move to device
    data = data.to(device)

    # Extract components
    x_dict = {ntype: data[ntype].x for ntype in data.node_types}
    edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}

    se_labels = data["drug"].side_effect_labels
    train_mask = data["drug"].train_mask
    val_mask = data["drug"].val_mask
    test_mask = data["drug"].test_mask

    bio_edge_type = ("drug", "associated_with", "biomarker")
    bio_edge_index = data[bio_edge_type].edge_index
    bio_labels = data[bio_edge_type].edge_label
    bio_train_mask = data[bio_edge_type].train_mask
    bio_val_mask = data[bio_edge_type].val_mask
    bio_test_mask = data[bio_edge_type].test_mask

    # ── Build model ──────────────────────────────────────────
    in_channels = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    metadata = data.metadata()

    model = HGTModel(
        node_types=data.node_types,
        metadata=metadata,
        in_channels_dict=in_channels,
        hidden_channels=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_se_classes=se_labels.shape[1],
        num_bio_classes=3,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {num_params:,}")

    # ── Loss functions ───────────────────────────────────────
    pos_weight = compute_pos_weight(se_labels, train_mask).to(device)
    se_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Class weights for biomarker (adverse=130, efficacy=187, other=0)
    bio_class_counts = torch.bincount(bio_labels[bio_train_mask], minlength=3).float()
    bio_class_weights = 1.0 / (bio_class_counts + 1.0)
    bio_class_weights = bio_class_weights / bio_class_weights.sum() * 3.0
    bio_criterion = nn.CrossEntropyLoss(weight=bio_class_weights.to(device))

    print(f"  SE pos_weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")
    print(f"  Bio class weights: {bio_class_weights.tolist()}")

    # ── Optimizer & Scheduler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ────────────────────────────────────────
    print(f"\n  Training for {args.epochs} epochs (patience={args.patience})")
    print("-" * 60)

    best_val_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "val_f1": [], "bio_val_f1": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        # Forward pass
        out = model(x_dict, edge_index_dict)

        # Task A: Side-effect prediction loss
        se_logits = model.predict_side_effects(out["drug"][train_mask])
        se_loss = se_criterion(se_logits, se_labels[train_mask])

        # Task B: Biomarker classification loss
        bio_logits = model.predict_biomarker_type(
            out["drug"], out["biomarker"], bio_edge_index[:, bio_train_mask]
        )
        bio_loss = bio_criterion(bio_logits, bio_labels[bio_train_mask])

        # Combined loss
        total_loss = args.se_weight * se_loss + args.bio_weight * bio_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Evaluate ─────────────────────────────────────────
        val_auc, val_f1 = evaluate_se(model, x_dict, edge_index_dict,
                                       se_labels, val_mask, device)
        bio_acc, bio_f1 = evaluate_bio(model, x_dict, edge_index_dict,
                                        bio_edge_index, bio_labels, bio_val_mask, device)

        elapsed = time.time() - t0
        history["train_loss"].append(total_loss.item())
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["bio_val_f1"].append(bio_f1)

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {total_loss:.4f} (SE:{se_loss:.4f} Bio:{bio_loss:.4f}) | "
                f"Val AUC: {val_auc:.4f} F1: {val_f1:.4f} | "
                f"Bio Acc: {bio_acc:.2f} F1: {bio_f1:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.1f}s"
            )

        # ── Early stopping ───────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "args": vars(args),
            }, os.path.join(model_dir, "best_hgt_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (best AUC: {best_val_auc:.4f})")
                break

    # ── Final Evaluation on Test Set ─────────────────────────
    print("\n" + "=" * 60)
    print(" TEST SET EVALUATION")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(os.path.join(model_dir, "best_hgt_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded best model from epoch {checkpoint['epoch']}")

    test_auc, test_f1 = evaluate_se(model, x_dict, edge_index_dict,
                                     se_labels, test_mask, device)
    bio_test_acc, bio_test_f1 = evaluate_bio(model, x_dict, edge_index_dict,
                                              bio_edge_index, bio_labels,
                                              bio_test_mask, device)

    print(f"\n  Side-Effect Prediction:")
    print(f"    Test AUC:  {test_auc:.4f}")
    print(f"    Test F1:   {test_f1:.4f}")
    print(f"\n  Biomarker Classification:")
    print(f"    Test Acc:  {bio_test_acc:.2f}")
    print(f"    Test F1:   {bio_test_f1:.4f}")

    # Detailed biomarker classification report
    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        bio_logits = model.predict_biomarker_type(
            out["drug"], out["biomarker"], bio_edge_index[:, bio_test_mask]
        )
        bio_preds = bio_logits.argmax(dim=-1).cpu().numpy()
        bio_targets = bio_labels[bio_test_mask].cpu().numpy()

    print("\n  Biomarker Classification Report:")
    print(classification_report(
        bio_targets, bio_preds,
        target_names=["adverse", "efficacy", "other"],
        zero_division=0
    ))

    # ── Save training history ────────────────────────────────
    history_path = os.path.join(model_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Save results summary ─────────────────────────────────
    results = {
        "test_auc": test_auc,
        "test_f1": test_f1,
        "bio_test_acc": float(bio_test_acc),
        "bio_test_f1": bio_test_f1,
        "best_epoch": checkpoint["epoch"],
        "best_val_auc": best_val_auc,
        "num_params": num_params,
    }
    results_path = os.path.join(model_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Model saved:   {os.path.join(model_dir, 'best_hgt_model.pt')}")
    print(f"  History saved: {history_path}")
    print(f"  Results saved: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()
