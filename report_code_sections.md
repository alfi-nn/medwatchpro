# MedWatchPro — Key Code Sections for Report

> [!TIP]
> Each section below is self-contained and labeled for easy inclusion in your report. Copy the section heading and code block directly.

---

## Section 1: Data Preprocessing — Side-Effect Label Matrix Construction

**File:** [01_prepare_nodes_and_edges.py](file:///d:/medwatchpro/scripts/01_prepare_nodes_and_edges.py#L98-L219)

This section parses the SIDER database (MedDRA side-effect terms) and cross-references them with DrugBank drugs to construct a binary label matrix of shape `[num_drugs × 100]` for multi-label ADR classification.

```python
def prepare_side_effect_nodes_and_labels(drug_id_map):
    # --- Load SIDER side effects ---
    with gzip.open(sider_path, "rt", encoding="utf-8", errors="replace") as f:
        sider_rows = []
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                stitch_flat = parts[0].strip()
                se_name     = parts[5].strip()
                if stitch_flat and se_name:
                    sider_rows.append((stitch_flat, se_name.lower()))

    df_sider = pd.DataFrame(sider_rows, columns=["stitch_id", "side_effect"])

    # --- Select top 100 most common side effects ---
    se_counts = Counter(df_sider["side_effect"])
    top_se = [se for se, _ in se_counts.most_common(100)]

    # --- Build binary label matrix: drugs × side_effects ---
    num_drugs = max(drug_id_map.values()) + 1
    num_se    = len(se_to_idx)
    label_matrix = np.zeros((num_drugs, num_se), dtype=np.float32)

    for _, row in df_filtered.iterrows():
        d_idx = int(row["drug_node_idx"])
        s_idx = int(row["se_node_idx"])
        label_matrix[d_idx, s_idx] = 1.0

    np.save(os.path.join(OUTPUT_DIR, "side_effect_labels.npy"), label_matrix)
```

---

## Section 2: Embedding Generation — ChemBERTa (Drug SMILES)

**File:** [02_generate_embeddings.py](file:///d:/medwatchpro/scripts/02_generate_embeddings.py#L31-L91)

Generates 384-dimensional embeddings for each drug molecule using the CLS token from ChemBERTa-77M-MLM.

```python
def generate_drug_embeddings(batch_size=32):
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    all_embeddings = torch.zeros(num_drugs, embed_dim)

    for i in tqdm(range(0, num_drugs, batch_size), desc="Drug batches"):
        batch_smiles = smiles_list[i:i + batch_size]
        inputs = tokenizer(
            batch_smiles, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS token embedding (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings[i:i + len(batch_smiles)] = cls_embeddings

    torch.save(all_embeddings, out_path)
```

---

## Section 3: Embedding Generation — ProtBERT (Protein Sequences)

**File:** [02_generate_embeddings.py](file:///d:/medwatchpro/scripts/02_generate_embeddings.py#L97-L168)

Generates 1024-dimensional embeddings for protein targets using mean pooling over ProtBERT token outputs. Sequences are space-separated as required by ProtBERT.

```python
def generate_protein_embeddings(batch_size=8):
    model_name = "Rostlab/prot_bert"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name)  # CPU mode
    model.eval()

    for i in tqdm(range(0, num_proteins, batch_size), desc="Protein batches"):
        batch_seqs = sequences[i:i + batch_size]
        # ProtBERT expects space-separated amino acids
        spaced_seqs = [" ".join(list(str(seq).strip())) for seq in batch_seqs]

        inputs = tokenizer(
            spaced_seqs, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling over sequence length
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeds = outputs.last_hidden_state
        mean_embeds = (token_embeds * attention_mask).sum(dim=1) \
                      / attention_mask.sum(dim=1)
        all_embeddings[i:i + len(batch_seqs)] = mean_embeds
```

---

## Section 4: Heterogeneous Graph Construction

**File:** [03_build_graph.py](file:///d:/medwatchpro/scripts/03_build_graph.py#L31-L172)

Assembles a PyG `HeteroData` graph with 4 node types and 5 edge types, including train/val/test masks.

```python
def build_hetero_graph():
    data = HeteroData()

    # ── 1. NODE FEATURES ──────────────────────────────
    data["drug"].x = torch.load("drug_embeddings.pt")       # [N_drugs, 384]
    data["protein"].x = torch.load("protein_embeddings.pt") # [N_prot, 1024]
    data["biomarker"].x = torch.randn(num_biomarkers, 128)  # Trainable
    data["side_effect"].x = torch.randn(num_se, 128)        # Trainable

    # ── 2. EDGE INDICES ───────────────────────────────
    data["drug", "targets", "protein"].edge_index = dti_edges
    data["protein", "targeted_by", "drug"].edge_index = dti_rev
    data["drug", "interacts", "drug"].edge_index = ddi_edges
    data["drug", "causes", "side_effect"].edge_index = dse_edges
    data["drug", "associated_with", "biomarker"].edge_index = bio_edges

    # Biomarker edge labels for classification
    label_map = {"adverse": 0, "efficacy": 1, "other": 2}
    data["drug", "associated_with", "biomarker"].edge_label = bio_labels

    # ── 3. TRAIN/VAL/TEST SPLIT (80/10/10) ────────────
    perm = torch.randperm(num_drugs)
    train_mask[perm[:int(0.8 * num_drugs)]] = True
    val_mask[perm[int(0.8*num_drugs):int(0.9*num_drugs)]] = True
    test_mask[perm[int(0.9 * num_drugs):]] = True

    data["drug"].train_mask = train_mask
    data["drug"].val_mask   = val_mask
    data["drug"].test_mask  = test_mask

    torch.save(data, "graph_data.pt")
```

---

## Section 5: HGT Model Architecture

**File:** [04_hgt_model.py](file:///d:/medwatchpro/scripts/04_hgt_model.py#L16-L147)

The core Heterogeneous Graph Transformer model with per-type input projections, HGTConv layers with residual connections and LayerNorm, and two task-specific decoder heads.

```python
class HGTModel(nn.Module):
    def __init__(self, node_types, metadata, in_channels_dict,
                 hidden_channels=128, num_heads=4, num_layers=2,
                 num_se_classes=100, num_bio_classes=3, dropout=0.4):
        super().__init__()

        # Per-type input projection
        self.input_projections = nn.ModuleDict()
        for ntype in node_types:
            self.input_projections[ntype] = Linear(
                in_channels_dict[ntype], hidden_channels
            )

        # HGT Convolution layers with LayerNorm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(hidden_channels, hidden_channels,
                        metadata=metadata, heads=num_heads)
            )
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_channels) for nt in node_types
            }))

        # Side-Effect decoder (multi-label)
        self.se_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_se_classes),
        )

        # Biomarker edge classifier
        self.bio_decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_bio_classes),
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v.clone() for k, v in x_dict.items()}

        # Project all node types to common hidden dimension
        for ntype in x_dict:
            if ntype in self.input_projections:
                x_dict[ntype] = self.input_projections[ntype](x_dict[ntype])

        # HGT message passing with residual + LayerNorm
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
```

---

## Section 6: Training Loop — Multi-Task Loss & Early Stopping

**File:** [colab_train.py](file:///d:/medwatchpro/colab_train.py#L137-L286)

Multi-task training with class-weighted losses, AdamW optimizer, cosine annealing, gradient clipping, and early stopping.

```python
def train_hgt(graph_path="graph_data.pt", epochs=100, lr=1e-3, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(graph_path, weights_only=False).to(device)

    # Loss functions with class balancing
    pw = compute_pos_weight(se_labels, train_m).to(device)
    se_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    bio_counts = torch.bincount(bio_labels[bio_train_m], minlength=3).float()
    bio_wt = (1.0 / (bio_counts + 1.0))
    bio_wt = (bio_wt / bio_wt.sum() * 3.0).to(device)
    bio_loss_fn = nn.CrossEntropyLoss(weight=bio_wt)

    # Optimizer with cosine annealing
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-6
    )

    # Training loop
    best_auc = 0.0; wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        out = model(x_dict, eid)

        # Multi-task loss: SE prediction + Biomarker classification
        se_l = se_loss_fn(
            model.predict_side_effects(out["drug"][train_m]),
            se_labels[train_m]
        )
        bio_l = bio_loss_fn(
            model.predict_biomarker_type(
                out["drug"], out["biomarker"], bio_ei[:, bio_train_m]
            ),
            bio_labels[bio_train_m]
        )
        loss = se_l + 2.0 * bio_l  # Weighted combination

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # Validation & early stopping
        va, vf = eval_se(model, x_dict, eid, se_labels, val_m)
        if va > best_auc:
            best_auc = va; wait = 0
            torch.save({"state": model.state_dict(), "epoch": ep,
                         "auc": va}, "best_hgt_model.pt")
        else:
            wait += 1
            if wait >= patience:
                break
```

---

## Section 7: Evaluation Metrics — Side-Effect & Biomarker

**File:** [colab_train.py](file:///d:/medwatchpro/colab_train.py#L103-L134)

Per-class ROC-AUC, micro-F1, and macro-F1 computation for multi-label ADR and three-class biomarker classification.

```python
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
    f1 = f1_score(tgt, (probs > 0.5).astype(int),
                  average="micro", zero_division=0)
    return auc, f1


def eval_bio(model, x_dict, eid, bio_ei, bio_labels, mask):
    model.eval()
    with torch.no_grad():
        out = model(x_dict, eid)
        logits = model.predict_biomarker_type(
            out["drug"], out["biomarker"], bio_ei[:, mask]
        )
        preds = logits.argmax(dim=-1).cpu().numpy()
        tgt = bio_labels[mask].cpu().numpy()
    return (preds == tgt).mean(), \
           f1_score(tgt, preds, average="macro", zero_division=0)
```

---

## Section 8: Temporal ADR Processing (FAERS Time-to-Onset)

**File:** [06_temporal_adr.py](file:///d:/medwatchpro/scripts/06_temporal_adr.py#L48-L101)

Computes drug side-effect time-to-onset from FAERS reporting data and categorizes into clinical temporal bins.

```python
# Compute time-to-onset in days
valid["onset_days"] = (valid["event_date"] - valid["start_date"]).dt.days

# Filter reasonable range (0 to 365 days)
valid = valid[(valid["onset_days"] >= 0) & (valid["onset_days"] <= 365)]

# Categorize into temporal bins
def onset_category(days):
    if days <= 1:    return 0  # Acute: 0-24 hours
    elif days <= 7:  return 1  # Early: 1-7 days
    elif days <= 28: return 2  # Delayed: 1-4 weeks
    elif days <= 180:return 3  # Late: 1-6 months
    else:            return 4  # Chronic: 6+ months

valid["onset_cat"] = valid["onset_days"].apply(onset_category)

# Compute median onset per side effect
se_onset = valid.groupby("pt_lower").agg(
    median_days=("onset_days", "median"),
    mean_days=("onset_days", "mean"),
    count=("onset_days", "count"),
    most_common_cat=("onset_cat", lambda x: x.mode().iloc[0]),
).reset_index()
```

---

## Section 9: XAI — Attention Capture for Explainability

**File:** [server.py](file:///d:/medwatchpro/server.py#L36-L53)

Custom HGTConv subclass that intercepts attention weights during message passing for post-hoc explainability.

```python
class HGTConvWithAttn(HGTConv):
    """Subclass of HGTConv that captures attention weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attn_weights = None
        self._attn_index = None

    def message(self, k_j, q_i, v_j, edge_attr, index, ptr, size_i):
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        # ── CAPTURE: store attention weights ──
        self._attn_weights = alpha.detach().mean(dim=-1)  # [E]
        self._attn_index = index.detach()
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)
```

---

## Section 10: Novel Compound Prediction — SMILES Injection into Graph

**File:** [server.py](file:///d:/medwatchpro/server.py#L602-L661)

For novel (unseen) drugs: generates a ChemBERTa embedding, appends it to the graph, injects user-specified protein target edges, and runs a full HGT forward pass for prediction.

```python
# 1. Generate embedding using ChemBERTa
with torch.no_grad():
    inputs = chemberta_tokenizer(smiles, return_tensors="pt",
                                  padding=True, truncation=True)
    outputs = chemberta_model(**inputs)
    drug_emb = outputs.last_hidden_state[:, 0, :]  # [1, 384]

    # 2. Append novel compound to graph
    new_x_dict = {k: v.clone() for k, v in x_dict.items()}
    new_x_dict["drug"] = torch.cat([new_x_dict["drug"], drug_emb], dim=0)
    novel_idx = len(x_dict["drug"])

    # 3. Inject protein target edges (bidirectional)
    new_eid = {k: v.clone() for k, v in eid.items()}
    if req.target_proteins:
        src_edges = torch.full((len(req.target_proteins),),
                                novel_idx, dtype=torch.long)
        dst_edges = torch.tensor(req.target_proteins, dtype=torch.long)
        injected_edges = torch.stack([src_edges, dst_edges], dim=0)
        new_eid[targets_et] = torch.cat(
            [new_eid[targets_et], injected_edges], dim=1
        )
        # Reverse edges for information flow back
        rev_injected = torch.stack([dst_edges, src_edges], dim=0)
        new_eid[rev_targets_et] = torch.cat(
            [new_eid[rev_targets_et], rev_injected], dim=1
        )

    # 4. Full HGT forward pass on augmented graph
    out_dict = model(new_x_dict, new_eid)
    drug_emb_128 = out_dict["drug"][novel_idx].unsqueeze(0)

    # 5. Side-effect prediction
    se_logits = model.predict_side_effects(drug_emb_128)
    se_probs = torch.sigmoid(se_logits).squeeze().numpy()
```

---

> [!NOTE]
> **How to use these in your LaTeX report:** Each section heading maps to a pipeline stage. Use them under methodology subsections (e.g., *§4.1 Data Preprocessing*, *§4.2 Feature Extraction*, *§4.3 Graph Construction*, *§4.4 Model Architecture*, *§4.5 Training*, *§4.6 Temporal Analysis*, *§4.7 Inference & Explainability*). Wrap in `\begin{lstlisting}[language=Python]...\end{lstlisting}` for LaTeX.
