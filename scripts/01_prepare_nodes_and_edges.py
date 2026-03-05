"""
============================================================
 STEP 1: Data Collection & Preprocessing
 Purpose: Parse all raw data sources (DrugBank CSVs, SIDER)
          into clean node and edge CSV files with contiguous
          integer IDs, ready for embedding and graph construction.
============================================================
"""

import os
import gzip
import pandas as pd
import numpy as np
from collections import Counter

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUGBANK_DIR  = os.path.join(BASE_DIR, "drugbank_all_full_database.xml")
RAW_DIR       = BASE_DIR                # drug_names.tsv, meddra_all_se.tsv.gz live here
OUTPUT_DIR    = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")


# ==============================================================
#  1A. DRUG NODES  (from DrugBank drugs_features.csv)
# ==============================================================
def prepare_drug_nodes():
    print("\n[1/6] Preparing Drug nodes...")
    df = pd.read_csv(os.path.join(DRUGBANK_DIR, "drugs_features.csv"))

    # Keep only drugs with a valid SMILES
    df = df.dropna(subset=["smiles"]).copy()
    df = df[["drug_id", "name", "smiles"]].drop_duplicates(subset=["drug_id"])
    df = df.reset_index(drop=True)
    df["node_idx"] = df.index

    df.to_csv(os.path.join(OUTPUT_DIR, "nodes_drugs.csv"), index=False)
    print(f"    [OK] {len(df)} drug nodes saved.")
    return dict(zip(df["drug_id"], df["node_idx"]))


# ==============================================================
#  1B. PROTEIN NODES  (from DrugBank proteins.csv)
# ==============================================================
def prepare_protein_nodes():
    print("\n[2/6] Preparing Protein nodes...")
    df = pd.read_csv(os.path.join(DRUGBANK_DIR, "proteins.csv"))

    # Clean FASTA-formatted sequences
    def clean_seq(s):
        if pd.isna(s):
            return None
        lines = str(s).strip().split("\n")
        if lines[0].startswith(">"):
            return "".join(lines[1:]).strip()
        return s.strip()

    df["clean_sequence"] = df["sequence"].apply(clean_seq)
    df = df.dropna(subset=["clean_sequence"]).copy()
    df = df[["target_id", "name", "clean_sequence"]].drop_duplicates(subset=["target_id"])
    df = df.reset_index(drop=True)
    df["node_idx"] = df.index

    # Truncate sequences to 1000 chars for ProtBERT efficiency
    df["clean_sequence"] = df["clean_sequence"].str[:1000]

    df.to_csv(os.path.join(OUTPUT_DIR, "nodes_proteins.csv"), index=False)
    print(f"    [OK] {len(df)} protein nodes saved.")
    return dict(zip(df["target_id"], df["node_idx"]))


# ==============================================================
#  1C. BIOMARKER NODES  (from DrugBank biomarkers.csv)
# ==============================================================
def prepare_biomarker_nodes():
    print("\n[3/6] Preparing Biomarker nodes...")
    df = pd.read_csv(os.path.join(DRUGBANK_DIR, "biomarkers.csv"))

    df["gene_symbol"]     = df["gene_symbol"].fillna("UNKNOWN")
    df["defining_change"] = df["defining_change"].fillna("UNKNOWN")
    df["biomarker_key"]   = df["gene_symbol"] + "||" + df["defining_change"]

    unique = df[["biomarker_key", "gene_symbol", "defining_change",
                  "protein_name", "uniprot_id"]].drop_duplicates(subset=["biomarker_key"])
    unique = unique.reset_index(drop=True)
    unique["node_idx"] = unique.index

    unique.to_csv(os.path.join(OUTPUT_DIR, "nodes_biomarkers.csv"), index=False)
    print(f"    [OK] {len(unique)} biomarker nodes saved.")
    return dict(zip(unique["biomarker_key"], unique["node_idx"])), df


# ==============================================================
#  1D. SIDE-EFFECT NODES  (from SIDER meddra_all_se.tsv.gz)
# ==============================================================
def prepare_side_effect_nodes_and_labels(drug_id_map):
    """
    Parse the SIDER database to extract:
      - Side-effect nodes
      - (Drug, causes, SideEffect) edges / label matrix

    SIDER file format (TSV, no header):
      col0: STITCH flat ID   e.g. CID100000085
      col1: STITCH stereo ID e.g. CID000000085
      col2: UMLS concept CUI (label)
      col3: MedDRA concept type
      col4: UMLS concept CUI (for MedDRA)
      col5: side effect name

    We also use drug_names.tsv to map STITCH CIDs -> drug names,
    which we can then cross-reference with our DrugBank drug IDs.
    """
    print("\n[4/6] Preparing Side-Effect nodes from SIDER...")

    sider_path = os.path.join(RAW_DIR, "meddra_all_se.tsv.gz")
    drug_names_path = os.path.join(RAW_DIR, "drug_names.tsv")

    if not os.path.exists(sider_path):
        print("    [WARN] meddra_all_se.tsv.gz not found. Skipping side-effect extraction.")
        return {}, None

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
    df_sider = df_sider.drop_duplicates()
    print(f"    Loaded {len(df_sider)} unique (drug, side_effect) pairs from SIDER.")

    # --- Load drug names mapping (STITCH CID -> name) ---
    if os.path.exists(drug_names_path):
        df_names = pd.read_csv(drug_names_path, sep="\t", header=None,
                                names=["stitch_id", "drug_name"])
        stitch_to_name = dict(zip(df_names["stitch_id"], df_names["drug_name"].str.lower()))
    else:
        print("    [WARN] drug_names.tsv not found. Cannot map SIDER to DrugBank.")
        return {}, None

    # --- Map SIDER STITCH IDs to our DrugBank drug IDs via name matching ---
    # Load our drug node names
    drugs_df = pd.read_csv(os.path.join(OUTPUT_DIR, "nodes_drugs.csv"))
    drugbank_name_to_id = {}
    for _, row in drugs_df.iterrows():
        name = str(row["name"]).lower().strip()
        drugbank_name_to_id[name] = row["drug_id"]
        # Also try the first word (common name)
        first_word = name.split()[0] if name else ""
        if first_word and first_word not in drugbank_name_to_id:
            drugbank_name_to_id[first_word] = row["drug_id"]

    # Build stitch_id -> drugbank_drug_id mapping
    stitch_to_drugbank = {}
    for stitch_id, drug_name in stitch_to_name.items():
        drug_name_lower = drug_name.lower().strip()
        if drug_name_lower in drugbank_name_to_id:
            stitch_to_drugbank[stitch_id] = drugbank_name_to_id[drug_name_lower]

    print(f"    Mapped {len(stitch_to_drugbank)} SIDER drugs to DrugBank IDs.")

    # --- Filter SIDER to only matched drugs ---
    df_sider["drug_id"] = df_sider["stitch_id"].map(stitch_to_drugbank)
    df_sider = df_sider.dropna(subset=["drug_id"])
    df_sider = df_sider[df_sider["drug_id"].isin(drug_id_map)]

    if len(df_sider) == 0:
        print("    [WARN] No SIDER drugs matched to DrugBank. Side effects skipped.")
        return {}, None

    # --- Select top N most common side effects ---
    se_counts = Counter(df_sider["side_effect"])
    top_se = [se for se, _ in se_counts.most_common(100)]

    # Create side-effect nodes
    se_df = pd.DataFrame({"side_effect": top_se})
    se_df["node_idx"] = se_df.index
    se_df.to_csv(os.path.join(OUTPUT_DIR, "nodes_side_effects.csv"), index=False)
    se_to_idx = dict(zip(se_df["side_effect"], se_df["node_idx"]))
    print(f"    [OK] {len(se_df)} side-effect nodes saved.")

    # --- Build binary label matrix: drugs × side_effects ---
    df_filtered = df_sider[df_sider["side_effect"].isin(se_to_idx)]
    df_filtered = df_filtered.copy()
    df_filtered["drug_node_idx"] = df_filtered["drug_id"].map(drug_id_map)
    df_filtered["se_node_idx"]   = df_filtered["side_effect"].map(se_to_idx)

    num_drugs = max(drug_id_map.values()) + 1
    num_se    = len(se_to_idx)
    label_matrix = np.zeros((num_drugs, num_se), dtype=np.float32)

    for _, row in df_filtered.iterrows():
        d_idx = int(row["drug_node_idx"])
        s_idx = int(row["se_node_idx"])
        label_matrix[d_idx, s_idx] = 1.0

    # Save label matrix
    np.save(os.path.join(OUTPUT_DIR, "side_effect_labels.npy"), label_matrix)

    # Save edges (Drug -> SideEffect)
    edges = df_filtered[["drug_node_idx", "se_node_idx"]].drop_duplicates()
    edges.columns = ["source", "target"]
    edges.to_csv(os.path.join(OUTPUT_DIR, "edges_drug_side_effect.csv"), index=False)
    print(f"    [OK] {len(edges)} (Drug -> SideEffect) edges saved.")

    # Report balance
    pos_rates = label_matrix.mean(axis=0)
    print(f"    Label matrix shape: {label_matrix.shape}")
    print(f"    Mean positive rate: {pos_rates.mean():.2%}")
    print(f"    Min / Max positive rate: {pos_rates.min():.2%} / {pos_rates.max():.2%}")

    return se_to_idx, label_matrix


# ==============================================================
#  1E. EDGES: Drug-Target Interaction (DTI)
# ==============================================================
def prepare_dti_edges(drug_map, protein_map):
    """
    dti.csv uses DrugBank internal target IDs (BE0000048),
    proteins.csv uses UniProt IDs (P00734).
    We join them by matching target_name (dti) -> name (proteins).
    """
    print("\n[5/6] Preparing DTI edges...")
    df_dti = pd.read_csv(os.path.join(DRUGBANK_DIR, "dti.csv"))
    df_prot = pd.read_csv(os.path.join(OUTPUT_DIR, "nodes_proteins.csv"))

    # Build protein name -> node_idx lookup
    prot_name_to_idx = {}
    for _, row in df_prot.iterrows():
        name = str(row["name"]).strip().lower()
        prot_name_to_idx[name] = int(row["node_idx"])

    # Map DTI rows
    df_dti = df_dti[df_dti["drug_id"].isin(drug_map)].copy()
    df_dti["target_name_lower"] = df_dti["target_name"].str.strip().str.lower()
    df_dti["source"] = df_dti["drug_id"].map(drug_map)
    df_dti["target"] = df_dti["target_name_lower"].map(prot_name_to_idx)

    # Keep only matched
    df_matched = df_dti.dropna(subset=["target"]).copy()
    df_matched["target"] = df_matched["target"].astype(int)

    edges = df_matched[["source", "target"]].drop_duplicates()
    edges.to_csv(os.path.join(OUTPUT_DIR, "edges_dti.csv"), index=False)
    print(f"    [OK] {len(edges)} DTI edges saved (matched by name).")
    print(f"    ({len(df_dti) - len(df_matched)} DTI rows unmatched)")

    # Also save reverse edges (protein -> drug)
    reverse = edges.rename(columns={"source": "target", "target": "source"})
    reverse.to_csv(os.path.join(OUTPUT_DIR, "edges_dti_reverse.csv"), index=False)
    print(f"    [OK] {len(reverse)} reverse DTI edges saved.")
    return edges


# ==============================================================
#  1F. EDGES: Drug-Drug Interaction (DDI)
# ==============================================================
def prepare_ddi_edges(drug_map):
    print("\n[6/6] Preparing DDI edges...")
    ddi_path = os.path.join(DRUGBANK_DIR, "ddi.csv")
    if not os.path.exists(ddi_path):
        print("    [WARN] ddi.csv not found. Skipping DDI edges.")
        return None

    # DDI file is very large (~415 MB), read in chunks
    chunks = pd.read_csv(ddi_path, chunksize=500_000)
    all_edges = []
    for chunk in chunks:
        chunk = chunk[chunk["drug_id_1"].isin(drug_map) & chunk["drug_id_2"].isin(drug_map)]
        chunk = chunk.copy()
        chunk["source"] = chunk["drug_id_1"].map(drug_map)
        chunk["target"] = chunk["drug_id_2"].map(drug_map)
        all_edges.append(chunk[["source", "target"]])

    edges = pd.concat(all_edges).drop_duplicates()
    edges.to_csv(os.path.join(OUTPUT_DIR, "edges_ddi.csv"), index=False)
    print(f"    [OK] {len(edges)} DDI edges saved.")
    return edges


# ==============================================================
#  1G. EDGES: Drug-Biomarker (target for edge classification)
# ==============================================================
def prepare_biomarker_edges(drug_map, biomarker_map, biomarkers_df):
    print("\n[BONUS] Preparing Drug-Biomarker edges (prediction target)...")
    df = biomarkers_df.copy()
    df["biomarker_key"] = df["gene_symbol"].fillna("UNKNOWN") + "||" + df["defining_change"].fillna("UNKNOWN")

    df = df[df["drug_id"].isin(drug_map)].copy()
    df["source"] = df["drug_id"].map(drug_map)
    df["target"] = df["biomarker_key"].map(biomarker_map)

    # Classify biomarker type into label
    def classify_label(row):
        btype = str(row.get("biomarker_type", "")).lower()
        desc  = str(row.get("description", "")).lower()
        if "adverse" in btype or "adverse" in desc or "toxicity" in desc or "risk" in desc:
            return "adverse"
        elif "effect" in btype or "efficacy" in desc or "response" in desc or "reduction" in desc:
            return "efficacy"
        else:
            return "other"

    df["label"] = df.apply(classify_label, axis=1)

    edges = df[["source", "target", "biomarker_type", "label", "description"]].copy()
    edges.to_csv(os.path.join(OUTPUT_DIR, "edges_biomarkers.csv"), index=False)

    # Print label distribution
    print(f"    [OK] {len(edges)} biomarker edges saved.")
    print(f"    Label distribution:")
    for label, count in edges["label"].value_counts().items():
        print(f"      {label}: {count}")

    return edges


# ==============================================================
#  MAIN
# ==============================================================
def main():
    print("=" * 60)
    print(" MedWatchPro — Step 1: Data Collection & Preprocessing")
    print("=" * 60)

    # Nodes
    drug_map    = prepare_drug_nodes()
    protein_map = prepare_protein_nodes()
    bio_map, bio_df = prepare_biomarker_nodes()
    se_map, labels  = prepare_side_effect_nodes_and_labels(drug_map)

    # Edges
    prepare_dti_edges(drug_map, protein_map)
    prepare_ddi_edges(drug_map)
    prepare_biomarker_edges(drug_map, bio_map, bio_df)

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Drug nodes:        {len(drug_map)}")
    print(f"  Protein nodes:     {len(protein_map)}")
    print(f"  Biomarker nodes:   {len(bio_map)}")
    print(f"  Side-effect nodes: {len(se_map)}")
    if labels is not None:
        print(f"  Label matrix:      {labels.shape}")
    print(f"\n  Output directory:  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
