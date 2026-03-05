"""
============================================================
 STEP 2: Generate Embeddings
 Purpose: Use ChemBERTa for drug SMILES embeddings and
          ProtBERT for protein sequence embeddings.
          Saves .pt files to data/processed/
============================================================
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel

# --- Paths ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ==============================================================
#  2A. DRUG EMBEDDINGS (ChemBERTa)
# ==============================================================
def generate_drug_embeddings(batch_size=32):
    print("\n" + "=" * 60)
    print(" Generating Drug Embeddings (ChemBERTa)")
    print("=" * 60)

    model_name = "DeepChem/ChemBERTa-77M-MLM"
    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    # Load drug nodes
    df = pd.read_csv(os.path.join(DATA_DIR, "nodes_drugs.csv"))
    smiles_list = df["smiles"].tolist()
    num_drugs = len(smiles_list)
    print(f"  Drugs to embed: {num_drugs}")

    # Get embedding dimension from a test forward pass
    test_input = tokenizer("C", return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        test_out = model(**test_input)
    embed_dim = test_out.last_hidden_state.shape[-1]
    print(f"  Embedding dimension: {embed_dim}")

    # Generate embeddings in batches
    all_embeddings = torch.zeros(num_drugs, embed_dim)

    for i in tqdm(range(0, num_drugs, batch_size), desc="  Drug batches"):
        batch_smiles = smiles_list[i:i + batch_size]

        # Some SMILES may be malformed; handle gracefully
        try:
            inputs = tokenizer(
                batch_smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)

            # CLS token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            all_embeddings[i:i + len(batch_smiles)] = cls_embeddings
        except Exception as e:
            print(f"  [WARN] Batch {i} failed: {e}. Using zero vectors.")
            # Leave as zeros (will be handled during training)

    # Save
    out_path = os.path.join(DATA_DIR, "drug_embeddings.pt")
    torch.save(all_embeddings, out_path)
    print(f"  [OK] Drug embeddings saved: {all_embeddings.shape}")
    print(f"       File: {out_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return all_embeddings


# ==============================================================
#  2B. PROTEIN EMBEDDINGS (ProtBERT)
# ==============================================================
def generate_protein_embeddings(batch_size=8):
    """
    ProtBERT requires spaced amino acid sequences.
    ProtBERT (~1.7GB) is too large for 4GB VRAM alongside other allocations,
    so we run it on CPU. This is slower but reliable.
    """
    print("\n" + "=" * 60)
    print(" Generating Protein Embeddings (ProtBERT) [CPU mode]")
    print("=" * 60)

    # Free GPU memory from previous step
    torch.cuda.empty_cache()

    model_name = "Rostlab/prot_bert"
    print(f"  Loading model: {model_name}")
    print("  NOTE: Running on CPU (model too large for 4GB VRAM)")
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name)  # stays on CPU
    model.eval()

    # Load protein nodes
    df = pd.read_csv(os.path.join(DATA_DIR, "nodes_proteins.csv"))
    sequences = df["clean_sequence"].tolist()
    num_proteins = len(sequences)
    print(f"  Proteins to embed: {num_proteins}")

    # Get embedding dimension
    test_seq = "M E T"
    test_input = tokenizer(test_seq, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        test_out = model(**test_input)
    embed_dim = test_out.last_hidden_state.shape[-1]
    print(f"  Embedding dimension: {embed_dim}")

    # Generate embeddings in batches
    all_embeddings = torch.zeros(num_proteins, embed_dim)

    for i in tqdm(range(0, num_proteins, batch_size), desc="  Protein batches"):
        batch_seqs = sequences[i:i + batch_size]

        # ProtBERT expects space-separated amino acids
        spaced_seqs = [" ".join(list(str(seq).strip())) for seq in batch_seqs]

        try:
            inputs = tokenizer(
                spaced_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512   # truncate long sequences
            )

            with torch.no_grad():
                outputs = model(**inputs)

            # Mean pooling over sequence length
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeds = outputs.last_hidden_state
            mean_embeds = (token_embeds * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            all_embeddings[i:i + len(batch_seqs)] = mean_embeds

        except Exception as e:
            print(f"  [WARN] Batch {i} failed: {e}. Using zero vectors.")

    # Save
    out_path = os.path.join(DATA_DIR, "protein_embeddings.pt")
    torch.save(all_embeddings, out_path)
    print(f"  [OK] Protein embeddings saved: {all_embeddings.shape}")
    print(f"       File: {out_path}")

    del model
    return all_embeddings


# ==============================================================
#  MAIN
# ==============================================================
def main():
    print("=" * 60)
    print(" MedWatchPro -- Step 2: Embedding Generation")
    print("=" * 60)

    drug_path = os.path.join(DATA_DIR, "drug_embeddings.pt")
    prot_path = os.path.join(DATA_DIR, "protein_embeddings.pt")

    if os.path.exists(drug_path):
        print(f"\n  [SKIP] Drug embeddings already exist: {drug_path}")
        drug_embeds = torch.load(drug_path, weights_only=True)
    else:
        drug_embeds = generate_drug_embeddings(batch_size=32)

    if os.path.exists(prot_path):
        print(f"\n  [SKIP] Protein embeddings already exist: {prot_path}")
        protein_embeds = torch.load(prot_path, weights_only=True)
    else:
        protein_embeds = generate_protein_embeddings(batch_size=4)

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Drug embeddings:    {drug_embeds.shape}")
    print(f"  Protein embeddings: {protein_embeds.shape}")
    print(f"  Output directory:   {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
