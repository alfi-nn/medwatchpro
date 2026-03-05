"""
============================================================
 HGT Model Definition
 Purpose: Heterogeneous Graph Transformer for:
   1) Drug side-effect prediction (multi-label classification)
   2) Drug-biomarker interaction type classification
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGTModel(nn.Module):
    """
    Heterogeneous Graph Transformer for adverse drug reaction prediction.

    Architecture:
      1. Per-type linear projections to a common hidden dimension
      2. N layers of HGTConv for message passing
      3. Task-specific decoder heads

    Memory-optimized for RTX 3050 (4GB VRAM):
      - Uses hidden_dim=128 (instead of 256)
      - 2 HGT layers (instead of 3)
      - Gradient checkpointing support
    """

    def __init__(
        self,
        node_types,
        metadata,
        in_channels_dict,
        hidden_channels=128,
        num_heads=4,
        num_layers=2,
        num_se_classes=100,
        num_bio_classes=3,
        dropout=0.4,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # 1. Per-type input projection
        self.input_projections = nn.ModuleDict()
        for ntype in node_types:
            in_dim = in_channels_dict[ntype]
            self.input_projections[ntype] = Linear(in_dim, hidden_channels)

        # 2. HGT Convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads,
                )
            )
            # Per-type LayerNorm
            norm_dict = nn.ModuleDict()
            for ntype in node_types:
                norm_dict[ntype] = nn.LayerNorm(hidden_channels)
            self.norms.append(norm_dict)

        # 3A. Side-Effect Prediction Head (multi-label, drug nodes only)
        self.se_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_se_classes),
        )

        # 3B. Biomarker Edge Classification Head
        # Input: concatenation of drug + biomarker embeddings
        self.bio_decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_bio_classes),
        )

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass: project features -> HGT layers -> return node embeddings.

        Args:
            x_dict: dict of {node_type: feature_tensor}
            edge_index_dict: dict of {(src, rel, dst): edge_index}

        Returns:
            x_dict: dict of updated node embeddings
        """
        # IMPORTANT: copy to avoid mutating the original input
        x_dict = {k: v.clone() for k, v in x_dict.items()}

        # Project all node types to common hidden dim
        for ntype in x_dict:
            if ntype in self.input_projections:
                x_dict[ntype] = self.input_projections[ntype](x_dict[ntype])

        # HGT message passing
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)

            # Residual + LayerNorm + Dropout
            for ntype in x_dict_new:
                x_dict_new[ntype] = self.norms[i][ntype](
                    x_dict_new[ntype] + x_dict[ntype]  # residual
                )
                x_dict_new[ntype] = F.dropout(
                    x_dict_new[ntype], p=self.dropout, training=self.training
                )
            x_dict = x_dict_new

        return x_dict

    def predict_side_effects(self, drug_embeddings):
        """Predict side effects for drug nodes (multi-label)."""
        return self.se_decoder(drug_embeddings)

    def predict_biomarker_type(self, drug_emb, biomarker_emb, edge_index):
        """
        Predict biomarker interaction type (adverse/efficacy/other).

        Args:
            drug_emb: [N_drugs, hidden] drug node embeddings
            biomarker_emb: [N_bio, hidden] biomarker node embeddings
            edge_index: [2, E] drug-biomarker edge indices
        """
        src_emb = drug_emb[edge_index[0]]
        dst_emb = biomarker_emb[edge_index[1]]
        edge_features = torch.cat([src_emb, dst_emb], dim=-1)
        return self.bio_decoder(edge_features)
