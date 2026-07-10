#!/usr/bin/env python3
"""
Extract edge importance explanations from a trained EXPASS model checkpoint.
Supports GraphSAGE / JK / custom activation and other train-time architecture params.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from batched_explainer import BatchedGNNExplainer
from model import Model


def infer_num_classes_from_state_dict(state_dict: dict) -> int:
    if "final.weight" not in state_dict:
        raise KeyError(
            "Could not find 'final.weight' in checkpoint state_dict. "
            "Pass --num-classes explicitly."
        )
    return int(state_dict["final.weight"].shape[0])


def infer_hidden_dim_from_state_dict(state_dict: dict) -> int:
    # Works with your Model where first conv layer has out_channels == hidden_dim
    # Typical key examples:
    # - conv_layers.0.lin_l.weight (SAGEConv)
    # - conv_layers.0.weight       (GCNConv variants)
    candidate_keys = [
        "conv_layers.0.lin_l.weight",
        "conv_layers.0.lin.weight",
        "conv_layers.0.weight",
    ]
    for k in candidate_keys:
        if k in state_dict:
            return int(state_dict[k].shape[0])

    # fallback: derive from final layer if JK is not cat
    if "final.weight" in state_dict:
        # shape = [nclass, final_in]
        final_in = int(state_dict["final.weight"].shape[1])
        return final_in  # may be hidden_dim or hidden_dim*num_layers (jk cat)
    raise KeyError("Could not infer hidden_dim from checkpoint. Pass --hidden-dim explicitly.")


def export_edge_explanations(
    model_path: str,
    embeddings_path: str,
    adjacency_path: str,
    output_path: str = "edge_explanations.csv",
    arch: str = "graphsage",
    num_layers: int = 3,
    hidden_dim: int = None,
    num_classes: int = None,
    jk_mode: str = "cat",
    activation: str = "elu",
    dropout: float = 0.0,
    explainer_lr: float = 0.01,
    explainer_epochs: int = 100,
):
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)

    print(f"Loading adjacency from {adjacency_path}...")
    with open(adjacency_path, "rb") as f:
        adj_matrix = pickle.load(f)

    edge_index = torch.from_numpy(np.array(adj_matrix.nonzero())).long()
    x = torch.from_numpy(embeddings).float()
    x.requires_grad_(True)

    num_features = x.shape[1]

    print(f"Loading model checkpoint from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")

    if num_classes is None:
        num_classes = infer_num_classes_from_state_dict(state_dict)
        print(f"  Auto-detected num_classes={num_classes}")
    else:
        print(f"  Using num_classes={num_classes} (provided)")

    if hidden_dim is None:
        hidden_dim = infer_hidden_dim_from_state_dict(state_dict)
        # if jk cat, infer_hidden_dim fallback may have returned hidden_dim*num_layers
        if jk_mode == "cat" and "final.weight" in state_dict:
            final_in = int(state_dict["final.weight"].shape[1])
            if final_in % num_layers == 0:
                hidden_dim = final_in // num_layers
        print(f"  Auto-detected hidden_dim={hidden_dim}")
    else:
        print(f"  Using hidden_dim={hidden_dim} (provided)")

    print(
        f"Creating model: arch={arch}, layers={num_layers}, hidden_dim={hidden_dim}, "
        f"jk_mode={jk_mode}, activation={activation}"
    )
    model = Model(
        nfeat=num_features,
        nhid=hidden_dim,
        nclass=num_classes,
        dropout=dropout,  # irrelevant in eval but keep explicit
        num_layers=num_layers,
        gnn_arch=arch,
        node_classification_mode=True,
        jk_mode=jk_mode,
        activation=activation,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {unexpected}")
    model.eval()

    print("Creating explainer...")
    explainer = BatchedGNNExplainer(
        model=model,
        lr=explainer_lr,
        epochs=explainer_epochs,
        return_type="raw",
        log=False,
    )

    print("Generating edge importance scores...")
    _, edge_mask = explainer.explain_graph(
        x=x,
        edge_index=edge_index,
        edge_weight=None,
    )

    print("Creating output table...")
    edges_df = pd.DataFrame(
        {
            "source_node": edge_index[0].cpu().numpy(),
            "target_node": edge_index[1].cpu().numpy(),
            "importance_score": edge_mask.detach().cpu().numpy(),
        }
    )
    edges_df = edges_df.sort_values("importance_score", ascending=False).reset_index(drop=True)
    edges_df["rank"] = np.arange(1, len(edges_df) + 1)

    edges_df.to_csv(output_path, index=False)
    print(f"✓ Saved edge explanations to {output_path}")
    print(f"  Total edges: {len(edges_df)}")
    print("\nTop 10 most important edges:")
    print(edges_df.head(10).to_string(index=False))
    return edges_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model .pth file")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .npy")
    parser.add_argument("--adjacency", required=True, help="Path to adjacency .pkl")
    parser.add_argument("--output", default="edge_explanations.csv", help="Output CSV")

    # Must match train-time model config
    parser.add_argument("--arch", default="graphsage", help="gnn architecture")
    parser.add_argument("--num-layers", type=int, default=3, help="number of GNN layers")
    parser.add_argument("--hidden-dim", type=int, default=None, help="hidden dim (auto if omitted)")
    parser.add_argument("--num-classes", type=int, default=None, help="num output classes (auto if omitted)")
    parser.add_argument("--jk-mode", default="cat", choices=["none", "cat", "max"])
    parser.add_argument("--activation", default="elu")
    parser.add_argument("--dropout", type=float, default=0.0)

    # Explainer config
    parser.add_argument("--explainer-lr", type=float, default=0.01)
    parser.add_argument("--explainer-epochs", type=int, default=100)

    args = parser.parse_args()

    export_edge_explanations(
        model_path=args.model,
        embeddings_path=args.embeddings,
        adjacency_path=args.adjacency,
        output_path=args.output,
        arch=args.arch,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        jk_mode=args.jk_mode,
        activation=args.activation,
        dropout=args.dropout,
        explainer_lr=args.explainer_lr,
        explainer_epochs=args.explainer_epochs,
    )