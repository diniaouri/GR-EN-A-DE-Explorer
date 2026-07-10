#!/usr/bin/env python3


import argparse
import copy
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl
import glob
import time
import shutil
from typing import Optional

from preprocessing import (
    ToxigenDataset,
    LGBTEnDataset,
    MigrantsEnDataset,
    MultilingualENCorpusGermanDataset,
    MultilingualENCorpusFrenchDataset,
    MultilingualENCorpusCypriotDataset,
    MultilingualENCorpusSloveneDataset,
)
from model import GCN, GCL
from graph_learners import FGP_learner, ATT_learner, GNN_learner, MLP_learner

from utils import (
    save_loss_plot, ExperimentParameters, accuracy, get_feat_mask, symmetrize, normalize,
    split_batch, torch_sparse_eye, torch_sparse_to_dgl_graph, dgl_graph_to_torch_sparse
)

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity  # added for attribute-based edge creation


def sanitize_filename_part(s):
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def build_run_suffix(args, max_parts=4):
    suffix = []
    suffix.append(f"exp{args.exp_nb}")
    if getattr(args, "context_columns", None):
        safe_columns = [sanitize_filename_part(col) for col in args.context_columns]
        suffix.append("ctxcols_" + "_".join(sorted(safe_columns)))
    included = 3
    for arg in vars(args):
        if arg not in ["exp_nb", "context_columns", "embeddings_path"] and getattr(args, arg) is not None:
            if included < max_parts:
                suffix.append(f"{sanitize_filename_part(arg)}_{sanitize_filename_part(getattr(args, arg))}")
                included += 1
    return "__".join(suffix)


def make_context_adjacency(context_data, context_columns):
    """Build adjacency matrix based on shared context attributes.
    
    Creates a graph where nodes (posts/documents) are connected if they share
    at least one attribute value in the specified context columns. This captures
    semantic relationships between posts based on narrative elements.
   
    """
    if not isinstance(context_data, pd.DataFrame):
        raise TypeError("context_data must be a pandas DataFrame")
    N = context_data.shape[0]
    adj_context = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        row_i = context_data.iloc[i]
        for j in range(N):
            any_shared = False
            for col in context_columns:
                # Early break when first match found - optimization
                if row_i[col] == context_data.iloc[j][col]:
                    any_shared = True
                    break
            adj_context[i, j] = 1.0 if any_shared else 0.0
    # Remove self-loops
    np.fill_diagonal(adj_context, 0)
    return adj_context


def encode_metadata(df, columns):
    parts = []
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not in dataset DataFrame.")
        ser = df[col].fillna("___MISSING___")
        if pd.api.types.is_numeric_dtype(ser):
            # Keep numeric columns as-is
            parts.append(ser.values.reshape(-1, 1).astype(np.float32))
        else:
            # One-hot encode categorical columns
            dummies = pd.get_dummies(ser, prefix=col)
            parts.append(dummies.values.astype(np.float32))
    if not parts:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(parts, axis=1)


# -------------------------
# Attribute co-membership edge builder (adapted from colleague code)
# -------------------------
def add_attribute_co_membership_edges_np(df_attrs: pd.DataFrame, max_edges_per_node: int = 5, embeddings: Optional[np.ndarray] = None) -> np.ndarray:
    """Build adjacency matrix based on shared attribute co-membership.
    
    This function creates graph edges between nodes (posts/documents) that share
    the same attribute values, which helps capture narrative structure. 
    """
    n = df_attrs.shape[0]
    # Track all edges to avoid duplicates
    existing = set()
    
    # Process each attribute column independently
    for col in df_attrs.columns:
        vals = df_attrs[col].fillna("").astype(str).tolist()
        
        # Group nodes into buckets by attribute value
        buckets = {}
        for i, v in enumerate(vals):
            buckets.setdefault(v, []).append(i)
        
        # Process each bucket (nodes sharing the same attribute value)
        for members in buckets.values():
            m = len(members)
            if m <= 1:
                # Skip buckets with only one member
                continue
                
            # EMBEDDING-BASED EDGE SELECTION (for larger buckets with embeddings)
            if embeddings is not None and m > max_edges_per_node:
                mem_arr = np.array(members)
                emb_sub = embeddings[mem_arr]  # Extract embeddings for bucket members
                
                if emb_sub.shape[0] < 2:
                    continue
                    
                # Compute pairwise cosine similarity within bucket
                sim = cosine_similarity(emb_sub)
                
                # For each node, find its top-k most similar neighbors in the bucket
                for idx_i, node in enumerate(mem_arr):
                    sims = sim[idx_i].copy()
                    sims[idx_i] = -1  # Exclude self-similarity
                    
                    # Select top-k most similar nodes (highest cosine similarity)
                    topk = sims.argsort()[-max_edges_per_node:]
                    for tidx in topk:
                        nbr = int(mem_arr[tidx])
                        # Add bidirectional edges (undirected graph)
                        existing.add((int(node), int(nbr)))
                        existing.add((int(nbr), int(node)))
            else:
                # RANDOM SAMPLING (for small buckets or when no embeddings available)
                # Simply connect each node to random neighbors within the bucket
                for node in members:
                    choices = [m for m in members if m != node]
                    random.shuffle(choices)
                    # Add edges to first max_edges_per_node neighbors
                    for nbr in choices[:max_edges_per_node]:
                        existing.add((int(node), int(nbr)))
                        existing.add((int(nbr), int(node)))
    
    # Convert edge set to dense adjacency matrix
    adj = np.zeros((n, n), dtype=np.float32)
    if existing:
        edge_index = np.array(list(existing), dtype=np.int64).T
        adj[edge_index[0], edge_index[1]] = 1.0
    
    # Ensure no self-loops (diagonal = 0)
    np.fill_diagonal(adj, 0.0)
    return adj


# -------------------------
# Small helper functions to run the attribute (Max) x K sweep inside main.py
# -------------------------
def get_dataset_class(exp_nb):
    mapping = {
        1: ToxigenDataset,
        2: LGBTEnDataset,
        3: MigrantsEnDataset,
        4: MultilingualENCorpusGermanDataset,
        5: MultilingualENCorpusFrenchDataset,
        6: MultilingualENCorpusCypriotDataset,
        7: MultilingualENCorpusSloveneDataset,
    }
    if exp_nb not in mapping:
        raise ValueError(f"Unknown exp_nb: {exp_nb}")
    return mapping[exp_nb]


def build_top_binary_lists(ds, candidate_cols, max_Ns):
    """
    Return dict N -> list of top-N binary column names (one-hot), ranked by document frequency.
    """
    df_meta = ds.data.reset_index(drop=True)
    onehot = pd.get_dummies(df_meta[candidate_cols].astype(str), prefix_sep='=')
    freqs = onehot.sum(axis=0).sort_values(ascending=False)
    binary_cols_sorted = freqs.index.tolist()
    out = {}
    for N in max_Ns:
        if N <= 0:
            out[N] = []
        else:
            out[N] = binary_cols_sorted[:N]
    return out, onehot


def save_combined_embeddings(ds, top_binary_cols, out_path):
    """
    Concatenate text embeddings with the selected binary columns and save numpy to out_path.
    """
    emb = ds.embeddings
    if hasattr(emb, "cpu"):
        emb_np = emb.cpu().numpy()
    else:
        emb_np = np.array(emb)
    df_meta = ds.data.reset_index(drop=True)
    if len(top_binary_cols) == 0:
        combined = emb_np.astype(np.float32)
    else:
        # select candidate columns automatically from CONTEXT_COLUMNS_FULL if present
        if hasattr(ds, "CONTEXT_COLUMNS_FULL"):
            candidate_cols = [c for c in ds.CONTEXT_COLUMNS_FULL if c in df_meta.columns]
        else:
            candidate_cols = [c for c in df_meta.columns if c.lower() not in ("text", "tweet, text")]
        onehot = pd.get_dummies(df_meta[candidate_cols].astype(str), prefix_sep='=')
        missing = [c for c in top_binary_cols if c not in onehot.columns]
        if missing:
            raise RuntimeError(f"Requested binary columns not present in one-hot frame: {missing}")
        meta_feats = onehot[top_binary_cols].values.astype(np.float32)
        combined = np.concatenate([emb_np.astype(np.float32), meta_feats], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, combined)
    print(f"[INFO] Saved combined features to {out_path} shape={combined.shape}")
    return out_path


def find_latest_adj_final():
    files = sorted(glob.glob("./adjacency_matrices/adjacency_final__*.pkl"), key=os.path.getmtime)
    if not files:
        return None
    return files[-1]


def find_latest_embeddings_file():
    files = sorted(glob.glob("./embeddings/embeddings__*.npy"), key=os.path.getmtime)
    if not files:
        return None
    return files[-1]


def compute_neighbor_context_match(adj, df_meta, cols_to_check=("In-Group", "Out-group"), top_k=None):
    N = adj.shape[0]
    stats = {}
    for col in cols_to_check:
        if col not in df_meta.columns:
            stats[col] = np.nan
            continue
        df_values = df_meta[col].astype(str).tolist()
        fracs = []
        for i in range(N):
            neigh = np.where(adj[i] != 0)[0]
            if len(neigh) == 0:
                continue
            if top_k is not None and len(neigh) > top_k:
                neigh = neigh[:top_k]
            same = sum(1 for j in neigh if df_values[j] == df_values[i])
            fracs.append(same / len(neigh))
        stats[col] = float(np.mean(fracs)) if len(fracs) > 0 else float(0.0)
    return stats


def safe_copy(src, dst):
    """
    Copy src -> dst, if dst exists append an incrementing suffix.
    Returns final dst path.
    """
    base, ext = os.path.splitext(dst)
    final = dst
    i = 1
    while os.path.exists(final):
        final = f"{base}__dup{i}{ext}"
        i += 1
    shutil.copy(src, final)
    return final



def run_attribute_k_sweep(args):
    """
    Sweep runner: iterates over all (Max, K) combinations.
      - Max controls attr_edges_max (anchor adjacency density).
      - K controls n_neighbors for the final kNN graph.
      - candidate_columns controls binary one-hot columns appended to embeddings.
        If not specified, NO metadata is appended (pure text embeddings).
    """
    exp_nb = args.exp_nb
    DatasetClass = get_dataset_class(exp_nb)
    print(f"[SWEEP] Instantiating dataset class {DatasetClass.__name__} (this will compute/load embeddings)")
    ds = DatasetClass(exp_nb, embeddings_path=None, skip_embeddings=False)
    df_meta = ds.data.reset_index(drop=True)

    if args.candidate_columns is None:
        candidate_columns = []
    elif isinstance(args.candidate_columns, str):
        if args.candidate_columns.strip() == "":
            candidate_columns = []
        else:
            candidate_columns = [c.strip() for c in args.candidate_columns.split(",") if c.strip()]
    elif isinstance(args.candidate_columns, (list, tuple)):
        candidate_columns = []
        for entry in args.candidate_columns:
            if not isinstance(entry, str):
                continue
            if "," in entry:
                candidate_columns.extend([c.strip() for c in entry.split(",") if c.strip()])
            else:
                candidate_columns.append(entry.strip())
    else:
        raise RuntimeError(
            "Unexpected type for --candidate_columns. "
            "Provide a comma-separated string or multiple arguments."
        )

    print("[SWEEP] Candidate columns:", candidate_columns if candidate_columns else "(none — pure text embeddings)")

    max_list = [int(x) for x in args.sweep_max_list.split(",") if x.strip() != ""]
    k_list = [int(x) for x in args.sweep_k_list.split(",") if x.strip() != ""]

    # Build top-binary map only when there are candidate columns to expand.
    if candidate_columns:
        top_binary_map, _ = build_top_binary_lists(ds, candidate_columns, max_list)
    else:
        # No candidate columns → every N maps to an empty binary column list.
        top_binary_map = {N: [] for N in max_list}

    combined_map = {}
    for N in sorted(set(max_list)):
        fname = f"./embeddings/combined_exp{exp_nb}_max{N}.npy"
        if os.path.exists(fname):
            print(f"[SWEEP] Combined file already exists: {fname}")
            combined_map[N] = fname
            continue
        top_cols = top_binary_map.get(N, [])
        print(f"[SWEEP] Building combined embeddings for Max={N} (binary dims={len(top_cols)}) ...")
        save_combined_embeddings(ds, top_cols, fname)
        combined_map[N] = fname

    sweep_csv = "./sweep_results_main.csv"
    header = ["timestamp", "exp_nb", "Max", "K", "combined_emb", "saved_emb", "saved_adj", "in_group_match", "out_group_match", "duration_s"]
    if not os.path.exists(sweep_csv):
        pd.DataFrame(columns=header).to_csv(sweep_csv, index=False)

    results = []
    for N in sorted(set(max_list)):
        embeddings_path = combined_map[N]
        for K in k_list:
            print(f"\n[SWEEP] Running Exp={exp_nb} Max={N} K={K}")
            ep = ExperimentParameters(exp_nb)
            for arg in vars(args):
                setattr(ep, arg, getattr(args, arg))
            ep.embeddings_path = embeddings_path
            ep.n_neighbors = K
            ep.add_attr_edges = True
            ep.attr_edges_max = N

            if getattr(args, "sweep_epochs", None) is not None:
                ep.epochs = args.sweep_epochs

            experiment = Experiment()
            start = time.time()
            experiment.train(ep)
            duration = time.time() - start

            adj_file = find_latest_adj_final()
            emb_file = find_latest_embeddings_file()

            stats = {"In-Group": None, "Out-group": None}
            if adj_file is not None:
                try:
                    with open(adj_file, "rb") as fh:
                        adj = pickle.load(fh)
                    if hasattr(adj, "toarray"):
                        adj_mat = adj.toarray()
                    else:
                        adj_mat = np.array(adj)
                    stats = compute_neighbor_context_match(adj_mat, df_meta, cols_to_check=("In-Group", "Out-group"), top_k=K)
                    print(f"[SWEEP METRIC] adj_file={adj_file} In-Group={stats['In-Group']:.4f} Out-group={stats['Out-group']:.4f}")
                except Exception as e:
                    print("[SWEEP] Failed to load/compute metric from", adj_file, ":", e)

            saved_emb = None
            saved_adj = None
            timestamp = int(time.time())
            try:
                if emb_file is not None:
                    emb_base = os.path.basename(emb_file)
                    new_emb_name = f"./embeddings/embeddings__exp{exp_nb}__max{N}__K{K}__{timestamp}__{emb_base}"
                    saved_emb = safe_copy(emb_file, new_emb_name)
                    print(f"[SWEEP] Copied embeddings: {emb_file} -> {saved_emb}")
            except Exception as e:
                print("[SWEEP] Failed to copy embeddings file:", e)
            try:
                if adj_file is not None:
                    adj_base = os.path.basename(adj_file)
                    new_adj_name = f"./adjacency_matrices/adj_final__exp{exp_nb}__max{N}__K{K}__{timestamp}__{adj_base}"
                    saved_adj = safe_copy(adj_file, new_adj_name)
                    print(f"[SWEEP] Copied adjacency: {adj_file} -> {saved_adj}")
            except Exception as e:
                print("[SWEEP] Failed to copy adjacency file:", e)

            row = {
                "timestamp": timestamp,
                "exp_nb": exp_nb,
                "Max": N,
                "K": K,
                "combined_emb": embeddings_path,
                "saved_emb": saved_emb,
                "saved_adj": saved_adj,
                "in_group_match": stats.get("In-Group"),
                "out_group_match": stats.get("Out-group"),
                "duration_s": duration,
            }
            pd.DataFrame([row]).to_csv(sweep_csv, mode="a", header=False, index=False)
            results.append(row)

    print("\n[SWEEP] Summary:")
    print(pd.DataFrame(results))
    print(f"[SWEEP] Detailed run records appended to {sweep_csv}")
    return results

# -------------------------
# End of sweep helpers
# -------------------------


class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)

    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def per_class_accuracy(self, predictions, labels):
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        num_classes = max(predictions.max(), labels.max()) + 1
        correct = predictions == labels
        accuracies = torch.tensor([
            correct[labels == i].float().mean() if (labels == i).any() else 0.0
            for i in range(num_classes)
        ])
        with open("./accuracies.txt", "a") as f:
            line = "/".join(map(str, accuracies.cpu().numpy().flatten()))
            f.write(line + '\n')
        return accuracies

    def per_class_acc_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        accu = self.per_class_accuracy(logp[mask], labels[mask])
        return accu

    def loss_gcl(self, model, graph_learner, features, anchor_adj, args, context_dataset=None, dynamic_context_weight=None):
        maskfeat_rate_anchor = args.maskfeat_rate_anchor
        maskfeat_rate_learner = args.maskfeat_rate_learner
        mask_v1, _ = get_feat_mask(features, maskfeat_rate_anchor)
        features_v1 = features * (1 - mask_v1)
        z1, _ = model(features_v1, anchor_adj, 'anchor')
        mask, _ = get_feat_mask(features, maskfeat_rate_learner)
        features_v2 = features * (1 - mask)
        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = learned_adj.detach()
            learned_adj = normalize(learned_adj, 'sym', args.sparse)
        z2, _ = model(features_v2, learned_adj, 'learner')

        # Compute only contrastive loss
        from model import contrastive_loss
        contrast_loss = contrastive_loss(z1, z2, temperature=args.temperature, sym=args.sym)
        total_loss = contrast_loss
        context_loss = torch.tensor(0.0)  # Keep for backward compatibility

        return total_loss, contrast_loss, context_loss, learned_adj

    def train(self, args):
        run_suffix = build_run_suffix(args, max_parts=4)
        context_dataset = None

        # ======= DATASET SELECTION =======
        if args.exp_nb == 1:
            dataset = ToxigenDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 2:
            dataset = LGBTEnDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 3:
            dataset = MigrantsEnDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 4:
            dataset = MultilingualENCorpusGermanDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 5:
            dataset = MultilingualENCorpusFrenchDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 6:
            dataset = MultilingualENCorpusCypriotDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 7:
            dataset = MultilingualENCorpusSloveneDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        else:
            raise ValueError(f"Unknown experiment number: {args.exp_nb}")

        # ======= GET DATASET (features, labels, masks, adjacency) =======
        if getattr(args, "gsl_mode", "structure_refinement") == 'structure_refinement':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = dataset.get_dataset()
        elif getattr(args, "gsl_mode", "structure_inference") == 'structure_inference':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = dataset.get_dataset()
        else:
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = dataset.get_dataset()

        # ======= If user provided context columns, encode metadata and optionally concatenate to features =======
        if getattr(args, "context_columns", None):
            # Prefer dataset.data for metadata encoding
            if context_dataset is not None:
                df_meta = context_dataset.data.reset_index(drop=True)
                try:
                    meta_feats = encode_metadata(df_meta, args.context_columns)
                except Exception as e:
                    raise RuntimeError(f"Failed to encode metadata columns {args.context_columns}: {e}")

                # Concatenate metadata to features (handle numpy, DataFrame, or torch tensor)
                if isinstance(features, np.ndarray):
                    features = np.concatenate([features.astype(np.float32), meta_feats], axis=1)
                    nfeats = features.shape[1]
                    os.makedirs("./embeddings", exist_ok=True)
                    combined_path = f"./embeddings/combined_exp{args.exp_nb}.npy"
                    np.save(combined_path, features)
                    print(f"[INFO] Saved combined features to {combined_path} (text embeddings + metadata). New shape: {features.shape}")
                elif isinstance(features, pd.DataFrame):
                    df_features = features.reset_index(drop=True)
                    meta_df = pd.DataFrame(meta_feats, index=df_features.index)
                    df_combined = pd.concat([df_features, meta_df], axis=1)
                    features = df_combined.values.astype(np.float32)
                    nfeats = features.shape[1]
                    os.makedirs("./embeddings", exist_ok=True)
                    combined_path = f"./embeddings/combined_exp{args.exp_nb}.npy"
                    np.save(combined_path, features)
                    print(f"[INFO] Saved combined features to {combined_path} (DataFrame->numpy). New shape: {features}")
                else:
                    # e.g., torch tensor
                    try:
                        feat_np = features.cpu().numpy()
                        features = np.concatenate([feat_np.astype(np.float32), meta_feats], axis=1)
                        nfeats = features.shape[1]
                        os.makedirs("./embeddings", exist_ok=True)
                        combined_path = f"./embeddings/combined_exp{args.exp_nb}.npy"
                        np.save(combined_path, features)
                        print(f"[INFO] Saved combined features to {combined_path} (tensor->numpy). New shape: {features.shape}")
                    except Exception as e:
                        raise RuntimeError(f"Unable to combine features with metadata: {e}")
            else:

                pass

        for trial in range(getattr(args, "ntrials", 1)):
            self.setup_seed(trial)

            # --- Context-based adjacency matrix ---
            use_context_adj = getattr(args, "use_context_adj", False)
            if use_context_adj and args.context_columns is not None:
                # Prefer dataset metadata if available
                if context_dataset is not None:
                    df_context = context_dataset.data.copy()
                    missing = [c for c in args.context_columns if c not in df_context.columns]
                    if missing:
                        raise ValueError(f"Context columns not found in dataset: {missing}")
                    df_context = df_context[list(args.context_columns)]
                elif isinstance(features, pd.DataFrame):
                    df_context = features.copy()
                    missing = [c for c in args.context_columns if c not in df_context.columns]
                    if missing:
                        raise ValueError(f"Context columns not found in provided features: {missing}")
                    df_context = df_context[list(args.context_columns)]
                else:
                    raise ValueError(
                        "Cannot build context adjacency: no dataset metadata available. "
                        "Provide --embeddings_path pointing to a combined features .npy or avoid --use_context_adj."
                    )

                # If user requested attribute-based co-membership edges,
                # build adjacency from attributes using add_attribute_co_membership_edges_np.
                if getattr(args, "add_attr_edges", False) and getattr(args, "attr_edges_max", None) is not None:
                    # Prepare embeddings array to pass into the function (prefer numpy)
                    try:
                        if isinstance(features, torch.Tensor):
                            emb_np = features.cpu().numpy()
                        else:
                            emb_np = np.array(features)
                    except Exception:
                        emb_np = None
                    try:
                        attr_adj = add_attribute_co_membership_edges_np(df_context, max_edges_per_node=int(args.attr_edges_max), embeddings=emb_np)
                        anchor_adj_raw = torch.from_numpy(attr_adj.astype(np.float32))
                        print(f"[INFO] Built attribute co-membership adjacency using max_edges_per_node={args.attr_edges_max}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to build attribute co-membership adjacency: {e}")
                else:
                    adj_context = make_context_adjacency(df_context, df_context.columns)
                    anchor_adj_raw = torch.from_numpy(adj_context.astype(np.float32)).float()
            else:
                if getattr(args, "gsl_mode", "structure_inference") == 'structure_inference':
                    if getattr(args, "sparse", False):
                        anchor_adj_raw = torch_sparse_eye(features.shape[0])
                    else:
                        anchor_adj_raw = torch.eye(features.shape[0]).float()
                elif getattr(args, "gsl_mode", "structure_refinement") == 'structure_refinement':
                    if getattr(args, "sparse", False):
                        anchor_adj_raw = adj_original
                    else:
                        if isinstance(adj_original, list):
                            adj_original = np.array(adj_original)
                        if isinstance(adj_original, np.ndarray):
                            if adj_original.ndim == 1 or adj_original.shape[0] == 0:
                                adj_original = np.eye(features.shape[0], dtype=np.float32)
                            elif adj_original.ndim == 0:
                                raise ValueError("adj_original is scalar, not adjacency matrix!")
                            elif adj_original.ndim == 2:
                                adj_original = adj_original.astype(np.float32)
                        print("adj_original shape:", adj_original.shape)
                        anchor_adj_raw = torch.from_numpy(adj_original).float()

            anchor_adj = normalize(anchor_adj_raw, 'sym', getattr(args, "sparse", False))
            if getattr(args, "sparse", False):
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            # Choose graph learner
            type_learner = args.type_learner
            if type_learner == 'fgp':
                graph_learner = FGP_learner(
                    features if isinstance(features, torch.Tensor) else torch.tensor(features), args.k, args.sim_function, 6, args.sparse)
            elif type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                            args.activation_learner)
            elif type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                            args.activation_learner)
            elif type_learner == 'gnn':
                graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                            args.activation_learner, anchor_adj)
            else:
                raise ValueError(f"Unknown type_learner: {type_learner}.")

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                        emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                        dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)

            # Move to GPU if available
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                model = model.cuda()
                graph_learner = graph_learner.cuda()
                train_mask = torch.tensor(train_mask).cuda()
                val_mask = torch.tensor(val_mask).cuda()
                test_mask = torch.tensor(test_mask).cuda()
                # features may be numpy -> convert to tensor first
                if not isinstance(features, torch.Tensor):
                    features = torch.tensor(features, dtype=torch.float32).cuda()
                else:
                    features = features.cuda()
                labels = torch.tensor(labels).cuda()
                if not args.sparse:
                    anchor_adj = anchor_adj.cuda()
            else:
                if not isinstance(features, torch.Tensor):
                    features = torch.tensor(features, dtype=torch.float32)
                labels = torch.tensor(labels)

            loss_list = []
            contrastive_loss_list = []

            # Bootstrapping parameters (read from args if present)
            try:
                tau = float(getattr(args, "bootstrap_tau", 1.0))
                interval = int(getattr(args, "bootstrap_interval", 10))
            except Exception:
                tau = 1.0
                interval = 10

            for epoch in range(1, args.epochs + 1):
                model.train()
                graph_learner.train()
                total_loss, contrast_loss, context_loss, Adj = self.loss_gcl(
                    model, graph_learner, features, anchor_adj, args, context_dataset=context_dataset,
                    dynamic_context_weight=None
                )

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                total_loss.backward()
                # optional: gradient clipping could be inserted here if desired
                optimizer_cl.step()
                optimizer_learner.step()

                # Bootstrapping anchor update (SUBLIME-style): A_a <- tau * A_a + (1-tau) * S
                if tau < 1.0 and (epoch % interval == 0):
                    try:
                        if isinstance(Adj, torch.Tensor):
                            device = Adj.device
                            if not isinstance(anchor_adj_raw, torch.Tensor):
                                anchor_adj_raw = torch.from_numpy(np.array(anchor_adj_raw)).to(device)
                            else:
                                anchor_adj_raw = anchor_adj_raw.to(device)
                            with torch.no_grad():
                                learned = Adj.detach()
                                new_anchor = tau * anchor_adj_raw + (1.0 - tau) * learned
                                new_anchor = normalize(new_anchor, 'sym', getattr(args, "sparse", False))
                                anchor_adj_raw = new_anchor
                                if getattr(args, "sparse", False):
                                    try:
                                        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_raw)
                                    except Exception:
                                        anchor_adj = anchor_adj_raw
                                else:
                                    anchor_adj = anchor_adj_raw
                                    if use_cuda and isinstance(anchor_adj, torch.Tensor):
                                        anchor_adj = anchor_adj.cuda()
                                # logging the mean change may help debugging:
                                try:
                                    delta = (new_anchor - learned).abs().mean().item()
                                except Exception:
                                    delta = None
                                print(f"[BOOTSTRAP] Epoch {epoch}: updated anchor adjacency with tau={tau} delta_mean={delta}")
                        else:
                            print("[BOOTSTRAP] Learned Adj not a dense torch.Tensor; skipping bootstrap this interval.")
                    except Exception as e:
                        print("[BOOTSTRAP] Exception during anchor bootstrap update:", e)

                loss_list.append(total_loss.item())
                contrastive_loss_list.append(contrast_loss.item())

                print(f"Epoch {epoch:05d} | Contrastive: {contrast_loss:.4f} | Total: {total_loss:.4f}")

                if epoch % 200 == 0 or epoch == args.epochs:
                    model.eval()
                    graph_learner.eval()
                    f_adj = Adj
                    # detach edge weights if sparse DGL graph
                    if args.sparse and hasattr(f_adj, 'edata'):
                        try:
                            f_adj.edata['w'] = f_adj.edata['w'].detach()
                        except Exception:
                            pass
                    else:
                        try:
                            f_adj = f_adj.detach()
                        except Exception:
                            pass
                    os.makedirs("./adjacency_matrices", exist_ok=True)
                    adj_file = f'./adjacency_matrices/adjacency_learned_epoch_{epoch}__{run_suffix}.pkl'
                    with open(adj_file, 'wb') as file:
                        pickle.dump(f_adj, file)

            # Plot losses and save
            epochs_arr = np.arange(1, len(loss_list) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_arr, loss_list, label="Total Loss")
            plt.plot(epochs_arr, contrastive_loss_list, label="Contrastive Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Evolution Over Epochs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"losses__{run_suffix}.png")
            plt.close()

            # Save losses as a single file with both loss types
            # To load: data = np.load('losses__.npy', allow_pickle=True).item()
            # Then access via data['total_loss'] and data['contrastive_loss']
            losses_dict = {
                'total_loss': np.array(loss_list),
                'contrastive_loss': np.array(contrastive_loss_list)
            }
            np.save(f"losses__{run_suffix}.npy", losses_dict)

            model.eval()
            graph_learner.eval()
            with torch.no_grad():
                learned_adj_final = graph_learner(features)
                if not args.sparse:
                    learned_adj_final = symmetrize(learned_adj_final)
                    learned_adj_final = normalize(learned_adj_final, 'sym', args.sparse)
                embeddings, _ = model(features, learned_adj_final)
                embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)
                emb_file = f'./embeddings/embeddings__{run_suffix}.npy'
                os.makedirs("./embeddings", exist_ok=True)
                np.save(emb_file, embeddings_np)
       

            adj_final = kneighbors_graph(
                embeddings_np, n_neighbors=args.n_neighbors, metric="cosine", mode='connectivity'
            ).toarray()
            os.makedirs("./adjacency_matrices", exist_ok=True)
            adj_final_file = f'./adjacency_matrices/adjacency_final__{run_suffix}.pkl'
            with open(adj_final_file, 'wb') as file:
                pickle.dump(adj_final, file)

            try:
                save_loss_plot(loss_list, args)
            except Exception:
                pass

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_nb', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_context_adj', action='store_true', help='Use context-based adjacency matrix')
    parser.add_argument('--context_columns', nargs='+', default=None, help='Columns to build context adjacency from')
    parser.add_argument('--embeddings_path', type=str, default=None)

    # paper-aligned defaults
    parser.add_argument('--temperature', type=float, default=0.08)
    parser.add_argument('--maskfeat_rate_anchor', type=float, default=0.35)
    parser.add_argument('--maskfeat_rate_learner', type=float, default=0.35)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--rep_dim', type=int, default=256)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dropedge_rate', type=float, default=0.2)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--type_learner', type=str, default='fgp', choices=['fgp', 'mlp', 'att', 'gnn'])
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--sim_function', type=str, default='cosine')
    parser.add_argument('--activation_learner', type=str, default='relu')
    parser.add_argument('--gsl_mode', type=str, default='structure_refinement', choices=['structure_refinement', 'structure_inference'])
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--w_decay', type=float, default=1e-4)
    parser.add_argument('--sym', action='store_true')

    # Sweep options (new). Use --run_sweep to activate.
    parser.add_argument('--run_sweep', action='store_true', help='Run attribute (Max) x K sweep inside main.py')
    parser.add_argument('--sweep_max_list', type=str, default="0,10,50", help='Comma-separated Max values (number of binary attribute dims)')
    parser.add_argument('--sweep_k_list', type=str, default="5,10,15", help='Comma-separated K values (n_neighbors for final kNN)')
    # candidate_columns accepts either multiple args or a single comma-separated string
    parser.add_argument('--candidate_columns', nargs='*', default=None, help='Comma-separated or space-separated candidate columns for one-hot expansion')
    parser.add_argument('--sweep_epochs', type=int, default=None, help='Override epochs during sweep runs (optional)')

    # attribute-edge behavior
    parser.add_argument('--add_attr_edges', action='store_true', help='When building context adjacency, add attribute co-membership edges (colleague behavior)')
    parser.add_argument('--attr_edges_max', type=int, default=None, help='Max edges per node to add from attributes (used when --add_attr_edges is set)')

    # SUBLIME-specific bootstrapping options
    parser.add_argument('--bootstrap_tau', type=float, default=1.0, help='Bootstrapping decay rate tau for anchor update (A_a <- tau*A_a + (1-tau)*S). Set <1 to enable.')
    parser.add_argument('--bootstrap_interval', type=int, default=10, help='Interval (in epochs) to perform anchor bootstrapping update.')

    return parser.parse_args()


if __name__ == '__main__':
    cl_args = parse_cli()

    # If user wants to run the attribute-K sweep, do it here and exit.
    if getattr(cl_args, "run_sweep", False):
        # Validate inputs quickly
        if not cl_args.sweep_max_list:
            raise ValueError("Please provide --sweep_max_list (e.g. 0,10,50)")
        if not cl_args.sweep_k_list:
            raise ValueError("Please provide --sweep_k_list (e.g. 5,10,15)")
        # run sweep (this will load embeddings and run multiple Experiment.train calls)
        run_attribute_k_sweep(cl_args)
        print("[MAIN] Sweep finished.")
        raise SystemExit(0)

    if torch.cuda.is_available():
        torch.cuda.set_device(cl_args.gpu)
    # Load experiment params (ExperimentParameters will read local CSV via utils.py)
    experiment_params = ExperimentParameters(cl_args.exp_nb)
    # Copy CLI args into experiment parameters object so downstream code uses them
    for arg in vars(cl_args):
        setattr(experiment_params, arg, getattr(cl_args, arg))
    experiment = Experiment()
    experiment.train(experiment_params)
