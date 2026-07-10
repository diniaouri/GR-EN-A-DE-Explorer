#!/usr/bin/env python3

import math
import torch
import numpy as np
from time import time
from pathlib import Path
from typing import NamedTuple, Optional, Any
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import normalize, softmax
from tqdm import tqdm
import csv
import re
import sys

from parser import argument_parser
from datasets import DATASET_LOADERS
from model import Model
# from torch_geometric.nn.models.gnn_explainer import GNNExplainer
from batched_explainer import BatchedGNNExplainer as GNNExplainer
from PGMEx import PGMExplainer
# from intgrad import IntegratedGradExplainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).parent
CONVERGENCE_DIR = HERE / "convergence_files"
CONVERGENCE_DIR.mkdir(exist_ok=True)

MULTILINGUAL_LABEL_COLS = [
    "Topic",
    "Initiating Problem",
    "Intolerance",
    "Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)",
    "Polarization/Othering",
    "Perceived Threat",
    "Solution",
]

ALL_SEEDS = [0, 21, 42, 84, 123]          # NEW: canonical seed list


# ─── Training utilities ───────────────────────────────────────────────────────

class FocalLoss(torch.nn.Module):
    """Focal loss with optional label smoothing and class weights."""

    def __init__(
        self,
        gamma: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(
            input,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class EarlyStopping:
    """Stops training when a validation metric stops improving."""

    def __init__(self, patience: int = 100, min_epochs: int = 100):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_val = -float("inf")
        self.counter = 0

    def step(self, epoch: int, val_metric: float) -> bool:
        if epoch < self.min_epochs:
            return False
        if val_metric > self.best_val:
            self.best_val = val_metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def apply_edge_dropout(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor],
    p: float,
    training: bool = True,
) -> tuple:
    if not training or p <= 0.0:
        return edge_index, edge_weight
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) > p
    filtered_weight = edge_weight[keep] if edge_weight is not None else None
    return edge_index[:, keep], filtered_weight


def get_lr_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up then cosine annealing down to min_lr."""

    def _lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

# ─────────────────────────────────────────────────────────────────────────────


def get_experiment_config(exp_nb: int, grenade_root: Path) -> dict:
    csv_path = grenade_root / "grenade_original/Contrastive_Learning_Approach/src/experiment_params.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"experiment_params.csv not found at {csv_path}")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        with open(csv_path, 'r', encoding='utf-16') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    for row in rows:
        if int(row.get('exp_nb', -1)) == exp_nb:
            return {
                'dataset': row.get('dataset', 'Toxigen'),
                'label_col': row.get('label_col', 'target_group'),
                'text_col': row.get('text_col', 'text'),
                'context_cols': [
                    col.strip()
                    for col in row.get('context_cols', 'In-Group,Out-group').split(',')
                ],
            }
    raise ValueError(f"Experiment {exp_nb} not found in experiment_params.csv")


def extract_exp_nb_from_path(embeddings_path: str) -> int:
    match = re.search(r'exp(\d+)', embeddings_path)
    if match:
        return int(match.group(1))
    return 1


class PerformanceResults(NamedTuple):
    train_acc: float
    val_acc: float
    test_acc: float
    train_auroc: float
    val_auroc: float
    test_auroc: float
    test_f1_score: float
    val_f1_score: float
    # ── Confusion-matrix counts at the (test-set) evaluation, defaults keep
    #    other branches/callers unaffected. Captured so the best-Val-F1
    #    checkpoint snapshot carries its CM into cross-seed aggregation. ──
    cm_tp: float = 0.0
    cm_fn: float = 0.0
    cm_fp: float = 0.0
    cm_tn: float = 0.0


# region main ---------------------------------

def main(
    grenade_embeddings: str,
    grenade_adjacency: str,
    grenade_labels: str = None,
    grenade_exp_nb: int = None,
    dataset_name: str = "toxigen",
    label_col: str = None,
    arch: str = "graphsage",
    explainer: str = "gnn_explainer",
    num_layers: int = 3,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    jk_mode: str = "cat",
    activation: str = "elu",
    batch_size: int = 200,
    seed: int = 912,
    epochs: int = 150,
    model_saving_lag: int = 25,
    vanilla_mode: bool = False,
    lr_gnn: float = 0.01,
    weight_decay: float = 0.001,
    focal_gamma: float = 1.0,
    label_smoothing: float = 0.05,
    grad_clip: float = 1.0,
    edge_dropout: float = 0.1,
    warmup_epochs: int = 20,
    min_lr: float = 1e-5,
    patience: int = 100,
    min_epochs: int = 100,
    oversample_minority: bool = False,
    explainer_iters: int = 5,
    correct_sampling_percent: float = 0.4,
    explanations_lag: int = 20,
    explanation_topk_thresh: float = 0.3,
    lr_gnnex: float = 0.01,
    explainer_epochs: int = 200,
    device: str = None,
    tune_threshold: bool = True,
) -> Optional[PerformanceResults]:          # CHANGED: now returns best PerformanceResults
    global DEVICE
    if device is not None:
        DEVICE = device
    print(f"  Running on: {DEVICE.upper()}")

    # ── Reproducibility: seed every RNG that affects training ─────────────────
    # The data split is fixed independently below (SPLIT_SEED), so across seeds
    # only weight init / dropout / edge-dropout / sampling vary — a clean
    # "same split, different seed" protocol.
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN determinism (some scatter/gather GPU ops may still be nondeterministic).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  Seeded RNGs with seed={seed}")
    preds = None
    use_explanations = False
    #best_auroc_val = 0
    best_val_f1 = 0.0              # renamed
    best_performance = None                 # NEW: track performance at best checkpoint
    oversmoothing = 0

    from grenade_dataset import GrenadeOutputDataset

    exp_nb = grenade_exp_nb if grenade_exp_nb else extract_exp_nb_from_path(grenade_embeddings)

    if grenade_labels:
        labels = np.load(grenade_labels)
        print(f"✓ Loaded {len(labels)} labels from {grenade_labels}")
    else:
        print(f"Loading labels for experiment {exp_nb}...")
        try:
            exp_config = get_experiment_config(exp_nb, HERE.parent)
            dataset_name = exp_config['dataset']
            label_col = label_col or exp_config['label_col']
            print(f"  Dataset: {dataset_name}")
            print(f"  Label column: {label_col}")
            sys.path.insert(0, str(HERE.parent / "grenade_original/Contrastive_Learning_Approach/src"))
            try:
                from preprocessing import (
                    ToxigenDataset,
                    LGBTEnDataset,
                    MigrantsEnDataset,
                    MultilingualENCorpusGermanDataset,
                    MultilingualENCorpusFrenchDataset,
                    MultilingualENCorpusCypriotDataset,
                    MultilingualENCorpusSloveneDataset,
                )
                DATASETS = {
                    "Toxigen": (ToxigenDataset, "csv"),
                    "LGBTEn": (LGBTEnDataset, "csv"),
                    "MigrantsEn": (MigrantsEnDataset, "csv"),
                    "Multilingual_EN_Corpus_DATA_FRENCH.xlsx": (MultilingualENCorpusFrenchDataset, "xlsx"),
                    "Multilingual_EN_Corpus_DATA_GERMAN.xlsx": (MultilingualENCorpusGermanDataset, "xlsx"),
                    "Multilingual_EN_Corpus_DATA_CYPRIOT.xlsx": (MultilingualENCorpusCypriotDataset, "xlsx"),
                    "Multilingual_EN_Corpus_DATA_SLOVENE.xlsx": (MultilingualENCorpusSloveneDataset, "xlsx"),
                    "MultilingualENCorpusFrench": (MultilingualENCorpusFrenchDataset, "xlsx"),
                    "MultilingualENCorpusGerman": (MultilingualENCorpusGermanDataset, "xlsx"),
                    "MultilingualENCorpusCypriot": (MultilingualENCorpusCypriotDataset, "xlsx"),
                    "MultilingualENCorpusSlovene": (MultilingualENCorpusSloveneDataset, "xlsx"),
                }
                if dataset_name not in DATASETS:
                    raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASETS.keys())}")
                dataset_cls, file_ext = DATASETS[dataset_name]
                datasets_dir = HERE.parent / "grenade_original/Contrastive_Learning_Approach/datasets"
                if file_ext == "xlsx":
                    _CLASSNAME_TO_FILE = {
                        "MultilingualENCorpusFrench": "Multilingual_EN_Corpus_DATA_FRENCH.xlsx",
                        "MultilingualENCorpusGerman": "Multilingual_EN_Corpus_DATA_GERMAN.xlsx",
                        "MultilingualENCorpusCypriot": "Multilingual_EN_Corpus_DATA_CYPRIOT.xlsx",
                        "MultilingualENCorpusSlovene": "Multilingual_EN_Corpus_DATA_SLOVENE.xlsx",
                    }
                    xlsx_filename = (
                        dataset_name
                        if dataset_name.endswith(".xlsx")
                        else _CLASSNAME_TO_FILE[dataset_name]
                    )
                    data_path = str(datasets_dir / xlsx_filename)
                    grenade_dataset = dataset_cls(
                        experiment_nb=exp_nb,
                        data_path=data_path,
                        skip_embeddings=True,
                    )
                else:
                    csv_path = str(datasets_dir / f"{dataset_name}.csv")
                    grenade_dataset = dataset_cls(
                        experiment_nb=exp_nb,
                        csv_path=csv_path,
                        skip_embeddings=True,
                    )
                if label_col not in grenade_dataset.data.columns:
                    raise ValueError(
                        f"Label column '{label_col}' not found in dataset. "
                        f"Available: {grenade_dataset.data.columns.tolist()}"
                    )
                label_values = grenade_dataset.data[label_col].values
                encoder = LabelEncoder()
                labels = encoder.fit_transform(label_values)
                print(f"✓ Loaded {len(labels)} labels")
                print(
                    f"  Classes: {len(encoder.classes_)} - "
                    f"{encoder.classes_[:5]}{'...' if len(encoder.classes_) > 5 else ''}"
                )
                print(f"  Distribution: {np.bincount(labels)}")
            finally:
                grenade_src = str(HERE.parent / "grenade_original/Contrastive_Learning_Approach/src")
                if grenade_src in sys.path:
                    sys.path.remove(grenade_src)
        except Exception as e:
            print(f"Warning: Could not load labels from GRENADE: {e}")
            print("Falling back to dummy labels")
            embeddings = np.load(grenade_embeddings)
            labels = np.random.randint(0, 2, size=len(embeddings))
            print(f"Warning: Using {len(labels)} random binary labels (0 or 1)")

    safe_label = re.sub(r'[^\w-]', '_', label_col).strip('_') if label_col else "default"
    out_dir = HERE / f"grenade-{arch}-{safe_label}"
    out_dir.mkdir(exist_ok=True)
    mode_tag = "vanilla" if vanilla_mode else f"expass_topk_{explanation_topk_thresh}"
    convergence_file_stem = f"{safe_label}-loss-lrgnn_{lr_gnn}-seed_{seed}-{mode_tag}"
    best_model_path = out_dir / f"{convergence_file_stem}-best.pth"

    # ── Fixed, reproducible 60/20/20 random split ────────────────────────────
    # SPLIT_SEED is a CONSTANT (not the run seed), so the split is identical for
    # every seed and identical across relaunches. Uses its own numpy Generator so
    # it does not consume/perturb the global RNG seeded above for training.
    SPLIT_SEED = 42
    _n = len(labels)
    _perm = np.random.default_rng(SPLIT_SEED).permutation(_n)
    _n_train = int(_n * 0.6)
    _n_val = int(_n * 0.2)
    _train_mask = np.zeros(_n, dtype=bool)
    _val_mask = np.zeros(_n, dtype=bool)
    _test_mask = np.zeros(_n, dtype=bool)
    _train_mask[_perm[:_n_train]] = True
    _val_mask[_perm[_n_train:_n_train + _n_val]] = True
    _test_mask[_perm[_n_train + _n_val:]] = True
    print(
        f"  Fixed split (SPLIT_SEED={SPLIT_SEED}): "
        f"train={_train_mask.sum()} | val={_val_mask.sum()} | test={_test_mask.sum()}"
    )

    dataset_obj = GrenadeOutputDataset(
        embeddings_path=grenade_embeddings,
        adjacency_path=grenade_adjacency,
        labels=labels,
        train_mask=_train_mask,
        val_mask=_val_mask,
        test_mask=_test_mask,
    )
    data = dataset_obj._data_obj
    data = data.to(DEVICE)

    use_node_classification = False
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        use_node_classification = True
        num_classes = dataset_obj.num_classes
        print("=" * 80)
        print("Detected node classification mode (GRENADE)")
        print(f"  Total nodes: {data.x.shape[0]}")
        print(
            f"  Train nodes: {data.train_mask.sum().item()} "
            f"({100 * data.train_mask.sum().item() / data.x.shape[0]:.1f}%)"
        )
        print(
            f"  Val nodes: {data.val_mask.sum().item()} "
            f"({100 * data.val_mask.sum().item() / data.x.shape[0]:.1f}%)"
        )
        print(
            f"  Test nodes: {data.test_mask.sum().item()} "
            f"({100 * data.test_mask.sum().item() / data.x.shape[0]:.1f}%)"
        )
        print(f"  Number of classes: {num_classes}")
        print("=" * 80)
    else:
        print("Using graph classification mode (original EXPASS)")
        num_classes = 2

    data.idx = 0
    train_loader = [data]
    val_loader   = [data]
    test_loader  = [data]
    n_feat = train_loader[0].x.shape[1]

    model = Model(
        nhid=hidden_dim,
        nfeat=n_feat,
        nclass=num_classes,
        dropout=dropout,
        num_layers=num_layers,
        gnn_arch=arch,
        jk_mode=jk_mode,
        activation=activation,
        node_classification_mode=use_node_classification,
    ).to(DEVICE)

    sample_weights = cal_weights_model(train_loader, use_node_classification).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_gnn,
        weight_decay=weight_decay,
    )

    criterion = FocalLoss(
        gamma=focal_gamma,
        weight=sample_weights,
        label_smoothing=label_smoothing,
    )

    scheduler = get_lr_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        base_lr=lr_gnn,
        min_lr=min_lr,
    )

    early_stopping = EarlyStopping(patience=patience, min_epochs=min_epochs)

    explainer_obj = get_explainer(
        explainer=explainer,
        model=model,
        explainer_epochs=explainer_epochs,
        lr_gnnex=lr_gnnex,
        criterion=criterion,
    )

    for epoch in tqdm(range(epochs)):
        epoch_start_time = time()
        if not vanilla_mode and epoch > explanations_lag:
            use_explanations = True

        avg_loss = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            preds=preds,
            explainer=explainer_obj,
            use_explanations=use_explanations,
            explainer_iters=explainer_iters,
            correct_sampling_percent=correct_sampling_percent,
            explanation_topk_thresh=explanation_topk_thresh,
            use_node_classification=use_node_classification,
            grad_clip=grad_clip,
            edge_dropout_rate=edge_dropout,
        )
        output_train, performance = evaluate_performance(
            train_loader,
            val_loader,
            test_loader,
            model,
            use_node_classification,
            num_classes,
            tune_threshold=tune_threshold,
        )
        preds = output_train
        model_saving_lag = 25 if model_saving_lag is None else model_saving_lag
        #if epoch >= model_saving_lag and performance.val_auroc >= best_auroc_val:
         #   best_auroc_val = performance.val_auroc
          #  torch.save(model.state_dict(), best_model_path)
           # best_performance = performance  # NEW: snapshot at best checkpoint
            
            
        if epoch >= model_saving_lag and performance.val_f1_score >= best_val_f1:
            best_val_f1 = performance.val_f1_score
            torch.save(model.state_dict(), best_model_path)
            best_performance = performance
            print(
                f"  ★ BEST Val F1 = {best_val_f1:.4f} at epoch {epoch} "
                f"→ checkpoint selected & saved to {best_model_path.name}"
            )

        log_progress(
            epoch, avg_loss, performance, oversmoothing, convergence_file_stem, epoch_start_time
        )

        scheduler.step()

        #if early_stopping.step(epoch, performance.val_auroc):
        if early_stopping.step(epoch, performance.val_f1_score):   # also change here

            print(
                f"\nEarly stopping triggered at epoch {epoch} "
                f"(no improvement for {patience} epochs)."
            )
            break

    return best_performance                 # NEW: return for aggregation

# endregion main


# region Functions ---------------------------------

def print_seed_aggregation(
    results: list,
    seeds: list = None,
    label: str = "",
) -> None:
    """Print mean ± std of every PerformanceResults field across seeds.

    Args:
        results: List of PerformanceResults (one per seed run).
        seeds:   Seed values that produced each result (for display only).
        label:   Optional header tag, e.g. a label-column name.
    """
    if not results:
        print("No results to aggregate.")
        return

    seeds = seeds or ALL_SEEDS[: len(results)]
    fields = PerformanceResults._fields

    header = f"AGGREGATE RESULTS OVER {len(results)} SEEDS  {seeds}"
    if label:
        header += f"  [{label}]"

    sep = "=" * 80
    print(f"\n{sep}")
    print(header)
    print(sep)
    print(f"  {'Metric':<25} {'Mean':>10}  {'Std':>10}  {'Per-seed values'}")
    print(f"  {'-'*25} {'-'*10}  {'-'*10}  {'-'*30}")

    cm_fields = {"cm_tp", "cm_fn", "cm_fp", "cm_tn"}
    for field in fields:
        if field in cm_fields:
            continue  # confusion-matrix counts handled separately below
        values = []
        for r in results:
            v = getattr(r, field)
            values.append(float(v.item() if hasattr(v, "item") else v))
        mean_val = float(np.mean(values))
        std_val  = float(np.std(values))
        per_seed = "  ".join(f"{v:.4f}" for v in values)
        print(f"  {field:<25} {mean_val:>10.4f}  {std_val:>10.4f}  {per_seed}")

    print(sep)

    # ── Mean confusion matrix across seeds (percentages) ──────────────────────
    # Each seed's CM is taken at its best-Val-F1 checkpoint. We row-normalise
    # per true class within each seed (so % is comparable across seeds with
    # different test sizes), then average those percentages across seeds.
    def _cell(r, name):
        v = getattr(r, name, 0.0)
        return float(v.item() if hasattr(v, "item") else v)

    have_cm = any(
        (_cell(r, "cm_tp") + _cell(r, "cm_fn") + _cell(r, "cm_fp") + _cell(r, "cm_tn")) > 0
        for r in results
    )
    if have_cm:
        # Per-seed row-normalised percentages (true-class 0 and true-class 1 rows).
        tn_pct, fp_pct, fn_pct, tp_pct = [], [], [], []
        recalls, precisions = [], []
        for r in results:
            tp, fn = _cell(r, "cm_tp"), _cell(r, "cm_fn")
            fp, tn = _cell(r, "cm_fp"), _cell(r, "cm_tn")
            row0 = tn + fp  # true class 0
            row1 = tp + fn  # true class 1
            if row0 > 0:
                tn_pct.append(100.0 * tn / row0)
                fp_pct.append(100.0 * fp / row0)
            if row1 > 0:
                fn_pct.append(100.0 * fn / row1)
                tp_pct.append(100.0 * tp / row1)
                recalls.append(100.0 * tp / row1)
            if (tp + fp) > 0:
                precisions.append(100.0 * tp / (tp + fp))

        mean_tp_cnt = float(np.mean([_cell(r, "cm_tp") for r in results]))
        mean_fn_cnt = float(np.mean([_cell(r, "cm_fn") for r in results]))
        mean_fp_cnt = float(np.mean([_cell(r, "cm_fp") for r in results]))
        mean_tn_cnt = float(np.mean([_cell(r, "cm_tn") for r in results]))

        def _m(xs):
            return float(np.mean(xs)) if xs else 0.0

        print(f"MEAN CONFUSION MATRIX OVER {len(results)} SEEDS "
              f"(row-normalised %, at best-Val-F1 checkpoint)")
        print(sep)
        print(f"  {'':<12}{'Pred 0':>12}{'Pred 1':>12}")
        print(f"  {'True 0':<12}{_m(tn_pct):>11.1f}%{_m(fp_pct):>11.1f}%")
        print(f"  {'True 1':<12}{_m(fn_pct):>11.1f}%{_m(tp_pct):>11.1f}%")
        print("  " + "-" * 36)
        print(f"  Mean counts  TP={mean_tp_cnt:.1f}  FN={mean_fn_cnt:.1f}  "
              f"FP={mean_fp_cnt:.1f}  TN={mean_tn_cnt:.1f}")
        print(f"  Mean class-1 recall    = {_m(recalls):.1f}%")
        print(f"  Mean class-1 precision = {_m(precisions):.1f}%")
        print(sep)


def guess_n_features(train_loader) -> int:
    first_batch = train_loader.dataset[0]
    return first_batch.num_node_features


def cal_weights_model(dataset, use_node_classification=False):
    "Calculate weights for weighted cross entropy loss to address data imbalance"
    labels = []
    if use_node_classification:
        for data in dataset:
            if hasattr(data, 'train_mask'):
                labels += data.y[data.train_mask].tolist()
            else:
                labels += data.y.tolist()
    else:
        for data in dataset:
            labels += data.y.tolist()

    labels_tensor = torch.tensor(labels).squeeze()
    num_classes = labels_tensor.max().item() + 1
    class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()

    n_full = labels_tensor.size(0)
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = n_full / (num_classes * class_counts[i])
        else:
            weights[i] = 0.0
    return weights


def get_explainer(
    explainer: str,
    model: Model,
    explainer_epochs: Optional[int] = None,
    lr_gnnex: Optional[float] = None,
    criterion: Optional[Any] = None,
):
    if explainer == "gnn_explainer":
        return GNNExplainer(model, epochs=explainer_epochs, lr=lr_gnnex, return_type="raw", log=False)
    if explainer == "pgmexplainer":
        return PGMExplainer(model=model, graph=None)
    if explainer == "intgradexplainer":
        return IntegratedGradExplainer(model, criterion)
    raise ValueError(
        '`explainer` must be one of: ("gnn_explainer", "pgmexplainer", "intgradexplainer")'
    )


def train(
    model,
    train_loader,
    optimizer,
    criterion,
    preds,
    explainer,
    use_explanations: bool,
    explainer_iters: int = 5,
    correct_sampling_percent: float = 0.05,
    explanation_topk_thresh: float = 0.25,
    use_node_classification: bool = False,
    grad_clip: float = 0.0,
    edge_dropout_rate: float = 0.0,
):
    losses = []
    for idx, data in enumerate(train_loader):
        model.eval()
        input_data = data.x
        scores = get_default_scores(data, explainer)
        if use_explanations and preds is not None:
            scores = []
            if use_node_classification:
                sampled_correct_indices = sample_correct_indices(
                    pred=preds[data.train_mask],
                    gtruth=data.y[data.train_mask],
                    correct_sampling_percent=correct_sampling_percent,
                )
            else:
                sampled_correct_indices = sample_correct_indices(
                    pred=preds[idx],
                    gtruth=data.y,
                    correct_sampling_percent=correct_sampling_percent,
                )
            scores = get_explainer_scores(
                data=data,
                model=model,
                explainer=explainer,
                sampled_correct_indices=sampled_correct_indices,
                explainer_iters=explainer_iters,
                use_explanations=use_explanations,
                explanation_topk_thresh=explanation_topk_thresh,
                use_node_classification=use_node_classification,
            )
            if isinstance(explainer, PGMExplainer):
                input_data = scores
                scores = None

        model.train()
        optimizer.zero_grad()
        batch_param = None if use_node_classification else data.batch
        train_edge_index, train_scores = apply_edge_dropout(
            data.edge_index, scores, edge_dropout_rate, training=True
        )
        out = model(input_data, train_edge_index, train_scores, batch_param)

        if use_node_classification:
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
        else:
            loss = criterion(out, data.y)

        loss.backward()

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        losses.append(loss)

    avg_loss = sum(l.item() for l in losses) / len(losses)
    return avg_loss


def get_default_scores(data, explainer):
    if isinstance(explainer, GNNExplainer):
        return torch.ones(data.edge_index.shape[1], device=data.edge_index.device)
    if isinstance(explainer, PGMExplainer):
        return None
    if isinstance(explainer, IntegratedGradExplainer):
        return None
    raise ValueError(f"Invalid explainer class passed: '{type(explainer)}'")


def sample_correct_indices(pred, gtruth, correct_sampling_percent: float = 0.5) -> np.ndarray:
    cor_idx = np.where(pred.cpu() == gtruth.cpu())[0]
    samples = max(1, int(correct_sampling_percent * cor_idx.size))
    if cor_idx.shape[0] == 0:
        return np.array([])
    return np.random.choice(cor_idx, samples)


def get_explainer_scores(
    data,
    model,
    explainer,
    sampled_correct_indices,
    explainer_iters: int,
    use_explanations: bool,
    explanation_topk_thresh: float,
    use_node_classification: bool = False,
):
    scores = []

    if use_node_classification:
        if len(sampled_correct_indices) > 0:
            graph_scores = _get_sampled_nodes_or_edge_scores(
                data=data,
                idx=0,
                model=model,
                explainer=explainer,
                explainer_iters=explainer_iters,
                use_explanations=use_explanations,
                explanation_topk_thresh=explanation_topk_thresh,
                use_node_classification=True,
            )
        else:
            graph_scores = _get_remaining_nodes_or_edge_scores(
                data=data, idx=0, explainer=explainer, use_node_classification=True,
            )
        scores = graph_scores
    else:
        for i in range(data.num_graphs):
            if i in sampled_correct_indices:
                graph_scores = _get_sampled_nodes_or_edge_scores(
                    data=data,
                    idx=i,
                    model=model,
                    explainer=explainer,
                    explainer_iters=explainer_iters,
                    use_explanations=use_explanations,
                    explanation_topk_thresh=explanation_topk_thresh,
                    use_node_classification=False,
                )
            else:
                graph_scores = _get_remaining_nodes_or_edge_scores(
                    data=data, idx=i, explainer=explainer, use_node_classification=False,
                )
            scores.extend(graph_scores)

    if isinstance(explainer, GNNExplainer):
        scores = torch.Tensor(scores).to(DEVICE)
    if isinstance(explainer, PGMExplainer):
        scores = torch.tensor(scores).view(data.x.shape[0], 1)
        scores = scores * data.x
    return scores


def _get_sampled_nodes_or_edge_scores(
    data,
    idx,
    model,
    explainer,
    explainer_iters: int,
    use_explanations: bool,
    explanation_topk_thresh: float,
    use_node_classification: bool = False,
):
    graph_data = data if use_node_classification else data[idx]

    if isinstance(explainer, GNNExplainer):
        scores_edges = normalized_explanation_median(
            graph_data, explainer_iters, explainer, use_explanations, explanation_topk_thresh
        )
        return scores_edges.detach().cpu().numpy()
    if isinstance(explainer, PGMExplainer):
        explainer = PGMExplainer(model, graph_data)
        _, p_values, _ = explainer.explain(
            num_samples=1000,
            percentage=10,
            top_node=3,
            p_threshold=0.05,
            pred_threshold=0.1,
        )
        scores_nodes = torch.tensor(
            [1 - j for j in p_values], dtype=graph_data.x.dtype
        )
        return scores_nodes
    if isinstance(explainer, IntegratedGradExplainer):
        batch_param = None if use_node_classification else graph_data.batch
        model_kwargs = {"batch": batch_param, "edge_weight": None}
        exp = explainer.get_explanation_graph(
            edge_index=graph_data.edge_index,
            x=graph_data.x,
            y=graph_data.y,
            forward_kwargs=model_kwargs,
        )
        scores_nodes = normalize(exp.node_imp, dim=0)
        return scores_nodes.detach().cpu()
    raise ValueError(f"Invalid explainer class passed: '{type(explainer)}'")


def _get_remaining_nodes_or_edge_scores(data, idx, explainer, use_node_classification: bool = False):
    graph_data = data if use_node_classification else data[idx]

    if isinstance(explainer, GNNExplainer):
        remaining_edges = torch.ones_like(graph_data.edge_index[1])
        return remaining_edges.detach().cpu().numpy()
    if isinstance(explainer, PGMExplainer):
        return torch.ones(graph_data.x.shape[0], dtype=graph_data.x.dtype, device=graph_data.x.device)
    if isinstance(explainer, IntegratedGradExplainer):
        remaining_nodes = torch.ones(graph_data.x.shape[0], device=graph_data.x.device)
        return remaining_nodes.detach().cpu()
    raise ValueError(f"Invalid explainer class passed: '{type(explainer)}'")


def normalized_explanation_median(
    data,
    iters: int,
    explainer: GNNExplainer,
    use_explanations: bool,
    explanation_topk_thresh: float,
):
    weigths_iters = []
    for it in range(iters):
        _, scores_edges = explainer.explain_graph(
            x=data.x,
            edge_index=data.edge_index,
            edge_weight=None,
            use_explanations=use_explanations,
        )
        weigths_iters.append(scores_edges)

    scores_edges = torch.stack(weigths_iters).median(0)[0]
    scores_edges = (scores_edges - scores_edges.min()) / (
        scores_edges.max() - scores_edges.min()
    )
    thresh = scores_edges.topk(int(explanation_topk_thresh * data.edge_index.shape[1]))[0][-1]
    scores_edges = torch.where(scores_edges >= thresh, 1.0, 0.0)
    return scores_edges


def test(loader, model, use_node_classification=False):
    model.eval()

    if use_node_classification:
        preds_list, labels_list, logits_list = [], [], []
        for data in loader:
            with torch.no_grad():
                out = model(data.x, data.edge_index, None, None)
                pred = out.argmax(dim=1)
                preds_list.append(pred)
                labels_list.append(data.y)
                logits_list.append(out)
        preds  = torch.cat(preds_list)
        labels = torch.cat(labels_list)
        logits = torch.cat(logits_list)
        return preds, labels, None, logits
    else:
        preds, labels, logits = [], [], []
        for data in loader:
            out  = model(data.x, data.edge_index, None, data.batch)
            pred = out.argmax(dim=1)
            preds.append(pred)
            labels.append(data.y)
            logits.append(out)
        preds    = torch.cat(preds)
        labels   = torch.cat(labels)
        logits   = torch.cat(logits)
        accuracy = (preds == labels).float().mean()
        return preds, labels, accuracy, logits


def evaluate_performance(
    train_loader,
    val_loader,
    test_loader,
    model,
    use_node_classification=False,
    num_classes=2,
    tune_threshold: bool = True,
):
    output_train, labels_train, train_acc, logits_train = test(train_loader, model, use_node_classification)
    output_val,   labels_val,   val_acc,   logits_val   = test(val_loader,   model, use_node_classification)
    output_test,  labels_test,  test_acc,  logits_test  = test(test_loader,  model, use_node_classification)

    # Confusion-matrix counts (test set). Filled in the binary branch below;
    # default 0 so multi-class / graph-classification branches stay valid.
    cm_tp = cm_fn = cm_fp = cm_tn = 0.0

    if use_node_classification:
        data = train_loader[0]

        train_pred   = output_train[data.train_mask]
        train_labels = labels_train[data.train_mask]
        train_logits = logits_train[data.train_mask]
        train_acc    = (train_pred == train_labels).float().mean()

        val_pred   = output_val[data.val_mask]
        val_labels = labels_val[data.val_mask]
        val_logits = logits_val[data.val_mask]
        val_acc    = (val_pred == val_labels).float().mean()

        test_pred   = output_test[data.test_mask]
        test_labels = labels_test[data.test_mask]
        test_logits = logits_test[data.test_mask]
        test_acc    = (test_pred == test_labels).float().mean()

        if num_classes == 2:
            train_probs = softmax(train_logits, dim=1)[:, 1]
            val_probs   = softmax(val_logits,   dim=1)[:, 1]
            test_probs  = softmax(test_logits,  dim=1)[:, 1]

            def _safe_auroc(y_true, y_score, fallback):
                # roc_auc_score raises if only one class is present in y_true.
                yt = y_true.cpu()
                if np.unique(yt.numpy()).size < 2:
                    return float(fallback)
                return roc_auc_score(yt, y_score.cpu())

            train_auroc = _safe_auroc(train_labels, train_probs, train_acc.item())
            val_auroc   = _safe_auroc(val_labels,   val_probs,   val_acc.item())
            test_auroc  = _safe_auroc(test_labels,  test_probs,  test_acc.item())
            best_t = 0.5
            if tune_threshold:
                # Threshold is chosen on validation and then applied to test (no leakage).
                # Coarse sweep is intentional because val positives are very few.
                val_labels_np = val_labels.cpu().numpy()
                val_probs_np = val_probs.cpu().numpy()
                if np.unique(val_labels_np).size >= 2:
                    best_val_f1 = -1.0
                    for t in np.linspace(0.05, 0.95, 19):
                        val_pred_t = (val_probs_np >= t).astype(int)
                        score_t = f1_score(
                            val_labels_np,
                            val_pred_t,
                            average='macro',
                            zero_division=0,
                        )
                        if score_t > best_val_f1:
                            best_val_f1 = score_t
                            best_t = float(t)

                val_pred = (val_probs >= best_t).int()
                test_pred = (test_probs >= best_t).int()
                val_acc = (val_pred == val_labels).float().mean()
                test_acc = (test_pred == test_labels).float().mean()

            test_f1_score = f1_score(test_labels.cpu(), test_pred.cpu(), average='macro', zero_division=0)
            val_f1_score = f1_score(val_labels.cpu(), val_pred.cpu(), average='macro', zero_division=0)
            from sklearn.metrics import confusion_matrix as _confusion_matrix
            _cm = _confusion_matrix(test_labels.cpu(), test_pred.cpu())
            if _cm.shape == (2, 2):
                cm_tp = float(_cm[1, 1])
                cm_fn = float(_cm[1, 0])
                cm_fp = float(_cm[0, 1])
                cm_tn = float(_cm[0, 0])
                print(f"[CM] thr={best_t:.2f} | TP={_cm[1,1]} | FN={_cm[1,0]} | FP={_cm[0,1]} | TN={_cm[0,0]} "
                      f"| Solution recall={_cm[1,1]/max(1,_cm[1,0]+_cm[1,1]):.2f}")

        
        
        else:
            train_probs = softmax(train_logits, dim=1)
            val_probs   = softmax(val_logits,   dim=1)
            test_probs  = softmax(test_logits,  dim=1)
            try:
                train_auroc   = roc_auc_score(train_labels.cpu(), train_probs.cpu().numpy(), multi_class='ovr', average='macro')
                val_auroc     = roc_auc_score(val_labels.cpu(),   val_probs.cpu().numpy(),   multi_class='ovr', average='macro')
                test_auroc    = roc_auc_score(test_labels.cpu(),  test_probs.cpu().numpy(),  multi_class='ovr', average='macro')
                test_f1_score = f1_score(test_labels.cpu(), test_pred.cpu(), average='macro')
                val_f1_score = f1_score(val_labels.cpu(), val_pred.cpu(), average='macro', zero_division=0)

            except ValueError as e:
                print(f"Warning: AUROC calculation failed ({e}), using accuracy as fallback")
                train_auroc   = train_acc.item()
                val_auroc     = val_acc.item()
                test_auroc    = test_acc.item()
                test_f1_score = f1_score(test_labels.cpu(), test_pred.cpu(), average='macro')
                val_f1_score = f1_score(val_labels.cpu(), val_pred.cpu(), average='macro', zero_division=0)


        output_for_next_epoch = output_train

    else:
        if num_classes == 2:
            train_probs = softmax(logits_train, dim=1)[:, 1]
            val_probs   = softmax(logits_val,   dim=1)[:, 1]
            test_probs  = softmax(logits_test,  dim=1)[:, 1]
            train_auroc = roc_auc_score(labels_train.cpu(), train_probs.cpu())
            val_auroc   = roc_auc_score(labels_val.cpu(),   val_probs.cpu())
            test_auroc  = roc_auc_score(labels_test.cpu(),  test_probs.cpu())
        else:
            train_probs = softmax(logits_train, dim=1)
            val_probs   = softmax(logits_val,   dim=1)
            test_probs  = softmax(logits_test,  dim=1)
            train_auroc = roc_auc_score(labels_train.cpu(), train_probs.cpu().numpy(), multi_class='ovr', average='macro')
            val_auroc   = roc_auc_score(labels_val.cpu(),   val_probs.cpu().numpy(),   multi_class='ovr', average='macro')
            test_auroc  = roc_auc_score(labels_test.cpu(),  test_probs.cpu().numpy(),  multi_class='ovr', average='macro')

        test_f1_score = f1_score(labels_test.cpu(), output_test.cpu())
        val_f1_score = f1_score(labels_val.cpu(), output_val.cpu())
        output_for_next_epoch = output_train

    performance = PerformanceResults(
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        train_auroc=train_auroc,
        val_auroc=val_auroc,
        test_auroc=test_auroc,
        test_f1_score=test_f1_score,
        val_f1_score=val_f1_score,
        cm_tp=cm_tp,
        cm_fn=cm_fn,
        cm_fp=cm_fp,
        cm_tn=cm_tn,
    )
    return output_for_next_epoch, performance


def log_progress(
    epoch: int,
    avg_loss: float,
    performance: PerformanceResults,
    oversmoothing: float,
    convergence_file_stem: str,
    epoch_start_time: float,
):
    metrics = {
        "Epoch": epoch,
        "Train Loss": avg_loss,
        "Train Acc": performance.train_acc,
        "Test Acc": performance.test_acc,
        "Train AUROC": performance.train_auroc,
        "Val AUROC": performance.val_auroc,
        "Test AUROC": performance.test_auroc,
        "Test F1": performance.test_f1_score,
        "Val F1": performance.val_f1_score,
        "Val Acc": performance.val_acc,
        "Oversmoothing": oversmoothing,
    }
    metrics_formatted = [
        f"{name}: {value:.4f}" for name, value in metrics.items()
    ]
    progress_string = ", ".join(metrics_formatted)
    # Print every epoch so Val F1 (the checkpoint-selection metric) is always visible.
    print(progress_string)
    with open(CONVERGENCE_DIR / f"{convergence_file_stem}.csv", "a") as f:
        f.write(progress_string + "\n")


def calculate_oversmoothing(model, dataset_loader, seed, batch_size, best_model_path):
    graph_embedding = torch.Tensor()
    graph_label     = torch.Tensor()
    model.eval()
    dataset = dataset_loader(seed=seed, batch_size=batch_size, split_train_val_test=False)
    model.load_state_dict(torch.load(best_model_path))
    for data in dataset:
        embedding = model.embed(data.x, data.edge_index, None, data.batch)
        graph_embedding = torch.cat((graph_embedding, embedding))
        graph_label     = torch.cat((graph_label, data.y))
    return calculate_gdr(graph_label, graph_embedding)


def calculate_gdr(label, embedding):
    X_labels = []
    for i in label.unique():
        X_label = embedding[label == i].data.cpu().numpy()
        h_norm  = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.0] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)

    dis_intra = 0.0
    for i in label.unique():
        x2 = np.sum(np.square(X_labels[int(i)]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[int(i)], X_labels[int(i)].T)
        dis_intra += np.mean(dists)
    dis_intra /= label.unique().shape[0]

    dis_inter = 0.0
    for i in range(label.unique().shape[0] - 1):
        for j in range(i + 1, label.unique().shape[0]):
            x2_i = np.sum(np.square(X_labels[int(i)]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[int(j)]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter += np.mean(dists)
    num_inter = float(label.unique().shape[0] * (label.unique().shape[0] - 1) / 2)
    dis_inter /= num_inter

    return dis_inter / dis_intra

# endregion


if __name__ == "__main__":
    # Register --all_seeds before parsing so it is captured cleanly.
    argument_parser.add_argument(             # NEW
        "--all_seeds",
        action="store_true",
        default=False,
        help=(
            f"Run training once per seed in {ALL_SEEDS} and report "
            "mean ± std across seeds at the end."
        ),
    )

    args = argument_parser.parse_args()
    args_dict = vars(args)
    all_label_cols = args_dict.pop('all_label_cols', False)
    all_seeds      = args_dict.pop('all_seeds',      False)  # NEW

    def _run_seeds(base_kwargs: dict, label: str = "") -> None:
        """Run main() for every seed and print aggregated statistics."""
        results, ran_seeds = [], []
        for seed in ALL_SEEDS:
            print(f"\n{'='*80}")
            print(f"Seed {seed}" + (f"  [{label}]" if label else ""))
            print("=" * 80)
            perf = main(**{**base_kwargs, "seed": seed})
            if perf is not None:
                results.append(perf)
                ran_seeds.append(seed)
            else:
                print(f"  Warning: no checkpoint was saved for seed {seed} – skipping.")
        if results:
            print_seed_aggregation(results, seeds=ran_seeds, label=label)
        else:
            print("No valid results collected – nothing to aggregate.")

    if all_label_cols and all_seeds:               # NEW branch: both flags
        for col in MULTILINGUAL_LABEL_COLS:
            print(f"\n{'#'*80}")
            print(f"Label column: {col}")
            print("#" * 80)
            _run_seeds({**args_dict, "label_col": col}, label=col)

    elif all_label_cols:                           # original behaviour
        for col in MULTILINGUAL_LABEL_COLS:
            print(f"\n{'='*80}")
            print(f"Running training for label column: {col}")
            print("=" * 80)
            main(**{**args_dict, "label_col": col})

    elif all_seeds:                                # NEW branch: seeds only
        _run_seeds(args_dict)

    else:                                          # original single-run behaviour
        main(**args_dict)
