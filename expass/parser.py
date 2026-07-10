#!/usr/bin/env python3

import argparse
argument_parser = argparse.ArgumentParser()

# Core arguments --------------------------------

argument_parser.add_argument(
    "--grenade-embeddings", type=str, required=True,
    help="Path to GRENADE embeddings .npy file"
)
argument_parser.add_argument(
    "--grenade-adjacency", type=str, required=True,
    help="Path to GRENADE adjacency .pkl file"
)
argument_parser.add_argument(
    "--grenade-labels", type=str, default=None,
    help="Path to labels .npy file for GRENADE data (optional, will try to auto-load)"
)
argument_parser.add_argument(
    "--grenade-exp-nb", type=int, default=None,
    help="GRENADE experiment number (reads from experiment_params.csv)"
)
argument_parser.add_argument(
    "--dataset-name", type=str, default="toxigen",
    choices=["toxigen", "lgbten", "migrantsen", "french", "german", "cypriot", "slovene"],
    help="GRENADE dataset name for auto-loading labels (default: toxigen)"
)
argument_parser.add_argument(
    "--label-col", type=str, default=None,
    help="Override label column from experiment_params.csv (e.g., 'Topic', 'Intolerance', 'Solution')"
)
argument_parser.add_argument(
    "--all-label-cols", action="store_true", default=False,
    help=(
        "Run training for every multilingual target column one by one "
        "(Topic, Initiating Problem, Intolerance, Hostility to out-group, "
        "Polarization/Othering, Perceived Threat, Solution). "
        "Results are saved in separate directories and files named after each column."
    )
)
argument_parser.add_argument(
    "--arch", type=str, default="gcn",
    choices=("gcn", "graphconv", "leconv", "graphsage", "ginconv", "sageconv"),
    help="GNN Architectures",
)
argument_parser.add_argument(
    "--explainer", type=str, default="gnn_explainer",
    choices=("gnn_explainer", "pgmexplainer", "intgradexplainer"),
    help="Explainer method to use. Ignored if flag --vanilla_mode is used",
)
argument_parser.add_argument("--weight_decay", type=float, default=0.001)
# Architecture hyperparams --------------------------------

argument_parser.add_argument(
    "--num_layers", type=int, default=3, 
    help="Number of GNN layers"
)
argument_parser.add_argument(
    "--hidden_dim", type=int, default=256,
    help="Hidden dimension size of the GNN"
)
argument_parser.add_argument(
    "--dropout", type=float, default=0.3,
    help="Dropout rate"
)
argument_parser.add_argument(
    "--jk_mode", type=str, default="cat",
    choices=("cat", "max", "lstm", "none"),
    help="Jumping-knowledge aggregation mode"
)
argument_parser.add_argument(
    "--activation", type=str, default="elu",
    choices=("relu", "elu", "prelu", "tanh"),
    help="Activation function"
)
argument_parser.add_argument(
    "--batch_size", type=int, default=200, 
    help="Batch size for the dataloader"
)
argument_parser.add_argument(
    "--seed", type=int, default=912, 
    help="Random seed."
)

# Training hyperparams --------------------------------

argument_parser.add_argument(
    "--epochs", default=150, type=int,
    help = "Number of epochs of the top-level loop"
)
argument_parser.add_argument(
    "--lr_gnn", default=0.01, type=float,
    help = "Learning rate of the GNN"
)
argument_parser.add_argument(
    "--lr_gnnex", default=0.01, type=float,
    help = "Learning rate of the GNN Explainer"
)
argument_parser.add_argument(
    "--explainer_iters", default=5, type=int,
    help="Number of times the explainer runs per epoch (median is taken). Default: 5"
)
argument_parser.add_argument(
    "--explainer_epochs", default=200, type=int,
    help="Number of epochs for the GNNExplainer optimization. Default: 200"
)
argument_parser.add_argument(
    "--correct_sampling_percent", default=0.4, type=float,
    help="Fraction of correctly predicted nodes used to generate explanations. Default: 0.4"
)
argument_parser.add_argument(
    "--explanation_topk_thresh", default=0.3, type=float,
    help="Fraction of edges to KEEP after explanation scoring (e.g. 0.3 = keep top 30%%, drop 70%%). Default: 0.3"
)
argument_parser.add_argument(
    "--explanations_lag", default=20, type=int,
    help="Epoch after which explanation-guided training starts. Default: 20"
)
argument_parser.add_argument(
    "--model_saving_lag", type=int, default=25, 
    help="Number of epochs to wait before saving model checkpoints."
)
argument_parser.add_argument("--focal_gamma", type=float, default=2.0)
argument_parser.add_argument("--label_smoothing", type=float, default=0.05)
argument_parser.add_argument("--warmup_epochs", type=int, default=20)
argument_parser.add_argument("--min_lr", type=float, default=1e-5)
argument_parser.add_argument("--grad_clip", type=float, default=1.0)
argument_parser.add_argument("--edge_dropout", type=float, default=0.1)
argument_parser.add_argument("--patience", type=int, default=50)
argument_parser.add_argument("--min_epochs", type=int, default=50)
argument_parser.add_argument("--oversample_minority", action="store_true", default=False)
argument_parser.add_argument(
    "--no_tune_threshold",
    action="store_false",
    dest="tune_threshold",
    default=True,
    help="Disable validation-tuned decision threshold calibration for binary node classification.",
)

# Boolean flags ---------------------------------

argument_parser.add_argument(
    "--vanilla_mode", action="store_true", default=False,
    help="Whether to run training in vanilla mode (i.e. not using explanation)",
)
argument_parser.add_argument(
    "--device", type=str, default=None,
    choices=("cpu", "cuda"),
    help="Device to run on: 'cpu' or 'cuda'. Auto-detects GPU if not specified."
)
