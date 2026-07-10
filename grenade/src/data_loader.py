import warnings
import pickle as pkl
import sys
import os
from preprocessing import (
    ToxigenDataset,
    LGBTEnDataset,
    MigrantsEnDataset,
    MultilingualENCorpusGermanDataset,
    MultilingualENCorpusFrenchDataset,
    MultilingualENCorpusCypriotDataset,
    MultilingualENCorpusSloveneDataset,
)
import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

def load_data(dataset_name="toxigen", csv_path=None, experiment_nb=1):
    """
    Load data for the specified dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "toxigen":
        return ToxigenDataset(experiment_nb=experiment_nb, csv_path=csv_path, skip_embeddings=True)
    elif dataset_name == "lgbten":
        return LGBTEnDataset(csv_path=csv_path, skip_embeddings=True)
    elif dataset_name == "migrantsen":
        return MigrantsEnDataset(csv_path=csv_path, skip_embeddings=True)
    elif dataset_name == "multilingualencorpusgerman":
        return MultilingualENCorpusGermanDataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "multilingualencorpusfrench":
        return MultilingualENCorpusFrenchDataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "multilingualencorpuscypriot":
        return MultilingualENCorpusCypriotDataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "multilingualencorpusslovene":
        return MultilingualENCorpusSloveneDataset(experiment_nb=experiment_nb, skip_embeddings=True)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
 