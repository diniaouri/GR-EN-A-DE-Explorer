"""
GR-EN-A-DE Dataset Preprocessing Module

================================================================================
DATASET STRUCTURE AND NARRATIVE ANALYSIS
================================================================================

This module handles preprocessing of multilingual extremist narrative datasets.
Each dataset contains social media posts/texts annotated with narrative elements
that help us understand how extremist content is structured.

================================================================================
HOW DATASETS ARE MATCHED AND USED
================================================================================

**Embeddings Matching:**
Each dataset class computes text embeddings using a multilingual transformer:
- All datasets use the same embedding model (paraphrase-multilingual-MiniLM-L12-v2)
- This ensures embeddings are comparable across languages
- Embeddings capture semantic meaning regardless of language

**Context Attributes Matching:**
Different datasets have slightly different attributes, but core ones overlap:
- French, German datasets: Same attribute structure (16 columns)
- Cypriot, Slovene datasets: Extended structure with additional fields
- English datasets (Toxigen, LGBT, Migrants): Simplified structure (In-Group, Out-Group)

**Graph Construction:**
When building the graph, we match posts based on:
1. **Semantic similarity** (embedding distance) - captures content similarity
2. **Attribute co-membership** (shared narrative elements) - captures structural similarity

================================================================================
USAGE PIPELINE
================================================================================

1. **Load Dataset**:
   ```python
   dataset = MultilingualENCorpusFrenchDataset(experiment_nb=1)
   ```

2. **Get Components**:
   ```python
   embeddings, num_features, labels, num_classes, train_mask, val_mask, test_mask, adj = dataset.get_dataset()
   ```

3. **Access Context**:
   ```python
   # Get narrative attributes for specific posts
   attributes = dataset.get_context_attributes(
       indices=[0, 5, 10],
       columns=['In-Group', 'Out-group', 'Topic']
   )
   ```

4. **Build Context-Aware Graph**:
   ```python
   # Use attributes to add meaningful edges
   context_adj = make_context_adjacency(dataset.data, context_columns)
   ```

================================================================================
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional

import emoji
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

class PreprocessedDataset(ABC):
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.experiment: int = experiment_nb
        self.embeddings_path: Optional[str] = embeddings_path

        # self.data is set by subclasses BEFORE this constructor is called!
        if not skip_embeddings:
            # Use multilingual model for embeddings across different languages
            self.calc_embeddings_if_not_already("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            print("Embeddings are loaded")
            self.embeddings = torch.from_numpy(self.embeddings)
        else:
            self.embeddings = None

    def _get_embeddings_path(self) -> str:
        if self.embeddings_path is not None:
            return self.embeddings_path
        else:
            return f"./embeddings/{self.dataset_name}_embeddings_exp{self.experiment}.npy"

    def save_embeddings(self) -> None:
        path = self._get_embeddings_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.embeddings)

    def calc_embeddings_if_not_already(self, model_name):
        path = self._get_embeddings_path()
        print(f"Using embeddings path: {path}")
        if not os.path.exists(path):
            self.embeddings = self.calc_embeddings(model_name)
            self.save_embeddings()
        else:
            self.embeddings = np.load(path)

    def calc_embeddings(self, model_name: str) -> np.ndarray:
        """Calculate embeddings for the dataset using a sentence transformer model.
        
        This method uses a pre-trained transformer model to generate embeddings
        for text data in the dataset. It automatically handles different column
        names across datasets.
        
        Args:
            model_name: Name of the sentence transformer model to use
            
        Returns:
            np.ndarray: Embeddings for all texts in the dataset
        """
        model = SentenceTransformer(model_name, trust_remote_code=True)
        if "text" in self.data.columns and getattr(self, 'dataset_name', None) == "Toxigen":
            sentences = self.data["text"].tolist()
        else:
            if "Text" in self.data.columns:
                sentences = self.data["Text"].tolist()
            elif "text" in self.data.columns:
                sentences = self.data["text"].tolist()
            else:
                raise ValueError("No valid text column found for embeddings.")
        embeddings: np.ndarray = model.encode(
            sentences,
            batch_size=1,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def get_dataset(self) -> Tuple[torch.Tensor, int, List[Any], int, List[Any], List[Any], List[Any], List[Any]]:
        return (
            self.embeddings,
            self.embeddings.shape[1] if self.embeddings is not None else 0,
            getattr(self, 'labels', []),
            len(getattr(self, 'labels', [])),
            getattr(self, 'train_mask', []),
            getattr(self, 'val_mask', []),
            getattr(self, 'test_mask', []),
            getattr(self, 'adjacency_matrix', [])
        )

    @abstractmethod
    def clean_up_dataset(self) -> pd.DataFrame:
        pass

    def clean_up_text(self, text: str) -> str:
        """Clean and normalize text data.
        
        Performs the following cleaning operations:
        - Removes URLs
        - Converts emojis to text representations
        - Removes mentions (@username)
        - Converts hashtags to plain text
        - Removes retweet indicators
        - Normalizes whitespace
        - Converts to lowercase
        
        Args:
            text: The text to clean
            
        Returns:
            str: Cleaned and normalized text
        """
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = emoji.demojize(text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return text

def remove_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[0]

# --- Datasets ---

# ---- FRENCH ----
class MultilingualENCorpusFrenchDataset(PreprocessedDataset):
    """Dataset for French extremist narrative corpus analysis.
    
    This dataset contains annotated French language texts with various context
    attributes related to extremist narratives, including in-group/out-group
    dynamics, perceived threats, and emotional responses.
    """
    CONTEXT_COLUMNS_FULL = [
        "Topic", "In-Group", "Out-group", "Initiating Problem", "Intolerance", "Superiority of in-group",
        "Hostility to out-group (e.g. verbal attacks, belittlement, instilment of fear, incitement to violence)",
        "Polarization/Othering", "Perceived Threat", "Setting", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability", "Conspiracy Theories", "Irony/Humor"
    ]
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "Multilingual_EN_Corpus_FRENCH"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/Multilingual_EN_Corpus_DATA_FRENCH.xlsx",
                           header=4, usecols="B, AH:BA")
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        df.to_csv("ty_1st_annotator.csv")
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.CONTEXT_COLUMNS_FULL
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Requested context columns not found in dataset: {missing_cols}")
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.data.iloc[i][col])
            out.append(tuple(entry))
        return out

# ---- GERMAN ----
class MultilingualENCorpusGermanDataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = MultilingualENCorpusFrenchDataset.CONTEXT_COLUMNS_FULL
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "Multilingual_EN_Corpus_GERMAN"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/Multilingual_EN_Corpus_DATA_GERMAN.xlsx",
                           header=4, usecols="B, AH:AZ")
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        df.to_csv("ty_german_1st_annotator.csv")
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        return MultilingualENCorpusFrenchDataset.get_context_attributes(self, indices, columns)

# ---- CYPRIOT ----
class MultilingualENCorpusCypriotDataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = [
        "Topic", "Tone of Post", "In-Group", "Out-group", "Narrator",
        "Intolerance", "Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)",
        "Polarization/Othering", "Perceived Threat", "Character(s)", "Setting",
        "Initiating Problem", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability",
        "Conspiracy Theories", "Irony/Humor"
    ]
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "Multilingual_EN_Corpus_CYPRIOT"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        # Try all header rows until you find the real header
        found = False
        for hdr in range(0, 15):
            df = pd.read_excel("./datasets/Multilingual_EN_Corpus_DATA_CYPRIOT.xlsx", header=hdr)
            colnames = list(df.columns)
            if "Tweet, text" in colnames and "In-Group" in colnames and "Out-group" in colnames:
                found = True
                print(f"USING header={hdr}")
                break
        if not found:
            print("DEBUG: Could not find header row with required columns. Here are some column samples:")
            for hdr in range(0, 15):
                df = pd.read_excel("./datasets/Multilingual_EN_Corpus_DATA_CYPRIOT.xlsx", header=hdr)
                print(f"header={hdr}: {list(df.columns)[:20]}")
            raise ValueError("Could not find header row with required columns. Please check your Excel file.")

        df = df.rename(columns={"Tweet, text": "Text"})
        select_cols = ["Text"] + self.CONTEXT_COLUMNS_FULL
        missing = [col for col in select_cols if col not in df.columns]
        if missing:
            print("DEBUG COLUMNS:", df.columns)
            print("DEBUG: Check for typos or extra spaces in your Excel columns. Here are the missing:", missing)
            raise ValueError(f"Missing columns: {missing}. Columns are: {df.columns}")
        df = df[select_cols]
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.CONTEXT_COLUMNS_FULL
        return [tuple(self.data.iloc[i][col] for col in columns) for i in indices]

# ---- SLOVENE ----
class MultilingualENCorpusSloveneDataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = [
        "Topic", "Tone of Post", "In-Group", "Out-group", "Narrator",
        "Intolerance", "Hostility to out-group", "Polarization/Othering", "Perceived Threat",
        "Character(s)", "Setting", "Initiating Problem", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability", "Conspiracy Theories", "Irony/Humor"
    ]
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "Multilingual_EN_Corpus_SLOVENE"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)
    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/Multilingual_EN_Corpus_DATA_SLOVENE.xlsx", header=4, usecols="E, AJ:BB")
        df = df.rename(columns={"Tweet, Text": "Text"})
        required_cols = ["Text", "In-Group", "Out-group"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print("DEBUG COLUMNS:", df.columns)
            raise ValueError(f"Missing Slovene columns: {missing}. Columns are: {df.columns}")
        df = df.dropna(subset=required_cols)
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        return df
    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.CONTEXT_COLUMNS_FULL
        return [tuple(self.data.iloc[i][col] for col in columns) for i in indices]

# ---- TOXIGEN, LGBTEn, MigrantsEn as before ----
class ToxigenDataset(PreprocessedDataset):
    DEFAULT_CONTEXT_COLUMNS = [
        "In-Group", "Out-group"
    ]
    def __init__(self, experiment_nb: int = 1, csv_path: Optional[str] = None, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name = "Toxigen"
        self.csv_path = csv_path or "datasets/Toxigen.csv"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, encoding="utf-8")
        if "text" in df.columns:
            df["text"] = df["text"].apply(self.clean_up_text)
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.DEFAULT_CONTEXT_COLUMNS
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Requested context columns not found in dataset: {missing_cols}")
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.data.iloc[i][col])
            out.append(tuple(entry))
        return out

class LGBTEnDataset(PreprocessedDataset):
    """Dataset for LGBT-related extremist narrative analysis in English."""

    def __init__(
        self,
        experiment_nb: int = 1,
        embeddings_path: Optional[str] = None,
        skip_embeddings: bool = False,
        csv_path: Optional[str] = None
    ):
        self.dataset_name = "LGBTEn"
        self.csv_path = csv_path or "datasets/LGBTEn.csv"
        self.label_col = "annotation_type"
        self.context_cols = [
            "In-Group", "Out-group"
        ]
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb=experiment_nb, embeddings_path=embeddings_path, skip_embeddings=skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        """Clean and prepare the LGBT dataset for processing.
        
        Performs the following operations:
        1. Loads CSV data from the specified path
        2. Cleans text content (removes URLs, normalizes whitespace, etc.)
        3. Encodes string labels to numeric format using LabelEncoder
        
        Label Encoding Strategy:
        - If labels are strings: Uses sklearn's LabelEncoder for numeric conversion
        - If labels are already numeric: Keeps them as-is
        - If no label column exists: Sets labels to empty list
        
        Returns:
            Cleaned DataFrame ready for embedding computation
        """
        df = pd.read_csv(self.csv_path)
        # Clean text data if text column exists
        if "text" in df.columns:
            df["text"] = df["text"].apply(self.clean_up_text)
        # Encode labels if annotation type is present
        if self.label_col in df.columns:
            labels_raw = df[self.label_col].tolist()
            if any(isinstance(lbl, str) for lbl in labels_raw):
                encoder = LabelEncoder()
                self.labels = encoder.fit_transform(labels_raw)
                self.label_encoder = encoder
            else:
                self.labels = labels_raw
        else:
            self.labels = []
        return df 

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        columns = columns or self.context_cols
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Requested context columns not found in dataset: {missing_cols}")
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.data.iloc[i][col])
            out.append(tuple(entry))
        return out

class MigrantsEnDataset(PreprocessedDataset):
    """Dataset for migrants-related extremist narrative analysis in English."""

    def __init__(
        self,
        experiment_nb: int = 1,
        embeddings_path: Optional[str] = None,
        skip_embeddings: bool = False,
        csv_path: Optional[str] = None
    ):
        self.dataset_name = "MigrantsEn"
        self.csv_path = csv_path or "datasets/MigrantsEn.csv"
        self.label_col = "annotation_type"
        self.context_cols = [
            "In-Group", "Out-group"
        ]
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb=experiment_nb, embeddings_path=embeddings_path, skip_embeddings=skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        """Clean and prepare the migrants dataset for processing.
        
        Performs the following operations:
        1. Loads CSV data from the specified path
        2. Cleans text content (removes URLs, normalizes whitespace, etc.)
        3. Encodes string labels to numeric format using LabelEncoder
        
        Label Encoding Strategy:
        - If labels are strings: Uses sklearn's LabelEncoder for numeric conversion
        - If labels are already numeric: Keeps them as-is
        - If no label column exists: Sets labels to empty list
        
        Returns:
            Cleaned DataFrame ready for embedding computation
        """
        df = pd.read_csv(self.csv_path)
        # Clean text data if text column exists
        if "text" in df.columns:
            df["text"] = df["text"].apply(self.clean_up_text)
        # Encode labels if annotation type is present
        if self.label_col in df.columns:
            labels_raw = df[self.label_col].tolist()
            if any(isinstance(lbl, str) for lbl in labels_raw):
                encoder = LabelEncoder()
                self.labels = encoder.fit_transform(labels_raw)
                self.label_encoder = encoder
            else:
                self.labels = labels_raw
        else:
            self.labels = []
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        columns = columns or self.context_cols
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Requested context columns not found in dataset: {missing_cols}")
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.data.iloc[i][col])
            out.append(tuple(entry))
        return out
