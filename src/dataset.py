"""
dataset.py
----------
Dataset classes and data loading utilities for:
  1) DCASE 2023 Task 2 development dataset
  2) MIMII dataset (additional dataset for Assignment 3)

Handles file discovery, label parsing from filenames, and
domain (source/target) separation.
"""

import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .feature_extraction import extract_features_from_file


class DCASEDataset(Dataset):
    """
    PyTorch Dataset for DCASE 2023 Task 2.

    Each .wav file in the dataset follows the naming convention:
       section_<ID>_<domain>_<split>_<label>_<index>_<attrs>.wav

    e.g. section_00_source_train_normal_0001_<attrs>.wav

    This class lazily loads and extracts features from individual files.
    """

    def __init__(self, file_list, sr=16000, n_fft=1024,
                 hop_length=512, n_mels=128, n_frames=5,
                 duration=10):
        """
        Parameters
        ----------
        file_list : list of str
            Full paths to .wav files.
        """
        self.file_list = file_list
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.duration = duration

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        vectors = extract_features_from_file(
            fpath, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
            n_frames=self.n_frames, duration=self.duration
        )
        return torch.tensor(vectors, dtype=torch.float32)


class PrecomputedDataset(Dataset):
    """
    Dataset wrapping pre-extracted feature vectors stored as a numpy array.
    Used during training after features have been computed and cached.
    """

    def __init__(self, features):
        """
        Parameters
        ----------
        features : np.ndarray
            Shape (N, input_dim) array of feature vectors.
        """
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def parse_filename(filename):
    """
    Parse a DCASE 2023 Task 2 filename to extract metadata.

    Example filename:
        section_00_source_test_anomaly_0003_voltage_... .wav

    Returns
    -------
    info : dict with keys:
        section_id, domain, split, label, index
    """
    basename = os.path.basename(filename)
    parts = basename.replace(".wav", "").split("_")

    info = {
        "section_id": parts[1] if len(parts) > 1 else "00",
        "domain": parts[2] if len(parts) > 2 else "source",
        "split": parts[3] if len(parts) > 3 else "train",
        "label": parts[4] if len(parts) > 4 else "normal",
        "index": parts[5] if len(parts) > 5 else "0000",
    }
    return info


def get_machine_type_files(data_root, machine_type, split="train"):
    """
    Collect all .wav file paths for a given machine type and split.

    Parameters
    ----------
    data_root : str
        Root directory, e.g. "data/dcase2023t2/dev_data"
    machine_type : str
        e.g. "fan", "ToyCar", etc.
    split : str
        "train" or "test"

    Returns
    -------
    files : list of str
        Sorted list of .wav file paths.
    """
    search_dir = os.path.join(data_root, machine_type, split)
    if not os.path.isdir(search_dir):
        # Try alternate structure (some datasets use section-level dirs)
        search_dir = os.path.join(data_root, machine_type)

    pattern = os.path.join(search_dir, "*.wav")
    files = sorted(glob.glob(pattern))

    if split == "train":
        files = [f for f in files if "train" in os.path.basename(f)]
    elif split == "test":
        files = [f for f in files if "test" in os.path.basename(f)]

    return files


def separate_by_domain(file_list):
    """
    Split file list into source-domain and target-domain files.

    Returns
    -------
    source_files, target_files : list of str, list of str
    """
    source_files = [f for f in file_list
                    if "source" in os.path.basename(f)]
    target_files = [f for f in file_list
                    if "target" in os.path.basename(f)]
    return source_files, target_files


def get_labels(file_list):
    """
    Extract binary labels from file list.
    normal -> 0, anomaly -> 1

    Returns
    -------
    labels : np.ndarray of int
    """
    labels = []
    for f in file_list:
        basename = os.path.basename(f)
        if "anomaly" in basename:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels, dtype=int)


def build_dataloader(features, batch_size=512, shuffle=True):
    """
    Convenience function: wrap a feature array into a DataLoader.

    Parameters
    ----------
    features : np.ndarray
        Shape (N, input_dim).
    batch_size : int
    shuffle : bool

    Returns
    -------
    loader : DataLoader
    """
    dataset = PrecomputedDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, drop_last=False)
    return loader
