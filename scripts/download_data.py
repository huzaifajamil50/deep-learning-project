"""
download_data.py
----------------
Helper script to download DCASE 2023 Task 2 development dataset
from Zenodo and the MIMII dataset for additional experiments.

Datasets are downloaded, extracted, and placed in the expected
directory structure under data/.

Usage:
    python scripts/download_data.py --dataset dcase2023
    python scripts/download_data.py --dataset mimii
    python scripts/download_data.py --dataset all
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from tqdm import tqdm


# DCASE 2023 Task 2 Development Dataset (Zenodo)
DCASE2023_URLS = {
    "bearing":  "https://zenodo.org/record/7882613/files/dev_bearing.zip",
    "fan":      "https://zenodo.org/record/7882613/files/dev_fan.zip",
    "gearbox":  "https://zenodo.org/record/7882613/files/dev_gearbox.zip",
    "slider":   "https://zenodo.org/record/7882613/files/dev_slider.zip",
    "ToyCar":   "https://zenodo.org/record/7882613/files/dev_ToyCar.zip",
    "ToyTrain": "https://zenodo.org/record/7882613/files/dev_ToyTrain.zip",
    "valve":    "https://zenodo.org/record/7882613/files/dev_valve.zip",
}

# MIMII Dataset (for additional experiments in Assignment 3)
MIMII_URLS = {
    "fan":    "https://zenodo.org/record/3384388/files/dev_data_fan.zip",
    "pump":   "https://zenodo.org/record/3384388/files/dev_data_pump.zip",
    "slider": "https://zenodo.org/record/3384388/files/dev_data_slider.zip",
    "valve":  "https://zenodo.org/record/3384388/files/dev_data_valve.zip",
}


class DownloadProgressBar(tqdm):
    """Progress bar hook for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, dest_path):
    """Download a file from a URL with a progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with DownloadProgressBar(unit="B", unit_scale=True,
                              miniters=1, desc=os.path.basename(dest_path)) as t:
        urllib.request.urlretrieve(url, filename=dest_path,
                                   reporthook=t.update_to)


def extract_zip(zip_path, extract_to):
    """Extract a zip file to the given directory."""
    print(f"  Extracting {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Done.")


def download_dcase2023(data_root="data/dcase2023t2/dev_data"):
    """Download and extract DCASE 2023 Task 2 development data."""
    print("\n=== Downloading DCASE 2023 Task 2 Development Dataset ===")
    os.makedirs(data_root, exist_ok=True)

    for mtype, url in DCASE2023_URLS.items():
        zip_name = f"dev_{mtype}.zip"
        zip_path = os.path.join(data_root, zip_name)

        if os.path.isdir(os.path.join(data_root, mtype)):
            print(f"  {mtype} already exists, skipping download.")
            continue

        print(f"\n  Downloading {mtype} ...")
        try:
            download_file(url, zip_path)
            extract_zip(zip_path, data_root)
            os.remove(zip_path)    # clean up zip after extraction
        except Exception as e:
            print(f"  [ERROR] Failed to download {mtype}: {e}")
            print(f"  You can manually download from: {url}")


def download_mimii(data_root="data/mimii"):
    """Download and extract MIMII dataset."""
    print("\n=== Downloading MIMII Dataset ===")
    os.makedirs(data_root, exist_ok=True)

    for mtype, url in MIMII_URLS.items():
        zip_name = f"dev_data_{mtype}.zip"
        zip_path = os.path.join(data_root, zip_name)

        if os.path.isdir(os.path.join(data_root, mtype)):
            print(f"  {mtype} already exists, skipping download.")
            continue

        print(f"\n  Downloading {mtype} ...")
        try:
            download_file(url, zip_path)
            extract_zip(zip_path, data_root)
            os.remove(zip_path)
        except Exception as e:
            print(f"  [ERROR] Failed to download {mtype}: {e}")
            print(f"  You can manually download from: {url}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for DCASE 2023 Task 2 ASD"
    )
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["dcase2023", "mimii", "all"],
                        help="Which dataset to download")
    args = parser.parse_args()

    if args.dataset in ("dcase2023", "all"):
        download_dcase2023()
    if args.dataset in ("mimii", "all"):
        download_mimii()

    print("\n=== Download complete ===")


if __name__ == "__main__":
    main()
