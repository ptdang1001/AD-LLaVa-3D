import os
import sys
import h5py
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
import torch.nn.functional as F
import pysnooper

console = Console()


def _read_ids(f: h5py.File) -> List[str]:
    if "ids" in f:
        raw = f["ids"][:]
        try:
            it = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        except Exception:
            it = list(raw)
        return [
            (x.decode("utf-8") if hasattr(x, "decode") else str(x)).strip() for x in it
        ]
    # Fallback to dataset keys (excluding meta and ids)
    return [k for k in f.keys() if k not in ("meta", "ids")]


def load_mri_h5_to_dict(
    h5_path: Path | str,
    ids: Optional[List[str]] = None,
    squeeze_singleton: bool = True,
) -> Dict[str, np.ndarray]:
    """Load an MRI H5 into a dict mapping SampleID -> numpy array.

    - If ids is None, loads all datasets (using 'ids' dataset if present; otherwise keys)
    - If arrays are 4D with trailing size 1, and squeeze_singleton=True, squeezes to 3D
    """
    h5_path = Path(h5_path)
    result: Dict[str, np.ndarray] = {}
    with h5py.File(str(h5_path), "r") as f:
        sid_list = list(ids) if ids is not None else _read_ids(f)
        for sid in sid_list:
            if sid not in f:
                print("SampleID %s not found in %s, skipping", sid, h5_path)
                continue
            arr = f[sid][...]
            if squeeze_singleton and arr.ndim == 4 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            # Ensure float32 for consistency
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            result[sid] = arr
    return result


def sample_mri_data(data: np.ndarray) -> torch.Tensor:
    # Convert to torch tensor and transpose to get axial view as first dimension
    data = torch.from_numpy(data).float()  # Shape: [H, W, D]
    data = data.permute(2, 0, 1)  # Shape: [D, H, W]

    # Get middle 30 slices from axial view (now first dimension)
    total_slices = data.shape[0]
    start_slice = (total_slices - 30) // 2
    middle_30_slices = data[start_slice : start_slice + 30]  # Shape: [30, H, W]

    # Choose every 3rd slice from the 30 middle slices to get 10 slices
    middle_slices = middle_30_slices[::3]  # Shape: [10, H, W]

    # Resize each slice to 224x224
    # First, add batch and channel dimensions for interpolation
    middle_slices = middle_slices.unsqueeze(1)  # Shape: [10, 1, H, W]

    # Resize using bilinear interpolation
    resized_slices = F.interpolate(
        middle_slices, size=(224, 224), mode="bilinear", align_corners=False
    )

    # Remove the channel dimension
    resized_slices = resized_slices.squeeze(1)  # Shape: [10, 224, 224]

    # Normalize the data
    min_val = resized_slices.min()
    max_val = resized_slices.max()
    normalized_slices = (resized_slices - min_val) / (max_val - min_val + 1e-8)

    return normalized_slices  # Shape: [10, 224, 224]


def load_mri_data(h5_path: str) -> torch.Tensor:
    """
    Load MRI data from h5 file, extract middle 30 axial slices, and resize to 224x224.
    The input data is in shape [H, W, D] where D is the axial view dimension.
    We transpose it to [D, H, W] format.

    Args:
        h5_path (str): Path to the h5 file

    Returns:
        torch.Tensor: Normalized middle 30 axial slices of size [30, 224, 224]
    """
    with h5py.File(h5_path, "r") as f:
        # Assuming the data is stored in the root level
        data = f["data"][:]  # Load the full volume, shape: [H, W, D]

    # Convert to torch tensor and transpose to get axial view as first dimension
    data = torch.from_numpy(data).float()  # Shape: [H, W, D]
    data = data.permute(2, 0, 1)  # Shape: [D, H, W]

    # Get middle 30 slices from axial view (now first dimension)
    total_slices = data.shape[0]
    start_slice = (total_slices - 30) // 2
    middle_30_slices = data[start_slice : start_slice + 30]  # Shape: [30, H, W]

    # Choose every 3rd slice from the 30 middle slices to get 10 slices
    middle_slices = middle_30_slices[::3]  # Shape: [10, H, W]

    # Resize each slice to 224x224
    # First, add batch and channel dimensions for interpolation
    middle_slices = middle_slices.unsqueeze(1)  # Shape: [10, 1, H, W]

    # Resize using bilinear interpolation
    resized_slices = F.interpolate(
        middle_slices, size=(224, 224), mode="bilinear", align_corners=False
    )

    # Remove the channel dimension
    resized_slices = resized_slices.squeeze(1)  # Shape: [10, 224, 224]

    # Normalize the data
    min_val = resized_slices.min()
    max_val = resized_slices.max()
    normalized_slices = (resized_slices - min_val) / (max_val - min_val + 1e-8)

    return normalized_slices  # Shape: [10, 224, 224]


def load_json_data(json_path: str) -> List[Dict]:
    """
    Load data from JSON file.

    Args:
        json_path (str): Path to the JSON file

    Returns:
        List[Dict]: List of data samples
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_jsonl_data(jsonl_path: str) -> List[Dict]:
    """
    Load data from JSONL file.
    """
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def display_data_info(data: List[Dict], split: str):
    """
    Display information about the dataset using rich.

    Args:
        data (List[Dict]): List of data samples
        split (str): Dataset split name (train/test)
    """
    table = Table(title=f"{split} Dataset Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Number of samples", str(len(data)))
    table.add_row("Unique subjects", str(len(set(d["object_id"] for d in data))))

    # Count diagnoses
    from_diagnoses = {}
    to_diagnoses = {}
    for sample in data:
        from_diagnoses[sample["from_diagnosis"]] = (
            from_diagnoses.get(sample["from_diagnosis"], 0) + 1
        )
        to_diagnoses[sample["to_diagnosis"]] = (
            to_diagnoses.get(sample["to_diagnosis"], 0) + 1
        )

    table.add_row("From diagnoses", str(from_diagnoses))
    table.add_row("To diagnoses", str(to_diagnoses))

    console.print(table)


class MRIDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_json: str,
        test_json: str,
        data_root: str,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.25,
        all_mri_file: str = None,
        **kwargs,
    ):
        """Initialize the MRI DataModule with dataset and dataloader functionality."""
        super().__init__()
        self.train_json = train_json
        self.test_json = test_json
        self.data_root = data_root
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.all_mri_file = all_mri_file
        self.all_mri_data = None
        if self.all_mri_file is not None:
            self.all_mri_path = Path(self.data_root) / self.all_mri_file
            self.all_mri_path = str(self.all_mri_path)
            self.all_mri_data = load_mri_h5_to_dict(self.all_mri_path)

        # Save hyperparameters
        self.save_hyperparameters(ignore=["train_json", "test_json", "data_root"])

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # @pysnooper.snoop()
    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for training, validation and testing."""
        if stage == "fit":
            # Load and process training data
            train_json_path = os.path.join(self.data_root, self.train_json)
            train_data = load_json_data(train_json_path)
            display_data_info(train_data, "Training")

            # Create dataset
            class MRIDataset(Dataset):
                def __init__(self, data, data_root):
                    self.data = data
                    self.data_root = data_root

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    sample = self.data[idx]
                    # from_mri_path = sample["from_image_path"]
                    # keep last 2 parts of the path
                    # from_mri_path = "/".join(from_mri_path.split("/")[-2:])
                    from_mri_path = Path(self.data_root) / sample["from_image_path"]
                    # print(f"from_mri_path: {from_mri_path}")
                    # sys.exit()
                    # to_mri_path = Path(self.data_root) / sample["to_image_path"]

                    from_mri = load_mri_data(str(from_mri_path))
                    # print(f"from_mri: {from_mri.shape}")
                    # to_mri = load_mri_data(str(to_mri_path))

                    return {
                        "sample_id": sample["sample_id"],
                        "object_id": sample["object_id"],
                        "from_mri": from_mri,
                        "prompt_no_answer": sample["prompt_no_answer"],
                        "answer": sample["answer"],
                    }

            # Create full training dataset
            full_train_dataset = MRIDataset(train_data, self.data_root)

            # Split into train and validation
            total_size = len(full_train_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            console.print(f"[bold green]Dataset split information:[/bold green]")
            console.print(f"Training samples: {train_size}")
            console.print(f"Validation samples: {val_size}")

        if stage == "test" or stage is None:
            # Load and process test data
            test_json_path = os.path.join(self.data_root, self.test_json)
            test_data = load_jsonl_data(test_json_path)
            # display_data_info(test_data, "Test")

            class MRIDataset(Dataset):
                def __init__(self, data, data_root, all_mri_data):
                    self.data = data
                    self.data_root = data_root
                    self.all_mri_data = all_mri_data

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    sample = self.data[idx]
                    #   from_mri_path = sample["from_image_path"]
                    # keep last 2 parts of the path
                    # from_mri_path = "/".join(from_mri_path.split("/")[-2:])
                    from_mri = None
                    if self.all_mri_data is not None:
                        from_mri = self.all_mri_data[sample["from"]["sample_id"]]
                        from_mri = sample_mri_data(from_mri)
                        return {
                            "sample_id": sample["vlm_sample_id"],
                            "object_id": sample["patient_id"],
                            "from_mri": from_mri,
                            "prompt_no_answer": sample["prompt"],
                            "answer": sample["answer"],
                        }

                    else:
                        from_mri_path = Path(self.data_root) / sample["from_image_path"]
                        from_mri = load_mri_data(str(from_mri_path))
                        # to_mri = load_mri_data(str(to_mri_path))
                        # to_mri_path = Path(self.data_root) / sample["to_image_path"]

                        return {
                            "sample_id": sample["sample_id"],
                            "object_id": sample["object_id"],
                            "from_mri": from_mri,
                            "prompt_no_answer": sample["prompt_no_answer"],
                            "answer": sample["answer"],
                        }

            self.test_dataset = MRIDataset(test_data, self.data_root, self.all_mri_data)

    def train_dataloader(self) -> DataLoader:
        """Create and return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
