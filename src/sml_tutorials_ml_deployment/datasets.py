import warnings
from pathlib import Path
from typing import Callable, Literal

import matplotlib as mpl
import numpy as np
import torch
from einops import rearrange
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms.functional import convert_image_dtype, to_tensor


def normalize_cmap(cmap: list[tuple[int, int, int]]) -> list[tuple[float, float, float]]:
    return [tuple(c / 255 for c in rgb) for rgb in cmap]


def make_classes_legend(colors: list[tuple[float, float, float]], labels: list[str]):
    colors = normalize_cmap(colors)
    return [mpl.patches.Patch(color=color, label=label) for color, label in zip(colors, labels)]


def transforms(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    img, mask = batch["image"], batch["mask"]
    batch["image"] = rearrange(torch.as_tensor(img, dtype=torch.float32), "h w c -> c h w") / 255
    batch["mask"] = torch.as_tensor(mask, dtype=torch.long)
    return batch


class DeepGlobeLandCover(Dataset):
    SPLITS = ["train", "test"]
    CLASSES = [
        "urban_land",
        "agriculture_land",
        "rangeland",
        "forest_land",
        "water",
        "barren_land",
        "unknown",
    ]
    COLORMAP = [
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
    ]

    def __init__(
        self,
        root_dir: Path,
        split: Literal["training", "test"] = "training",
        transforms: Callable[[dict[str, np.ndarray]], dict[str, torch.Tensor]] | None = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self._ids: list[str] | None = None

    @property
    def ids(self) -> list[str]:
        if self._ids is None:
            imgs_root = self.root_dir / f"{self.split}_data" / "images"
            imgs_paths = sorted(list(imgs_root.glob("*sat.jpg")))
            self._ids = [path.stem.split("_sat", 1)[0] for path in imgs_paths]
            if len(self._ids) == 0:
                warnings.warn(f"Found zero samples in the provided Deep Globe dataset root: {self.root_dir.resolve()}")
        return self._ids

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        sample_id = self.ids[index]
        img_path = self.root_dir / f"{self.split}_data" / "images" / f"{sample_id}_sat.jpg"
        mask_path = self.root_dir / f"{self.split}_data" / "masks" / f"{sample_id}_mask.png"
        sample = {
            "id": sample_id,
            "image": io.imread(img_path),
            "mask": self.mask_image_to_class_indices_map(io.imread(mask_path)),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(self.ids)

    def mask_image_to_class_indices_map(self, mask_img: np.ndarray) -> np.ndarray:
        h, w, c = mask_img.shape
        indices_map = np.empty((h, w), dtype=mask_img.dtype)
        for idx, color in enumerate(self.COLORMAP):
            indices_map[(mask_img[..., :] == color).all(axis=-1)] = idx
        return indices_map
