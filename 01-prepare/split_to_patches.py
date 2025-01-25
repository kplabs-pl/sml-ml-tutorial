import argparse
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
from skimage import io
from skimage.util import view_as_blocks
from tqdm import tqdm

DEEP_GLOBE_SCENE_SIZE = 2448


def center_pad_to_divisible_by(img: np.ndarray, divisible_by: int) -> np.ndarray:
    h, w, c = img.shape
    target_shape = (divisible_by * ceil(h / divisible_by), divisible_by * ceil(w / divisible_by), c)
    padding = tuple(target - dim for target, dim in zip(target_shape, img.shape))
    center_padding = tuple((pad_dim // 2, pad_dim - pad_dim // 2) for pad_dim in padding)
    return np.pad(img, center_padding)


def main(patch_size: int, input_dir: Path, output_dir: Path):
    img_paths = list(input_dir.rglob("*sat.jpg"))
    mask_paths = list(input_dir.rglob("*mask.png"))

    for img_path in (progress := tqdm(img_paths + mask_paths)):
        progress.set_description(f"Processing {img_path}")

        img = io.imread(img_path)
        h, w, c = img.shape
        assert h == DEEP_GLOBE_SCENE_SIZE and w == DEEP_GLOBE_SCENE_SIZE

        img = center_pad_to_divisible_by(img, patch_size)
        patch_shape = (patch_size, patch_size, c)
        patches = view_as_blocks(img, patch_shape)

        out_img_dir_path = output_dir / img_path.parent.relative_to(input_dir)
        out_img_dir_path.mkdir(exist_ok=True, parents=True)

        patch_rows, patch_cols, *_ = patches.shape
        for r, c in product(range(patch_rows), range(patch_cols)):
            # There is a extraneous dimension in the patches from the stride along channels
            patch = np.squeeze(patches[r, c], axis=0)
            img_id, img_name_rest = img_path.name.split("_", 1)
            io.imsave(out_img_dir_path / f"{img_id}_{r:02d}_{c:02d}_{img_name_rest}", patch, check_contrast=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.patch_size, args.input_dir, args.output_dir)
