import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional, Sequence, Union

import numpy as np
import SimpleITK as sitk
import scipy.ndimage
import os
import yaml
import matplotlib.pyplot as plt

def load_image_as_array(image_path: str) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """Read image with SimpleITK and return numpy array (z, y, x) plus spacing/origin."""
    image = sitk.ReadImage(str(image_path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = tuple(reversed(image.GetSpacing()))  # sitk spacing is (x, y, z)
    origin = tuple(reversed(image.GetOrigin()))
    return array, spacing, origin


def resample_image(image: np.ndarray,
                   current_spacing: Tuple[float, float, float],
                   target_spacing: np.ndarray,
                   *,
                   is_mask: bool = False) -> np.ndarray:
    zoom_factors = np.asarray(current_spacing) / target_spacing
    order = 0 if is_mask else 3
    return scipy.ndimage.zoom(image, zoom=zoom_factors, order=order, prefilter=not is_mask)


def crop_or_pad(image: np.ndarray,
                resolution: np.ndarray,
                pad_value: float,
                *,
                is_mask: bool = False) -> np.ndarray:
    processed = []
    original = []
    shape = image.shape
    for axis in range(3):
        if shape[axis] >= resolution[axis]:
            processed.append({'left': 0, 'right': resolution[axis]})
            offset = (shape[axis] - resolution[axis]) // 2
            original.append({'left': offset, 'right': offset + resolution[axis]})
        else:
            offset = (resolution[axis] - shape[axis]) // 2
            processed.append({'left': offset, 'right': offset + shape[axis]})
            original.append({'left': 0, 'right': shape[axis]})

    def slice_array(target: np.ndarray, target_idx, source: np.ndarray, source_idx):
        target[
            target_idx[0]['left']:target_idx[0]['right'],
            target_idx[1]['left']:target_idx[1]['right'],
            target_idx[2]['left']:target_idx[2]['right'],
        ] = source[
            source_idx[0]['left']:source_idx[0]['right'],
            source_idx[1]['left']:source_idx[1]['right'],
            source_idx[2]['left']:source_idx[2]['right'],
        ]
        return target

    dtype = np.int32 if is_mask else np.float32
    output = np.full(resolution, fill_value=pad_value, dtype=dtype)
    output = slice_array(output, processed, image, original)
    return output


def normalize_image(image: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    min_value, max_value = value_range
    image = np.clip(image, a_min=min_value, a_max=max_value)
    return (image - min_value) / (max_value - min_value)


def process_volume(
    image: np.ndarray,
    spacing: Tuple[float, float, float],
    config: Dict,
    mask: Optional[np.ndarray] = None,
    mask_labels: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    spacing_target = np.asarray(config['dataset']['spacing'], dtype=np.float32)
    resolution_target = np.asarray(config['dataset']['resolution'], dtype=np.int32)
    value_range = np.asarray(config['dataset']['value_range'], dtype=np.float32)

    resampled = resample_image(image, spacing, spacing_target)
    cropped = crop_or_pad(resampled, resolution_target, pad_value=image.min())

    processed_mask = None
    if mask is not None:
        if mask_labels is not None:
            filtered_mask = np.zeros_like(mask)
            for lbl in mask_labels:
                filtered_mask[mask == lbl] = lbl
        else:
            filtered_mask = mask

        resampled_mask = resample_image(filtered_mask, spacing, spacing_target, is_mask=True)
        resampled_mask = np.round(resampled_mask).astype(np.int32)
        processed_mask = crop_or_pad(resampled_mask, resolution_target, pad_value=0, is_mask=True)

    return cropped, processed_mask


def save_numpy(array: np.ndarray, output_path: Path) -> None:
    np.save(output_path, array)
    print(f"Processed volume saved to {output_path}")


def save_nifti(array: np.ndarray, spacing: Tuple[float, float, float], origin: Tuple[float, float, float], output_path: Path) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(reversed(spacing)))
    image.SetOrigin(tuple(reversed(origin)))
    sitk.WriteImage(image, str(output_path))
    print(f"Processed volume saved to {output_path}")


def save_center_slices(volume: np.ndarray, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2

    slices = {
        "axial": volume[z_mid, :, :],
        "coronal": volume[:, y_mid, :],
        "sagittal": volume[:, :, x_mid],
    }

    v_min = float(volume.min())
    v_max = float(volume.max())
    if v_max > v_min:
        def normalize(img):
            return (img - v_min) / (v_max - v_min)
    else:
        def normalize(img):
            return np.zeros_like(img, dtype=np.float32)

    for plane, img in slices.items():
        img_norm = normalize(img)
        out_path = output_dir / f"{prefix}_{plane}.png"
        plt.imsave(out_path, img_norm, cmap="gray")
        print(f"Saved slice: {out_path}")


def load_mask_with_labels(mask_path: str, labels: Optional[Sequence[int]]) -> np.ndarray:
    mask_image = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask_image).astype(np.int32)
    if labels is not None:
        filtered = np.zeros_like(mask_array)
        for lbl in labels:
            filtered[mask_array == lbl] = lbl
        return filtered
    return mask_array


def run_processing(
    image_path: str,
    config_path: Optional[str] = None,
    *,
    config: Optional[Dict] = None,
    output_npy: Optional[str] = None,
    output_nii: Optional[str] = None,
    mask_path: Optional[str] = None,
    mask_output_nii: Optional[str] = None,
    mask_labels: Optional[Sequence[int]] = None,
    slice_output_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience wrapper to process a volume by providing python variables directly.
    Either pass `config` as a dictionary or provide `config_path`.
    """
    if config is None:
        if config_path is None:
            raise ValueError("Either `config` or `config_path` must be provided.")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    image_array, spacing, origin = load_image_as_array(image_path)
    mask_array = None
    if mask_path:
        mask_array = load_mask_with_labels(mask_path, mask_labels)

    processed, processed_mask = process_volume(
        image_array,
        spacing,
        config,
        mask=mask_array,
        mask_labels=None,  # mask already filtered if needed
    )
    print(f"Processed volume shape: {processed.shape}")

    if output_npy:
        save_numpy(processed, Path(output_npy))

    if output_nii:
        target_spacing = tuple(config['dataset']['spacing'])
        save_nifti(processed, spacing=target_spacing, origin=origin, output_path=Path(output_nii))

    if mask_output_nii and processed_mask is not None:
        target_spacing = tuple(config['dataset']['spacing'])
        save_nifti(processed_mask, spacing=target_spacing, origin=origin, output_path=Path(mask_output_nii))

    if slice_output_dir:
        prefix = Path(image_path).stem
        save_center_slices(processed, Path(slice_output_dir), prefix)

    if not any([output_npy, output_nii, mask_output_nii]):
        print("Processing complete (no output paths provided).")

    return processed, processed_mask


def main(
    image: Optional[str] = None,
    config: Optional[str] = None,
    *,
    config_dict: Optional[Dict] = None,
    output_npy: Optional[str] = None,
    output_nii: Optional[str] = None,
    mask: Optional[str] = None,
    mask_output_nii: Optional[str] = None,
    mask_labels: Optional[Sequence[int]] = None,
    slice_output_dir: Optional[str] = None,
    argv=None,
):
    """
    Entry point usable from Python or CLI.

    - Pass `image`, `config` (path), optional outputs to call programmatically.
    - Alternatively leave them as None and provide CLI args (or `argv` list).
    """
    run_processing(
        image_path=image,
        config_path=config,
        config=config_dict,
        output_npy=output_npy,
        output_nii=output_nii,
        mask_path=mask,
        mask_output_nii=mask_output_nii,
        mask_labels=mask_labels,
        slice_output_dir=slice_output_dir,
    )

def process_volume_all(image_path_root, mask_path_root, mode, config,
                       output_mask_path, output_nii_path,
                       slice_output_root: Optional[str] = None):
    case_names_list = os.listdir(image_path_root)
    mask_label = [33, 34, 35, 36, 37]
    for case_name in case_names_list:
        image_path = os.path.join(image_path_root, case_name, f"{mode}_cropped.nii.gz")
        mask_path = os.path.join(mask_path_root, case_name, f"{mode}_mask_cropped.nii.gz")

        if not os.path.exists(image_path):
            print(f"[Skip] image not found: {image_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"[Skip] mask not found: {mask_path}")
            continue

        case_volume_dir = os.path.join(output_nii_path, case_name)
        case_mask_dir = os.path.join(output_mask_path, case_name)
        os.makedirs(case_volume_dir, exist_ok=True)
        os.makedirs(case_mask_dir, exist_ok=True)

        volume_out = os.path.join(case_volume_dir, f"{mode}_processed_volume.nii.gz")
        mask_out = os.path.join(case_mask_dir, f"{mode}_processed_mask.nii.gz")

        slice_dir = None
        if slice_output_root:
            slice_dir = os.path.join(slice_output_root, case_name)

        try:
            run_processing(
                image_path=image_path,
                config_path=config,
                output_nii=volume_out,
                mask_path=mask_path,
                mask_output_nii=mask_out,
                mask_labels=mask_label,
                slice_output_dir=slice_dir,
            )
            print(f"[Done] {case_name} -> {volume_out}")
        except Exception as exc:
            print(f"[Error] {case_name}: {exc}")


if __name__ == "__main__":
    # image = 'D:/data_space/Zhongrifriendly/paired_data_cropped/final_data_set/anlinv/pre_cropped.nii.gz'
    # mask = 'D:/data_space/Zhongrifriendly/paired_data_cropped/crop_mask/anlinv/pre_mask_cropped.nii.gz'
    # config = './config.yaml'
    # output_nii_path = './anlinv_processed/'
    # os.makedirs(output_nii_path, exist_ok=True)
    # output_nii = os.path.join(output_nii_path, 'pre_volume.nii.gz')
    # mask_output_nii = os.path.join(output_nii_path, 'pre_mask.nii.gz')
    # main(image=image, config=config, output_nii=output_nii, mask=mask, mask_output_nii=mask_output_nii, mask_labels=[33,34,35,36,37])

    image_path_root = 'D:/data_space/Zhongrifriendly/paired_data_cropped/final_data_set'
    mask_path_root = 'D:/data_space/Zhongrifriendly/paired_data_cropped/crop_mask'
    mode = 'intra'
    config = './Dataset_Preprocess/config_pre/config_128_1.yaml'
    output_mask_path = 'D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/processed_176_1_mask'
    output_nii_path = 'D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/processed_176_1_volume'
    os.makedirs(output_mask_path, exist_ok=True)
    os.makedirs(output_nii_path, exist_ok=True)
    slice_output_root = 'D:/data_space/Zhongrifriendly/paired_data_cropped_176_1/vis_cropped_172_1_ct'
    process_volume_all(image_path_root, mask_path_root, mode, config,
                       output_mask_path, output_nii_path,
                       slice_output_root=slice_output_root)


