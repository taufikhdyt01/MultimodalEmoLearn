"""
Generate Landmark Heatmap Files for Early Fusion
=================================================
Converts existing 136-dim landmark vectors to 224x224 Gaussian heatmaps,
one heatmap per image. Used as additional channel input for Early Fusion.

For each landmark (x_norm, y_norm), place Gaussian blob on heatmap.
Take element-wise max across 68 landmarks to merge into single channel.
Heatmap values: [0, 1] float32, same normalization as image.

Output: X_{split}_heatmaps.npy (N, 224, 224) alongside existing X_{split}_*.npy
files. At training time, stack with image to form 4-channel input.

Usage:
    python scripts/generate_landmark_heatmaps.py              # all datasets
    python scripts/generate_landmark_heatmaps.py --only primer  # primer only
    python scripts/generate_landmark_heatmaps.py --sigma 5    # custom Gaussian std
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMG_SIZE = 224
DEFAULT_SIGMA = 3.0  # Gaussian std in pixels


def gen_heatmap(landmark_136, img_size=IMG_SIZE, sigma=DEFAULT_SIGMA,
                x_grid=None, y_grid=None):
    """Generate single 224x224 heatmap from 136-dim landmark vector.

    Takes element-wise max across all 68 landmarks (not sum) to keep
    heatmap in [0, 1] range even when landmarks overlap.
    """
    if x_grid is None or y_grid is None:
        y_grid, x_grid = np.ogrid[:img_size, :img_size]
    coords = landmark_136.reshape(-1, 2)  # (68, 2)
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)
    denom = 2.0 * sigma * sigma
    for x_norm, y_norm in coords:
        cx = x_norm * img_size
        cy = y_norm * img_size
        g = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / denom)
        heatmap = np.maximum(heatmap, g.astype(np.float32))
    return heatmap


def gen_heatmap_batch(landmarks, img_size=IMG_SIZE, sigma=DEFAULT_SIGMA, batch_log=500):
    """Generate heatmaps for batch of landmarks. Returns (N, 224, 224) float32."""
    N = len(landmarks)
    heatmaps = np.zeros((N, img_size, img_size), dtype=np.float32)
    y_grid, x_grid = np.ogrid[:img_size, :img_size]
    for i in range(N):
        heatmaps[i] = gen_heatmap(landmarks[i], img_size, sigma, x_grid, y_grid)
        if (i + 1) % batch_log == 0 or (i + 1) == N:
            print(f'    {i + 1}/{N}')
    return heatmaps


DATASETS = [
    # (name, root_dir, list_of_splits)
    ('Primer conf60',
     PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60',
     ['train', 'val', 'test']),
    ('Primer conf60 augmented',  # For B3 scenario in Early Fusion (nb 64)
     PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60_augmented',
     ['train']),  # Only train is augmented; val/test copied from base
    ('Primer conf60 under_660',
     PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60_under_660',
     ['train', 'val', 'test']),
    ('Primer conf60 under_660 4class',
     PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60_under_660_4class',
     ['train', 'val', 'test']),
    # Benchmarks
    ('RAF-DB 7class',
     PROJECT_ROOT / 'data' / 'benchmark' / 'rafdb_7class',
     ['train', 'test']),
    ('RAF-DB 4class',
     PROJECT_ROOT / 'data' / 'benchmark' / 'rafdb_4class',
     ['train', 'test']),
    ('KDEF 7class',
     PROJECT_ROOT / 'data' / 'benchmark' / 'kdef_7class',
     ['train', 'val', 'test']),
    ('KDEF 4class',
     PROJECT_ROOT / 'data' / 'benchmark' / 'kdef_4class',
     ['train', 'val', 'test']),
    ('CK+ 7class',
     PROJECT_ROOT / 'data' / 'benchmark' / 'ckplus_7class',
     ['all']),  # single file (no pre-split)
    ('CK+ 4class',
     PROJECT_ROOT / 'data' / 'benchmark' / 'ckplus_4class',
     ['all']),
    ('CK+ 4class contempt',
     PROJECT_ROOT / 'data' / 'benchmark' / 'ckplus_4class_contempt',
     ['all']),
    ('JAFFE 7class',
     PROJECT_ROOT / 'data' / 'benchmark' / 'jaffe_7class',
     ['all']),
    ('JAFFE 4class',
     PROJECT_ROOT / 'data' / 'benchmark' / 'jaffe_4class',
     ['all']),
]


def landmark_filename(root, split):
    """Get the landmark file for this split. Handle 'all' (no split prefix) case."""
    if split == 'all':
        candidate = root / 'X_landmarks.npy'
    else:
        candidate = root / f'X_{split}_landmarks.npy'
    return candidate


def heatmap_filename(root, split):
    if split == 'all':
        return root / 'X_heatmaps.npy'
    return root / f'X_{split}_heatmaps.npy'


def process_dataset(name, root, splits, sigma):
    if not root.exists():
        print(f'  [SKIP] {name}: root {root} not found')
        return
    print(f'\n  {name}  ({root.name})')
    for split in splits:
        lm_file = landmark_filename(root, split)
        out_file = heatmap_filename(root, split)
        if not lm_file.exists():
            print(f'    [SKIP] {split}: no landmark file at {lm_file.name}')
            continue
        if out_file.exists():
            print(f'    [SKIP] {split}: heatmap already exists ({out_file.name})')
            continue
        print(f'    Generating {split} ...')
        landmarks = np.load(lm_file)
        if landmarks.ndim != 2 or landmarks.shape[1] != 136:
            print(f'    [WARN] {split}: unexpected shape {landmarks.shape}, skipping')
            continue
        heatmaps = gen_heatmap_batch(landmarks, sigma=sigma)
        np.save(out_file, heatmaps)
        print(f'    Saved: {out_file.name}  shape={heatmaps.shape}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, default=None,
                    help='Process only datasets whose name contains this substring')
    ap.add_argument('--sigma', type=float, default=DEFAULT_SIGMA,
                    help=f'Gaussian std in pixels (default {DEFAULT_SIGMA})')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite existing heatmap files')
    args = ap.parse_args()

    if args.force:
        # monkey-patch to always regenerate
        global heatmap_filename

        def _fn(root, split):
            if split == 'all':
                return root / 'X_heatmaps.npy'
            return root / f'X_{split}_heatmaps.npy'
        heatmap_filename = _fn  # no effect since logic stays — force via unlinking
        for name, root, splits in DATASETS:
            for split in splits:
                out = _fn(root, split)
                if out.exists():
                    out.unlink()
                    print(f'  [FORCE] removed {out.name}')

    print(f'Generating landmark heatmaps (sigma={args.sigma} px)')
    for name, root, splits in DATASETS:
        if args.only and args.only.lower() not in name.lower():
            continue
        process_dataset(name, root, splits, args.sigma)

    print('\nDone.')


if __name__ == '__main__':
    main()
