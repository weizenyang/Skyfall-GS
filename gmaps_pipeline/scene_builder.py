"""Assemble a Skyfall-GS Satellite-format dataset from fetched Google Maps data.

Produces the directory layout expected by ``readSatelliteInfo`` in
``scene/dataset_readers.py``::

    <output>/
    ├── images/           # RGB crops (PNG)
    ├── masks/            # binary masks (.npy + .png)
    ├── transforms_train.json
    ├── transforms_test.json
    └── points3D.txt      # COLMAP text format
"""

import json
import math
import logging

import numpy as np
from PIL import Image
from pathlib import Path

log = logging.getLogger(__name__)


def build_dataset(
    mosaic: np.ndarray,
    mosaic_meta: dict,
    elevation: dict,
    output_dir: str,
    *,
    view_size: int = 512,
    view_stride: int = 384,
    fov_deg: float = 15.0,
    test_ratio: float = 0.1,
) -> None:
    """Build a complete Skyfall-GS Satellite-format dataset on disk."""
    out = Path(output_dir)
    img_dir = out / "images"
    mask_dir = out / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    views = _generate_views(
        mosaic, mosaic_meta, view_size, view_stride, fov_deg,
        elevation["center_elevation"],
    )
    if not views:
        raise RuntimeError(
            f"No views generated. Mosaic is {mosaic.shape[1]}x{mosaic.shape[0]} "
            f"but view_size={view_size}. Try a larger --radius or smaller --view-size."
        )
    log.info("Generated %d views", len(views))

    if len(views) < 4:
        log.warning(
            "Only %d views were generated — reconstruction quality may be poor. "
            "Consider a larger --radius or smaller --view-stride.",
            len(views),
        )

    n_test = 0 if len(views) < 5 else max(1, int(len(views) * test_ratio))
    indices = list(range(len(views)))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    test_ids = set(indices[:n_test])

    train_frames: list[dict] = []
    test_frames: list[dict] = []

    for idx, (crop, frame_dict) in enumerate(views):
        name = f"view_{idx:04d}"
        Image.fromarray(crop).save(img_dir / f"{name}.png")

        mask = 1 - np.all(crop == 0, axis=-1).astype(np.uint8)
        np.save(mask_dir / f"{name}.npy", mask)
        Image.fromarray(mask * 255).save(mask_dir / f"{name}.png")

        frame_dict["file_path"] = f"images/{name}.png"
        if idx in test_ids:
            test_frames.append(frame_dict)
        else:
            train_frames.append(frame_dict)

    R_fix = np.eye(4).tolist()
    T_fix = [0.0, 0.0, 0.0]

    _write_transforms(train_frames, R_fix, T_fix, out / "transforms_train.json")
    _write_transforms(test_frames, R_fix, T_fix, out / "transforms_test.json")
    log.info("Wrote %d train / %d test frames", len(train_frames), len(test_frames))

    _write_point_cloud(mosaic, mosaic_meta, elevation, out / "points3D.txt")
    log.info("Dataset written to %s", out)


# ---------------------------------------------------------------------------
# View generation
# ---------------------------------------------------------------------------

def _generate_views(
    mosaic: np.ndarray,
    meta: dict,
    view_size: int,
    view_stride: int,
    fov_deg: float,
    center_elev: float,
) -> list[tuple[np.ndarray, dict]]:
    """Create overlapping nadir crops and their camera parameters."""
    h, w = mosaic.shape[:2]

    actual_vs = min(view_size, h, w)
    if actual_vs < view_size:
        log.warning(
            "Mosaic (%dx%d) smaller than view_size (%d); using %d instead",
            w, h, view_size, actual_vs,
        )
        view_size = actual_vs
        view_stride = min(view_stride, view_size)

    res = meta["resolution"]
    half_fov = math.radians(fov_deg / 2.0)
    focal = (view_size / 2.0) / math.tan(half_fov)
    altitude = focal * res

    cx_m = w / 2.0
    cy_m = h / 2.0

    y_starts = list(range(0, h - view_size + 1, view_stride))
    x_starts = list(range(0, w - view_size + 1, view_stride))
    if not y_starts:
        y_starts = [max(0, (h - view_size) // 2)]
    if not x_starts:
        x_starts = [max(0, (w - view_size) // 2)]

    views: list[tuple[np.ndarray, dict]] = []

    for py in y_starts:
        for px in x_starts:
            crop = mosaic[py : py + view_size, px : px + view_size]
            if crop.shape[0] != view_size or crop.shape[1] != view_size:
                continue

            crop_cx = px + view_size / 2.0
            crop_cy = py + view_size / 2.0

            east = (crop_cx - cx_m) * res
            north = -(crop_cy - cy_m) * res

            c2w = _make_nadir_c2w(east, north, altitude)

            frame = {
                "transform_matrix_rotated": c2w.tolist(),
                "fl_x": float(focal),
                "fl_y": float(focal),
                "cx": view_size / 2.0,
                "cy": view_size / 2.0,
            }
            views.append((crop, frame))

    return views


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _make_nadir_c2w(east: float, north: float, up: float) -> np.ndarray:
    """Camera-to-world for a straight-down camera in COLMAP convention.

    COLMAP camera axes: X = right, Y = down, Z = forward.
    World axes (ENU) : X = east,  Y = north, Z = up.

    Mapping
    -------
    cam-X (right in image)   → world-east   = ( 1,  0,  0)
    cam-Y (down  in image)   → world-south  = ( 0, -1,  0)
    cam-Z (forward / into scene) → world-down = ( 0,  0, -1)
    """
    return np.array([
        [1.0,  0.0,  0.0, east],
        [0.0, -1.0,  0.0, north],
        [0.0,  0.0, -1.0, up],
        [0.0,  0.0,  0.0, 1.0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_transforms(
    frames: list[dict],
    R_fix: list,
    T_fix: list,
    path: Path,
) -> None:
    """Write a ``transforms_*.json`` compatible with the Satellite reader."""
    data = {"R": R_fix, "T": T_fix, "frames": frames}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_point_cloud(
    mosaic: np.ndarray,
    meta: dict,
    elevation: dict,
    path: Path,
) -> None:
    """Write ``points3D.txt`` in COLMAP text format."""
    res = meta["resolution"]
    h, w = mosaic.shape[:2]
    cx_m, cy_m = w / 2.0, h / 2.0
    center_elev = elevation["center_elevation"]

    n = len(elevation["lats"])
    log.info("Writing %d points to %s", n, path)

    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")

        pid = 1
        for i in range(n):
            px = float(elevation["px"][i])
            py_val = float(elevation["py"][i])

            east = (px - cx_m) * res
            north = -(py_val - cy_m) * res
            up = elevation["elevations"][i] - center_elev

            ix = int(np.clip(px, 0, w - 1))
            iy = int(np.clip(py_val, 0, h - 1))
            r = int(mosaic[iy, ix, 0])
            g = int(mosaic[iy, ix, 1])
            b = int(mosaic[iy, ix, 2])

            f.write(f"{pid} {east:.6f} {north:.6f} {up:.6f} {r} {g} {b} 1.0\n")
            pid += 1
