"""Fetch and stitch satellite tiles from the Google Maps Tiles API.

Supports on-disk caching so repeated runs never re-download the same data.
"""

import io
import json
import sys
import time
import logging
from pathlib import Path

import requests
import numpy as np
from PIL import Image

from .geo import latlon_to_tile, ground_resolution, TILE_SIZE

log = logging.getLogger(__name__)


def create_session(api_key: str) -> str:
    """Obtain a Map Tiles API session token for satellite imagery."""
    url = f"https://tile.googleapis.com/v1/createSession?key={api_key}"
    resp = requests.post(url, json={
        "mapType": "satellite",
        "language": "en-US",
        "region": "US",
    })
    resp.raise_for_status()
    return resp.json()["session"]


def fetch_tile(
    z: int, x: int, y: int,
    session: str, api_key: str,
    cache_dir: Path | None = None,
    retries: int = 3,
) -> Image.Image:
    """Download one 256x256 satellite tile, with automatic retry.

    If *cache_dir* is set, tiles are saved as PNGs and reused on subsequent
    runs — no API call is made for tiles that already exist on disk.
    """
    if cache_dir is not None:
        cached = cache_dir / f"{z}_{x}_{y}.png"
        if cached.exists():
            return Image.open(cached).convert("RGB")

    url = (
        f"https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}"
        f"?session={session}&key={api_key}"
    )
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                img.save(cached)
            return img
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            log.warning(
                "Tile %s/%s/%s attempt %d failed (%s), retrying in %ds",
                z, x, y, attempt + 1, exc, wait,
            )
            time.sleep(wait)


def fetch_mosaic(
    lat: float, lon: float, radius_m: float, zoom: int,
    api_key: str,
    session: str | None = None,
    rate_limit: float = 0.05,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Fetch tiles covering a circular area and stitch into one mosaic.

    If a cached mosaic + metadata already exist under *cache_dir*, they are
    loaded directly and no API calls are made.

    Returns
    -------
    mosaic : np.ndarray, shape (H, W, 3), dtype uint8
    meta   : dict with zoom, resolution, tile_range, mosaic_size, center coords
    """
    mosaic_npz = cache_dir / "mosaic.npz" if cache_dir else None
    meta_json = cache_dir / "mosaic_meta.json" if cache_dir else None

    if mosaic_npz and meta_json and mosaic_npz.exists() and meta_json.exists():
        log.info("Loading cached mosaic from %s", cache_dir)
        mosaic = np.load(mosaic_npz)["mosaic"]
        with open(meta_json) as f:
            meta = json.load(f)
        return mosaic, meta

    if session is None:
        session = create_session(api_key)

    res = ground_resolution(lat, zoom)
    radius_tiles = int(np.ceil(radius_m / res / TILE_SIZE)) + 1

    cx, cy = latlon_to_tile(lat, lon, zoom)
    x_min, x_max = cx - radius_tiles, cx + radius_tiles
    y_min, y_max = cy - radius_tiles, cy + radius_tiles

    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    total = nx * ny
    log.info("Fetching %d tiles (%dx%d) at zoom %d …", total, nx, ny, zoom)

    tile_cache = cache_dir / "tiles" if cache_dir else None
    mosaic = np.zeros((ny * TILE_SIZE, nx * TILE_SIZE, 3), dtype=np.uint8)

    fetched = 0
    cached_count = 0
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            fetched += 1
            sys.stdout.write(f"\r  Tile {fetched}/{total}")
            sys.stdout.flush()

            tile_img = fetch_tile(zoom, tx, ty, session, api_key,
                                  cache_dir=tile_cache)
            is_cached = (
                tile_cache is not None
                and (tile_cache / f"{zoom}_{tx}_{ty}.png").exists()
            )
            if is_cached:
                cached_count += 1
            px = (tx - x_min) * TILE_SIZE
            py = (ty - y_min) * TILE_SIZE
            mosaic[py : py + TILE_SIZE, px : px + TILE_SIZE] = np.array(tile_img)

            if not is_cached:
                time.sleep(rate_limit)

    sys.stdout.write("\n")
    if cached_count:
        log.info("  %d/%d tiles loaded from cache", cached_count, total)

    meta = {
        "zoom": zoom,
        "resolution": res,
        "center_lat": lat,
        "center_lon": lon,
        "tile_range": {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
        },
        "mosaic_size": {
            "width": nx * TILE_SIZE,
            "height": ny * TILE_SIZE,
        },
    }

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mosaic_npz, mosaic=mosaic)
        with open(meta_json, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Mosaic cached to %s", cache_dir)

    return mosaic, meta
