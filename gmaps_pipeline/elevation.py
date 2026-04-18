"""Fetch elevation data for the pipeline.

Providers
---------
- ``opentopodata`` (default) — free SRTM 30 m data, no API key needed.
- ``flat``          — all elevations set to 0; no network calls at all.
- ``google``        — Google Maps Elevation API (requires paid tier + API key).

All providers support on-disk caching so repeated runs skip the network.
"""

import sys
import time
import logging
from pathlib import Path

import requests
import numpy as np

from .geo import pixel_to_latlon

log = logging.getLogger(__name__)

PROVIDERS = ("opentopodata", "flat", "google")


# ── public entry point ─────────────────────────────────────────────────

def fetch_elevation_grid(
    mosaic_meta: dict,
    grid_size: int,
    provider: str = "opentopodata",
    api_key: str | None = None,
    rate_limit: float = 0.1,
    cache_dir: Path | None = None,
) -> dict:
    """Query elevation for an NxN grid covering the mosaic area.

    Returns a dict with 1-D arrays ``lats``, ``lons``, ``elevations``,
    mosaic-pixel coords ``px`` / ``py``, ``grid_shape``, and
    ``center_elevation``.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown elevation provider {provider!r}. "
            f"Choose from: {', '.join(PROVIDERS)}"
        )

    cache_file = cache_dir / "elevation.npz" if cache_dir else None
    if cache_file and cache_file.exists():
        log.info("Loading cached elevation from %s", cache_file)
        data = np.load(cache_file, allow_pickle=True)
        return {
            "lats": data["lats"],
            "lons": data["lons"],
            "elevations": data["elevations"],
            "grid_shape": tuple(data["grid_shape"]),
            "px": data["px"],
            "py": data["py"],
            "center_elevation": float(data["center_elevation"]),
        }

    h = mosaic_meta["mosaic_size"]["height"]
    w = mosaic_meta["mosaic_size"]["width"]

    margin = 0.05
    px_coords = np.linspace(w * margin, w * (1 - margin), grid_size)
    py_coords = np.linspace(h * margin, h * (1 - margin), grid_size)
    px_grid, py_grid = np.meshgrid(px_coords, py_coords)
    px_flat = px_grid.ravel()
    py_flat = py_grid.ravel()

    lats = np.empty(len(px_flat))
    lons = np.empty(len(px_flat))
    for i, (px, py) in enumerate(zip(px_flat, py_flat)):
        lat, lon = pixel_to_latlon(float(px), float(py), mosaic_meta)
        lats[i] = lat
        lons[i] = lon

    if provider == "flat":
        elevations = np.zeros(len(lats), dtype=np.float64)
        center_elev = 0.0
        log.info("Using flat elevation (all z = 0)")
    elif provider == "opentopodata":
        elevations, center_elev = _fetch_opentopodata(
            lats, lons, mosaic_meta, rate_limit,
        )
    elif provider == "google":
        if not api_key:
            raise RuntimeError(
                "Google elevation provider requires --api-key "
                "(or GOOGLE_MAPS_API_KEY in .env)."
            )
        elevations, center_elev = _fetch_google(
            lats, lons, mosaic_meta, api_key, rate_limit,
        )

    result = {
        "lats": lats,
        "lons": lons,
        "elevations": elevations,
        "grid_shape": (grid_size, grid_size),
        "px": px_flat,
        "py": py_flat,
        "center_elevation": center_elev,
    }

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            lats=lats, lons=lons, elevations=elevations,
            grid_shape=np.array([grid_size, grid_size]),
            px=px_flat, py=py_flat,
            center_elevation=np.array(center_elev),
        )
        log.info("Elevation cached to %s", cache_file)

    return result


# ── Open Topo Data (free, no key) ──────────────────────────────────────

_OTD_BATCH = 100
_OTD_ENDPOINT = "https://api.opentopodata.org/v1/srtm30m"


def _fetch_opentopodata(
    lats: np.ndarray, lons: np.ndarray,
    mosaic_meta: dict, rate_limit: float,
) -> tuple[np.ndarray, float]:
    elevations = np.zeros(len(lats), dtype=np.float64)
    n_batches = int(np.ceil(len(lats) / _OTD_BATCH))
    log.info(
        "Fetching elevation from Open Topo Data (%d points, %d batches) …",
        len(lats), n_batches,
    )

    for b in range(n_batches):
        sys.stdout.write(f"\r  Elevation batch {b + 1}/{n_batches}")
        sys.stdout.flush()

        start = b * _OTD_BATCH
        end = min(start + _OTD_BATCH, len(lats))
        locations = "|".join(
            f"{lats[i]:.8f},{lons[i]:.8f}" for i in range(start, end)
        )
        resp = requests.get(
            _OTD_ENDPOINT,
            params={"locations": locations},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            raise RuntimeError(
                f"Open Topo Data error: {data.get('error', data)}"
            )
        for i, result in enumerate(data["results"]):
            val = result.get("elevation")
            elevations[start + i] = val if val is not None else 0.0
        time.sleep(max(rate_limit, 1.0))

    sys.stdout.write("\n")

    h = mosaic_meta["mosaic_size"]["height"]
    w = mosaic_meta["mosaic_size"]["width"]
    center_lat, center_lon = pixel_to_latlon(w / 2, h / 2, mosaic_meta)
    c_resp = requests.get(
        _OTD_ENDPOINT,
        params={"locations": f"{center_lat:.8f},{center_lon:.8f}"},
        timeout=30,
    )
    c_resp.raise_for_status()
    c_val = c_resp.json()["results"][0].get("elevation")
    center_elev = c_val if c_val is not None else 0.0

    return elevations, center_elev


# ── Google Maps Elevation API (paid) ───────────────────────────────────

_GOOGLE_BATCH = 512
_GOOGLE_ENDPOINT = "https://maps.googleapis.com/maps/api/elevation/json"


def _fetch_google(
    lats: np.ndarray, lons: np.ndarray,
    mosaic_meta: dict, api_key: str, rate_limit: float,
) -> tuple[np.ndarray, float]:
    elevations = np.zeros(len(lats), dtype=np.float64)
    n_batches = int(np.ceil(len(lats) / _GOOGLE_BATCH))
    log.info(
        "Fetching elevation from Google (%d points, %d batches) …",
        len(lats), n_batches,
    )

    for b in range(n_batches):
        sys.stdout.write(f"\r  Elevation batch {b + 1}/{n_batches}")
        sys.stdout.flush()

        start = b * _GOOGLE_BATCH
        end = min(start + _GOOGLE_BATCH, len(lats))
        locations = "|".join(
            f"{lats[i]:.8f},{lons[i]:.8f}" for i in range(start, end)
        )
        resp = requests.get(
            _GOOGLE_ENDPOINT,
            params={"locations": locations, "key": api_key},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["status"] != "OK":
            hint = ""
            if data["status"] == "REQUEST_DENIED":
                hint = (
                    "\n\nHINT: Check the following in Google Cloud Console:\n"
                    '  1. "Maps Elevation API" is enabled '
                    "(APIs & Services → Library)\n"
                    "  2. Billing is active on the project\n"
                    "  3. Your API key is not restricted to other APIs only "
                    "(APIs & Services → Credentials → edit key → "
                    "API restrictions)\n"
                    "  4. No IP / referrer restrictions are blocking "
                    "this machine"
                )
            raise RuntimeError(
                f"Elevation API error: {data['status']}"
                f" — {data.get('error_message', '')}"
                f"{hint}"
            )
        for i, result in enumerate(data["results"]):
            elevations[start + i] = result["elevation"]
        time.sleep(rate_limit)

    sys.stdout.write("\n")

    h = mosaic_meta["mosaic_size"]["height"]
    w = mosaic_meta["mosaic_size"]["width"]
    center_lat, center_lon = pixel_to_latlon(w / 2, h / 2, mosaic_meta)
    c_resp = requests.get(
        _GOOGLE_ENDPOINT,
        params={
            "locations": f"{center_lat:.8f},{center_lon:.8f}",
            "key": api_key,
        },
        timeout=30,
    )
    c_resp.raise_for_status()
    center_elev = c_resp.json()["results"][0]["elevation"]

    return elevations, center_elev
