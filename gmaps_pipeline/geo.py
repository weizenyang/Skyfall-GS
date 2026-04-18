"""Geographic coordinate utilities for the Google Maps → Skyfall-GS pipeline.

Handles Web Mercator tile math, lat/lon ↔ mosaic-pixel conversion, and
pixel → local East-North-Up (ENU) mapping used by the scene builder.
"""

import math

TILE_SIZE = 256


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert latitude / longitude to tile (x, y) at *zoom*."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return x, y


def tile_to_latlon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Return the (lat, lon) of the NW corner of tile (x, y)."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad), lon


def ground_resolution(lat: float, zoom: int) -> float:
    """Meters per pixel at *lat* and *zoom*."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def pixel_to_latlon(px: float, py: float, meta: dict) -> tuple[float, float]:
    """Convert mosaic-pixel coordinates to (lat, lon).

    *meta* must contain ``zoom`` and ``tile_range.{x_min, y_min}``.
    """
    zoom = meta["zoom"]
    n = 2 ** zoom
    x_min = meta["tile_range"]["x_min"]
    y_min = meta["tile_range"]["y_min"]
    global_px = px + x_min * TILE_SIZE
    global_py = py + y_min * TILE_SIZE
    lon = global_px / (n * TILE_SIZE) * 360.0 - 180.0
    lat_rad = math.atan(
        math.sinh(math.pi * (1 - 2 * global_py / (n * TILE_SIZE)))
    )
    return math.degrees(lat_rad), lon


def latlon_to_pixel(lat: float, lon: float, meta: dict) -> tuple[float, float]:
    """Convert (lat, lon) to mosaic-pixel coordinates."""
    zoom = meta["zoom"]
    n = 2 ** zoom
    x_min = meta["tile_range"]["x_min"]
    y_min = meta["tile_range"]["y_min"]
    global_px = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    global_py = (
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
        * TILE_SIZE
    )
    return global_px - x_min * TILE_SIZE, global_py - y_min * TILE_SIZE
