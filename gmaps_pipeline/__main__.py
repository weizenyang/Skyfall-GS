"""CLI entry-point: ``python -m gmaps_pipeline …``

Example
-------
python -m gmaps_pipeline \
    --lat 40.748817 --lon -73.985428 \
    --radius 300 --zoom 19 \
    -o data/my_scene
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .tiles import create_session, fetch_mosaic
from .elevation import fetch_elevation_grid, PROVIDERS as ELEV_PROVIDERS
from .scene_builder import build_dataset


def _resolve_api_key(cli_value: str | None) -> str:
    """Return the API key from the CLI flag, .env file, or environment."""
    if cli_value:
        return cli_value
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if key:
        return key
    raise SystemExit(
        "ERROR: No Google Maps API key found.\n"
        "Provide one via --api-key, the GOOGLE_MAPS_API_KEY env var, "
        "or a .env file in the repo root (see gmaps_pipeline/.env.example)."
    )


def _dataset_is_complete(out: Path) -> bool:
    """Return True if the output directory already has all expected files."""
    required = [
        out / "transforms_train.json",
        out / "points3D.txt",
    ]
    if not all(f.exists() for f in required):
        return False
    img_dir = out / "images"
    if not img_dir.is_dir():
        return False
    return any(img_dir.iterdir())


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(
        description="Generate a Skyfall-GS dataset from Google Maps satellite tiles.",
    )
    p.add_argument("--lat", type=float, required=True,
                   help="Center latitude of the area of interest")
    p.add_argument("--lon", type=float, required=True,
                   help="Center longitude of the area of interest")
    p.add_argument("--radius", type=float, default=500.0,
                   help="Radius around the center in meters (default: 500)")
    p.add_argument("--zoom", type=int, default=19,
                   help="Tile zoom level, 1-22 (default: 19, ~0.3 m/px)")
    p.add_argument("--api-key", default=None,
                   help="Google Maps Platform API key "
                        "(default: reads GOOGLE_MAPS_API_KEY from .env or environment)")
    p.add_argument("--elevation-source", default="opentopodata",
                   choices=ELEV_PROVIDERS,
                   help="Elevation data provider: opentopodata (free, default), "
                        "flat (no network, all z=0), google (paid, needs API key)")
    p.add_argument("--grid-size", type=int, default=100,
                   help="Elevation grid N×N (default: 100 → 10 000 points)")
    p.add_argument("--view-size", type=int, default=512,
                   help="Training image size in pixels (default: 512)")
    p.add_argument("--view-stride", type=int, default=384,
                   help="Stride between views; smaller → more overlap (default: 384)")
    p.add_argument("--fov", type=float, default=15.0,
                   help="Simulated camera vertical FoV in degrees (default: 15)")
    p.add_argument("--test-ratio", type=float, default=0.1,
                   help="Fraction of views held out for testing (default: 0.1)")
    p.add_argument("-o", "--output", required=True,
                   help="Output directory, e.g. data/my_scene")
    p.add_argument("--force", action="store_true",
                   help="Re-generate the dataset even if it already exists "
                        "(cached tiles/elevation are still reused)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    out = Path(args.output)

    if not args.force and _dataset_is_complete(out):
        log.info(
            "Dataset already exists at %s — skipping. "
            "Use --force to regenerate.",
            out,
        )
        return

    api_key = _resolve_api_key(args.api_key)
    cache_dir = out / ".cache"

    log.info("Creating Maps Tiles API session …")
    session = create_session(api_key)

    log.info("Fetching satellite mosaic …")
    mosaic, meta = fetch_mosaic(
        args.lat, args.lon, args.radius, args.zoom,
        api_key, session,
        cache_dir=cache_dir,
    )
    log.info(
        "Mosaic size: %d × %d px  (%.2f m/px)",
        meta["mosaic_size"]["width"],
        meta["mosaic_size"]["height"],
        meta["resolution"],
    )

    log.info("Fetching elevation data (%s) …", args.elevation_source)
    elev = fetch_elevation_grid(
        meta, args.grid_size,
        provider=args.elevation_source,
        api_key=api_key,
        cache_dir=cache_dir,
    )
    log.info(
        "Elevation range: %.1f – %.1f m  (center %.1f m)",
        elev["elevations"].min(),
        elev["elevations"].max(),
        elev["center_elevation"],
    )

    log.info("Building dataset …")
    build_dataset(
        mosaic, meta, elev, args.output,
        view_size=args.view_size,
        view_stride=args.view_stride,
        fov_deg=args.fov,
        test_ratio=args.test_ratio,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
