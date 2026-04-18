# Google Maps Pipeline for Skyfall-GS

Download satellite imagery and elevation data from Google Maps APIs, then
transform them into the dataset format expected by Skyfall-GS for 3D urban
scene reconstruction.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Same environment as Skyfall-GS |
| Google Maps Platform API key | Must have **Map Tiles API** and **Elevation API** enabled |
| Billing enabled on your GCP project | Both APIs are pay-per-use (see [pricing](https://developers.google.com/maps/documentation/tile/usage-and-billing)) |

## Installation

From the repository root, install the pipeline's lightweight dependencies
(these do **not** conflict with Skyfall-GS's own `requirements.txt`):

```bash
pip install -r gmaps_pipeline/requirements.txt
```

## API Key Setup

Copy the example env file to the repo root and fill in your key:

```bash
cp gmaps_pipeline/.env.example .env
```

Then edit `.env`:

```
GOOGLE_MAPS_API_KEY=your_actual_key_here
```

The `.env` file is already in `.gitignore` so your key will never be committed.
You can also pass `--api-key` on the command line or export the variable in
your shell — the lookup order is: `--api-key` flag > env var > `.env` file.

## Quick Start

```bash
python -m gmaps_pipeline \
    --lat 40.748817 --lon -73.985428 \
    --radius 300 \
    --zoom 19 \
    -o data/my_scene
```

Then train Skyfall-GS on the result:

```bash
python train.py \
    -s data/my_scene \
    -m outputs/my_scene \
    --eval --port 6209 \
    --kernel_size 0.1 --resolution 1 --sh_degree 1 \
    --appearance_enabled \
    --lambda_depth 0 --lambda_opacity 10 \
    --densify_until_iter 21000 --densify_grad_threshold 0.0001 \
    --lambda_pseudo_depth 0.5 \
    --start_sample_pseudo 1000 --end_sample_pseudo 21000 \
    --size_threshold 20 --scaling_lr 0.001 --rotation_lr 0.001 \
    --opacity_reset_interval 3000 --sample_pseudo_interval 10
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--lat` | *(required)* | Center latitude |
| `--lon` | *(required)* | Center longitude |
| `--radius` | 500 | Area radius in meters |
| `--zoom` | 19 | Tile zoom level (higher = more detail; 19 ≈ 0.3 m/px) |
| `--api-key` | `.env` / env var | Google Maps Platform API key (see [API Key Setup](#api-key-setup)) |
| `--grid-size` | 100 | Elevation sample grid N×N (100 → 10 000 points) |
| `--view-size` | 512 | Output image size in pixels |
| `--view-stride` | 384 | Stride between overlapping views (smaller → more overlap) |
| `--fov` | 15.0 | Simulated camera field-of-view in degrees |
| `--test-ratio` | 0.1 | Fraction of views held out for testing |
| `-o / --output` | *(required)* | Output directory (e.g. `data/my_scene`) |

## Output Format

The pipeline writes exactly the directory layout that `readSatelliteInfo` in
`scene/dataset_readers.py` expects:

```
data/my_scene/
├── images/
│   ├── view_0000.png
│   ├── view_0001.png
│   └── …
├── masks/
│   ├── view_0000.npy
│   ├── view_0000.png
│   └── …
├── transforms_train.json
├── transforms_test.json
└── points3D.txt
```

### `transforms_train.json`

Contains a normalisation rotation `R` (identity) and translation `T` (zero),
plus per-frame entries with:

- `file_path` — relative path to the image
- `transform_matrix_rotated` — 4×4 camera-to-world matrix (COLMAP convention)
- `fl_x`, `fl_y` — focal length in pixels
- `cx`, `cy` — principal point in pixels

### `points3D.txt`

COLMAP text format: `POINT3D_ID X Y Z R G B ERROR`

Point positions are in a local East-North-Up coordinate system (meters)
centered on the requested lat/lon.  Colour is sampled from the satellite
mosaic.

## How It Works

1. **Tile fetch** — Downloads satellite imagery tiles from the
   [Google Maps Tiles API](https://developers.google.com/maps/documentation/tile/satellite)
   and stitches them into a high-resolution mosaic.
2. **Elevation fetch** — Queries the
   [Google Elevation API](https://developers.google.com/maps/documentation/elevation/overview)
   on a regular grid covering the mosaic.
3. **View generation** — Slides an overlapping window across the mosaic to
   produce training images, each assigned a nadir (straight-down) pinhole
   camera whose focal length and altitude are derived from the tile zoom
   level and the configured FoV.
4. **Point cloud** — Combines the elevation grid with colours sampled from
   the mosaic to write `points3D.txt`.
5. **Normalisation** — The identity `R` / zero `T` in the JSON triggers
   Skyfall-GS's built-in normalisation, which rescales the point cloud to a
   sphere of radius 256 and shifts the ground plane to z = 0.

## Tips

- **More overlap → better reconstruction.** Reducing `--view-stride` gives
  the optimiser more multi-view signal, at the cost of more training images.
- **Lower `--fov`** makes the cameras more orthographic (closer to the
  tile projection), which improves pixel-level consistency between views.
  Higher `--fov` introduces more perspective parallax, which can help
  depth estimation but slightly mismatches the orthorectified tile content.
- **Higher `--grid-size`** gives a denser initial point cloud, which can
  speed up Stage 1 convergence.
- **Stage 2 (IDU)** is where Skyfall-GS generates novel street-level views
  via diffusion, so even all-nadir input works well for the full pipeline.

## Limitations

- Google Maps tiles are **orthorectified** — every view is strictly top-down.
  True multi-angle satellite imagery (e.g. WorldView) provides stronger
  geometric cues; this pipeline compensates via overlapping crops and defers
  to IDU for novel-view synthesis.
- Elevation API accuracy varies by region (~1 m in dense urban areas,
  up to 30 m in remote terrain).
- Subject to [Google Maps Platform Terms of Service](https://cloud.google.com/maps-platform/terms).
