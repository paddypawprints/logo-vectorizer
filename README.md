# png-vectorizer

Converts a raster PNG into a clean multi-layer SVG using divide-and-conquer
least-squares cubic Bézier fitting. No gradient descent — the fit is computed
analytically in a single pass, so there are no spikes and no hyperparameters
to tune.

## Algorithm

1. **Background removal** — flood-fill from corners to transparent (optional)
2. **Color quantization** — k-means clustering in LAB space → N color layers
3. **Contour tracing** — marching squares per region (outer boundary + holes)
4. **Corner detection** — curvature profile; weak corners coalesced away
5. **D&C Bézier fitting** — each corner-to-corner arc is fitted with the
   fewest cubic Bézier spans that keep max per-point error ≤ `MAX_BEZIER_ERR_PX`
   (default 2px). Split point at worst-fitting arc position, recurse.
   Control points solved via `np.linalg.lstsq` with chord-length
   parameterization — guaranteed spike-free global optimum.
6. **SVG export** — one `<path>` per region with `fill-rule="evenodd"` for holes;
   preview PNG and pixel-diff PNG written alongside the SVG.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Apple Silicon: optional MLX for GPU-accelerated corner coalescing
pip install mlx
```

## Usage

```bash
# Basic
python fit_all.py --input logo.png --output logo.svg

# Remove background before vectorizing
python fit_all.py --input logo.png --flood-bg

# Tune color count and minimum region size
python fit_all.py --input logo.png --colors 5 --min-px 500

# Custom output names
python fit_all.py --input logo.png --output out.svg --preview out_preview.png --diff out_diff.png
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `logo-plain-transparent.png` | Input PNG |
| `--output` | `taos_pass1.svg` | Output SVG |
| `--preview` | `taos_pass1_preview.png` | Rasterized SVG preview |
| `--diff` | `taos_pass1_diff.png` | 4× pixel diff vs quantized reference |
| `--colors` | `8` | Number of colors to quantize to |
| `--min-px` | `300` | Ignore regions smaller than this many pixels |
| `--flood-bg` | off | Flood-fill background to transparent before vectorizing |
| `--flood-tol` | `40` | RGB tolerance for `--flood-bg` |
| `--quiet` | off | Suppress per-region progress output |

## Tunable parameters

At the top of `fit_all.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `N_PIX` | `10` | Max anchor spacing = √2 × N_PIX px |
| `SHARP_THRESH` | `40.0°` | Curvature above this → sharp corner (C0 kink allowed) |
| `MAX_BEZIER_ERR_PX` | `2.0` | Max per-point error before D&C splits a span |
| `DILATION_PX` | `1` | Mask dilation before contouring (fills 1px gaps) |

## License

MIT — see [LICENSE](LICENSE).
