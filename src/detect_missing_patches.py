"""Find thick, high-error patches between a quantized reference and an SVG render.

Usage:
    python src/detect_missing_patches.py \
        --reference uzumaki_quantized.png \
        --render uzumaki_preview.png \
        --overlay /tmp/uzumaki_patches.png \
        --report /tmp/uzumaki_patches.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def find_patches(reference, rendered, error_threshold, min_area, min_thickness):
    """Return high-error connected components with their size and thickness."""
    ref = np.asarray(Image.open(reference).convert("RGBA"), dtype=np.int16)
    pred = np.asarray(Image.open(rendered).convert("RGBA"), dtype=np.int16)
    if ref.shape != pred.shape:
        raise ValueError(
            f"Image dimensions differ: reference {ref.shape[1]}x{ref.shape[0]}, "
            f"render {pred.shape[1]}x{pred.shape[0]}"
        )

    visible = (ref[:, :, 3] > 0) | (pred[:, :, 3] > 0)
    rgb_error = np.abs(ref[:, :, :3] - pred[:, :, :3]).max(axis=2)
    alpha_error = np.abs(ref[:, :, 3] - pred[:, :, 3])
    error = np.maximum(rgb_error, alpha_error)
    threshold_mask = visible & (error > error_threshold)

    connectivity = np.ones((3, 3), dtype=bool)
    _, component_count = ndi.label(threshold_mask, structure=connectivity)
    distances = ndi.distance_transform_edt(threshold_mask)
    thick_mask = threshold_mask & (distances >= min_thickness / 2)
    core_labels, _ = ndi.label(thick_mask, structure=connectivity)

    # Grow each thick core back over the pixels the thickness test eroded, so
    # area/bbox describe the whole patch while thin bridges stay excluded.
    grow_dist, nearest = ndi.distance_transform_edt(
        core_labels == 0, return_indices=True)
    grown = core_labels[nearest[0], nearest[1]]
    labels = np.where(threshold_mask & (grow_dist <= min_thickness / 2 + 1.5),
                      grown, 0)
    objects = ndi.find_objects(labels)
    patches = []
    for label, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        y_slice, x_slice = bounds
        component = labels[bounds] == label
        area = int(component.sum())
        if area < min_area:
            continue

        local_y, local_x = np.nonzero(component)
        global_y = local_y + y_slice.start
        global_x = local_x + x_slice.start
        ref_visible = ref[global_y, global_x, 3] > 0
        ref_colors = ref[global_y[ref_visible], global_x[ref_visible], :3]
        if len(ref_colors):
            colors, color_counts = np.unique(ref_colors, axis=0, return_counts=True)
            dominant_color = "#{:02x}{:02x}{:02x}".format(*colors[color_counts.argmax()])
        else:
            dominant_color = ""

        radius = float(distances[bounds][component].max())
        patches.append({
            "id": label,
            "area_px": area,
            "max_thickness_px": round(2 * radius, 2),
            "mean_error": round(float(error[global_y, global_x].mean()), 2),
            "x_min": int(global_x.min()),
            "y_min": int(global_y.min()),
            "x_max": int(global_x.max()) + 1,
            "y_max": int(global_y.max()) + 1,
            "reference_color": dominant_color,
        })

    patches.sort(key=lambda patch: (-patch["area_px"], -patch["max_thickness_px"]))
    return patches, int(threshold_mask.sum()), component_count, labels


def write_overlay(render_path, output_path, patches, labels):
    rendered = Image.open(render_path).convert("RGBA")
    background = Image.new("RGBA", rendered.size, (0, 0, 0, 255))
    canvas = Image.alpha_composite(background, rendered)

    tint = Image.new("RGBA", canvas.size, (255, 0, 0, 0))
    tint_pixels = np.asarray(tint).copy()
    for patch in patches:
        x0, y0, x1, y1 = (
            patch["x_min"], patch["y_min"], patch["x_max"], patch["y_max"]
        )
        area = labels[y0:y1, x0:x1] == patch["id"]
        region = tint_pixels[y0:y1, x0:x1]
        region[area] = (255, 0, 0, 90)
    tint = Image.fromarray(tint_pixels)
    canvas = Image.alpha_composite(canvas, tint)

    draw = ImageDraw.Draw(canvas)
    for patch in patches:
        box = (patch["x_min"], patch["y_min"], patch["x_max"] - 1, patch["y_max"] - 1)
        draw.rectangle(box, outline=(255, 40, 40, 255), width=2)
        draw.text((box[0] + 3, box[1] + 3), str(patch["id"]), fill=(255, 40, 40, 255))
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="uzumaki_quantized.png",
                        help="Quantized target image")
    parser.add_argument("--render", default="uzumaki_preview.png",
                        help="Rasterized SVG preview")
    parser.add_argument("--threshold", type=int, default=30,
                        help="Minimum per-pixel RGB/alpha error (0-255)")
    parser.add_argument("--min-area", type=int, default=24,
                        help="Minimum connected-patch area in pixels")
    parser.add_argument("--min-thickness", type=float, default=5.0,
                        help="Minimum maximum inscribed diameter in pixels")
    parser.add_argument("--overlay", default="missing_patches.png",
                        help="Output image with candidate bounding boxes")
    parser.add_argument("--report", default="missing_patches.csv",
                        help="Output CSV with detected component measurements")
    args = parser.parse_args()
    if args.threshold < 0 or args.threshold > 255:
        parser.error("--threshold must be between 0 and 255")
    if args.min_area < 1 or args.min_thickness < 0:
        parser.error("--min-area must be positive and --min-thickness nonnegative")

    patches, high_error_pixels, component_count, labels = find_patches(
        args.reference, args.render, args.threshold, args.min_area, args.min_thickness
    )
    fields = [
        "id", "area_px", "max_thickness_px", "mean_error",
        "x_min", "y_min", "x_max", "y_max", "reference_color",
    ]
    with Path(args.report).open("w", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(patches)
    write_overlay(args.render, args.overlay, patches, labels)

    print(f"High-error pixels: {high_error_pixels}")
    print(f"High-error components before thickness filter: {component_count}")
    print(f"Thick patches at least {args.min_area}px: {len(patches)}")
    print(f"Overlay: {args.overlay}")
    print(f"Report: {args.report}")
    print("\nLargest candidates:")
    for patch in patches[:20]:
        print(
            f"  #{patch['id']}: area={patch['area_px']}px, "
            f"thickness={patch['max_thickness_px']}px, "
            f"bbox=({patch['x_min']},{patch['y_min']})-"
            f"({patch['x_max']},{patch['y_max']}), "
            f"color={patch['reference_color']}"
        )


if __name__ == "__main__":
    main()
