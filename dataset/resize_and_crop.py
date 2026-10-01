#!/usr/bin/env python3
"""Resize the shortest image side and take a deterministic centered crop."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a directory of images to lossless square PNGs by resizing "
            "the shortest side and then taking a centered crop."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256, help="Output side length.")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=30_000,
        help="Required input count; set to 0 to disable this check.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000,
        help="Print progress after this many images; set to 0 to disable.",
    )
    return parser.parse_args()


def resize_and_center_crop(image: Image.Image, size: int) -> Image.Image:
    """Match the MSCOCO-30K preprocessing used for the released evaluation."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")

    # Resize the shortest side to `size` while preserving the aspect ratio.
    # int() deliberately matches the floor rounding used to create the dataset.
    if width < height:
        new_width = size
        new_height = int(height * size / width)
    else:
        new_height = size
        new_width = int(width * size / height)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Use integer-centered offsets, matching the original preprocessing script.
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    return image.crop((left, top, left + size, top + size))


def main() -> None:
    args = parse_args()
    if args.size < 1:
        raise SystemExit("--size must be positive")
    if args.expected_count < 0:
        raise SystemExit("--expected-count cannot be negative")
    if args.progress_every < 0:
        raise SystemExit("--progress-every cannot be negative")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise SystemExit("Input and output directories must be different")

    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.expected_count and len(files) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} input images, found {len(files)} in {input_dir}"
        )

    output_names = [path.with_suffix(".png").name for path in files]
    if len(output_names) != len(set(output_names)):
        raise SystemExit("Multiple input files would map to the same output PNG name")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(files)} images from {input_dir}")
    for index, source_path in enumerate(files, start=1):
        with Image.open(source_path) as source:
            image = resize_and_center_crop(source.convert("RGB"), args.size)
            image.save(output_dir / source_path.with_suffix(".png").name, format="PNG")
        if args.progress_every and index % args.progress_every == 0:
            print(f"  {index}/{len(files)} done")

    missing_outputs = [name for name in output_names if not (output_dir / name).is_file()]
    if missing_outputs:
        raise RuntimeError(f"Failed to write {len(missing_outputs)} output images")
    print(f"Done. Wrote {len(files)} RGB {args.size}x{args.size} PNGs to {output_dir}")


if __name__ == "__main__":
    main()
