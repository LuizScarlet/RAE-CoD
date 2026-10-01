from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .schemas import ImageInfo


class ImageLoadError(ValueError):
    pass


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def load_image(path_like: str | Path) -> tuple[Image.Image, ImageInfo]:
    path = Path(path_like).expanduser().resolve()
    if not path.is_file():
        raise ImageLoadError(f"Image does not exist or is not a file: {path}")
    try:
        with Image.open(path) as opened:
            oriented = ImageOps.exif_transpose(opened)
            original_mode = oriented.mode
            image = _to_rgb(oriented).copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Cannot decode image {path}: {exc}") from exc
    info = ImageInfo(
        path=str(path),
        sha256=sha256_file(path),
        width=image.width,
        height=image.height,
        mode=original_mode,
    )
    return image, info
