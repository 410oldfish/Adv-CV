from pathlib import Path
import re
from typing import Callable, List, Tuple

from PIL import Image, ImageEnhance, ImageFilter

# -------------------- Augmentations --------------------

def aug_identity(img: Image.Image) -> Image.Image:
    return img.copy()

def aug_hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

def aug_vflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

def aug_rotate_p15(img: Image.Image) -> Image.Image:
    return img.rotate(+15, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))

def aug_rotate_n15(img: Image.Image) -> Image.Image:
    return img.rotate(-15, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))

def aug_tx_right2(img: Image.Image) -> Image.Image:
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, 2, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )

def aug_ty_down2(img: Image.Image) -> Image.Image:
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, 0, 0, 1, 2),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )



AUGS: List[Tuple[str, Callable[[Image.Image], Image.Image]]] = [
    ("orig",            aug_identity),
    ("hflip",           aug_hflip),
    # ("vflip",           aug_vflip),
    ("rotp15",          aug_rotate_p15),
    ("rotn15",          aug_rotate_n15),
    # ("txr2",            aug_tx_right2),
    # ("tyd2",            aug_ty_down2)
]

# -------------------- Filename helpers --------------------

EXT_RE = re.compile(r'(\.(jpg|jpeg|png|bmp|gif|webp))+$', flags=re.IGNORECASE)

def strip_all_suffixes(name: str) -> str:
    """Remove chained extensions like '.png.jpg' -> base name without extension(s)."""
    return EXT_RE.sub('', name)

def infer_class(name_no_ext: str) -> str:
    """Class is the token before the first underscore."""
    return name_no_ext.split('_', 1)[0].lower()

def build_output_stem(orig_name: str) -> Tuple[str, str]:
    """
    Returns (cls, base_stem_without_exts) ensuring the class token is first.
    Example:
      'airplane_aeroplane_s_000002.png.jpg' -> ('airplane', 'airplane_aeroplane_s_000002')
    """
    base = strip_all_suffixes(orig_name)
    cls = infer_class(base)
    parts = base.split('_')
    if parts and parts[0].lower() == cls:
        stem = base
    else:
        stem = f"{cls}_{base}"
    return cls, stem

# -------------------- Main driver --------------------

def enhance_dataset_separate(
    input_dir: str,
    output_dir: str,
    write_original_clean_copy: bool = False
) -> None:
    """
    Reads RGB 32x32 images from input_dir.
    For each input image, applies each augmentation independently (no chaining, no randomness).
    Writes outputs to output_dir as JPEGs, keeping class prefix and adding an augmentation suffix.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    files = [p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if not files:
        print(f"No images found in {input_dir}")
        return

    total_written = 0
    for p in files:
        try:
            with Image.open(p) as img:
                # Assumptions: RGB and 32x32
                if img.mode != "RGB":
                    # If your dataset is guaranteed RGB, this should never hit.
                    img = img.convert("RGB")
                if img.size != (32, 32):
                    # If your dataset is guaranteed 32x32, this should never hit.
                    img = img.resize((32, 32), Image.BILINEAR)

                _, base_stem = build_output_stem(p.name)

                for suffix, fn in AUGS:
                    if suffix == "orig" and not write_original_clean_copy:
                        continue
                    out_img = fn(img)  # apply augmentation
                    out_name = f"{base_stem}_{suffix}.jpg"
                    out_img.save(out_path / out_name, format="JPEG", quality=95)
                    total_written += 1

        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    print(f"Done. Wrote {total_written} images to {out_path}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "data" / "train"
    output_dir = script_dir / "data" / "train_enhance"

    enhance_dataset_separate(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        write_original_clean_copy=True,
    )