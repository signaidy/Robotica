from __future__ import annotations

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "imagen"
INTERPOLATION_DIR = PROJECT_ROOT / "manipulated" / "interpolation"
ROTATION_DIR = PROJECT_ROOT / "manipulated" / "rotations"
ROTATE_SCALE_DIR = PROJECT_ROOT / "manipulated" / "RnS"
TEXT_ROTATE_SCALE_DIR = PROJECT_ROOT / "manipulated" / "RnSText"
TEXT_SCALE50_DIR = PROJECT_ROOT / "manipulated" / "TextInterpolate50"
TEXT_SCALE200_DIR = PROJECT_ROOT / "manipulated" / "TextInterpolate200"
SCALE_FACTOR = 2.0
TEXT_IMAGE_NAME = "text.png"

INTERPOLATIONS: list[tuple[str, int]] = [
    ("nearest", cv2.INTER_NEAREST),
    ("bilinear", cv2.INTER_LINEAR),
    ("bicubic", cv2.INTER_CUBIC),
]


def iter_images(folder: Path) -> list[Path]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    images: list[Path] = []
    for pattern in patterns:
        images.extend(folder.glob(pattern))
    return sorted(images)


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    return (scaled_width, scaled_height)


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR.resolve()}")

    INTERPOLATION_DIR.mkdir(parents=True, exist_ok=True)
    ROTATION_DIR.mkdir(parents=True, exist_ok=True)
    ROTATE_SCALE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_ROTATE_SCALE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_SCALE50_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_SCALE200_DIR.mkdir(parents=True, exist_ok=True)

    images = iter_images(INPUT_DIR)
    if not images:
        raise FileNotFoundError(f"No images found in: {INPUT_DIR.resolve()}")

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Skipping unreadable file: {image_path}")
            continue

        height, width = image.shape[:2]
        target_size = scaled_size(width, height, SCALE_FACTOR)
        is_text_image = image_path.name.lower() == TEXT_IMAGE_NAME

        for name, interpolation in INTERPOLATIONS:
            resized = cv2.resize(image, target_size, interpolation=interpolation)
            output_name = f"{image_path.stem}_{name}{image_path.suffix}"
            output_path = INTERPOLATION_DIR / output_name
            cv2.imwrite(str(output_path), resized)

        for angle in (45, 90):
            center = (width / 2.0, height / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            output_name = f"{image_path.stem}_rot{angle}{image_path.suffix}"
            output_path = ROTATION_DIR / output_name
            cv2.imwrite(str(output_path), rotated)

            for name, interpolation in INTERPOLATIONS:
                rotated_scaled = cv2.resize(
                    rotated,
                    target_size,
                    interpolation=interpolation,
                )
                output_name = (
                    f"{image_path.stem}_rot{angle}_{name}{image_path.suffix}"
                )
                output_path = ROTATE_SCALE_DIR / output_name
                cv2.imwrite(str(output_path), rotated_scaled)

        if is_text_image:
            angle = 90
            center = (width / 2.0, height / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            for name, interpolation in INTERPOLATIONS:
                rotated = cv2.warpAffine(
                    image,
                    matrix,
                    (width, height),
                    flags=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                rotated_scaled = cv2.resize(
                    rotated,
                    target_size,
                    interpolation=interpolation,
                )
                output_name = (
                    f"{image_path.stem}_rot{angle}_scale2x_{name}"
                    f"{image_path.suffix}"
                )
                output_path = TEXT_ROTATE_SCALE_DIR / output_name
                cv2.imwrite(str(output_path), rotated_scaled)

            size_50 = scaled_size(width, height, 0.5)
            size_200 = scaled_size(width, height, 2.0)

            for name, interpolation in INTERPOLATIONS:
                resized_50 = cv2.resize(image, size_50, interpolation=interpolation)
                output_name = (
                    f"{image_path.stem}_scale50_{name}{image_path.suffix}"
                )
                output_path = TEXT_SCALE50_DIR / output_name
                cv2.imwrite(str(output_path), resized_50)

                resized_200 = cv2.resize(image, size_200, interpolation=interpolation)
                output_name = (
                    f"{image_path.stem}_scale200_{name}{image_path.suffix}"
                )
                output_path = TEXT_SCALE200_DIR / output_name
                cv2.imwrite(str(output_path), resized_200)

    print(f"Saved resized images to: {INTERPOLATION_DIR.resolve()}")
    print(f"Saved rotated images to: {ROTATION_DIR.resolve()}")
    print(f"Saved rotated+scaled images to: {ROTATE_SCALE_DIR.resolve()}")
    print(f"Saved text rotate+scale images to: {TEXT_ROTATE_SCALE_DIR.resolve()}")
    print(f"Saved text 50% images to: {TEXT_SCALE50_DIR.resolve()}")
    print(f"Saved text 200% images to: {TEXT_SCALE200_DIR.resolve()}")


if __name__ == "__main__":
    main()
