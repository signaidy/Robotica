#!/usr/bin/env python3
"""
Clean noisy text images (captcha-like) using OpenCV.

Usage:
  python3 procesamiento/captcha_cleaner.py \
    --input procesamiento/imagen/image.png \
    --output-dir procesamiento/output \
    --method adaptive --deskew --debug
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, median_ksize: int, bilateral: bool) -> np.ndarray:
    out = cv2.medianBlur(gray, median_ksize)
    if bilateral:
        out = cv2.bilateralFilter(out, d=7, sigmaColor=50, sigmaSpace=50)
    return out


def threshold_image(gray: np.ndarray, method: str, block_size: int, c: int) -> np.ndarray:
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if method == "adaptive":
        # block_size must be odd and > 1
        block_size = max(3, block_size | 1)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )
    raise ValueError(f"Unknown method: {method}")


def infer_text_is_white(binary: np.ndarray) -> bool:
    white_ratio = np.mean(binary == 255)
    # If background is dark, white pixels will be fewer and likely text.
    return white_ratio < 0.5


def remove_small_components(fg: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return fg
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = np.zeros_like(fg)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label] = 255
    return out


def deskew_binary(fg: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(fg > 0))
    if coords.size < 20:
        return fg
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # OpenCV returns angle in [-90, 0); correct to small rotation
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return fg
    (h, w) = fg.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(fg, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)


def save_debug(debug_dir: Path, name: str, img: np.ndarray) -> None:
    ensure_dir(debug_dir)
    cv2.imwrite(str(debug_dir / f"{name}.png"), img)


def prepare_for_ocr(
    image: np.ndarray,
    scale: float,
    border: int,
    pre_blur: int,
    post_blur: int,
    interp: str,
    erode: int,
    dilate: int,
    sharpen: float,
    sharpen_ksize: int,
) -> np.ndarray:
    out = image
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    if pre_blur and pre_blur > 0:
        k = max(3, pre_blur | 1)
        out = cv2.GaussianBlur(out, (k, k), 0)

    if sharpen and sharpen > 0:
        k = max(3, sharpen_ksize | 1)
        blur = cv2.GaussianBlur(out, (k, k), 0)
        out = cv2.addWeighted(out, 1.0 + sharpen, blur, -sharpen, 0)

    if scale and scale != 1.0:
        h, w = out.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if interp == "lanczos":
            interp_flag = cv2.INTER_LANCZOS4
        elif interp == "linear":
            interp_flag = cv2.INTER_LINEAR
        elif interp == "area":
            interp_flag = cv2.INTER_AREA
        else:
            interp_flag = cv2.INTER_CUBIC
        out = cv2.resize(out, (new_w, new_h), interpolation=interp_flag)

    if border > 0:
        out = cv2.copyMakeBorder(
            out,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    if post_blur and post_blur > 0:
        k = max(3, post_blur | 1)
        out = cv2.GaussianBlur(out, (k, k), 0)

    if np.any((out != 0) & (out != 255)):
        _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if erode > 0 or dilate > 0:
        inv = cv2.bitwise_not(out)
        if erode > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
            inv = cv2.erode(inv, k, iterations=1)
        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
            inv = cv2.dilate(inv, k, iterations=1)
        out = cv2.bitwise_not(inv)

    return out


def build_tess_config(psm: int, oem: int, whitelist: str, no_dawg: bool) -> str:
    config_parts = []
    if oem is not None:
        config_parts.append(f"--oem {oem}")
    if psm is not None:
        config_parts.append(f"--psm {psm}")
    if whitelist:
        config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
    if no_dawg:
        config_parts.append("-c load_system_dawg=0 -c load_freq_dawg=0")
    return " ".join(config_parts)


def clean_ocr_text(text: str, whitelist: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if whitelist:
        cleaned = "".join(ch for ch in cleaned if ch in whitelist)
    return cleaned


def merge_dot_components(boxes: list[dict]) -> list[dict]:
    if len(boxes) < 2:
        return boxes

    heights = [b["h"] for b in boxes]
    areas = [b["area"] for b in boxes]
    med_h = float(np.median(heights)) if heights else 1.0
    med_area = float(np.median(areas)) if areas else 1.0

    merged = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if b["area"] >= 0.35 * med_area or b["h"] >= 0.6 * med_h:
            continue

        best_j = None
        best_gap = None
        for j, c in enumerate(boxes):
            if i == j:
                continue
            if c["y"] <= b["y"]:
                continue
            x_overlap = max(
                0,
                min(b["x"] + b["w"], c["x"] + c["w"]) - max(b["x"], c["x"]),
            )
            if x_overlap <= 0:
                continue
            overlap_ratio = x_overlap / max(1, min(b["w"], c["w"]))
            if overlap_ratio < 0.2:
                continue
            gap = c["y"] - (b["y"] + b["h"])
            if gap < 0 or gap > med_h:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_j = j

        if best_j is not None:
            c = boxes[best_j]
            x1 = min(b["x"], c["x"])
            y1 = min(b["y"], c["y"])
            x2 = max(b["x"] + b["w"], c["x"] + c["w"])
            y2 = max(b["y"] + b["h"], c["y"] + c["h"])
            c["x"], c["y"], c["w"], c["h"] = x1, y1, x2 - x1, y2 - y1
            c["area"] += b["area"]
            merged[i] = True

    return [b for i, b in enumerate(boxes) if not merged[i]]


def count_acute_angles(binary: np.ndarray, epsilon_ratio: float, acute_deg: float) -> int:
    inv = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return 0

    eps = max(1.0, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 3:
        return 0

    count = 0
    for i in range(len(approx)):
        p0 = approx[i - 1][0].astype(np.float32)
        p1 = approx[i][0].astype(np.float32)
        p2 = approx[(i + 1) % len(approx)][0].astype(np.float32)
        v1 = p0 - p1
        v2 = p2 - p1
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = max(-1.0, min(1.0, cosang))
        angle = float(np.degrees(np.arccos(cosang)))
        if angle < acute_deg:
            count += 1
    return count


def is_i_like(binary: np.ndarray) -> bool:
    def components(fg: np.ndarray):
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
        comps = []
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            cx, cy = centroids[idx]
            comps.append({"x": x, "y": y, "w": w, "h": h, "area": area, "cx": cx, "cy": cy})
        return comps

    def dot_stem_ok(comps: list[dict]) -> bool:
        if len(comps) != 2:
            return False
        comps = sorted(comps, key=lambda c: c["area"])
        dot, stem = comps[0], comps[1]
        if dot["area"] > stem["area"] * 0.6:
            return False
        if dot["y"] >= stem["y"]:
            return False
        if dot["y"] + dot["h"] > stem["y"] + stem["h"] * 0.45:
            return False
        x_tol = max(stem["w"] * 0.9, 2.0)
        if abs(dot["cx"] - stem["cx"]) > x_tol:
            return False
        return True

    fg = (binary == 0).astype(np.uint8) * 255
    comps = components(fg)
    if dot_stem_ok(comps):
        return True

    # Try a light erosion to separate a weakly connected dot.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(fg, k, iterations=1)
    comps = components(eroded)
    if dot_stem_ok(comps):
        return True

    # Projection heuristic: two foreground clusters separated by a gap.
    h, w = binary.shape[:2]
    fg01 = (binary == 0).astype(np.uint8)
    row_sum = fg01.sum(axis=1)
    if row_sum.size == 0:
        return False
    thresh = max(1, int(round(0.1 * float(row_sum.max()))))
    active = row_sum > thresh
    segments = []
    start = None
    for i, val in enumerate(active):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(active) - 1))
    if len(segments) >= 2:
        top = segments[0]
        bottom = segments[-1]
        gap = bottom[0] - top[1] - 1
        top_h = top[1] - top[0] + 1
        bottom_h = bottom[1] - bottom[0] + 1
        width_active = int(np.count_nonzero(fg01.sum(axis=0)))
        if gap >= max(1, int(round(0.02 * h))) and top_h <= max(2, int(round(0.5 * bottom_h))):
            if width_active <= max(3, int(round(0.8 * w))):
                return True

    return False


def has_full_height_left_stem(binary: np.ndarray, coverage_ratio: float = 0.75) -> bool:
    h, w = binary.shape[:2]
    if w <= 2:
        return False
    fg = (binary == 0).astype(np.uint8)
    search_cols = max(1, int(round(w * 0.25)))
    max_col = 0
    for x in range(search_cols):
        col_count = int(fg[:, x].sum())
        if col_count > max_col:
            max_col = col_count
    return max_col >= int(round(h * coverage_ratio))


def extract_char_boxes(binary: np.ndarray, min_area: int, merge_dots: bool) -> list[tuple]:
    fg = (binary == 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    boxes = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})

    if merge_dots:
        boxes = merge_dot_components(boxes)

    boxes.sort(key=lambda b: b["x"])
    return [(b["x"], b["y"], b["w"], b["h"]) for b in boxes]


def ocr_per_char(
    pytesseract,
    image: np.ndarray,
    lang: str,
    config: str,
    whitelist: str,
    char_psm: int,
    char_oem: int,
    no_dawg: bool,
    min_area: int,
    pad: int,
    merge_dots: bool,
    digit_acute_min: int,
    digit_acute_angle: float,
    digit_acute_eps: float,
    debug_dir: Path | None,
) -> str:
    boxes = extract_char_boxes(image, min_area=min_area, merge_dots=merge_dots)
    if debug_dir is not None:
        box_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in boxes:
            cv2.rectangle(box_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        save_debug(debug_dir, "07_char_boxes", box_img)

    chars = []
    height, width = image.shape[:2]
    for x, y, w, h in boxes:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        crop = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(crop, lang=lang, config=config)
        cleaned = clean_ocr_text(text, whitelist)
        if cleaned:
            candidate = cleaned[0]
        else:
            candidate = ""

        if not candidate and is_i_like(crop):
            candidate = "i"

        if candidate:
            if candidate.isdigit() and digit_acute_min > 0:
                acute_count = count_acute_angles(
                    binary=crop,
                    epsilon_ratio=digit_acute_eps,
                    acute_deg=digit_acute_angle,
                )
                if acute_count < digit_acute_min:
                    letters_only = "".join(ch for ch in whitelist if ch.isalpha())
                    if letters_only:
                        letter_config = build_tess_config(
                            psm=char_psm,
                            oem=char_oem,
                            whitelist=letters_only,
                            no_dawg=no_dawg,
                        )
                        alt = pytesseract.image_to_string(crop, lang=lang, config=letter_config)
                        alt_clean = clean_ocr_text(alt, letters_only)
                        if alt_clean:
                            candidate = alt_clean[0]

            if candidate == "e" and has_full_height_left_stem(crop):
                candidate = "b"

        if candidate:
            chars.append(candidate)
    return "".join(chars)


def run_ocr(
    image: np.ndarray,
    lang: str,
    psm: int,
    oem: int,
    whitelist: str,
    tesseract_cmd: str,
    no_dawg: bool,
    ocr_scale: float,
    ocr_border: int,
    ocr_pre_blur: int,
    ocr_post_blur: int,
    ocr_interp: str,
    ocr_erode: int,
    ocr_dilate: int,
    ocr_sharpen: float,
    ocr_sharpen_ksize: int,
    per_char: bool,
    char_psm: int,
    char_min_area: int,
    char_pad: int,
    char_merge_dots: bool,
    digit_acute_min: int,
    digit_acute_angle: float,
    digit_acute_eps: float,
    debug_dir: Path | None,
) -> None:
    try:
        import pytesseract
    except ModuleNotFoundError:
        print("OCR skipped: pytesseract is not installed. Install with: pip install pytesseract")
        return

    ocr_img = prepare_for_ocr(
        image=image,
        scale=ocr_scale,
        border=ocr_border,
        pre_blur=ocr_pre_blur,
        post_blur=ocr_post_blur,
        interp=ocr_interp,
        erode=ocr_erode,
        dilate=ocr_dilate,
        sharpen=ocr_sharpen,
        sharpen_ksize=ocr_sharpen_ksize,
    )
    if debug_dir is not None:
        save_debug(debug_dir, "06_ocr", ocr_img)

    if not tesseract_cmd:
        tesseract_cmd = shutil.which("tesseract") or ""
        if not tesseract_cmd:
            win_paths = [
                Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
                Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
            ]
            for cand in win_paths:
                if cand.exists():
                    tesseract_cmd = str(cand)
                    break

    if not tesseract_cmd:
        print("OCR failed: tesseract is not installed or it's not in your PATH.")
        print("Tip: install Tesseract and/or set --tesseract-cmd to its path.")
        return

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    config = build_tess_config(psm=psm, oem=oem, whitelist=whitelist, no_dawg=no_dawg)

    try:
        if per_char:
            char_config = build_tess_config(psm=char_psm, oem=oem, whitelist=whitelist, no_dawg=no_dawg)
            text = ocr_per_char(
                pytesseract=pytesseract,
                image=ocr_img,
                lang=lang,
                config=char_config,
                whitelist=whitelist,
                char_psm=char_psm,
                char_oem=oem,
                no_dawg=no_dawg,
                min_area=char_min_area,
                pad=char_pad,
                merge_dots=char_merge_dots,
                digit_acute_min=digit_acute_min,
                digit_acute_angle=digit_acute_angle,
                digit_acute_eps=digit_acute_eps,
                debug_dir=debug_dir,
            )
        else:
            text = pytesseract.image_to_string(ocr_img, lang=lang, config=config)
    except Exception as exc:
        print(f"OCR failed: {exc}")
        print("Tip: install Tesseract and/or set --tesseract-cmd to its path.")
        return

    cleaned = text if per_char else clean_ocr_text(text, whitelist)
    if cleaned:
        label = "OCR (per-char)" if per_char else "OCR"
        print(f"{label}: {cleaned}")
    else:
        print("OCR: (no text detected)")


def process_image(
    input_path: Path,
    output_dir: Path,
    method: str,
    block_size: int,
    c: int,
    median_ksize: int,
    bilateral: bool,
    morph_open: int,
    morph_close: int,
    min_area: int,
    deskew: bool,
    debug: bool,
) -> Path:
    img = read_image(input_path)
    gray = to_gray(img)
    smooth = denoise(gray, median_ksize, bilateral)
    binary = threshold_image(smooth, method, block_size, c)
    text_is_white = infer_text_is_white(binary)

    # Ensure text is white for morphology
    fg = binary if text_is_white else cv2.bitwise_not(binary)

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open, morph_open))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close, morph_close))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

    fg = remove_small_components(fg, min_area)

    if deskew:
        fg = deskew_binary(fg)

    # Output black text on white background for readability / OCR
    out = cv2.bitwise_not(fg)

    ensure_dir(output_dir)
    out_path = output_dir / f"{input_path.stem}_clean.png"
    cv2.imwrite(str(out_path), out)

    if debug:
        debug_dir = output_dir / "steps"
        save_debug(debug_dir, "01_gray", gray)
        save_debug(debug_dir, "02_smooth", smooth)
        save_debug(debug_dir, "03_binary", binary)
        save_debug(debug_dir, "04_fg", fg)
        save_debug(debug_dir, "05_out", out)

    return out_path, out


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    image_dir = script_dir / "imagen"
    default_input = image_dir / "image.png"
    if image_dir.exists():
        pngs = sorted(image_dir.glob("*.png"))
        if pngs:
            default_input = pngs[0]
    default_output = script_dir / "output"
    parser = argparse.ArgumentParser(description="Clean noisy text images (captcha-like) using OpenCV.")
    parser.add_argument("--input", type=Path, default=default_input, help="Input image path")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Output directory")
    parser.add_argument("--method", choices=["adaptive", "otsu"], default="otsu")
    parser.add_argument("--block-size", type=int, default=21, help="Adaptive threshold block size")
    parser.add_argument("--c", type=int, default=10, help="Adaptive threshold C")
    parser.add_argument("--median-ksize", type=int, default=3, help="Median blur kernel size (odd)")
    parser.add_argument("--bilateral", action="store_true", help="Apply bilateral filter")
    parser.add_argument("--morph-open", type=int, default=2, help="Morph open kernel size (0 to disable)")
    parser.add_argument("--morph-close", type=int, default=2, help="Morph close kernel size (0 to disable)")
    parser.add_argument("--min-area", type=int, default=20, help="Remove components smaller than this area")
    parser.add_argument("--deskew", action="store_true", help="Estimate and correct rotation")
    parser.add_argument("--debug", action="store_true", help="Save intermediate images")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR output")
    parser.add_argument("--lang", type=str, default="eng", help="Tesseract language")
    parser.add_argument("--psm", type=int, default=7, help="Tesseract page segmentation mode")
    parser.add_argument("--oem", type=int, default=1, help="Tesseract engine mode")
    parser.add_argument(
        "--whitelist",
        type=str,
        default="abcdefghijklmnopqrstuvwxyz0123456789",
        help="Restrict OCR characters",
    )
    parser.add_argument("--tesseract-cmd", type=str, default="", help="Path to tesseract executable")
    parser.add_argument("--no-dawg", action="store_true", help="Disable Tesseract dictionaries")
    parser.add_argument("--with-dawg", action="store_false", dest="no_dawg", help="Enable dictionaries")
    parser.add_argument("--ocr-scale", type=float, default=3.0, help="Scale image before OCR")
    parser.add_argument("--ocr-border", type=int, default=10, help="Border (px) added before OCR")
    parser.add_argument("--ocr-pre-blur", type=int, default=0, help="Gaussian blur kernel before scaling (odd)")
    parser.add_argument("--ocr-post-blur", type=int, default=0, help="Gaussian blur kernel after scaling (odd)")
    parser.add_argument(
        "--ocr-interp",
        choices=["cubic", "lanczos", "linear", "area"],
        default="cubic",
        help="Interpolation for scaling before OCR",
    )
    parser.add_argument("--ocr-erode", type=int, default=0, help="Erode text before OCR")
    parser.add_argument("--ocr-dilate", type=int, default=0, help="Dilate text before OCR")
    parser.add_argument("--ocr-sharpen", type=float, default=0.8, help="Unsharp mask amount before OCR")
    parser.add_argument("--ocr-sharpen-ksize", type=int, default=5, help="Unsharp mask kernel size (odd)")
    parser.add_argument("--per-char", action="store_true", help="OCR each character separately")
    parser.add_argument("--char-psm", type=int, default=10, help="PSM for per-character OCR")
    parser.add_argument("--char-min-area", type=int, default=15, help="Min area for character components")
    parser.add_argument("--char-pad", type=int, default=2, help="Padding around each character crop")
    parser.add_argument("--digit-acute-min", type=int, default=2, help="Min acute angles required to keep digits")
    parser.add_argument("--digit-acute-angle", type=float, default=60.0, help="Angle threshold (deg) for acute")
    parser.add_argument("--digit-acute-eps", type=float, default=0.03, help="Contour approx epsilon ratio")
    parser.add_argument(
        "--no-char-merge-dot",
        action="store_false",
        dest="char_merge_dots",
        help="Disable merging dot with stem for i/j",
    )
    parser.set_defaults(char_merge_dots=True, per_char=True, no_dawg=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path, out_img = process_image(
        input_path=args.input,
        output_dir=args.output_dir,
        method=args.method,
        block_size=args.block_size,
        c=args.c,
        median_ksize=args.median_ksize,
        bilateral=args.bilateral,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        min_area=args.min_area,
        deskew=args.deskew,
        debug=args.debug,
    )
    print(f"Saved: {out_path}")
    if not args.no_ocr:
        run_ocr(
            image=out_img,
            lang=args.lang,
            psm=args.psm,
            oem=args.oem,
            whitelist=args.whitelist,
            tesseract_cmd=args.tesseract_cmd,
            no_dawg=args.no_dawg,
            ocr_scale=args.ocr_scale,
            ocr_border=args.ocr_border,
            ocr_pre_blur=args.ocr_pre_blur,
            ocr_post_blur=args.ocr_post_blur,
            ocr_interp=args.ocr_interp,
            ocr_erode=args.ocr_erode,
            ocr_dilate=args.ocr_dilate,
            ocr_sharpen=args.ocr_sharpen,
            ocr_sharpen_ksize=args.ocr_sharpen_ksize,
            per_char=args.per_char,
            char_psm=args.char_psm,
            char_min_area=args.char_min_area,
            char_pad=args.char_pad,
            char_merge_dots=args.char_merge_dots,
            digit_acute_min=args.digit_acute_min,
            digit_acute_angle=args.digit_acute_angle,
            digit_acute_eps=args.digit_acute_eps,
            debug_dir=(args.output_dir / "steps") if args.debug else None,
        )


if __name__ == "__main__":
    main()
