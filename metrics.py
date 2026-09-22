import json
import re
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No supported images found in: {folder}")
    return files


def pair_keys(path_or_name):
    path = Path(path_or_name)
    filename = path.name.lower()
    stem = path.stem.lower()
    keys = [filename, stem]
    stripped = re.sub(r"^(low|normal|high|gt|input|image|img)[_-]*", "", stem)
    if stripped != stem:
        keys.append(stripped)
    groups = re.findall(r"\d+", stem)
    if groups:
        digits = groups[-1]
        keys.extend([digits, digits.lstrip("0") or "0"])
    return list(dict.fromkeys(key for key in keys if key))


def build_index(folder):
    index = {}
    for path in image_files(folder):
        for key in pair_keys(path):
            index.setdefault(key, path)
    return index


def find_match(source, target_index):
    for key in pair_keys(source):
        if key in target_index:
            return target_index[key]
    raise FileNotFoundError(f"Could not match {Path(source).name} to a GT image")


def calculate_psnr(prediction, target):
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mse = np.mean(np.square(prediction - target))
    return 10.0 * np.log10(255.0 * 255.0 / (mse + 1e-8))


def _ssim_channel(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    score = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return score.mean()


def calculate_ssim(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Input images must have the same dimensions")
    if prediction.ndim == 2:
        return _ssim_channel(prediction, target)
    if prediction.ndim == 3 and prediction.shape[2] == 3:
        return float(np.mean([
            _ssim_channel(prediction[:, :, i], target[:, :, i]) for i in range(3)
        ]))
    raise ValueError(f"Unsupported image shape: {prediction.shape}")


def evaluate_folder(output_dir, gt_dir, device, json_path=None):
    output_files = image_files(output_dir)
    gt_index = build_index(gt_dir)
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    records = []

    for output_path in tqdm(output_files, desc="Metrics"):
        gt_path = find_match(output_path, gt_index)
        prediction_pil = Image.open(output_path).convert("RGB")
        target_pil = Image.open(gt_path).convert("RGB")

        # Paper protocol: align the enhanced result to the GT resolution.
        if prediction_pil.size != target_pil.size:
            prediction_pil = prediction_pil.resize(target_pil.size)
        prediction = np.asarray(prediction_pil)
        target = np.asarray(target_pil)

        psnr = float(calculate_psnr(prediction, target))
        ssim = float(calculate_ssim(prediction, target))
        pred_tensor = lpips.im2tensor(prediction).to(device)
        gt_tensor = lpips.im2tensor(target).to(device)
        with torch.inference_mode():
            lpips_score = float(loss_fn(gt_tensor, pred_tensor).item())

        records.append({
            "image": output_path.name,
            "gt": gt_path.name,
            "psnr_db": psnr,
            "ssim": ssim,
            "lpips_alex": lpips_score,
        })

    summary = {
        "images": len(records),
        "psnr_db": float(np.mean([item["psnr_db"] for item in records])),
        "ssim": float(np.mean([item["ssim"] for item in records])),
        "lpips_alex": float(np.mean([item["lpips_alex"] for item in records])),
        "metric_alignment": "output_to_gt",
        "per_image": records,
    }
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
