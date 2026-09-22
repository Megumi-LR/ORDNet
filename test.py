import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from metrics import image_files, evaluate_folder
from net import ORDNet


DATASET_ALIASES = {
    "lol_v1": "lolv1",
    "lolv1": "lolv1",
    "lol_v2_real": "lolv2-real",
    "lolv2_real": "lolv2-real",
    "lolv2-real": "lolv2-real",
    "lol_v2_syn": "lolv2-synthetic",
    "lolv2_syn": "lolv2-synthetic",
    "lolv2-syn": "lolv2-synthetic",
    "lolv2-synthetic": "lolv2-synthetic",
    "sid": "sid",
}


def retention_factor(value):
    factor = float(value)
    if not 0.0 <= factor <= 1.0:
        raise argparse.ArgumentTypeError("the value must be within [0, 1]")
    return factor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ORDNet inference and optionally reproduce PSNR/SSIM/LPIPS."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["lolv1", "lolv2-real", "lolv2-synthetic", "sid"],
        help="Selects the checkpoint-compatible inference post-processing.",
    )
    parser.add_argument("--input-dir", required=True, help="Folder of low-light RGB images.")
    parser.add_argument("--gt-dir", default=None, help="Optional folder of reference RGB images.")
    parser.add_argument("--weight", required=True, help="Path to the downloaded checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Folder for enhanced PNG images.")
    parser.add_argument(
        "--alpha-l",
        type=retention_factor,
        default=0.5,
        help="Luminance high-frequency retention factor (default: 0.5).",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:0."
    )
    return parser.parse_args()


def resolve_device(spec):
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("The checkpoint does not contain a valid state_dict")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return checkpoint, state_dict


def validate_dataset(checkpoint, requested):
    if not isinstance(checkpoint, dict) or "dataset" not in checkpoint:
        return
    saved = DATASET_ALIASES.get(str(checkpoint["dataset"]).lower())
    if saved is not None and saved != requested:
        raise ValueError(
            f"Checkpoint dataset is {saved!r}, but --dataset is {requested!r}."
        )


def configure_postprocessing(model, dataset):
    model.trans.gated = dataset == "lolv1"
    model.trans.gated2 = dataset == "lolv2-real"
    model.trans.alpha = 0.84 if dataset == "lolv2-real" else 1.0


def image_to_tensor(path, device):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def save_tensor(tensor, path):
    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    image = Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")
    image.save(path)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    input_files = image_files(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, state_dict = load_checkpoint(args.weight)
    validate_dataset(checkpoint, args.dataset)
    model = ORDNet(alpha_l=args.alpha_l)
    model.load_state_dict(state_dict, strict=True)
    configure_postprocessing(model, args.dataset)
    model.to(device).eval()

    print(f"Device: {device}")
    print(f"Images: {len(input_files)}")
    print(f"alpha_L: {args.alpha_l}")
    with torch.inference_mode():
        for source in tqdm(input_files, desc="Inference"):
            prediction = model(image_to_tensor(source, device))
            save_tensor(prediction, output_dir / f"{source.stem}.png")

    print(f"Enhanced images: {output_dir}")
    if args.gt_dir is not None:
        metrics_path = output_dir / "metrics.json"
        summary = evaluate_folder(output_dir, args.gt_dir, device, metrics_path)
        print(f"PSNR: {summary['psnr_db']:.4f} dB")
        print(f"SSIM: {summary['ssim']:.4f}")
        print(f"LPIPS-Alex: {summary['lpips_alex']:.4f}")
        print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
