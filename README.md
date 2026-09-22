## Checkpoints

Checkpoint download: [Google Drive](https://drive.google.com/drive/folders/15s0OFnre_H5kr-T7YsUopUJi4mXAFPQu?usp=drive_link)

Download the checkpoint for the target dataset and pass its path through
`--weight`. Checkpoints are not included in this repository.

## Environment

Python 3.8 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Install a CUDA-enabled PyTorch build from the official PyTorch instructions if
GPU inference is required. LPIPS may download the AlexNet backbone on its first
run.

## Data preparation

Prepare two folders with paired RGB images. Paths are supplied only at runtime;
no dataset path is hard-coded in the repository.

```text
path/to/low-light-images/
path/to/ground-truth-images/
```

Input and GT images should have matching filenames or matching numeric IDs.
Supported formats are PNG, JPG, JPEG, and BMP.

## Test

### LOL-v1

```bash
python test.py --dataset lolv1 --input-dir path/to/LOL-v1/low --gt-dir path/to/LOL-v1/high --weight weights/lolv1_best.pth --output-dir results/lolv1 --alpha-l 0.5 --device cuda:0
```

### LOL-v2-Real

```bash
python test.py --dataset lolv2-real --input-dir path/to/LOL-v2-Real/low --gt-dir path/to/LOL-v2-Real/high --weight weights/lolv2real_best.pth --output-dir results/lolv2-real --alpha-l 0.5 --device cuda:0
```

### LOL-v2-Synthetic

```bash
python test.py --dataset lolv2-synthetic --input-dir path/to/LOL-v2-Synthetic/low --gt-dir path/to/LOL-v2-Synthetic/high --weight weights/lolv2syn_best.pth --output-dir results/lolv2-synthetic --alpha-l 0.75 --device cuda:0
```

### SID (Sony-Total-Dark)

```bash
python test.py --dataset sid --input-dir path/to/SID/eval/short --gt-dir path/to/SID/eval/long --weight weights/sid_best.pth --output-dir results/sid --alpha-l 0.5 --device cuda:0
```

## Reference results

| Dataset | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| LOL-v1 | 25.290 | 0.870 | 0.074 |
| LOL-v2-Real | 24.006 | 0.875 | 0.105 |
| LOL-v2-Synthetic | 25.977 | 0.939 | 0.044 |
| SID (Sony-Total-Dark) | 22.919 | 0.684 | 0.399 |

The implementation reports RGB PSNR, channel-averaged RGB SSIM, and AlexNet
LPIPS. Enhanced images are aligned to the GT resolution before metric
calculation, matching the released evaluation protocol.

## License

This code is released under the MIT License. See `LICENSE`.
