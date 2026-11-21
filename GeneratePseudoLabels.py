"""Stage I: generate DESPL pseudo-labels from unlabeled RGB images.

Default output is an 8-bit grayscale encoding of the soft Eq. (10) average.
Use --binary-mask only when an additional hard output threshold is desired.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from pseudo_labeling.despl import DESPLConfig, generate_pseudo_label_with_augmentation


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=("Fixed paper settings: DINOv2 ViT-L/14, lambda=0.1, sigma=0.1, "
                "B=50, alpha=0.1, T=8, first two Laplacian eigenvectors. "
                "Outputs are original-size same-stem PNGs plus despl_manifest.json. "
                "No ground-truth labels are read. No connected-component or border "
                "post-processing is added. Eigenvector signs use the supplied source's "
                "joint-column convention, which the paper does not specify."))
    parser.add_argument("--data_path", type=Path, default=Path("./data/TrainDataset/Imgs"),
                        help="directory of unlabeled JPG/JPEG/PNG images with unique stems")
    parser.add_argument("--save_path", type=Path, default=Path("./data/TrainDataset/PseudoMask_Enhanced"),
                        help="output mask directory; existing results require --overwrite")
    parser.add_argument("--device", default="cuda", help="feature-extraction device, e.g. cuda:0 or cpu; spectral computation uses CPU")
    parser.add_argument("--imgsz", type=int, choices=(476,), default=476,
                        help="paper setting: 476 pixels, producing a 34 x 34 grid")
    parser.add_argument("--weights", type=Path, help="local DINOv2 ViT-L/14 state_dict; otherwise use official pretrained weights")
    parser.add_argument("--binary-mask", action="store_true", help="apply an extra >0.5 threshold after Eq. (10)")
    parser.add_argument("--save-npy", action="store_true", help="also save unquantized float32 Eq. (10) outputs as .npy files")
    parser.add_argument("--overwrite", action="store_true", help="regenerate and replace existing masks and manifest")
    return parser


def discover_images(directory):
    if not directory.is_dir():
        raise ValueError(f"Input image directory does not exist: {directory}")
    images = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise ValueError(f"No JPG/JPEG/PNG images found in {directory}")
    names = set()
    for path in images:
        key = path.stem.casefold()
        if key in names:
            raise ValueError(f"Duplicate image stem would overwrite an output mask: {path.stem}")
        names.add(key)
        with Image.open(path) as image:
            image.verify()
    return images


def load_model(weights, device):
    from dinov2.hub.backbones import dinov2_vitl14

    model = dinov2_vitl14(pretrained=weights is None)
    if weights is not None:
        load_options = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_options["weights_only"] = True
        state = torch.load(weights, **load_options)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args, model=None):
    images = discover_images(args.data_path)
    stems = {path.stem.casefold() for path in images}
    existing = (list(args.save_path.glob("*.png")) + list(args.save_path.glob("*.npy"))) if args.save_path.is_dir() else []
    unrelated = [path.name for path in existing if path.stem.casefold() not in stems]
    if unrelated:
        raise ValueError(f"Output directory contains masks for other inputs: {unrelated[:5]}. Use a fresh --save_path; unrelated files are never deleted.")
    if args.overwrite and not args.save_npy and any(path.suffix.lower() == ".npy" for path in existing):
        raise ValueError("Existing .npy outputs would become stale. Use --save-npy with --overwrite, or use a fresh --save_path.")
    manifest_path = args.save_path / "despl_manifest.json"
    if not args.overwrite and (existing or manifest_path.exists()):
        raise ValueError("Output masks or a manifest already exist. Use a fresh --save_path or --overwrite; existing results are never silently reused.")
    if args.weights is not None and not args.weights.is_file():
        raise ValueError(f"Local weights do not exist: {args.weights}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; pass --device cpu explicitly")
    if args.data_path.resolve() == args.save_path.resolve():
        raise ValueError("Input and output directories must differ")
    args.save_path.mkdir(parents=True, exist_ok=True)
    if model is None:
        model = load_model(args.weights, device)
    config = DESPLConfig(image_size=args.imgsz)
    records = []
    for path in tqdm(images, desc="DESPL"):
        with Image.open(path) as image:
            image = image.convert("RGB")
            soft_mask = generate_pseudo_label_with_augmentation(model, image, device, config)
        mask = (soft_mask > 0.5).astype(np.float32) if args.binary_mask else soft_mask
        output = args.save_path / (path.stem + ".png")
        Image.fromarray(np.rint(mask * 255).astype(np.uint8)).save(output)
        if args.save_npy:
            np.save(output.with_suffix(".npy"), soft_mask.astype(np.float32))
        records.append({"image": str(path.resolve()), "mask": output.name,
                        "width": image.width, "height": image.height})
    manifest = {
        "method": "DualUCOD DESPL, Eqs. (1)-(10)",
        "backbone": "DINOv2 ViT-L/14",
        "config": config.__dict__,
        "augmentations": ["normal", "r90", "r180", "r270", "hflip", "vflip", "bright_up_1.2", "contrast_up_1.2"],
        "eigenvectors": "first two in ascending eigenvalue order; lowest mode is not discarded; signed affinity can change degree/null mode ordering",
        "sign_convention": "joint positive-entry fraction flip from supplied new_label2.py; unspecified in paper",
        "output": "binary >0.5 after fusion" if args.binary_mask else "soft Eq. (10) average, bilinear resized, rounded to 8-bit PNG",
        "weights": str(args.weights.resolve()) if args.weights else "official DINOv2 ViT-L/14 pretrained weights",
        "weights_sha256": sha256_file(args.weights) if args.weights else None,
        "float32_npy": args.save_npy,
        "npy_output": "unquantized soft Eq. (10) output, also when --binary-mask is enabled" if args.save_npy else None,
        "images": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} masks to {args.save_path}")
    return records


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
