"""ICML DualUCOD: Dual-Eigenvector Spectral Pseudo-Labeling (DESPL).

Adapted from the supplied local get_new_label_me.py, retaining its DINOv2,
affinity, eigendecomposition, and augmentation pipeline. Implements Sec. 3.1,
Eqs. (1)-(10), with Appendix A.3 defaults. The aligned multi-view mean is
saved as a soft grayscale PNG; --binary-mask enables an extra hard threshold.

Paper: https://proceedings.mlr.press/v306/liu26au.html
Eigenvector signs and the exact eight augmentation choices follow the local
source; these implementation conventions are not uniquely specified by the paper.
"""

import argparse
import inspect
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from skimage.filters import threshold_otsu
from scipy.stats import entropy

from hubconf import dinov2_vitl14, dinov2_vits14, dinov2_vitb14

def get_augmentations(use_color=True):
    augs = {
        "normal": lambda x: x,
        "r90": lambda x: TF.rotate(x, 90, expand=True),
        "r180": lambda x: TF.rotate(x, 180, expand=True),
        "r270": lambda x: TF.rotate(x, 270, expand=True),
        "hflip": lambda x: TF.hflip(x),
        "vflip": lambda x: TF.vflip(x),
    }
    if use_color:
        augs.update({
            "bright_up": lambda x: TF.adjust_brightness(x, 1.2),
            "contrast_up": lambda x: TF.adjust_contrast(x, 1.2),
        })
    return augs

def get_inverse_aug(aug_name):
    inv_map = {
        "normal": lambda x: x,
        "r90": lambda x: np.rot90(x, k=3),
        "r180": lambda x: np.rot90(x, k=2),
        "r270": lambda x: np.rot90(x, k=1),
        "hflip": lambda x: np.fliplr(x),
        "vflip": lambda x: np.flipud(x),
        "bright_up": lambda x: x,
        "contrast_up": lambda x: x,
    }
    return inv_map[aug_name]

def ensure_symmetric(M):
    return (M + M.t()) / 2

def normalized_laplacian_from_W(W, eps=1e-6):
    """
    Stable computation of symmetric normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    """
    N = W.shape[0]
    device = W.device

    # Degree vector
    deg = W.sum(dim=1)

    # Clamp to avoid extreme values and division by zero
    deg = torch.clamp(deg, min=eps, max=1e6)

    # Compute D^{-1/2}
    deg_inv_sqrt = deg ** (-0.5)
    D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # (N, N)

    # Compute normalized affinity: W_norm = D^{-1/2} W D^{-1/2}
    W_norm = D_inv_sqrt @ W @ D_inv_sqrt
    W_norm = ensure_symmetric(W_norm)  # Numerical symmetry

    # Laplacian: L = I - W_norm
    L_sym = torch.eye(N, device=device) - W_norm
    L_sym = ensure_symmetric(L_sym)

    return L_sym

def pairwise_distances(x, y=None, w=None):
    if w is None:
        w = torch.ones(x.size(1))

    x_w = x * w
    x_norm = (x_w**2).sum(1).view(-1, 1)

    if y is not None:
        y = y * w
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x_w, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x_w, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def color_affinity(img):

    normalized_image = img / 255.0
    pixels = normalized_image.view(-1, 3)
    color_distances = torch.cdist(pixels, pixels, p=2.0)
    color_affinity = torch.exp(-color_distances ** 2 / (2 * (0.1 ** 2)))

    return color_affinity

def normalized_laplacian(L, D):
    D_inv_sqrt = torch.inverse(torch.sqrt(D))
    D_inv_sqrt = D_inv_sqrt.diagonal(dim1=-2, dim2=-1)

    D_inv_sqrt_diag = torch.diag_embed(D_inv_sqrt)

    L_norm = torch.bmm(D_inv_sqrt_diag, torch.bmm(L, D_inv_sqrt_diag))

    return L_norm

def get_diagonal(W):
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W, dtype=torch.float64)
    degree = W.sum(dim=1)
    if not torch.isfinite(degree).all() or torch.any(degree <= 0):
        raise ValueError("Nonpositive or nonfinite graph degree in the ICML affinity")
    return torch.diag(degree)

def laplacian(adj, W):
    """Eqs. (1)-(3): raw cosine affinity + 0.1 * Gaussian RGB affinity."""
    # Preserve the paper's affinity, including negative cosine entries.
    W_comb = adj + 0.1 * W.unsqueeze(0).to(adj)
    D_comb = torch.stack([get_diagonal(w_comb) for w_comb in W_comb])
    L_comb = D_comb - W_comb
    return normalized_laplacian(L_comb, D_comb)

def eigen(lap, K):
    eigenvalues_all, eigenvectors_all = torch.linalg.eigh(lap, UPLO='U')
    eigenvalues = eigenvalues_all[:, :K]
    eigenvectors = eigenvectors_all[:, :, :K]

    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:
            eigenvectors[k] = 0 - eigenvectors[k]
    return eigenvalues, eigenvectors

def laplacian_matrix(img_raw, local_query, img_size=34):

    adj = cosine_similarity(local_query.astype(np.float64))

    img_raw = np.array(img_raw)

    img_raw = F.interpolate(
        torch.from_numpy(img_raw).permute(2, 0, 1).float().unsqueeze(0),
        size=(img_size, img_size) if isinstance(img_size, int) else img_size,
        mode='bilinear',
        align_corners=True
    ).squeeze(0).permute(1, 2, 0).numpy()
    img_raw = torch.from_numpy(img_raw).unsqueeze(0)
    color_W = color_affinity(img_raw)
    adj = torch.from_numpy(adj).unsqueeze(0)
    nor_adj_lap = laplacian(adj, color_W)

    nor_adj_lap_detach = torch.clone(nor_adj_lap.detach())
    eigenvalues, eigenvectors = eigen(nor_adj_lap_detach, K=2)

    return eigenvectors.squeeze(0)



def _minmax_normalize(vec):
    """Eq. (4), with a roundoff guard for a numerically constant vector."""
    vec = np.asarray(vec, dtype=np.float64)
    span = np.ptp(vec)
    tolerance = 32 * np.finfo(vec.dtype).eps * np.max(np.abs(vec))
    if span <= tolerance:
        return np.zeros_like(vec)
    return (vec - vec.min()) / span


def _compute_entropy(normalized_vec, n_bins=50):
    """Eq. (5): Shannon entropy of B=50 bins over the normalized vector."""
    hist, _ = np.histogram(normalized_vec, bins=n_bins, range=(0, 1))
    probs = hist[hist > 0].astype(np.float64) / hist.sum()
    return entropy(probs)


def dual_eigen_fusion_adaptive(embedding, img_size=34, alpha=0.1):
    """ICML Eqs. (4)-(9): entropy selection, Otsu, and AUX correction."""
    H, W = (img_size, img_size) if isinstance(img_size, int) else img_size
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.shape != (H * W, 2) or not np.isfinite(embedding).all():
        raise ValueError("Expected a finite H*W by 2 eigenvector embedding")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")

    eigen0 = _minmax_normalize(embedding[:, 0])
    eigen1 = _minmax_normalize(embedding[:, 1])

    # Step 1: use the lower-entropy normalized eigenvector as MAIN (Eq. 6).
    if _compute_entropy(eigen1) < _compute_entropy(eigen0):
        eigen_main, eigen_aux = eigen1, eigen0
    else:
        eigen_main, eigen_aux = eigen0, eigen1

    # Step 2: Otsu produces the initial binary partition (Eq. 7).
    thresh_main = threshold_otsu(eigen_main)
    final_mask = eigen_main > thresh_main

    # Step 3: AUX refines only the low-confidence band (Eqs. 8-9).
    margin = alpha * np.ptp(eigen_main)
    ambiguous = np.abs(eigen_main - thresh_main) <= margin
    final_mask[ambiguous] = eigen_aux[ambiguous] > 0.5
    return final_mask.astype(np.int64)


def generate_pseudo_label(model, img_orig, transform, device, alpha=0.1, return_grid=False):
    """Generate a per-view ICML mask and return its original (height, width).

    return_grid=True preserves the binary patch mask for Eq. (10) fusion.
    Otherwise, return its bilinearly resized original-resolution soft mask.
    """
    w_orig, h_orig = img_orig.size
    img_tensor = transform(img_orig).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model.get_intermediate_layers(img_tensor, n=1, reshape=True)[0]

    grid_size = tuple(feats.shape[-2:])
    local_query = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1]).cpu().numpy()
    embedding = laplacian_matrix(img_orig, local_query, img_size=grid_size).cpu().numpy()
    labels = dual_eigen_fusion_adaptive(embedding, img_size=grid_size, alpha=alpha).reshape(grid_size)
    labels_tensor = torch.from_numpy(labels.astype(np.float32))
    if return_grid:
        return labels_tensor, (h_orig, w_orig)

    labels_up = F.interpolate(
        labels_tensor[None, None], size=(h_orig, w_orig),
        mode='bilinear', align_corners=False
    )[0, 0]
    return labels_up, (h_orig, w_orig)

def generate_pseudo_label_with_augmentation(model, img_path, transform, device, use_color=True, alpha=0.1):
    with Image.open(img_path) as image:
        img_orig = image.convert("RGB")
    w_orig, h_orig = img_orig.size

    augs = get_augmentations(use_color=use_color)
    aligned_masks = []
    for aug_name, aug_func in augs.items():
        aug_img = aug_func(img_orig)
        mask, _ = generate_pseudo_label(
            model, aug_img, transform, device, alpha=alpha, return_grid=True
        )
        aug_mask = get_inverse_aug(aug_name)(mask.cpu().numpy())
        aligned_masks.append(aug_mask.astype(np.float32))

    # Eq. (10): inverse-align on the common patch grid, then average.
    mean_grid = np.mean(np.stack(aligned_masks, axis=0), axis=0, dtype=np.float64)
    labels_tensor = torch.from_numpy(mean_grid.astype(np.float32))[None, None]
    soft_mask = F.interpolate(
        labels_tensor, size=(h_orig, w_orig), mode='bilinear', align_corners=False
    )[0, 0]
    return soft_mask, (h_orig, w_orig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino", type=str, default="vit-l14", choices=("vit-s14", "vit-b14", "vit-l14"))
    parser.add_argument("--imgsz", type=int, default=476, choices=(476,))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", type=str, default="./data/TrainDataset/Imgs")
    parser.add_argument("--save_path", type=str, default="./data/TrainDataset/PseudoMask_Enhanced")
    parser.add_argument("--alpha", type=float, default=0.1, help="Eq. (8) uncertainty-band fraction; ICML default: 0.1")
    parser.add_argument("--weights", type=str, help="local DINOv2 state dict; otherwise use official pretrained weights")
    parser.add_argument("--binary-mask", action="store_true", help="apply an extra >0.5 threshold after Eq. (10)")
    parser.add_argument("--save-npy", action="store_true", help="also save the unquantized float32 soft mask")
    parser.add_argument("--overwrite", action="store_true", help="regenerate existing masks")
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or args.alpha < 0:
        parser.error("--alpha must be finite and nonnegative")
    if not os.path.isdir(args.data_path):
        parser.error("--data_path must be an existing image directory")
    if os.path.normcase(os.path.realpath(args.data_path)) == os.path.normcase(os.path.realpath(args.save_path)):
        parser.error("input and output directories must differ")
    img_files = sorted(f for f in os.listdir(args.data_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                       and os.path.isfile(os.path.join(args.data_path, f)))
    if not img_files:
        parser.error("no JPG/JPEG/PNG images found in --data_path")
    stems = [os.path.splitext(f)[0].casefold() for f in img_files]
    if len(stems) != len(set(stems)):
        parser.error("duplicate image stems would overwrite an output mask")
    if args.weights is not None and not os.path.isfile(args.weights):
        parser.error("--weights must be an existing state-dict file")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.Resize((args.imgsz, args.imgsz)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    builders = {'vit-s14': dinov2_vits14, 'vit-b14': dinov2_vitb14, 'vit-l14': dinov2_vitl14}
    model = builders[args.dino](pretrained=args.weights is None)
    if args.weights is not None:
        load_options = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_options["weights_only"] = True
        state = torch.load(args.weights, **load_options)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    os.makedirs(args.save_path, exist_ok=True)
    for img_name in tqdm(img_files):
        img_path = os.path.join(args.data_path, img_name)
        save_path = os.path.join(args.save_path, os.path.splitext(img_name)[0] + '.png')
        npy_path = os.path.splitext(save_path)[0] + '.npy'
        if os.path.exists(save_path) and not args.overwrite:
            continue
        if not args.save_npy and os.path.exists(npy_path):
            raise ValueError("Existing .npy output would become stale; use --save-npy or a fresh --save_path")
        pseudo_soft, _ = generate_pseudo_label_with_augmentation(
            model, img_path, transform, device, alpha=args.alpha
        )
        soft_mask = pseudo_soft.cpu().numpy().astype(np.float32)
        if args.save_npy:
            np.save(npy_path, soft_mask)
        final_mask = (soft_mask > 0.5).astype(np.float32) if args.binary_mask else soft_mask
        Image.fromarray(np.rint(final_mask * 255).astype(np.uint8)).save(save_path)

if __name__ == "__main__":
    main()
