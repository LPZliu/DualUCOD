import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse
# Assume hubconf.py is in the same directory
from hubconf import dinov2_vitl14, dinov2_vits14, dinov2_vitb14
import scipy
from scipy import ndimage
import cv2
from scipy.ndimage import label
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab

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

def get_diagonal(W, threshold: float=1e-12):
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W, dtype=torch.float32)

    D = torch.matmul(W, torch.ones(W.shape[1], dtype=W.dtype).to(W.device))
    D[D < threshold] = 1.0  # Prevent division by zero.

    D_diag = torch.diag(D)
    return D_diag

def laplacian(adj, W):
    adj = (adj * (adj > 0))
    max_values = adj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    adj = adj / max_values 

    W = W.unsqueeze(0)
    W = (W * (W > 0))
    w_combs = W.to(adj.device)
    max_values = w_combs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    w_combs = w_combs / max_values

    W_comb = 0.1 * w_combs + adj
    D_comb = torch.stack([get_diagonal(w_comb) for w_comb in W_comb])
    L_comb = D_comb - W_comb
    lap = normalized_laplacian(L_comb, D_comb)
    return lap

def eigen(lap, K):
    eigenvalues_all, eigenvectors_all = torch.linalg.eigh(lap, UPLO='U')
    eigenvalues = eigenvalues_all[:, :K]
    eigenvectors = eigenvectors_all[:, :, :K]    
    eigenvalues = eigenvalues.float()
    eigenvectors = eigenvectors.float()

    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  
            eigenvectors[k] = 0 - eigenvectors[k]
    return eigenvalues, eigenvectors

def laplacian_matrix(img_raw, local_query):

    adj = cosine_similarity(local_query) 

    img_raw = np.array(img_raw)

    img_raw = F.interpolate(
        torch.from_numpy(img_raw).permute(2, 0, 1).float().unsqueeze(0),
        size=(34, 34),
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

def count_edge_point(mask):
    zero_nums = 0
    one_nums = 0

    zero_nums += (mask[0, :] == 0).sum()
    one_nums += (mask[0, :] == 1).sum()

    zero_nums += (mask[-1, :] == 0).sum()
    one_nums += (mask[-1, :] == 1).sum()

    zero_nums += (mask[1:-1, 0] == 0).sum()
    one_nums += (mask[1:-1, 0] == 1).sum()

    zero_nums += (mask[1:-1, -1] == 0).sum()
    one_nums += (mask[1:-1, -1] == 1).sum()

    assert zero_nums + one_nums == (2 * (mask.shape[0] + mask.shape[1]) - 4)

    return zero_nums, one_nums


import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.stats import entropy

def _compute_entropy(vec, n_bins=50):
    """计算向量的归一化信息熵（基于直方图）"""
    # 归一化到 [0, 1] 便于统一 binning
    eps = 1e-8
    vec_norm = (vec - vec.min()) / (vec.max() - vec.min() + eps)
    hist, _ = np.histogram(vec_norm, bins=n_bins, range=(0, 1), density=False)
    probs = hist / (hist.sum() + eps)
    return entropy(probs)

def dual_eigen_fusion_adaptive(embedding, img_size=34):
    H = W = img_size
    N = H * W

    eigen0 = embedding[:, 0]
    eigen1 = embedding[:, 1]

    # === Step 1: 自适应选择主特征向量（基于信息熵）===
    entropy0 = _compute_entropy(eigen0)
    entropy1 = _compute_entropy(eigen1)

    if entropy1 < entropy0:
        eigen_main = eigen1
        eigen_aux = eigen0
        main_idx = 1
    else:
        eigen_main = eigen0
        eigen_aux = eigen1
        main_idx = 0

    # === Step 2: 对主向量做稳健阈值分割 ===
    if len(np.unique(eigen_main)) > 2:
        try:
            thresh_main = threshold_otsu(eigen_main)
        except Exception:
            thresh_main = np.median(eigen_main)
    else:
        thresh_main = np.median(eigen_main)

    mask_primary = (eigen_main > thresh_main).astype(float)  # (N,)

    # === Step 3: 定义模糊区域（基于主向量）===
    eps = 1e-8
    margin = 0.1 * (eigen_main.max() - eigen_main.min() + eps)
    low_confidence = (eigen_main >= thresh_main - margin) & (eigen_main <= thresh_main + margin)

    # === Step 4: 用辅助向量修正模糊区域 ===
    aux_norm = (eigen_aux - eigen_aux.min()) / (eigen_aux.max() - eigen_aux.min() + eps)
    final_mask = mask_primary.copy()
    if np.any(low_confidence):
        final_mask[low_confidence] = (aux_norm[low_confidence] > 0.5).astype(float)

    # === Step 5: 强后处理 —— 保留最大连通域 ===
    mask_2d = final_mask.reshape(H, W).astype(bool)
    labeled, num_feat = label(mask_2d)

    if num_feat == 0:
        final_mask_2d = np.zeros((H, W), dtype=bool)
    else:
        props = regionprops(labeled)
        largest_region = max(props, key=lambda x: x.area)
        final_mask_2d = (labeled == largest_region.label)

    return final_mask_2d.ravel().astype(np.int64)

def generate_pseudo_label(model, img_orig, transform, device):
    """
    Generate pseudo-label using dual-eigen fusion.
    
    Args:
        model: DINO model (with get_intermediate_layers)
        img_orig: PIL.Image.Image
        transform: torchvision transform
        device: torch.device
        debug_save_path: 
            - None: 不保存
            - str: 如 './vis' 或 './vis/001.png'
    
    Returns:
        binary_mask_tensor: (H_orig, W_orig) float32 {0,1}
        orig_size: (h, w)
    """
    from torchvision.transforms.functional import to_tensor  # 确保有 transform

    w_orig, h_orig = img_orig.size
    img_tensor = transform(img_orig).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model.get_intermediate_layers(img_tensor, n=1, reshape=True)[0]  # [1, C, 34, 34]

    # Reshape DINO features to (N, C)
    local_query = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1]).cpu().numpy()  # (1156, C)

    # Get Laplacian embedding (you must have this function!)
    embedding = laplacian_matrix(img_orig, local_query)  # Should return (1156, K) tensor
    embedding = embedding.cpu().numpy()

    # Use dual-eigen fusion
    labels_flat = dual_eigen_fusion_adaptive(embedding, img_size=34)
    labels = labels_flat.reshape(34, 34)

    # Flip based on edge prior (ensure background on borders)
    zero_nums_at_edge, one_nums_at_edge = count_edge_point(labels)
    if one_nums_at_edge > zero_nums_at_edge:
        labels = 1 - labels

    # Upsample to original resolution
    labels_tensor = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    labels_up = F.interpolate(
        labels_tensor,
        size=(h_orig, w_orig),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    binary_mask = (labels_up > 0.5).astype(np.float32)
    return torch.from_numpy(binary_mask), (h_orig, w_orig)

def generate_pseudo_label_with_augmentation(model, img_path, transform, device, use_color=True):
    img_orig = Image.open(img_path).convert("RGB")
    H_orig, W_orig = img_orig.size
    
    augs = get_augmentations(use_color=use_color)
    masks = []
    fusion_mask = 0
    for aug_name, aug_func in augs.items():
        # Apply augmentation to image (PIL)
        aug_img = aug_func(img_orig)
        
        # Generate pseudo-label (uses PIL image)
        mask, _ = generate_pseudo_label(model, aug_img, transform, device)
        
        # Convert to numpy
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Inverse transform mask
        aug_mask = get_inverse_aug(aug_name)(mask)
        fusion_mask += aug_mask
    fusion_mask /= len(augs.keys())
    # Fuse masks (average)
    # fused_mask = np.mean(masks, axis=0)
    binary_mask = (fusion_mask > 0.5).astype(np.float32)
    return torch.from_numpy(binary_mask), (H_orig, W_orig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino", type=str, default="vit-l14")
    parser.add_argument("--imgsz", type=int, default=476)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", type=str, default="./TrainDataset/Imgs")
    parser.add_argument("--save_path", type=str, default="./PseudoMask_s1")
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    dino_dim = {'vit-s14': 384, 'vit-b14': 768, 'vit-l14': 1024}
    dinotype = args.dino
    imgsz = args.imgsz
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    if dinotype == 'vit-s14':
        model = dinov2_vits14().to(device).eval()
    elif dinotype == 'vit-b14':
        model = dinov2_vitb14().to(device).eval()
    elif dinotype == 'vit-l14':
        model = dinov2_vitl14().to(device).eval()

    os.makedirs(args.save_path, exist_ok=True)
    img_files = [f for f in os.listdir(args.data_path) if f.lower().endswith(('.jpg', '.png'))]

    for img_name in tqdm(img_files):
        img_path = os.path.join(args.data_path, img_name)
        save_path = os.path.join(args.save_path, os.path.splitext(img_name)[0] + '.png')
        if os.path.exists(save_path):
            continue
        pseudo_soft, (h, w) = generate_pseudo_label_with_augmentation(
            model, img_path, transform, device
        )
        # pseudo_soft = 1 - pseudo_soft
        final_mask = pseudo_soft.cpu().numpy().astype(np.uint8)
        Image.fromarray(final_mask * 255, mode='L').save(save_path)

if __name__ == "__main__":
    main()