"""DESPL as specified in DualUCOD, Sec. 3.1 and Appendix A.3.

This module implements Eqs. (1)-(10), selecting the FIRST two Laplacian
eigenvectors in ascending eigenvalue order without discarding the lowest
mode. For nonnegative affinity this includes the degree/null eigenvector;
signed cosine affinities can change that ordering. It does not
add connected-component filtering, a border prior, affinity clipping, or a
hard threshold after the multi-view average. Those are additional operations
in the supplied development script, not operations in Eqs. (1)-(10).

The paper leaves the arbitrary signs of the eigenvectors unspecified. We
retain the supplied new_label2.py convention: flip BOTH columns together
when strictly between half and all entries of the pair are positive. This
convention is an implementation choice, not an additional paper equation;
it cannot guarantee bitwise agreement across eigensolver implementations.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance
from scipy.linalg import eigh
from skimage.filters import threshold_otsu


@dataclass(frozen=True)
class DESPLConfig:
    """Paper settings; the 34 x 34 grid comes from Appendix A.3."""

    image_size: int = 476
    grid_size: int = 34
    color_weight: float = 0.1
    color_sigma: float = 0.1
    entropy_bins: int = 50
    ambiguous_fraction: float = 0.1


def affinity_matrix(features, rgb, color_weight=0.1, color_sigma=0.1):
    """Eqs. (1)-(3): raw cosine affinity plus Gaussian RGB affinity.

    features is N x D; rgb contains N RGB values in the original 0..255
    scale. In particular, negative cosine affinities are NOT truncated.
    """
    features = np.asarray(features, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if features.ndim != 2 or len(features) != len(rgb) or len(features) < 2:
        raise ValueError("Expected N x D features and N RGB pixels, with N >= 2")
    if not np.isfinite(features).all() or not np.isfinite(rgb).all():
        raise ValueError("Features and RGB values must be finite")
    if color_sigma <= 0 or color_weight < 0:
        raise ValueError("color_sigma must be positive and color_weight nonnegative")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cosine affinity is undefined for a zero feature vector")
    unit = features / norms
    semantic = unit @ unit.T
    pixels = rgb / 255.0
    squared_distances = np.sum((pixels[:, None] - pixels[None, :]) ** 2, axis=-1)
    color = np.exp(-squared_distances / (2.0 * color_sigma**2))
    return semantic + color_weight * color


def normalized_laplacian(affinity):
    """L_sym = D^-1/2 (D-A) D^-1/2, directly following Sec. 3.1.

    Fail on a nonpositive degree rather than silently modifying the graph.
    """
    affinity = np.asarray(affinity, dtype=np.float64)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("Affinity must be square")
    if not np.isfinite(affinity).all() or not np.allclose(affinity, affinity.T):
        raise ValueError("Affinity must be finite and symmetric")
    degree = affinity.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("Nonpositive graph degree; Eq. (3) was not clipped or rescaled")
    inv_sqrt = degree**-0.5
    laplacian = (np.diag(degree) - affinity) * inv_sqrt[:, None] * inv_sqrt[None, :]
    return (laplacian + laplacian.T) / 2.0


def smallest_eigenvectors(laplacian):
    """Return the first two eigenpairs in ascending eigenvalue order.

    The lowest mode is not discarded. A nonnegative affinity includes the
    degree/null eigenvector here; signed cosine affinity can change ordering.
    Signs follow the joint two-column convention documented above. Repeated
    eigenvalues can still produce different valid bases on different systems.
    """
    values, vectors = eigh(laplacian, subset_by_index=(0, 1), check_finite=True)
    positive_fraction = np.mean(vectors > 0)
    if 0.5 < positive_fraction < 1.0:
        vectors = -vectors
    return values, vectors


def minmax_normalize(vector):
    """Eq. (4), with a numerical-constant guard for degenerate embeddings.

    Spans within 32 machine epsilons of the vector magnitude map to zero,
    so eigensolver roundoff does not amplify a theoretical constant mode.
    This degeneracy convention is not specified in the paper.
    """
    vector = np.asarray(vector, dtype=np.float64)
    span = np.ptp(vector)
    tolerance = 32 * np.finfo(vector.dtype).eps * np.max(np.abs(vector))
    if span <= tolerance:
        return np.zeros_like(vector)
    return (vector - vector.min()) / span


def histogram_entropy(normalized_vector, bins=50):
    """Eq. (5), using B=50 equal bins on [0, 1]."""
    histogram, _ = np.histogram(normalized_vector, bins=bins, range=(0, 1))
    probabilities = histogram[histogram > 0].astype(np.float64) / histogram.sum()
    return -np.sum(probabilities * np.log(probabilities))


def dual_eigen_fusion(embedding, entropy_bins=50, ambiguous_fraction=0.1, return_details=False):
    """Eqs. (4)-(9): entropy selection, Otsu, and auxiliary refinement.

    Returns an N-element binary mask. There is no connected-component or
    border post-processing. Constants map to zero under the Eq. (4) guard.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2 or embedding.shape[1] != 2 or not np.isfinite(embedding).all():
        raise ValueError("Expected a finite N x 2 eigenvector embedding")
    if entropy_bins < 2 or ambiguous_fraction < 0:
        raise ValueError("Invalid entropy or ambiguity settings")
    normalized = np.column_stack([minmax_normalize(embedding[:, k]) for k in range(2)])
    entropies = [histogram_entropy(normalized[:, k], entropy_bins) for k in range(2)]
    main_index = 1 if entropies[1] < entropies[0] else 0
    main, auxiliary = normalized[:, main_index], normalized[:, 1 - main_index]
    threshold = float(threshold_otsu(main))
    mask = main > threshold
    margin = ambiguous_fraction * np.ptp(main)
    ambiguous = np.abs(main - threshold) <= margin
    mask[ambiguous] = auxiliary[ambiguous] > 0.5
    if return_details:
        return mask, {"main_index": main_index, "entropies": entropies,
                      "threshold": threshold, "ambiguous": ambiguous,
                      "normalized": normalized}
    return mask


def get_augmentations():
    """The eight views in supplied source, with T=8 from Appendix A.3."""
    transpose = getattr(Image, "Transpose", Image)
    return (
        ("normal", lambda image: image),
        ("r90", lambda image: image.transpose(transpose.ROTATE_90)),
        ("r180", lambda image: image.transpose(transpose.ROTATE_180)),
        ("r270", lambda image: image.transpose(transpose.ROTATE_270)),
        ("hflip", lambda image: image.transpose(transpose.FLIP_LEFT_RIGHT)),
        ("vflip", lambda image: image.transpose(transpose.FLIP_TOP_BOTTOM)),
        ("bright_up", lambda image: ImageEnhance.Brightness(image).enhance(1.2)),
        ("contrast_up", lambda image: ImageEnhance.Contrast(image).enhance(1.2)),
    )


def inverse_augmentation(mask, name):
    if name == "r90":
        return np.rot90(mask, k=3)
    if name == "r180":
        return np.rot90(mask, k=2)
    if name == "r270":
        return np.rot90(mask, k=1)
    if name == "hflip":
        return np.fliplr(mask)
    if name == "vflip":
        return np.flipud(mask)
    if name in ("normal", "bright_up", "contrast_up"):
        return mask
    raise ValueError(f"Unknown augmentation: {name}")


def image_transform(image_size=476):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def generate_grid_mask(model, image, transform, device, config=DESPLConfig()):
    """Extract the last DINOv2 feature map, then apply Eqs. (1)-(9)."""
    import torch
    import torch.nn.functional as functional

    with torch.no_grad():
        features = model.get_intermediate_layers(transform(image).unsqueeze(0).to(device), n=1, reshape=True)[0]
    if tuple(features.shape[-2:]) != (config.grid_size, config.grid_size):
        raise ValueError(f"Expected a {config.grid_size} x {config.grid_size} feature grid; got {features.shape}")
    features = features[0].flatten(1).T.cpu().numpy()
    # Preserve the supplied source's interpolation convention for RGB sampling.
    rgb = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()[None]
    rgb = functional.interpolate(rgb, size=(config.grid_size, config.grid_size), mode="bilinear", align_corners=True)
    rgb = rgb[0].permute(1, 2, 0).numpy()
    affinity = affinity_matrix(features, rgb, config.color_weight, config.color_sigma)
    _, embedding = smallest_eigenvectors(normalized_laplacian(affinity))
    mask = dual_eigen_fusion(embedding, config.entropy_bins, config.ambiguous_fraction)
    return mask.reshape(config.grid_size, config.grid_size)


def generate_pseudo_label_with_augmentation(model, image, device, config=DESPLConfig()):
    """Eq. (10): inverse-align eight binary grid masks and average them.

    The average is bilinearly resized once to the original image dimensions.
    The floating output remains in [0, 1]; it is NOT thresholded after fusion.
    All rotations/flips are aligned on the common grid before resizing, so
    nonsquare original images retain their correct orientation and size.
    """
    import torch
    import torch.nn.functional as functional

    image = image.convert("RGB")
    transform = image_transform(config.image_size)
    masks = [inverse_augmentation(generate_grid_mask(model, augment(image), transform, device, config), name)
             for name, augment in get_augmentations()]
    mean_grid = np.mean(np.stack(masks), axis=0, dtype=np.float64)
    tensor = torch.from_numpy(mean_grid.astype(np.float32))[None, None]
    result = functional.interpolate(tensor, size=(image.height, image.width), mode="bilinear", align_corners=False)
    return result[0, 0].numpy().clip(0, 1)
