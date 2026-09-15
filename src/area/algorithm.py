"""
Pure-function implementations of the three AREA decision axes.

All functions operate on numpy arrays and are model-agnostic.
They are called by AREAInferenceModel after the probe pass extracts raw attention arrays.
"""
import math
import numpy as np
from typing import List, Tuple, Optional


# ─── Stage C: EAEG ────────────────────────────────────────────────────────────

def compute_sentence_scores(
    a_txt: np.ndarray,
    sent_spans: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Aggregate per-token attention scores into per-sentence scores.
    Implements pseudocode lines 15-16.

    a_txt: shape [N_C] — attention from probe token to context tokens (already averaged
           over layers and heads by the caller, then row-normalised per layer).
    sent_spans: list of (start, end) token offsets relative to context start.

    Returns g: shape [M] — raw sentence scores (non-negative, sum > 0 unless trivial).
    """
    scores = np.zeros(len(sent_spans), dtype=np.float32)
    for m, (s, e) in enumerate(sent_spans):
        if e > s:
            scores[m] = float(a_txt[s:e].mean())
    return scores


def compute_eaeg_text(
    sent_scores: np.ndarray,
    k_max: int = 3,
    eps: float = 1e-8,
) -> Tuple[int, float]:
    """
    EAEG text-granularity decision (pseudocode lines 16-19).

    Returns:
        k_txt  — adaptive number of sentences to highlight
        H_txt  — Shannon entropy in bits of the sentence-score distribution
    """
    g = sent_scores.astype(np.float64)
    total = g.sum()
    if total <= eps:
        # No attention on context at all → default to 1 sentence (conservative; avoids
        # treating a zero-signal distribution as uniform and falsely selecting k_max sents)
        return 1, 0.0
    p = g / total

    # Shannon entropy H = -Σ p_m * log2(p_m)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    H_txt = -float((p * log_p).sum())

    # Perplexity PP = 2^H; adaptive k
    PP_txt = 2.0 ** H_txt
    k_txt = min(math.ceil(PP_txt), k_max)
    return k_txt, H_txt


def compute_visual_centroid(
    a_vis: np.ndarray,
    grid_shape: Tuple[int, int],
    eps: float = 1e-8,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Compute weighted centroid and spread from filtered visual attention (pseudocode lines 10-13).

    a_vis: shape [N_V] — filtered visual token attention (sinks zeroed).
    grid_shape: (H_grid, W_grid) — spatial layout of visual tokens.

    Returns:
        c_x, c_y   — centroid (in grid coordinates)
        sigma_x, sigma_y — weighted standard deviations
        M_vis_tilde — normalised spatial probability map, shape [H_grid, W_grid]
    """
    H_grid, W_grid = grid_shape
    M_vis = a_vis.reshape(grid_shape).astype(np.float64)
    total = M_vis.sum()
    if total <= eps:
        # Degenerate: uniform
        M_vis_tilde = np.full_like(M_vis, 1.0 / (H_grid * W_grid))
    else:
        M_vis_tilde = M_vis / total

    hs = np.arange(H_grid, dtype=np.float64)
    ws = np.arange(W_grid, dtype=np.float64)
    W_mesh, H_mesh = np.meshgrid(ws, hs)  # both shape [H, W]

    c_x = float((W_mesh * M_vis_tilde).sum())
    c_y = float((H_mesh * M_vis_tilde).sum())
    sigma_x = float(np.sqrt(((W_mesh - c_x) ** 2 * M_vis_tilde).sum()))
    sigma_y = float(np.sqrt(((H_mesh - c_y) ** 2 * M_vis_tilde).sum()))

    # Guard against degenerate (point) distributions
    sigma_x = max(sigma_x, 0.5)
    sigma_y = max(sigma_y, 0.5)

    return c_x, c_y, sigma_x, sigma_y, M_vis_tilde


def compute_eaeg_vision(
    M_vis_tilde: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    beta_min: float = 1.5,
    beta_max: float = 2.5,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """
    EAEG vision-granularity decision (pseudocode lines 20-22).

    Returns:
        beta   — adaptive scale factor for the bounding box
        H_vis  — Shannon entropy in bits of the spatial attention map
    """
    p = M_vis_tilde.ravel().astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    H_vis = -float((p * log_p).sum())

    A_eff = 2.0 ** H_vis  # effective #grid cells
    beta_raw = 0.5 * math.sqrt(A_eff / (sigma_x * sigma_y + eps))
    beta = float(np.clip(beta_raw, beta_min, beta_max))
    return beta, H_vis


# ─── Stage D: DG-LT ────────────────────────────────────────────────────────────

def compute_dg_lt_signals(
    a_txt: np.ndarray,
    top_k_sent_spans: List[Tuple[int, int]],
    a_vis: np.ndarray,
    grid_shape: Optional[Tuple[int, int]],
    grid_bbox: Optional[Tuple[int, int, int, int]],
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """
    Compute DG-LT gate signals Δ_txt and Δ_vis (pseudocode lines 23-27).

    a_txt: [N_C] — normalised attention from probe token to context (per-token).
    top_k_sent_spans: token spans of the top-k_txt candidate evidence sentences.
    a_vis: [N_V] — filtered + [0,1]-normalised visual attention (sinks and border zeroed).
    grid_shape: (H_grid, W_grid) spatial layout of visual tokens.
    grid_bbox: (x1, y1, x2, y2) tentative crop bbox in grid coordinates.

    Returns:
        delta_txt — surprisal: high ↔ model not attending evidence → highlighting valuable
        delta_vis — bbox concentration: high ↔ crop is spatially focused → cropping reliable
    """
    # π_txt = fraction of attention mass on evidence tokens; δ_txt = surprisal
    ctx_total = float(a_txt.sum()) + eps
    ev_mass = 0.0
    for (s, e) in top_k_sent_spans:
        if e > s:
            ev_mass += float(a_txt[s:e].sum())
    pi_txt = ev_mass / ctx_total
    delta_txt = float(-np.log2(pi_txt + eps))

    # δ_vis = fraction of clean visual attention concentrated inside the tentative bbox
    delta_vis = 0.0
    if (len(a_vis) > 0 and grid_shape is not None and grid_bbox is not None):
        x1g, y1g, x2g, y2g = grid_bbox
        if x2g > x1g and y2g > y1g:
            a_vis_2d = a_vis.reshape(grid_shape)
            vis_total = float(a_vis.sum()) + eps
            bbox_mass = float(a_vis_2d[y1g:y2g, x1g:x2g].sum())
            delta_vis = bbox_mass / vis_total
    return delta_txt, delta_vis


# ─── Stage F: ETML ─────────────────────────────────────────────────────────────

def compute_step_entropy(
    step_attn: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Shannon entropy (bits) of per-step context attention (pseudocode lines 38-39).

    step_attn: [N_C] — averaged attention from the current decode position to context tokens.
    """
    total = float(step_attn.sum()) + eps
    p = step_attn.astype(np.float64) / total
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    return float(-(p * log_p).sum())


def is_entropy_peak(
    current_entropy: float,
    entropy_history: List[float],
    kappa: float = 1.0,
    eps: float = 1e-8,
) -> bool:
    """
    Detect whether current_entropy is a significant peak above recent history (pseudocode line 40).

    Uses streaming z-score: E(t') > mean(E_hist) + kappa * std(E_hist).
    Returns False when fewer than 2 history points (need mean & std).
    """
    if len(entropy_history) < 2:
        return False
    arr = np.array(entropy_history, dtype=np.float64)
    mu = arr.mean()
    sigma = arr.std() + eps
    return current_entropy > mu + kappa * sigma


# ─── Bbox in pixel space ───────────────────────────────────────────────────────

def compute_bbox_pixels(
    c_x: float,
    c_y: float,
    sigma_x: float,
    sigma_y: float,
    beta: float,
    grid_shape: Tuple[int, int],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Convert grid-space centroid+sigma to pixel-space bounding box (pseudocode line 32).

    grid_shape: (H_grid, W_grid)
    image_size: (W_img, H_img)  — PIL convention

    Returns (x1, y1, x2, y2) in pixel coordinates.
    """
    H_grid, W_grid = grid_shape
    W_img, H_img = image_size

    scale_x = W_img / W_grid
    scale_y = H_img / H_grid

    x1 = max(0, int((c_x - beta * sigma_x) * scale_x))
    y1 = max(0, int((c_y - beta * sigma_y) * scale_y))
    x2 = min(W_img, int((c_x + beta * sigma_x) * scale_x))
    y2 = min(H_img, int((c_y + beta * sigma_y) * scale_y))

    # Ensure non-degenerate box
    if x2 <= x1:
        x2 = min(W_img, x1 + 1)
    if y2 <= y1:
        y2 = min(H_img, y1 + 1)

    return x1, y1, x2, y2
