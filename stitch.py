#!/usr/bin/env python3
"""
stitch.py — unified screenshot / image stitcher.

Stitches a sequence of overlapping images into one continuous image. Combines
the detection quality of the "enhanced" lineage (template + multi-scale matching,
overlap acceptance tests, quality scoring, logging) with the practical features
of the original viewport stitcher (vertical OR horizontal direction, permanent
chrome cropping, edge-masking during matching, and debug seam previews).

Internals always work in a "vertical frame" (images stacked top-to-bottom).
For horizontal stitching the images are transposed on load and the result is
transposed back on save — so there is a single, well-tested code path.

Usage:
  python3 stitch.py ./screens                 # whole folder, auto-named output
  python3 stitch.py a.png b.png c.png          # specific files, in this order
  python3 stitch.py shots/*.png -o out.png     # shell glob + explicit name
  python3 stitch.py ./screens -H -e -v         # horizontal, edges, verbose
  python3 stitch.py ./screens --crop-right 16 --ignore-perp 24   # strip macOS chrome
"""
import os
import argparse
import logging
from typing import Tuple

import cv2
import numpy as np

# ---------- logging ----------

def setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )

# ---------- IO / small helpers ----------

def imread_alpha(path: str, crop_right: int = 0, crop_bottom: int = 0):
    """The image's alpha channel (cropped like the image), or None if opaque."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    alpha = apply_crop(img[:, :, 3], crop_right, crop_bottom)
    return None if (alpha == 255).all() else np.ascontiguousarray(alpha)

def imread_rgb(path: str) -> np.ndarray:
    """Load image as 3-channel BGR (drops alpha, expands grayscale)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")

def resolve_inputs(inputs):
    """Turn CLI inputs into an ordered list of image paths.

    A single directory is expanded to its images, sorted by name. Explicit files
    (and shell-expanded globs) keep the order given. Directories listed among
    files are expanded in place. Non-images and the implicit absence of args are
    handled by the caller.
    """
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files.extend(sorted(
                os.path.join(item, f) for f in os.listdir(item)
                if not f.startswith(".") and f.lower().endswith(IMG_EXTS)))
        elif os.path.isfile(item) and item.lower().endswith(IMG_EXTS):
            files.append(item)
        else:
            logging.warning(f"skipping (not an image / not found): {item}")
    return files

def auto_output_name(inputs, files):
    """Derive an output path when -o is omitted: '<folder>-stitched.png' beside
    the source folder; 'stitched.png' in the cwd for loose files. Never clobbers
    an existing file — appends -2, -3, ... until the name is free.
    """
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        src = inputs[0]
    else:
        dirs = {os.path.dirname(os.path.abspath(f)) for f in files}
        src = os.path.commonpath(list(dirs)) if len(dirs) == 1 else None

    if src and os.path.abspath(src) != os.path.abspath("."):
        base = os.path.basename(os.path.abspath(src))
        parent = os.path.dirname(os.path.abspath(src))
        stem = os.path.join(parent, f"{base}-stitched")
    else:
        stem = "stitched"

    candidate = f"{stem}.png"
    n = 2
    while os.path.exists(candidate):
        candidate = f"{stem}-{n}.png"
        n += 1
    return candidate

def to_work_frame(img: np.ndarray, direction: str) -> np.ndarray:
    """Map an image into the internal vertical frame.

    Transpose is its own inverse, so the same function maps back out.
    For horizontal stitching, swapping rows<->cols turns a left-to-right
    sequence into a top-to-bottom one.
    """
    if direction == "horizontal":
        if img.ndim == 3:
            return np.ascontiguousarray(np.transpose(img, (1, 0, 2)))
        return np.ascontiguousarray(img.T)
    return img

def apply_crop(img: np.ndarray, crop_right: int, crop_bottom: int) -> np.ndarray:
    """Permanently remove px from the right/bottom edges (real screen coords)."""
    h, w = img.shape[:2]
    if crop_right >= w or crop_bottom >= h:
        logging.warning(f"crop ({crop_right}r/{crop_bottom}b) >= image size {w}x{h}; clamping")
    w2 = max(1, w - crop_right) if crop_right > 0 else w
    h2 = max(1, h - crop_bottom) if crop_bottom > 0 else h
    if w2 == w and h2 == h:
        return img
    return img[:h2, :w2]

def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    """Left-align the image in a target-width canvas, replicating the right edge.

    Used to normalise frames to a common width without discarding content: the
    narrower frame gets a margin rather than the wider frame being cropped. A
    no-op when widths already match (the usual scroll-capture case).
    """
    w = img.shape[1]
    if w >= target_w:
        return img
    return cv2.copyMakeBorder(img, 0, 0, 0, target_w - w, cv2.BORDER_REPLICATE)

def gray(img: np.ndarray, edges: bool = False) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if edges:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        g = cv2.convertScaleAbs(np.sqrt(gx * gx + gy * gy))
    return g

def mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float('inf')
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(d * d))

def trim_perp(g: np.ndarray, ignore_perp: int) -> np.ndarray:
    """Drop the trailing `ignore_perp` columns (perpendicular edge) before matching.

    In the vertical frame the perpendicular axis is columns (image width). For
    horizontal input these were the bottom rows before transpose — i.e. exactly
    the place a scrollbar / preview thumbnail tends to sit.
    """
    if ignore_perp > 0 and g.shape[1] - ignore_perp >= 10:
        return g[:, :g.shape[1] - ignore_perp]
    return g

def adaptive_max_shift(img_width: int) -> int:
    """Cap the perpendicular-drift search to a sane fraction of image width."""
    return min(50, max(20, img_width // 80))

def pad_canvas(stitched: np.ndarray, curr_w: int, curr_x: int) -> Tuple[np.ndarray, int]:
    """Grow the canvas horizontally so it covers curr_x .. curr_x+curr_w."""
    h, w = stitched.shape[:2]
    left = max(0, -curr_x)
    right = max(0, curr_x + curr_w - w)
    if left or right:
        stitched = cv2.copyMakeBorder(stitched, 0, 0, left, right, cv2.BORDER_REPLICATE)
    return stitched, left

def pad_image(img: np.ndarray, total_w: int, left_pad: int) -> np.ndarray:
    h, w = img.shape[:2]
    right = max(0, total_w - (left_pad + w))
    if left_pad == 0 and right == 0:
        return img
    return cv2.copyMakeBorder(img, 0, 0, left_pad, right, cv2.BORDER_REPLICATE)

# ---------- joint alignment detection ----------
#
# Overlap and perpendicular shift cannot be estimated independently: the seam
# strips only line up once BOTH are right. So we search the (overlap, shift)
# plane jointly, coarse-to-fine, minimising MSE over the actual overlap strip.

def _strip_mse(r1: np.ndarray, r2: np.ndarray, shift: int, rstep: int = 1) -> float:
    """MSE of two equal-height gray strips after a horizontal shift.

    r1/r2 share the same row count (the overlap); shift slides columns only.
    rstep subsamples rows to speed up the coarse pass.
    """
    if shift > 0:
        a = r1[:, shift:]
        b = r2[:, :r2.shape[1] - shift] if r2.shape[1] > shift else r2[:, :0]
    elif shift < 0:
        s = -shift
        a = r1[:, :r1.shape[1] - s] if r1.shape[1] > s else r1[:, :0]
        b = r2[:, s:]
    else:
        a, b = r1, r2
    w = min(a.shape[1], b.shape[1])
    if w < 10:
        return float('inf')
    a, b = a[:, :w], b[:, :w]
    if rstep > 1:
        a, b = a[::rstep], b[::rstep]
    return mse(a, b)

def detect_alignment(prev: np.ndarray, curr: np.ndarray,
                     min_overlap: int = 50, max_overlap: int = 800, max_shift: int = 30,
                     edges: bool = False, ignore_perp: int = 0,
                     shift_accept_ratio: float = 0.008, overlap_accept_ratio: float = 0.05,
                     coarse: bool = True) -> dict:
    """Jointly estimate vertical overlap and perpendicular shift between two images.

    Returns a dict with accepted/raw shift and overlap plus MSE diagnostics.
    `coarse=True` runs a strided coarse pass then refines (fast); `coarse=False`
    searches the full grid at single-pixel resolution (slower, most precise).
    """
    pg = trim_perp(gray(prev, edges), ignore_perp)
    cg = trim_perp(gray(curr, edges), ignore_perp)
    h1, h2 = pg.shape[0], cg.shape[0]
    max_search = min(max_overlap, h1, h2)
    if max_search <= min_overlap or max_shift < 0:
        # too small to search a real overlap -> concatenate (0), don't trim content
        return dict(shift=0, raw_shift=0, overlap=0, raw_overlap=0,
                    overlap_found=False, mse_zero=0.0, mse_best=0.0)

    enable_shift = max_shift > 0

    def strips(off):
        return pg[h1 - off:h1, :], cg[0:off, :]

    # Coarse pass. Tall frames are searched on a downscaled copy (every offset at
    # 1/f scale); small ones with a strided full-res search.
    f = (4 if min(h1, h2) >= 800 else 2 if min(h1, h2) >= 400 else 1) if coarse else 1
    if f > 1 and max_search // f < -(-min_overlap // f):
        f = 1                                            # range too narrow to downscale
    if f > 1:
        ps = cv2.resize(pg, (pg.shape[1] // f, h1 // f), interpolation=cv2.INTER_AREA)
        cs = cv2.resize(cg, (cg.shape[1] // f, h2 // f), interpolation=cv2.INTER_AREA)
        hs1 = ps.shape[0]
        lo, hi = -(-min_overlap // f), max_search // f
        ms = max_shift // f if enable_shift else 0
        best = (float('inf'), 0, lo)
        for off in range(lo, hi + 1):
            r1, r2 = ps[hs1 - off:hs1, :], cs[0:off, :]
            for sh in range(-ms, ms + 1):
                m = _strip_mse(r1, r2, sh, 1)
                if m < best[0]:
                    best = (m, sh, off)
        _, c_sh, c_off = best[0], best[1] * f, best[2] * f
        off_step = sh_step = 2 * f                       # refine window: +-2 coarse px
    else:
        off_step, sh_step, rstep = (3, 2, 2) if coarse else (1, 1, 1)
        offs = list(range(min_overlap, max_search + 1, off_step))
        if offs[-1] != max_search:
            offs.append(max_search)
        coarse_shifts = list(range(-max_shift, max_shift + 1, sh_step)) if enable_shift else [0]

        best = (float('inf'), 0, min_overlap)  # (mse, shift, overlap)
        for off in offs:
            r1, r2 = strips(off)
            for sh in coarse_shifts:
                m = _strip_mse(r1, r2, sh, rstep)
                if m < best[0]:
                    best = (m, sh, off)
        _, c_sh, c_off = best

    # refine around the coarse optimum at full row resolution
    off_lo, off_hi = max(min_overlap, c_off - off_step), min(max_search, c_off + off_step)
    sh_lo, sh_hi = max(-max_shift, c_sh - sh_step), min(max_shift, c_sh + sh_step)
    best = (float('inf'), c_sh, c_off)
    for off in range(off_lo, off_hi + 1):
        r1, r2 = strips(off)
        fine_shifts = range(sh_lo, sh_hi + 1) if enable_shift else [0]
        for sh in fine_shifts:
            m = _strip_mse(r1, r2, sh, 1)
            if m < best[0]:
                best = (m, sh, off)
    mse_best, raw_shift, raw_off = best

    # --- acceptance: only deviate from the safe default if it clearly helps ---
    r1_min, r2_min = strips(min_overlap)
    mse_zero = _strip_mse(r1_min, r2_min, 0, 1)            # shift 0 @ min overlap

    r1_off, r2_off = strips(raw_off)
    mse_shift0 = _strip_mse(r1_off, r2_off, 0, 1)          # shift 0 @ chosen overlap
    min_eff = max(1, max_shift // 15)
    shift_improve = (mse_shift0 - mse_best) / (mse_shift0 + 1e-12)
    if enable_shift and shift_improve >= shift_accept_ratio and abs(raw_shift) >= min_eff:
        acc_shift, mse_at_acc = raw_shift, mse_best
    else:
        acc_shift, mse_at_acc = 0, mse_shift0

    mse_min_acc = _strip_mse(r1_min, r2_min, acc_shift, 1)  # accepted shift @ min overlap
    overlap_improve = (mse_min_acc - mse_at_acc) / (mse_min_acc + 1e-12)
    # No real overlap -> 0, so the next frame is concatenated whole (nothing trimmed)
    # rather than losing min_overlap px of genuine content to a phantom seam.
    found = overlap_improve >= overlap_accept_ratio
    acc_overlap = raw_off if found else 0

    logging.debug(f"align raw(shift={raw_shift}, overlap={raw_off}) "
                  f"accepted(shift={acc_shift}, overlap={acc_overlap}) found={found} "
                  f"shift_improve={shift_improve:.4f} overlap_improve={overlap_improve:.4f}")
    return dict(shift=int(acc_shift), raw_shift=int(raw_shift),
                overlap=int(acc_overlap), raw_overlap=int(raw_off),
                overlap_found=bool(found),
                mse_zero=float(mse_zero), mse_best=float(mse_best))

# ---------- quality assessment ----------

def assess_stitch_quality(prev: np.ndarray, curr: np.ndarray, shift: int, overlap: int) -> dict:
    try:
        h1 = prev.shape[0]
        if overlap <= 0 or overlap >= min(h1, curr.shape[0]):
            return {"quality_score": 0.0, "confidence": "low"}
        po = prev[h1 - overlap:, :]
        co = curr[:overlap, :]
        if shift > 0:
            po = po[:, shift:] if shift < po.shape[1] else po[:, :0]
            co = co[:, :-shift] if shift < co.shape[1] else co[:, :0]
        elif shift < 0:
            s = -shift
            po = po[:, :-s] if s < po.shape[1] else po[:, :0]
            co = co[:, s:] if s < co.shape[1] else co[:, :0]
        w = min(po.shape[1], co.shape[1])
        if w < 10:
            return {"quality_score": 0.0, "confidence": "low"}
        mse_score = mse(gray(po[:, :w]), gray(co[:, :w]))
        # Calibrated to 8-bit content: an aligned seam is MSE ~0-150; a poorly
        # matched one runs into the thousands. (Old /10000 scaling rated almost
        # everything "high".)
        quality_score = max(0.0, 1.0 - mse_score / 4000.0)
        confidence = "high" if mse_score < 300 else "medium" if mse_score < 1500 else "low"
        return {"quality_score": quality_score, "confidence": confidence,
                "mse": mse_score, "overlap_size": (overlap, w)}
    except Exception:
        return {"quality_score": 0.0, "confidence": "error"}

# ---------- debug ----------

def save_debug(prev: np.ndarray, curr: np.ndarray, shift: int, overlap: int,
               index: int, direction: str, debug_dir: str):
    """Save a human-readable seam preview for a pair (transposed back if needed)."""
    os.makedirs(debug_dir, exist_ok=True)
    h1 = prev.shape[0]
    off = max(1, min(overlap, h1, curr.shape[0]))
    prev_tail = prev[h1 - off:, :].copy()
    curr_head = curr[:off, :].copy()
    cv2.rectangle(prev_tail, (0, 0), (prev_tail.shape[1] - 1, prev_tail.shape[0] - 1), (0, 255, 0), 2)
    cv2.rectangle(curr_head, (0, 0), (curr_head.shape[1] - 1, curr_head.shape[0] - 1), (0, 0, 255), 2)
    try:
        preview = np.vstack([prev_tail, curr_head])
    except ValueError:
        preview = prev_tail
    preview = to_work_frame(preview, direction)  # back to display orientation
    cv2.imwrite(os.path.join(debug_dir, f"seam_{index:03d}.png"), preview)

# ---------- sticky header/footer detection ----------

def detect_sticky_bands(imgs, max_frac: float = 0.4, tol: float = 25.0,
                        min_std: float = 8.0) -> Tuple[int, int]:
    """Find leading/trailing rows that are near-identical across ALL frames.

    These are fixed-position UI chrome (nav bars, footers) that repeat in every
    screenshot rather than scrolling. Returns (top_height, bottom_height) in
    work-frame rows. Compares against the first frame at aligned columns (sticky
    chrome doesn't drift). Capped at max_frac of the shortest frame.

    A band must also carry real content (`min_std`): a plain white/uniform margin
    matches across frames too, but it isn't chrome — de-duplicating it is
    pointless, so blank bands are reported as 0.
    """
    if len(imgs) < 2:
        return 0, 0
    H = min(im.shape[0] for im in imgs)
    B = max(0, int(H * max_frac))
    if B < 2:
        return 0, 0
    grays = [gray(im).astype(np.float32) for im in imgs]

    def band_height(slicer):
        ref = slicer(grays[0])
        err = np.stack([((slicer(g) - ref) ** 2).mean(axis=1) for g in grays[1:]])
        row_err = err.max(axis=0)                       # worst-matching frame per row
        bad = np.where(row_err > tol)[0]
        n = int(bad[0]) if bad.size else B
        # ignore blank/uniform bands (whitespace margins, not chrome)
        if n and slicer(grays[0])[:n].std() < min_std:
            return 0
        return n

    th = band_height(lambda g: g[:B])
    # bottom band: measure from the edge inward, then map back to a row count
    bh = band_height(lambda g: g[g.shape[0] - B:][::-1])
    return th, bh

# ---------- scrolling-region detection ----------

def detect_motion_box(imgs, tol: int = 16, min_frac: float = 0.1):
    """Find the region whose pixels change between frames — the scrolling area.

    For whole-window captures: toolbars, sidebars and fixed headers stay put
    while the scrolled content moves. Returns (y0, y1, x0, x1) in real image
    coords, or None if frames differ in size or nothing changed.

    Changed pixels (any adjacent pair, gray diff > `tol`) are closed into blobs;
    blobs smaller than `min_frac` of the largest (a spinner, a blinking cursor)
    are dropped, and the box spans the rest. Blank margins inside the scrolling
    area never change, so they fall outside the box.
    """
    if len(imgs) < 2 or any(im.shape != imgs[0].shape for im in imgs):
        return None
    H, W = imgs[0].shape[:2]
    grays = [gray(im) for im in imgs]
    mask = np.zeros((H, W), np.uint8)
    for a, b in zip(grays, grays[1:]):
        mask[cv2.absdiff(a, b) > tol] = 255
    if not mask.any():
        return None
    k = max(3, min(H, W) // 20) | 1        # odd, so the closing doesn't shift the box
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = stats[1:]
    comps = comps[comps[:, cv2.CC_STAT_AREA] >= comps[:, cv2.CC_STAT_AREA].max() * min_frac]
    x0 = int(comps[:, cv2.CC_STAT_LEFT].min())
    y0 = int(comps[:, cv2.CC_STAT_TOP].min())
    x1 = int((comps[:, cv2.CC_STAT_LEFT] + comps[:, cv2.CC_STAT_WIDTH]).max())
    y1 = int((comps[:, cv2.CC_STAT_TOP] + comps[:, cv2.CC_STAT_HEIGHT]).max())
    return y0, y1, x0, x1

def apply_fixed_cols(box, fixed_cols, width: int):
    """Shrink the scrolling region's columns to exclude declared fixed UI.

    `fixed_cols` are (a, b) column ranges, e.g. a scrollbar reported by the OS.
    A range in the region's right half moves its right edge in; one in the left
    half moves its left edge in. Ranges outside the region change nothing.
    Returns (box, ranges that applied).
    """
    y0, y1, x0, x1 = box
    applied = []
    for a, b in fixed_cols:
        a, b = max(0, a), min(width, b)
        if b <= x0 or a >= x1 or a >= b:
            continue
        nx0, nx1 = (x0, min(x1, a)) if a > (x0 + x1) / 2 else (max(x0, b), x1)
        if nx1 - nx0 < 10:
            logging.warning(f"--fixed-cols {a}:{b} would leave no scrolling region; ignored")
            continue
        x0, x1 = nx0, nx1
        applied.append((a, b))
    return (y0, y1, x0, x1), applied

def transparent_rows(alpha, from_end: bool = False) -> int:
    """Rows at the top (or bottom) edge that contain any transparency — the
    window's rounded corners in a macOS window capture."""
    if alpha is None:
        return 0
    opaque = (alpha == 255).all(axis=1)
    if from_end:
        opaque = opaque[::-1]
    idx = np.where(opaque)[0]
    # no opaque row at all (a translucent window) or a deep band: not corners
    if not idx.size or idx[0] > len(opaque) // 8:
        return 0
    return int(idx[0])

def panes(strip: np.ndarray, tol: int = 20):
    """Split a side strip into panes where the background colour changes
    (a grey sidebar next to a white list), as [start, end) column ranges."""
    med = np.median(strip, axis=0).astype(np.int16)              # (w, 3)
    cuts = np.flatnonzero(np.abs(np.diff(med, axis=0)).max(axis=1) > tol) + 1
    bounds = [0, *cuts.tolist(), strip.shape[1]]
    return list(zip(bounds[:-1], bounds[1:]))

def pinned_bottom(strip: np.ndarray, zone: float = 0.15, min_gap: float = 0.05,
                  tol: int = 12) -> int:
    """Row where a pane's bottom-pinned controls start, or the strip height.

    Pinned controls (a "..." button at a sidebar's foot) sit in the bottom
    `zone` of the pane, below a band of empty rows at least `min_gap` of its
    height. Content that runs to the bottom (a list) has no such band, so it
    stays in place and nothing is moved.
    """
    vh = strip.shape[0]
    g = gray(strip).astype(np.int16)
    flat = np.r_[False, (g.max(axis=1) - g.min(axis=1)) <= tol, False]
    edges = np.flatnonzero(np.diff(flat.astype(np.int8)))
    starts, ends = edges[::2], edges[1::2]          # empty-row runs [start, end)
    need = max(8, int(vh * min_gap))
    ok = (ends - starts >= need) & (ends >= vh * (1 - zone)) & (ends < vh)
    return int(ends[ok].max()) if ok.any() else vh

def compose_window(first: np.ndarray, last: np.ndarray, body: np.ndarray, box,
                   a_first=None, a_last=None, blank_cols=()):
    """Put the fixed UI back around the stitched scrolling region, once.

    Top band from the first frame, bottom band from the last (its end state).
    Side columns come from the first frame only, followed by each column's
    median (its background) to fill the extra height. Exception: controls
    pinned to a pane's bottom, below a band of empty space, are moved to the
    bottom (see `pinned_bottom`). `blank_cols` (a declared scrollbar)
    are filled with their median throughout: with all content shown, there is
    nothing left to scroll. With alphas given, the window's top and bottom rows
    (rounded corners) are copied from the first and last frame with their
    transparency. Returns (bgr, alpha); alpha is None when there's nothing
    transparent to keep.
    """
    y0, y1, x0, x1 = box
    bh = body.shape[0]
    # side rows inside the window's transparent bottom corners; the real corners
    # are restored from the last frame below
    corner = max(0, y1 - (first.shape[0] - transparent_rows(a_first, from_end=True)))

    def side(c0, c1):
        sf, sl = first[y0:y1, c0:c1], last[y0:y1, c0:c1]
        vh, w = sf.shape[:2]
        if w == 0:
            return np.zeros((bh, 0, 3), np.uint8)
        fill = np.median(sf, axis=0).astype(np.uint8)[None]   # (1, w, 3)
        if bh <= vh:
            out = sf[:bh].copy()
        else:
            src = sf[:max(1, vh - corner)]
            parts = []
            for p0, p1 in panes(src):              # each pane decides on its own
                pane, pfill = src[:, p0:p1], fill[:, p0:p1]
                split = pinned_bottom(pane) if p1 - p0 >= 8 else len(pane)
                # the dropped corner rows go back as filler at the very bottom
                # (overwritten by the restored corners), keeping pinned controls
                # at their distance from the window's edge
                parts.append(np.vstack([pane[:split], np.repeat(pfill, bh - vh, axis=0),
                                        pane[split:], np.repeat(pfill, vh - len(pane), axis=0)]))
            out = np.hstack(parts)
        if blank_cols:
            # track colour: the thumb starts at the top of the first frame and
            # ends at the bottom of the last, so these quarters show the track
            track = np.median(np.vstack([sf[3 * vh // 4:], sl[:max(1, vh // 4)]]),
                              axis=0).astype(np.uint8)[None]
            for a, b in blank_cols:
                a, b = max(a, c0) - c0, min(b, c1) - c0
                if a < b:
                    out[:, a:b] = track[:, a:b]
        return out

    mid = np.hstack([side(0, x0), body, side(x1, first.shape[1])])
    W = mid.shape[1]

    def fit(band):                             # body may be wider if frames drifted
        if band.shape[1] < W:
            return cv2.copyMakeBorder(band, 0, 0, 0, W - band.shape[1], cv2.BORDER_REPLICATE)
        return band[:, :W]

    top = [fit(first[:y0])] if y0 > 0 else []
    bottom = [fit(last[y1:])] if last.shape[0] > y1 else []
    out = np.vstack(top + [mid] + bottom)

    k = transparent_rows(a_first)
    m = transparent_rows(a_last, from_end=True)
    if not (k or m) or W != first.shape[1] or out.shape[0] < k + m:
        return out, None
    # The output's first rows are the first frame's, its last rows the last
    # frame's; restoring them restores the rounded corners exactly.
    alpha = np.full(out.shape[:2], 255, np.uint8)
    if k:
        out[:k], alpha[:k] = first[:k], a_first[:k]
    if m:
        out[-m:], alpha[-m:] = last[-m:], a_last[-m:]
    return out, alpha

def seam_row(prev_ov: np.ndarray, curr_ov: np.ndarray, band: int = 64) -> int:
    """Row of the overlap to cut at for --prefer middle.

    Searches the overlap, minus a margin at each end, for the row where the two frames agree
    best across a band of rows around it, so content that changed between the
    captures (an animation) is taken whole from one frame instead of being cut
    through. Ties go to the exact middle, so identical frames cut there.
    """
    O = prev_ov.shape[0]
    q = max(O // 16, min(48, O // 4))       # margin: clear of edge fades and shadows
    if O - 2 * q < 3:
        return O // 2
    diff = cv2.absdiff(gray(prev_ov), gray(curr_ov)).mean(axis=1).astype(np.float32)
    k = max(1, min(band, (O - 2 * q) // 2)) | 1
    smooth = cv2.blur(diff.reshape(-1, 1), (1, k)).ravel()
    rows = np.arange(q, O - q)
    cost = smooth[rows] + 1e-3 * np.abs(rows - O // 2)   # tiny pull toward the middle
    return int(rows[np.argmin(cost)])

# ---------- main stitcher ----------

def stitch_images(files, output: str,
                  direction: str = "vertical",
                  min_overlap: int = 50, max_overlap: int = 800,
                  max_shift: int = 30, shift_accept_ratio: float = 0.008,
                  overlap_accept_ratio: float = 0.05,
                  guard: int = 0, prefer: str = "earlier", feather: int = 0,
                  crop_right: int = 0, crop_bottom: int = 0, ignore_perp: int = 0,
                  window: bool = False, content_only: bool = False, fixed_cols=(),
                  sticky: bool = False, sticky_top: int = -1, sticky_bottom: int = -1,
                  no_shift: bool = False, no_overlap: bool = False,
                  edges: bool = False, multiscale: bool = True,
                  strict: bool = False, dry_run: bool = False,
                  debug: bool = False, verbose: bool = False):
    """Stitch `files` into `output`.

    Returns the number of low-quality seams (>=0) on success, or None if nothing
    could be produced (fewer than 2 usable images).
    """
    setup_logging(verbose)

    files = [f for f in files if os.path.abspath(f) != os.path.abspath(output)]
    if len(files) < 2:
        print("Need at least 2 images to stitch.")
        return None

    # Load -> permanent crop (real coords) -> work frame (transpose if horizontal).
    # A single unreadable file is skipped, not fatal, so one stray file in a
    # folder doesn't abort the whole batch.
    imgs, loaded = [], []
    for f in files:
        try:
            img = apply_crop(imread_rgb(f), crop_right, crop_bottom)
        except RuntimeError as e:
            logging.warning(f"skipping unreadable image: {e}")
            continue
        imgs.append(img)
        loaded.append(f)
        logging.debug(f"Loaded {os.path.basename(f)}: {img.shape}")
    if len(imgs) < 2:
        print(f"Need at least 2 readable images to stitch (got {len(imgs)}).")
        return None

    imgs = [to_work_frame(im, direction) for im in imgs]

    # Whole-window frames: stitch only the scrolling region; the fixed UI around
    # it is put back once at the end (unless content_only).
    window = window or content_only
    full, box, alphas, scrollbars = imgs, None, [None, None], []
    if window:
        box = detect_motion_box(imgs)
        if box is None:
            logging.warning("--window: no common scrolling region found "
                            "(frames identical or differ in size); stitching whole frames")
        else:
            if fixed_cols:
                box, scrollbars = apply_fixed_cols(box, fixed_cols, imgs[0].shape[1])
            y0, y1, x0, x1 = box
            logging.info(f"Scrolling region: rows {y0}..{y1}, cols {x0}..{x1} "
                         f"(of {imgs[0].shape[0]}x{imgs[0].shape[1]}, work frame)")
            imgs = [im[y0:y1, x0:x1] for im in imgs]
            if not content_only:
                # transparency (rounded window corners) of the frames the fixed
                # UI is taken from
                alphas = [imread_alpha(f, crop_right, crop_bottom) for f in (loaded[0], loaded[-1])]
                alphas = [to_work_frame(a, direction) if a is not None else None for a in alphas]

    # Normalize perpendicular extent (work-frame width) so seams line up.
    common_w = max(img.shape[1] for img in imgs)
    if any(img.shape[1] != common_w for img in imgs):
        logging.warning(f"frames differ in width; padding all to {common_w}px (no cropping)")
    imgs = [pad_to_width(img, common_w) for img in imgs]

    # Sticky chrome: strip the repeated band from interior frames, keeping the
    # header on the first frame and the footer on the last.
    sticky_requested = sticky or sticky_top >= 0 or sticky_bottom >= 0
    if not sticky_requested:
        # Detect-and-hint only: never strip content unasked. Restrict to a
        # chrome-like size range so we don't mistake heavy overlap (small scroll
        # step) or 1px noise for fixed chrome.
        H = min(im.shape[0] for im in imgs)
        dt, db = detect_sticky_bands(imgs)
        hint = [(name, n) for name, n in (("header", dt), ("footer", db))
                if 12 <= n < 0.35 * H]
        if hint:
            bits = ", ".join(f"{name} ~{n}px" for name, n in hint)
            print(f"ℹ Looks like a fixed {bits} repeats across all frames. "
                  f"Re-run with --sticky to de-duplicate it.")
    if sticky_requested:
        th = sticky_top if sticky_top >= 0 else 0
        bh = sticky_bottom if sticky_bottom >= 0 else 0
        if sticky and sticky_top < 0 and sticky_bottom < 0:
            th, bh = detect_sticky_bands(imgs)
        logging.info(f"Sticky bands: top={th}px (kept on first), bottom={bh}px (kept on last)")
        last = len(imgs) - 1
        stripped = []
        for i, im in enumerate(imgs):
            top_cut = 0 if i == 0 else th
            bot_cut = 0 if i == last else bh
            if top_cut + bot_cut >= im.shape[0]:
                logging.warning(f"    frame {i} (height {im.shape[0]}) is shorter than the "
                                f"sticky bands (top={top_cut}+bottom={bot_cut}); not stripping it")
                stripped.append(im)
            elif top_cut or bot_cut:
                stripped.append(im[top_cut: im.shape[0] - bot_cut])
            else:
                stripped.append(im)
        imgs = stripped

    prev_x = 0
    actual_max_shift = min(max_shift, adaptive_max_shift(common_w))
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(output)) or ".", "debug_seams")

    logging.info(f"{'Dry-run: analysing' if dry_run else 'Stitching'} {len(imgs)} images "
                 f"(direction={direction}). max_shift={actual_max_shift}, edges={edges}, "
                 f"search={'coarse-to-fine' if multiscale else 'exhaustive'}")

    # Pass 1: measure every seam. Pass 2 assembles, after the window crop has been
    # refined with what the seams revealed.
    quality_scores = []
    low_seams = []
    aligns = []
    for i in range(1, len(imgs)):
        prev, curr = imgs[i - 1], imgs[i]

        eff_max_shift = 0 if (no_shift or actual_max_shift <= 0) else actual_max_shift
        if no_overlap:
            accepted_shift = best_shift = 0
            accepted_overlap = best_off = 0
            mse_zero = mse_best = 0.0
            overlap_found = True   # user explicitly chose concatenation
        else:
            a = detect_alignment(
                prev, curr, min_overlap, max_overlap, eff_max_shift, edges,
                ignore_perp, shift_accept_ratio, overlap_accept_ratio, coarse=multiscale)
            accepted_shift, best_shift = a["shift"], a["raw_shift"]
            accepted_overlap, best_off = a["overlap"], a["raw_overlap"]
            mse_zero, mse_best = a["mse_zero"], a["mse_best"]
            overlap_found = a["overlap_found"]

        quality = assess_stitch_quality(prev, curr, accepted_shift, accepted_overlap)
        # An undetected overlap means these frames were just concatenated — that's
        # not a confident seam, regardless of how the boundary pixels happened to score.
        if not no_overlap and not overlap_found:
            quality = {"quality_score": 0.0, "confidence": "low"}
        quality_scores.append(quality)

        logging.info(f"[{i}] {os.path.basename(files[i-1])} <-> {os.path.basename(files[i])}")
        logging.info(f"    shift: raw={best_shift:3d} accepted={accepted_shift:3d}  "
                     f"overlap: raw={best_off:3d} accepted={accepted_overlap:3d}  "
                     f"(MSE {mse_zero:.1f} -> {mse_best:.1f})")
        logging.info(f"    quality: {quality['confidence']} (score {quality['quality_score']:.3f})")
        if not no_overlap and not overlap_found:
            logging.warning("    no overlap detected — frames concatenated as-is")
            low_seams.append((i, os.path.basename(files[i-1]), os.path.basename(files[i])))
        elif quality['confidence'] == 'low':
            logging.warning("    low-quality stitch")
            low_seams.append((i, os.path.basename(files[i-1]), os.path.basename(files[i])))

        if debug:
            save_debug(prev, curr, accepted_shift, accepted_overlap, i, direction, debug_dir)
        aligns.append((accepted_shift, accepted_overlap))

    stitched = None if dry_run else imgs[0].copy()
    for i, (accepted_shift, accepted_overlap) in enumerate(aligns, start=1):
        if dry_run:
            break
        curr = imgs[i]

        # place curr in stitched coordinates
        curr_x = prev_x + accepted_shift
        stitched, left_added = pad_canvas(stitched, curr.shape[1], curr_x)
        if left_added:
            prev_x += left_added
            curr_x += left_added
        curr_padded = pad_image(curr, stitched.shape[1], left_pad=curr_x)

        O = min(accepted_overlap, stitched.shape[0], curr_padded.shape[0])
        if feather > 0 and O > 0:
            # blend a transition across the overlap instead of a hard cut
            F = min(feather, O)
            prev_ov = stitched[-O:].astype(np.float32)
            curr_ov = curr_padded[:O].astype(np.float32)
            w = np.zeros(O, np.float32)
            w[O - F:] = np.linspace(0.0, 1.0, F)
            w = w[:, None, None]
            blended = ((1 - w) * prev_ov + w * curr_ov).astype(np.uint8)
            stitched = np.vstack((stitched[:-O], blended))
            tail = curr_padded[O:, :]
            if tail.size:
                stitched = np.vstack((stitched, tail))
        elif prefer == "middle":
            # cut away from the overlap's ends (keeps frame edges with fades or
            # shadows out of the seam), where the frames agree best
            keep = seam_row(stitched[-O:], curr_padded[:O]) if O > 0 else 0
            if O - keep > 0:
                stitched = stitched[:stitched.shape[0] - (O - keep)]
            stitched = np.vstack((stitched, curr_padded[keep:, :]))
        elif prefer == "earlier":
            cut = min(curr_padded.shape[0], accepted_overlap + guard)
            tail = curr_padded[cut:, :]
            if tail.size == 0:
                logging.warning(f"    overlap ({cut}px) covers the whole frame; appending it "
                                f"untrimmed to avoid dropping content")
                stitched = np.vstack((stitched, curr_padded))
            else:
                stitched = np.vstack((stitched, tail))
        else:
            cut_st = min(stitched.shape[0], accepted_overlap + guard)
            if cut_st >= stitched.shape[0]:
                logging.warning(f"    overlap+guard ({cut_st}px) covers the whole canvas; "
                                f"appending without trimming to avoid losing earlier content")
                stitched = np.vstack((stitched, curr_padded))
            elif cut_st > 0:
                stitched = np.vstack((stitched[:-cut_st, :], curr_padded))
            else:
                stitched = np.vstack((stitched, curr_padded))

        prev_x = curr_x

    if not dry_run:
        alpha = None
        if box is not None and not content_only:
            stitched, alpha = compose_window(full[0], full[-1], stitched, box, *alphas,
                                            blank_cols=scrollbars)
        stitched = to_work_frame(stitched, direction)  # transpose back for horizontal
        if alpha is not None:
            stitched = np.dstack([stitched, to_work_frame(alpha, direction)])
        out_parent = os.path.dirname(os.path.abspath(output))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        # OpenCV's default PNG settings are ~5x larger than zlib level 6 for screenshots
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6] if output.lower().endswith(".png") else []
        if not cv2.imwrite(output, stitched, params):
            raise RuntimeError(f"could not write '{output}' "
                               f"(check the path, extension, and permissions)")
        logging.info(f"Saved {output}  ({stitched.shape[1]} x {stitched.shape[0]})")

    if quality_scores:
        avg = float(np.mean([q['quality_score'] for q in quality_scores]))
        high = sum(1 for q in quality_scores if q['confidence'] == 'high')
        logging.info(f"Quality: {high}/{len(quality_scores)} high, avg {avg:.3f}")
    if low_seams:
        print(f"\n⚠ {len(low_seams)} low-quality seam(s) — inspect these pairs"
              f"{' (try --edges, --max-shift, or --debug)' if not debug else ''}:")
        for idx, a_name, b_name in low_seams:
            print(f"    [{idx}] {a_name} <-> {b_name}")
    if debug:
        logging.info(f"Debug seam previews in {debug_dir}/")
    return len(low_seams)

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(
        description="Unified screenshot/image stitcher (vertical or horizontal).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 stitch.py ./screens                 # whole folder -> ./screens-stitched.png
  python3 stitch.py a.png b.png c.png          # specific files, in this order
  python3 stitch.py shots/*.png -o out.png     # shell glob + explicit name
  python3 stitch.py ./screens -H -e -v         # horizontal, edges, verbose
  python3 stitch.py ./screens --crop-right 16 --ignore-perp 24 -d
""")
    p.add_argument("inputs", nargs="*", default=["."],
                   help="image files (in order), a shell glob, or a single folder "
                        "(default: current folder)")
    p.add_argument("-o", "--output", default=None,
                   help="output path (default: '<folder>-stitched.png', auto-numbered "
                        "if it exists)")
    p.add_argument("--direction", choices=["vertical", "horizontal"], default="vertical",
                   help="stitch axis (default: vertical)")
    p.add_argument("-H", "--horizontal", action="store_true",
                   help="shorthand for --direction horizontal")

    p.add_argument("--min-overlap", type=int, default=50)
    p.add_argument("--max-overlap", type=int, default=800)
    p.add_argument("--max-shift", "--max-x-shift", dest="max_shift", type=int, default=30,
                   help="max perpendicular drift in px to search")
    p.add_argument("--shift-accept-ratio", type=float, default=0.008,
                   help="relative MSE improvement over zero-shift to accept a drift")
    p.add_argument("--overlap-accept-ratio", type=float, default=0.05,
                   help="relative MSE improvement over min-overlap to accept an overlap")

    p.add_argument("--guard", type=int, default=0, help="extra px removed at the seam")
    p.add_argument("--prefer", choices=["earlier", "later", "middle"], default="earlier",
                   help="which frame's pixels fill the overlap; 'middle' cuts mid-overlap, "
                        "keeping frame edges (fades, shadows) out of the seam")
    p.add_argument("--feather", type=int, default=0, metavar="N",
                   help="blend an N-px transition across each seam (hides faint seam lines)")

    p.add_argument("--crop-right", type=int, default=0,
                   help="permanently crop px from the right edge of each image")
    p.add_argument("--crop-bottom", type=int, default=0,
                   help="permanently crop px from the bottom edge of each image")
    p.add_argument("--ignore-perp", "--ignore-right", "--ignore-bottom", dest="ignore_perp",
                   type=int, default=0,
                   help="ignore px on the trailing perpendicular edge during matching "
                        "(e.g. macOS scrollbar/thumbnail)")
    p.add_argument("--window", action="store_true",
                   help="frames are whole-window captures: stitch only the region that "
                        "scrolls and keep the fixed UI (toolbars, sidebars, scrollbar) once")
    p.add_argument("--content-only", action="store_true",
                   help="like --window, but drop the fixed UI and output only the "
                        "scrolled content")
    def col_range(s):
        try:
            a, b = (int(v) for v in s.split(":"))
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected A:B (two integers), got {s!r}")
        if not 0 <= a < b:
            raise argparse.ArgumentTypeError(f"expected 0 <= A < B, got {s!r}")
        return a, b

    p.add_argument("--fixed-cols", action="append", default=[], metavar="A:B", type=col_range,
                   help="with --window: px range A..B across the stitch axis (columns "
                        "for vertical) that is fixed UI even though it changes, e.g. a "
                        "scrollbar; kept once instead of stitched. Repeatable")

    p.add_argument("--sticky", action="store_true",
                   help="auto-detect & de-duplicate fixed headers/footers repeated in every shot")
    p.add_argument("--sticky-top", type=int, default=-1, metavar="N",
                   help="force a sticky header of N px (implies --sticky behaviour)")
    p.add_argument("--sticky-bottom", type=int, default=-1, metavar="N",
                   help="force a sticky footer of N px (implies --sticky behaviour)")

    p.add_argument("--no-shift", "--no-x-shift", dest="no_shift", action="store_true")
    p.add_argument("--no-overlap", "--no-y-overlap", dest="no_overlap", action="store_true")
    p.add_argument("-e", "--edges", action="store_true",
                   help="match on Sobel edges (better for text/UI)")
    p.add_argument("--multiscale", action="store_true", default=True)
    p.add_argument("--no-multiscale", dest="multiscale", action="store_false")

    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any seam is low-quality")
    p.add_argument("--dry-run", action="store_true",
                   help="report per-seam overlap/shift/quality without writing output")
    p.add_argument("-d", "--debug", action="store_true", help="save seam previews to debug_seams/")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.horizontal:
        args.direction = "horizontal"

    inputs = args.inputs or ["."]
    setup_logging(args.verbose)
    files = resolve_inputs(inputs)
    if len(files) < 2:
        print(f"Need at least 2 images to stitch (found {len(files)}).")
        return 1
    output = args.output if (args.output or args.dry_run) else auto_output_name(inputs, files)

    try:
        low = stitch_images(
            files=files, output=output or "", direction=args.direction,
            min_overlap=args.min_overlap, max_overlap=args.max_overlap,
            max_shift=args.max_shift, shift_accept_ratio=args.shift_accept_ratio,
            overlap_accept_ratio=args.overlap_accept_ratio,
            guard=args.guard, prefer=args.prefer, feather=args.feather,
            crop_right=args.crop_right, crop_bottom=args.crop_bottom,
            ignore_perp=args.ignore_perp, window=args.window, content_only=args.content_only,
            fixed_cols=args.fixed_cols,
            sticky=args.sticky, sticky_top=args.sticky_top, sticky_bottom=args.sticky_bottom,
            no_shift=args.no_shift, no_overlap=args.no_overlap,
            edges=args.edges, multiscale=args.multiscale,
            strict=args.strict, dry_run=args.dry_run,
            debug=args.debug, verbose=args.verbose)
        if low is None:                       # nothing produced (too few usable images)
            return 1
        if args.strict and low:
            print(f"--strict: {low} low-quality seam(s); exiting non-zero.")
            return 1
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
