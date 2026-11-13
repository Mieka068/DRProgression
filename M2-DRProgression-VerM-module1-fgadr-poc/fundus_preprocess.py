"""
Shared fundus-image preprocessing for every Module 2 / Module 3 loader (round 4).

Retina crop + pad to square (item R4-1.2). Tianjin (3:2) and LongDR (2000x1312) photos are not
square; resizing them straight to image_size x image_size squashed the retina into an oval and
shrank small lesions further. load_fundus() instead crops a square box around the retina's field of
view (as wide as its larger side, centred on it) -- PIL pads with black where the box runs past the
image edge -- so the retina keeps its shape and fills the frame. It is the same box
module1/apply_to_progression_data.py's retina_box() uses for grading/segmentation.

Module 1 lesion masks are cached in the ORIGINAL image frame (apply_to_progression_data.py un-crops
them, and --mask-size stores them squashed to N x N), so crop_mask() maps the same box onto the
mask's own resolution before the loaders resize it.

Settings (process-wide, so a notebook can set them once for every script it runs):
    DRP_RETINA_CROP=0                          -> the earlier plain resize (no crop)
    DRP_LONGDR_FOLDER=FundusImagesNormalized   -> LongDR's contrast-normalised images (earlier runs);
                                                  default FundusImagesColor (item R4-1.1: Module 1's
                                                  segmenters mark ~10-30 % of a normal retina as
                                                  lesion on the normalised images)
"""
import os

import numpy as np
from PIL import Image

_BOX_CACHE = {}


def retina_crop_enabled():
    return os.environ.get("DRP_RETINA_CROP", "1").strip().lower() not in ("0", "false", "no", "off")


def longdr_folder():
    return os.environ.get("DRP_LONGDR_FOLDER", "FundusImagesColor")


def main_fov_region(fov: np.ndarray) -> np.ndarray:
    """Keep only the largest connected bright region of a field-of-view mask: the retina disc
    (round 5). Tianjin's follow-up photos print '#2984', the date/time and 'R / Color' in the black
    border; those characters are separate blobs, but stacked in the same columns they passed the
    2 % row/column cut-off in retina_box() and stretched the box over them -- the retina then
    filled only ~60 % of the frame, off-centre (and SIFT rejected the scale change needed to align
    it). DRP_TEXT_SAFE_CROP=0 restores the earlier mask."""
    if os.environ.get("DRP_TEXT_SAFE_CROP", "1").strip().lower() in ("0", "false", "no", "off"):
        return fov
    from scipy import ndimage
    labels, n = ndimage.label(fov)
    if n <= 1:
        return fov
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def retina_box(img: Image.Image, threshold: int = 10, work_size: int = 512):
    """Square (left, top, right, bottom) box around the retina in img's pixel coordinates, or None
    if no clear field of view is found. Found on a <= work_size px copy (fast), then scaled back.
    Same rule as module1/apply_to_progression_data.py's retina_box(). Only the largest bright
    region counts as the retina, so text overlays in the border are ignored (main_fov_region)."""
    w, h = img.size
    scale = min(1.0, work_size / max(w, h))
    small = img if scale == 1.0 else img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                                Image.BILINEAR)
    fov = np.array(small.convert("RGB"))[:, :, 1] > threshold  # green channel
    if fov.sum() < 0.05 * fov.size:
        return None
    fov = main_fov_region(fov)
    if fov.sum() < 0.05 * fov.size:
        return None
    rows, cols = fov.sum(axis=1), fov.sum(axis=0)
    ys = np.nonzero(rows > 0.02 * rows.max())[0]
    xs = np.nonzero(cols > 0.02 * cols.max())[0]
    y0, y1, x0, x1 = ys[0] / scale, (ys[-1] + 1) / scale, xs[0] / scale, (xs[-1] + 1) / scale
    side = int(round(max(y1 - y0, x1 - x0)))
    left = int(round((x0 + x1) / 2 - side / 2))
    top = int(round((y0 + y1) / 2 - side / 2))
    return left, top, left + side, top + side


def cached_retina_box(path, img=None):
    """retina_box() of the image at path, memoised per process (the loaders re-read every epoch)."""
    key = os.path.abspath(path)
    if key not in _BOX_CACHE:
        if img is None:
            img = Image.open(path).convert("RGB")
        _BOX_CACHE[key] = retina_box(img)
    return _BOX_CACHE[key]


def load_fundus(path, retina_crop=None):
    """(RGB image, box, original (W, H)). With the retina crop on (default: DRP_RETINA_CROP), the image
    is cropped to its square retina box; box is None when the crop is off or no retina was found."""
    img = Image.open(path).convert("RGB")
    size = img.size
    if retina_crop is None:
        retina_crop = retina_crop_enabled()
    box = cached_retina_box(path, img) if retina_crop else None
    return (img.crop(box) if box is not None else img), box, size


def crop_mask(mask: np.ndarray, box, image_size):
    """Crop a 0/1 HxW mask cached in the original image frame (full resolution, or squashed to N x N
    by --mask-size) with the image's retina box. image_size = the original image's (W, H)."""
    if box is None:
        return mask
    w, h = image_size
    mh, mw = mask.shape[:2]
    sx, sy = mw / w, mh / h
    left, top, right, bottom = box
    box_m = (int(round(left * sx)), int(round(top * sy)), int(round(right * sx)), int(round(bottom * sy)))
    return np.array(Image.fromarray((mask > 0).astype(np.uint8) * 255).crop(box_m)) > 0


def mask_to_tensor(mask, out_size):
    """0/1 mask (any HxW) -> 1 x out_size x out_size float tensor (nearest neighbour)."""
    import torch
    img = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).resize((out_size, out_size), Image.NEAREST)
    return torch.from_numpy(np.array(img) > 0).float().unsqueeze(0)


def map_points_to_crop(pts_px, box, image_size):
    """Original-pixel (x, y) points -> [0, 1] coordinates of the (cropped, square) frame the loaders
    produce. box None = the whole image (the plain resize)."""
    pts = np.asarray(pts_px, dtype=np.float64)
    if box is None:
        w, h = image_size
        return pts / [w, h]
    left, top, right, _bottom = box
    return (pts - [left, top]) / (right - left)


def map_points_from_crop(pts01, box, image_size):
    """Inverse of map_points_to_crop."""
    pts = np.asarray(pts01, dtype=np.float64)
    if box is None:
        w, h = image_size
        return pts * [w, h]
    left, top, right, _bottom = box
    return pts * (right - left) + [left, top]


def load_module2_image(path, image_size, retina_crop=None):
    """1 x 3 x image_size x image_size tensor in Module 2's [-1, 1] convention, retina-cropped like
    the training loaders -- for evaluate_trajectory.py / visualize_trajectory.py /
    generate_progression.py, so inference sees the same framing as training."""
    from torchvision import transforms
    img, _box, _size = load_fundus(path, retina_crop)
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return tfm(img).unsqueeze(0)


def load_module2_mask(mask, path, image_size, retina_crop=None):
    """A Module 1 cache mask (original image frame) of the image at path -> 1 x 1 x s x s tensor,
    cropped with the same retina box as load_module2_image()."""
    if retina_crop is None:
        retina_crop = retina_crop_enabled()
    box, size = None, None
    if retina_crop:
        with Image.open(path) as im:
            size = im.size
        box = cached_retina_box(path)
    return mask_to_tensor(crop_mask(mask, box, size), image_size).unsqueeze(0)
