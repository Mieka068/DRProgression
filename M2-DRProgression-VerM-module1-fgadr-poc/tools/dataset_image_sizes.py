"""
Image-size distribution per dataset (CODE_FIXES item 7.3) -- LongDR, Tianjin and EyePACS image
sizes were undocumented. Reads only the image headers (PIL opens lazily), so it is fast even on
EyePACS.

Usage:
    python tools/dataset_image_sizes.py fire=/content/data/FIRE_dataset \
        longdr=/content/data/LongDRScreening_20150209 tianjin=/content/data/retinal-dr-longitudinal \
        [--max-images 5000] [--out sizes.json]
Prints, per dataset: image count, the most common (W x H) sizes, width/height min / median / max,
and the share of non-square images; --out also writes it as JSON.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def image_sizes(root, max_images=None):
    sizes = []
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith(IMAGE_EXTS) or "mask" in f.lower():
                continue
            try:
                with Image.open(os.path.join(dirpath, f)) as im:
                    sizes.append(im.size)  # (W, H), header only
            except Exception:
                continue
            if max_images and len(sizes) >= max_images:
                return sizes
    return sizes


def summarize(sizes, top=5):
    if not sizes:
        return {"n_images": 0}
    w = np.array([s[0] for s in sizes])
    h = np.array([s[1] for s in sizes])
    return {
        "n_images": len(sizes),
        "most_common": [{"size": f"{a}x{b}", "count": c} for (a, b), c in Counter(sizes).most_common(top)],
        "width": {"min": int(w.min()), "median": float(np.median(w)), "max": int(w.max())},
        "height": {"min": int(h.min()), "median": float(np.median(h)), "max": int(h.max())},
        "non_square_fraction": float(np.mean(w != h)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+", metavar="NAME=DIR")
    p.add_argument("--max-images", type=int, default=None, help="Stop after this many images per dataset")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    report = {}
    for item in args.datasets:
        name, root = item.split("=", 1) if "=" in item else (os.path.basename(item.rstrip("/")), item)
        report[name] = summarize(image_sizes(root, args.max_images))
        r = report[name]
        print(f"\n{name} ({root}): {r['n_images']} images")
        if r["n_images"]:
            print("  most common: " + ", ".join(f"{m['size']} ({m['count']})" for m in r["most_common"]))
            print(f"  width  min/median/max: {r['width']['min']} / {r['width']['median']:.0f} / {r['width']['max']}")
            print(f"  height min/median/max: {r['height']['min']} / {r['height']['median']:.0f} / {r['height']['max']}")
            print(f"  non-square: {r['non_square_fraction']:.1%}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
