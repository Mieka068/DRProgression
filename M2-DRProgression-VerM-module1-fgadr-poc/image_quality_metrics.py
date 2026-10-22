"""
Shared FID/KID computation with an honest treatment of small-sample instability
(IMPLEMENTATION_PLAN_3.md's Task L addendum).

FID at feature=2048 is a *biased* estimator at small N, not just a noisy one -- the point
estimate is systematically inflated with few samples and only shrinks toward the true value as
N grows (Binkowski et al. 2018 introduced Kernel Inception Distance, KID, specifically because
of this). feature=64 avoids the bias by measuring a different, lower-level property of the
images (texture/edges, not semantic content) -- it trades comparability against DRForecastGAN's
own (assumed feature=2048) numbers away for stability, which is the wrong trade. So:

  1. Always compute FID at feature=2048 (comparable to DRForecastGAN's own reported numbers),
     never fall back to feature=64 to make it look more stable.
  2. Report a resampled FID range alongside the point estimate (2.5th/97.5th percentile over
     n_bootstrap resamples). A wide range is itself an honest, reportable finding at POC scale,
     not something to hide by only reporting the point estimate. Default (CODE_FIXES item 5.5):
     SUBSAMPLING without replacement (subsample_fraction of the images, real and fake
     separately). The earlier with-replacement bootstrap duplicates images, which shrinks the
     sample's diversity and inflates FID, so its range could sit entirely above the point
     estimate (e.g. 233.7 vs [235.8, 243.2]). Subsampled FID still runs a little high (fewer
     images), so read the range as the spread at subsample_fraction x N, not a confidence
     interval for the point estimate; KID +/- std is the sample-size-robust number.
     fid_range_method says which was used; range_method="bootstrap" is the earlier behaviour.
  3. Report Kernel Inception Distance (KID) as a supplementary, sample-size-robust (unbiased
     at any N) complement -- NOT a substitute for the FID comparison, since DRForecastGAN's
     paper reports FID, not KID.

Used by both train_module2_poc.py (single-step POC metrics) and evaluate_trajectory.py
(per-cascade-step and self-consistency metrics), so this same honest-uncertainty treatment
applies everywhere FID is reported, not just one call site.
"""
import numpy as np


def compute_fid_kid_stats(real_imgs, fake_imgs, device, feature=2048, min_n_for_fid=50,
                           n_bootstrap=10, kid_subset_size=None, bootstrap_seed=0,
                           range_method="subsample", subsample_fraction=0.8):
    """
    real_imgs, fake_imgs: lists of [1,3,H,W] CPU or device tensors in [0,1] range (already
    denormalized). FID/KID are unpaired distributional comparisons -- real_imgs[i] need not
    correspond to fake_imgs[i], and the two lists don't need equal length.

    Returns a dict:
      n_samples: int, min(len(real_imgs), len(fake_imgs))
      fid: float or None (point estimate on the full set; None if n_samples < min_n_for_fid,
        or torchmetrics/torch-fidelity isn't installed)
      fid_bootstrap_range: [low, high] or None (2.5th/97.5th percentile FID across
        n_bootstrap resamples -- subsamples without replacement by default, see point 2 above;
        None under the same gates as fid, or if fewer than 2 resamples computed without error)
      fid_range_method: "subsample (<fraction>, without replacement)" or "bootstrap (with replacement)"
      kid_mean, kid_std: float or None (torchmetrics' own KID mean/std estimate across its
        internal subsets; None under the same n_samples gate, or on any KID-specific failure)
      note: str or None, explains any None fields
    """
    n_samples = min(len(real_imgs), len(fake_imgs))
    result = {
        "n_samples": n_samples, "fid": None, "fid_bootstrap_range": None,
        "fid_range_method": (f"subsample ({subsample_fraction:g}, without replacement)"
                             if range_method == "subsample" else "bootstrap (with replacement)"),
        "kid_mean": None, "kid_std": None, "note": None,
    }
    if n_samples < min_n_for_fid:
        result["note"] = (
            f"n_samples ({n_samples}) < min_n_for_fid ({min_n_for_fid}) -- FID/KID are "
            "distributional metrics unreliable on this few samples, omitted rather than "
            "reported misleadingly."
        )
        return result

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        result["note"] = "torchmetrics/torch-fidelity not installed -- skipping FID/KID."
        return result

    def _fid_on(real_subset, fake_subset):
        m = FrechetInceptionDistance(feature=feature, normalize=True).to(device)
        for img in real_subset:
            m.update(img.to(device), real=True)
        for img in fake_subset:
            m.update(img.to(device), real=False)
        return float(m.compute())

    result["fid"] = _fid_on(real_imgs, fake_imgs)

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_vals = []
    for _ in range(n_bootstrap):
        if range_method == "subsample":
            real_idx = rng.choice(len(real_imgs), size=max(2, int(len(real_imgs) * subsample_fraction)), replace=False)
            fake_idx = rng.choice(len(fake_imgs), size=max(2, int(len(fake_imgs) * subsample_fraction)), replace=False)
        else:
            real_idx = rng.integers(0, len(real_imgs), size=len(real_imgs))
            fake_idx = rng.integers(0, len(fake_imgs), size=len(fake_imgs))
        try:
            bootstrap_vals.append(_fid_on([real_imgs[i] for i in real_idx], [fake_imgs[i] for i in fake_idx]))
        except Exception:
            continue  # a degenerate resample (e.g. near-duplicate covariance) -- skip, don't crash the run
    if len(bootstrap_vals) >= 2:
        result["fid_bootstrap_range"] = [
            float(np.percentile(bootstrap_vals, 2.5)), float(np.percentile(bootstrap_vals, 97.5))
        ]

    try:
        from torchmetrics.image.kid import KernelInceptionDistance

        subset_size = kid_subset_size or max(2, min(50, n_samples // 2))
        kid_metric = KernelInceptionDistance(
            subset_size=subset_size, subsets=min(100, n_bootstrap * 10), feature=feature, normalize=True
        ).to(device)
        for img in real_imgs:
            kid_metric.update(img.to(device), real=True)
        for img in fake_imgs:
            kid_metric.update(img.to(device), real=False)
        kid_mean, kid_std = kid_metric.compute()
        result["kid_mean"] = float(kid_mean)
        result["kid_std"] = float(kid_std)
    except Exception as e:
        kid_note = f"KID computation failed ({e}) -- omitted."
        result["note"] = f"{result['note']} {kid_note}" if result["note"] else kid_note

    return result
