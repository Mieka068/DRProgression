# TO-DO LIST
[] Module 1
[] Module 2
[] Module 3

# Branches
- *Feilong-DRForestGAN* is the original code found via https://zenodo.org/records/15747872.
- *module-2-inference*
- *module-2-batch-inference*

# Dataset
As of May 3, 2026, I used the following dataset to try to apply to *module-2-inference* branch.
You can download the archive.zip through this link: https://www.kaggle.com/datasets/lawateaditya/idrid-dataset?resource=download
It is too big for Github.

As of May 5, I used the ff link for the FIRE dataset: https://projects.ics.forth.gr/cvrl/fire/
and Longitudinal DR dataset: https://academictorrents.com/details/744717095e59373186abec814c86de4831d889e9

# Updates
May 3: inference preliminary testing is a success.
May 5: batch inference (with 5 images) preliminary testing is a success.

# Module 2 FID comparison -- sentence to use in the manuscript/defense
FID (feature=2048) is biased at small sample sizes, not just noisy (Binkowski et al. 2018),
so `train_module2_poc.py`/`evaluate_trajectory.py` also report a bootstrap 95% range and
Kernel Inception Distance (KID, unbiased at any N) alongside the point estimate -- see
`image_quality_metrics.py`. Frame the comparison directionally, not competitively:

> "Our FID is computed on a substantially smaller held-out set than DRForecastGAN's (n=X vs.
> their 2,734-8,523), so it indicates general range rather than a strict head-to-head result;
> KID is reported alongside as a sample-size-robust complement."

Say it once, plainly, rather than either overclaiming comparability or quietly reporting
`feature=64` and hoping it isn't questioned.