"""
Live progress for DRG-Net's segmentation training in notebook 11 (pipe its output through this).

DRG-Net prints "Starting epoch N/M." every epoch, the bare eval AUC every eval interval, and (FGADR)
the shape/range of every eval image. This shows the first epoch, every --every-th, the last one
and everything else except the per-image debug lines, flushing each line so Colab displays it as
it happens; the full, unfiltered output is appended to --log (put it on Drive).

Usage:  python -u train_fgadr.py ... 2>&1 | python -u seg_progress.py --every 10 --log run.log
"""
import argparse
import re
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--every", type=int, default=10)
parser.add_argument("--log", default=None)
args = parser.parse_args()

log = open(args.log, "a") if args.log else None
start = time.time()
EPOCH = re.compile(r"Starting epoch (\d+)/(\d+)")
for line in sys.stdin:
    if log:
        log.write(line)
        log.flush()
    s = line.strip()
    if not s or s.startswith(("img:", "Mask:")):
        continue
    m = EPOCH.search(s)
    if m:
        n, total = int(m.group(1)), int(m.group(2))
        if n == 1 or n % args.every == 0 or n == total:
            print(f"epoch {n}/{total}  ({(time.time() - start) / 60:.1f} min this session)", flush=True)
        continue
    try:  # eval_model() prints the eval-set ROC-AUC as a bare number every eval interval
        print(f"  eval ROC-AUC: {float(s):.4f}", flush=True)
    except ValueError:
        print(s, flush=True)
if log:
    log.close()
