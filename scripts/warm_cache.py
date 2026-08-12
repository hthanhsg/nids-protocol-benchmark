"""Build the subsample cache once, sequentially, before launching parallel jobs.

Reading a raw NF-* CSV is the memory-hungry step: NF-BoT-IoT-v2 is 37M rows and
peaks well above 20 GB as a DataFrame. Two training jobs starting at the same
time would each do that read independently and can exhaust system memory.

This does every read once, one file at a time, and writes the parquet cache the
training jobs then share. Cache keys depend on (path, subsample, seed,
subsample_by, min_per_host), so the values here must match what the jobs use.

    python warm_cache.py --data-dir data --cache-dir cache_entity \
        --subsample 1000000 --subsample-by entity

Then launch the parallel jobs with the same --subsample / --subsample-by /
--cache-dir and they will hit the cache instead of re-reading.
"""
import argparse
import gc
import glob
import logging
import os
import time

from nidsbench import data as D

log = logging.getLogger("warm_cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--subsample", type=int, default=1_000_000)
    ap.add_argument("--subsample-by", choices=["flow", "entity"], default="entity")
    ap.add_argument("--min-per-host", type=int, default=10)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="one cache entry per seed; give every seed you plan to run")
    ap.add_argument("--datasets", nargs="+", default=None)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    files = []
    for pat in ("*.csv", "*.csv.gz", "*.parquet"):
        files += glob.glob(os.path.join(a.data_dir, "**", pat), recursive=True)
    files = sorted(set(files))
    if a.datasets:
        files = [f for f in files
                 if any(d.lower() in os.path.basename(f).lower() for d in a.datasets)]
    if not files:
        log.error("no dataset files under %s", a.data_dir)
        return 1

    total = len(files) * len(a.seeds)
    log.info("warming %d cache entries (%d files x %d seeds)",
             total, len(files), len(a.seeds))
    os.makedirs(a.cache_dir, exist_ok=True)

    i = 0
    for seed in a.seeds:
        for f in files:
            i += 1
            name = os.path.basename(f).split(".")[0]
            t0 = time.time()
            log.info("[%d/%d] %s (seed %d)", i, total, name, seed)
            ds = D.load(f, name=name, subsample=a.subsample, seed=seed,
                        cache_dir=a.cache_dir, subsample_by=a.subsample_by,
                        min_per_host=a.min_per_host)
            log.info("      %d rows, %d features, time=%s, %.0fs",
                     len(ds.df), len(ds.feature_cols), ds.time_col or "NONE",
                     time.time() - t0)
            # Release before the next file: peak memory is one dataset, not all.
            del ds
            gc.collect()

    log.info("cache ready in %s -- parallel jobs can now start safely", a.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())