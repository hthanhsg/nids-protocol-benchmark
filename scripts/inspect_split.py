"""Inspect the exact split a training run uses, and verify the feature matrix.

Reproduces one cell of the sweep without training anything: it loads the same
cached sample, builds the same split with the same seed, and prints what went
where. Two things it answers:

  * Is the split what it claims to be -- are entities really disjoint, is the
    temporal split really ordered in time, how is each class distributed?
  * Does any label column survive into the features? The check is printed
    explicitly rather than asserted quietly, because "no leakage" is a claim a
    reader should be able to see verified.

    python inspect_split.py --data-dir data --dataset NF-BoT-IoT-v2 \
        --cache-dir cache_entity --subsample 1000000 --subsample-by entity \
        --min-per-host 4 --split disjoint_ip --task binary

Add --dump-indices to write the row indices of each split to CSV.
"""
from __future__ import annotations
import argparse
import glob
import os
import numpy as np
import pandas as pd

from nidsbench import data as D
from nidsbench import splits as S


def find_file(data_dir, name):
    for pat in ("*.csv", "*.csv.gz", "*.parquet"):
        for f in glob.glob(os.path.join(data_dir, "**", pat), recursive=True):
            if os.path.basename(f).split(".")[0] == name:
                return f
    raise SystemExit(f"dataset {name} not found under {data_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--subsample", type=int, default=1_000_000)
    ap.add_argument("--subsample-by", choices=["flow", "entity"], default="entity")
    ap.add_argument("--min-per-host", type=int, default=10)
    ap.add_argument("--split", default="disjoint_ip",
                    choices=["random", "temporal", "disjoint_ip"])
    ap.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    ap.add_argument("--identity", default="keep", choices=["keep", "drop"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump-indices", action="store_true")
    a = ap.parse_args()

    path = find_file(a.data_dir, a.dataset)
    ds = D.load(path, name=a.dataset, subsample=a.subsample, seed=a.seed,
                cache_dir=a.cache_dir, subsample_by=a.subsample_by,
                min_per_host=a.min_per_host)
    print(f"\nloaded {ds}")

    X, y, classes = D.build_matrix(ds, a.identity, a.task)

    # ---------- 1. does any label column survive into X? ----------
    print("\n" + "=" * 70)
    print("FEATURE MATRIX CHECK")
    print("=" * 70)
    print(f"  label columns in this file : binary={ds.label_bin!r}  multi={ds.label_multi!r}")
    print(f"  timestamp column           : {ds.time_col!r}")
    print(f"  identity columns           : {ds.identity_cols}")
    print(f"  X has {X.shape[1]} columns, {len(X):,} rows")

    leaked = [c for c in X.columns
              if c in {ds.label_bin, ds.label_multi, ds.time_col} and c is not None]
    suspicious = [c for c in X.columns
                  if any(k in str(c).lower() for k in
                         ("label", "attack", "class", "target", "malicious"))]
    print(f"\n  label/timestamp columns present in X : {leaked or 'NONE'}")
    print(f"  columns whose name looks label-like  : {suspicious or 'NONE'}")
    print("  VERDICT:", "LEAK -- a label column is a feature!" if leaked
          else "clean, no label column is used as a feature")
    print(f"\n  first 12 feature columns: {list(X.columns[:12])}")

    # ---------- 2. the split ----------
    print("\n" + "=" * 70)
    print(f"SPLIT: {a.split}   (task={a.task}, identity={a.identity}, seed={a.seed})")
    print("=" * 70)
    try:
        tr, va, te = S.make_split(a.split, ds, y, seed=a.seed)
    except S.SplitUnavailable as e:
        print(f"  UNAVAILABLE: {e}")
        return
    n = len(y)
    for nm, idx in (("train", tr), ("val", va), ("test", te)):
        print(f"  {nm:6s} {len(idx):>9,} rows  ({len(idx)/n*100:5.1f}%)")

    # overlap must be empty for any split
    print(f"\n  row overlap train&test: {len(np.intersect1d(tr, te))}   "
          f"train&val: {len(np.intersect1d(tr, va))}   "
          f"val&test: {len(np.intersect1d(va, te))}")

    # ---------- 3. class distribution per split ----------
    print("\n  class distribution (share within each split):")
    rows = []
    for nm, idx in (("train", tr), ("val", va), ("test", te)):
        cnt = pd.Series(y[idx]).value_counts(normalize=True)
        raw = pd.Series(y[idx]).value_counts()
        for ci, cname in enumerate(classes):
            rows.append({"split": nm, "class": cname,
                         "n": int(raw.get(ci, 0)), "share": float(cnt.get(ci, 0.0))})
    dist = pd.DataFrame(rows).pivot(index="class", columns="split", values="share")
    cnts = pd.DataFrame(rows).pivot(index="class", columns="split", values="n")
    print(dist[["train", "val", "test"]].round(4).to_string())
    print("\n  counts:")
    print(cnts[["train", "val", "test"]].to_string())
    missing = [c for ci, c in enumerate(classes) if (y[va] == ci).sum() == 0]
    if missing:
        print(f"\n  classes ABSENT from validation: {missing}")
        print("  -> validation cannot judge these; model selection is partly blind.")

    # ---------- 4. entity disjointness ----------
    ent_col = "IPV4_SRC_ADDR"
    if ent_col in ds.df.columns:
        e = ds.df[ent_col].astype(str).values
        etr, eva, ete = set(e[tr]), set(e[va]), set(e[te])
        print(f"\n  source hosts  train {len(etr):,}  val {len(eva):,}  test {len(ete):,}")
        print(f"  host overlap  train&test {len(etr & ete):,}   "
              f"train&val {len(etr & eva):,}")
        if a.split == "disjoint_ip":
            print("  VERDICT:", "disjoint as intended" if not (etr & ete)
                  else "NOT DISJOINT -- hosts appear on both sides!")

    dst = "IPV4_DST_ADDR"
    if dst in ds.df.columns and a.split == "disjoint_ip":
        d = ds.df[dst].astype(str).values
        ov = len(set(d[tr]) & set(d[te]))
        print(f"  destination hosts shared train&test: {ov:,}")
        print("  (the split partitions sources only, so destinations recur by design)")

    # ---------- 5. temporal ordering ----------
    if ds.time_col is not None:
        t = pd.to_numeric(ds.df[ds.time_col], errors="coerce").values
        print(f"\n  {ds.time_col} range per split:")
        for nm, idx in (("train", tr), ("val", va), ("test", te)):
            v = t[idx]
            v = v[np.isfinite(v)]
            if len(v):
                print(f"    {nm:6s} {v.min():.0f} .. {v.max():.0f}")
        if a.split == "temporal":
            ok = t[tr].max() <= t[va].min() and t[va].max() <= t[te].min()
            print("  VERDICT:", "strictly ordered past -> future" if ok
                  else "ranges overlap (ties at the boundary are normal)")

    if a.dump_indices:
        for nm, idx in (("train", tr), ("val", va), ("test", te)):
            f = f"split_{a.dataset}_{a.split}_{a.task}_{nm}.csv"
            pd.DataFrame({"row_index": np.sort(idx)}).to_csv(f, index=False)
            print(f"  wrote {f}")


if __name__ == "__main__":
    main()