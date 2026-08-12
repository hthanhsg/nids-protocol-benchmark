"""Merge result directories into one analysis frame.

Stage 1 and the parallel Stage 2 write to separate trees so that two processes
never append to the same CSV. This recombines them, checks that the run keys do
not overlap, and reports what arrived.

    python merge_results.py results_v2 results_v2_seeds -o merged
"""
import argparse
import glob
import os
import pandas as pd

KEY = ["dataset", "model", "split", "identity", "task", "seed", "sampling"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="directories to search recursively")
    ap.add_argument("-o", "--out", default="merged")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for kind in ["results", "per_class", "confusion"]:
        frames = []
        for d in a.dirs:
            for f in sorted(glob.glob(os.path.join(d, "**", f"{kind}.csv"), recursive=True)):
                try:
                    df = pd.read_csv(f)
                except Exception as e:
                    print(f"  !! could not read {f}: {e}")
                    continue
                df["source_file"] = os.path.relpath(f)
                frames.append(df)
                print(f"  + {f}  ({len(df)} rows)")
        if not frames:
            print(f"  no {kind}.csv found")
            continue
        m = pd.concat(frames, ignore_index=True)

        if kind == "results" and set(KEY).issubset(m.columns):
            dup = m[m.duplicated(KEY, keep=False)].sort_values(KEY)
            if len(dup):
                print(f"\n  WARNING: {len(dup)} rows share a run key across directories.")
                print("  The same configuration was run twice; keeping the first of each.")
                print(dup[KEY + ["source_file"]].head(12).to_string(index=False))
                m = m.drop_duplicates(KEY, keep="first")

        out = os.path.join(a.out, f"{kind}.csv")
        m.to_csv(out, index=False)
        print(f"  -> {out}  ({len(m)} rows)\n")

    r = pd.read_csv(os.path.join(a.out, "results.csv"))
    ok = r[r.status == "ok"]
    print("=" * 60)
    print(f"total {len(r)} | ok {len(ok)} | "
          f"skipped {(r.status=='skipped').sum()} | failed {(r.status=='failed').sum()}")
    if len(ok):
        print("\nruns per seed:")
        print(ok.groupby(["task", "seed"]).size().to_string())


if __name__ == "__main__":
    main()