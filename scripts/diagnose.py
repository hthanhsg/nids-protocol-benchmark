"""Diagnose suspicious results without stopping the sweep.

Three things produce a high score that looks wrong, and they need different
responses:

  1. Leakage        -- test information reaching the model. Would be a bug.
  2. Degenerate val -- validation holding one class, so model selection is blind.
                       Not a bug, but the selected model is arbitrary.
  3. Easy dataset   -- the corpus is trivially separable, so every protocol
                       scores high. Not a bug; it is a finding about the data.

Run against a partially written results directory:

    python diagnose.py results_v2/binary results_v2/multiclass
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

KEY = ["dataset", "model", "split", "identity", "task", "seed"]


def load(d):
    r = pd.read_csv(os.path.join(d, "results.csv"))
    pc_path = os.path.join(d, "per_class.csv")
    pc = pd.read_csv(pc_path) if os.path.exists(pc_path) else None
    return r[r.status == "ok"].copy(), pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()

    oks, pcs = [], []
    for d in a.dirs:
        try:
            o, p = load(d)
        except FileNotFoundError:
            print(f"  (no results yet in {d})")
            continue
        o["src"] = d
        oks.append(o)
        if p is not None:
            p["src"] = d
            pcs.append(p)
    if not oks:
        sys.exit("no results found")
    ok = pd.concat(oks, ignore_index=True)
    pc = pd.concat(pcs, ignore_index=True) if pcs else None
    print(f"analysing {len(ok)} completed runs\n")

    # ---- 2. Degenerate validation --------------------------------------
    print("=" * 66)
    print("VALIDATION QUALITY  (does val predict test?)")
    print("=" * 66)
    if "classes_missing_in_val" in ok:
        g = ok.groupby("split").agg(
            runs=("val_macro_f1", "size"),
            val_missing_cls=("classes_missing_in_val", "mean"),
            test_missing_cls=("classes_missing_in_test", "mean"),
            val_f1=("val_macro_f1", "mean"),
            test_f1=("test_macro_f1", "mean"),
        )
        g["val_minus_test"] = (g.val_f1 - g.test_f1).round(3)
        print(g.round(3).to_string())

    print("\ncorrelation between val and test macro-F1, by split:")
    for s, sub in ok.groupby("split"):
        if len(sub) > 2 and sub.val_macro_f1.notna().sum() > 2:
            c = sub.val_macro_f1.corr(sub.test_macro_f1)
            verdict = ("val tracks test" if c > 0.7 else
                       "WEAK -- model selection is unreliable here" if c > 0.2 else
                       "BROKEN -- val carries no signal about test")
            print(f"  {s:12s} r = {c:6.3f}   {verdict}")

    bad = ok[(ok.val_macro_f1 - ok.test_macro_f1).abs() > 0.3]
    if len(bad):
        print(f"\n{len(bad)} run(s) where |val - test| > 0.30 "
              f"(selection was effectively blind):")
        cols = ["dataset", "model", "split", "identity", "task",
                "classes_missing_in_val", "val_macro_f1", "test_macro_f1"]
        print(bad[[c for c in cols if c in bad]].round(3).head(15).to_string(index=False))

    # ---- 3. Easy dataset vs real protocol effect -----------------------
    print("\n" + "=" * 66)
    print("PROTOCOL EFFECT PER DATASET  (is the corpus just easy?)")
    print("=" * 66)
    piv = ok.pivot_table(index="dataset", columns="split",
                         values="test_macro_f1", aggfunc="mean")
    if {"random"}.issubset(piv.columns):
        for s in ("temporal", "disjoint_ip"):
            if s in piv.columns:
                piv[f"drop_{s}"] = (piv["random"] - piv[s]).round(3)
    print(piv.round(3).to_string())
    print("\nA dataset whose score barely moves across splits is not evidence of")
    print("leakage; it is a corpus on which the protocol cannot discriminate.")

    # ---- 1. Leakage signature ------------------------------------------
    print("\n" + "=" * 66)
    print("LEAKAGE SIGNATURE  (does keeping identity features help?)")
    print("=" * 66)
    q = ok.pivot_table(index=["dataset", "model", "split", "task"],
                       columns="identity", values="test_macro_f1")
    if {"keep", "drop"}.issubset(q.columns):
        q = q.dropna()
        q["gain"] = (q["keep"] - q["drop"]) * 100
        by = q.reset_index().groupby("split")["gain"].agg(["mean", "max", "size"])
        print("identity gain (keep - drop), percentage points:")
        print(by.round(2).to_string())
        print("\nUnder a random split this gain is the shortcut the survey warns")
        print("about. Under disjoint splits it should shrink towards zero,")
        print("because test entities were never seen during training.")

    # ---- class balance --------------------------------------------------
    if pc is not None and len(pc):
        print("\n" + "=" * 66)
        print("CLASS BALANCE IN TEST  (is the score carried by one class?)")
        print("=" * 66)
        tot = pc.groupby(KEY[:5])["support"].transform("sum")
        pc = pc.assign(share=pc["support"] / tot)
        worst = (pc.sort_values("share", ascending=False)
                   .groupby(["dataset", "split"], as_index=False).first())
        print(worst[["dataset", "split", "class", "share", "f1"]]
              .rename(columns={"class": "largest_class", "share": "its_share"})
              .round(4).to_string(index=False))
        print("\nWhere one class holds >99% of the test set, accuracy is")
        print("meaningless and macro-F1 rests on very few minority samples.")


if __name__ == "__main__":
    main()