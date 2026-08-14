"""Experiment runner.

Design goal: you start it once and walk away. Every finished run is appended to
results/results.csv immediately, so an interrupted job loses at most the run in
flight. Re-running the same command skips everything already recorded, which
means "resume" and "start" are the same command.

Usage
-----
    python -m nidsbench.run --data-dir /path/to/netflow --subsample 1000000

    # smaller first pass, to confirm the pipeline works end to end
    python -m nidsbench.run --data-dir ... --subsample 50000 --models mlp xgboost

    # after everything finishes
    python -m nidsbench.run --data-dir ... --report
"""
from __future__ import annotations
import argparse
import glob
import json
import logging
import os
import platform
import sys
import time
import traceback
import numpy as np
import pandas as pd

from . import data as D
from . import graph as G
from . import metrics as M
from . import models as MO
from . import splits as S

log = logging.getLogger("nidsbench")

SPLITS = ["random", "temporal", "disjoint_ip"]
IDENTITY = ["keep", "drop"]
TASKS = ["binary", "multiclass"]

RESULT_KEY = ["dataset", "model", "split", "identity", "task", "seed", "sampling"]

# Fixed column order. Every row -- completed, skipped or failed -- is written with
# exactly these columns, so appending never misaligns against the existing header.
COLUMNS = RESULT_KEY + [
    "status", "error", "n_features", "n_classes", "n_train", "n_val", "n_test",
    "classes_missing_in_train", "classes_missing_in_val", "classes_missing_in_test",
    "n_entities_train", "n_entities_val", "n_entities_test",
    "best_epoch", "epochs_run", "val_macro_f1",
    "runtime_sec", "test_accuracy", "test_macro_f1", "test_weighted_f1",
    "test_binary_f1", "test_macro_precision", "test_macro_recall", "test_fpr",
    "test_support_test", "test_n_classes_present",
    "test_macro_f1_learnable", "test_n_classes_learnable",
    "identity_semantics", "n_nodes", "test_unseen_node_share", "shuffle_labels",
    "val_uninformative",
]
COLUMNS = ["sampling"] + [c for c in COLUMNS if c != "sampling"]


def find_datasets(data_dir):
    """Locate NF-* files. Version and family are inferred from the filename."""
    pats = ["*.csv", "*.csv.gz", "*.parquet"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(data_dir, "**", p), recursive=True)
    out = {}
    for f in sorted(files):
        base = os.path.basename(f)
        name = base.split(".")[0]
        out[name] = f
    return out


def run_id(dn, model, split, idm, task, seed, sampling="flow"):
    """Filesystem-safe identifier for one run, used to name saved artefacts."""
    return f"{dn}__{model}__{split}__id-{idm}__{task}__s{seed}__{sampling}"


def load_done(path):
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if not set(RESULT_KEY).issubset(df.columns):
        return set()
    return {tuple(str(r[k]) for k in RESULT_KEY) for _, r in df.iterrows()}


def append_row(path, row):
    """Append one row, padding missing keys so the column order is always identical.

    The directory is (re)created on every write. It is created once at startup
    too, but a long sweep can outlive its output directory -- deleting it from
    another shell while the job runs would otherwise lose the run and then lose
    the error row recording that loss.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    full = {c: row.get(c, "") for c in COLUMNS}
    extra = {k: v for k, v in row.items() if k not in full}
    df = pd.DataFrame([{**full, **extra}], columns=COLUMNS + sorted(extra))
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_one(ds, model_name, split_kind, identity, task, seed, outdir, cfg,
            shuffle_labels=False, save_models=False, save_predictions=False,
            sampling="flow"):
    t0 = time.time()
    X, y, classes = D.build_matrix(ds, identity, task)

    if shuffle_labels:
        # Leakage probe. Labels are permuted before splitting, so no feature
        # carries information about them. Any test score meaningfully above
        # chance would mean the pipeline leaks. Diagnostic only.
        y = np.random.RandomState(seed).permutation(y)

    if len(classes) < 2 or len(np.unique(y)) < 2:
        raise S.SplitUnavailable(f"only {len(np.unique(y))} class(es) present")

    tr, va, te = S.make_split(split_kind, ds, y, seed=seed)
    if min(len(tr), len(va), len(te)) < 50:
        raise S.SplitUnavailable(f"split too small: {len(tr)}/{len(va)}/{len(te)}")

    Xtr, Xva, Xte = D.encode(X.iloc[tr], X.iloc[va], X.iloc[te])
    ytr, yva, yte = y[tr], y[va], y[te]
    extra = {}

    # Classes absent from training are unlearnable; record how many so that a
    # low macro-F1 under disjoint splits can be interpreted rather than guessed at.
    # Under disjoint-entity splits, classes routinely appear in only some
    # splits. Recording this per split is what makes a low macro-F1 there
    # interpretable rather than mysterious.
    missing_tr = int(len(classes) - len(np.unique(ytr)))
    missing_va = int(len(classes) - len(np.unique(yva)))
    missing_te = int(len(classes) - len(np.unique(yte)))

    # Entities per split. A test fold drawn from one or two hosts scores well
    # while measuring almost nothing, so the count belongs beside the metric.
    ent_counts = {}
    if "IPV4_SRC_ADDR" in ds.df.columns:
        ent_all = ds.df["IPV4_SRC_ADDR"].astype(str).values
        for nm, idx in (("train", tr), ("val", va), ("test", te)):
            ent_counts[f"n_entities_{nm}"] = int(len(np.unique(ent_all[idx])))

    if model_name == "egraphsage":
        if "IPV4_SRC_ADDR" not in ds.df.columns or "IPV4_DST_ADDR" not in ds.df.columns:
            raise S.SplitUnavailable("no IPV4_SRC_ADDR/IPV4_DST_ADDR -- cannot build a flow graph")
        nsrc, ndst, n_nodes = G.build_node_index(
            ds.df["IPV4_SRC_ADDR"].values, ds.df["IPV4_DST_ADDR"].values)
        # Encode all rows on one shared scaler fitted on train, then index by split.
        Xall = np.zeros((len(X), Xtr.shape[1]), dtype=np.float32)
        Xall[tr], Xall[va], Xall[te] = Xtr, Xva, Xte
        pred_te, pred_va, info = G.fit_predict(
            nsrc, ndst, n_nodes, Xall, y, tr, va, te, len(classes), seed=seed, cfg=cfg)
        extra = {
            # A GNN needs IP to build the graph, so "drop" only removes IP/port
            # from the edge features -- topology still carries host identity.
            "identity_semantics": ("graph_topology_retains_identity" if identity == "drop"
                                   else "identity_in_features_and_topology"),
            "n_nodes": info.get("n_nodes"),
            "test_unseen_node_share": info.get("test_unseen_node_share"),
        }
    else:
        pred_te, pred_va, info = MO.fit_predict(
            model_name, Xtr, ytr, Xva, yva, Xte, len(classes), seed=seed, cfg=cfg)
        extra = {"identity_semantics": f"identity_features_{identity}"}

    test_m, per_cls, cm = M.compute(yte, pred_te, classes, y_train=ytr)
    val_m, _, _ = M.compute(yva, pred_va, classes, y_train=ytr)
    rid = run_id(ds.name, model_name, split_kind, identity, task, seed, sampling)

    row = {
        "dataset": ds.name, "model": model_name, "split": split_kind,
        "identity": identity, "task": task, "seed": seed, "sampling": sampling,
        "n_features": Xtr.shape[1], "n_classes": len(classes),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "classes_missing_in_train": missing_tr,
        "classes_missing_in_val": missing_va,
        "classes_missing_in_test": missing_te,
        **ent_counts,
        "best_epoch": info.get("best_epoch", -1),
        "val_uninformative": info.get("val_uninformative", False),
        "epochs_run": info.get("epochs_run", -1),
        "val_macro_f1": val_m["macro_f1"],
        "runtime_sec": round(time.time() - t0, 1),
        "status": "ok", "error": "",
    }
    row.update({f"test_{k}": v for k, v in test_m.items()})
    row.update(extra)
    fitted = info.pop("_model", None)

    # Per-class report (precision / recall / F1 / support) and confusion matrix.
    for name, records in (("per_class", per_cls),
                          ("confusion", M.confusion_long(cm, classes))):
        if not records:
            continue
        df = pd.DataFrame(records)
        for k in RESULT_KEY:
            df[k] = row[k]
        os.makedirs(outdir, exist_ok=True)
        fp = os.path.join(outdir, f"{name}.csv")
        df.to_csv(fp, mode="a", header=not os.path.exists(fp), index=False)

    if save_predictions:
        pdir = os.path.join(outdir, "predictions")
        os.makedirs(pdir, exist_ok=True)
        np.savez_compressed(os.path.join(pdir, f"{rid}.npz"),
                            y_true=yte.astype(np.int16), y_pred=pred_te.astype(np.int16),
                            classes=np.array(classes, dtype=object),
                            test_index=np.asarray(te, dtype=np.int64))

    if save_models and fitted is not None:
        mdir = os.path.join(outdir, "models")
        os.makedirs(mdir, exist_ok=True)
        _save_model(fitted, model_name, os.path.join(mdir, rid))

    return row


def _save_model(fitted, model_name, stem):
    """Persist a fitted model. Torch models go to .pt, XGBoost to .json."""
    try:
        if model_name == "xgboost":
            fitted.save_model(stem + ".json")
        else:
            import torch
            torch.save({"state_dict": fitted.state_dict(),
                        "class_name": type(fitted).__name__}, stem + ".pt")
    except Exception as e:  # noqa: BLE001 - never let artefact saving kill a run
        log.warning("could not save model %s: %s", os.path.basename(stem), e)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Protocol-vs-architecture benchmark on NF-* datasets")
    ap.add_argument("--data-dir", required=True, help="directory containing NF-*.csv / .parquet")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--subsample", type=int, default=1_000_000,
                    help="max rows per dataset (stratified; rare classes kept whole)")
    ap.add_argument("--models", nargs="+", default=MO.ALL_MODELS)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--identity", nargs="+", default=IDENTITY)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--datasets", nargs="+", default=None, help="filter by name substring")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-epochs", type=int, default=5,
                    help="floor before early stopping may keep a checkpoint")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--eval-batch", type=int, default=8192,
                    help="batch size for validation/test forward passes")
    ap.add_argument("--subsample-by", choices=["flow", "entity"], default="flow",
                    help="flow: stratified on class, preserves class balance but "
                         "thins the host graph. entity: keeps whole hosts, "
                         "preserving graph density. Use entity for graph models.")
    ap.add_argument("--min-per-host", type=int, default=10,
                    help="entity sampling: minimum rows allocated per host. "
                         "Hosts are dropped (smallest first) only when the "
                         "budget cannot give every host this many.")
    ap.add_argument("--cache-dir", default=None, help="cache subsampled data here (recommended)")
    ap.add_argument("--save-models", action="store_true",
                    help="save each fitted model (torch .pt / xgboost .json). "
                         "Adds up over a full sweep -- check disk space first.")
    ap.add_argument("--save-predictions", action="store_true",
                    help="save y_true/y_pred and test indices per run as compressed .npz")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="LEAKAGE PROBE: permute labels before splitting. Test scores "
                         "should collapse to chance. Write to a separate --out.")
    ap.add_argument("--report", action="store_true", help="summarise existing results and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    os.makedirs(a.out, exist_ok=True)
    respath = os.path.join(a.out, "results.csv")

    if a.report:
        return report(respath, a.out)

    with open(os.path.join(a.out, "environment.json"), "w") as f:
        json.dump({
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "torch": getattr(MO, "torch", None) and MO.torch.__version__,
            "cuda": bool(MO.HAS_TORCH and MO.torch.cuda.is_available()),
            "argv": vars(a), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, default=str)

    found = find_datasets(a.data_dir)
    if a.datasets:
        found = {k: v for k, v in found.items() if any(d.lower() in k.lower() for d in a.datasets)}
    if not found:
        log.error("no dataset files under %s", a.data_dir)
        return 1
    log.info("found %d dataset file(s): %s", len(found), ", ".join(found))

    done = load_done(respath)
    log.info("%d run(s) already recorded -- these will be skipped", len(done))

    cfg = {"max_epochs": a.max_epochs, "patience": a.patience,
           "batch_size": a.batch_size, "min_epochs": a.min_epochs,
           "eval_batch": a.eval_batch}

    combos = [(dn, m, sp, idm, tk, sd, a.subsample_by)
              for dn in found for tk in a.tasks for sp in a.splits
              for idm in a.identity for m in a.models for sd in a.seeds]
    todo = [c for c in combos if tuple(str(x) for x in c) not in done]
    log.info("%d run(s) planned, %d to execute", len(combos), len(todo))

    loaded, n_ok, n_skip, n_fail = {}, 0, 0, 0
    for i, (dn, model, split, idm, task, seed, samp) in enumerate(todo, 1):
        tag = f"{dn} | {model} | {split} | id={idm} | {task} | seed={seed} | {samp}"
        log.info("[%d/%d] %s", i, len(todo), tag)
        try:
            if dn not in loaded:
                loaded.clear()  # keep only one dataset in memory at a time
                loaded[dn] = D.load(found[dn], name=dn, subsample=a.subsample,
                                    cache_dir=a.cache_dir, seed=a.seeds[0],
                                    subsample_by=a.subsample_by,
                                    min_per_host=a.min_per_host)
                log.info("  %s", loaded[dn])
            row = run_one(loaded[dn], model, split, idm, task, seed, a.out, cfg,
                          shuffle_labels=a.shuffle_labels,
                          save_models=a.save_models,
                          save_predictions=a.save_predictions,
                          sampling=samp)
            row["shuffle_labels"] = bool(a.shuffle_labels)
            append_row(respath, row)
            n_ok += 1
            log.info("  -> macro-F1 %.4f | acc %.4f | FPR %.4f | %.0fs",
                     row["test_macro_f1"], row["test_accuracy"], row["test_fpr"],
                     row["runtime_sec"])
        except S.SplitUnavailable as e:
            append_row(respath, _stub(dn, model, split, idm, task, seed, "skipped", str(e), samp))
            n_skip += 1
            log.warning("  skipped: %s", e)
        except Exception as e:  # noqa: BLE001 - one bad cell must not kill the sweep
            append_row(respath, _stub(dn, model, split, idm, task, seed, "failed", repr(e), samp))
            n_fail += 1
            log.error("  FAILED: %s", e)
            log.debug(traceback.format_exc())

    log.info("done: %d ok, %d skipped, %d failed -> %s", n_ok, n_skip, n_fail, respath)
    report(respath, a.out)
    return 0


def _stub(dn, model, split, idm, task, seed, status, err, sampling="flow"):
    return {"dataset": dn, "model": model, "split": split, "identity": idm,
            "task": task, "seed": seed, "sampling": sampling,
            "status": status, "error": err[:500]}


def report(respath, outdir):
    """Print the two contrasts the study exists to measure."""
    if not os.path.exists(respath):
        log.error("no results at %s", respath)
        return 1
    df = pd.read_csv(respath)
    ok = df[df["status"] == "ok"].copy()
    print(f"\n{'='*70}\nRUNS: {len(df)} total | {len(ok)} ok | "
          f"{(df['status']=='skipped').sum()} skipped | {(df['status']=='failed').sum()} failed")
    if ok.empty:
        return 0

    print(f"\n--- mean test macro-F1 by protocol cell ---")
    piv = ok.pivot_table(index=["split", "identity"], columns="model",
                         values="test_macro_f1", aggfunc="mean")
    print(piv.round(4).to_string())

    base = ok[(ok.split == "random") & (ok.identity == "keep")]["test_macro_f1"].mean()
    strict = ok[(ok.split != "random") & (ok.identity == "drop")]["test_macro_f1"].mean()
    if np.isfinite(base) and np.isfinite(strict):
        print(f"\nOptimism gap: random+keep = {base:.4f} -> strict = {strict:.4f} "
              f"({(base-strict)*100:+.1f} pp)")

    print(f"\n--- variance decomposition (test macro-F1) ---")
    for factor in ["model", "split", "identity", "dataset"]:
        g = ok.groupby(factor)["test_macro_f1"].mean()
        print(f"  {factor:10s} spread = {g.max()-g.min():.4f}   ({dict(g.round(3))})")
    print("\nIf the split/identity spread exceeds the model spread, protocol "
          "dominates architecture -- the survey's central claim.")

    print(f"\n--- accuracy vs macro-F1 (the metric paradox) ---")
    ok["gap"] = ok["test_accuracy"] - ok["test_macro_f1"]
    print(ok.groupby("task")[["test_accuracy", "test_macro_f1", "gap"]].mean().round(4).to_string())

    summ = os.path.join(outdir, "summary.txt")
    with open(summ, "w") as f:
        f.write(piv.round(4).to_string())
    print(f"\nFull results: {respath}\nPer-class:    {os.path.join(outdir,'per_class.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())