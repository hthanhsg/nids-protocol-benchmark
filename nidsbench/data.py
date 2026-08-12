"""Loading and preprocessing for the standardised NetFlow datasets (NF-*-v2 / v3)."""
from __future__ import annotations
import hashlib
import logging
import os
import numpy as np
import pandas as pd

log = logging.getLogger("nidsbench.data")

# Identity fields. Dropped under identity="drop"; also used to build disjoint-IP splits.
IDENTITY_COLS = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
    "IPV6_SRC_ADDR", "IPV6_DST_ADDR",
]
# Candidate timestamp columns, in priority order (v2 and v3 differ).
TIME_COLS = [
    "FLOW_START_MILLISECONDS", "FLOW_START_MS", "FLOW_END_MILLISECONDS",
    "FIRST_SWITCHED", "timestamp", "Timestamp", "ts",
]
LABEL_BIN = ["Label", "label", "binary_label"]
LABEL_MULTI = ["Attack", "attack", "attack_cat", "Attack_cat", "category"]


def _find(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


class Dataset:
    """A loaded NF-* dataset with its schema resolved."""

    def __init__(self, name, df, label_bin, label_multi, time_col, identity_cols, feature_cols):
        self.name = name
        self.df = df
        self.label_bin = label_bin
        self.label_multi = label_multi
        self.time_col = time_col
        self.identity_cols = identity_cols
        self.feature_cols = feature_cols

    def __repr__(self):
        return (f"<Dataset {self.name} n={len(self.df)} feats={len(self.feature_cols)} "
                f"time={self.time_col} identity={len(self.identity_cols)}>")


def load(path, name=None, subsample=None, seed=42, cache_dir=None,
         subsample_by="flow", min_per_host=10):
    """Load one NF-* file (csv/csv.gz/parquet), resolve schema, optionally subsample.

    Two sampling modes, and the choice matters more than it looks:

    ``flow``   Stratified on the class label. Preserves class balance, but it
               draws flows scattered across the whole capture, so the hosts
               behind them are sampled thinly. On NF-BoT-IoT a 500k flow sample
               recovered 44 of 293 hosts and on NF-ToN-IoT 2,172 of 29,245,
               leaving any graph model to learn from a far sparser topology
               than the corpus actually contains.

    ``entity`` Selects whole hosts and keeps every flow they sent. Host density
               and therefore graph structure are preserved at their true value;
               class balance is whatever those hosts happen to produce, so rare
               classes are protected explicitly by reserving hosts that carry
               them before the remaining budget is filled at random.

    Use ``entity`` whenever a graph model is involved. Record which mode
    produced a result: the two are not interchangeable.
    """
    name = name or os.path.splitext(os.path.basename(path))[0].replace(".csv", "")
    key = hashlib.md5(
        f"{os.path.abspath(path)}|{subsample}|{seed}|{subsample_by}|{min_per_host}".encode()
    ).hexdigest()[:12]
    cache = os.path.join(cache_dir, f"{name}_{key}.parquet") if cache_dir else None

    if cache and os.path.exists(cache):
        log.info("cache hit %s", cache)
        try:
            df = pd.read_parquet(cache)
        except ImportError:
            log.warning("parquet engine missing; ignoring cache "
                        "(pip install pyarrow to enable it)")
            cache = None
            df = None
    else:
        df = None

    if df is None:
        log.info("reading %s", path)
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        lb = _find(df.columns, LABEL_BIN)
        lm = _find(df.columns, LABEL_MULTI)
        if lb is None and lm is None:
            raise ValueError(f"{name}: no label column found in {list(df.columns)[:40]}")

        if subsample and len(df) > subsample:
            if subsample_by == "entity":
                df = _entity_subsample(df, lm or lb, subsample, seed,
                                       min_per_host=min_per_host)
                log.info("%s entity-subsampled to %d rows", name, len(df))
            else:
                strat = df[lm] if lm else df[lb]
                df = _stratified_subsample(df, strat, subsample, seed)
                log.info("%s subsampled to %d rows", name, len(df))

        if cache:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                df.to_parquet(cache, index=False)
            except ImportError:
                log.warning("parquet engine missing; not caching "
                            "(pip install pyarrow to enable it)")

    lb = _find(df.columns, LABEL_BIN)
    lm = _find(df.columns, LABEL_MULTI)
    tc = _find(df.columns, TIME_COLS)
    ident = [c for c in IDENTITY_COLS if c in df.columns]

    drop = set(filter(None, [lb, lm]))
    feature_cols = [c for c in df.columns if c not in drop]

    if tc is None:
        log.warning("%s: no timestamp column -> temporal split unavailable", name)
    if not ident:
        log.warning("%s: no identity columns found -> identity ablation is a no-op", name)

    return Dataset(name, df, lb, lm, tc, ident, feature_cols)


def _stratified_subsample(df, strat, n, seed):
    rng = np.random.RandomState(seed)
    strat = strat.astype(str)
    counts = strat.value_counts()
    frac = n / len(df)
    keep = []
    for cls, cnt in counts.items():
        idx = np.where(strat.values == cls)[0]
        # Keep rare classes whole; sample proportionally from common ones.
        take = cnt if cnt <= 1000 else max(1000, int(round(cnt * frac)))
        take = min(take, cnt)
        keep.append(rng.choice(idx, size=take, replace=False))
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return df.iloc[keep].reset_index(drop=True)


def build_matrix(ds: Dataset, identity: str, task: str):
    """Return (X_df, y, class_names) for the requested identity mode and task.

    identity: "keep" retains IP/port columns as features; "drop" removes them.
    task:     "binary" or "multiclass".
    """
    assert identity in ("keep", "drop")
    assert task in ("binary", "multiclass")

    cols = list(ds.feature_cols)
    if identity == "drop":
        cols = [c for c in cols if c not in ds.identity_cols]
    # The timestamp orders the temporal split; it must never be a feature.
    if ds.time_col in cols:
        cols.remove(ds.time_col)

    X = ds.df[cols].copy()

    if task == "binary":
        if ds.label_bin is not None:
            y_raw = ds.df[ds.label_bin].astype(str)
            y = (~y_raw.isin(["0", "0.0", "False", "false", "Benign", "benign"])).astype(int).values
        else:
            y_raw = ds.df[ds.label_multi].astype(str).str.lower()
            y = (~y_raw.isin(["benign", "normal", "none", "0"])).astype(int).values
        classes = ["benign", "attack"]
    else:
        if ds.label_multi is None:
            raise ValueError(f"{ds.name}: multiclass requested but no attack-category column")
        y_raw = ds.df[ds.label_multi].astype(str).fillna("unknown")
        classes = sorted(y_raw.unique())
        mapping = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(mapping).values

    return X, y, classes


def encode(X_tr, X_va, X_te):
    """Fit encoders on train only, apply to all three splits.

    Numeric -> median impute + standardise. Categorical (incl. IP strings) ->
    frequency encoding learned on train; unseen categories map to 0.
    """
    num = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X_tr.columns if c not in num]

    out = []
    med = mu = sd = None
    if num:
        # Infinities must be removed *before* the statistics are fitted. Several
        # NF-* columns contain inf (e.g. rate features divided by a zero
        # duration); leaving them in makes the fitted mean inf and the fitted
        # std nan, which then poisons every scaled row.
        tr_num = X_tr[num].replace([np.inf, -np.inf], np.nan)
        med = tr_num.median()
        med = med.fillna(0.0)
        tr_num = tr_num.fillna(med)
        mu = tr_num.mean()
        sd = tr_num.std().replace(0, 1.0).fillna(1.0)

    freqs = {c: X_tr[c].astype(str).value_counts(normalize=True) for c in cat}

    for X in (X_tr, X_va, X_te):
        parts = []
        if num:
            a = X[num].replace([np.inf, -np.inf], np.nan).fillna(med)
            z = ((a - mu) / sd).values.astype(np.float32)
            # Clip so a single extreme outlier cannot dominate a batch.
            parts.append(np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -10, 10))
        if cat:
            b = np.column_stack([
                X[c].astype(str).map(freqs[c]).fillna(0.0).values for c in cat
            ]).astype(np.float32)
            parts.append(b)
        out.append(np.hstack(parts) if parts else np.zeros((len(X), 0), np.float32))

    return out[0], out[1], out[2]


def _entity_subsample(df, label_col, n, seed, entity_col="IPV4_SRC_ADDR",
                      min_per_host=10):
    """Sample so that host coverage is preserved, not just class balance.

    Flow-level stratification samples hosts in proportion to their traffic, so
    on a skewed corpus the long tail of small hosts disappears and the graph a
    GNN sees is far sparser than the real one. This allocates the row budget
    *per host* instead: every host keeps a share, capped so a few hubs cannot
    consume everything, and within a host the quota is spread across the classes
    that host actually emits.

    The hosts are the constraint, not the choice. NF-BoT-IoT-v2 has 20 distinct
    source addresses across 37M flows and NF-UNSW-NB15 has 44 across 2.4M; no
    sampling scheme can manufacture host diversity the capture never had. What
    this guarantees is that whatever diversity exists survives into the sample.

    Implemented with groupby transforms rather than a per-host loop: corpora
    here reach 200k+ distinct hosts over 20M rows, and any loop that scans the
    row array once per host costs trillions of comparisons.
    """
    if entity_col not in df.columns:
        log.warning("no %s column; falling back to flow-level sampling", entity_col)
        return _stratified_subsample(df, df[label_col].astype(str), n, seed)

    rng = np.random.RandomState(seed)
    work = pd.DataFrame({
        "ent": df[entity_col].astype(str).values,
        "lab": df[label_col].astype(str).values,
        "pos": np.arange(len(df), dtype=np.int64),
    })

    sizes = work["ent"].value_counts()
    n_hosts_total = len(sizes)

    # Drop hosts only when the budget cannot give each one a usable share.
    if n // max(n_hosts_total, 1) < min_per_host:
        keep = sizes.index[: max(1, n // min_per_host)]
        work = work[work["ent"].isin(set(keep))]
        sizes = sizes.loc[keep]
    n_hosts = len(sizes)

    # Even per-host quota, with headroom left by small hosts redistributed.
    per = max(min_per_host, n // max(n_hosts, 1))
    quota = np.minimum(sizes.values, per)
    spare = n - int(quota.sum())
    if spare > 0:
        room = sizes.values - quota
        growable = room > 0
        if growable.any():
            extra = np.minimum(room, spare // max(growable.sum(), 1))
            quota = quota + np.where(growable, extra, 0)
    host_quota = pd.Series(quota, index=sizes.index)

    # Shuffle first so that the per-group rank below is a random selection.
    work = work.iloc[rng.permutation(len(work))]

    grp = work.groupby(["ent", "lab"], sort=False)
    grp_size = grp["pos"].transform("size").values
    host_size = work.groupby("ent", sort=False)["pos"].transform("size").values
    rank = grp.cumcount().values
    hq = work["ent"].map(host_quota).values

    # Split each host's quota across its classes in proportion, with a floor so
    # a rare class on that host is not rounded away entirely.
    share = np.ceil(hq * grp_size / np.maximum(host_size, 1))
    floor = np.minimum(grp_size, np.maximum(1, hq // 8))
    grp_quota = np.minimum(grp_size, np.maximum(share, floor))

    sel = work["pos"].values[rank < grp_quota]

    # Proportional rounding overshoots slightly; trim at random to the budget.
    if len(sel) > n:
        sel = rng.choice(sel, size=n, replace=False)

    out = df.iloc[np.sort(sel)].reset_index(drop=True)
    log.info("  entity sampling kept %d of %d source hosts (%.1f%%), %d rows, "
             "%.0f rows/host", n_hosts, n_hosts_total,
             n_hosts / n_hosts_total * 100, len(out), len(out) / max(n_hosts, 1))
    return out