"""Train/validation/test splitting strategies.

All three return index arrays for a 70/15/15 partition. The validation split
exists so that early stopping never touches the test set: this matters here
because the whole point of the study is that evaluation protocol drives results.
"""
from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger("nidsbench.splits")

RATIOS = (0.70, 0.15, 0.15)


class SplitUnavailable(Exception):
    """Raised when a split cannot be built for this dataset (e.g. no timestamp)."""


def random_split(n, y=None, seed=42, ratios=RATIOS):
    """Stratified random split -- the leakage-prone baseline used by ~70% of the corpus."""
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    if y is None:
        rng.shuffle(idx)
        return _cut(idx, ratios)

    tr, va, te = [], [], []
    for cls in np.unique(y):
        c = idx[y == cls]
        rng.shuffle(c)
        a, b, d = _cut(c, ratios)
        tr.append(a); va.append(b); te.append(d)
    tr, va, te = np.concatenate(tr), np.concatenate(va), np.concatenate(te)
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


def temporal_split(ts, ratios=RATIOS):
    """Past -> future. Train on the earliest 70%, validate on the next 15%, test on the last 15%."""
    ts = np.asarray(ts)
    if ts is None or len(ts) == 0:
        raise SplitUnavailable("no timestamp column")
    finite = np.isfinite(ts.astype(float)) if ts.dtype.kind in "if" else np.array([True] * len(ts))
    if finite.sum() < len(ts) * 0.5:
        raise SplitUnavailable("timestamp column mostly missing")
    if len(np.unique(ts)) < 10:
        raise SplitUnavailable(f"timestamp has only {len(np.unique(ts))} distinct values")
    order = np.argsort(ts, kind="mergesort")
    return _cut(order, ratios)


def disjoint_entity_split(entity, y=None, seed=42, ratios=RATIOS, max_tries=40,
                          min_entities=3, min_share=0.05):
    """Partition by entity (source IP) so no entity appears in more than one split.

    Row-count balance is not enough. On NF-BoT-IoT-v2, twenty source hosts of
    wildly unequal size can be balanced to roughly 70/15/15 by rows while the
    test split holds a *single* host: the resulting score then measures how well
    one host's traffic is classified, and validation drawn from sixteen other
    hosts carries an entirely different class mix. That configuration produces
    numbers that look fine and mean nothing.

    Two guards therefore apply. Every split must receive at least
    ``min_entities`` distinct entities and at least ``min_share`` of the rows,
    and at least two classes. When no partition satisfies them the split is
    declared unavailable rather than returned: a corpus with too few entities
    cannot support entity-disjoint evaluation, and saying so is more useful than
    reporting a number derived from one host.
    """
    entity = np.asarray(entity).astype(str)
    uniq, inv, counts = np.unique(entity, return_inverse=True, return_counts=True)
    n_ent = len(uniq)
    if n_ent < 3 * min_entities:
        raise SplitUnavailable(
            f"only {n_ent} distinct entities; need >= {3*min_entities} to give each "
            f"split {min_entities}")

    n = len(entity)
    targets = np.array(ratios) * n
    best = None

    for t in range(max_tries):
        rng = np.random.RandomState(seed + t)
        assign = np.full(n_ent, -1)
        filled = np.zeros(3)
        n_assigned = np.zeros(3, dtype=int)

        # Seed each split with one entity so none can be starved outright, then
        # fill greedily by row deficit. Later attempts perturb the order.
        order = np.argsort(-counts)
        if t > 0:
            noise = rng.uniform(0.85, 1.15, n_ent)
            order = np.argsort(-(counts * noise))

        for s, e in enumerate(order[:3]):
            assign[e] = s
            filled[s] += counts[e]
            n_assigned[s] += 1

        for e in order[3:]:
            deficit = targets - filled
            # Force-feed any split still short of its entity floor.
            starved = n_assigned < min_entities
            if starved.any():
                s = int(np.argmax(np.where(starved, deficit, -np.inf)))
            else:
                s = int(np.argmax(deficit + rng.uniform(0, 1e-6, 3)))
            assign[e] = s
            filled[s] += counts[e]
            n_assigned[s] += 1

        splits = [np.where(assign[inv] == s)[0] for s in range(3)]

        if (n_assigned < min_entities).any():
            continue
        if min(len(s) for s in splits) < min_share * n:
            continue
        if y is not None and min(len(np.unique(y[s])) for s in splits) < 2:
            continue

        dev = max(abs(len(s) / n - r) for s, r in zip(splits, ratios))
        if best is None or dev < best[0]:
            best = (dev, splits, n_assigned.copy())
        if dev < 0.05:
            break

    if best is None:
        raise SplitUnavailable(
            f"no entity-disjoint partition over {n_ent} entities gives every split "
            f">= {min_entities} entities, >= {min_share:.0%} of rows and >= 2 classes")

    dev, splits, n_assigned = best
    log.info("  disjoint split: entities %s, rows %s, max deviation %.1f pp",
             list(n_assigned), [len(s) for s in splits], dev * 100)
    if dev > 0.10:
        log.warning("disjoint split deviates %.1f pp from target ratios", dev * 100)
    return splits[0], splits[1], splits[2]


def _cut(idx, ratios):
    n = len(idx)
    a = int(round(n * ratios[0]))
    b = a + int(round(n * ratios[1]))
    return idx[:a], idx[a:b], idx[b:]


def make_split(kind, ds, y, seed=42):
    """Dispatch to the right splitter. Raises SplitUnavailable if impossible."""
    n = len(ds.df)
    if kind == "random":
        return random_split(n, y=y, seed=seed)
    if kind == "temporal":
        if ds.time_col is None:
            raise SplitUnavailable("dataset has no timestamp column")
        return temporal_split(ds.df[ds.time_col].values)
    if kind == "disjoint_ip":
        src = "IPV4_SRC_ADDR"
        if src not in ds.df.columns:
            raise SplitUnavailable("no IPV4_SRC_ADDR column")
        return disjoint_entity_split(ds.df[src].values, y=y, seed=seed)
    raise ValueError(f"unknown split kind {kind}")