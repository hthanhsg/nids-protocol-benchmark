"""E-GraphSAGE: edge-level classification over a flow graph.

Nodes are hosts (IP addresses); edges are flows. Flow statistics are *edge*
features, so intrusion detection becomes edge classification -- the formulation
that makes GNNs native to flow data.

Implemented in plain PyTorch with scatter-add aggregation rather than
torch_geometric or DGL, so the benchmark has no extra dependency and the message
passing is auditable.

Two things differ from the tabular models and must be reported as such:

1. **Message passing uses training edges only.** Node embeddings are computed
   from the training subgraph; validation and test edges are scored against
   those embeddings but never contribute to them. Letting test edges propagate
   is a form of leakage that the transductive setups common in the literature
   do not always avoid.

2. **`identity` means something different here.** A GNN needs IP addresses to
   build the graph at all, so `drop` removes IP and port from the *edge feature
   vector* while the graph topology still encodes host identity. The runner
   records this as `identity_semantics="graph_topology_retains_identity"` so the
   cell is never compared naively against the tabular models.
"""
from __future__ import annotations
import logging
import numpy as np
from sklearn.metrics import f1_score

log = logging.getLogger("nidsbench.graph")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def build_node_index(src, dst):
    """Map host addresses to contiguous node ids over the whole dataset."""
    src = np.asarray(src).astype(str)
    dst = np.asarray(dst).astype(str)
    vocab = {h: i for i, h in enumerate(np.unique(np.concatenate([src, dst])))}
    return (np.array([vocab[h] for h in src], dtype=np.int64),
            np.array([vocab[h] for h in dst], dtype=np.int64),
            len(vocab))


class EGraphSAGELayer(nn.Module if HAS_TORCH else object):
    """One round of message passing.

    A node collects [neighbour embedding || edge features] from every incident
    edge, averages them, and updates. Edges are treated as undirected for
    aggregation so that a victim host also learns from flows arriving at it.
    """

    def __init__(self, d_node, d_edge, d_out, dropout=0.2):
        super().__init__()
        self.msg = nn.Linear(d_node + d_edge, d_out)
        self.upd = nn.Linear(d_node + d_out, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_src, edge_dst, edge_attr, n_nodes):
        both_src = torch.cat([edge_src, edge_dst])
        both_dst = torch.cat([edge_dst, edge_src])
        attr = torch.cat([edge_attr, edge_attr], dim=0)

        m = torch.relu(self.msg(torch.cat([h[both_src], attr], dim=1)))

        agg = torch.zeros(n_nodes, m.size(1), device=h.device, dtype=m.dtype)
        agg.index_add_(0, both_dst, m)
        deg = torch.zeros(n_nodes, device=h.device, dtype=m.dtype)
        deg.index_add_(0, both_dst, torch.ones_like(both_dst, dtype=m.dtype))
        agg = agg / deg.clamp(min=1).unsqueeze(1)

        return self.drop(torch.relu(self.norm(self.upd(torch.cat([h, agg], dim=1)))))


class EGraphSAGE(nn.Module if HAS_TORCH else object):
    def __init__(self, d_edge, n_classes, d_hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.d_hidden = d_hidden
        self.layers = nn.ModuleList()
        d_in = d_hidden
        for _ in range(layers):
            self.layers.append(EGraphSAGELayer(d_in, d_edge, d_hidden, dropout))
            d_in = d_hidden
        # Edge score from both endpoint embeddings plus the edge's own features.
        self.head = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes))

    def embed(self, n_nodes, edge_src, edge_dst, edge_attr, device):
        # E-GraphSAGE initialises node features as constants: all host
        # information must arrive through edges, never from node identity.
        h = torch.ones(n_nodes, self.d_hidden, device=device)
        for layer in self.layers:
            h = layer(h, edge_src, edge_dst, edge_attr, n_nodes)
        return h

    def score(self, h, edge_src, edge_dst, edge_attr):
        return self.head(torch.cat([h[edge_src], h[edge_dst], edge_attr], dim=1))


def _batched_score(model, h, src, dst, attr, batch=200_000):
    outs = []
    with torch.no_grad():
        for i in range(0, len(src), batch):
            sl = slice(i, i + batch)
            outs.append(model.score(h, src[sl], dst[sl], attr[sl]).argmax(1).cpu().numpy())
    return np.concatenate(outs) if outs else np.array([], dtype=int)


def fit_predict(node_src, node_dst, n_nodes, X, y, tr, va, te, n_classes,
                seed=42, cfg=None):
    """Train E-GraphSAGE on the training subgraph and score val/test edges.

    Returns (test predictions, val predictions, info).
    """
    if not HAS_TORCH:
        raise RuntimeError("torch not installed")
    cfg = {"lr": 1e-3, "weight_decay": 1e-4, "max_epochs": 100, "patience": 10,
           "min_epochs": 5, "d_hidden": 64, "layers": 2, "edge_batch": 100_000,
           **(cfg or {})}

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attr = torch.tensor(X, dtype=torch.float32, device=dev)
    src = torch.tensor(node_src, dtype=torch.long, device=dev)
    dst = torch.tensor(node_dst, dtype=torch.long, device=dev)

    tr_t = torch.tensor(tr, dtype=torch.long, device=dev)
    va_t = torch.tensor(va, dtype=torch.long, device=dev)
    te_t = torch.tensor(te, dtype=torch.long, device=dev)
    ytr = torch.tensor(y[tr], dtype=torch.long, device=dev)

    model = EGraphSAGE(X.shape[1], n_classes, cfg["d_hidden"], cfg["layers"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])

    counts = np.bincount(y[tr], minlength=n_classes).astype(np.float64)
    w = np.clip(np.where(counts > 0, len(tr) / (n_classes * np.maximum(counts, 1)), 0.0), 0, 50)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))

    # The message-passing graph is the training edge set. Test edges are scored
    # against it but never propagate through it.
    g_src, g_dst, g_attr = src[tr_t], dst[tr_t], attr[tr_t]

    best, best_state, bad, best_epoch = -np.inf, None, 0, -1
    history = []
    n_tr = len(tr)
    bs = min(cfg["edge_batch"], n_tr)

    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        tot = 0.0
        for i in range(0, n_tr, bs):
            sel = perm[i:i + bs]
            opt.zero_grad()
            h = model.embed(n_nodes, g_src, g_dst, g_attr, dev)
            logits = model.score(h, g_src[sel], g_dst[sel], g_attr[sel])
            loss = crit(logits, ytr[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(sel)

        model.eval()
        with torch.no_grad():
            h = model.embed(n_nodes, g_src, g_dst, g_attr, dev)
        pv = _batched_score(model, h, src[va_t], dst[va_t], attr[va_t])
        present = np.unique(y[va])
        vf1 = f1_score(y[va], pv, average="macro", labels=present, zero_division=0)
        history.append({"epoch": epoch, "train_loss": tot / n_tr, "val_macro_f1": float(vf1)})

        if epoch < cfg["min_epochs"] or vf1 > best + 1e-4:
            best, bad, best_epoch = vf1, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log.info("  egraphsage early-stopped at epoch %d (best %d, val macro-F1 %.4f)",
                         epoch, best_epoch, best)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(dev).eval()

    with torch.no_grad():
        h = model.embed(n_nodes, g_src, g_dst, g_attr, dev)
    pred_te = _batched_score(model, h, src[te_t], dst[te_t], attr[te_t])
    pred_va = _batched_score(model, h, src[va_t], dst[va_t], attr[va_t])

    # Test nodes absent from the training graph keep their constant initial
    # embedding. Under disjoint-IP splits that is most of them, and the count
    # explains any drop far better than the metric alone does.
    seen = torch.zeros(n_nodes, dtype=torch.bool, device=dev)
    seen[g_src] = True
    seen[g_dst] = True
    unseen = int((~(seen[src[te_t]] & seen[dst[te_t]])).sum())

    info = {"best_epoch": best_epoch, "epochs_run": len(history),
            "best_val_macro_f1": float(best), "n_nodes": int(n_nodes),
            "val_uninformative": bool(best_epoch <= cfg["min_epochs"]
                                      and len(history) > cfg["min_epochs"] + 1),
            "test_edges_with_unseen_node": unseen,
            "test_unseen_node_share": round(unseen / max(len(te), 1), 4),
            "_model": model}
    return pred_te, pred_va, info