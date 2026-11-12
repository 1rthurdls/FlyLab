"""Connectome queries: annotations, neuron search and pathway tracing (FlyWire v783).

Graph conventions
-----------------
* Nodes are the 138,639 neurons of the model (indices = rows of Completeness_783.csv).
* An edge pre->post carries the FlyWire synapse count and a sign (+1 excitatory,
  -1 inhibitory) from the neurotransmitter prediction, exactly the data the LIF
  model uses ('Excitatory x Connectivity' = sign x count).
* Pathway strength uses the *input fraction*: syn(pre->post) / total input synapses
  of post. A path's strength is the product of its edges' input fractions; the
  strongest paths are found with Dijkstra on cost = -log(input fraction).
  Weak edges (< MIN_SYN synapses, the usual FlyWire threshold of 5) are ignored.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
ANNOT_TSV = DATA / "Supplemental_file1_neuron_annotations.tsv"
ANNOT_PQ = DATA / "annotations_783.parquet"
MIN_SYN = 5

ANNOT_COLS = ["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type",
              "ito_lee_hemilineage", "top_nt", "side", "flow", "synonyms"]
SEARCH_COLS = ["cell_type", "hemibrain_type", "cell_class", "cell_sub_class", "super_class",
               "ito_lee_hemilineage", "top_nt", "synonyms"]


@lru_cache(maxsize=1)
def annotations() -> pd.DataFrame:
    """One row per model neuron, in model index order (column 'idx')."""
    if ANNOT_PQ.exists():
        return pd.read_parquet(ANNOT_PQ)
    comp = pd.read_csv(DATA / "Completeness_783.csv", index_col=0)
    a = pd.read_csv(ANNOT_TSV, sep="\t", dtype={"root_id": "int64"}, usecols=ANNOT_COLS, low_memory=False)
    a = a.drop_duplicates("root_id")
    df = pd.DataFrame({"root_id": comp.index.astype("int64"), "idx": np.arange(len(comp))})
    df = df.merge(a, on="root_id", how="left")
    for c in ANNOT_COLS[1:]:
        df[c] = df[c].astype("string").fillna("")
    df["label"] = np.where(df.cell_type != "", df.cell_type,
                           np.where(df.cell_sub_class != "", df.cell_sub_class, df.cell_class))
    df.loc[df.label == "", "label"] = "unannotated"
    df = df.sort_values("idx").reset_index(drop=True)
    df.to_parquet(ANNOT_PQ)
    return df


def describe(idx: int) -> dict:
    r = annotations().iloc[int(idx)]
    return {"root_id": int(r.root_id), "cell_type": r.label, "class": r.cell_class or r.super_class,
            "sub_class": r.cell_sub_class, "side": r.side, "nt": r.top_nt}


def short_label(idx: int) -> str:
    r = annotations().iloc[int(idx)]
    side = {"left": "L", "right": "R"}.get(r.side, "")
    return f"{r.label}{'_' + side if side else ''}"


def search_neurons(query: str, limit: int = 20) -> dict:
    """Case-insensitive search over cell type, class, sub-class, super-class, hemilineage,
    neurotransmitter and synonyms. Results are aggregated per cell type."""
    a = annotations()
    q = query.strip().lower()
    hit = np.zeros(len(a), bool)
    exact = np.zeros(len(a), bool)
    for c in SEARCH_COLS:
        col = a[c].str.lower()
        hit |= col.str.contains(q, regex=False).to_numpy()
        exact |= (col == q).to_numpy()
    sub = a[hit].copy()
    sub["exact"] = exact[hit]
    if sub.empty:
        return {"query": query, "n_neurons": 0, "matches": []}
    g = sub.groupby("label").agg(
        n=("root_id", "size"), exact=("exact", "max"), super_class=("super_class", "first"),
        cell_class=("cell_class", "first"), sub_class=("cell_sub_class", "first"),
        nt=("top_nt", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
        sides=("side", lambda s: ",".join(sorted(set(x for x in s if x)))),
        example_root_ids=("root_id", lambda s: [int(x) for x in s.head(3)]),
    ).reset_index().sort_values(["exact", "n"], ascending=[False, False])
    return {
        "query": query,
        "n_neurons": int(len(sub)),
        "n_cell_types": int(len(g)),
        "matches": [
            {"cell_type": r.label, "n": int(r.n), "super_class": r.super_class, "class": r.cell_class,
             "sub_class": r.sub_class, "nt": r.nt, "sides": r.sides, "example_root_ids": r.example_root_ids}
            for r in g.head(limit).itertuples()
        ],
    }


def neurons_of_type(cell_type: str) -> list[int]:
    a = annotations()
    return a.loc[a.label == cell_type, "root_id"].astype("int64").tolist()


# ---------------------------------------------------------------------------- graph
@lru_cache(maxsize=1)
def graph():
    """Returns (A_count csr pre x post, sign csr, in_total per post, cost csr)."""
    from .brain import PATH_CSR, PARAMS, Brain

    if not PATH_CSR.exists():
        Brain()  # builds the CSR cache
    z = np.load(PATH_CSR)
    indptr, post, w = z["indptr"], z["post"], z["w"]
    N = len(indptr) - 1
    signed = np.rint(w / PARAMS["wScale"]).astype(np.float32)  # sign x synapse count
    count = np.abs(signed)
    A = sp.csr_matrix((count, post, indptr), shape=(N, N))
    S = sp.csr_matrix((np.sign(signed), post, indptr), shape=(N, N))
    in_total = np.asarray(A.sum(axis=0)).ravel()
    # strongest-path cost on edges with >= MIN_SYN synapses
    keep = count >= MIN_SYN
    pre_of_edge = np.repeat(np.arange(N), np.diff(indptr))
    frac = count[keep] / np.maximum(in_total[post[keep]], 1)
    C = sp.csr_matrix((-np.log(frac) + 1e-9, (pre_of_edge[keep], post[keep])), shape=(N, N))
    return A, S, in_total, C


def top_inputs(idx: int, k: int = 10) -> list[dict]:
    A, S, in_total, _ = graph()
    col = A[:, idx].tocoo()
    order = np.argsort(-col.data)[:k]
    return [{"from": short_label(col.row[o]), "root_id": describe(col.row[o])["root_id"], "synapses": int(col.data[o]),
             "sign": "+" if S[col.row[o], idx] > 0 else "-", "input_fraction": round(float(col.data[o] / max(in_total[idx], 1)), 3)}
            for o in order]


def trace_pathway(source_idx: list[int], target_idx: list[int], max_hops: int = 4, k: int = 8,
                  restrict_to: np.ndarray | None = None) -> dict:
    """Strongest paths from any source neuron to each target neuron.

    restrict_to: optional boolean mask (N,) of allowed intermediate neurons
    (e.g. neurons active in a simulation -> 'functional' pathway)."""
    A, S, in_total, C = graph()
    N = C.shape[0]
    src = np.unique(np.asarray(source_idx, dtype=np.int64))
    tgt = np.unique(np.asarray(target_idx, dtype=np.int64))
    if restrict_to is not None:
        allowed = restrict_to.copy()
        allowed[src] = True
        allowed[tgt] = True
        D = sp.diags(allowed.astype(np.float32))
        C = (D @ C @ D).tocsr()
        C.eliminate_zeros()
    # super-source trick: one Dijkstra from a virtual node connected to all sources
    C2 = sp.vstack([C, sp.csr_matrix((np.full(len(src), 1e-9), (np.zeros(len(src), int), src)), shape=(1, N))])
    C2 = sp.hstack([C2, sp.csr_matrix((N + 1, 1))]).tocsr()
    dist, pred = dijkstra(C2, directed=True, indices=N, return_predecessors=True, limit=np.inf)
    paths = []

    def walk(n):
        nodes = []
        while n != N and n >= 0:
            nodes.append(int(n))
            n = pred[n]
        return nodes[::-1]

    # several distinct routes per target: best path to each strong direct input of the target
    for t in tgt:
        col = C[:, t].tocoo()
        cands = [(dist[p_] + c_, p_) for p_, c_ in zip(col.row, col.data) if np.isfinite(dist[p_]) and p_ != t]
        cands.sort()
        for d_, p_ in cands[:k]:
            nodes = walk(p_) + [int(t)]
            if len(set(nodes)) != len(nodes):
                continue
            hops = len(nodes) - 1
            edges, sign = [], 1
            for a_, b_ in zip(nodes[:-1], nodes[1:]):
                s_ = int(S[a_, b_])
                sign *= s_
                edges.append({"syn": int(A[a_, b_]), "sign": "+" if s_ > 0 else "-",
                              "input_fraction": round(float(A[a_, b_] / max(in_total[b_], 1)), 3)})
            paths.append({
                "target": short_label(t), "hops": hops, "within_max_hops": hops <= max_hops,
                "strength": float(np.exp(-d_)), "net_sign": "excitatory" if sign > 0 else "inhibitory",
                "path": [short_label(n) for n in nodes], "root_ids": [describe(n)["root_id"] for n in nodes],
                "edges": edges, "_idx": nodes,
            })
    paths = [p for p in paths if p["within_max_hops"]] or paths
    paths.sort(key=lambda p: -p["strength"])
    top = paths[0]["strength"] if paths else 1.0
    for p in paths:
        p["relative_strength"] = round(p["strength"] / top, 3)
        p["strength"] = float(f"{p['strength']:.3g}")
    # intermediate neurons ranked by how many/strong best-paths pass through them
    inter: dict[int, float] = {}
    for p in paths:
        for n in p["_idx"][1:-1]:
            inter[n] = inter.get(n, 0.0) + p["strength"]
    inter_sorted = sorted(inter.items(), key=lambda x: -x[1])
    wmax = inter_sorted[0][1] if inter_sorted else 1.0
    # layer analysis: how many neurons are reachable at each hop (strong edges only)
    Cb = (C > 0).astype(np.int8).tocsr()
    frontier = np.zeros(N, bool)
    frontier[src] = True
    seen = frontier.copy()
    layers = []
    for h in range(1, max_hops + 1):
        nxt = np.asarray(Cb.T @ frontier.astype(np.int8)).ravel() > 0
        nxt &= ~seen
        seen |= nxt
        layers.append({"hop": h, "n_new_neurons": int(nxt.sum()), "targets_reached": int(nxt[tgt].sum())})
        frontier = nxt
    for p in paths:
        p.pop("_idx")
    return {
        "n_sources": int(len(src)), "n_targets": int(len(tgt)),
        "n_targets_reached": len(paths),
        "min_synapses_per_edge": MIN_SYN,
        "n_paths_found": len(paths),
        "paths": paths[:k],
        "key_intermediates": [{"neuron": short_label(n), "root_id": describe(n)["root_id"], "path_weight": round(w_ / wmax, 3),
                               **{kk: vv for kk, vv in describe(n).items() if kk in ("class", "nt")}}
                              for n, w_ in inter_sorted[:10]],
        "_intermediate_idx": [n for n, _ in inter_sorted],
        "hop_layers": layers,
    }
