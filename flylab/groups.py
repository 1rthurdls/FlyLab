"""Named neuron groups, resolved from data (never hard-coded from memory).

Sources:
  * FlyWire v783 annotations: data/Supplemental_file1_neuron_annotations.tsv
    (Schlegel et al. 2024, github.com/flyconnectome/flywire_annotations)
  * Shiu et al. 2024 stimulus sets as shipped in code:
      - external/fly-brain/code/benchmark.py  (EXPERIMENTS['sugar'], ['p9'])
      - external/Drosophila_brain_model/example.ipynb (id_mn9)
The resolved lists are written to data/neuron_groups.json with their source.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
GROUPS_JSON = DATA / "neuron_groups.json"
ANNOT_TSV = DATA / "Supplemental_file1_neuron_annotations.tsv"
BENCH_PY = REPO / "external" / "fly-brain" / "code" / "benchmark.py"
EXAMPLE_NB = REPO / "external" / "Drosophila_brain_model" / "example.ipynb"

# Output groups the body interface reads (see interface.py)
OUTPUT_GROUPS = ["MN9", "P9", "DNa01_left", "DNa01_right", "DNa02_left", "DNa02_right", "MDN",
                 "giant_fiber", "proboscis_motor_neurons", "DNg12"]


def _bench_experiment_ids(key: str) -> list[int]:
    """Parse neu_exc of EXPERIMENTS[key] out of fly-brain/code/benchmark.py without importing it."""
    src = BENCH_PY.read_text(encoding="utf8")
    m = re.search(r"EXPERIMENTS\s*=\s*(\{.*?\n\})", src, re.S)
    tree = ast.parse(m.group(1), mode="eval")
    for k, v in zip(tree.body.keys, tree.body.values):
        if ast.literal_eval(k) == key:
            for kk, vv in zip(v.keys, v.values):
                if ast.literal_eval(kk) == "neu_exc":
                    return [int(x) for x in ast.literal_eval(vv)]
    raise KeyError(key)


def _notebook_mn9_id() -> int:
    nb = json.loads(EXAMPLE_NB.read_text(encoding="utf8"))
    for c in nb["cells"]:
        m = re.search(r"id_mn9\s*=\s*(\d+)", "".join(c["source"]))
        if m:
            return int(m.group(1))
    raise KeyError("id_mn9")


def build_groups() -> dict:
    a = pd.read_csv(ANNOT_TSV, sep="\t", dtype={"root_id": "int64"}, low_memory=False)
    comp = pd.read_csv(DATA / "Completeness_783.csv", index_col=0)
    in_model = set(comp.index.astype("int64"))
    a = a[a.root_id.isin(in_model)]
    G: dict[str, dict] = {}

    def add(name, ids, source, description, role):
        ids = sorted({int(i) for i in ids if int(i) in in_model})
        sub = a[a.root_id.isin(ids)]
        G[name] = {
            "root_ids": ids,
            "n": len(ids),
            "role": role,
            "description": description,
            "source": source,
            "annotation_summary": {
                "cell_type": sub.cell_type.fillna("NA").value_counts().to_dict(),
                "side": sub.side.fillna("NA").value_counts().to_dict(),
            },
        }

    # --- sensory ---------------------------------------------------------------
    sugar = _bench_experiment_ids("sugar")
    add("sugar_GRNs", sugar,
        "Shiu et al. 2024 sugar GRN stimulus set (v783 IDs) from external/fly-brain/code/benchmark.py EXPERIMENTS['sugar']",
        "Sugar-sensing gustatory receptor neurons (labellum, one side) used as the sugar stimulus in Shiu et al. 2024",
        "sensory")
    side = a[a.root_id.isin(sugar)].side.mode().iloc[0]
    add("bitter_GRNs", a[(a.cell_sub_class == "bitter") & (a.side == side)].root_id,
        f"FlyWire annotations: cell_sub_class == 'bitter', side == '{side}' (same side as sugar_GRNs)",
        "Bitter-sensing gustatory receptor neurons", "sensory")
    add("water_GRNs", a[(a.cell_sub_class == "water") & (a.side == side)].root_id,
        f"FlyWire annotations: cell_sub_class == 'water', side == '{side}'",
        "Water-sensing gustatory receptor neurons", "sensory")
    add("sugar_GRNs_flywire_all", a[a.cell_sub_class.isin(["sugar", "sugar/low_salt"])].root_id,
        "FlyWire annotations: cell_sub_class in {'sugar','sugar/low_salt'} (both sides)",
        "All neurons annotated as sugar GRNs in FlyWire (alternative, larger stimulus set)", "sensory")
    add("JO_grooming", a[a.cell_sub_class == "grooming"].root_id,
        "FlyWire annotations: cell_class == 'mechanosensory', cell_sub_class == 'grooming' (Johnston's organ JO-F types)",
        "Antennal mechanosensory (Johnston's organ) neurons annotated as grooming-related", "sensory")
    add("JO_CE", a[a.cell_type.fillna("").str.match(r"^JO-(C|E)")].root_id,
        "FlyWire annotations: cell_type starting with 'JO-C' or 'JO-E'",
        "Johnston's organ C/E (wind/gravity) mechanosensory neurons", "sensory")

    # --- motor / descending outputs -----------------------------------------------
    mn9_id = _notebook_mn9_id()
    mn9_type = a.loc[a.root_id == mn9_id, "cell_type"].iloc[0]
    add("MN9", a[a.cell_type == mn9_type].root_id,
        f"Shiu et al. example.ipynb id_mn9={mn9_id} -> FlyWire cell_type '{mn9_type}'; group = all '{mn9_type}' neurons",
        "MN9 proboscis motor neurons (drive proboscis extension)", "output")
    p9 = _bench_experiment_ids("p9")
    p9_type = a.loc[a.root_id.isin(p9), "cell_type"].mode().iloc[0]
    add("P9", a[a.cell_type == p9_type].root_id,
        f"external/fly-brain/code/benchmark.py EXPERIMENTS['p9'] ('P9s forward walking') -> FlyWire cell_type '{p9_type}'",
        "P9 descending neurons (forward walking)", "output")
    for t in ["DNa01", "DNa02"]:
        for s in ["left", "right"]:
            add(f"{t}_{s}", a[(a.cell_type == t) & (a.side == s)].root_id,
                f"FlyWire annotations: cell_type == '{t}', side == '{s}'",
                f"{t} descending neuron, {s} side (steering)", "output")
    add("MDN", a[a.cell_type == "MDN"].root_id, "FlyWire annotations: cell_type == 'MDN'",
        "Moonwalker descending neurons (backward walking)", "output")
    add("proboscis_motor_neurons", a[a.cell_sub_class == "proboscis_motor_neuron"].root_id,
        "FlyWire annotations: cell_sub_class == 'proboscis_motor_neuron'",
        "All annotated proboscis motor neurons", "output")
    add("DNg12", a[a.cell_type.fillna("").str.startswith("DNg12")].root_id,
        "FlyWire annotations: cell_type starting with 'DNg12'",
        "DNg12 descending neurons (reported by FlyWire; monitored as a candidate grooming output, unverified)",
        "output")
    # --- visual looming detectors + escape command neuron ----------------------------
    add("LPLC2", a[a.cell_type == "LPLC2"].root_id, "FlyWire annotations: cell_type == 'LPLC2'",
        "LPLC2 visual projection neurons (looming-sensitive)", "sensory")
    add("LC4", a[a.cell_type == "LC4"].root_id, "FlyWire annotations: cell_type == 'LC4'",
        "LC4 visual projection neurons (looming-sensitive)", "sensory")
    add("giant_fiber", a[a.cell_type == "DNp01"].root_id,
        "FlyWire annotations: cell_type == 'DNp01' (synonym: Giant Fiber)",
        "Giant fiber descending neurons (escape take-off command)", "output")
    return G


def load_groups(rebuild: bool = False) -> dict:
    if rebuild or not GROUPS_JSON.exists():
        G = build_groups()
        GROUPS_JSON.write_text(json.dumps(G, indent=1), encoding="utf8")
        return G
    return json.loads(GROUPS_JSON.read_text(encoding="utf8"))


if __name__ == "__main__":
    G = load_groups(rebuild=True)
    for k, v in G.items():
        print(f"{k:26s} n={v['n']:4d}  {v['annotation_summary']['cell_type']}")
