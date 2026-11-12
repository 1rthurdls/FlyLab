#!/usr/bin/env bash
# FlyLab one-command setup (Linux, macOS, or Windows Git Bash).
#   bash scripts/setup.sh
# Creates .venv (Python 3.11), installs pinned deps, fetches the brain-model code + FlyWire data,
# builds caches (connectome CSR, annotations, neuron groups) and pre-compiles the numba kernel.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

if [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
else
  echo ">> creating .venv (Python 3.11)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv || uv venv --native-tls --python 3.11 .venv
  else
    python3.11 -m venv .venv || python -m venv .venv
  fi
  if [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
fi
echo ">> python: $PY ($($PY --version))"

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" "$@" || uv pip install --native-tls --python "$PY" "$@"
  else
    "$PY" -m pip install "$@"
  fi
}
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo ">> installing torch (CPU wheel; numba does the heavy lifting anyway)"
  pip_install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu || pip_install torch
fi
echo ">> installing requirements"
pip_install -r requirements.txt

echo ">> fetching brain-model code and FlyWire v783 data"
mkdir -p external data
[[ -d external/fly-brain ]] || git clone --depth 1 https://github.com/eonsystemspbc/fly-brain.git external/fly-brain
[[ -d external/Drosophila_brain_model ]] || git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git external/Drosophila_brain_model
[[ -f data/Completeness_783.csv ]] || cp external/Drosophila_brain_model/Completeness_783.csv data/Completeness_783.csv
[[ -f data/Connectivity_783.parquet ]] || cp external/Drosophila_brain_model/Connectivity_783.parquet data/Connectivity_783.parquet
[[ -f data/Supplemental_file1_neuron_annotations.tsv ]] || curl -sSL -o data/Supplemental_file1_neuron_annotations.tsv \
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv

# point .mcp.json at this platform's venv interpreter
if [[ "$PY" == ".venv/bin/python" ]]; then
  sed -i.bak 's#\.venv/Scripts/python\.exe#.venv/bin/python#' .mcp.json && rm -f .mcp.json.bak
fi

echo ">> building caches + compiling the LIF kernel (first time ~1 min)"
"$PY" - <<'EOF'
import time
t = time.time()
from flylab.brain import Brain, Stimulus
from flylab import connectome
from flylab.groups import load_groups
b = Brain(); connectome.annotations(); G = load_groups(rebuild=True)
r = b.run([Stimulus(b.indices(G["sugar_GRNs"]["root_ids"]), 200)], duration_s=0.05, n_trials=2)
print(f"   ok: {b.N} neurons, {b.n_synapse_pairs} connections, {len(G)} groups, {time.time()-t:.0f}s")
EOF

if [[ ! -f .claude/settings.json ]]; then
  echo ">> NOTE: to enable the dashboard's live tool feed (Claude Code hooks), run:"
  echo "     cp .claude/settings.flylab.json .claude/settings.json"
fi
echo ">> setup done. Next: $PY scripts/smoke_test.py"
