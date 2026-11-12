"""Exploratory runs used to choose behaviours, thresholds and the lesion target (log -> runs/explore.log)."""
import json, sys, time
sys.path.insert(0, '.')
from flylab import lab

def show(r):
    og = {g: v['mean_rate_hz'] for g, v in r['output_groups'].items() if v['mean_rate_hz'] > 0}
    print(f"{r['label']:45s} wall={r['sim']['wall_s']}s active={r['sim']['n_active_neurons']} outputs={og}", flush=True)
    print("   top:", [(t['neuron'], t['rate_hz']) for t in r['top_responders'][:8]], flush=True)

for act, rate in [(['sugar_GRNs'], 200), (['sugar_GRNs', 'bitter_GRNs'], 200), (['bitter_GRNs'], 200),
                  (['water_GRNs'], 200), (['P9'], 100), (['DNa02_left'], 100), (['DNa01_left', 'DNa02_left'], 100),
                  (['MDN'], 100), (['JO_grooming'], 200), (['JO_CE'], 200)]:
    show(lab.run_simulation(act, stim_rate_hz=rate, duration_s=1.0, n_trials=8))
for rate in [50, 100, 150]:
    show(lab.run_simulation(['sugar_GRNs'], stim_rate_hz=rate, duration_s=1.0, n_trials=8))
t = time.time()
ls = lab.lesion_screen(['sugar_GRNs'], 'MN9', n_candidates=14, duration_s=1.0, n_trials=8)
print('lesion screen', round(time.time() - t), 's', json.dumps(ls, indent=1), flush=True)
