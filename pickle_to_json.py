import os, json, math, pickle
from typing import Any, Dict, List
import numpy as np

IN_PATH = "outputs/plot_accuracy_vs_gap_Depolar.pickle"
OUT_PATH = "outputs/plot_accuracy_vs_gap_Depolar.json"

def normal_ci95_for_bernoulli_mean(samples: List[float]):
    if not samples:
        return (0.0, 0.0)
    p = float(np.mean(samples))
    n = len(samples)
    ci = 1.96 * math.sqrt(max(p*(1-p), 0.0) / max(n,1))
    return (max(0.0, p-ci), min(1.0, p+ci))

def guess_noise_from_filename(path: str) -> str:
    base = os.path.basename(path).replace(".pickle","")
    parts = base.split("_")
    return parts[-1] if parts else ""

def convert_results(results: Dict[str, Any], noise_model: str) -> Dict[str, Any]:
    out = {"file": IN_PATH, "derived": "plot_accuracy_vs_gap",
           "noise_model": noise_model, "algorithms": []}
    for algo, val in results.items():
        xs, per_x_samples = val  # xs = gap_list, per_x_samples = [[acc,..] per gap]
        x_key = "gap" if xs and not isinstance(xs[0], (int, np.integer)) else "x"
        points = []
        for x, accs in zip(xs, per_x_samples):
            accs = [float(a) for a in accs]
            mean = float(np.mean(accs)) if accs else 0.0
            lo, hi = normal_ci95_for_bernoulli_mean(accs)
            points.append({x_key: float(x) if isinstance(x,(float,np.floating)) else x,
                           "accs": accs, "mean": mean, "ci95": [lo, hi]})
        out["algorithms"].append({"name": algo, "points": points})
    return out

with open(IN_PATH,"rb") as f:
    data = pickle.load(f)
if not isinstance(data, dict):
    raise TypeError(f"Unexpected top-level type: {type(data)}")
noise = guess_noise_from_filename(IN_PATH)
json_obj = convert_results(data, noise)
with open(OUT_PATH,"w",encoding="utf-8") as f:
    json.dump(json_obj, f, ensure_ascii=False, indent=2)
print(f"Wrote: {OUT_PATH}")
