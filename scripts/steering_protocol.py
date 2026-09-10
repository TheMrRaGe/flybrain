#!/usr/bin/env python3
"""
steering_protocol.py - does the connectome actually steer?

    python3 steering_protocol.py --brain brain_mirrored.npz --trials 30

Every steering number reported before this script was read from a single 250 ms
window. At biological firing rates DNa02 - one neuron per side - fires a handful of
spikes in that time, so those readings were noise dressed as signal. This follows
the protocol Shiu et al. (2024) used instead: repeat trials, 1,000 ms each, and
compare conditions rather than trusting any single run.

DESIGN
  Two conditions, exact mirror images of each other:
      LEFT   left antenna 1.0, right antenna 0.4
      RIGHT  left antenna 0.4, right antenna 1.0
  Any consistent difference in the response is either real lateralisation or a
  residual asymmetry in the model. The shuffle control separates those.

  Readouts, from most specific to least:
      DNa02        the identified turning neuron, one cell per side
      DNa family   its broader class
      all DNs      the whole descending population

  Statistic: turn index = (right - left) / (right + left), per trial.
  Reported with a Mann-Whitney U test, as in the reference paper.

SHUFFLE CONTROL
  The connectome's weights are permuted across edges, preserving the global weight
  distribution but destroying which neuron connects to which. If the real network
  steers and the shuffled one does not, the behaviour came from the wiring. If both
  steer, it came from my parameters, and the result is worthless.
"""
from __future__ import annotations

import argparse, json, time
import numpy as np
from scipy.stats import mannwhitneyu

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def spike_counts(brain: FlyBrain, groups: dict[str, np.ndarray],
                 settle_ms: float, record_ms: float) -> dict[str, float]:
    """Step the brain and count spikes per group. Never materialises a full raster."""
    brain.reset()
    for _ in range(int(settle_ms / brain.p.dt)):
        brain.step()
    counts = {k: 0 for k in groups}
    for _ in range(int(record_ms / brain.p.dt)):
        spk = brain.step()
        for k, idx in groups.items():
            counts[k] += int(spk[idx].sum())
    return counts


def run_condition(brain: FlyBrain, groups, left: float, right: float,
                  trials: int, settle_ms: float, record_ms: float, seed0: int):
    out = {k: [] for k in groups}
    for t in range(trials):
        brain.rng = np.random.default_rng(seed0 + t)
        brain._noise_pool = (brain.rng.standard_normal(brain._noise_pool.size,
                                                       dtype=np.float32) * brain.p.noise
                             if brain._noise_pool is not None else None)
        brain._noise_off = 0
        brain.smell_bilateral({"food": left}, {"food": right})
        brain.set_drive("hunger", 0.6)
        c = spike_counts(brain, groups, settle_ms, record_ms)
        for k in groups:
            out[k].append(c[k])
    return out


def turn_index(R: list[int], L: list[int]) -> np.ndarray:
    R = np.asarray(R, float); L = np.asarray(L, float)
    tot = R + L
    return np.where(tot > 0, (R - L) / np.maximum(tot, 1e-9), 0.0)


def analyse(name, condL, condR, label):
    """condL/condR: dict side -> list of per-trial counts, for one readout."""
    tiL = turn_index(condL["R"], condL["L"])      # stimulus on the LEFT
    tiR = turn_index(condR["R"], condR["L"])      # stimulus on the RIGHT
    d = float(np.mean(tiR) - np.mean(tiL))
    pooled = np.sqrt((np.std(tiL, ddof=1) ** 2 + np.std(tiR, ddof=1) ** 2) / 2)
    dprime = d / pooled if pooled > 0 else 0.0
    try:
        u, p = mannwhitneyu(tiL, tiR, alternative="two-sided")
    except ValueError:
        p = 1.0
    spikes = np.mean(condL["L"]) + np.mean(condL["R"])
    log(f"  {label:<13}{name:<12}"
        f"stim-L {np.mean(tiL):+.4f}  stim-R {np.mean(tiR):+.4f}  "
        f"diff {d:+.4f}  d'={dprime:>5.2f}  p={p:.2g}  ({spikes:.0f} spikes/trial)")
    return {"readout": name, "network": label, "stim_left": float(np.mean(tiL)),
            "stim_right": float(np.mean(tiR)), "difference": d,
            "dprime": float(dprime), "p": float(p), "spikes_per_trial": float(spikes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_mirrored.npz")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--record-ms", type=float, default=1000.0)
    ap.add_argument("--settle-ms", type=float, default=200.0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--only-shuffle", action="store_true",
                    help="run the shuffle control alone (the two halves take "
                         "~190s each, which exceeds a single command budget)")
    ap.add_argument("--out", default="steering_result.json")
    a = ap.parse_args()

    results = []
    order = [True] if a.only_shuffle else ([False] if a.no_shuffle else [False, True])
    for shuffled in order:
        label = "SHUFFLED" if shuffled else "real"
        t0 = time.perf_counter()
        b = FlyBrain(a.brain, Params(gain=a.gain), seed=0)
        if shuffled:
            # permute weights across edges: same global weight distribution, but which
            # neuron drives which is destroyed. Both the event and dense paths are
            # permuted with the SAME permutation so they stay consistent.
            rng = np.random.default_rng(12345)
            perm = rng.permutation(b._out_w.size)
            b._out_w = b._out_w[perm].copy()
        dn = b.pop["DN"]; sd = b.side[dn]; ty = b.type[dn].astype(str)
        a02 = np.char.startswith(ty, "DNa02"); afam = np.char.startswith(ty, "DNa")
        readouts = {
            "DNa02": {"L": dn[a02 & (sd == "L")], "R": dn[a02 & (sd == "R")]},
            "DNa family": {"L": dn[afam & (sd == "L")], "R": dn[afam & (sd == "R")]},
            "all DNs": {"L": dn[sd == "L"], "R": dn[sd == "R"]},
        }
        groups = {f"{n}|{s}": idx for n, sides in readouts.items()
                  for s, idx in sides.items()}
        for k in ("food",):
            b.define_odor(k, n_channels=20, seed=777)

        log(f"\n=== {label} network  ({a.trials} trials x {a.record_ms:.0f} ms per condition)")
        cl = run_condition(b, groups, 1.0, 0.4, a.trials, a.settle_ms, a.record_ms, 1000)
        cr = run_condition(b, groups, 0.4, 1.0, a.trials, a.settle_ms, a.record_ms, 5000)
        for n in readouts:
            results.append(analyse(
                n, {"L": cl[f"{n}|L"], "R": cl[f"{n}|R"]},
                   {"L": cr[f"{n}|L"], "R": cr[f"{n}|R"]}, label))
        log(f"  ({time.perf_counter()-t0:.0f}s)")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
