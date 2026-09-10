#!/usr/bin/env python3
"""
kc_sparsity.py - find the Kenyon-cell regime where the code is sparse AND selective.

    python3 kc_sparsity.py --odours ../results/odours.json

WHY
    odour_design.py measured that at kc_thresh 1.5 / apl 200 EVERY single glomerulus
    fires 600-790 Kenyon cells (15-19% of 4,064). One glomerulus is enough to fire a
    KC, so there is no coincidence detection and the code for any odour is the same
    ~700 easily-excited cells plus a fringe. Two designed, drive-matched 5-glomerulus
    odours still shared 852 of ~1,140 cells (Jaccard 0.60; independent 27% sets would
    give 0.16).

    The sparse regime tried earlier (3.0 / 3000) left 33 cells. Somewhere between is
    the regime the mushroom body is built for: ~5% of KCs active, each needing several
    coincident glomeruli, so different odours recruit different cells.

WHAT IT MEASURES
    For each (kc_thresh, apl) on the grid, present CS+ and CS- (same seed) and report
    active fraction, spikes, and their Jaccard overlap. Also the single-glomerulus
    footprint of one glomerulus, so you can see when a lone input stops being enough.
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def present(b, odour, seed, settle_ms, ms):
    b.rng = np.random.default_rng(seed)
    if b._noise_pool is not None:
        b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                         * b.p.noise)
        b._noise_off = 0
    b.reset()
    b.smell({odour: 1.0})
    for _ in range(int(settle_ms / b.p.dt)):
        b.step()
    tot = np.zeros(len(b._kc), dtype=np.float32)
    for _ in range(int(ms / b.p.dt)):
        tot += b.step()[b._kc]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--odours", default="../results/odours.json")
    ap.add_argument("--thresh", default="1.5,2.0,2.5")
    ap.add_argument("--apl", default="0.02,0.05,0.1,0.2,0.5,1.0",
                    help="apl_scale values: gain on the real APL neuron (decision 12)")
    ap.add_argument("--kk", default="1.0,0.0",
                    help="kc_kc_scale values; 0 ablates KC->KC recurrence (diagnostic)")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--ms", type=float, default=600.0)
    ap.add_argument("--settle-ms", type=float, default=250.0)
    ap.add_argument("--out", default="../results/kc_sparsity.json")
    a = ap.parse_args()

    od = json.load(open(a.odours))
    A, B = od["CS+"], od["CS-"]
    single = A[0]
    threshs = [float(x) for x in a.thresh.split(",")]
    apls = [float(x) for x in a.apl.split(",")]
    kks = [float(x) for x in a.kk.split(",")]

    log(f"CS+ = {A}\nCS- = {B}\nsingle-glomerulus probe: {single}\n")
    log(f"{'thresh':>6} {'aplsc':>6} {'kk':>4} | {'CS+ act':>8} {'CS- act':>8} {'%KC':>5} | "
        f"{'shared':>6} {'jaccard':>7} {'indep':>6} | {'CS+ spk':>8} {'CS- spk':>8} | {'1-glom act':>10}")
    rows = []
    for kk in kks:
      for th in threshs:
        for apl in apls:
            t0 = time.time()
            b = FlyBrain(a.brain, Params(gain=1.0, kc_thresh_scale=th, apl_scale=apl,
                                         kc_kc_scale=kk, noise=a.noise), seed=11)
            b._odor_map["CS+"] = {t: 1.0 for t in A}
            b._odor_map["CS-"] = {t: 1.0 for t in B}
            b._odor_map["one"] = {single: 1.0}
            p = present(b, "CS+", 1000, a.settle_ms, a.ms)
            m = present(b, "CS-", 1000, a.settle_ms, a.ms)
            g = present(b, "one", 1000, a.settle_ms, a.ms)
            ap_, am = p > 0, m > 0
            union = (ap_ | am).sum()
            jac = float((ap_ & am).sum() / union) if union else 0.0
            pct = 100 * (ap_.sum() + am.sum()) / 2 / len(p)
            f = pct / 100
            indep = f / (2 - f) if f else 0.0   # jaccard of two independent sets this size
            row = {"thresh": th, "apl_scale": apl, "kc_kc_scale": kk,
                   "active": [int(ap_.sum()), int(am.sum())],
                   "pct": pct, "shared": int((ap_ & am).sum()), "jaccard": jac,
                   "jaccard_independent": indep,
                   "spikes": [float(p.sum()), float(m.sum())],
                   "single_active": int((g > 0).sum())}
            rows.append(row)
            log(f"{th:>6.1f} {apl:>6.2f} {kk:>4.1f} | {ap_.sum():>8d} {am.sum():>8d} {pct:>5.1f} | "
                f"{(ap_ & am).sum():>6d} {jac:>7.3f} {indep:>6.3f} | {p.sum():>8.0f} {m.sum():>8.0f} | "
                f"{(g > 0).sum():>10d}   ({time.time()-t0:.0f}s)")
            del b

    # what would independent sets of the same size give?
    log("\nreference: two INDEPENDENT sets covering fraction f each have jaccard "
        "f/(2-f): f=0.27 -> 0.157, f=0.10 -> 0.053, f=0.05 -> 0.026")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"CS+": A, "CS-": B, "single": single, "rows": rows}, f, indent=2)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
