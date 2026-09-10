#!/usr/bin/env python3
"""
odour_design.py - choose a CS+/CS- pair from the Kenyon-cell footprint of each glomerulus.

    python3 odour_design.py --brain ../brain_mirrored.npz --out ../results/odours.json

WHY THIS EXISTS
    Every conditioning protocol so far defined CS+ as receptor types [0:5] and CS- as
    [5:10] of the sorted list - alphabetical, not designed. Measured (conditioning4):

        X = [ORN_D, DA1, DA2, DA3, DA4l]     1,087 active KCs, 23,171 spikes
        Y = [DA4m, DC1, DC2, DC3, DC4]         842 active KCs, 16,225 spikes
        744 of Y's 842 cells are also in X  (88%)

    ORN_DA1 alone has 204 receptors - the cVA pheromone glomerulus, enormous in a
    male - so X drives 43% more Kenyon activity and Y is nearly a subset of it. Any
    depression, paired with EITHER odour, then removes more of Y's response than X's.
    That is the -0.44 contingency-independent shift the differential readout showed.

    Real experiments counterbalance odours to cancel this. Better to not have it:
    measure the KC footprint of every glomerulus, then pick two sets of k glomeruli
    that are matched in total drive and share as few Kenyon cells as possible.

WHAT IT MEASURES
    One presentation per receptor type, same seed, `--ms` of activity after settling:
    which KCs fire and how much. Then a greedy search over pairs of k-sets scoring

        score = jaccard(KC_A, KC_B) + lambda * |log(drive_A / drive_B)|

    Greedy is fine: 53 glomeruli, k=5, and the answer only needs to be good, not optimal.
"""
from __future__ import annotations

import argparse, itertools, json, os, time
import numpy as np

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def footprint(b, rtype, seed, settle_ms, ms):
    b.rng = np.random.default_rng(seed)
    if b._noise_pool is not None:
        b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                         * b.p.noise)
        b._noise_off = 0
    b.reset()
    b._odor_map["_probe"] = {rtype: 1.0}
    b.smell({"_probe": 1.0})
    for _ in range(int(settle_ms / b.p.dt)):
        b.step()
    tot = np.zeros(len(b._kc), dtype=np.float32)
    for _ in range(int(ms / b.p.dt)):
        tot += b.step()[b._kc]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--k", type=int, default=5, help="glomeruli per odour")
    ap.add_argument("--ms", type=float, default=600.0)
    ap.add_argument("--settle-ms", type=float, default=250.0)
    ap.add_argument("--kc-thresh", type=float, default=1.5)
    ap.add_argument("--apl-scale", type=float, default=0.1, help="gain on the real APL (decision 12)")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--balance", type=float, default=1.0,
                    help="weight on |log drive ratio| against jaccard overlap")
    ap.add_argument("--exclude-min-receptors", type=int, default=15,
                    help="skip glomeruli with fewer receptors than this")
    ap.add_argument("--exclude-max-receptors", type=int, default=120,
                    help="skip giant glomeruli (DA1 has 204: cVA, male-specific)")
    ap.add_argument("--out", default="../results/odours.json")
    a = ap.parse_args()

    b = FlyBrain(a.brain, Params(gain=1.0, kc_thresh_scale=a.kc_thresh, apl_scale=a.apl_scale,
                                 noise=a.noise), seed=11)
    types = list(b.receptor_types)
    nrec = {t: len(b._receptor_index[t]) for t in types}
    keep = [t for t in types
            if a.exclude_min_receptors <= nrec[t] <= a.exclude_max_receptors]
    log(f"{len(types)} receptor types, {len(keep)} within "
        f"[{a.exclude_min_receptors}, {a.exclude_max_receptors}] receptors")
    log(f"excluded: {[(t, nrec[t]) for t in types if t not in keep]}\n")

    t0 = time.time()
    fp = {}
    log(f"{'type':<10} {'recs':>5} {'KC spikes':>10} {'active KCs':>11}")
    for t in keep:
        tot = footprint(b, t, 1000, a.settle_ms, a.ms)
        fp[t] = tot
        log(f"{t:<10} {nrec[t]:>5} {int(tot.sum()):>10} {int((tot > 0).sum()):>11}")
    log(f"\nfootprints in {time.time()-t0:.0f}s")

    # single-glomerulus footprints add roughly; use the union/sum as the odour estimate
    active = {t: fp[t] > 0 for t in keep}
    drive = {t: float(fp[t].sum()) for t in keep}
    usable = [t for t in keep if drive[t] > 0]

    def score(A, B):
        ma = np.any([active[t] for t in A], axis=0)
        mb = np.any([active[t] for t in B], axis=0)
        jac = (ma & mb).sum() / max((ma | mb).sum(), 1)
        da, db = sum(drive[t] for t in A), sum(drive[t] for t in B)
        bal = abs(np.log(max(da, 1) / max(db, 1)))
        return jac + a.balance * bal, jac, da, db, int(ma.sum()), int(mb.sum()), int((ma & mb).sum())

    # greedy: seed with the least-overlapping pair of single glomeruli, then grow
    # each side alternately with the glomerulus that keeps the score lowest
    best = None
    for x, y in itertools.combinations(usable, 2):
        s = score([x], [y])
        if best is None or s[0] < best[0]:
            best, seed = s, ([x], [y])
    A, B = seed
    while len(A) < a.k or len(B) < a.k:
        side = A if len(A) <= len(B) else B
        other = B if side is A else A
        cand = None
        for t in usable:
            if t in A or t in B:
                continue
            trial = side + [t]
            s = score(trial, other) if side is A else score(other, trial)
            if cand is None or s[0] < cand[0]:
                cand, pick = s, t
        side.append(pick)
    s, jac, da, db, na, nb, nshared = score(A, B)

    log(f"\nCS+ = {A}")
    log(f"CS- = {B}")
    log(f"predicted: {na} vs {nb} active KCs, {nshared} shared, jaccard {jac:.3f}, "
        f"drive {da:.0f} vs {db:.0f} (ratio {da/max(db,1):.2f})")

    # verify by presenting the composed odours, not just summing footprints
    b._odor_map["CS+"] = {t: 1.0 for t in A}
    b._odor_map["CS-"] = {t: 1.0 for t in B}
    meas = {}
    for o in ("CS+", "CS-"):
        b.rng = np.random.default_rng(1000)
        if b._noise_pool is not None:
            b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                             * b.p.noise)
            b._noise_off = 0
        b.reset()
        b.smell({o: 1.0})
        for _ in range(int(a.settle_ms / b.p.dt)):
            b.step()
        tot = np.zeros(len(b._kc), dtype=np.float32)
        for _ in range(int(a.ms / b.p.dt)):
            tot += b.step()[b._kc]
        meas[o] = tot
    ma, mb = meas["CS+"] > 0, meas["CS-"] > 0
    log(f"measured:  {int(ma.sum())} vs {int(mb.sum())} active KCs, "
        f"{int((ma & mb).sum())} shared, jaccard {(ma & mb).sum()/max((ma|mb).sum(),1):.3f}, "
        f"spikes {int(meas['CS+'].sum())} vs {int(meas['CS-'].sum())} "
        f"(ratio {meas['CS+'].sum()/max(meas['CS-'].sum(),1):.2f})")
    log(f"KC sparsity: {100*ma.sum()/len(ma):.1f}% / {100*mb.sum()/len(mb):.1f}% "
        f"of {len(ma)} (biological target ~5%)")

    out = {"CS+": A, "CS-": B, "k": a.k, "kc_thresh": a.kc_thresh, "apl_scale": a.apl_scale,
           "noise": a.noise,
           "footprints": {t: {"receptors": nrec[t], "spikes": drive[t],
                              "active_kc": int(active[t].sum())} for t in keep},
           "predicted": {"active": [na, nb], "shared": nshared, "jaccard": jac,
                         "drive": [da, db]},
           "measured": {"active": [int(ma.sum()), int(mb.sum())],
                        "shared": int((ma & mb).sum()),
                        "jaccard": float((ma & mb).sum() / max((ma | mb).sum(), 1)),
                        "spikes": [float(meas["CS+"].sum()), float(meas["CS-"].sum())]}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
