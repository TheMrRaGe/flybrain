#!/usr/bin/env python3
"""
learned_dn.py - does the learned MBON change reach ANY descending neuron?

    python3 learned_dn.py --repeats 8 --trials 12

learned_steering.py read one family (DNa) under a lateralised odour and found
nothing. That presupposes which DNs carry valence. This asks the population: for
every one of the 1,310 DNs, how does its (CS+ - CS-) response change with training,
and does that change REVERSE when the contingency does?

    S_i  = r_i(CS+) - r_i(CS-)         per DN, per seed, bilateral odour
    dS_i = S_i(post) - S_i(pre)         paired seed, so the control is exactly 0

Readouts
    corr(dS_both, dS_reversed) over DNs  - negative if learning reaches the bus
    number of DNs whose dS flips sign between arms and exceeds noise
    the DN types that carry it, with their sign in each arm
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np

from conditioning4 import build, reseed, quiet, compartments, train_block, order


def log(m): print(m, flush=True)


def present_all(b, odour, seed, settle_ms, ms, idx, strength=1.0):
    reseed(b, seed)
    b.reset()
    b.smell({odour: strength})
    quiet(b)
    for _ in range(int(settle_ms / b.p.dt)):
        b.step()
    tot = np.zeros(len(idx), dtype=np.float32)
    for _ in range(int(ms / b.p.dt)):
        tot += b.step()[idx]
    return tot


def measure(b, seeds, a, idx):
    was, b.plastic_on = b.plastic_on, False
    S = np.zeros((len(seeds), len(idx)), dtype=np.float32)
    for k, s in enumerate(seeds):
        p = present_all(b, "CS+", s, a.settle_ms, a.test_ms, idx, a.strength)
        m = present_all(b, "CS-", s, a.settle_ms, a.test_ms, idx, a.strength)
        S[k] = p - m
    b.plastic_on = was
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--aversive", default="PPL105")
    ap.add_argument("--appetitive", default="PAM08")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--train-ms", type=float, default=800.0)
    ap.add_argument("--test-ms", type=float, default=800.0)
    ap.add_argument("--settle-ms", type=float, default=250.0)
    ap.add_argument("--punish-hz", type=float, default=180.0)
    ap.add_argument("--punish-mv", type=float, default=70.0)
    ap.add_argument("--rate", type=float, default=0.0003)
    ap.add_argument("--core", type=float, default=0.2)
    ap.add_argument("--bg-hold", type=float, default=0.0, help="candidate decision 15")
    ap.add_argument("--mbon-hold", type=float, default=0.85)
    ap.add_argument("--kc-thresh", type=float, default=1.5)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--apl-scale", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--odours", default="../results/odours3.json")
    ap.add_argument("--channels", type=int, default=5)
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--cs-minus-first", action="store_true")
    ap.add_argument("--out", default="../results/learned_dn.json")
    a = ap.parse_args()
    AV, AP = a.aversive, a.appetitive
    arms = {"both": {"CS+": AV, "CS-": AP}, "reversed": {"CS+": AP, "CS-": AV}}
    seeds = [1000 + 97 * i for i in range(a.repeats)]

    res = {}
    for name, sched in arms.items():
        t0 = time.time()
        b = build(a.brain, a)
        dn = b.pop["DN"]; mbon = b.pop["MBON"]
        idx = np.concatenate([dn, mbon])
        ty = b.type.astype(str)[idx]; side = b.side[idx]
        pre = measure(b, seeds, a, idx)
        train_block(b, a.trials, a.train_ms, True, sched, a.punish_hz, a.punish_mv, order(a),
                    a.strength)
        post = measure(b, seeds, a, idx)
        dS = post - pre                                  # seeds x cells
        res[name] = {"dS_mean": dS.mean(0), "dS_sd": dS.std(0, ddof=1), "pre": pre.mean(0)}
        nd = len(dn)
        moved = np.abs(dS.mean(0)[:nd]) > 2 * dS.std(0, ddof=1)[:nd] / np.sqrt(a.repeats) + 0.5
        log(f"  {name:<9} DNs with |dS| > noise+0.5 spikes: {int(moved.sum())}/{nd}   "
            f"mean |dS| over DNs {np.abs(dS.mean(0)[:nd]).mean():.3f}   "
            f"MBON |dS| {np.abs(dS.mean(0)[nd:]).mean():.3f}   ({time.time()-t0:.0f}s)")
        del b

    nd = len(dn)
    A, R = res["both"]["dS_mean"][:nd], res["reversed"]["dS_mean"][:nd]
    Am, Rm = res["both"]["dS_mean"][nd:], res["reversed"]["dS_mean"][nd:]
    corr_dn = float(np.corrcoef(A, R)[0, 1]) if A.std() and R.std() else float("nan")
    corr_mb = float(np.corrcoef(Am, Rm)[0, 1]) if Am.std() and Rm.std() else float("nan")
    sd = np.sqrt((res["both"]["dS_sd"][:nd] ** 2 + res["reversed"]["dS_sd"][:nd] ** 2) / 2)
    z = (A - R) / np.maximum(sd / np.sqrt(a.repeats) * np.sqrt(2), 1e-9)
    flip = (np.sign(A) != np.sign(R)) & (np.abs(z) > 3) & (np.abs(A - R) > 1.0)
    log(f"\n  corr(dS_both, dS_reversed):  MBONs {corr_mb:+.3f}   DNs {corr_dn:+.3f}")
    log(f"  DNs whose (CS+ - CS-) change reverses with the contingency (|z|>3, |diff|>1 spike): "
        f"{int(flip.sum())}/{nd}")
    rows = []
    for i in np.argsort(-np.abs(z))[:25]:
        if abs(z[i]) < 2:
            break
        rows.append({"type": ty[i], "side": side[i], "dS_both": float(A[i]),
                     "dS_reversed": float(R[i]), "z": float(z[i]),
                     "pre_S": float(res["both"]["pre"][i])})
        log(f"    {ty[i]:<12} {side[i]}  both {A[i]:+7.2f}  reversed {R[i]:+7.2f}  "
            f"z {z[i]:+6.1f}  naive S {res['both']['pre'][i]:+7.1f}")
    out = {"corr_dn": corr_dn, "corr_mbon": corr_mb, "n_flip": int(flip.sum()),
           "n_dn": nd, "top": rows,
           "arms": {k: {"dS_mean": v["dS_mean"].tolist(), "dS_sd": v["dS_sd"].tolist()}
                    for k, v in res.items()},
           "types": ty.tolist(), "sides": [str(s) for s in side]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
