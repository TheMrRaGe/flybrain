#!/usr/bin/env python3
"""
learned_steering.py - does conditioning change what the fly DOES?

    python3 learned_steering.py --repeats 8 --trials 12

conditioning4 showed the mushroom-body output shifts with the contingency and reverses
with it. That is a readout of 45 MBONs. This asks whether it reaches the action bus:
after punishing CS+ and rewarding CS-, does the steering asymmetry on the descending
neurons turn AWAY from CS+ and TOWARD CS-?

READOUT
    The DNa family, split by side, under a lateralised odour: `contra` on one antenna,
    1.0 on the other (0.4 / 1.0 is the mirror-image stimulus the steering d'=4.21 was
    measured with). Turn index T = (R - L)/(R + L) on the DNa family; DNa02 drives an
    ipsilateral turn, so T > 0 is a right turn.

        approach index  A(o) = T(o on the right) - T(o on the left)   > 0: toward
        valence         V    = A(CS+) - A(CS-)

    Structural left/right bias (measured 16x the side signal) cancels in A, and again
    in post - pre on a paired seed. Nothing is calibrated away by hand.

PREDICTION
    punish CS+ / reward CS-  ->  dV < 0.    Swap the contingency  ->  dV > 0.
    Plasticity off           ->  dV = 0.0000 exactly.

    The training and the odours are conditioning4's, unchanged. The MBON compartments
    are counted alongside so the synaptic result is confirmed in the same run.
"""
from __future__ import annotations

import argparse, json, os, time
from types import SimpleNamespace
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from flysim import FlyBrain, Params
from conditioning4 import build, reseed, quiet, compartments, train_block, order


def log(m): print(m, flush=True)


def turn(R, L):
    tot = R + L
    return (R - L) / tot if tot else 0.0


def readouts(b):
    dn = b.pop["DN"]; sd = b.side[dn]; ty = b.type[dn].astype(str)
    fam = np.char.startswith(ty, "DNa")
    return {"DNa|L": dn[fam & (sd == "L")], "DNa|R": dn[fam & (sd == "R")],
            "DN|L": dn[sd == "L"], "DN|R": dn[sd == "R"]}


def present(b, groups, odour, side, contra, seed, settle_ms, ms):
    reseed(b, seed)
    b.reset()
    ipsi, other = {odour: 1.0}, {odour: contra}
    b.smell_bilateral(ipsi if side == "L" else other, ipsi if side == "R" else other)
    quiet(b)
    for _ in range(int(settle_ms / b.p.dt)):
        b.step()
    c = {k: 0 for k in groups}
    for _ in range(int(ms / b.p.dt)):
        s = b.step()
        for k, idx in groups.items():
            c[k] += int(s[idx].sum())
    return c


def measure(b, groups, seed, a):
    """Returns dict with V, A(CS+), A(CS-), per-condition T and raw counts."""
    was, b.plastic_on = b.plastic_on, False
    T, raw = {}, {}
    for o in ("CS+", "CS-"):
        for side in ("L", "R"):
            c = present(b, groups, o, side, a.contra, seed, a.settle_ms, a.test_ms)
            T[(o, side)] = {"DNa": turn(c["DNa|R"], c["DNa|L"]),
                            "DN": turn(c["DN|R"], c["DN|L"])}
            raw[f"{o}|{side}"] = c
    b.plastic_on = was
    out = {}
    for ro in ("DNa", "DN"):
        Ap = T[("CS+", "R")][ro] - T[("CS+", "L")][ro]
        Am = T[("CS-", "R")][ro] - T[("CS-", "L")][ro]
        out[ro] = {"V": Ap - Am, "A+": Ap, "A-": Am}
    out["raw"] = raw
    return out


def run_arm(a, plastic, label, schedule):
    t0 = time.time()
    b = build(a.brain, a)
    groups = readouts(b)
    ro_a, ro_p, _ = compartments(b, a.aversive, a.appetitive)
    groups["A"] = ro_a; groups["P"] = ro_p
    seeds = [1000 + 97 * i for i in range(a.repeats)]

    pre = [measure(b, groups, s, a) for s in seeds]
    train_block(b, a.trials, a.train_ms, plastic, schedule, a.punish_hz, a.punish_mv,
                order(a))
    post = [measure(b, groups, s, a) for s in seeds]

    res = {}
    for ro in ("DNa", "DN"):
        dV = np.array([q[ro]["V"] - p[ro]["V"] for p, q in zip(pre, post)])
        dAp = np.array([q[ro]["A+"] - p[ro]["A+"] for p, q in zip(pre, post)])
        dAm = np.array([q[ro]["A-"] - p[ro]["A-"] for p, q in zip(pre, post)])
        res[ro] = {"dV": dV.tolist(), "dA+": dAp.tolist(), "dA-": dAm.tolist(),
                   "V_pre": [p[ro]["V"] for p in pre], "V_post": [q[ro]["V"] for q in post]}
    # MBON compartments, summed over both sides of presentation
    def d_comp(m, key):
        cp = sum(m["raw"][f"CS+|{s}"][key] for s in "LR")
        cm = sum(m["raw"][f"CS-|{s}"][key] for s in "LR")
        return (cp - cm) / (cp + cm) if cp + cm else 0.0
    dD = np.array([(d_comp(q, "A") - d_comp(q, "P")) - (d_comp(p, "A") - d_comp(p, "P"))
                   for p, q in zip(pre, post)])
    res["MBON_dD"] = dD.tolist()
    spikes = np.mean([sum(p["raw"][k]["DNa|L"] + p["raw"][k]["DNa|R"]
                          for k in p["raw"]) / 4 for p in pre])
    dV = np.array(res["DNa"]["dV"])
    log(f"  {label:<10} DNa dV {dV.mean():+.4f}+-{dV.std(ddof=1):.4f}   "
        f"[dA+ {np.mean(res['DNa']['dA+']):+.4f}  dA- {np.mean(res['DNa']['dA-']):+.4f}]   "
        f"all-DN dV {np.mean(res['DN']['dV']):+.4f}   MBON dD {dD.mean():+.4f}   "
        f"({spikes:.0f} DNa spikes/presentation, {time.time()-t0:.0f}s)")
    return res


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
    ap.add_argument("--contra", type=float, default=0.4,
                    help="odour strength on the far antenna (near antenna is 1.0)")
    ap.add_argument("--punish-hz", type=float, default=180.0)
    ap.add_argument("--punish-mv", type=float, default=70.0)
    ap.add_argument("--rate", type=float, default=0.02)
    ap.add_argument("--kc-thresh", type=float, default=1.5)
    ap.add_argument("--apl-scale", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--odours", default="../results/odours3.json")
    ap.add_argument("--channels", type=int, default=5)
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--cs-minus-first", action="store_true")
    ap.add_argument("--arms", default="both,reversed,control")
    ap.add_argument("--naive-only", action="store_true",
                    help="just print the naive lateralised response for 3 seeds")
    ap.add_argument("--out", default="../results/learned_steering.json")
    a = ap.parse_args()

    AV, AP = a.aversive, a.appetitive
    arms = {"both":     (True,  {"CS+": AV, "CS-": AP}),
            "reversed": (True,  {"CS+": AP, "CS-": AV}),
            "control":  (False, {"CS+": AV, "CS-": AP})}

    log(f"learned steering: train through {AV} / {AP}, read the DNa family")
    log(f"odours {a.odours}{' (swapped)' if a.swap else ''}, near 1.0 / far {a.contra}, "
        f"{a.repeats} paired seeds, {a.trials} trials\n")

    if a.naive_only:
        b = build(a.brain, a)
        groups = readouts(b)
        ro_a, ro_p, _ = compartments(b, AV, AP)
        groups["A"] = ro_a; groups["P"] = ro_p
        log(f"DNa family: {len(groups['DNa|L'])} L / {len(groups['DNa|R'])} R cells")
        for s in (1000, 1097, 1194):
            m = measure(b, groups, s, a)
            for k, c in m["raw"].items():
                log(f"  seed {s} {k}: DNa L {c['DNa|L']:4d} R {c['DNa|R']:4d}  "
                    f"T {turn(c['DNa|R'], c['DNa|L']):+.3f}   all-DN L {c['DN|L']:5d} "
                    f"R {c['DN|R']:5d}   MBON A {c['A']:3d} P {c['P']:3d}")
            log(f"  seed {s}: DNa  A+ {m['DNa']['A+']:+.3f}  A- {m['DNa']['A-']:+.3f}  "
                f"V {m['DNa']['V']:+.3f}    all-DN V {m['DN']['V']:+.3f}")
        return

    out = {"args": vars(a), "arms": {}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for w in [s.strip() for s in a.arms.split(",") if s.strip()]:
        plastic, sched = arms[w]
        out["arms"][w] = run_arm(a, plastic, w, sched)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)

    r = out["arms"]
    if "both" in r and "reversed" in r:
        for ro in ("DNa", "DN"):
            tr, rv = np.array(r["both"][ro]["dV"]), np.array(r["reversed"][ro]["dV"])
            p_mwu = float(mannwhitneyu(tr, rv, alternative="two-sided")[1])
            try:
                p_w = float(wilcoxon(tr - rv, alternative="two-sided")[1])
            except ValueError:
                p_w = float("nan")
            pooled = np.sqrt((tr.var(ddof=1) + rv.var(ddof=1)) / 2) or 1e-9
            dp = float((tr.mean() - rv.mean()) / pooled)
            flipped = (tr.mean() < 0) != (rv.mean() < 0)
            n_sign = int(np.sum(np.sign(tr) != np.sign(rv)))
            log(f"\n  {ro:<4} both dV {tr.mean():+.4f}   reversed dV {rv.mean():+.4f}   "
                f"flipped {'YES' if flipped else 'NO'}   MWU p={p_mwu:.4f}  "
                f"Wilcoxon p={p_w:.4f}  d'={dp:+.2f}   opposite-sign seeds {n_sign}/{len(tr)}")
            out[f"{ro}_p_reversal"] = p_mwu
            out[f"{ro}_p_paired"] = p_w
            out[f"{ro}_dprime"] = dp
            out[f"{ro}_flipped"] = bool(flipped)
        tr, rv = np.array(r["both"]["DNa"]["dV"]), np.array(r["reversed"]["DNa"]["dV"])
        pred = (tr.mean() < 0) and (rv.mean() > 0)
        verdict = ("LEARNED BEHAVIOUR - steering turns away from the punished odour and "
                   "reverses with the contingency" if pred and out["DNa_p_reversal"] < 0.05
                   else "arms differ but not in the predicted direction" if out["DNa_p_reversal"] < 0.05
                   else "NOT DEMONSTRATED at the action bus")
        out["verdict"] = verdict
        log(f"\n  {verdict}")
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
