#!/usr/bin/env python3
"""
conditioning4.py - conditioning read out as a DIFFERENCE BETWEEN COMPARTMENTS.

    python3 conditioning4.py --brain ../brain_mirrored.npz --repeats 8 --trials 12

WHAT CHANGED FROM conditioning3.py, AND ONLY THIS
    conditioning3 taught through one compartment (PPL105) and read that compartment's
    MBONs. Measured: punishing CS+ gave -0.101, punishing CS- gave -0.017. A real
    shift, but the same sign both ways, so it does not follow the contingency.

    That result was PREDICTED by the earlier measurements and it is not a bug. The
    eligibility tags for the two odours have cosine 0.846, so depressing "CS-" cells
    hits CS+ cells too and both responses fall. A depression-only rule driven by
    trace x dopamine only ever subtracts; it never computes which odour is larger.

    The fly does not compute valence inside one compartment either. Punishment
    (PPL1) depresses Kenyon input to APPROACH-driving MBONs; reward (PAM) depresses
    Kenyon input to AVOIDANCE-driving MBONs; behaviour is the balance between the two
    MBON populations. So a depression-only rule becomes bidirectional at the level of
    the readout, not the synapse. (Aso et al. 2014, Owald et al. 2015.)

    The anatomy here supports exactly that split, measured from the DAN->MBON edges:

        PPL105   2 cells -> 25 MBONs   MBON06/11/12/13/18/19/23/31 (gamma1pedc, alpha)
        PAM08   50 cells -> 20 MBONs   MBON01/04/05/09/21/26/27/29/33 (gamma4/5, beta'2)
        shared: 0

    So: punish CS+ through PPL105, reward CS- through PAM08, and read

        D = d_A - d_P

    where d_A is the discrimination index (CS+ - CS-)/(CS+ + CS-) over the PPL105
    MBONs and d_P the same over the PAM08 MBONs. D is the CS+ preference relative to
    CS-. Both training channels should push D the same way, and reversing which odour
    gets which channel should flip its sign.

    The two single-channel arms decompose the effect: they say whether the reversal,
    if it appears, needs both channels or comes from one of them.

THE TEST IS STILL THE REVERSAL
    Paired noise seeds, so the plasticity-off arm is exactly 0.0000 and each seed is
    its own control. Every arm gets the SAME amount of dopamine - one DAN type driven
    on each odour - so "any dopamine depresses everything" cannot masquerade as
    learning.
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def build(path, args, seed=11):
    b = FlyBrain(path, Params(gain=1.0, learn_rate=args.rate,
                              kc_thresh_scale=args.kc_thresh, apl_scale=args.apl_scale,
                              mbon_hold_frac=args.mbon_hold, noise=args.noise), seed=seed)
    b.enable_plasticity()
    b.enable_compartments()
    # CORE MEMBERS ONLY. A DAN type's direct-synapse targets include strays at ~2% of
    # its peak weight (PPL105 -> MBON11: 12 synapses vs 680 onto MBON13). At the old
    # learn rate 2% still drove those synapses to the floor and the strays carried
    # the whole readout. Dopamine now reaches only members at >= `core` of the peak.
    if args.core > 0:
        for d in b._da_by_type.values():
            d["w"] = np.where(d["w"] >= args.core, d["w"], 0.0).astype(np.float32)
    if args.odours:
        od = json.load(open(args.odours))
        X, Y = list(od["CS+"]), list(od["CS-"])
    else:                              # the old alphabetical pair; see odour_design.py
        rt = b.receptor_types
        k = args.channels
        X, Y = rt[:k], rt[k:2 * k]
    if args.swap:                      # counterbalance: the other odour is CS+
        X, Y = Y, X
    b._odor_map["CS+"] = {t: 1.0 for t in X}
    b._odor_map["CS-"] = {t: 1.0 for t in Y}
    return b


def order(args):
    return ("CS-", "CS+") if args.cs_minus_first else ("CS+", "CS-")


def reseed(b, seed):
    b.rng = np.random.default_rng(seed)
    if b._noise_pool is not None:
        b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                         * b.p.noise)
        b._noise_off = 0


def quiet(b):
    """All dopamine off."""
    for t in b._da_by_type:
        b._ext[b._da_by_type[t]["cells"]] = 0.0


def compartments(b, aversive, appetitive):
    """MBON index sets for the two compartments, shared MBONs removed from both."""
    mbon = b.pop["MBON"]
    da, dp = b._da_by_type.get(aversive), b._da_by_type.get(appetitive)
    for name, d in ((aversive, da), (appetitive, dp)):
        if d is None:
            raise SystemExit(f"no such compartment {name!r}; have {sorted(b._da_by_type)}")
    A = set(mbon[da["w"][mbon] > 0].tolist())
    P = set(mbon[dp["w"][mbon] > 0].tolist())
    shared = A & P
    return (np.array(sorted(A - shared)), np.array(sorted(P - shared)), len(shared))


def disc(c):
    tot = c["CS+"] + c["CS-"]
    return (c["CS+"] - c["CS-"]) / tot if tot else 0.0


def measure(b, ro_a, ro_p, seed, settle_ms, ms, odours=("CS+", "CS-")):
    """Returns (D, d_A, d_P, raw counts). Plasticity off, paired seed."""
    was, b.plastic_on = b.plastic_on, False
    ca, cp = {}, {}
    for o in odours:
        reseed(b, seed)
        b.reset()
        b.smell({o: 1.0})
        quiet(b)
        for _ in range(int(settle_ms / b.p.dt)):
            b.step()
        va = vp = 0
        for _ in range(int(ms / b.p.dt)):
            s = b.step()
            va += int(s[ro_a].sum())
            vp += int(s[ro_p].sum())
        ca[o], cp[o] = va, vp
    b.plastic_on = was
    da, dp = disc(ca), disc(cp)
    return da - dp, da, dp, {"A": ca, "P": cp}


def train_block(b, trials, train_ms, plastic, schedule, hz, mv, odours=("CS+", "CS-")):
    """schedule: {odour: dan_type or None} - which DAN type fires on which odour."""
    was, b.plastic_on = b.plastic_on, plastic
    for _ in range(trials):
        for o in odours:
            b.reset()
            b.smell({o: 1.0})
            quiet(b)
            if schedule.get(o):
                b.stimulate_type(schedule[o], hz, mv)
            for _ in range(int(train_ms / b.p.dt)):
                b.step()
                if plastic:
                    b.learn()
    quiet(b)
    b.plastic_on = was


def teaching_check(b, dan_type, odour, mv, ms=800.0):
    """Spikes the driven DAN type fires with the odour present - the trap that sank
    every stimulate()-based protocol was a punishment that never arrived."""
    b.reset()
    b.smell({odour: 1.0})
    quiet(b)
    b.stimulate_type(dan_type, 180.0, mv)
    cells = b._da_by_type[dan_type]["cells"]
    n = 0
    for _ in range(int(ms / b.p.dt)):
        n += int(b.step()[cells].sum())
    quiet(b)
    b.reset()
    return n, len(cells)


def run_arm(path, args, plastic, label, schedule):
    t0 = time.time()
    b = build(path, args)
    ro_a, ro_p, n_shared = compartments(b, args.aversive, args.appetitive)
    seeds = [1000 + 97 * i for i in range(args.repeats)]

    od = order(args)
    pre = np.array([measure(b, ro_a, ro_p, s, args.settle_ms, args.test_ms, od)[:3]
                    for s in seeds])
    train_block(b, args.trials, args.train_ms, plastic, schedule, args.punish_hz,
                args.punish_mv, od)
    post = np.array([measure(b, ro_a, ro_p, s, args.settle_ms, args.test_ms, od)[:3]
                     for s in seeds])

    delta = post - pre                        # columns: D, d_A, d_P
    frac = float(np.mean(b._out_w[b._plastic] / np.maximum(b._w0, 1e-9)))
    wf = b._out_w[b._plastic] / np.maximum(b._w0, 1e-9)
    frac_a = float(wf[np.isin(b._plastic_post, ro_a)].mean())
    frac_p = float(wf[np.isin(b._plastic_post, ro_p)].mean())
    D, dA, dP = delta[:, 0], delta[:, 1], delta[:, 2]
    log(f"  {label:<16} D {pre[:,0].mean():+.4f} -> {post[:,0].mean():+.4f}   "
        f"dD {D.mean():+.4f}+-{D.std(ddof=1):.4f}   "
        f"[d_A {dA.mean():+.4f}  d_P {dP.mean():+.4f}]   "
        f"weights all {100*frac:.1f}% A {100*frac_a:.0f}% P {100*frac_p:.0f}%   "
        f"({time.time()-t0:.0f}s)")
    return {"delta_D": D.tolist(), "delta_dA": dA.tolist(), "delta_dP": dP.tolist(),
            "pre": pre.tolist(), "post": post.tolist(), "weights_frac": frac,
            "weights_frac_A": frac_a, "weights_frac_P": frac_p,
            "n_A": int(len(ro_a)), "n_P": int(len(ro_p)), "n_shared": n_shared}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--aversive", default="PPL105", help="punishment DAN type")
    ap.add_argument("--appetitive", default="PAM08", help="reward DAN type")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--train-ms", type=float, default=800.0)
    ap.add_argument("--test-ms", type=float, default=800.0)
    ap.add_argument("--settle-ms", type=float, default=250.0)
    ap.add_argument("--punish-hz", type=float, default=180.0)
    ap.add_argument("--punish-mv", type=float, default=70.0,
                    help="tonic drive on a taught DAN type. 12.6 mV (stimulate's "
                         "ceiling) fires PPL105 0 times during an odour.")
    # the regime conditioning3's reported run actually used: a live mushroom body
    ap.add_argument("--rate", type=float, default=0.0003,
                    help="0.02 drove synapses at 2%% dopamine weight to the floor")
    ap.add_argument("--core", type=float, default=0.2,
                    help="dopamine-weight cut for compartment membership (0 = all)")
    ap.add_argument("--mbon-hold", type=float, default=0.85, help="decision 14")
    ap.add_argument("--kc-thresh", type=float, default=1.5)
    ap.add_argument("--apl-scale", type=float, default=0.1,
                    help="gain on the real APL neuron (decision 12); 0.1 gives ~5%% KCs")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--odours", default="../results/odours3.json",
                    help="odour pair from odour_design.py; '' for the alphabetical pair")
    ap.add_argument("--channels", type=int, default=5, help="only without --odours")
    ap.add_argument("--swap", action="store_true",
                    help="counterbalance: use receptor types [k:2k] as CS+ and [0:k] "
                         "as CS-. If a contingency-independent shift flips sign with "
                         "this, it is an odour-identity effect.")
    ap.add_argument("--cs-minus-first", action="store_true",
                    help="present CS- before CS+ in every trial. If a "
                         "contingency-independent shift flips with this, it is a "
                         "presentation-order effect.")
    ap.add_argument("--arms", default="both,reversed,control,punish-only,reward-only",
                    help="comma list; the first three are the experiment, the last two "
                         "decompose it")
    ap.add_argument("--out", default="../results/conditioning4.json")
    a = ap.parse_args()

    AV, AP = a.aversive, a.appetitive
    arms = {
        "both":        (True,  {"CS+": AV, "CS-": AP}),
        "reversed":    (True,  {"CS+": AP, "CS-": AV}),
        "control":     (False, {"CS+": AV, "CS-": AP}),
        "punish-only": (True,  {"CS+": AV, "CS-": None}),
        "reward-only": (True,  {"CS+": None, "CS-": AP}),
    }
    wanted = [s.strip() for s in a.arms.split(",") if s.strip()]
    for w in wanted:
        if w not in arms:
            raise SystemExit(f"unknown arm {w!r}; have {list(arms)}")

    log(f"differential conditioning: punish through {AV}, reward through {AP}")
    log(f"readout D = d_A({AV} MBONs) - d_P({AP} MBONs)")
    log(f"{a.repeats} paired noise seeds, {a.trials} training trials, "
        f"{a.punish_mv:.0f} mV DAN drive")
    log(f"odours: {a.odours or 'alphabetical'}{' (swapped)' if a.swap else ''}, "
        f"order per trial: {' then '.join(order(a))}, "
        f"kc_thresh {a.kc_thresh}, apl_scale {a.apl_scale}, mbon_hold {a.mbon_hold}, "
        f"core {a.core}, rate {a.rate}\n")

    # the teaching signals must actually arrive during the odour
    b = build(a.brain, a)
    ro_a, ro_p, n_shared = compartments(b, AV, AP)
    log(f"  readout: {len(ro_a)} MBONs in {AV}, {len(ro_p)} in {AP}, "
        f"{n_shared} shared (excluded)")
    checks = {}
    for dan, od in ((AV, "CS+"), (AP, "CS-")):
        n, k = teaching_check(b, dan, od, a.punish_mv)
        checks[dan] = {"spikes": n, "cells": k}
        log(f"  {dan:<7} at {a.punish_mv:.0f} mV with {od} present: "
            f"{n} spikes / {k} cells over 800 ms  ({n/k/0.8:.0f} Hz per cell)")
    # do the two compartments even see the odours at rest?
    _, dA0, dP0, raw = measure(b, ro_a, ro_p, 1000, a.settle_ms, a.test_ms)
    log(f"  naive response, seed 1000:  {AV} MBONs {raw['A']}  d_A {dA0:+.4f}    "
        f"{AP} MBONs {raw['P']}  d_P {dP0:+.4f}\n")
    del b

    out = {"brain": a.brain, "aversive": AV, "appetitive": AP,
           "swap": a.swap, "cs_minus_first": a.cs_minus_first,
           "repeats": a.repeats, "trials": a.trials, "punish_mv": a.punish_mv,
           "rate": a.rate, "core": a.core, "mbon_hold": a.mbon_hold,
           "kc_thresh": a.kc_thresh, "apl_scale": a.apl_scale,
           "noise": a.noise, "odours": a.odours,
           "teaching_check": checks, "arms": {}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = out["arms"]
    for w in wanted:
        plastic, sched = arms[w]
        res[w] = run_arm(a.brain, a, plastic, w, sched)
        with open(a.out, "w") as f:      # save after every arm: a killed run keeps
            json.dump(out, f, indent=2)  # the arms it finished

    if "both" in res and "reversed" in res:
        tr = np.array(res["both"]["delta_D"])
        rv = np.array(res["reversed"]["delta_D"])
        log(f"\n  both      dD  {tr.mean():+.4f}  (sd {tr.std(ddof=1):.4f})")
        log(f"  reversed  dD  {rv.mean():+.4f}  (sd {rv.std(ddof=1):.4f})")
        if "control" in res:
            ct = np.array(res["control"]["delta_D"])
            log(f"  control   dD  {ct.mean():+.4f}  (sd {ct.std(ddof=1):.4f})")
        p_rev = float(mannwhitneyu(tr, rv, alternative="two-sided")[1])
        # seeds are paired across arms - identical pre, identical noise - so the
        # per-seed difference is a legitimate paired test as well
        try:
            p_rev_paired = float(wilcoxon(tr - rv, alternative="two-sided")[1])
        except ValueError:
            p_rev_paired = float("nan")
        pooled = np.sqrt((tr.var(ddof=1) + rv.var(ddof=1)) / 2) or 1e-9
        d_rev = float((tr.mean() - rv.mean()) / pooled)
        flipped = (tr.mean() < 0) != (rv.mean() < 0)
        n_sign = int(np.sum(np.sign(tr) != np.sign(rv)))

        log(f"\n  THE SPECIFICITY TEST - swapping which odour is punished/rewarded "
            f"must flip the sign of dD")
        log(f"  sign flipped: {'YES' if flipped else 'NO'}   "
            f"p(reversal) MWU = {p_rev:.4f}   paired Wilcoxon = {p_rev_paired:.4f}   "
            f"d' = {d_rev:+.2f}")
        log(f"  seeds with opposite sign in the two arms: {n_sign}/{len(tr)}")

        if flipped and p_rev < 0.05:
            verdict = "ASSOCIATIVE - the shift follows the contingency and reverses with it"
        elif p_rev < 0.05:
            verdict = ("SPECIFIC BUT NOT REVERSED - the arms differ significantly "
                       "without flipping sign")
        else:
            verdict = "NOT DEMONSTRATED - the two arms are indistinguishable"
        log(f"\n  {verdict}")
        out.update({"p_reversal": p_rev, "p_reversal_paired": p_rev_paired,
                    "dprime_reversal": d_rev, "sign_flipped": bool(flipped),
                    "verdict": verdict})

    if "punish-only" in res and "reward-only" in res and "both" in res:
        po = np.array(res["punish-only"]["delta_D"]).mean()
        ro = np.array(res["reward-only"]["delta_D"]).mean()
        bo = np.array(res["both"]["delta_D"]).mean()
        log(f"\n  decomposition:  punish-only {po:+.4f}  +  reward-only {ro:+.4f}  "
            f"=  {po+ro:+.4f}   vs both {bo:+.4f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
