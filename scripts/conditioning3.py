#!/usr/bin/env python3
"""
conditioning3.py - conditioning with COMPARTMENT-SPECIFIC dopamine.

    python3 conditioning3.py --brain brain_mirrored.npz --repeats 12 --trials 20

WHAT CHANGED FROM conditioning2.py, AND ONLY THIS
    Everything - protocol, seeds, arms, measurement - is conditioning2's. The single
    difference is where the teaching signal goes.

    conditioning2 delivers punishment with `stimulate("PPL1", 180)`, and
    enable_plasticity() collapses all 24 PPL1 cells into ONE normalised weight vector
    over the MBONs. Measured: that vector is non-zero at 80 of 97 MBONs, so every
    punishment depresses essentially the whole mushroom-body output at once. Nothing
    about that arrangement can be selective, and the measured result was exactly what
    you would predict - punishing CS+ gave -0.040, punishing CS- gave -0.025, same
    sign, p=0.878 for the reversal.

    Real mushroom-body learning is compartmentalised. Individual PPL1 types innervate
    individual compartments and teach only the Kenyon synapses there. Those types are
    in this dataset and were being averaged away:

        PPL105  2 cells -> 25 MBONs      PPL104  2 cells ->  8 MBONs
        PPL107  2 cells -> 25 MBONs      PPL106  2 cells ->  6 MBONs
        PPL102  2 cells -> 22 MBONs      PPL108  2 cells ->  6 MBONs
        PPL101  2 cells -> 18 MBONs      PPL201  2 cells ->  4 MBONs

    and they really are separate - PPL104's targets share NOTHING with PPL101, PPL102,
    PPL103 or PPL107 (Jaccard 0.00).

    So: punish through ONE type, and read out that type's MBONs.

THE TEST IS STILL THE REVERSAL
    A shift in the trained arm alone proves nothing. If learning is associative,
    punishing CS- instead of CS+ must FLIP THE SIGN. Paired noise seeds, so the
    plasticity-off arm is exactly 0.0000 and each seed is its own control.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
from scipy.stats import mannwhitneyu

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def build(path, args, seed=11):
    b = FlyBrain(path, Params(gain=1.0, learn_rate=args.rate,
                              kc_thresh_scale=args.kc_thresh, apl_gain=args.apl,
                              noise=args.noise), seed=seed)
    b.enable_plasticity()
    b.enable_compartments()
    rt = b.receptor_types
    k = args.channels
    b._odor_map["CS+"] = {t: 1.0 for t in rt[:k]}
    b._odor_map["CS-"] = {t: 1.0 for t in rt[k:2 * k]}
    return b


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


def measure(b, readout, seed, settle_ms, ms):
    was, b.plastic_on = b.plastic_on, False
    counts = {}
    for o in ("CS+", "CS-"):
        reseed(b, seed)
        b.reset()
        b.smell({o: 1.0})
        quiet(b)
        for _ in range(int(settle_ms / b.p.dt)):
            b.step()
        v = 0
        for _ in range(int(ms / b.p.dt)):
            v += int(b.step()[readout].sum())
        counts[o] = v
    b.plastic_on = was
    tot = counts["CS+"] + counts["CS-"]
    return ((counts["CS+"] - counts["CS-"]) / tot if tot else 0.0), counts


def train_block(b, trials, train_ms, plastic, compartment, punished, hz, mv):
    was, b.plastic_on = b.plastic_on, plastic
    for _ in range(trials):
        for o in ("CS+", "CS-"):
            b.reset()
            b.smell({o: 1.0})
            quiet(b)
            if o == punished:
                b.stimulate_type(compartment, hz, mv)
            for _ in range(int(train_ms / b.p.dt)):
                b.step()
                if plastic:
                    b.learn()
    quiet(b)
    b.plastic_on = was


def run_arm(path, args, plastic, label, punished):
    b = build(path, args)
    d = b._da_by_type.get(args.compartment)
    if d is None:
        raise SystemExit(f"no such compartment {args.compartment!r}; "
                         f"have {sorted(b._da_by_type)}")
    mbon = b.pop["MBON"]
    # read out ONLY the compartment being taught
    readout = mbon[d["w"][mbon] > 0]
    seeds = [1000 + 97 * i for i in range(args.repeats)]

    pre = [measure(b, readout, s, args.settle_ms, args.test_ms)[0] for s in seeds]
    train_block(b, args.trials, args.train_ms, plastic, args.compartment,
                punished, args.punish_hz, args.punish_mv)
    post = [measure(b, readout, s, args.settle_ms, args.test_ms)[0] for s in seeds]

    pre, post = np.array(pre), np.array(post)
    delta = post - pre
    frac = float(np.mean(b._out_w[b._plastic] / np.maximum(b._w0, 1e-9)))
    log(f"  {label:<14} d_pre {pre.mean():+.4f}   d_post {post.mean():+.4f}   "
        f"delta {delta.mean():+.4f}+-{delta.std(ddof=1):.4f}   weights {100*frac:.1f}%")
    return delta, len(readout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_mirrored.npz")
    ap.add_argument("--compartment", default="PPL105")
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--train-ms", type=float, default=800.0)
    ap.add_argument("--test-ms", type=float, default=800.0)
    ap.add_argument("--settle-ms", type=float, default=250.0)
    ap.add_argument("--punish-hz", type=float, default=180.0)
    ap.add_argument("--punish-mv", type=float, default=70.0,
                    help="tonic drive on the taught compartment. 12.6 mV "
                         "(stimulate's ceiling) fires it 0 times during an odour.")
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--kc-thresh", type=float, default=3.0)
    ap.add_argument("--apl", type=float, default=3000.0)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--channels", type=int, default=5)
    ap.add_argument("--out", default="results/conditioning3.json")
    a = ap.parse_args()

    log(f"compartment-specific conditioning: teaching through {a.compartment} only")
    log(f"{a.repeats} paired noise seeds, {a.trials} training trials\n")

    trained, n_read = run_arm(a.brain, a, True, "punish CS+", "CS+")
    reversed_, _ = run_arm(a.brain, a, True, "punish CS-", "CS-")
    control, _ = run_arm(a.brain, a, False, "no plasticity", "CS+")

    log(f"\n  readout: {n_read} MBONs in compartment {a.compartment}")
    log(f"\n  punish CS+ delta  {trained.mean():+.4f}  (sd {trained.std(ddof=1):.4f})")
    log(f"  punish CS- delta  {reversed_.mean():+.4f}  (sd {reversed_.std(ddof=1):.4f})")
    log(f"  control    delta  {control.mean():+.4f}  (sd {control.std(ddof=1):.4f})")

    p_rev = float(mannwhitneyu(trained, reversed_, alternative="two-sided")[1])
    p_ctl = float(mannwhitneyu(trained, control, alternative="two-sided")[1])
    pooled = np.sqrt((trained.var(ddof=1) + control.var(ddof=1)) / 2) or 1e-9
    dprime = float((trained.mean() - control.mean()) / pooled)

    flipped = (trained.mean() < 0) != (reversed_.mean() < 0)
    log(f"\n  THE SPECIFICITY TEST — reversing which odour is punished must flip the sign")
    log(f"  sign flipped: {'YES' if flipped else 'NO'}    p(reversal) = {p_rev:.4f}")
    log(f"  vs control:   d' = {dprime:+.2f}   p = {p_ctl:.4f}")

    if flipped and p_rev < 0.05:
        verdict = "ASSOCIATIVE — the shift follows the contingency and reverses with it"
    elif p_rev < 0.05:
        verdict = ("SPECIFIC BUT NOT REVERSED — the arms differ significantly without "
                   "flipping sign")
    else:
        verdict = "NOT DEMONSTRATED — the two arms are indistinguishable"
    log(f"\n  {verdict}")

    out = {"brain": a.brain, "compartment": a.compartment, "readout_n": n_read,
           "repeats": a.repeats, "trials": a.trials, "punish_mv": a.punish_mv,
           "trained_delta": trained.tolist(), "reversed_delta": reversed_.tolist(),
           "control_delta": control.tolist(),
           "p_reversal": p_rev, "p_vs_control": p_ctl, "dprime": dprime,
           "sign_flipped": bool(flipped), "verdict": verdict}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
