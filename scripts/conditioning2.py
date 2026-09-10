#!/usr/bin/env python3
"""
conditioning2.py - does the connectome learn? Properly measured this time.

    python3 conditioning2.py --brain brain_mirrored.npz --repeats 12 --trials 20

WHY THIS EXISTS
    Every conditioning number produced before this script came from ONE presentation
    of each odour. Measured directly, the noise floor of that measurement is mean
    -0.108 with sd 0.053 - a 95% band of +-0.103 with zero learning happening. Every
    result claimed (-0.012, -0.014, -0.059, +0.183, +0.179) sat inside it. They were
    measurements of noise reported as findings.

    The steering experiment was only settled by repeating trials and comparing
    distributions (d'=4.21, p=3e-11). This applies the same discipline here.

DESIGN - paired, two-armed

    discrimination index per repeat:   d = (CS+ - CS-) / (CS+ + CS-)

    For each of N independent noise seeds, d is measured BEFORE and AFTER the
    training block using that same seed, so each seed is its own control and
    between-repeat variance drops out:

        delta(s) = d_post(s) - d_pre(s)

    Two arms run the identical protocol - same stimuli, same dopamine, same number
    of steps, same seeds - differing in one respect only:

        TRAINED    plasticity on during the training block
        CONTROL    plasticity off during the training block

    If plasticity is doing something, delta differs between the arms. If it is not,
    the two distributions overlap. That comparison is the whole experiment; a shift
    in the trained arm alone proves nothing, because the measurement itself drifts.
"""
from __future__ import annotations

import argparse, json
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def build(path, args, seed=11):
    b = FlyBrain(path, Params(gain=1.0, learn_rate=args.rate,
                              kc_thresh_scale=args.kc_thresh, apl_gain=args.apl,
                              noise=args.noise), seed=seed)
    b.enable_plasticity()
    # NARROW, disjoint odours. Giving each odour half the antennal lobe (26 of 53
    # glomeruli) made 41.4% of CS+ Kenyon cells also respond to CS-, despite the
    # input channels being disjoint: a KC samples ~5 glomeruli, so with half the lobe
    # active nearly every KC catches something and identity stops selecting the code.
    # Real odours drive a handful of glomeruli. Narrow sets restore separable codes.
    rt = b.receptor_types
    k = args.channels
    b._odor_map["CS+"] = {t: 1.0 for t in rt[:k]}
    b._odor_map["CS-"] = {t: 1.0 for t in rt[k:2 * k]}
    return b


def reseed(b, seed):
    """Independent noise for this repeat, so repeats are genuinely independent."""
    b.rng = np.random.default_rng(seed)
    if b._noise_pool is not None:
        b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                         * b.p.noise)
        b._noise_off = 0


def measure(b, readout, seed, settle_ms, ms):
    """Discrimination index for one noise seed. Plasticity is off while measuring."""
    was, b.plastic_on = b.plastic_on, False
    counts = {}
    for o in ("CS+", "CS-"):
        reseed(b, seed)
        b.reset()
        b.smell({o: 1.0})
        b.stimulate("PPL1", 0.0)
        for _ in range(int(settle_ms / b.p.dt)):
            b.step()
        v = 0
        for _ in range(int(ms / b.p.dt)):
            v += int(b.step()[readout].sum())
        counts[o] = v
    b.plastic_on = was
    tot = counts["CS+"] + counts["CS-"]
    d = (counts["CS+"] - counts["CS-"]) / tot if tot else 0.0
    return d, counts


def train_block(b, trials, train_ms, plastic: bool, punished: str = "CS+"):
    """Identical across arms except for plasticity and which odour is punished."""
    was, b.plastic_on = b.plastic_on, plastic
    for _ in range(trials):
        for o, punish in (("CS+", punished == "CS+"), ("CS-", punished == "CS-")):
            b.reset()
            b.smell({o: 1.0})
            b.stimulate("PPL1", 180.0 if punish else 0.0)
            for _ in range(int(train_ms / b.p.dt)):
                b.step()
                if plastic:
                    b.learn()
    b.plastic_on = was


def run_arm(path, args, plastic: bool, label: str, punished: str = "CS+"):
    b = build(path, args)
    mbon = b.pop["MBON"]
    pun = b._da_w["PPL1"][mbon] > 0
    rew = b._da_w["PAM"][mbon] > 0
    readout = mbon[pun & ~rew]        # punishment-only MBONs: a clean, uncontaminated arm
    seeds = [1000 + 97 * i for i in range(args.repeats)]

    pre = [measure(b, readout, s, args.settle_ms, args.test_ms)[0] for s in seeds]
    train_block(b, args.trials, args.train_ms, plastic, punished)
    post = [measure(b, readout, s, args.settle_ms, args.test_ms)[0] for s in seeds]

    pre, post = np.array(pre), np.array(post)
    delta = post - pre
    frac = float(np.mean(b._out_w[b._plastic] / np.maximum(b._w0, 1e-9)))
    log(f"  {label:<9} d_pre {pre.mean():+.4f}+-{pre.std(ddof=1):.4f}   "
        f"d_post {post.mean():+.4f}+-{post.std(ddof=1):.4f}   "
        f"delta {delta.mean():+.4f}+-{delta.std(ddof=1):.4f}   "
        f"weights {100*frac:.1f}%")
    return delta, len(readout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_mirrored.npz")
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--train-ms", type=float, default=800.0)
    ap.add_argument("--test-ms", type=float, default=800.0)
    ap.add_argument("--settle-ms", type=float, default=250.0)
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--kc-thresh", type=float, default=3.0)
    ap.add_argument("--apl", type=float, default=3000.0)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--channels", type=int, default=5,
                    help="glomeruli per odour (of 53). Real odours use few.")
    ap.add_argument("--out", default="conditioning2_result.json")
    a = ap.parse_args()

    log(f"paired two-arm design: {a.repeats} noise seeds, {a.trials} training trials\n")
    trained, n_read = run_arm(a.brain, a, plastic=True, label="punish CS+")
    reversed_, _ = run_arm(a.brain, a, plastic=True, label="punish CS-", punished="CS-")
    control, _ = run_arm(a.brain, a, plastic=False, label="no plastic")

    log(f"\n  readout: {n_read} MBONs reached by punishment dopamine only")

    # Did plasticity change anything, over and above the measurement's own drift?
    u, p_between = mannwhitneyu(trained, control, alternative="two-sided")
    try:
        _, p_trained = wilcoxon(trained)
    except ValueError:
        p_trained = 1.0
    pooled = np.sqrt((trained.std(ddof=1) ** 2 + control.std(ddof=1) ** 2) / 2)
    effect = (trained.mean() - control.mean()) / pooled if pooled else 0.0

    ur, p_rev = mannwhitneyu(trained, reversed_, alternative="two-sided")
    log(f"\n  punish CS+ delta  {trained.mean():+.4f}  (sd {trained.std(ddof=1):.4f})")
    log(f"  punish CS- delta  {reversed_.mean():+.4f}  (sd {reversed_.std(ddof=1):.4f})")
    log(f"  THE SPECIFICITY TEST: reversing which odour is punished should flip the sign.")
    log(f"    difference {trained.mean()-reversed_.mean():+.4f}   p={p_rev:.3g}   "
        f"signs {'OPPOSITE - associative' if trained.mean()*reversed_.mean() < 0 else 'SAME - non-specific depression'}")
    log(f"  control delta  {control.mean():+.4f}  (sd {control.std(ddof=1):.4f})")
    log(f"  difference     {trained.mean()-control.mean():+.4f}   d'={effect:.2f}   "
        f"p={p_between:.3g}")

    specific = trained.mean() * reversed_.mean() < 0 and p_rev < 0.05
    verdict = ("ASSOCIATIVE LEARNING DEMONSTRATED - the shift follows which odour was "
               "punished and reverses when the contingency reverses"
               if specific and p_between < 0.05 else
               "NON-SPECIFIC - plasticity changes the output but not according to the "
               "contingency" if p_between < 0.05 else
               "NOT DEMONSTRATED - indistinguishable from the untrained control")
    log(f"\n  {verdict}")

    json.dump({"trained_delta": trained.tolist(), "reversed_delta": reversed_.tolist(),
               "control_delta": control.tolist(), "p_reversal": float(p_rev),
               "p_between": float(p_between), "effect_size": float(effect),
               "repeats": a.repeats, "trials": a.trials, "readout_n": int(n_read),
               "verdict": verdict}, open(a.out, "w"), indent=1)
    log(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
