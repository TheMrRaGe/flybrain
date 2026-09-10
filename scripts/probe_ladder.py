#!/usr/bin/env python3
"""
probe_ladder.py - where does odour discrimination actually die?

    python3 probe_ladder.py --brain brain_mirrored.npz --trials 60 --channels 5

WHY THIS EXISTS
    conditioning2.py measured that plasticity produces a small NON-SPECIFIC
    depression (punish CS+ -> -0.040, punish CS- -> -0.025, same sign, p=0.878 for
    the reversal). FINDINGS.md attributes that to Kenyon-cell code overlap: 36.1% of
    CS+ KCs also fire for CS-, so depression hits shared cells and lowers both.

    That is a HYPOTHESIS about which constraint destroys the capability, and it has
    never been tested directly. This tests it, by asking three increasingly
    constrained readouts to separate the two odours from Kenyon-cell activity alone.
    Each rung adds one biological constraint. The rung where accuracy collapses is
    the thing to fix; every rung above it is exonerated.

        rung 1  unconstrained linear readout over all Kenyon cells
                -> are the two odour codes separable AT ALL?

        rung 2  readout restricted to the real KC->MBON wiring, weights forced
                non-negative (they are synapses, they cannot become inhibitory)
                -> does the anatomy have the capacity?

        rung 3  same wiring, but weights may only DECREASE from their measured
                values, and the decision is total MBON drive with a fixed positive
                sign - which is exactly what the dopamine rule can express
                -> can depression-only learning express the discrimination?

    Rung 3 has a closed-form optimum. The objective is linear in the per-cell
    depression factor, so the best possible setting is at the box corners: depress a
    Kenyon cell's output fully if it prefers CS+, leave it alone otherwise. That
    means rung 3 is not an optimisation result that might have got stuck - it is a
    CEILING. No dopamine rule, however well tuned, can beat it.

    These probes are instruments, not models. They are trained, read, and thrown
    away. Nothing here feeds back into the simulation.

HONEST LIMITS
    - Rungs 2 and 3 model the MBON layer as the LINEAR synaptic drive its inputs
      deliver (sum of weight x presynaptic spikes), not as spiking LIF cells. That
      is the right call for a ceiling test - thresholding can only discard
      information - but it means these rungs bound what is available to the readout,
      not what the spiking network currently extracts.
    - There are ~4,000 Kenyon cells and a few hundred trials, so rung 1 can separate
      anything on the training set. Every number reported is held-out, 5-fold, split
      by trial.
"""
from __future__ import annotations

import argparse, json
import numpy as np

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


# --------------------------------------------------------------------------- #
#  data collection
# --------------------------------------------------------------------------- #

def build(path, args, seed=11):
    """Same construction as conditioning2.py, so the numbers are comparable."""
    b = FlyBrain(path, Params(gain=1.0, kc_thresh_scale=args.kc_thresh,
                              apl_gain=args.apl, noise=args.noise), seed=seed)
    n_plastic = b.enable_plasticity()
    rt = b.receptor_types
    k = args.channels
    b._odor_map["CS+"] = {t: 1.0 for t in rt[:k]}
    b._odor_map["CS-"] = {t: 1.0 for t in rt[k:2 * k]}
    return b, n_plastic


def reseed(b, seed):
    b.rng = np.random.default_rng(seed)
    if b._noise_pool is not None:
        b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                         * b.p.noise)
        b._noise_off = 0


def kc_counts(b, kc, ms):
    """Spike count per Kenyon cell over a window, without allocating (steps, N)."""
    acc = np.zeros(len(kc), dtype=np.int32)
    for _ in range(int(round(ms / b.p.dt))):
        acc += b.step()[kc]
    return acc


def collect(b, kc, trials, settle_ms, ms):
    """One KC population vector per odour per trial. Plasticity stays OFF throughout."""
    b.plastic_on = False
    X, y = [], []
    for i in range(trials):
        for label, odour in ((0, "CS+"), (1, "CS-")):
            reseed(b, 1000 + i)
            b.reset()
            b.smell({odour: 1.0})
            for _ in range(int(round(settle_ms / b.p.dt))):
                b.step()
            X.append(kc_counts(b, kc, ms))
            y.append(label)
        if (i + 1) % 10 == 0:
            log(f"      {i + 1}/{trials} trials")
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


# --------------------------------------------------------------------------- #
#  rung 1 - unconstrained linear readout
# --------------------------------------------------------------------------- #

def logreg(Xtr, ytr, Xte, l2=1.0, iters=600, lr=0.5):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    w, b0 = np.zeros(A.shape[1]), 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(A @ w + b0)))
        g = A.T @ (p - ytr) / len(ytr) + l2 * w / len(ytr)
        w -= lr * g
        b0 -= lr * float((p - ytr).mean())
    return (B @ w + b0) > 0


# --------------------------------------------------------------------------- #
#  rung 2 - the real wiring, weights non-negative
# --------------------------------------------------------------------------- #

def anatomical(Xtr, ytr, Xte, W0, iters=4000, lr=0.5):
    """
    Readout = sum over Kenyon cells of (measured weight x depression factor) x spikes,
    with the factor free in [0, 1] per cell - i.e. the anatomy may reweight its own
    synapses arbitrarily downward or leave them, but cannot invent connections and
    cannot make a synapse inhibitory. A free bias absorbs the overall level, so this
    rung asks only whether the ANATOMY separates, not whether the sign is right.

    SCALING MATTERS. Kenyon-cell spike counts times summed synaptic weight put the
    logit in the thousands, the sigmoid saturates, and the gradient vanishes - the
    first version of this returned 50% (chance) while the MORE constrained rung 3
    returned 95%, which is impossible: rung 3's solution is inside rung 2's feasible
    set. That gap was the optimiser failing, not the anatomy. The drive is normalised
    to unit scale and s starts at 1.0, the measured, unlearned state.
    """
    Dtr, Dte = Xtr * W0, Xte * W0
    sc = Dtr.sum(1).std() + 1e-9
    Dtr, Dte = Dtr / sc, Dte / sc
    s, b0 = np.ones(Dtr.shape[1]), 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Dtr @ s + b0, -30, 30)))
        s -= lr * (Dtr.T @ (p - ytr) / len(ytr))
        b0 -= lr * float((p - ytr).mean())
        np.clip(s, 0.0, 1.0, out=s)
    return (Dte @ s + b0) > 0


# --------------------------------------------------------------------------- #
#  rung 3 - depression only, closed form, fixed positive sign
# --------------------------------------------------------------------------- #

def depression_ceiling(Xtr, ytr, Xte, yte, W0, min_frac):
    """
    The most a depression-only rule could ever do.

    MBON drive for an odour is  sum_k  x_k * W0_k * s_k,  s_k in [min_frac, 1].
    The discrimination the fly expresses is CS- drive minus CS+ drive, so the
    objective is LINEAR in s and its maximum sits at a corner:

        s_k = min_frac  if cell k prefers CS+,  else 1

    Fitted on training trials only, then applied to held-out trials. Returns the
    held-out discrimination index before and after, and held-out accuracy using a
    threshold set on the training split.
    """
    plus = Xtr[ytr == 0].mean(0)
    minus = Xtr[ytr == 1].mean(0)
    s = np.where(plus > minus, min_frac, 1.0)

    eff = W0 * s
    d_te = Xte @ eff
    naive_te = Xte @ W0

    def index(d, yy):
        p, m = d[yy == 0].mean(), d[yy == 1].mean()
        return float((m - p) / (m + p)) if (m + p) > 1e-12 else 0.0

    thr = float((Xtr @ eff)[ytr == 0].mean() + (Xtr @ eff)[ytr == 1].mean()) / 2.0
    pred = d_te > thr
    return {
        "index_naive": index(naive_te, yte),
        "index_trained": index(d_te, yte),
        "pred": pred,
        "depressed_cells": int((s < 1.0).sum()),
        "n_cells": int(len(s)),
    }


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #

def folds(n, k=5):
    idx = np.arange(n)
    for f in range(k):
        te = idx[f::k]
        yield np.setdiff1d(idx, te), te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_mirrored.npz")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--channels", type=int, default=5)
    ap.add_argument("--settle", type=float, default=100.0)
    ap.add_argument("--ms", type=float, default=400.0)
    ap.add_argument("--kc-thresh", type=float, default=1.5)
    ap.add_argument("--apl", type=float, default=200.0)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--out", default="results/probe_ladder.json")
    a = ap.parse_args()

    log(f"[1/5] loading {a.brain}")
    b, n_plastic = build(a.brain, a)
    kc = b.pop["KC"]
    log(f"      {len(kc)} Kenyon cells, {n_plastic:,} KC->MBON synapses")

    # measured outgoing KC->MBON weight per Kenyon cell
    W0_all = np.zeros(b.N, dtype=np.float64)
    np.add.at(W0_all, b._plastic_pre, b._w0.astype(np.float64))
    W0 = W0_all[kc]
    connected = W0 > 0
    log(f"      {int(connected.sum())} of {len(kc)} Kenyon cells reach an MBON")

    log(f"[2/5] collecting {a.trials} paired trials ({a.ms:.0f} ms each, plasticity off)")
    X, y = collect(b, kc, a.trials, a.settle, a.ms)

    # code overlap, the number FINDINGS blames
    act = X > 0
    plus_cells = act[y == 0].mean(0) > 0.5
    minus_cells = act[y == 1].mean(0) > 0.5
    overlap = float((plus_cells & minus_cells).sum() / max(plus_cells.sum(), 1))
    log(f"      CS+ active KCs {int(plus_cells.sum())}, CS- {int(minus_cells.sum())}, "
        f"overlap {overlap:.1%}")

    X, W0 = X[:, connected], W0[connected]

    # trial-wise folds: a trial's CS+ and CS- rows travel together
    n = a.trials
    r1, r2, r3 = [], [], []
    idx_plus = np.arange(0, 2 * n, 2)
    idx_minus = np.arange(1, 2 * n, 2)

    log("[3/5] rung 1 - unconstrained linear readout over Kenyon cells")
    log("[4/5] rung 2 - restricted to real KC->MBON wiring, non-negative")
    log("[5/5] rung 3 - depression-only ceiling, closed form")
    ceil = []
    for tr, te in folds(n):
        rows_tr = np.concatenate([idx_plus[tr], idx_minus[tr]])
        rows_te = np.concatenate([idx_plus[te], idx_minus[te]])
        Xtr, ytr, Xte, yte = X[rows_tr], y[rows_tr], X[rows_te], y[rows_te]
        r1.append(float((logreg(Xtr, ytr, Xte) == yte).mean()))
        r2.append(float((anatomical(Xtr, ytr, Xte, W0) == yte).mean()))
        c = depression_ceiling(Xtr, ytr, Xte, yte, W0, b.p.min_weight_frac)
        r3.append(float((c["pred"] == yte).mean()))
        ceil.append(c)

    out = {
        "brain": a.brain,
        "trials": a.trials,
        "channels": a.channels,
        "window_ms": a.ms,
        "n_kenyon": int(len(kc)),
        "n_connected": int(connected.sum()),
        "n_plastic_synapses": int(n_plastic),
        "kc_overlap": overlap,
        "rung1_unconstrained": {"acc": float(np.mean(r1)), "sd": float(np.std(r1))},
        "rung2_anatomical": {"acc": float(np.mean(r2)), "sd": float(np.std(r2))},
        "rung3_depression_only": {
            "acc": float(np.mean(r3)), "sd": float(np.std(r3)),
            "index_naive": float(np.mean([c["index_naive"] for c in ceil])),
            "index_trained": float(np.mean([c["index_trained"] for c in ceil])),
            "depressed_cells": int(np.mean([c["depressed_cells"] for c in ceil])),
        },
        "measured_conditioning_delta": -0.040,
    }
    d = out["rung3_depression_only"]
    out["ceiling_shift"] = d["index_trained"] - d["index_naive"]

    log("")
    log(f"  rung 1  unconstrained        {out['rung1_unconstrained']['acc']:.1%} held-out")
    log(f"  rung 2  real wiring          {out['rung2_anatomical']['acc']:.1%} held-out")
    log(f"  rung 3  depression only      {d['acc']:.1%} held-out")
    log(f"          discrimination index {d['index_naive']:+.4f} -> {d['index_trained']:+.4f}"
        f"  (shift {out['ceiling_shift']:+.4f})")
    log(f"          vs. measured learning shift {out['measured_conditioning_delta']:+.4f}")

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
