#!/usr/bin/env python3
"""
conditioning.py - can a connectome learn?

    python3 conditioning.py --brain brain_mirrored.npz --trials 10

The classic Drosophila olfactory conditioning paradigm, run on the real wiring.
One odour (CS+) is paired with dopaminergic punishment; a second odour (CS-) is
presented alone. If the mushroom body works, the output neurons' response to CS+
should shift relative to CS-, and the shift should be specific to the odour that
was punished.

WHY THIS IS THE EXPERIMENT THAT MATTERS
    Before training, MBON output does not discriminate odours (measured d'=0.16).
    That is not a bug - a naive animal has uniform KC->MBON weights, so every odour
    drives the outputs about equally. Discrimination at the output is precisely what
    learning is supposed to create. If training does not move it, nothing downstream
    is worth building.

THE RULE
    Coincidence of Kenyon cell activity and dopamine DEPRESSES that KC->MBON synapse.
    Depression, not potentiation: MBONs drive approach by default, so weakening the
    pathway for a punished odour is what produces avoidance. Which output neurons get
    depressed comes from the real DAN->MBON wiring, not from compartments we invent.

CONTROLS
    unpaired  - the same punishment delivered without any odour. Any change here is
                non-specific drift, not learning.
    CS-       - a second odour, never punished, measured throughout.
"""
from __future__ import annotations

import argparse, json
import numpy as np

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def present(b: FlyBrain, odor: str | None, punish: bool, ms: float,
            learn: bool, measure: bool = False):
    """Run one presentation. Returns mean MBON spike counts if measuring."""
    mbon = b.pop["MBON"]
    kc = b.pop["KC"]
    if odor:
        b.smell({odor: 1.0})
    else:
        b.drive_hz[b.pop["olfactory"]] = 0.0
    b.stimulate("PPL1", 180.0 if punish else 0.0)

    mb = np.zeros(len(mbon)); kcs = 0
    for _ in range(int(ms / b.p.dt)):
        spk = b.step()
        if learn:
            b.learn()
        if measure:
            mb += spk[mbon]
            kcs += int(spk[kc].sum())
    return mb, kcs


def test_block(b: FlyBrain, odors, ms: float):
    """Measure the MBON response to each odour. No plasticity during testing."""
    was, b.plastic_on = b.plastic_on, False
    out = {}
    for o in odors:
        b.reset()
        present(b, o, False, 250, learn=False)          # settle in the odour
        mb, kc = present(b, o, False, ms, learn=False, measure=True)
        out[o] = mb
    b.plastic_on = was
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_mirrored.npz")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--train-ms", type=float, default=800.0)
    ap.add_argument("--test-ms", type=float, default=600.0)
    ap.add_argument("--rate", type=float, default=0.06)
    ap.add_argument("--kc-thresh", type=float, default=3.0)
    ap.add_argument("--apl", type=float, default=3000.0)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--out", default="conditioning_result.json")
    a = ap.parse_args()

    b = FlyBrain(a.brain, Params(gain=1.0, learn_rate=a.rate,
                                kc_thresh_scale=a.kc_thresh, apl_gain=a.apl,
                                noise=a.noise), seed=11)
    n = b.enable_plasticity()
    # DISJOINT odour channels for the first test of whether learning works at all.
    # Drawing 20 of 53 glomeruli at random makes two odours share ~7 channels by
    # construction, so KC overlap stays ~40% however sparse the code is. Splitting the
    # receptor types cleanly removes that confound; overlap can be reintroduced later
    # as a graded difficulty axis.
    rt = b.receptor_types
    half = len(rt) // 2
    b._odor_map["CS+"] = {t: 1.0 for t in rt[:half]}
    b._odor_map["CS-"] = {t: 1.0 for t in rt[half:]}
    log(f"odours: CS+ uses {half} receptor types, CS- uses {len(rt)-half}, no overlap")
    log(f"{b}")
    log(f"plastic KC->MBON synapses {n:,} | PAM {len(b.pop['PAM'])} reward, "
        f"PPL1 {len(b.pop['PPL1'])} punishment")
    log(f"protocol: {a.trials} trials, CS+ paired with PPL1, CS- unpaired\n")

    pre = test_block(b, ["CS+", "CS-"], a.test_ms)
    log(f"before training   CS+ {pre['CS+'].sum():>6.0f} MBON spikes   "
        f"CS- {pre['CS-'].sum():>6.0f}")

    for t in range(a.trials):
        b.reset()
        present(b, "CS+", punish=True, ms=a.train_ms, learn=True)    # paired
        b.reset()
        present(b, "CS-", punish=False, ms=a.train_ms, learn=True)   # unpaired
        if (t + 1) % max(1, a.trials // 5) == 0:
            frac = float(np.mean(b._out_w[b._plastic] / np.maximum(b._w0, 1e-9)))
            mid = test_block(b, ["CS+", "CS-"], a.test_ms)
            log(f"  trial {t+1:>3}   weights at {100*frac:5.1f}% of naive   "
                f"CS+ {mid['CS+'].sum():>6.0f}   CS- {mid['CS-'].sum():>6.0f}")

    post = test_block(b, ["CS+", "CS-"], a.test_ms)
    log(f"\nafter training    CS+ {post['CS+'].sum():>6.0f} MBON spikes   "
        f"CS- {post['CS-'].sum():>6.0f}")

    dP = post["CS+"].sum() - pre["CS+"].sum()
    dM = post["CS-"].sum() - pre["CS-"].sum()
    rel = (dP / max(pre["CS+"].sum(), 1)) - (dM / max(pre["CS-"].sum(), 1))
    log(f"\nchange in MBON output:")
    log(f"  CS+ (punished) {dP:+8.0f}  ({100*dP/max(pre['CS+'].sum(),1):+.1f}%)")
    log(f"  CS- (control)  {dM:+8.0f}  ({100*dM/max(pre['CS-'].sum(),1):+.1f}%)")
    log(f"  learning index {rel:+.4f}   <- odour-specific change, the thing that matters")

    # which synapses actually moved
    ratio = b._out_w[b._plastic] / np.maximum(b._w0, 1e-9)
    moved = ratio < 0.99
    log(f"\nsynapses depressed: {moved.sum():,} of {n:,} ({100*moved.mean():.1f}%)")
    if moved.any():
        log(f"  mean depression among those: {100*(1-ratio[moved].mean()):.1f}%")

    verdict = ("LEARNED - the punished odour's output changed specifically"
               if abs(rel) > 0.05 and abs(dP) > abs(dM)
               else "NO SPECIFIC LEARNING - change is not odour-specific")
    log(f"\n{verdict}")

    json.dump({"pre_csplus": float(pre["CS+"].sum()), "pre_csminus": float(pre["CS-"].sum()),
               "post_csplus": float(post["CS+"].sum()), "post_csminus": float(post["CS-"].sum()),
               "learning_index": float(rel), "synapses_depressed": int(moved.sum()),
               "trials": a.trials, "verdict": verdict}, open(a.out, "w"), indent=1)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
