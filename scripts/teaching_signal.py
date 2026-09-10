#!/usr/bin/env python3
"""
teaching_signal.py - is the dopamine that teaches actually about the punished odour?

    python3 teaching_signal.py --brain brain_mirrored.npz --repeats 6

WHY THIS EXISTS
    probe_ladder.py ruled out the Kenyon-cell code: depression-only could shift
    discrimination by +0.217 and the real rule delivers -0.040. So the fault is in
    credit assignment, and credit assignment has exactly two inputs:

        WHICH cells are tagged   the Kenyon eligibility trace at the moment dopamine
                                 arrives - must be the CS+ code, not both codes
        HOW MUCH dopamine        the phasic PPL1 signal at each MBON - must be much
                                 larger when punishment is delivered than when an
                                 odour merely drives PPL1 through the network

    FINDINGS already records the second one looking dangerous: PPL1 fires 941 spikes
    during an UNPAIRED odour and 1,348 with punishment, a ratio of only 1.43. This
    measures both inputs directly, so the next fix is aimed rather than guessed.

WHAT A FAILURE LOOKS LIKE
    If the tagged population is the same for CS+ and CS-, no dopamine schedule can be
    specific: both odours share the tag.
    If phasic dopamine is nearly equal with and without punishment, the teaching signal
    carries no information about what just happened, and the rule correctly learns
    "everything is bad" - which is exactly the small non-specific depression measured.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def build(path, args, seed=11):
    b = FlyBrain(path, Params(gain=1.0, kc_thresh_scale=args.kc_thresh,
                              apl_gain=args.apl, noise=args.noise,
                              da_baseline_ms=args.da_baseline_ms,
                              kc_trace_ms=args.kc_trace_ms), seed=seed)
    b.enable_plasticity()
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


def trial(b, odour, punish, seed, odour_ms, punish_hz, settle_ms=100.0):
    """
    One presentation. Returns the KC eligibility trace and the phasic dopamine at the
    MBONs, sampled at the moment dopamine is largest - which is the only moment that
    changes any weight.
    """
    reseed(b, seed)
    b.reset()
    b.plastic_on = True
    b._kc_trace[:] = 0.0
    b._da_at_mbon[:] = 0.0
    b._da_base[:] = 0.0

    kc = b.pop["KC"]
    ppl1 = b.pop["PPL1"]
    mbon = b.pop["MBON"]

    b.smell({odour: 1.0})
    for _ in range(int(settle_ms / b.p.dt)):
        b.step(); b.learn()

    ppl1_spikes = 0
    kc_counts = np.zeros(len(kc), dtype=np.int64)
    best_da, best_trace, best_t = -1.0, None, -1

    n = int(odour_ms / b.p.dt)
    for t in range(n):
        # punishment is delivered over the second half, as in conditioning2
        # PPL1 are DANs, not receptors. drive_hz only reaches neurons in
        # SENSORY_CLASSES, so setting it here does NOTHING - a first version of this
        # script did exactly that and measured 829 vs 828 PPL1 spikes, "proving" that
        # punishment had no effect. stimulate() applies tonic current via _ext, which
        # is how conditioning2.py delivers it and the only way that works.
        if punish and t > n * 0.5:
            b.stimulate("PPL1", punish_hz)
        else:
            b.stimulate("PPL1", 0.0)
        spk = b.step()
        b.learn()
        ppl1_spikes += int(spk[ppl1].sum())
        kc_counts += spk[kc]

        phasic = np.maximum(b._da_at_mbon - b._da_base, 0.0)
        tot = float(phasic[mbon].sum())
        if tot > best_da:
            best_da, best_trace, best_t = tot, b._kc_trace[kc].copy(), t

    b.stimulate("PPL1", 0.0)
    return {"ppl1": ppl1_spikes, "kc_counts": kc_counts,
            "peak_da": best_da, "trace": best_trace, "peak_t": best_t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_mirrored.npz")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--channels", type=int, default=5)
    ap.add_argument("--odour-ms", type=float, default=600.0)
    ap.add_argument("--punish-hz", type=float, default=180.0)
    ap.add_argument("--kc-thresh", type=float, default=1.5)
    ap.add_argument("--apl", type=float, default=200.0)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--da-baseline-ms", type=float, default=4000.0)
    ap.add_argument("--kc-trace-ms", type=float, default=1200.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="results/teaching_signal.json")
    a = ap.parse_args()

    log(f"[1/3] loading {a.brain}")
    b = build(a.brain, a)
    log(f"      PPL1 {len(b.pop['PPL1'])} | PAM {len(b.pop['PAM'])} | "
        f"MBON {len(b.pop['MBON'])} | KC {len(b.pop['KC'])}")

    conds = [("CS+", False), ("CS-", False), ("CS+", True), ("CS-", True)]
    res = {f"{o}{'+punish' if p else ''}": [] for o, p in conds}

    log(f"[2/3] {a.repeats} repeats x 4 conditions ({a.odour_ms:.0f} ms each)")
    for r in range(a.repeats):
        for o, p in conds:
            res[f"{o}{'+punish' if p else ''}"].append(
                trial(b, o, p, 500 + r, a.odour_ms, a.punish_hz))
        log(f"      repeat {r + 1}/{a.repeats}")

    log("[3/3] results\n")
    out = {"brain": a.brain, "repeats": a.repeats, "channels": a.channels,
           "odour_ms": a.odour_ms, "punish_hz": a.punish_hz, "conditions": {}}

    log("  DOPAMINE — does punishment stand out from network-driven PPL1?")
    log(f"  {'condition':<14}{'PPL1 spikes':>13}{'peak phasic DA':>17}")
    for k, v in res.items():
        ppl1 = float(np.mean([x["ppl1"] for x in v]))
        da = float(np.mean([x["peak_da"] for x in v]))
        out["conditions"][k] = {"ppl1_spikes": ppl1, "peak_phasic_da": da}
        log(f"  {k:<14}{ppl1:>13.0f}{da:>17.4f}")

    base = out["conditions"]["CS+"]["peak_phasic_da"]
    pun = out["conditions"]["CS++punish"]["peak_phasic_da"]
    out["da_punish_over_unpaired"] = (pun / base) if base > 1e-9 else float("inf")
    log(f"\n  phasic DA ratio, punished / unpaired = "
        f"{out['da_punish_over_unpaired']:.2f}x")

    log("\n  TAGGING — is the eligibility trace selective for the odour present?")
    tp = np.mean([x["trace"] for x in res["CS++punish"]], axis=0)
    tm = np.mean([x["trace"] for x in res["CS-+punish"]], axis=0)
    denom = np.linalg.norm(tp) * np.linalg.norm(tm)
    cos = float(tp @ tm / denom) if denom > 1e-12 else 0.0
    top_p = set(np.argsort(tp)[::-1][:200].tolist())
    top_m = set(np.argsort(tm)[::-1][:200].tolist())
    jac = len(top_p & top_m) / len(top_p | top_m)
    out["trace_cosine_CSplus_vs_CSminus"] = cos
    out["trace_top200_jaccard"] = jac
    log(f"  cosine(trace | CS+, trace | CS-)      = {cos:.3f}   (1.0 = identical tag)")
    log(f"  overlap of top-200 tagged cells       = {jac:.1%}")

    verdicts = []
    if out["da_punish_over_unpaired"] < 1.8:
        verdicts.append("DOPAMINE CARRIES LITTLE INFORMATION — punishment barely exceeds "
                        "the dopamine an odour drives on its own")
    if cos > 0.75:
        verdicts.append("TAG IS NOT ODOUR-SPECIFIC — both odours tag nearly the same cells")
    if not verdicts:
        verdicts.append("both inputs look usable — the fault is elsewhere in the rule")
    out["verdict"] = verdicts
    log("")
    for v in verdicts:
        log(f"  >> {v}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({k: v for k, v in out.items()}, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
