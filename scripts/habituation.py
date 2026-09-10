#!/usr/bin/env python3
"""
habituation.py - does the escape pathway stop responding to a stimulus that keeps
being harmless?

    python3 habituation.py --brain brain_whole.npz --seeds 8 --pulses 20

WHY THIS EXISTS
    Olfactory conditioning is stuck behind the Kenyon-cell code (see probe_ladder.py).
    Habituation of escape is a second, INDEPENDENT learning paradigm that does not
    touch the mushroom body at all, so it can be measured while the odour pathway is
    still being argued about. It is also the best-studied learning behaviour in larval
    zebrafish - habituation of the Mauthner-cell startle - and the fly's Giant Fiber
    (DNp01) is the direct functional counterpart: one enormous descending neuron per
    side that triggers the escape jump.

THE PROTOCOL - three blocks, one continuous session

    HABITUATE   N identical stimulus pulses. A habituating circuit responds less each
                time. This is the curve everything else is compared against.
    NOVEL       one pulse driving a DIFFERENT, disjoint set of receptors. If the decrement
                is habituation it is stimulus-SPECIFIC, so a novel stimulus must still
                evoke a full response. If the novel pulse is equally suppressed, the
                circuit is merely fatigued or saturated - that is not learning.
    RETEST      the original stimulus again. Recovery after the novel stimulus is
                dishabituation; recovery over time alone is spontaneous recovery.

    Specificity is the whole experiment. A response that just gets smaller is not
    evidence of anything - synapses deplete, membranes adapt, and both look like
    learning on a single curve.

WHAT TO EXPECT, STATED BEFORE RUNNING
    There is NO plastic synapse anywhere in this pathway. enable_plasticity() makes
    KC->MBON learnable and nothing else, and the Giant Fiber is not downstream of the
    mushroom body. So any decrement measured here cannot be synaptic learning - it
    can only come from recurrent network dynamics: inhibition accumulating in the
    sensory relays and the descending bus.

    That makes this a real question rather than a foregone one. Circuits DO habituate
    without plasticity, through recurrent inhibition alone. If the decrement is there,
    the connectome produces it structurally. If it is flat, the model is missing the
    mechanism and that is the thing to add - either way the answer is worth having,
    and it is worth having BEFORE building a learning rule for this pathway.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def pick_channels(b, rng, n, modality="mechano"):
    """
    Two disjoint stimulus sets in the same modality, so the novelty control is a
    change of WHICH cells, not of what kind of stimulus.

    NOT VISUAL, AND THE REASON MATTERS. This experiment was written for a looming
    visual stimulus - the fly counterpart of zebrafish Mauthner startle habituation.
    Measured: driving all 4,107 photoreceptors at 400 Hz produces 224,794 receptor
    spikes and EXACTLY ZERO spikes in the visual projection neurons or anywhere in
    the descending bus.

    The cause is in the data. All 4,107 visual receptors are histaminergic and all
    29,469 of their outgoing edges carry sign -1: in the fly, photoreceptor output is
    INHIBITORY (histamine gates chloride channels in the lamina). Combine that with
    the model's 0 Hz basal firing - correct per Shiu et al., "inhibitory connections
    to an inactive neuron have no effect" - and the entire visual system is a switch
    wired to nothing. There is no activity for it to suppress. 58% of the connectome
    is optic lobe and it contributes exactly zero spikes.

    This is the same family of error as the two already in FINDINGS (glutamate is
    inhibitory; sensory somas sit outside the volume), one layer further out, and it
    cannot be fixed by driving the receptors harder. Vision needs a tonic baseline it
    can modulate downward. Until then, mechanosensation is the honest choice - and
    tap-habituation of escape is the canonical paradigm in C. elegans and zebrafish
    anyway, so nothing about the experiment is weakened.
    """
    src = b.pop[modality]
    order = rng.permutation(len(src))
    return src[order[:n]], src[order[n:2 * n]]


def pulse(b, channels, hz, pulse_ms, gap_ms, watch):
    """One stimulus presentation. Returns spike counts for each watched population."""
    counts = {k: 0 for k in watch}
    b.drive_hz[:] = 0.0
    b.drive_hz[channels] = hz
    for _ in range(int(round(pulse_ms / b.p.dt))):
        spk = b.step()
        for k, idx in watch.items():
            counts[k] += int(spk[idx].sum())
    b.drive_hz[:] = 0.0
    for _ in range(int(round(gap_ms / b.p.dt))):
        spk = b.step()
        for k, idx in watch.items():
            counts[k] += int(spk[idx].sum())
    return counts


def run_seed(b, seed, a, watch):
    rng = np.random.default_rng(seed)
    b.rng = np.random.default_rng(seed)
    if b._noise_pool is not None:
        b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                         * b.p.noise)
        b._noise_off = 0
    b.reset()

    chan_a, chan_b = pick_channels(b, rng, a.channels, a.modality)

    # let the network settle so the first pulse is not measuring the transient
    b.drive_hz[:] = 0.0
    for _ in range(int(round(300.0 / b.p.dt))):
        b.step()

    hab = [pulse(b, chan_a, a.hz, a.pulse_ms, a.gap_ms, watch) for _ in range(a.pulses)]
    novel = pulse(b, chan_b, a.hz, a.pulse_ms, a.gap_ms, watch)
    retest = [pulse(b, chan_a, a.hz, a.pulse_ms, a.gap_ms, watch)
              for _ in range(a.retests)]
    return {"habituate": hab, "novel": novel, "retest": retest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_whole.npz")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--pulses", type=int, default=20)
    ap.add_argument("--retests", type=int, default=3)
    ap.add_argument("--channels", type=int, default=600)
    ap.add_argument("--modality", default="mechano")
    ap.add_argument("--hz", type=float, default=180.0)
    ap.add_argument("--pulse-ms", type=float, default=80.0)
    ap.add_argument("--gap-ms", type=float, default=220.0)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--std-scope", default="sensory", choices=["sensory","all"])
    ap.add_argument("--std", action="store_true",
                    help="enable short-term synaptic depression")
    # single source of truth: the model's own defaults. A duplicated 0.45 here silently
    # overrode the tuned 0.08 in Params and produced a run 14x less responsive.
    _d = Params()
    ap.add_argument("--std-u", type=float, default=_d.std_u)
    ap.add_argument("--std-tau", type=float, default=_d.std_tau_rec_ms)
    ap.add_argument("--out", default="results/habituation.json")
    a = ap.parse_args()

    log(f"[1/3] loading {a.brain}")
    b = FlyBrain(a.brain, Params(gain=1.0, noise=a.noise, std_u=a.std_u,
                                 std_tau_rec_ms=a.std_tau), seed=7)
    if a.std:
        n = b.enable_std(None if a.std_scope == "all" else
                         ("visual","mechano","olfactory","gustatory","VPN","ALPN"))
        log(f"      short-term depression ON ({a.std_scope}) for {n:,} presynaptic neurons "
            f"(U={a.std_u}, tau_rec={a.std_tau:.0f} ms)")

    dn = b.pop["DN"]
    names = b.type[dn].astype(str)
    gf = dn[np.char.startswith(names, "DNp01")]
    dnp = dn[np.char.startswith(names, "DNp")]
    if len(gf) == 0:
        log("      DNp01 absent from this build - aborting")
        return
    watch = {"giant_fiber": gf, "DNp_escape": dnp, "all_DN": dn,
             "MBON": b.pop["MBON"]}
    log(f"      Giant Fiber {len(gf)} cells | DNp escape family {len(dnp)} | "
        f"all DN {len(dn)} | visual projection {len(b.pop['VPN'])}")
    log("      NOTE: no plastic synapse exists in this pathway. Any decrement is "
        "short-term depression or network dynamics, not associative learning.")

    log(f"[2/3] {a.seeds} seeds x ({a.pulses} habituating + 1 novel + {a.retests} retest) "
        f"pulses")
    runs = []
    for s in range(a.seeds):
        runs.append(run_seed(b, 400 + s, a, watch))
        log(f"      seed {s + 1}/{a.seeds}")

    log("[3/3] summarising")
    out = {"brain": a.brain, "std": bool(a.std), "std_scope": a.std_scope, "seeds": a.seeds, "pulses": a.pulses,
           "pulse_ms": a.pulse_ms, "gap_ms": a.gap_ms, "hz": a.hz,
           "channels": a.channels, "populations": {}}

    for k in watch:
        hab = np.array([[p[k] for p in r["habituate"]] for r in runs], dtype=float)
        nov = np.array([r["novel"][k] for r in runs], dtype=float)
        ret = np.array([[p[k] for p in r["retest"]] for r in runs], dtype=float)

        first = hab[:, 0]
        last = hab[:, -1]
        # normalise every seed to its own first pulse, so seeds with different
        # absolute excitability contribute equally
        denom = np.where(first > 0, first, np.nan)
        norm = hab / denom[:, None]

        with np.errstate(invalid="ignore"):
            decrement = float(np.nanmean(last / denom))
            novel_rel = float(np.nanmean(nov / denom))
            retest_rel = float(np.nanmean(ret[:, 0] / denom))

        out["populations"][k] = {
            "n_cells": int(len(watch[k])),
            "curve_mean": [float(np.nanmean(norm[:, i])) for i in range(a.pulses)],
            "curve_sd": [float(np.nanstd(norm[:, i])) for i in range(a.pulses)],
            "first_pulse_spikes": float(first.mean()),
            "last_over_first": decrement,
            "novel_over_first": novel_rel,
            "retest_over_first": retest_rel,
            "specific": bool(novel_rel - decrement > 0.15),
        }
        log(f"  {k:<12} first={first.mean():>9.0f} spikes  last/first={decrement:6.3f}  "
            f"novel/first={novel_rel:6.3f}  retest/first={retest_rel:6.3f}"
            f"  {'SPECIFIC' if out['populations'][k]['specific'] else 'not specific'}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
