#!/usr/bin/env python3
"""
mbon_hold.py - measure what tonic drive does to the mushroom-body output layer.

    python3 mbon_hold.py --fracs 0,0.5,0.7,0.85,0.95

WHY
    With the antennal lobe and APL fixed, the principal MBONs of both taught
    compartments (MBON05/13/18/21) fire 0-4 spikes per odour; the readouts were carried
    by strays (MBON11, MBON09) getting ~2% of the compartment's dopamine. Each principal
    cell is silenced by REAL inhibition (LHCENT, APL, MBON09) that in life is balanced
    by tonic drive the 0 Hz-rest model does not have. Decision 14 adds that drive as
    `mbon_hold_frac` x threshold. This script says what value to use.

WHAT IT MEASURES, per hold fraction
    no odour   spontaneous MBON rate (real MBONs: ~5-20 Hz), DNa spikes
    CS+ / CS-  spikes per principal MBON, number of MBON types responding, core-
               compartment discrimination (w >= 0.2 members only), DNa spikes
    The target is the principal MBONs responding to odours with a resting rate that
    is not a seizure, and the core compartments giving a non-degenerate readout.
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)

PRINCIPAL = ("MBON05", "MBON13", "MBON18", "MBON21", "MBON11", "MBON09")


def present(b, odour, seed, settle_ms, ms):
    b.rng = np.random.default_rng(seed)
    if b._noise_pool is not None:
        b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                         * b.p.noise)
        b._noise_off = 0
    b.reset()
    if odour:
        b.smell({odour: 1.0})
    else:
        b.drive_hz[b.pop["olfactory"]] = 0.0
    for t in b._da_by_type:
        b._ext[b._da_by_type[t]["cells"]] = 0.0
    for _ in range(int(settle_ms / b.p.dt)):
        b.step()
    tot = np.zeros(b.N, dtype=np.float32)
    for _ in range(int(ms / b.p.dt)):
        tot += b.step()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--odours", default="../results/odours3.json")
    ap.add_argument("--fracs", default="0,0.5,0.7,0.85,0.95")
    ap.add_argument("--core", type=float, default=0.2, help="dopamine-weight cut for core members")
    ap.add_argument("--ms", type=float, default=800.0)
    ap.add_argument("--settle-ms", type=float, default=250.0)
    ap.add_argument("--out", default="../results/mbon_hold.json")
    a = ap.parse_args()
    od = json.load(open(a.odours))
    rows = []
    for frac in [float(x) for x in a.fracs.split(",")]:
        t0 = time.time()
        b = FlyBrain(a.brain, Params(gain=1.0, mbon_hold_frac=frac), seed=11)
        b.enable_plasticity(); b.enable_compartments(); b.plastic_on = False
        b._odor_map["CS+"] = {t: 1.0 for t in od["CS+"]}
        b._odor_map["CS-"] = {t: 1.0 for t in od["CS-"]}
        ty = b.type.astype(str); mbon = b.pop["MBON"]
        dn = b.pop["DN"]; dna = dn[np.char.startswith(ty[dn], "DNa")]
        core = {c: [i for i in mbon if b._da_by_type[c]["w"][i] >= a.core]
                for c in ("PPL105", "PAM08")}
        res = {"frac": frac}
        log(f"\n== mbon_hold_frac {frac} ==")
        for o in (None, "CS+", "CS-"):
            tot = present(b, o, 1000, a.settle_ms, a.ms)
            name = o or "rest"
            m_spk = float(tot[mbon].sum())
            n_types = len(set(ty[mbon][tot[mbon] > 0]))
            per = {p: int(sum(tot[i] for i in mbon if ty[i] == p)) for p in PRINCIPAL}
            core_spk = {c: int(tot[idx].sum()) for c, idx in core.items()}
            res[name] = {"mbon_spikes": m_spk, "mbon_hz_per_cell": m_spk / len(mbon) / (a.ms / 1000),
                         "types_active": n_types, "principal": per, "core": core_spk,
                         "dna_spikes": float(tot[dna].sum()), "kc_active": int((tot[b._kc] > 0).sum()),
                         "brain_spikes": float(tot.sum())}
            log(f"  {name:<5} MBON {m_spk:6.0f} spk ({m_spk/len(mbon)/(a.ms/1000):5.1f} Hz/cell, "
                f"{n_types:2d} types)  principal {per}  core A {core_spk['PPL105']:4d} "
                f"P {core_spk['PAM08']:4d}   DNa {tot[dna].sum():5.0f}  KC act {(tot[b._kc]>0).sum():4d}  "
                f"brain {tot.sum():7.0f}")
        for c in core:
            p_, m_ = res["CS+"]["core"][c], res["CS-"]["core"][c]
            res[f"d_core_{c}"] = (p_ - m_) / (p_ + m_) if p_ + m_ else None
        log(f"  core d: PPL105 {res['d_core_PPL105']}   PAM08 {res['d_core_PAM08']}   ({time.time()-t0:.0f}s)")
        rows.append(res)
        del b
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"core_cut": a.core, "rows": rows}, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
