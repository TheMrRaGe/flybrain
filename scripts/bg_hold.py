#!/usr/bin/env python3
"""
bg_hold.py - measure a low-rate spontaneous regime before adopting it (decision 15).

    python3 bg_hold.py --fracs 0,0.3,0.5,0.6,0.7

WHY
    Four stages in a row - lamina, Kenyon cells, MBONs, the MB -> CX/LAL layer - were
    silent under the 0 Hz-rest convention until given the tonic drive they have in
    life. `bg_hold_frac` gives it to every central neuron at once. Before that
    replaces Shiu's convention, this says what it does.

WHAT IT MEASURES, per fraction
    rest    brain rate (Hz/neuron), classes with spontaneous activity, DN rate, MBON
            rate, whether it is stable over 1.5 s (seizure = rate climbing)
    CS+/CS- KC sparsity and overlap (must survive), MBON core responses, DN
            response and the CS+ vs CS- difference at the DNs, and the fraction of
            the core MBONs' targets that are now alive
"""
from __future__ import annotations

import argparse, json, os, time, collections
import numpy as np

from flysim import FlyBrain, Params
from conditioning4 import compartments


def log(m): print(m, flush=True)


def run(b, odour, seed, ms, settle_ms=250, halves=False):
    b.rng = np.random.default_rng(seed)
    if b._noise_pool is not None:
        b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                         * b.p.noise)
        b._noise_off = 0
    b.reset()
    b.drive_hz[b.pop["olfactory"]] = 0.0
    if odour:
        b.smell({odour: 1.0})
    for t in b._da_by_type:
        b._ext[b._da_by_type[t]["cells"]] = 0.0
    for _ in range(int(settle_ms / b.p.dt)):
        b.step()
    n = int(ms / b.p.dt)
    tot = np.zeros(b.N, dtype=np.float32)
    first = second = 0.0
    for i in range(n):
        s = b.step()
        tot += s
        if i < n // 2: first += s.sum()
        else: second += s.sum()
    return tot, first, second


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--odours", default="../results/odours3.json")
    ap.add_argument("--fracs", default="0,0.3,0.5,0.6,0.7")
    ap.add_argument("--syn-sat", default="0", help="syn_sat_k values (candidate 1)")
    ap.add_argument("--ms", type=float, default=800.0)
    ap.add_argument("--out", default="../results/bg_hold.json")
    a = ap.parse_args()
    od = json.load(open(a.odours))
    rows = []
    for frac in [float(x) for x in a.fracs.split(",")]:
      for sat in [float(x) for x in a.syn_sat.split(",")]:
        t0 = time.time()
        b = FlyBrain(a.brain, Params(gain=1.0, bg_hold_frac=frac, syn_sat_k=sat), seed=11)
        b.enable_plasticity(); b.enable_compartments(); b.plastic_on = False
        for d in b._da_by_type.values():
            d["w"] = np.where(d["w"] >= 0.2, d["w"], 0.0).astype(np.float32)
        b._odor_map["CS+"] = {t: 1.0 for t in od["CS+"]}
        b._odor_map["CS-"] = {t: 1.0 for t in od["CS-"]}
        cls = b.cls.astype(str); ty = b.type.astype(str)
        dn = b.pop["DN"]; mbon = b.pop["MBON"]; kc = b._kc
        ro_a, ro_p, _ = compartments(b, "PPL105", "PAM08")
        # 1-hop targets of the core MBONs
        tg = set()
        for i in np.concatenate([ro_a, ro_p]):
            tg.update(b._out_tgt[b._out_ptr[i]:b._out_ptr[i + 1]].tolist())
        tg = np.array(sorted(tg))
        res = {"frac": frac}
        log(f"\n== bg_hold_frac {frac} ==")
        # rest, 1.6 s for stability
        tot, f1, f2 = run(b, None, 1000, 1600)
        sec = 1.6
        by_cls = collections.Counter()
        for c in set(cls):
            m = cls == c
            if m.sum() >= 20:
                by_cls[c] = float(tot[m].sum() / m.sum() / sec)
        active_cls = sorted([(c, r) for c, r in by_cls.items() if r > 0.5], key=lambda x: -x[1])
        res["rest"] = {"hz_per_neuron": float(tot.sum() / b.N / sec),
                       "second_half_over_first": float(f2 / max(f1, 1)),
                       "dn_hz": float(tot[dn].sum() / len(dn) / sec),
                       "mbon_hz": float(tot[mbon].sum() / len(mbon) / sec),
                       "kc_active": int((tot[kc] > 0).sum()),
                       "classes_over_0.5hz": active_cls[:12]}
        log(f"  rest: {tot.sum()/b.N/sec:6.2f} Hz/neuron  2nd/1st half {f2/max(f1,1):.2f}  "
            f"DN {tot[dn].sum()/len(dn)/sec:.1f} Hz  MBON {tot[mbon].sum()/len(mbon)/sec:.1f} Hz  "
            f"KC active {(tot[kc]>0).sum()}")
        log(f"        spontaneous classes: {[(c, round(r,1)) for c, r in active_cls[:10]]}")
        acts = {}
        for o in ("CS+", "CS-"):
            tot, f1, f2 = run(b, o, 1000, a.ms)
            acts[o] = tot
            sec = a.ms / 1000
            res[o] = {"hz_per_neuron": float(tot.sum() / b.N / sec),
                      "kc_active": int((tot[kc] > 0).sum()),
                      "core_A": int(tot[ro_a].sum()), "core_P": int(tot[ro_p].sum()),
                      "dn_spikes": float(tot[dn].sum()),
                      "targets_alive": float((tot[tg] > 0).mean())}
            log(f"  {o}: {tot.sum()/b.N/sec:5.2f} Hz/neuron  KC active {(tot[kc]>0).sum():4d}  "
                f"core A {tot[ro_a].sum():4.0f} P {tot[ro_p].sum():4.0f}  DN {tot[dn].sum():6.0f}  "
                f"MBON-target cells alive {100*(tot[tg]>0).mean():.0f}%")
        p_, m_ = acts["CS+"][kc] > 0, acts["CS-"][kc] > 0
        jac = float((p_ & m_).sum() / max((p_ | m_).sum(), 1))
        dS = acts["CS+"][dn] - acts["CS-"][dn]
        res["kc_jaccard"] = jac
        res["dn_diff_l1"] = float(np.abs(dS).sum())
        res["dn_diff_cells"] = int((np.abs(dS) > 5).sum())
        log(f"  KC jaccard {jac:.3f}   DNs differing by >5 spikes between odours: "
            f"{(np.abs(dS)>5).sum()}/{len(dn)}   ({time.time()-t0:.0f}s)")
        rows.append(res)
        del b
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
