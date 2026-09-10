#!/usr/bin/env python3
"""
al_selectivity.py - does the antennal lobe keep glomerular identity, and under which
local-neuron sign assignment?

    python3 al_selectivity.py --odours ../results/odours.json

THE MEASUREMENT THAT MOTIVATED THIS
    Stimulating ONE receptor type (ORN_DM6) drove projection neurons in 45 of 53
    glomeruli above 5 Hz, twelve of them at 250 Hz (saturation). Only 7% of
    uniglomerular PN spikes were in DM6 itself. The PN code carried no glomerular
    identity, so the Kenyon code could not either - which is the root of every
    non-specific conditioning result.

    96% of the excitatory drive reaching an unstimulated glomerulus's PNs came from
    antennal-lobe local neurons, lLN1_bc alone 45%. The transmitter table calls
    lLN1 acetylcholine (42 of 59 cells) and splits lLN2 44 GABA / 40 acetylcholine.
    A cell type is transmitter-homogeneous; a 50/50 split inside one type is the
    classifier saying it does not know. Physiologically, lateral interaction in the
    fly antennal lobe is predominantly GABAergic and divisive (Olsen & Wilson 2008).

    Rather than assert the fix, measure the variants against the known physiology.

READOUTS, per variant, DM6 alone / CS+ / CS-
    glomeruli >5 Hz      how far one input spreads
    own share            fraction of uniglomerular PN spikes in the STIMULATED glomeruli
    own PN rate          the stimulated glomeruli must still respond
    KC %, jaccard        the code the mushroom body sees, at the same apl_w
"""
from __future__ import annotations

import argparse, json, os, re, time
import numpy as np

from flysim import FlyBrain, Params


def log(m): print(m, flush=True)


def glom(t):
    m = re.match(r"^([A-Z]+[0-9]*[a-z]?)_(?:ad|l|v|il|vl|lv)?PN$", t)
    return m.group(1) if m else None


VARIANTS = {
    "as-is":        (),
    "lLN1+lLN2 inhibitory": (("lLN1", -1), ("lLN2", -1)),
    "all LN inhibitory":    (("lLN", -1), ("vLN", -1), ("v2LN", -1), ("il3LN", -1),
                             ("LN", -1)),
}


def present(b, odour, seed, settle_ms, ms):
    b.rng = np.random.default_rng(seed)
    if b._noise_pool is not None:
        b._noise_pool = (b.rng.standard_normal(b._noise_pool.size, dtype=np.float32)
                         * b.p.noise)
        b._noise_off = 0
    b.reset()
    b.smell({odour: 1.0})
    for _ in range(int(settle_ms / b.p.dt)):
        b.step()
    tot = np.zeros(b.N, dtype=np.float32)
    for _ in range(int(ms / b.p.dt)):
        tot += b.step()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--odours", default="../results/odours.json")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--kc-thresh", type=float, default=1.5)
    ap.add_argument("--apl-w", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--ms", type=float, default=600.0)
    ap.add_argument("--settle-ms", type=float, default=250.0)
    ap.add_argument("--out", default="../results/al_selectivity.json")
    a = ap.parse_args()

    od = json.load(open(a.odours))
    stim = {"DM6 alone": ["ORN_DM6"], "CS+": od["CS+"], "CS-": od["CS-"]}
    out = {"stimuli": stim, "variants": {}}
    for vname in [v.strip() for v in a.variants.split(",")]:
        ov = VARIANTS[vname]
        t0 = time.time()
        b = FlyBrain(a.brain, Params(gain=1.0, kc_thresh_scale=a.kc_thresh, apl_w=a.apl_w,
                                     noise=a.noise, sign_override=ov), seed=11)
        ty = b.type.astype(str)
        alpn = b.pop["ALPN"]
        pn_glom = {i: glom(ty[i]) for i in alpn}
        uni = np.array([i for i in alpn if pn_glom[i]])
        gl_of = np.array([pn_glom[i] for i in uni])
        gloms = sorted(set(gl_of))
        n_pn = {g: int((gl_of == g).sum()) for g in gloms}
        kc = b._kc
        log(f"\n== {vname}: {b.n_sign_overridden} cells re-signed ==")
        log(f"{'stimulus':<10} {'glom>5Hz':>9} {'own share':>9} {'own Hz':>7} {'other Hz':>9} "
            f"| {'KC act':>6} {'%KC':>5} {'KC spk':>7}")
        res = {}
        acts = {}
        for sname, types in stim.items():
            b._odor_map["_s"] = {t: 1.0 for t in types}
            tot = present(b, "_s", 1000, a.settle_ms, a.ms)
            own = set(t.replace("ORN_", "") for t in types)
            rate = {g: float(tot[uni][gl_of == g].sum()) / n_pn[g] / (a.ms / 1000)
                    for g in gloms}
            n_hot = sum(1 for g in gloms if rate[g] > 5)
            uni_spk = float(tot[uni].sum())
            own_spk = float(sum(tot[uni][gl_of == g].sum() for g in own if g in n_pn))
            own_hz = np.mean([rate[g] for g in own if g in rate])
            oth_hz = np.mean([rate[g] for g in gloms if g not in own])
            kact = tot[kc] > 0
            acts[sname] = kact
            res[sname] = {"glom_hot": n_hot, "own_share": own_spk / max(uni_spk, 1),
                          "own_hz": float(own_hz), "other_hz": float(oth_hz),
                          "kc_active": int(kact.sum()), "kc_spikes": float(tot[kc].sum()),
                          "rates": rate}
            log(f"{sname:<10} {n_hot:>6d}/53 {own_spk/max(uni_spk,1):>9.2f} {own_hz:>7.0f} "
                f"{oth_hz:>9.1f} | {kact.sum():>6d} {100*kact.mean():>5.1f} {tot[kc].sum():>7.0f}")
        p, m = acts["CS+"], acts["CS-"]
        jac = float((p & m).sum() / max((p | m).sum(), 1))
        f = (p.mean() + m.mean()) / 2
        indep = f / (2 - f) if f else 0.0
        s1 = acts["DM6 alone"]
        log(f"KC overlap CS+/CS-: {int((p & m).sum())} shared, jaccard {jac:.3f} "
            f"(independent sets: {indep:.3f}, ratio {jac/max(indep,1e-9):.1f}x)   "
            f"DM6-alone cells also in CS+: {int((s1 & p).sum())}/{int(s1.sum())}   "
            f"({time.time()-t0:.0f}s)")
        res["kc_jaccard"] = jac
        res["kc_jaccard_independent"] = indep
        res["n_resigned"] = b.n_sign_overridden
        out["variants"][vname] = res
        del b

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
