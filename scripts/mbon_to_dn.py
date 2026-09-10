#!/usr/bin/env python3
"""
mbon_to_dn.py - the influence map: does driving an MBON move the descending neurons?

    python3 mbon_to_dn.py

The learned signal reverses across the MBON layer but arrives at the DNs with no
sign. Before opening intermediate layers, ask the direct question: if a core MBON is
driven on its own at a physiological rate, do any DNs change? With and without an
odour present, because the intermediates may need the odour's drive to be alive.

Per MBON type: tonic current sized to fire it at ~`hz`; count DN spikes vs the same
condition with the MBON quiet; report the DN types that move, both hemispheres.
Paired seed, so the difference is exact.
"""
from __future__ import annotations

import argparse, json, os, time, collections
import numpy as np

from types import SimpleNamespace
from conditioning4 import build, reseed, quiet


def log(m): print(m, flush=True)


def run(b, odour, seed, drive_idx, mv, ms, settle_ms):
    reseed(b, seed)
    b.reset()
    b.drive_hz[b.pop["olfactory"]] = 0.0
    if odour:
        b.smell({odour: 1.0})
    quiet(b)
    saved = b._ext[drive_idx].copy() if len(drive_idx) else None
    if len(drive_idx):
        b._ext[drive_idx] = mv
    for _ in range(int(settle_ms / b.p.dt)):
        b.step()
    tot = np.zeros(b.N, dtype=np.float32)
    for _ in range(int(ms / b.p.dt)):
        tot += b.step()
    if len(drive_idx):
        b._ext[drive_idx] = saved
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--types", default="MBON05,MBON13,MBON21,MBON18,MBON09,MBON11,MBON30")
    ap.add_argument("--mv", type=float, default=20.0, help="tonic drive on the MBON")
    ap.add_argument("--ms", type=float, default=800.0)
    ap.add_argument("--settle-ms", type=float, default=250.0)
    ap.add_argument("--odours", default="../results/odours3.json")
    ap.add_argument("--out", default="../results/mbon_to_dn.json")
    a = ap.parse_args()
    args = SimpleNamespace(rate=0.0003, kc_thresh=1.5, apl_scale=0.1, mbon_hold=0.85, bg_hold=0.0,
                           noise=0.15, odours=a.odours, swap=False, channels=5, core=0.2)
    b = build(a.brain, args)
    b.plastic_on = False
    ty = b.type.astype(str); dn = b.pop["DN"]; tdn = ty[dn]; sdn = b.side[dn]
    out = {}
    for odour in (None, "CS+"):
        base = run(b, odour, 1000, np.zeros(0, int), 0.0, a.ms, a.settle_ms)
        log(f"\n== odour {odour or 'none'}: baseline DN spikes {base[dn].sum():.0f}, MBON spikes {base[b.pop['MBON']].sum():.0f}")
        for t in a.types.split(","):
            idx = np.flatnonzero(ty == t)
            if not len(idx):
                continue
            t0 = time.time()
            tot = run(b, odour, 1000, idx, a.mv, a.ms, a.settle_ms)
            m_hz = tot[idx].sum() / len(idx) / (a.ms / 1000)
            d = tot[dn] - base[dn]
            n_moved = int((np.abs(d) > 3).sum())
            by_type = collections.defaultdict(float)
            for k, x in zip(tdn, d):
                by_type[k] += x
            top = sorted(by_type.items(), key=lambda kv: -abs(kv[1]))[:6]
            log(f"  {t:<7} driven to {m_hz:5.0f} Hz/cell ({len(idx)} cells): DN total {d.sum():+7.0f} "
                f"spikes, |d|>3 in {n_moved:3d}/{len(dn)} DNs, brain {tot.sum()-base.sum():+8.0f}   "
                f"top: {[(k, round(v)) for k, v in top]}   ({time.time()-t0:.0f}s)")
            out[f"{odour or 'none'}|{t}"] = {"mbon_hz": float(m_hz), "dn_total": float(d.sum()),
                                             "n_moved": n_moved, "top": top,
                                             "brain_delta": float(tot.sum() - base.sum())}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
