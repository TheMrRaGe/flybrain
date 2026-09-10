#!/usr/bin/env python3
"""
mirror_connectome.py - use the fly's own bilateral symmetry as a second measurement.

    python3 mirror_connectome.py --in brain_whole.npz --out brain_mirrored.npz

WHY
    A real fly is bilaterally symmetric. This specimen's reconstruction is not:
    the right hemisphere carries 9.7% more synaptic weight than the left on only
    1.7% more neurons. That gap is tracing coverage, not biology.

    It matters because behaviour is decoded from left/right differences. Uncorrected,
    an identical stimulus to both antennae drives the right side harder and the
    creature turns right on literally every tick of a run. Scaling total weight per
    hemisphere only fixed the aggregate (bias +0.120 -> +0.047) because the shortfall
    is not uniform - it is scattered connection by connection.

HOW
    Every connection is reduced to a canonical form:

        (presynaptic cell type, postsynaptic cell type, same side or crossing)

    Each canonical form is observed TWICE in a symmetric animal - once originating
    from the left, once from the right. Where those two disagree, the larger is
    treated as the better-resolved measurement and both sides are brought up to it.
    Where one side saw a connection the other missed entirely, it is instantiated
    on the missing side across the same cell types.

    This does not invent connectivity. It uses the contralateral homologue as a
    replicate measurement of the same wiring, which is what bilateral symmetry
    means.

THE ONE EXCEPTION
    Kenyon cells. Their input wiring is genuinely random in each individual animal
    rather than stereotyped by cell type, so the left mushroom body is not a
    measurement of the right one. Mirroring there would invent structure instead of
    recovering it, so KC connections are passed through untouched (--mirror-kc to
    override). The asymmetry that remains is honest.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def log(m): print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="brain_whole.npz")
    ap.add_argument("--out", default="brain_mirrored.npz")
    ap.add_argument("--stat", choices=["max", "mean"], default="max",
                    help="max recovers under-traced connections; mean averages both "
                         "observations and is more conservative (default max)")
    ap.add_argument("--mirror-kc", action="store_true",
                    help="also mirror Kenyon cell connections (off by default - their "
                         "wiring is random per animal, not stereotyped)")
    a = ap.parse_args()

    d = np.load(a.inp, allow_pickle=False)
    ty, side, cls = d["type"].astype(str), d["side"].astype(str), d["cls"].astype(str)
    pre, post, w = d["pre"].astype(np.int64), d["post"].astype(np.int64), d["w"].astype(np.float64)
    N, E = len(ty), len(pre)
    log(f"[1/6] loaded {N:,} neurons, {E:,} connections from {a.inp}")

    inL0 = w[side[post] == "L"].sum(); inR0 = w[side[post] == "R"].sum()
    log(f"      synaptic weight  L {inL0:,.0f}  R {inR0:,.0f}  R/L {inR0/inL0:.4f}")

    log("[2/6] reducing every connection to its canonical form")
    tcode = pd.factorize(ty)[0].astype(np.int64)
    sp, sq = side[pre], side[post]
    lateral = np.where(sp == sq, 0, 1)                 # 0 = same side, 1 = crossing
    both = np.isin(sp, ["L", "R"]) & np.isin(sq, ["L", "R"])

    # a canonical connection is (pre type, post type, crossing?) - identical for the
    # left-origin and right-origin copies of the same wiring
    key = (tcode[pre] * len(ty) + tcode[post]) * 2 + lateral
    origin = np.where(sp == "R", 1, 0)                 # which copy we are looking at

    kc = cls == "Kenyon_Cell"
    # NB: ~a.mirror_kc on a Python bool is -1, not False. Using it in a mask
    # silently turns `usable` into an integer array, and key[usable] becomes
    # fancy indexing instead of boolean selection.
    protect = np.zeros(E, dtype=bool) if a.mirror_kc else (kc[pre] | kc[post])
    usable = both & ~protect
    assert usable.dtype == bool, "usable must stay a boolean mask"
    log(f"      {usable.sum():,} connections eligible "
        f"({(~both).sum():,} touch a midline neuron, {protect.sum():,} are Kenyon cell)")

    log("[3/6] pairing each canonical form with its mirror image")
    df = pd.DataFrame({"key": key[usable], "origin": origin[usable], "w": w[usable]})
    g = df.groupby(["key", "origin"], sort=False)["w"].sum().unstack(fill_value=0.0)
    for c in (0, 1):
        if c not in g.columns:
            g[c] = 0.0
    obsL, obsR = g[0].to_numpy(), g[1].to_numpy()
    target = np.maximum(obsL, obsR) if a.stat == "max" else (obsL + obsR) / 2.0

    seen_both = ((obsL > 0) & (obsR > 0)).sum()
    only_one = ((obsL > 0) ^ (obsR > 0)).sum()
    disagree = np.abs(obsL - obsR)[(obsL > 0) & (obsR > 0)]
    log(f"      {len(g):,} canonical connections: {seen_both:,} seen on both sides, "
        f"{only_one:,} on one side only")
    if len(disagree):
        rel = disagree / np.maximum(obsL, obsR)[(obsL > 0) & (obsR > 0)]
        log(f"      where both were seen, the two measurements differ by "
            f"{100*rel.mean():.1f}% on average")

    log(f"[4/6] bringing both copies up to the better measurement ({a.stat})")
    kidx = pd.Series(np.arange(len(g)), index=g.index)
    row = kidx.reindex(df["key"].to_numpy()).to_numpy()
    obs = np.where(df["origin"].to_numpy() == 1, obsR[row], obsL[row])
    scale = np.where(obs > 0, target[row] / np.maximum(obs, 1e-9), 0.0)

    w_new = w.copy()
    w_new[usable] = df["w"].to_numpy() * scale
    gained = w_new[usable].sum() - w[usable].sum()
    log(f"      existing connections scaled: {gained:+,.0f} synaptic weight recovered")

    log("[5/6] instantiating connections one side missed entirely")
    missing = np.where((obsL == 0) ^ (obsR == 0))[0]
    add_pre, add_post, add_w = [], [], []
    if len(missing):
        by_type_side: dict[tuple[int, str], np.ndarray] = {}
        for i in range(N):
            by_type_side.setdefault((tcode[i], side[i]), []).append(i)
        by_type_side = {k: np.array(v) for k, v in by_type_side.items()}
        keys = g.index.to_numpy()
        nt = len(ty)
        made = 0
        for r in missing:
            k = keys[r]
            lat = k % 2; rest = k // 2
            tpost = rest % nt; tpre = rest // nt
            have_R = obsR[r] > 0
            src_side = "L" if have_R else "R"           # the side that is missing it
            dst_side = src_side if lat == 0 else ("R" if src_side == "L" else "L")
            P = by_type_side.get((tpre, src_side)); Q = by_type_side.get((tpost, dst_side))
            if P is None or Q is None or not len(P) or not len(Q):
                continue                                 # no homologue to attach to
            n = max(len(P), len(Q))
            per = target[r] / n
            if per <= 0:
                continue
            for j in range(n):
                add_pre.append(int(P[j % len(P)])); add_post.append(int(Q[j % len(Q)]))
                add_w.append(per)
            made += 1
        log(f"      {made:,} of {len(missing):,} one-sided connections rebuilt "
            f"({len(add_pre):,} new edges)")
    else:
        log("      none missing")

    if add_pre:
        pre = np.concatenate([pre, np.array(add_pre, dtype=np.int64)])
        post = np.concatenate([post, np.array(add_post, dtype=np.int64)])
        w_new = np.concatenate([w_new, np.array(add_w)])

    inL1 = w_new[side[post] == "L"].sum(); inR1 = w_new[side[post] == "R"].sum()
    log(f"[6/6] writing {a.out}")
    log(f"      synaptic weight  L {inL1:,.0f}  R {inR1:,.0f}  R/L {inR1/inL1:.4f}   "
        f"(was {inR0/inL0:.4f})")
    log(f"      connections {E:,} -> {len(pre):,}")

    np.savez_compressed(
        a.out, bodyId=d["bodyId"], type=d["type"], cls=d["cls"], sc=d["sc"],
        nt=d["nt"], sign=d["sign"], side=d["side"],
        pre=pre.astype(np.int32), post=post.astype(np.int32),
        w=np.round(w_new).astype(np.int32),
    )


if __name__ == "__main__":
    main()
