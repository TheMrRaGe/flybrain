#!/usr/bin/env python3
"""
eye.py - a compound eye at the fly's own resolution, from the connectome.

    from eye import Eye
    eye = Eye(brain)                       # brain: a FlyBrain (CPU) with the graph loaded
    rates = eye.render(objects, heading, ambient)   # Hz per photoreceptor

WHERE THE RETINA COMES FROM
    The photoreceptors carry no coordinates in the annotations, but the lamina
    cartridges they synapse into do: every L1 cell has `assignedOlHex1/2`, a hex-grid
    column index (1-36 x 1-39, ~880 columns per eye). 1,352 of the 1,394 R1-R6
    photoreceptors in the graph have an L1 target, so each photoreceptor's place on the
    retina is the column its own axon goes to - read from the wiring, not assigned.
    R7/R8 (colour/polarisation) are left at baseline: the motion and looming pathway
    is R1-R6 -> L1/L2 -> T4/T5 / LPLC2.

THE ONE CHOICE
    Which hex axis is front-to-back and which is up-and-down is not in the data. Here
    hex1 -> azimuth (front 0 deg to back 170 deg on that eye's side), hex2 -> elevation
    (-60..+60 deg). A wrong orientation rotates the retina; looming (expansion anywhere)
    and left/right brightness are unaffected, optic-flow direction is not. Labelled.

RESOLUTION
    ~880 pixels per eye, ~5 deg each. A person at 3 tiles is a dark blob 20 deg wide;
    at 12 tiles it is one pixel. That is what a fly sees.

LIGHT - corrected 10 Sept 2026
    Insect photoreceptors DEPOLARISE to light and release MORE histamine, which
    hyperpolarises the lamina cells L1-L3 (OFF cells); light off abolishes release and
    excites them (Hardie 1989; Frontiers Neural Circuits 2016). The earlier vision fix in
    FINDINGS had this inverted, and with it the ON pathway (L1 -> Mi1 -> T4) was crushed
    by tonically firing L1. Rate = photoreceptor_hz * brightness.
"""
from __future__ import annotations

import math
import numpy as np
import pyarrow.feather as pf


class Eye:
    def __init__(self, brain, data_dir="../data"):
        b = brain
        ty = b.type.astype(str)
        ann = pf.read_table(f"{data_dir}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                            columns=["bodyId", "assignedOlHex1", "assignedOlHex2"]).to_pandas()
        ann = ann[ann["assignedOlHex1"].notna()].set_index("bodyId")
        hex_of = {int(k): (float(r.assignedOlHex1), float(r.assignedOlHex2)) for k, r in ann.iterrows()}
        raw = np.load(b._path)
        pre, post = raw["pre"], raw["post"]
        isR = np.flatnonzero(ty == "R1-R6")
        isL1 = ty == "L1"
        m = np.isin(pre, isR) & isL1[post]
        # each photoreceptor -> the L1 it contacts most (first edge is enough: one cartridge)
        target = {}
        for p, q in zip(pre[m], post[m]):
            target.setdefault(int(p), int(q))
        recs, az, el, side = [], [], [], []
        for p, q in target.items():
            h = hex_of.get(int(b.bodyId[q]))
            if h is None:
                continue
            recs.append(p); side.append(str(b.side[p]) if str(b.side[p]) in ("L", "R") else str(b.side[q]))
            az.append(h[0]); el.append(h[1])
        self.idx = np.array(recs)
        az, el = np.array(az), np.array(el)
        sd = np.array(side)
        # normalise hex axes to angles; the right eye looks right (0..170 deg), the left mirrors
        a = (az - az.min()) / max(az.max() - az.min(), 1) * math.radians(170.0)
        e = ((el - el.min()) / max(el.max() - el.min(), 1) - 0.5) * math.radians(120.0)
        self.az = np.where(sd == "R", a, -a)          # radians, relative to the body axis, + = right
        self.el = e
        self.side = sd
        self.base_hz = b.p.photoreceptor_hz
        self.n = len(self.idx)
        self.per_eye = {s: int((sd == s).sum()) for s in ("L", "R")}

    def render(self, objects, heading, ambient):
        """objects: list of (bearing_world_rad, distance_tiles, size_tiles, brightness 0..1).
        Returns Hz per photoreceptor (in self.idx order)."""
        bright = np.full(self.n, float(ambient))
        for (bearing, d, size, lum) in objects:
            d = max(d, 0.3)
            half = math.atan2(size * 0.5, d)               # angular half-width
            if half < math.radians(2.0):                   # smaller than a pixel: invisible
                continue
            rel = (bearing - heading + math.pi) % (2 * math.pi) - math.pi   # -pi..pi, + = right
            dz = np.abs((self.az - rel + math.pi) % (2 * math.pi) - math.pi)
            hit = (dz < half) & (np.abs(self.el) < half)
            bright[hit] = lum
        return self.base_hz * np.clip(bright, 0.0, 1.0)
