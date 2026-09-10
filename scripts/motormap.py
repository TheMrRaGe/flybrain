#!/usr/bin/env python3
"""
motormap.py - individual limb and wing control, read from the real motor neurons.

    python3 motormap.py --brain brain_whole.npz --data data      # report what it finds

WHAT THIS REPLACES
    fly3d.py and desktop_fly.py drive the legs from a gait generator: a sine wave per
    leg with a fixed tripod phase offset. That is animation. The connectome contains
    the actual motor pool, and it was never being read.

WHAT IS ACTUALLY IN THE DATA
    `vnc_motor` holds 699 motor neurons in the simulation (708 in the annotations),
    and they are labelled BY MUSCLE, with `somaNeuromere` giving the segment and
    `exitNerve` confirming it:

        T1  173 MNs   prothoracic   front legs    (ProLN)
        T2  175 MNs   mesothoracic  middle legs + wings  (MesoLN)
        T3  152 MNs   metathoracic  hind legs     (MetaLN)
        A1-A10        abdominal - not legs

    Crossed with soma side that is six legs, each with named antagonist pairs:

        Ti flexor / Acc. ti flexor  vs  Ti extensor      femur-tibia joint
        Tr flexor / Acc. tr flexor  vs  Tr extensor      trochanter (leg levation)
        Sternal anterior rotator    vs  Sternal posterior rotator   coxa swing
        ltm / ltm1-tibia / ltm2-femur vs Ta depressor    tarsus

    and the wing power and steering muscles:

        DLMn a-f     dorsal longitudinal, the downstroke power muscles
        DVMn 1-3     dorsoventral, the upstroke power muscles
        b1-b3, hg1-4, iii1-4, ps1, tp1-2, MNwm*   steering muscles

    Those joints line up one-for-one with the DesktopFly leg chain (coxa yaw, femur,
    tibia, tarsus), so the brain's output can drive the model directly.

HOW A JOINT IS READ
    A joint angle is the BALANCE of its antagonists, not the rate of either:

        angle = (flexor rate - extensor rate) / (flexor rate + extensor rate + eps)

    which is bounded in [-1, 1], is zero when the pair is balanced, and does not move
    when the animal simply gets more excited overall. Same reasoning as the DNa
    steering readout - a difference, normalised by the total, so a global gain change
    cannot masquerade as a command.

HONEST LIMITS
    - There is no muscle and no limb inertia here. A motor-neuron balance is being read
      as a joint ANGLE, where a real leg integrates force against load. Treat it as
      "which way this joint is being pulled", not as physics.
    - Some muscle types carry no somaNeuromere and are dropped rather than guessed at.
    - Whether these pools produce a coordinated gait is a QUESTION, not an assumption.
      Run this file directly to see what they actually do.
"""
from __future__ import annotations

import argparse, json, os
import numpy as np

LEGS = ("T1", "T2", "T3")           # front, middle, hind
SIDES = ("L", "R")

# joint -> (muscles pulling one way, muscles pulling the other)
JOINTS = {
    "coxa": (   # protraction / retraction: the swing of the whole leg
        ["Sternal anterior rotator MN", "Tergopleural/Pleural promotor MN"],
        ["Sternal posterior rotator MN", "Pleural remotor/abductor MN"],
    ),
    "trochanter": (   # levation / depression: lifting the leg off the ground
        ["Tr flexor MN", "Acc. tr flexor MN"],
        ["Tr extensor MN", "Sternotrochanter MN", "Tergotr. MN"],
    ),
    "tibia": (   # the femur-tibia joint, the best-characterised leg joint in the fly
        ["Ti flexor MN", "Acc. ti flexor MN"],
        ["Ti extensor MN"],
    ),
    "tarsus": (
        ["ltm MN", "ltm1-tibia MN", "ltm2-femur MN"],
        ["Ta depressor MN"],
    ),
}

WING_POWER_DOWN = "DLMn"      # dorsal longitudinal - downstroke
WING_POWER_UP = "DVMn"        # dorsoventral - upstroke
WING_STEER_PREFIXES = ("MNwm", "b1", "b2", "b3", "hg", "iii", "ps1", "tp1", "tp2",
                       "i1", "i2", "tt")


def _load_neuromere(brain, data_dir):
    """somaNeuromere is not in the npz; join it on bodyId from the annotation table."""
    import pyarrow.feather as pf
    path = None
    for f in os.listdir(data_dir):
        if f.startswith("body-annotations") and f.endswith(".feather"):
            path = os.path.join(data_dir, f)
            break
    if path is None:
        raise FileNotFoundError(f"no body-annotations feather in {data_dir}")
    t = pf.read_table(path, columns=["bodyId", "somaNeuromere", "exitNerve"])
    bid = t.column("bodyId").to_numpy(zero_copy_only=False)
    nm = t.column("somaNeuromere").to_numpy(zero_copy_only=False).astype(str)
    order = np.argsort(bid)
    pos = np.searchsorted(bid[order], brain.bodyId)
    pos = np.clip(pos, 0, len(bid) - 1)
    idx = order[pos]
    ok = bid[idx] == brain.bodyId
    out = np.full(brain.N, "", dtype=object)
    out[ok] = nm[idx[ok]]
    return np.array([str(x) for x in out])


class MotorMap:
    """Reads six legs and two wings out of the motor pool."""

    def __init__(self, brain, data_dir="data"):
        self.b = brain
        self.neuromere = _load_neuromere(brain, data_dir)
        ty = brain.type.astype(str)
        side = brain.side.astype(str)
        sc = brain.sc.astype(str)
        motor = (sc == "vnc_motor")

        self.legs = {}
        for seg in LEGS:
            for sd in SIDES:
                base = motor & (self.neuromere == seg) & (side == sd)
                joints = {}
                for jname, (a_names, b_names) in JOINTS.items():
                    a = np.flatnonzero(base & np.isin(ty, a_names))
                    b = np.flatnonzero(base & np.isin(ty, b_names))
                    if len(a) or len(b):
                        joints[jname] = (a, b)
                if joints:
                    self.legs[(seg, sd)] = joints

        self.wings = {}
        for sd in SIDES:
            base = motor & (side == sd)
            down = np.flatnonzero(base & np.char.startswith(ty, WING_POWER_DOWN))
            up = np.flatnonzero(base & np.char.startswith(ty, WING_POWER_UP))
            steer = np.flatnonzero(base & np.array(
                [t.startswith(WING_STEER_PREFIXES) for t in ty]))
            if len(down) or len(up) or len(steer):
                self.wings[sd] = {"down": down, "up": up, "steer": steer}

        self.all_motor = np.flatnonzero(motor)

    # -- reading ---------------------------------------------------------- #

    @staticmethod
    def _balance(counts, a, b):
        ra = float(counts[a].sum()) / max(len(a), 1)
        rb = float(counts[b].sum()) / max(len(b), 1)
        tot = ra + rb
        return ((ra - rb) / tot) if tot > 1e-9 else 0.0, ra + rb

    def read(self, counts):
        """
        counts: spike counts per neuron over the window just simulated.
        Returns per-leg joint balances in [-1,1] and per-wing power/steer.
        """
        legs = {}
        for key, joints in self.legs.items():
            j = {}
            for jname, (a, b) in joints.items():
                bal, drive = self._balance(counts, a, b)
                j[jname] = bal
                j[jname + "_drive"] = drive
            legs["%s%s" % key] = j
        wings = {}
        for sd, w in self.wings.items():
            bal, power = self._balance(counts, w["down"], w["up"])
            steer = float(counts[w["steer"]].sum()) / max(len(w["steer"]), 1)
            wings[sd] = {"phase": bal, "power": power, "steer": steer}
        return {"legs": legs, "wings": wings}

    def summary(self):
        out = {"legs": {}, "wings": {}}
        for key, joints in self.legs.items():
            out["legs"]["%s%s" % key] = {j: [int(len(a)), int(len(b))]
                                         for j, (a, b) in joints.items()}
        for sd, w in self.wings.items():
            out["wings"][sd] = {k: int(len(v)) for k, v in w.items()}
        out["total_vnc_motor"] = int(len(self.all_motor))
        return out


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_whole.npz")
    ap.add_argument("--data", default="data")
    ap.add_argument("--ms", type=float, default=400.0)
    ap.add_argument("--std", action="store_true", default=True)
    ap.add_argument("--out", default="results/motormap.json")
    a = ap.parse_args()

    from flysim import FlyBrain, Params
    print(f"loading {a.brain} ...", flush=True)
    b = FlyBrain(a.brain, Params(gain=1.0), seed=0)
    if a.std:
        b.enable_std(None)
    b.enable_vision()
    mm = MotorMap(b, a.data)
    s = mm.summary()

    print(f"\nvnc_motor neurons: {s['total_vnc_motor']}")
    print(f"legs mapped: {len(mm.legs)} of 6\n")
    print(f"  {'leg':<6}" + "".join(f"{j:>14}" for j in JOINTS))
    for key in sorted(s["legs"]):
        row = s["legs"][key]
        cells = "".join(f"{str(row.get(j, [0,0])[0])+'/'+str(row.get(j,[0,0])[1]):>14}"
                        for j in JOINTS)
        print(f"  {key:<6}{cells}")
    print("\n  (cells shown as flexor/extensor pool sizes)")
    print("\nwings:")
    for sd, w in s["wings"].items():
        print(f"  {sd}: downstroke {w['down']:>3}  upstroke {w['up']:>3}  "
              f"steering {w['steer']:>3}")

    # what do they actually do when the animal is driven?
    print(f"\ndriving mechanosensory at 180 Hz for {a.ms:.0f} ms ...")
    b.reset()
    b.see(0.35)
    b.drive_hz[b.pop["mechano"]] = 180.0
    counts = np.zeros(b.N, dtype=np.int64)
    for _ in range(int(a.ms / b.p.dt)):
        counts += b.step()
    r = mm.read(counts)
    tot = int(counts[mm.all_motor].sum())
    print(f"  motor spikes in window: {tot}")
    print(f"\n  {'leg':<6}" + "".join(f"{j:>12}" for j in JOINTS))
    for k in sorted(r["legs"]):
        print(f"  {k:<6}" + "".join(f"{r['legs'][k].get(j,0.0):>+12.3f}" for j in JOINTS))
    print("\n  wing        phase       power      steer")
    for sd, w in r["wings"].items():
        print(f"  {sd:<6}{w['phase']:>+12.3f}{w['power']:>12.2f}{w['steer']:>11.2f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": s, "sample_read": r, "motor_spikes": tot}, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
