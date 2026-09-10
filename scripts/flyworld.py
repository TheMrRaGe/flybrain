#!/usr/bin/env python3
"""
flyworld.py - a survival sandbox driven by a real fly connectome.

    python3 flyworld.py --brain brain_whole.npz --ticks 200 --out run.jsonl

ARCHITECTURE
    The brain runs at ~0.08x real time. A browser wants 60fps. Those cannot share a
    loop, so Python owns BOTH the world and the brain and steps them in lockstep with
    no wall-clock pressure at all. Every tick is written to a JSONL trajectory that
    the three.js viewer replays at whatever speed you like. Nothing is dropped, nothing
    is rushed, and you can scrub back to the moment the creature changed its mind.

THE SENSE -> BRAIN -> ACT LOOP
    world state  ->  receptor firing rates  ->  100 ms of spiking  ->  descending
    neurons  ->  body command  ->  world state

ACTION DECODING - the part where it is easiest to cheat
    A fly steers by firing its descending neurons asymmetrically: more on the right,
    it turns. That is real, and this dataset labels soma side for all 1,310 of them
    (653 L / 647 R / 10 midline). So steering here is population asymmetry, not a
    lookup table I invented:

        turn  = (right_rate - left_rate) / (right_rate + left_rate)
        speed = total descending drive, normalised

    Escape is the one named exception. DNp01 is the Giant Fiber, the best-characterised
    descending neuron in the animal, and it triggers the escape jump. It gets its own
    channel because its function is established, not guessed.

    GRAB is deliberately different. It is bound to a cluster of descending neurons
    that have no known role, and the fly has no grasping behaviour to inherit. If the
    creature ever learns to use it, that is the experiment: can a brain drive an
    effector it never evolved for?

WHAT THE CREATURE CANNOT DO
    There is no circuit for planning, caching, or deferred reward. The prediction is a
    competent foragerch that dies the moment survival needs a plan. Being wrong about
    that is the interesting outcome.
"""
from __future__ import annotations

import argparse, json, math, time
from dataclasses import dataclass, field, asdict
import numpy as np

from flysim import FlyBrain, Params


# --------------------------------------------------------------------------- #
#  world
# --------------------------------------------------------------------------- #

WORLD_R = 40.0          # metres, circular arena
TICK_MS = 200.0         # brain time simulated per world tick (a 1-neuron
                        # readout needs a longer window to be worth reading)

@dataclass
class Thing:
    kind: str           # food | water | wood | hazard | shelter
    x: float
    y: float
    amount: float = 1.0
    r: float = 1.6      # interaction radius

    def dist(self, x, y): return math.hypot(self.x - x, self.y - y)


# what each thing smells like: a set of glomerular channels, stable per kind
ODOR_OF = {
    "food":    "odor_food",
    "water":   "odor_water",
    "wood":    "odor_wood",
    "hazard":  "odor_hazard",
    "shelter": "odor_shelter",
}


class World:
    """A small survival arena. Resources deplete, hazards hurt, night is cold."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.t = 0
        self.things: list[Thing] = []
        for kind, n, amt in [("food", 7, 3.0), ("water", 4, 5.0),
                             ("wood", 6, 2.0), ("hazard", 3, 1.0), ("shelter", 2, 1.0)]:
            for _ in range(n):
                a, d = rng.uniform(0, 2 * math.pi), rng.uniform(6, WORLD_R * 0.92)
                self.things.append(Thing(kind, d * math.cos(a), d * math.sin(a), amt))

    @property
    def is_night(self) -> bool:
        return (self.t // 300) % 2 == 1        # 300 ticks of day, 300 of night

    def ambient_temp(self) -> float:
        return 14.0 if self.is_night else 26.0

    def odor_field(self, x, y) -> dict[str, float]:
        """Concentration by kind at a point. Inverse-square, as a real plume roughly is."""
        out: dict[str, float] = {}
        for th in self.things:
            if th.amount <= 0:
                continue
            d = max(th.dist(x, y), 0.5)
            c = min(1.0, (th.amount * 4.0) / (d * d))
            k = ODOR_OF[th.kind]
            out[k] = max(out.get(k, 0.0), c)
        return out

    def nearest(self, x, y, kind=None, within=None):
        best, bd = None, 1e9
        for th in self.things:
            if th.amount <= 0 or (kind and th.kind != kind):
                continue
            d = th.dist(x, y)
            if d < bd:
                best, bd = th, d
        if within is not None and bd > within:
            return None, bd
        return best, bd


# --------------------------------------------------------------------------- #
#  body
# --------------------------------------------------------------------------- #

@dataclass
class Body:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    energy: float = 1.0        # starves at 0
    water: float = 1.0
    warmth: float = 1.0        # freezes at 0
    alive: bool = True
    carrying: str | None = None
    built: int = 0             # shelters crafted - the thing it has no circuit for
    age: int = 0

    def drives(self) -> dict[str, float]:
        """Needs as levels in [0,1]. These become tonic current on endocrine neurons."""
        return {"hunger": 1.0 - self.energy,
                "thirst": 1.0 - self.water,
                "cold":   1.0 - self.warmth}


# --------------------------------------------------------------------------- #
#  the creature: senses in, descending neurons out
# --------------------------------------------------------------------------- #

class Creature:
    MAX_SPEED = 1.1            # metres per tick
    MAX_TURN = 0.9             # radians per tick
    TURN_GAIN = 3.0
    TURN_SMOOTH = 0.25         # how fast the turn command follows the readout           # scales evoked asymmetry into a usable turn range
    ADAPT = 0.18               # how fast the reference tracks conditions

    def __init__(self, brain: FlyBrain, body: Body | None = None):
        self.b = brain
        self.body = body or Body()

        dn = brain.pop["DN"]
        side = brain.side[dn]
        names = brain.type[dn].astype(str)
        self.dn = dn

        # STEERING READOUT - DNa02, not the population mean.
        # DNa02 is the best-characterised turning neuron in the fly: one cell per
        # side, and ipsilateral activity drives an ipsilateral turn. Averaging all
        # 1,310 descending neurons buries it under 1,308 cells doing other things.
        # Measured on the mirrored connectome, the DNa02 pair gives 13x more
        # directional signal than the population mean and a signal-to-bias ratio of
        # 3.8x against 0.4x.
        a02 = np.char.startswith(names, "DNa02")
        if a02.sum() >= 2:
            self.dn_L, self.dn_R = dn[a02 & (side == "L")], dn[a02 & (side == "R")]
            self.steer_source = "DNa02"
        else:
            self.dn_L, self.dn_R = dn[side == "L"], dn[side == "R"]
            self.steer_source = "DN population (DNa02 absent)"

        # total descending drive still sets speed - that is a population property
        self.dn_all_L, self.dn_all_R = dn[side == "L"], dn[side == "R"]

        # DNp01 - the Giant Fiber. Established function: escape.
        self.dn_escape = dn[np.char.startswith(names, "DNp01")]

        # GRAB: descending neurons with no established role, chosen by name so the
        # set is reproducible. The fly has no grasping behaviour; this is the
        # novel-effector channel.
        # novel-effector channel: descending neurons with no established behaviour.
        unnamed = dn[np.char.startswith(names, "CB") | np.char.startswith(names, "DNg")]
        self.dn_grab = unnamed if len(unnamed) >= 8 else dn[:32]

        for d in ("hunger", "thirst", "cold"):
            brain.define_drive(d, "endocrine")
        for k in set(ODOR_OF.values()):
            brain.define_odor(k, n_channels=20, seed=abs(hash(k)) % 9999)

        self.base_L = self.base_R = None   # driven descending rates at the reference point
        self.ref = 0.5                     # right-share that means 'no information'
        self._cal_tot = self._cal_ref = None   # bias-vs-drive calibration curve
        self._turn_s = 0.0                     # smoothed turn command

    def calibrate(self, levels=(0.0, 0.15, 0.4, 0.8), ms: float = 300.0) -> list:
        """
        Map the structural left/right bias as a function of how hard the brain is driven.

        Three attempts got here. Subtracting the resting rate failed: with no input
        the network is silent, 0.00 Hz both sides, so there is nothing to subtract.
        A single reference measured under one symmetric stimulus fixed the +5.55-point
        structural split but left -0.63 in an empty arena. An adaptive reference that
        tracked recent conditions removed the bias and the signal with it, because a
        standing creature in a fixed gradient sees a constant, and adaptation exists
        precisely to discard constants.

        What actually varies is this: the two halves respond to drive unequally, and
        by an amount that depends on the drive LEVEL. So the reference is measured at
        several symmetric intensities and interpolated. Whatever right-share comes out
        of a symmetric stimulus carries no information about the world, at any level,
        and that is the zero point.
        """
        keep_hz = self.b.drive_hz.copy(); keep_ext = self.b._ext.copy()
        curve = []
        for lv in levels:
            sym = {k: lv for k in set(ODOR_OF.values())}
            self.b.smell_bilateral(sym, sym)
            for d in ("hunger", "thirst", "cold"):
                self.b.set_drive(d, 0.5)
            self.b.reset(); self.b.run(100, record=False)
            r = self.b.run(ms); rates = r.rates
            L = float(rates[self.dn_L].mean()); R = float(rates[self.dn_R].mean())
            tot = L + R
            curve.append((tot, R / tot if tot > 1e-6 else 0.5))
        curve.sort()
        self._cal_tot = np.array([c[0] for c in curve], dtype=float)
        self._cal_ref = np.array([c[1] for c in curve], dtype=float)
        self.b.drive_hz[:] = keep_hz; self.b._ext[:] = keep_ext; self.b.reset()
        return curve

    def _ref_at(self, tot: float) -> float:
        """Interpolated no-information reference for this level of descending drive."""
        if self._cal_tot is None:
            self.calibrate()
        return float(np.interp(tot, self._cal_tot, self._cal_ref))

    # -- sensing ---------------------------------------------------------- #

    def sense(self, w: World) -> dict:
        bd = self.body
        # two antennae, 0.9 m either side of the midline
        ax, ay = math.cos(bd.heading), math.sin(bd.heading)
        lx, ly = bd.x - ay * 0.9, bd.y + ax * 0.9
        rx, ry = bd.x + ay * 0.9, bd.y - ax * 0.9
        left, right = w.odor_field(lx, ly), w.odor_field(rx, ry)
        self.b.smell_bilateral(left, right)
        odors = {k: max(left.get(k, 0), right.get(k, 0)) for k in set(left) | set(right)}

        for name, lvl in bd.drives().items():
            self.b.set_drive(name, lvl)

        # contact chemosensation: only when actually touching something
        th, d = w.nearest(bd.x, bd.y)
        if th is not None and d <= th.r:
            self.b.taste(+1.0 if th.kind in ("food", "water") else -1.0)
        else:
            self.b.taste(0.0)

        # thermosensation: 25 neurons, driven by how far ambient is from preference
        temp = w.ambient_temp()
        shelter, sd = w.nearest(bd.x, bd.y, "shelter")
        if shelter is not None and sd <= shelter.r * 2:
            temp += 6.0
        therm = self.b.pop.get("thermo", np.array([], int))
        if len(therm):
            self.b.drive_hz[therm] = 200.0 * min(1.0, abs(temp - 24.0) / 12.0)

        return {"odors": odors, "temp": temp, "touching": th.kind if (th and d <= th.r) else None}

    # -- acting ----------------------------------------------------------- #

    def decide(self, ms: float = TICK_MS) -> dict:
        if self._cal_tot is None:
            self.calibrate()

        r = self.b.run(ms)
        rates = r.rates

        L = float(rates[self.dn_L].mean())
        R = float(rates[self.dn_R].mean())
        tot = L + R

        # steering is the EVOKED asymmetry: how each side departs from its own
        # resting rate, not the raw difference between two unequal populations.
        # Steering is the departure of the left/right split from its OWN recent
        # average, not from a fixed reference.
        #
        # A fixed reference does not survive contact with a real world: calibrating
        # under a symmetric stimulus fixed the +5.55-point structural bias, but an
        # empty arena then read -0.63, because the split depends on how much total
        # stimulus there is and not only on how symmetric it is.
        #
        # An adaptive reference absorbs both. It also happens to be what the animal
        # does - flies track odour by comparing against a moment ago, not by trusting
        # an absolute left/right reading, because their antennae are under a
        # millimetre apart and that comparison is nearly useless on its own.
        ref = self._ref_at(tot)
        frac = R / tot if tot > 1e-6 else ref
        raw = (frac - ref) * self.TURN_GAIN

        # DNa02 is ONE neuron per side. Over a 100 ms window that is a couple of
        # spikes, so the instantaneous left/right ratio slams to the rails and the
        # creature spins: straightness fell to 0.08 with the raw signal. The pair
        # carries real direction information when integrated over seconds - it just
        # cannot be read tick by tick. A real fly's turn is not instantaneous either;
        # body inertia does this smoothing in the animal.
        self._turn_s += (raw - self._turn_s) * self.TURN_SMOOTH
        turn = float(np.clip(self._turn_s, -1.0, 1.0))
        pop = float(rates[self.dn_all_L].mean()) + float(rates[self.dn_all_R].mean())
        speed = min(1.0, pop / 60.0)                          # total descending drive
        escape = float(rates[self.dn_escape].mean()) if len(self.dn_escape) else 0.0
        grab = float(rates[self.dn_grab].mean()) if len(self.dn_grab) else 0.0

        return {"turn": turn, "speed": speed, "escape": escape, "grab": grab,
                "dn_L": L, "dn_R": R,
                "mbon": float(rates[self.b.pop["MBON"]].mean()),
                "kc": float(rates[self.b.pop["KC"]].mean()),
                "cx": float(rates[self.b.pop["CX"]].mean())}

    # -- physics and metabolism ------------------------------------------- #

    def act(self, w: World, cmd: dict) -> list[str]:
        bd, events = self.body, []
        if not bd.alive:
            return events

        jump = cmd["escape"] > 25.0
        bd.heading += cmd["turn"] * self.MAX_TURN * (2.0 if jump else 1.0)
        step = self.MAX_SPEED * (1.0 if jump else cmd["speed"])
        bd.x += math.cos(bd.heading) * step
        bd.y += math.sin(bd.heading) * step

        d = math.hypot(bd.x, bd.y)                    # arena wall
        if d > WORLD_R:
            bd.x *= WORLD_R / d; bd.y *= WORLD_R / d
            bd.heading += math.pi * 0.5
            events.append("wall")

        th, dist = w.nearest(bd.x, bd.y)
        if th is not None and dist <= th.r and th.amount > 0:
            if th.kind == "food":
                bd.energy = min(1.0, bd.energy + 0.30); th.amount -= 0.5; events.append("ate")
            elif th.kind == "water":
                bd.water = min(1.0, bd.water + 0.35); th.amount -= 0.4; events.append("drank")
            elif th.kind == "hazard":
                bd.energy -= 0.12; events.append("hurt")
            elif th.kind == "wood" and cmd["grab"] > 18.0 and bd.carrying is None:
                bd.carrying = "wood"; th.amount -= 1.0; events.append("grabbed")
            elif th.kind == "shelter":
                bd.warmth = min(1.0, bd.warmth + 0.25); events.append("sheltered")
                if bd.carrying == "wood":
                    bd.carrying = None; bd.built += 1; events.append("built")

        # metabolism
        cold = w.ambient_temp() < 18.0
        bd.energy -= 0.004 + 0.006 * cmd["speed"]
        bd.water  -= 0.005
        bd.warmth += (-0.010 if cold else 0.006)
        bd.energy = max(0.0, bd.energy); bd.water = max(0.0, bd.water)
        bd.warmth = float(np.clip(bd.warmth, 0.0, 1.0))
        bd.age += 1

        if bd.energy <= 0 or bd.water <= 0 or bd.warmth <= 0:
            bd.alive = False
            events.append("died:" + ("starved" if bd.energy <= 0 else
                                     "dehydrated" if bd.water <= 0 else "froze"))
        return events


# --------------------------------------------------------------------------- #
#  episode
# --------------------------------------------------------------------------- #

def run_episode(brain_path: str, ticks: int, out: str, seed: int = 0,
                gain: float = 0.15, log_every: int = 20) -> dict:
    t0 = time.perf_counter()
    brain = FlyBrain(brain_path, Params(gain=gain), seed=seed)
    print(f"brain: {brain}", flush=True)
    w = World(seed)
    c = Creature(brain)
    print(f"world: {len(w.things)} objects | steering from {len(c.dn_L)}L/{len(c.dn_R)}R "
          f"descending neurons | escape={len(c.dn_escape)} DNp01 | grab={len(c.dn_grab)} novel",
          flush=True)

    frames, tally = [], {}
    for t in range(ticks):
        w.t = t
        s = c.sense(w)
        cmd = c.decide()
        ev = c.act(w, cmd)
        for e in ev:
            tally[e] = tally.get(e, 0) + 1
        b = c.body
        frames.append({
            "t": t, "x": round(b.x, 3), "y": round(b.y, 3), "h": round(b.heading, 3),
            "e": round(b.energy, 3), "w": round(b.water, 3), "c": round(b.warmth, 3),
            "turn": round(cmd["turn"], 4), "speed": round(cmd["speed"], 3),
            "dnL": round(cmd["dn_L"], 1), "dnR": round(cmd["dn_R"], 1),
            "esc": round(cmd["escape"], 1), "grab": round(cmd["grab"], 1),
            "kc": round(cmd["kc"], 2), "mbon": round(cmd["mbon"], 1), "cx": round(cmd["cx"], 1),
            "night": w.is_night, "carry": b.carrying, "built": b.built,
            "ev": ev, "alive": b.alive,
        })
        if t % log_every == 0:
            print(f"  t={t:>4} pos=({b.x:>6.1f},{b.y:>6.1f}) e={b.energy:.2f} "
                  f"w={b.water:.2f} c={b.warmth:.2f} turn={cmd['turn']:+.2f} "
                  f"spd={cmd['speed']:.2f} {'/'.join(ev) if ev else ''}", flush=True)
        if not b.alive:
            print(f"  t={t}: {ev[-1]}", flush=True)
            break

    meta = {"ticks": len(frames), "seed": seed, "world_r": WORLD_R,
            "things": [asdict(th) for th in w.things],
            "events": tally, "alive": c.body.alive, "built": c.body.built,
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    with open(out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for fr in frames:
            f.write(json.dumps(fr) + "\n")
    print(f"\nwrote {out}: {len(frames)} ticks in {meta['wall_seconds']}s")
    print(f"events: {tally}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_whole.npz")
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--out", default="run.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float, default=0.15)
    a = ap.parse_args()
    run_episode(a.brain, a.ticks, a.out, a.seed, a.gain)
