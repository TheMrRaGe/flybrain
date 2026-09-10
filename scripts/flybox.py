#!/usr/bin/env python3
"""
flybox.py - a continuously running survival world with a brain that learns in it.

    python3 flybox.py                       # runs until Ctrl+C
    python3 -m http.server 8000             # from the project root, in another shell
    -> http://localhost:8000/web/flybox.html   live view

THE BOX
    A circular arena with food, water, heat hazards and shelter. Day and night. The
    fly has energy, water and warmth; all three drain, faster when walking; any at
    zero is death. Food and water are on the ground and deplete and regrow. Hazards
    burn. Night is cold; shelter is warm.

    Death is a consequence: the body is replaced, the world keeps going, and by
    default the BRAIN PERSISTS - its learned KC->MBON weights carry into the next
    life. Lifespan across lives is the first metric. `--no-inherit` gives each life a
    naive brain, which is the control.

THE WORLD IS THE TEACHER
    Nothing is pre-trained. Reinforcement is delivered by contact with the world:
        eating or drinking   -> PAM08 driven while it happens   (reward channel)
        touching a hazard    -> PPL105 driven while it happens  (punishment channel)
    with plasticity on throughout. Each kind of thing has its own odour, a disjoint
    set of glomeruli, so the association the mushroom body can form is
    odour -> outcome. Whether it forms it, and whether that changes where the fly
    goes, is what the box measures.

WHAT IS THE CONNECTOME'S
    turn      calibrated DNa left/right asymmetry (fly3d.py's method)
    speed     total leg motor-neuron rate (T1-T3), the measured walking axis
    stop      leg drive below a floor -> stationary; eating/drinking need it
    escape    DNp01, the Giant Fiber
    learning  KC->MBON depression under compartment dopamine (decisions 12-14)
    Imposed: the plume model, the metabolism, the arena, the mapping rate -> speed.

REGIME
    The one where the mushroom body can move the action bus: kc_thresh 1.0, odours at
    <= 0.35, MBON hold 0.85, core compartments, learn_rate 0.0003. Measured, not tuned.
"""
from __future__ import annotations

import argparse, json, math, os, signal, time
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

from conditioning4 import build
from motormap import _load_neuromere


def log(m): print(m, flush=True)


# --------------------------------------------------------------------------- #
#  world
# --------------------------------------------------------------------------- #

WORLD_R = 40.0
TICK_MS = 150.0
DAY_TICKS = 400            # 60 s of brain time per half-day


@dataclass
class Thing:
    kind: str
    x: float
    y: float
    amount: float
    r: float = 2.0
    max_amount: float = 1.0

    def dist(self, x, y): return math.hypot(self.x - x, self.y - y)


class World:
    KINDS = {"food": (6, 3.0), "water": (4, 4.0), "hazard": (3, 1.0), "shelter": (2, 1.0)}

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.things = []
        for kind, (n, amt) in self.KINDS.items():
            for _ in range(n):
                a, d = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(8, WORLD_R * 0.9)
                self.things.append(Thing(kind, d * math.cos(a), d * math.sin(a), amt, 2.0, amt))

    @property
    def is_night(self): return (self.t // DAY_TICKS) % 2 == 1

    def ambient(self): return 14.0 if self.is_night else 26.0

    def odour_field(self, x, y):
        """Concentration 0..1 per kind at a point: inverse-square plume, capped."""
        out = {}
        for th in self.things:
            if th.amount <= 0:
                continue
            d = max(th.dist(x, y), 1.0)
            c = min(1.0, 3.0 * th.amount / th.max_amount / (d * d) * 4.0)
            out[th.kind] = max(out.get(th.kind, 0.0), c)
        return out

    def heat_at(self, x, y):
        h = 0.0
        for th in self.things:
            if th.kind == "hazard":
                h = max(h, max(0.0, 1.0 - th.dist(x, y) / 5.0))
        return h

    def nearest(self, x, y, kind=None):
        best, bd = None, 1e9
        for th in self.things:
            if th.amount <= 0 or (kind and th.kind != kind):
                continue
            d = th.dist(x, y)
            if d < bd:
                best, bd = th, d
        return best, bd

    def regrow(self):
        for th in self.things:
            if th.kind in ("food", "water") and th.amount < th.max_amount:
                th.amount = min(th.max_amount, th.amount + 0.002 * th.max_amount)
        self.t += 1


# --------------------------------------------------------------------------- #
#  body
# --------------------------------------------------------------------------- #

@dataclass
class Body:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    energy: float = 1.0
    water: float = 1.0
    warmth: float = 1.0
    age: int = 0
    alive: bool = True
    speed: float = 0.0
    stopped: bool = False


# --------------------------------------------------------------------------- #
#  brain wrapper
# --------------------------------------------------------------------------- #

class BoxBrain:
    def __init__(self, path, seed, kc_thresh, strength, mbon_hold, rate, data_dir, odours):
        args = SimpleNamespace(rate=rate, kc_thresh=kc_thresh, apl_scale=0.1,
                               mbon_hold=mbon_hold, bg_hold=0.0, noise=0.15,
                               odours=odours, swap=False, channels=5, core=0.2)
        self.b = b = build(path, args, seed=seed)
        b.plastic_on = True
        self.strength = strength
        # four disjoint odours from the two designed 8-glomerulus sets
        od = json.load(open(odours))
        types = list(od["CS+"]) + list(od["CS-"])
        self.odours = {"food": types[0:4], "water": types[4:8],
                       "hazard": types[8:12], "shelter": types[12:16]}
        for k, ts in self.odours.items():
            b._odor_map[k] = {t: 1.0 for t in ts}
        dn = b.pop["DN"]; names = b.type[dn].astype(str); side = b.side[dn]
        fam = np.char.startswith(names, "DNa")
        self.L, self.R = dn[fam & (side == "L")], dn[fam & (side == "R")]
        self.escape = dn[np.char.startswith(names, "DNp01")]
        mech = b.pop["mechano"]
        self.mL, self.mR = mech[b.side[mech] == "L"], mech[b.side[mech] == "R"]
        ty = b.type.astype(str); sc = b.sc.astype(str)
        nm = _load_neuromere(b, data_dir)
        motor = sc == "vnc_motor"
        wingish = (np.char.startswith(ty, "DLMn") | np.char.startswith(ty, "DVMn")
                   | np.char.startswith(ty, "MNwm") | np.char.startswith(ty, "MNhm"))
        self.leg = np.flatnonzero(motor & np.isin(nm, ["T1", "T2", "T3"]) & ~wingish)
        self.thermo = b.pop.get("thermo", np.zeros(0, int))
        cm = b.pop["motor"]
        self.feed = cm[np.isin(ty[cm], b.FEEDING_MN)]     # proboscis: the feeding readout
        # HUNGER THROUGH ITS TARGETS. The peptide cells (IPC, DH44, LK, NPF, Hugin) are
        # 'unclear' in the transmitter table, sign 0, no outputs - they cannot speak in
        # a fast-synapse LIF. Two documented targets instead:
        #   PPL101 = PPL1-gamma1pedc (MB-MP1): active when fed, blocks appetitive memory
        #   expression; NPF silences it when hungry (Krashes et al. 2009, Cell).
        #   Sugar GRN gain roughly doubles with starvation via dopamine/DopEcR
        #   (Inagaki et al. 2012, Cell).
        self.ppl101 = b._da_by_type["PPL101"]["cells"]
        self.fed_level, self.taste_gain = 0.0, 1.0
        self.steps = int(round(TICK_MS / b.p.dt))
        self.base = None
        self.mbon = b.pop["MBON"]
        self._calibrate()

    def _run(self, hzL, hzR, smell, steps, reinforce=None, touch=None, temp_err=0.0):
        b = self.b
        b.drive_hz[:] = 0.0
        b.drive_hz[self.mL] = hzL
        b.drive_hz[self.mR] = hzR
        if smell:
            b.smell_bilateral(smell[0], smell[1])
        if touch is not None:
            b.taste(self.taste_gain if touch in ("food", "water") else -1.0)
        if len(self.thermo) and temp_err > 0:
            b.drive_hz[self.thermo] = 200.0 * min(1.0, temp_err / 12.0)
        for t in b._da_by_type:
            b._ext[b._da_by_type[t]["cells"]] = 0.0
        if reinforce:
            b.stimulate_type(reinforce, 180.0, 70.0)
        elif self.fed_level > 0:
            b._ext[self.ppl101] = 8.0 * self.fed_level          # MB-MP1 tonic when fed
        aL = aR = aLeg = aE = aM = aF = 0
        for _ in range(steps):
            spk = b.step()
            if b.plastic_on:
                b.learn()
            aL += int(spk[self.L].sum()); aR += int(spk[self.R].sum())
            aLeg += int(spk[self.leg].sum()); aE += int(spk[self.escape].sum())
            aM += int(spk[self.mbon].sum()); aF += int(spk[self.feed].sum())
        for t in b._da_by_type:
            b._ext[b._da_by_type[t]["cells"]] = 0.0
        return aL / max(len(self.L), 1), aR / max(len(self.R), 1), aLeg, aE, aM, aF

    def _calibrate(self):
        was, self.b.plastic_on = self.b.plastic_on, False
        cal = []
        for lv in (0.0, 0.3, 0.6, 1.0):
            self.b.reset()
            self._run(180.0 * lv, 180.0 * lv, None, int(120 / self.b.p.dt))
            L, R, *_ = self._run(180.0 * lv, 180.0 * lv, None, int(260 / self.b.p.dt))
            tot = L + R
            cal.append((tot, R / tot if tot > 1e-6 else 0.5))
        cal.sort()
        tots, refs = [], []
        for t, r in cal:
            if tots and t <= tots[-1]:
                refs[-1] = (refs[-1] + r) / 2.0; continue
            tots.append(t); refs.append(r)
        if len(tots) < 2:
            tots, refs = [0.0, 1.0], [0.5, 0.5]
        self.cal_tot, self.cal_ref = np.array(tots), np.array(refs)
        self.b.reset()
        self.b.plastic_on = was

    def weights(self):
        b = self.b
        return float(np.mean(b._out_w[b._plastic] / np.maximum(b._w0, 1e-9)))

    def tick(self, left, right, drives, temp_err, touch, reinforce):
        b = self.b
        hunger = drives["hunger"]
        self.fed_level = 1.0 - hunger
        self.taste_gain = 0.5 + 0.5 * hunger
        smell = ({k: self.strength * v for k, v in left.items()},
                 {k: self.strength * v for k, v in right.items()})
        L, R, leg, esc, mb, feed = self._run(0.0, 0.0, smell, self.steps, reinforce, touch, temp_err)
        tot = L + R
        ref = float(np.interp(tot, self.cal_tot, self.cal_ref))
        frac = (R / tot) if tot > 1e-6 else ref
        ev = frac - ref
        self.base = ev if self.base is None else self.base + (ev - self.base) * 0.02
        return {"turn": float(np.clip((ev - self.base) * 12.0, -1.0, 1.0)),
                "leg": int(leg), "escape": int(esc), "mbon": int(mb), "feed": int(feed)}


# --------------------------------------------------------------------------- #
#  the box
# --------------------------------------------------------------------------- #

class Box:
    MAX_SPEED = 1.0            # arena units per tick at full leg drive
    MAX_TURN = 0.7
    LEG_REST = 262.0           # measured leg MN spikes/tick at rest
    LEG_MAX = 520.0
    LEG_STOP = 130.0           # below this the fly is stationary
    FEED_MN = 60.0             # proboscis motor spikes/tick that count as feeding
                               # (rest ~5, sweet taste ~450 - measured)

    def __init__(self, a):
        self.a = a
        self.world = World(a.seed)
        self.brain = BoxBrain(a.brain, a.seed, a.kc_thresh, a.strength, a.mbon_hold,
                              a.rate, a.data_dir, a.odours)
        self.body = Body(heading=float(np.random.default_rng(a.seed).uniform(0, 2 * math.pi)))
        self.life = 1
        self.lives = []
        self.events = []
        self.tick_n = 0
        self.stats = {"ate": 0, "drank": 0, "hurt": 0, "sheltered": 0, "escapes": 0}
        self.zone_ticks = {k: 0 for k in World.KINDS}
        self.feeding = False
        self.out_dir = a.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.logf = open(os.path.join(self.out_dir, "events.jsonl"), "a")

    def spawn(self):
        rng = np.random.default_rng(self.a.seed + self.life)
        self.body = Body(heading=float(rng.uniform(0, 2 * math.pi)))
        if not self.a.inherit:
            self.brain.b._out_w[self.brain.b._plastic] = self.brain.b._w0.copy()
        self.brain.b.reset()
        self.stats = {"ate": 0, "drank": 0, "hurt": 0, "sheltered": 0, "escapes": 0}
        self.zone_ticks = {k: 0 for k in World.KINDS}

    def step(self):
        w, bd, br = self.world, self.body, self.brain
        # -- sense --
        ax, ay = math.cos(bd.heading), math.sin(bd.heading)
        lx, ly = bd.x - ay * 0.9, bd.y + ax * 0.9
        rx, ry = bd.x + ay * 0.9, bd.y - ax * 0.9
        left, right = w.odour_field(lx, ly), w.odour_field(rx, ry)
        th, d = w.nearest(bd.x, bd.y)
        touch = th.kind if (th is not None and d <= th.r) else None
        temp = w.ambient() + 12.0 * w.heat_at(bd.x, bd.y)
        sh, sd = w.nearest(bd.x, bd.y, "shelter")
        if sh is not None and sd <= sh.r * 2:
            temp += 6.0
        drives = {"hunger": 1.0 - bd.energy, "thirst": 1.0 - bd.water, "cold": 1.0 - bd.warmth}

        # -- reinforcement from the world: only while the outcome is happening --
        reinforce = None
        if touch == "hazard":
            reinforce = "PPL105"
        elif touch in ("food", "water") and self.feeding:
            reinforce = "PAM08"      # reward while the proboscis is actually feeding

        cmd = br.tick(left, right, drives, abs(temp - 24.0), touch, reinforce)

        # -- act --
        leg = cmd["leg"]
        self.feeding = touch in ("food", "water") and cmd["feed"] >= self.FEED_MN
        bd.stopped = leg < self.LEG_STOP or self.feeding
        speed = 0.0 if bd.stopped else min(1.0, max(0.0, (leg - self.LEG_STOP) / (self.LEG_MAX - self.LEG_STOP)))
        jump = cmd["escape"] > 0
        if jump:
            self.stats["escapes"] += 1
        bd.heading += cmd["turn"] * self.MAX_TURN * (2.0 if jump else 1.0)
        step = self.MAX_SPEED * (2.5 if jump else speed)
        bd.x += ax * step; bd.y += ay * step
        bd.speed = step
        dd = math.hypot(bd.x, bd.y)
        if dd > WORLD_R:
            bd.x *= WORLD_R / dd; bd.y *= WORLD_R / dd
            bd.heading += math.pi * 0.5

        ev = []
        if touch == "food" and self.feeding and th.amount > 0:
            bd.energy = min(1.0, bd.energy + 0.05); th.amount -= 0.05; self.stats["ate"] += 1; ev.append("ate")
        elif touch == "water" and self.feeding and th.amount > 0:
            bd.water = min(1.0, bd.water + 0.06); th.amount -= 0.05; self.stats["drank"] += 1; ev.append("drank")
        elif touch == "hazard":
            bd.energy -= 0.03; bd.warmth = min(1.0, bd.warmth + 0.02); self.stats["hurt"] += 1; ev.append("hurt")
        elif touch == "shelter":
            bd.warmth = min(1.0, bd.warmth + 0.03); self.stats["sheltered"] += 1
        for k in self.zone_ticks:
            t_, d_ = w.nearest(bd.x, bd.y, k)
            if t_ is not None and d_ <= 6.0:
                self.zone_ticks[k] += 1

        # -- metabolism --
        bd.energy -= 0.00025 + 0.00035 * speed      # ~5-7 min of brain time to starve
        bd.water -= 0.0003
        bd.warmth += -0.0008 if temp < 18.0 else (0.0006 if temp < 30.0 else -0.0008)
        bd.energy = max(0.0, bd.energy); bd.water = max(0.0, bd.water)
        bd.warmth = float(np.clip(bd.warmth, 0.0, 1.0))
        bd.age += 1
        w.regrow()
        self.tick_n += 1

        if bd.energy <= 0 or bd.water <= 0 or bd.warmth <= 0:
            cause = "starved" if bd.energy <= 0 else "dehydrated" if bd.water <= 0 else "froze"
            bd.alive = False
            rec = {"life": self.life, "age_ticks": bd.age, "age_s": bd.age * TICK_MS / 1000,
                   "cause": cause, "stats": dict(self.stats), "zone_ticks": dict(self.zone_ticks),
                   "weights": br.weights(), "world_t": w.t}
            self.lives.append(rec)
            self.logf.write(json.dumps({"event": "death", **rec}) + "\n"); self.logf.flush()
            log(f"  LIFE {self.life} ended at {rec['age_s']:.0f} s: {cause}   ate {self.stats['ate']} "
                f"drank {self.stats['drank']} hurt {self.stats['hurt']}   weights {100*rec['weights']:.2f}%")
            self.life += 1
            self.spawn()
        return cmd, ev, temp

    def state(self, cmd, temp, wall):
        w, bd = self.world, self.body
        return {"t": w.t, "sim_s": self.tick_n * TICK_MS / 1000, "wall_s": wall,
                "life": self.life, "night": w.is_night, "ambient": w.ambient(), "temp": temp,
                "fly": {"x": bd.x, "y": bd.y, "heading": bd.heading, "energy": bd.energy,
                        "water": bd.water, "warmth": bd.warmth, "age_s": bd.age * TICK_MS / 1000,
                        "speed": bd.speed, "stopped": bd.stopped, "feeding": self.feeding},
                "cmd": cmd, "stats": self.stats, "zone_ticks": self.zone_ticks,
                "weights": self.brain.weights(),
                "things": [{"kind": t.kind, "x": t.x, "y": t.y, "amount": t.amount / t.max_amount, "r": t.r}
                           for t in w.things],
                "lives": self.lives[-20:], "arena_r": WORLD_R, "inherit": self.a.inherit,
                "odours": self.brain.odours}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--odours", default="../results/odours3.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kc-thresh", type=float, default=1.0)
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--mbon-hold", type=float, default=0.85)
    ap.add_argument("--rate", type=float, default=0.0003)
    ap.add_argument("--no-inherit", dest="inherit", action="store_false",
                    help="each life starts with a naive brain (control)")
    ap.add_argument("--max-lives", type=int, default=0, help="0 = run until Ctrl+C")
    ap.add_argument("--max-sim-s", type=float, default=0.0)
    ap.add_argument("--out-dir", default="../results/box")
    a = ap.parse_args()

    log(f"loading {a.brain} ...")
    t0 = time.perf_counter()
    box = Box(a)
    log(f"  {box.brain.b.N:,} neurons, {len(box.brain.leg)} leg MNs, calibrated R-share "
        f"{list(np.round(box.brain.cal_ref, 3))}   ({time.perf_counter()-t0:.0f}s)")
    log(f"  odours: " + ", ".join(f"{k}={v}" for k, v in box.brain.odours.items()))
    log(f"  brain {'persists' if a.inherit else 'is reset'} across lives. Ctrl+C to stop.\n")
    state_path = os.path.join(a.out_dir, "state.json")
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
    cmd, temp = {"turn": 0.0, "leg": 0, "escape": 0, "mbon": 0, "feed": 0}, 24.0
    while not stop["now"]:
        cmd, ev, temp = box.step()
        if box.tick_n % 2 == 0:
            st = box.state(cmd, temp, time.perf_counter() - t0)
            tmp = state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, state_path)
        if box.tick_n % 40 == 0:
            bd = box.body
            log(f"  life {box.life} t={box.tick_n*TICK_MS/1000:6.1f}s pos=({bd.x:5.1f},{bd.y:5.1f}) "
                f"e={bd.energy:.2f} w={bd.water:.2f} c={bd.warmth:.2f} leg={cmd['leg']:4d} "
                f"{'feed' if box.feeding else 'stop' if bd.stopped else 'walk'} {'night' if box.world.is_night else 'day'} "
                f"ate {box.stats['ate']} drank {box.stats['drank']} hurt {box.stats['hurt']}  "
                f"w {100*box.brain.weights():.2f}%")
        if a.max_lives and box.life > a.max_lives:
            break
        if a.max_sim_s and box.tick_n * TICK_MS / 1000 >= a.max_sim_s:
            break
    log(f"\nstopped after {box.tick_n*TICK_MS/1000:.0f} s of brain time, {len(box.lives)} deaths")
    for r in box.lives:
        log(f"  life {r['life']}: {r['age_s']:.0f} s, {r['cause']}, ate {r['stats']['ate']} "
            f"drank {r['stats']['drank']} hurt {r['stats']['hurt']}")


if __name__ == "__main__":
    main()
