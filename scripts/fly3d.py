#!/usr/bin/env python3
"""
fly3d.py - a room the connectome can walk and fly around in.

    python3 fly3d.py --brain brain_whole.npz --seconds 45 --out results/fly3d.jsonl

WHY A ROOM AND NOT THE ARENA
    flyworld.py's arena is a flat disc and the creature is a point on it. A fly walks
    up walls, hangs off the ceiling, takes off, and spends most of its airtime turning
    in place. None of that fits in two dimensions, so this is a box: floor, four walls,
    ceiling, all landable.

THE SAME TWO CLOCKS AS EVERYTHING ELSE
    The brain runs at ~0.45x real time and the viewer wants 60 fps, so Python owns both
    and steps them in lockstep with no wall-clock pressure: a brain tick every
    `--tick-ms` of simulated time produces a command, the body integrates at
    `--fps` between commands, and every frame is written out for the viewer to replay.

WHAT THE CONNECTOME DECIDES
    turn evidence   calibrated DNa left/right asymmetry (see below)
    walk drive      total descending rate -> ground speed
    escape          DNp01, the Giant Fiber -> escape takeoff, which is its real job

    THE CALIBRATION IS NOT OPTIONAL. Raw (R-L)/(R+L) on the DNa family measures
    -1.304 under a SYMMETRIC stimulus, -1.263 with the left antenna driven and -1.185
    with the right. The structural left/right bias is 16x the actual side signal, so
    the raw readout is a constant hard turn and the fly walks in perfect circles. The
    zero point is whatever a symmetric stimulus produces, measured at several drive
    levels and interpolated - the same fix flyworld.py's calibrate() applies.

WHAT IS IMPOSED AND NOT DERIVED - say this out loud when showing the output
    Flies do not steer continuously. They hold a course and turn in discrete body
    saccades, roughly one a second walking and far faster in flight, and they stop and
    groom constantly. Integrating any continuous turn signal gives circles no matter
    how well calibrated it is. So the brain's asymmetry is treated as EVIDENCE that
    biases a saccade generator, and the walk/stop/groom/fly state machine sits on top.
    The connectome contains the circuitry for all of it. This file does not read it out.
    Wingbeat, banking and leg placement are animation.
"""
from __future__ import annotations

import argparse, json, math, os, random, time
import numpy as np

from flysim import FlyBrain, Params

# room, in fly body-lengths (the model is ~23 units nose to tail, scaled 0.115 in the
# viewer, so these are comfortable numbers to think in)
ROOM = {"w": 64.0, "d": 44.0, "h": 30.0}
MODES = ["stop", "walk", "groom", "takeoff", "flight", "land"]
# spontaneous takeoff probability per walk-bout decision. IMPOSED, not derived (see
# the docstring). fly3d_learn.py sets it to 0 and derives takeoff from the wing-power
# motor neurons instead.
RANDOM_TAKEOFF = 0.26


# --------------------------------------------------------------------------- #
#  brain
# --------------------------------------------------------------------------- #

class Brain:
    """The connectome, wrapped so the body only ever sees three numbers."""

    def __init__(self, path, tick_ms, gain, seed):
        self.b = b = FlyBrain(path, Params(gain=gain), seed=seed)
        self.tick_ms = tick_ms
        dn = b.pop["DN"]
        names = b.type[dn].astype(str)
        side = b.side[dn]
        fam = np.char.startswith(names, "DNa")
        self.L = dn[fam & (side == "L")]
        self.R = dn[fam & (side == "R")]
        if len(self.L) < 2 or len(self.R) < 2:
            self.L, self.R = dn[side == "L"], dn[side == "R"]
        self.escape = dn[np.char.startswith(names, "DNp01")]
        mech = b.pop["mechano"]
        self.mL = mech[b.side[mech] == "L"]
        self.mR = mech[b.side[mech] == "R"]
        self.steps = int(round(tick_ms / b.p.dt))
        self.base = None
        self._calibrate()

    def _drive(self, hzL, hzR, steps):
        b = self.b
        b.drive_hz[:] = 0.0
        b.drive_hz[self.mL] = hzL
        b.drive_hz[self.mR] = hzR
        aL = aR = aE = 0
        for _ in range(steps):
            spk = b.step()
            aL += int(spk[self.L].sum())
            aR += int(spk[self.R].sum())
            aE += int(spk[self.escape].sum())
        return aL / max(len(self.L), 1), aR / max(len(self.R), 1), aE

    def _calibrate(self):
        cal = []
        for lv in (0.0, 0.3, 0.6, 1.0):
            self.b.reset()
            self._drive(180.0 * lv, 180.0 * lv, int(120 / self.b.p.dt))
            L, R, _ = self._drive(180.0 * lv, 180.0 * lv, int(260 / self.b.p.dt))
            tot = L + R
            cal.append((tot, R / tot if tot > 1e-6 else 0.5))
        cal.sort()
        # np.interp needs strictly increasing x; duplicate drive levels do occur
        tots, refs = [], []
        for t, r in cal:
            if tots and t <= tots[-1]:
                refs[-1] = (refs[-1] + r) / 2.0
                continue
            tots.append(t); refs.append(r)
        if len(tots) < 2:
            tots, refs = [0.0, 1.0], [0.5, 0.5]
        self.cal_tot, self.cal_ref = np.array(tots), np.array(refs)
        self.b.reset()

    def tick(self, threat, bias):
        """threat 0..1 how close the disturbance is; bias -1 left .. +1 right."""
        if threat <= 0.01:
            hzL = hzR = 0.0
        else:
            hzL = 180.0 * threat * (1.0 - 0.6 * max(0.0, bias))
            hzR = 180.0 * threat * (1.0 - 0.6 * max(0.0, -bias))
        L, R, aE = self._drive(hzL, hzR, self.steps)
        tot = L + R
        ref = float(np.interp(tot, self.cal_tot, self.cal_ref))
        frac = (R / tot) if tot > 1e-6 else ref
        ev = frac - ref
        self.base = ev if self.base is None else self.base + (ev - self.base) * 0.02
        return {"evidence": float(np.clip((ev - self.base) * 12.0, -1.0, 1.0)),
                "drive": float(tot), "escape": int(aE)}


# --------------------------------------------------------------------------- #
#  body
# --------------------------------------------------------------------------- #

class Body:
    """
    Position, orientation and behavioural state in the room.

    Surfaces are indexed 0 floor, 1..4 walls, 5 ceiling. While walking, the fly is
    stuck to one of them and `up` is that surface's normal; in flight it is free.
    """

    def __init__(self, rng):
        self.rng = rng
        self.x, self.y, self.z = 0.0, 0.0, 0.0        # y is up
        self.yaw = rng.uniform(0, 2 * math.pi)
        self.pitch = 0.0
        self.roll = 0.0
        self.surface = 0
        self.mode = "walk"
        self.mode_until = 0.0
        self.speed = 0.0
        self.gait = 0.0
        self.groom = 0.0
        self.wing = 0.0                                # 0 folded .. 1 beating
        self.sacc_t = 0.0
        self.sacc_rate = 0.0
        self.next_sacc = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.target = None
        self.evidence = 0.0
        self.drive = 0.0
        self.escaping = False

    # -- surface helpers -------------------------------------------------- #
    def surface_normal(self, s):
        return [(0, 1, 0), (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, -1, 0)][s]

    def clamp_to_surface(self):
        w, d, h = ROOM["w"] / 2, ROOM["d"] / 2, ROOM["h"]
        m = 2.0
        if self.surface == 0:
            self.y = 0.0
            self.x = min(max(self.x, -w + m), w - m)
            self.z = min(max(self.z, -d + m), d - m)
        elif self.surface == 5:
            self.y = h
            self.x = min(max(self.x, -w + m), w - m)
            self.z = min(max(self.z, -d + m), d - m)
        elif self.surface in (1, 2):
            self.z = -d if self.surface == 1 else d
            self.x = min(max(self.x, -w + m), w - m)
            self.y = min(max(self.y, m), h - m)
        else:
            self.x = -w if self.surface == 3 else w
            self.z = min(max(self.z, -d + m), d - m)
            self.y = min(max(self.y, m), h - m)

    def on_surface_bounds(self):
        w, d, h = ROOM["w"] / 2, ROOM["d"] / 2, ROOM["h"]
        m = 3.0
        if self.surface in (0, 5):
            return (abs(self.x) > w - m) or (abs(self.z) > d - m)
        if self.surface in (1, 2):
            return (abs(self.x) > w - m) or self.y < m or self.y > h - m
        return (abs(self.z) > d - m) or self.y < m or self.y > h - m


def step_body(b: Body, cmd: dict, dt: float, t: float, threat: float, bias: float):
    rng = b.rng
    b.evidence = cmd["evidence"]
    b.drive = cmd["drive"]
    if cmd["escape"] > 0 and b.mode in ("walk", "stop", "groom"):
        # DNp01 fires -> escape takeoff. This is the Giant Fiber's actual job.
        b.mode, b.mode_until, b.escaping = "takeoff", t + 0.18, True

    # ---------- state machine ----------
    if b.mode in ("walk", "stop", "groom") and t >= b.mode_until:
        r = rng.random()
        if b.mode == "walk":
            if r < RANDOM_TAKEOFF:
                b.mode, b.mode_until, b.escaping = "takeoff", t + 0.20, False
            elif r < 0.48:
                b.mode, b.mode_until = "stop", t + rng.uniform(0.35, 1.1)
            else:
                b.mode_until = t + rng.uniform(0.8, 2.4)
        elif b.mode == "stop":
            if r < 0.45:
                b.mode, b.mode_until, b.groom = "groom", t + rng.uniform(1.0, 2.6), 0.0
            else:
                b.mode, b.mode_until = "walk", t + rng.uniform(0.8, 2.4)
        else:
            b.mode, b.mode_until = "walk", t + rng.uniform(0.8, 2.4)

    if b.mode in ("stop", "groom") and threat > 0.5:
        b.mode, b.mode_until = "walk", t + rng.uniform(0.6, 1.4)

    # ---------- saccades ----------
    flying = b.mode in ("flight", "takeoff", "land")
    if b.sacc_t > 0.0:
        b.sacc_t -= dt
        b.yaw += b.sacc_rate * dt
    elif b.mode in ("walk", "flight") and t >= b.next_sacc:
        ev = b.evidence
        if threat > 0.3:
            sign = -1.0 if bias > 0 else 1.0        # turn away from the disturbance
        elif abs(ev) > 0.06:
            sign = 1.0 if ev > 0 else -1.0
        else:
            sign = rng.choice((-1.0, 1.0))
        if flying:
            mag = math.radians(rng.uniform(40.0, 130.0))     # flight saccades are big
            dur = rng.uniform(0.05, 0.09)                    # and fast
            gap = rng.uniform(0.10, 0.42)
        else:
            mag = math.radians(rng.uniform(28.0, 110.0))
            dur = rng.uniform(0.09, 0.17)
            gap = rng.uniform(0.45, 1.5)
        mag *= 1.0 + 1.2 * min(1.0, abs(ev) * 3.0)
        b.sacc_t, b.sacc_rate = dur, sign * mag / dur
        b.next_sacc = t + dur + gap
    else:
        b.yaw += math.sin(t * 1.7) * 0.2 * dt

    b.roll += (-b.sacc_rate * 0.16 * (1.0 if b.sacc_t > 0 else 0.0) - b.roll) * min(1, 8 * dt)

    # ---------- translation ----------
    if b.mode == "takeoff":
        b.wing = min(1.0, b.wing + dt * 9.0)
        n = b.surface_normal(b.surface)
        push = 26.0 if b.escaping else 15.0
        b.vx += n[0] * push * dt * 6
        b.vy += n[1] * push * dt * 6
        b.vz += n[2] * push * dt * 6
        if t >= b.mode_until:
            b.mode, b.mode_until = "flight", t + rng.uniform(2.4, 6.5)
            b.surface = -1
    elif b.mode == "flight":
        b.wing = 1.0
        fwd = (math.cos(b.yaw), math.sin(b.yaw))
        cruise = 26.0 + 16.0 * min(1.0, b.drive / 6.0) + (22.0 if threat > 0.4 else 0.0)
        b.vx += (fwd[0] * cruise - b.vx) * min(1.0, 2.6 * dt)
        b.vz += (fwd[1] * cruise - b.vz) * min(1.0, 2.6 * dt)
        # hold a height, wander in it
        want_y = ROOM["h"] * (0.45 + 0.28 * math.sin(t * 0.7 + 1.3))
        b.vy += ((want_y - b.y) * 1.6 - b.vy) * min(1.0, 3.0 * dt)
        if t >= b.mode_until:
            b.mode, b.mode_until = "land", t + 4.0
            b.target = pick_landing(b, rng)
    elif b.mode == "land":
        b.wing = 1.0
        tx, ty, tz, surf = b.target
        dx, dy, dz = tx - b.x, ty - b.y, tz - b.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        sp = max(8.0, min(34.0, dist * 2.2))
        b.vx += (dx / dist * sp - b.vx) * min(1.0, 4.0 * dt)
        b.vy += (dy / dist * sp - b.vy) * min(1.0, 4.0 * dt)
        b.vz += (dz / dist * sp - b.vz) * min(1.0, 4.0 * dt)
        b.yaw += (math.atan2(dz, dx) - b.yaw) * min(1.0, 3.0 * dt)
        if dist < 1.6 or t >= b.mode_until:
            b.surface = surf
            b.vx = b.vy = b.vz = 0.0
            b.wing = 0.0
            b.escaping = False
            b.mode, b.mode_until = "walk", t + rng.uniform(0.8, 2.0)
            b.clamp_to_surface()
    else:
        b.wing = max(0.0, b.wing - dt * 6.0)
        target = 0.0 if b.mode in ("stop", "groom") else 5.5 + 9.0 * min(1.0, b.drive / 6.0)
        b.speed += (target - b.speed) * min(1.0, 7.0 * dt)
        step_on_surface(b, dt)

    if b.mode in ("takeoff", "flight", "land"):
        b.x += b.vx * dt
        b.y += b.vy * dt
        b.z += b.vz * dt
        w, d, h = ROOM["w"] / 2, ROOM["d"] / 2, ROOM["h"]
        if abs(b.x) > w - 1.5 or abs(b.z) > d - 1.5 or b.y < 1.2 or b.y > h - 1.2:
            b.x = min(max(b.x, -w + 1.5), w - 1.5)
            b.z = min(max(b.z, -d + 1.5), d - 1.5)
            b.y = min(max(b.y, 1.2), h - 1.2)
            b.yaw += math.pi * rng.uniform(0.4, 0.8)
            b.vx *= -0.3; b.vz *= -0.3
        b.pitch += ((-0.22 if b.mode != "land" else 0.16) - b.pitch) * min(1, 4 * dt)
    else:
        b.pitch += (0.0 - b.pitch) * min(1, 6 * dt)

    f = 3.0 + 8.0 * min(1.0, b.speed / 14.0)
    b.gait = (b.gait + f * dt) % 1.0
    if b.mode == "groom":
        b.groom += dt * 7.5


def step_on_surface(b: Body, dt: float):
    """Walking is 2D in the surface's own plane, then re-clamped to it."""
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    if b.surface in (0, 5):
        b.x += c * b.speed * dt
        b.z += s * b.speed * dt
    elif b.surface in (1, 2):
        b.x += c * b.speed * dt
        b.y += s * b.speed * dt
    else:
        b.z += c * b.speed * dt
        b.y += s * b.speed * dt
    if b.on_surface_bounds():
        b.yaw += math.pi * b.rng.uniform(0.45, 0.75)
        b.sacc_t = 0.0
    b.clamp_to_surface()


def pick_landing(b: Body, rng):
    w, d, h = ROOM["w"] / 2, ROOM["d"] / 2, ROOM["h"]
    surf = rng.choice([0, 0, 0, 1, 2, 3, 4, 5])          # floor is the usual choice
    m = 4.0
    if surf == 0:
        return (rng.uniform(-w + m, w - m), 0.0, rng.uniform(-d + m, d - m), 0)
    if surf == 5:
        return (rng.uniform(-w + m, w - m), h, rng.uniform(-d + m, d - m), 5)
    if surf in (1, 2):
        return (rng.uniform(-w + m, w - m), rng.uniform(m, h - m),
                -d if surf == 1 else d, surf)
    return (-w if surf == 3 else w, rng.uniform(m, h - m),
            rng.uniform(-d + m, d - m), surf)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_whole.npz")
    ap.add_argument("--seconds", type=float, default=45.0,
                    help="0 = run forever; Ctrl+C writes what it has")
    ap.add_argument("--respawn-every", type=float, default=0.0,
                    help="seconds between forced respawns (0 = never)")
    ap.add_argument("--fps", type=int, default=40)
    ap.add_argument("--tick-ms", type=float, default=150.0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="results/fly3d.jsonl")
    a = ap.parse_args()

    t0 = time.perf_counter()
    print(f"loading {a.brain} ...", flush=True)
    brain = Brain(a.brain, a.tick_ms, a.gain, a.seed)
    print(f"  {brain.b.N:,} neurons | DNa {len(brain.L)}L/{len(brain.R)}R | "
          f"DNp01 {len(brain.escape)} | mechano {len(brain.mL)}L/{len(brain.mR)}R",
          flush=True)
    print(f"  bias calibrated: R-share {list(np.round(brain.cal_ref,4))} "
          f"at drive {list(np.round(brain.cal_tot,2))}", flush=True)

    rng = random.Random(a.seed)
    b = Body(rng)
    b.clamp_to_surface()

    dt = 1.0 / a.fps
    endless = a.seconds <= 0
    n_frames = (1 << 62) if endless else int(a.seconds * a.fps)
    brain_every = max(1, int(round((a.tick_ms / 1000.0) / dt)))
    frames, modes_seen = [], {}
    cmd = {"evidence": 0.0, "drive": 0.0, "escape": 0}
    respawns = []
    if endless:
        print("  endless mode - Ctrl+C to stop and write the episode", flush=True)

    # a wandering disturbance the fly can feel - stands in for the cursor
    i = -1
    try:
      while i + 1 < n_frames:
        i += 1
        t = i * dt

        if a.respawn_every > 0 and t > 0 and (i % int(a.respawn_every * a.fps) == 0):
            # put it back on the floor in the middle, keep the brain's state: a
            # respawn is a change to the BODY, not a new animal
            b.surface, b.mode = 0, "walk"
            b.x = b.y = b.z = 0.0
            b.vx = b.vy = b.vz = 0.0
            b.wing = 0.0
            b.yaw = rng.uniform(0, 2 * math.pi)
            b.mode_until = t + rng.uniform(0.8, 2.0)
            b.clamp_to_surface()
            respawns.append(i)
            print(f"  t={t:6.1f}s  RESPAWN", flush=True)
        dx = 26.0 * math.sin(t * 0.31) - b.x
        dz = 18.0 * math.cos(t * 0.23) - b.z
        dy = 12.0 + 6.0 * math.sin(t * 0.17) - b.y
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        threat = max(0.0, min(1.0, 1.0 - dist / 26.0)) ** 1.5
        bias = math.sin(math.atan2(dz, dx) - b.yaw)

        if i % brain_every == 0:
            cmd = brain.tick(threat, bias)

        step_body(b, cmd, dt, t, threat, bias)
        modes_seen[b.mode] = modes_seen.get(b.mode, 0) + 1
        frames.append([
            round(b.x, 2), round(b.y, 2), round(b.z, 2),
            round(b.yaw, 3), round(b.pitch, 3), round(b.roll, 3),
            MODES.index(b.mode) if b.mode in MODES else 1,
            round(b.gait, 3), round(b.wing, 2), round(b.groom, 2),
            round(b.drive, 2), round(b.evidence, 3), round(threat, 2),
        ])
        if i % (a.fps * 5) == 0:
            print(f"  t={t:5.1f}s  {b.mode:<8} pos=({b.x:6.1f},{b.y:5.1f},{b.z:6.1f}) "
                  f"surf={b.surface} drive={b.drive:5.2f}", flush=True)
    except KeyboardInterrupt:
        print(f"\n  stopped at {len(frames)/a.fps:.1f}s", flush=True)

    meta = {"room": ROOM, "fps": a.fps, "seconds": len(frames) / a.fps,
            "endless": endless, "respawn_frames": respawns, "modes": MODES,
            "fields": ["x","y","z","yaw","pitch","roll","mode","gait","wing",
                       "groom","drive","evidence","threat"],
            "mode_frames": modes_seen, "tick_ms": a.tick_ms,
            "neurons": int(brain.b.N),
            "calibration": [[float(x), float(y)] for x, y in
                            zip(brain.cal_tot, brain.cal_ref)],
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for fr in frames:
            f.write(json.dumps(fr) + "\n")
    print(f"\nwrote {a.out}: {len(frames)} frames in {meta['wall_seconds']}s")
    pct = {k: f"{100.0*v/len(frames):.0f}%" for k, v in
           sorted(modes_seen.items(), key=lambda kv: -kv[1])}
    print(f"time budget: {pct}")


if __name__ == "__main__":
    main()
