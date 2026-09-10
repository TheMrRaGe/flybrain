#!/usr/bin/env python3
"""
desktop_fly.py - the connectome walking on your Windows desktop.

    python3 desktop_fly.py --brain brain_whole.npz

    Esc or right-click quits.  --debug draws a readout panel.

WHY THIS EXISTS AND WHAT IT IS NOT
    DesktopFly-Linux is GTK3 + wlr-layer-shell and does not run on Windows. This is
    the same idea rebuilt on Tkinter, which ships with Python: a borderless,
    always-on-top, click-through window whose background colour is keyed out, so only
    the fly is drawn over whatever is already on screen.

    The body geometry - segment sizes, the six-leg table, the tripod phases - is
    ported from DesktopFly-Linux (MIT, (c) 2026 Denis Shiryaev and contributors),
    projected to a top-down view. The behaviour is this repository's own brain.

THE TWO CLOCKS
    The brain runs at ~0.45x real time, and a desktop pet has to move at 60 fps.
    Those cannot share a loop, so they do not: a worker thread runs the connectome and
    publishes a movement command roughly twice a second, while the UI thread
    interpolates smoothly between commands. Same split as flyworld.py, for the same
    reason - nothing is rushed and nothing is dropped.

WHAT DRIVES IT
    The cursor is the world. Distance to the pointer drives the mechanosensory
    receptors, which is the ONE sensory channel measured to actually reach the
    descending bus in this model - 10,324 descending spikes at 180 Hz. Vision would be
    the obvious choice and it is unusable: all 4,107 photoreceptors are histaminergic
    with sign -1, and against the model's 0 Hz resting rate they produce literally zero
    downstream spikes. See habituation.py.

TWO BUGS THIS FILE HAD, AND WHAT THEY LOOKED LIKE
    The first version walked in perfect circles. Measured cause, both parts:

    1. The receptors were split into "left" and "right" by ARRAY INDEX. That split is
       68%/71% impure - it is not a left/right split at all. Sensory somas sit outside
       the imaged volume so somaSide is "M"; laterality comes from rootSide, which is
       what `side` carries. Split on that.

    2. Far worse: the raw left/right readout is dominated by structural bias.
       Measured on the DNa family, turn_raw = (R-L)/(R+L)*3 came out

           symmetric stimulus   -1.304
           left antenna only    -1.263
           right antenna only   -1.185

       The bias is -1.3 and constant; the actual side signal is 0.078. Bias is 16x
       signal, the command clips to -1.0 every tick, and the fly turns at maximum rate
       forever. flyworld.py already solves this with calibrate(): measure the R-share
       under a SYMMETRIC stimulus at several drive levels, interpolate, and subtract.
       Whatever comes out of a symmetric stimulus carries no information about the
       world, at any level, and that is the zero point. Ported here.

WHAT IS CONNECTOME AND WHAT IS NOT - read before quoting this
    Connectome: the turn EVIDENCE (calibrated DNa left/right asymmetry), the escape
    trigger (DNp01), and the walking drive (total descending rate).

    NOT connectome: the saccade generator and the walk/stop/groom state machine. Real
    flies do not steer continuously - they hold a fairly straight course and turn in
    discrete body saccades, and they stop and groom often. Integrating a continuous
    turn signal produces circles by construction, however well calibrated. So the
    brain's asymmetry is treated as EVIDENCE that accumulates, and a saccade fires
    when it crosses threshold. That is imposed, not derived. The connectome contains
    the circuitry for it; this file does not read it out.
"""
from __future__ import annotations

import argparse, math, queue, random, sys, threading, time
import tkinter as tk

import numpy as np

from flysim import FlyBrain, Params

CHROMA = "#FF00FE"          # keyed out to transparent; nothing else may use it

# --- body, ported from DesktopFly-Linux geometry.py, projected top-down ---------
# (side, (x, y), yaw offset, gait phase, is_front, femur, tibia, tarsus)
LEG_SPECS = (
    ( 1.0, ( 3.1,  5.3),  0.95, 0.0, True,  4.2, 4.8, 3.2),
    (-1.0, (-3.1,  5.3),  0.95, 0.5, True,  4.2, 4.8, 3.2),
    ( 1.0, ( 3.7,  2.0), -0.10, 0.5, False, 4.8, 5.6, 3.8),
    (-1.0, (-3.7,  2.0), -0.10, 0.0, False, 4.8, 5.6, 3.8),
    ( 1.0, ( 3.3, -1.2), -0.95, 0.0, False, 5.8, 7.0, 4.6),
    (-1.0, (-3.3, -1.2), -0.95, 0.5, False, 5.8, 7.0, 4.6),
)
BODY_BROWN, LEG_BROWN = "#806224", "#543d24"
EYE_RED, WING = "#9e1a12", "#d8d8d8"
HEAD_BROWN, ABDOMEN = "#93794f", "#b88c52"
ABDOMEN_BAND = "#382617"

GAIT_FREQ = (3.0, 11.0)     # Hz, upstream constants.py
GAIT_AMP = (0.20, 0.50)     # rad
STANCE_FRACTION = 0.6


# --------------------------------------------------------------------------- #
#  the brain, on its own thread
# --------------------------------------------------------------------------- #

class BrainWorker(threading.Thread):
    """Publishes {'turn','speed','escape'} about twice a second. Never blocks the UI."""

    daemon = True

    def __init__(self, path: str, tick_ms: float, gain: float):
        super().__init__()
        self.path, self.tick_ms, self.gain = path, tick_ms, gain
        self.out: "queue.Queue[dict]" = queue.Queue(maxsize=4)
        self.state = {"threat": 0.0}          # written by the UI thread, read here
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.status = "loading connectome..."

    def run(self) -> None:
        try:
            b = FlyBrain(self.path, Params(gain=self.gain), seed=0)
        except Exception as exc:                              # noqa: BLE001
            self.status = f"brain failed: {exc}"
            self.ready.set()
            return

        dn = b.pop["DN"]
        names = b.type[dn].astype(str)
        side = b.side[dn]
        a02 = np.char.startswith(names, "DNa02")
        if a02.sum() >= 2:
            dn_L, dn_R = dn[a02 & (side == "L")], dn[a02 & (side == "R")]
        else:
            dn_L, dn_R = dn[side == "L"], dn[side == "R"]
        # the DNa FAMILY is the readout that actually carries direction (d'=4.21);
        # DNa02 alone is right but fires ~1 spike per trial
        fam = np.char.startswith(names, "DNa")
        fam_L, fam_R = dn[fam & (side == "L")], dn[fam & (side == "R")]
        if len(fam_L) >= 2 and len(fam_R) >= 2:
            dn_L, dn_R = fam_L, fam_R
        escape = dn[np.char.startswith(names, "DNp01")]
        mech = b.pop["mechano"]
        # LATERALITY COMES FROM rootSide, NOT array position (bug 1)
        mL = mech[b.side[mech] == "L"]
        mR = mech[b.side[mech] == "R"]
        if len(mL) < 20 or len(mR) < 20:
            half = len(mech) // 2
            mL, mR = mech[:half], mech[half:]

        steps = int(round(self.tick_ms / b.p.dt))

        def measure(hz_L, hz_R, ms=None):
            n = steps if ms is None else int(round(ms / b.p.dt))
            b.drive_hz[:] = 0.0
            b.drive_hz[mL] = hz_L
            b.drive_hz[mR] = hz_R
            aL = aR = aE = 0
            for _ in range(n):
                spk = b.step()
                aL += int(spk[dn_L].sum())
                aR += int(spk[dn_R].sum())
                aE += int(spk[escape].sum())
            return (aL / max(len(dn_L), 1), aR / max(len(dn_R), 1), aE)

        # CALIBRATION (bug 2). The zero point is whatever a SYMMETRIC stimulus produces,
        # measured at several intensities because the split depends on drive level.
        self.status = "calibrating the left/right bias..."
        cal = []
        for lv in (0.0, 0.25, 0.55, 1.0):
            b.reset()
            measure(180.0 * lv, 180.0 * lv, ms=120.0)          # settle
            L, R, _ = measure(180.0 * lv, 180.0 * lv, ms=260.0)
            tot = L + R
            cal.append((tot, R / tot if tot > 1e-6 else 0.5))
        cal.sort()
        cal_tot = np.array([c[0] for c in cal])
        cal_ref = np.array([c[1] for c in cal])
        self.calibration = [(round(t, 2), round(r, 4)) for t, r in cal]
        b.reset()

        self.status = (f"{b.N:,} neurons | {len(dn_L)}L/{len(dn_R)}R DNa | "
                       f"bias zeroed at R-share {cal_ref.mean():.3f}")
        self.ready.set()

        base = None                     # slow adaptive baseline for residual drift
        while not self.stop.is_set():
            threat = float(self.state.get("threat", 0.0))
            bias = float(self.state.get("bias", 0.0))       # -1 left .. +1 right
            lo = 180.0 * threat * (1.0 - 0.6 * max(0.0, bias))
            hi = 180.0 * threat * (1.0 - 0.6 * max(0.0, -bias))
            if threat <= 0.01:
                lo = hi = 0.0
            L, R, accE = measure(lo, hi)

            tot = L + R
            frac = (R / tot) if tot > 1e-6 else float(np.interp(tot, cal_tot, cal_ref))
            ref = float(np.interp(tot, cal_tot, cal_ref))
            ev = (frac - ref)
            # residual drift the interpolated reference does not catch
            base = ev if base is None else base + (ev - base) * 0.02
            evidence = (ev - base) * 12.0
            cmd = {"evidence": float(np.clip(evidence, -1.0, 1.0)),
                   "speed": float(min(1.0, tot / 40.0)),
                   "escape": accE,
                   "drive": tot,
                   "rshare": frac, "ref": ref}
            try:
                self.out.put_nowait(cmd)
            except queue.Full:
                try:
                    self.out.get_nowait()
                    self.out.put_nowait(cmd)
                except queue.Empty:
                    pass


# --------------------------------------------------------------------------- #
#  drawing
# --------------------------------------------------------------------------- #

def rot(px, py, c, s):
    return px * c - py * s, px * s + py * c


class DesktopFly:
    def __init__(self, root: tk.Tk, worker: BrainWorker, scale: float, debug: bool):
        self.root, self.worker, self.scale, self.debug = root, worker, scale, debug
        self.W = root.winfo_screenwidth()
        self.H = root.winfo_screenheight()
        self.canvas = tk.Canvas(root, width=self.W, height=self.H, bg=CHROMA,
                                highlightthickness=0, bd=0)
        self.canvas.pack()

        self.x, self.y = self.W * 0.5, self.H * 0.6
        self.heading = -math.pi / 2          # screen coords: -y is up
        self.speed = self.drive = 0.0
        self.evidence = 0.0                  # accumulated turn evidence from the brain
        self.gait = 0.0
        self.escape_until = 0.0
        self.fleeing = False
        self.bias_side = 0.0
        # A fly holds a course and turns in discrete saccades; it does not steer
        # continuously. See the header - this generator is imposed, not derived.
        self.state = "walk"
        self.state_until = time.perf_counter() + random.uniform(0.6, 1.8)
        self.sacc_t = 0.0                    # remaining saccade time
        self.sacc_rate = 0.0                 # rad/s during the current saccade
        self.next_sacc = time.perf_counter() + random.uniform(0.5, 1.4)
        self.groom_phase = 0.0
        self.walk_speed = 0.0
        self.last = time.perf_counter()
        self.fps_t, self.fps_n, self.fps = self.last, 0, 0.0

        root.bind("<Escape>", lambda e: self.quit())
        root.bind("<Button-3>", lambda e: self.quit())
        self.tick()

    def quit(self):
        self.worker.stop.set()
        self.root.destroy()

    # -- the world is the cursor -------------------------------------------- #
    def sense(self):
        try:
            cx = self.root.winfo_pointerx()
            cy = self.root.winfo_pointery()
        except tk.TclError:
            return
        dx, dy = cx - self.x, cy - self.y
        dist = math.hypot(dx, dy)
        threat = max(0.0, min(1.0, 1.0 - dist / 340.0)) ** 1.5
        # which side of the fly the cursor is on, in body coordinates
        ang = math.atan2(dy, dx) - self.heading
        bias = math.sin(ang)
        self.worker.state["threat"] = threat
        self.worker.state["bias"] = bias
        self.bias_side = bias
        self.threat, self.cursor_dist = threat, dist

    # -- motion -------------------------------------------------------------- #
    def integrate(self, dt):
        now = time.perf_counter()
        while True:
            try:
                cmd = self.worker.out.get_nowait()
            except queue.Empty:
                break
            self.evidence = cmd["evidence"]
            self.speed, self.drive = cmd["speed"], cmd["drive"]
            if cmd["escape"] > 0:
                self.escape_until = now + 0.5
        self.fleeing = now < self.escape_until
        threat = getattr(self, "threat", 0.0)

        # ---- behaviour state machine (imposed, not connectome) ----
        if self.fleeing:
            self.state = "flee"
            self.state_until = max(self.state_until, self.escape_until)
        elif now >= self.state_until:
            r = random.random()
            if self.state == "walk":
                if r < 0.42:
                    self.state, self.state_until = "stop", now + random.uniform(0.35, 1.1)
                else:
                    self.state_until = now + random.uniform(0.7, 2.2)
            elif self.state == "stop":
                if r < 0.45:
                    self.state, self.state_until = "groom", now + random.uniform(0.9, 2.4)
                    self.groom_phase = 0.0
                else:
                    self.state, self.state_until = "walk", now + random.uniform(0.7, 2.2)
            else:
                self.state, self.state_until = "walk", now + random.uniform(0.7, 2.2)

        # a close cursor cancels grooming and gets it moving
        if self.state in ("stop", "groom") and threat > 0.45:
            self.state, self.state_until = "walk", now + random.uniform(0.6, 1.4)

        # ---- saccades: discrete turns, straight runs between ----
        if self.sacc_t > 0.0:
            self.sacc_t -= dt
            self.heading += self.sacc_rate * dt
        elif self.state in ("walk", "flee") and now >= self.next_sacc:
            ev = self.evidence
            if self.fleeing or threat > 0.3:
                # flee AWAY from the cursor; a fly does not orient to a looming threat
                sign = -1.0 if self.bias_side > 0 else 1.0
            elif abs(ev) > 0.06:
                sign = 1.0 if ev > 0 else -1.0
            else:
                sign = random.choice((-1.0, 1.0))
            mag = math.radians(random.uniform(28.0, 115.0))
            mag *= 1.0 + 1.4 * min(1.0, abs(ev) * 3.0)
            if self.fleeing:
                mag *= 1.5
            dur = random.uniform(0.09, 0.17)
            self.sacc_t, self.sacc_rate = dur, sign * mag / dur
            gap = random.uniform(0.45, 1.5)
            self.next_sacc = now + dur + gap * (0.35 if self.fleeing else 1.0)
        else:
            self.heading += math.sin(now * 1.7) * 0.25 * dt   # slight course wander

        # ---- translation ----
        if self.state in ("stop", "groom"):
            target = 0.0
        elif self.fleeing:
            target = 430.0
        else:
            target = 55.0 + 200.0 * self.speed
        self.walk_speed += (target - self.walk_speed) * min(1.0, 7.0 * dt)
        self.x += math.cos(self.heading) * self.walk_speed * dt
        self.y += math.sin(self.heading) * self.walk_speed * dt

        m = 60.0
        if self.x < m or self.x > self.W - m or self.y < m or self.y > self.H - m:
            self.x = min(max(self.x, m), self.W - m)
            self.y = min(max(self.y, m), self.H - m)
            # steer back toward the middle; a fixed angular kick made its own orbit
            self.heading = math.atan2(self.H * 0.5 - self.y, self.W * 0.5 - self.x)
            self.heading += random.uniform(-0.5, 0.5)
            self.sacc_t = 0.0

        f = GAIT_FREQ[0] + (GAIT_FREQ[1] - GAIT_FREQ[0]) * min(1.0, self.walk_speed / 260.0)
        self.gait = (self.gait + f * dt) % 1.0
        if self.state == "groom":
            self.groom_phase += dt * 7.5

    # -- the fly ------------------------------------------------------------- #
    def draw(self):
        c = self.canvas
        c.delete("fly")
        s = self.scale
        # body frame is +y forward; screen heading points along +x, so rotate by
        # heading + 90deg to line the model's nose up with the direction of travel
        th = self.heading + math.pi / 2
        cs, sn = math.cos(th), math.sin(th)

        def P(bx, by):
            rx, ry = rot(bx * s, -by * s, cs, sn)
            return self.x + rx, self.y + ry

        amp = GAIT_AMP[0] + (GAIT_AMP[1] - GAIT_AMP[0]) * min(1.0, self.walk_speed / 260.0)
        grooming = self.state == "groom"
        for side, at, yaw, phase, is_front, femur, tibia, tarsus in LEG_SPECS:
            if grooming and is_front:
                # the front pair carries the grooming motion (upstream's note on
                # LEG_SPECS); they sweep together over the head
                g = math.sin(self.groom_phase) * 0.5
                base = (yaw if side > 0 else math.pi - yaw) + side * (0.85 + g)
                jx, jy = at
                pts = [P(jx, jy)]
                for seg, bend in ((femur, 0.0), (tibia, 0.9 + g), (tarsus, 1.3)):
                    bs = base + side * bend
                    jx += math.cos(bs) * seg * 0.82
                    jy += math.sin(bs) * seg * 0.82
                    pts.append(P(jx, jy))
                flat = [v for pt in pts for v in pt]
                c.create_line(*flat, fill=LEG_BROWN, width=max(1, int(2.2 * s)),
                              capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="fly")
                continue
            p = (self.gait + phase) % 1.0
            if p < STANCE_FRACTION:
                ang = amp * (1.0 - 2.0 * (p / STANCE_FRACTION))
                lift = 1.0
            else:
                u = (p - STANCE_FRACTION) / (1.0 - STANCE_FRACTION)
                ang = amp * (2.0 * u - 1.0)
                lift = 1.0 - 0.35 * math.sin(math.pi * u)   # foreshortens in top-down
            base = (yaw if side > 0 else math.pi - yaw) + side * ang
            jx, jy = at
            pts = [P(jx, jy)]
            for seg, bend in ((femur, 0.0), (tibia, 0.62), (tarsus, 0.95)):
                base_seg = base + side * bend
                jx += math.cos(base_seg) * seg * lift
                jy += math.sin(base_seg) * seg * lift
                pts.append(P(jx, jy))
            flat = [v for pt in pts for v in pt]
            c.create_line(*flat, fill=LEG_BROWN, width=max(1, int(2.2 * s)),
                          capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="fly")

        def oval(bx, by, rx, ry, fill, outline=""):
            pts = []
            for i in range(20):
                a = 2 * math.pi * i / 20
                pts.append(P(bx + rx * math.cos(a), by + ry * math.sin(a)))
            c.create_polygon([v for pt in pts for v in pt], fill=fill,
                             outline=outline, smooth=True, tags="fly")

        if self.walk_speed > 300.0 or self.fleeing:          # wingbeat smear
            for sd in (-1, 1):
                oval(sd * 5.2, 1.0, 5.0, 2.3, "#e8e8e8")
        else:
            for sd in (-1, 1):
                oval(sd * 1.9, -7.0, 2.6, 8.2, WING)

        oval(0.0, -6.5, 4.5, 7.5, ABDOMEN)                   # abdomen
        for band in (-10.5, -7.5, -4.6):
            oval(0.0, band, 4.2, 0.85, ABDOMEN_BAND)
        oval(0.0, 2.5, 4.4, 5.3, BODY_BROWN)                 # thorax
        oval(0.0, 9.0, 3.0, 2.6, HEAD_BROWN)                 # head
        for sd in (-1, 1):
            oval(sd * 2.1, 9.7, 1.6, 2.0, EYE_RED)           # compound eyes
            ax, ay = P(sd * 0.9, 11.4)
            bx, by = P(sd * 1.5, 13.4)
            c.create_line(ax, ay, bx, by, fill=LEG_BROWN,
                          width=max(1, int(1.4 * s)), tags="fly")

        if self.debug:
            self.draw_debug()

    def draw_debug(self):
        c = self.canvas
        lines = [
            self.worker.status,
            "state {:<6} evidence {:+.3f}".format(self.state, self.evidence),
            "descending {:6.2f} Hz   speed {:5.0f} px/s".format(self.drive, self.walk_speed),
            "cursor {:5.0f} px   threat {:.2f}".format(
                getattr(self, "cursor_dist", 0.0), getattr(self, "threat", 0.0)),
            "saccade" if self.sacc_t > 0 else "",
            "GIANT FIBER" if self.fleeing else "",
            "{:.0f} fps".format(self.fps),
        ]
        y = 24
        for t in lines:
            if not t:
                continue
            c.create_text(21, y + 1, text=t, anchor="w", fill="#000000",
                          font=("Consolas", 11), tags="fly")
            c.create_text(20, y, text=t, anchor="w", fill="#7CE0C0",
                          font=("Consolas", 11), tags="fly")
            y += 17

    def tick(self):
        now = time.perf_counter()
        dt = min(now - self.last, 0.05)
        self.last = now
        self.fps_n += 1
        if now - self.fps_t >= 0.5:
            self.fps = self.fps_n / (now - self.fps_t)
            self.fps_t, self.fps_n = now, 0
        self.sense()
        self.integrate(dt)
        self.draw()
        self.root.after(16, self.tick)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="brain_whole.npz")
    ap.add_argument("--scale", type=float, default=2.2, help="pixels per body unit")
    ap.add_argument("--tick-ms", type=float, default=200.0,
                    help="brain time simulated per command")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    worker = BrainWorker(a.brain, a.tick_ms, a.gain)
    worker.start()

    root = tk.Tk()
    root.title("desktop fly")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-transparentcolor", CHROMA)
    except tk.TclError:
        print("transparency unavailable - this needs Windows; the window will be opaque",
              file=sys.stderr)
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
    root.config(bg=CHROMA)

    print("loading the connectome (a few seconds)...  Esc or right-click quits")
    worker.ready.wait(timeout=180)
    print(worker.status)

    DesktopFly(root, worker, a.scale, a.debug)
    root.mainloop()
    worker.stop.set()


if __name__ == "__main__":
    main()
