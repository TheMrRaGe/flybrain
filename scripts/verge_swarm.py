#!/usr/bin/env python3
"""
verge_swarm.py - a population of fly-brained souls living in the Verge.

    # in the Verge repo (prototypes/stage-b):   SLOW=15 npm run serve
    python3 verge_swarm.py --flies 8 --url ws://127.0.0.1:8000

WHAT THIS IS
    N souls, one connectome. Each soul is a normal Verge agent client: it opens the
    same WebSocket a browser opens, gets the same fogged snapshot, and sends the same
    {t:"in"} - no privileged view, no privileged verb, and it announces itself as an
    agent. Behind each soul is one fly in a FlySwarm (flysim_gpu.py), stepped for one
    game tick (100 ms of brain time) per snapshot.

    Even-numbered souls KEEP their brain across deaths (the soulbound lineage that
    remembers); odd-numbered souls start each life naive. Same world, same Lieutenant,
    same seed: the inherit-vs-control experiment runs inside the game.

SENSES (world -> receptors), all by the same rules as flybox.py
    bush (food), water, fire, other souls, the Lieutenant, hostile creatures -> six
    odour plumes falling with distance, split across the two antennae by bearing
    cold                  -> thermosensory, from needs.warmth
    standing on a bush    -> sugar GRNs (LB3c/b + taste pegs), gain by hunger
    standing on water     -> water GRNs (LB3a), gain by thirst
    satiety went UP       -> PAM08 (reward)            } the world is the teacher,
    health went DOWN      -> PPL105 (punishment)       } plasticity always on
    PPL101 (MB-MP1) tonic when fed (Krashes 2009); sugar gain by hunger (Inagaki 2012)

ACTIONS (descending / motor neurons -> {t:"in"})
    heading      calibrated DNa left/right asymmetry -> continuous heading -> dx, dy
    speed        leg motor-neuron rate; below the stop floor the soul stands still
    escape       DNp01 -> a burst of speed away from the nearest threat
    action verb  the game's one contextual button (forage / drink / loot ...),
                 pressed when the PROBOSCIS motor neurons fire while standing on a
                 bush (derived), or when standing on water and thirsty (IMPOSED - the
                 water -> proboscis pathway is dark in the model; see FINDINGS)
    Crafting verbs are NOT bound yet. That is the next experiment, not this one.

RECORD
    results/verge/events.jsonl: every death (soul, lineage, cause, age, forage
    counts, inherit flag, weight state) and periodic population summaries.
"""
from __future__ import annotations

import argparse, asyncio, json, math, os, time
import numpy as np
import torch
import websockets

from flysim import Params
from flysim_gpu import FlySwarm
from motormap import _load_neuromere

TILE, WORLD_W = 1000, 576
T_WATER, T_BUSH, T_BARE, T_FIRE = 3, 4, 5, 6
HOSTILE = {"wolf", "hedge-boar", "bog-lynx", "ash-hound", "fen-wraith"}
NEED_MAX = 1000.0


def log(m): print(m, flush=True)


class Soul:
    def __init__(self, i, inherit, tribe=0):
        self.i, self.inherit, self.tribe = i, inherit, tribe
        self.kin_d, self.str_d = [], []      # distance to nearest kin / stranger, per tick
        self.id = None
        self.snap = None
        self.tick = -1
        self.heading = float(np.random.default_rng(i).uniform(0, 2 * math.pi))
        self.ev_base = None
        self.prev = None                     # previous (satiety, hydration, warmth, health)
        self.life = {"born": time.time(), "ticks": 0, "forage": 0, "drink": 0, "hits": 0,
                     "escapes": 0, "rewards": 0, "punish": 0}
        self.lives = 0
        self.dead_sent = False


class VergeSwarm:
    def __init__(self, a):
        self.a = a
        self.sw = FlySwarm(a.brain, n=a.flies, params=Params(gain=1.0, kc_thresh_scale=a.kc_thresh),
                           seed=a.seed, plasticity=True)
        cpu = self.sw.cpu
        od = json.load(open(a.odours))
        types = list(od["CS+"]) + list(od["CS-"])
        # five disjoint odours from the 16 designed glomeruli, plus the two PHEROMONE
        # glomeruli for other souls: kin on VA1v, strangers on DA1 (cVA - the male
        # pheromone channel, 204 receptors, the one we excluded from odour design).
        # Which one is "kin" is a modelling choice, labelled here.
        names = ["bush", "water", "fire", "lieutenant", "beast"]
        chunks = [types[0:3], types[3:6], types[6:9], types[9:13], types[13:16]]
        self.odours = dict(zip(names, chunks))
        self.odours["kin"] = ["ORN_VA1v"]
        self.odours["stranger"] = ["ORN_DA1"]
        for k, ts in self.odours.items():
            cpu._odor_map[k] = {t: 1.0 for t in ts}
        dev = self.sw.device
        dn = cpu.pop["DN"]; nm = cpu.type[dn].astype(str); sd = cpu.side[dn]
        fam = np.char.startswith(nm, "DNa")
        self.L = torch.as_tensor(dn[fam & (sd == "L")], device=dev)
        self.R = torch.as_tensor(dn[fam & (sd == "R")], device=dev)
        self.escape = torch.as_tensor(dn[np.char.startswith(nm, "DNp01")], device=dev)
        ty = cpu.type.astype(str); sc = cpu.sc.astype(str)
        neuro = _load_neuromere(cpu, a.data_dir)
        motor = sc == "vnc_motor"
        wingish = (np.char.startswith(ty, "DLMn") | np.char.startswith(ty, "DVMn")
                   | np.char.startswith(ty, "MNwm") | np.char.startswith(ty, "MNhm"))
        self.leg = torch.as_tensor(np.flatnonzero(motor & np.isin(neuro, ["T1", "T2", "T3"]) & ~wingish), device=dev)
        cm = cpu.pop["motor"]
        self.feed = torch.as_tensor(cm[np.isin(ty[cm], cpu.FEEDING_MN)], device=dev)
        self.thermo = torch.as_tensor(cpu.pop.get("thermo", np.zeros(0, int)), device=dev)
        g = cpu.pop["gustatory"]; gty = ty[g]
        self.g_all = torch.as_tensor(g, device=dev)
        self.g_sweet = torch.as_tensor(g[np.isin(gty, cpu.TASTE_SWEET)], device=dev)
        self.g_water = torch.as_tensor(g[np.isin(gty, cpu.TASTE_WATER)], device=dev)
        self.ppl101 = self.sw.da_types["PPL101"][0]
        inherit_tribes = {int(x) for x in a.inherit_tribes.split(",") if x.strip() != ""}
        self.souls = [Soul(i, inherit=(i // a.per_tribe) in inherit_tribes, tribe=i // a.per_tribe)
                      for i in range(a.flies)]
        self.tribe_of_id = {}                 # player id -> tribe, learned from welcomes
        self.steps = int(round(a.tick_ms / self.sw.dt))
        os.makedirs(a.out_dir, exist_ok=True)
        self.logf = open(os.path.join(a.out_dir, "events.jsonl"), "a")
        self.game_ticks = 0

    # -- senses ---------------------------------------------------------------- #

    def sense(self, s: Soul):
        sw, snap = self.sw, s.snap
        me = snap["players"][s.id] if s.id is not None and s.id < len(snap["players"]) else None
        if not me or not me.get("alive", True):
            return None
        x, y = me["x"], me["y"]
        needs = me["needs"]
        hunger = 1.0 - needs["satiety"] / NEED_MAX
        thirst = 1.0 - needs["hydration"] / NEED_MAX
        cold = 1.0 - needs["warmth"] / NEED_MAX
        # where things are (fogged snapshot: only what this soul can see)
        pts = {k: [] for k in self.odours}
        for idx, tile in snap.get("visibleTiles", []):
            tx, ty_ = (idx % WORLD_W) * TILE + TILE // 2, (idx // WORLD_W) * TILE + TILE // 2
            if tile == T_BUSH: pts["bush"].append((tx, ty_))
            elif tile == T_WATER: pts["water"].append((tx, ty_))
            elif tile == T_FIRE: pts["fire"].append((tx, ty_))
        for idx, _fuel in snap.get("fires", []):
            pts["fire"].append(((idx % WORLD_W) * TILE + TILE // 2, (idx // WORLD_W) * TILE + TILE // 2))
        for j, p in enumerate(snap.get("players", [])):
            if p and j != s.id and p.get("alive", True):
                same = self.tribe_of_id.get(j) == s.tribe
                pts["kin" if same else "stranger"].append((p["x"], p["y"]))
        lt = snap.get("lieutenant")
        if lt: pts["lieutenant"].append((lt["x"], lt["y"]))
        for c in snap.get("creatures", []):
            if c.get("kind") in HOSTILE: pts["beast"].append((c["x"], c["y"]))
        # plumes: concentration 0..1 by distance (in tiles), bearing relative to heading
        ax, ay = math.cos(s.heading), math.sin(s.heading)
        left, right = {}, {}
        nearest = {}
        for k, plist in pts.items():
            best = None
            for (px, py) in plist:
                d = math.hypot(px - x, py - y) / TILE
                if best is None or d < best[0]: best = (d, px, py)
            if best is None: continue
            d, px, py = best
            nearest[k] = best
            if k == "kin": s.kin_d.append(d)
            elif k == "stranger": s.str_d.append(d)
            c = max(0.0, 1.0 - d / self.a.plume_tiles)
            if c <= 0.01: continue
            br = math.sin(math.atan2(py - y, px - x) - s.heading)      # -1 left .. +1 right
            left[k] = c * (1.0 - 0.4 * max(0.0, br))
            right[k] = c * (1.0 - 0.4 * max(0.0, -br))
        f = s.i
        sw.clear_senses(f)
        if left or right:
            sw.smell(f, {k: self.a.strength * v for k, v in left.items()},
                        {k: self.a.strength * v for k, v in right.items()})
        # contact chemosensation: the tile under the soul
        on_idx = (y // TILE) * WORLD_W + (x // TILE)
        tile_here = dict(snap.get("visibleTiles", [])).get(on_idx, 0)
        on_bush = tile_here == T_BUSH or ("bush" in nearest and nearest["bush"][0] < 0.8)
        on_water = tile_here == T_WATER or ("water" in nearest and nearest["water"][0] < 1.0)
        sw.drive_hz[f, self.g_all] = 0.0
        if on_bush:
            sw.drive_hz[f, self.g_sweet] = sw.p.max_rate_hz * (0.5 + 0.5 * hunger)
        elif on_water:
            sw.drive_hz[f, self.g_water] = sw.p.max_rate_hz * min(1.0, 0.2 + 0.8 * thirst)
        if len(self.thermo):
            sw.drive_hz[f, self.thermo] = 200.0 * min(1.0, cold)
        # hunger through its targets (Krashes 2009): MB-MP1 tonic when fed
        sw.quiet_dopamine(f)
        sw.ext[f, self.ppl101] = 8.0 * (1.0 - hunger)
        # the world teaches: reward on satiety/hydration gain, punishment on damage
        cur = (needs["satiety"], needs["hydration"], needs["warmth"], me.get("health", 0))
        if s.prev is not None:
            if cur[0] > s.prev[0] + 1 or cur[1] > s.prev[1] + 1:
                sw.stimulate_type(f, "PAM08", 70.0); s.life["rewards"] += 1
            if cur[3] < s.prev[3]:
                sw.stimulate_type(f, "PPL105", 70.0); s.life["punish"] += 1; s.life["hits"] += 1
        s.prev = cur
        return {"x": x, "y": y, "hunger": hunger, "thirst": thirst, "cold": cold,
                "on_bush": on_bush, "on_water": on_water, "nearest": nearest}

    # -- one game tick for the whole population --------------------------------- #

    def step_all(self, ctx):
        sw = self.sw
        B = sw.n
        acc = {k: torch.zeros(B, device=sw.device) for k in ("L", "R", "leg", "feed", "esc")}
        for _ in range(self.steps):
            spk = sw.step()
            acc["L"] += spk[:, self.L].sum(1); acc["R"] += spk[:, self.R].sum(1)
            acc["leg"] += spk[:, self.leg].sum(1); acc["feed"] += spk[:, self.feed].sum(1)
            acc["esc"] += spk[:, self.escape].sum(1)
        out = {k: v.cpu().numpy() for k, v in acc.items()}
        cmds = {}
        for s in self.souls:
            c = ctx.get(s.i)
            if c is None: continue
            f = s.i
            L, R = out["L"][f] / max(len(self.L), 1), out["R"][f] / max(len(self.R), 1)
            tot = L + R
            frac = R / tot if tot > 1e-6 else 0.5
            s.ev_base = frac if s.ev_base is None else s.ev_base + (frac - s.ev_base) * 0.02
            evidence = float(np.clip((frac - s.ev_base) * 12.0, -1.0, 1.0))
            leg = out["leg"][f] * (150.0 / self.a.tick_ms)         # scale to the box's per-150ms numbers
            stopped = leg < 130.0
            esc = out["esc"][f] > 0
            if esc:
                s.life["escapes"] += 1
                # away from the nearest threat if one is in view
                thr = [c["nearest"][k] for k in ("lieutenant", "beast") if k in c["nearest"]]
                if thr:
                    d, px, py = min(thr)
                    s.heading = math.atan2(c["y"] - py, c["x"] - px)
            s.heading += evidence * 0.5
            if stopped and not esc:
                dx = dy = 0
            else:
                dx = int(round(math.cos(s.heading))); dy = int(round(math.sin(s.heading)))
            verbs = []
            feed_ok = out["feed"][f] * (150.0 / self.a.tick_ms) >= 60.0
            if c["on_bush"] and feed_ok:
                verbs.append("action"); s.life["forage"] += 1; dx = dy = 0
            elif c["on_water"] and c["thirst"] > 0.1:
                verbs.append("action"); s.life["drink"] += 1; dx = dy = 0      # IMPOSED
            cmds[s.i] = {"t": "in", "dx": dx, "dy": dy, "verbs": verbs, "slot": 0}
            s.life["ticks"] += 1
        return cmds

    # -- deaths ------------------------------------------------------------------ #

    def on_death(self, s: Soul, me):
        rec = {"event": "death", "soul": s.i, "tribe": s.tribe, "inherit": s.inherit, "lineage": me.get("lineage"),
               "kin_dist": float(np.mean(s.kin_d)) if s.kin_d else None,
               "stranger_dist": float(np.mean(s.str_d)) if s.str_d else None,
               "life_ticks": s.life["ticks"], "life": dict(s.life),
               "weights": float(self.sw.weights_frac()[s.i]), "game_tick": s.snap.get("snap", {}).get("tick", s.tick)}
        self.logf.write(json.dumps(rec) + "\n"); self.logf.flush()
        log(f"  soul {s.i} ({'inherit' if s.inherit else 'naive'}) died after {s.life['ticks']} ticks: "
            f"forage {s.life['forage']} drink {s.life['drink']} hits {s.life['hits']} rewards {s.life['rewards']} "
            f"weights {100*rec['weights']:.2f}%")
        s.lives += 1
        s.life = {"born": time.time(), "ticks": 0, "forage": 0, "drink": 0, "hits": 0,
                  "escapes": 0, "rewards": 0, "punish": 0}
        s.prev = None; s.kin_d, s.str_d = [], []
        self.sw.reset(flies=[s.i])
        if not s.inherit:
            self.sw.reset_weights(flies=[s.i])

    # -- network -------------------------------------------------------------- #

    async def soul_link(self, s: Soul, inbox: asyncio.Queue):
        async with websockets.connect(self.a.url, max_size=None) as ws:
            s.ws = ws
            await ws.send(json.dumps({"t": "hello", "name": f"Fly-{s.tribe}-{s.i:02d}", "kind": "agent",
                                      "family": f"Tribe-{s.tribe}"}))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("t") == "welcome":
                    s.id = msg["id"]
                    self.tribe_of_id[s.id] = s.tribe
                    log(f"  soul {s.i} joined as player {s.id} ({'inherit' if s.inherit else 'naive'})")
                elif msg.get("t") == "state":
                    s.snap = msg.get("snap", msg)
                    s.tick = s.snap.get("tick", s.tick + 1)
                    await inbox.put(s.i)

    async def run(self):
        inbox: asyncio.Queue = asyncio.Queue()
        links = [asyncio.create_task(self.soul_link(s, inbox)) for s in self.souls]
        log(f"  {len(self.souls)} souls connecting to {self.a.url} ...")
        seen_tick = {}
        t0 = time.time()
        last_report = 0
        try:
            while True:
                # wait until every joined soul has a snapshot newer than the last one we acted on
                await inbox.get()
                ready = [s for s in self.souls if s.snap is not None and s.id is not None and s.tick > seen_tick.get(s.i, -1)]
                joined = [s for s in self.souls if s.id is not None]
                if len(ready) < len(joined) or not joined:
                    continue
                ctx = {}
                for s in ready:
                    seen_tick[s.i] = s.tick
                    me = s.snap["players"][s.id] if s.id < len(s.snap["players"]) else None
                    if me and not me.get("alive", True):
                        if not s.dead_sent:
                            self.on_death(s, me)
                            await s.ws.send(json.dumps({"t": "respawn"}))
                            s.dead_sent = True
                        continue
                    s.dead_sent = False
                    c = self.sense(s)
                    if c: ctx[s.i] = c
                cmds = self.step_all(ctx)
                for s in ready:
                    if s.i in cmds:
                        await s.ws.send(json.dumps(cmds[s.i]))
                self.game_ticks += 1
                if time.time() - last_report > 30:
                    last_report = time.time()
                    alive = sum(1 for s in self.souls if s.id is not None and not s.dead_sent)
                    wf = self.sw.weights_frac()
                    log(f"  t={self.game_ticks:6d} ticks ({time.time()-t0:5.0f}s wall)  alive {alive}/{len(joined)}")
                    for tr in sorted({s.tribe for s in self.souls}):
                        ss = [s for s in self.souls if s.tribe == tr]
                        kd = [np.mean(s.kin_d[-50:]) for s in ss if s.kin_d]; sd_ = [np.mean(s.str_d[-50:]) for s in ss if s.str_d]
                        log(f"    tribe {tr} ({'inherit' if ss[0].inherit else 'naive'}): forage {sum(s.life['forage'] for s in ss)} "
                            f"drink {sum(s.life['drink'] for s in ss)} hits {sum(s.life['hits'] for s in ss)} deaths {sum(s.lives for s in ss)} "
                            f"weights {100*np.mean([wf[s.i] for s in ss]):.2f}%  nearest kin {np.mean(kd) if kd else float('nan'):.1f} "
                            f"stranger {np.mean(sd_) if sd_ else float('nan'):.1f} tiles")
                    self.logf.write(json.dumps({"event": "summary", "ticks": self.game_ticks,
                                                "deaths": [s.lives for s in self.souls],
                                                "weights": wf.tolist()}) + "\n"); self.logf.flush()
                if self.a.max_ticks and self.game_ticks >= self.a.max_ticks:
                    break
        finally:
            for t in links: t.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000")
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--odours", default="../results/odours3.json")
    ap.add_argument("--tribes", type=int, default=2)
    ap.add_argument("--per-tribe", type=int, default=4)
    ap.add_argument("--inherit-tribes", default="0", help="comma list of tribes whose brains persist across deaths")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--kc-thresh", type=float, default=1.0)
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--plume-tiles", type=float, default=12.0)
    ap.add_argument("--tick-ms", type=float, default=100.0, help="brain time per game tick (10 Hz sim)")
    ap.add_argument("--max-ticks", type=int, default=0)
    ap.add_argument("--out-dir", default="../results/verge")
    a = ap.parse_args()
    a.flies = a.tribes * a.per_tribe
    log(f"loading swarm of {a.flies} ({a.tribes} tribes x {a.per_tribe}) ...")
    vs = VergeSwarm(a)
    log(f"  device {vs.sw.device}; odours " + ", ".join(f"{k}={v}" for k, v in vs.odours.items()))
    asyncio.run(vs.run())


if __name__ == "__main__":
    main()
