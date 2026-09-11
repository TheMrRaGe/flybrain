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
from eye import Eye

DAY_TICKS, NIGHT_TICKS = 3000, 3000          # the Verge's calendar (calendar.ts)
# The Verge's crafting table (prototypes/stage-b/src/sim/tick.ts RECIPES + the free verbs):
# verb -> (what it makes, materials that must be in the pack). The game enforces amounts
# and the workbench/fire requirement; a press without them does nothing.
VERBS = {
    "gather":          ("wood/stone/berries", []),          # the action button beside a tree, rock or bush
    "eat":             ("a meal", ["cookedMeat"]),
    "cook":            ("cooked meat", ["rawMeat"]),
    "build":           ("fire", ["wood"]),
    "makeSpear":       ("spear", ["wood"]),
    "makeHammer":      ("hammer", ["stone", "wood"]),
    "makeKnife":       ("knife", ["stone", "wood"]),
    "makeAxe":         ("axe", ["stone", "wood"]),
    "makePickaxe":     ("pickaxe", ["stone", "wood"]),
    "makeHoe":         ("hoe", ["stone", "wood"]),
    "makeCordage":     ("cordage", ["hide", "knife"]),
    "makeCloak":       ("cloak", ["hide"]),
    "makeBandage":     ("bandage", ["hide", "cordage"]),
    "makeBoots":       ("boots", ["hide", "cordage"]),
    "makeGloves":      ("gloves", ["hide", "cordage"]),
    "makeBackpack":    ("backpack", ["hide", "cordage"]),
    "makeWaterskin":   ("waterskin", ["hide", "cordage"]),
    "fillWaterskin":   ("full waterskin", ["waterskin"]),
    "setSnare":        ("snare", ["cordage", "wood"]),
    "makeTorch":       ("torch", ["wood", "pitch"]),
    "makeCharcoal":    ("charcoal", ["wood"]),
    "makePot":         ("pot", ["clay"]),
    "makeSkep":        ("skep", ["cordage", "clay"]),
    "smelt":           ("bar", ["ore", "charcoal"]),
    "makeSword":       ("sword", ["bar", "wood", "cordage"]),
    "makeBow":         ("bow", ["wood", "cordage", "pitch"]),
    "makeArrow":       ("arrow", ["wood", "glue", "knife"]),
    "dryMeat":         ("dried meat", ["rawMeat"]),
    "sawPlanks":       ("plank", ["wood"]),
    "dressBlocks":     ("block", ["stone"]),
    "buildMasonBench": ("mason bench", ["stone", "wood", "hammer"]),
    "buildSawpit":     ("sawpit", ["block", "wood", "hammer"]),
    "buildWoodWall":   ("wood wall", ["plank", "hammer"]),
    "buildStoneWall":  ("stone wall", ["block", "hammer"]),
    "buildDoor":       ("door", ["plank", "hammer"]),
    "buildBed":        ("bed", ["plank", "hammer"]),
    "buildChest":      ("chest", ["plank", "hammer"]),
    "buildWallTorch":  ("wall torch", ["wood", "hammer"]),
    "buildDryingRack": ("drying rack", ["plank", "cordage", "hammer"]),
    "buildWell":       ("well", ["block", "clay", "hammer"]),
    "buildHearth":     ("hearth", ["block", "stone", "hammer"]),
    "buildSignpost":   ("signpost", ["plank", "hammer"]),
    "buildBridge":     ("bridge", ["plank", "hammer"]),
    "buildWard":       ("ward", ["block", "crowns", "hammer"]),
    "setHome":         ("a home", []),
}
# pack fields that are THINGS MADE (a tool, a garment, a container); materials gained
# count as gathers. Both are rewarded; only these count as crafts.
MADE = {"spear", "hammer", "knife", "axe", "pickaxe", "hoe", "cordage", "cloak", "bandage", "boots", "gloves",
        "backpack", "waterskin", "snare", "torch", "charcoal", "pot", "skep", "bar", "sword", "bow", "arrow",
        "driedMeat", "plank", "block", "cookedMeat", "ironKnife", "ironAxe", "ironPickaxe", "ironHoe"}


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
                     "escapes": 0, "rewards": 0, "punish": 0, "overeat": 0, "gathered": 0, "gather_tries": 0, "craft_tries": 0, "crafted": []}
        self.crafted_ever = []
        self.prev_pack = None
        self.lives = 0
        self.dead_sent = False
        self.intent = None
        self.decisions = 0
        self.err_x = self.err_y = 0.0


class VergeSwarm:
    def __init__(self, a):
        self.a = a
        self.sw = FlySwarm(a.brain, n=a.flies, params=Params(gain=1.0, kc_thresh_scale=a.kc_thresh),
                           seed=a.seed, plasticity=True, vision=not a.blind)
        cpu = self.sw.cpu
        # SIGHT, limited to what the model reproduces: luminance and dark blobs reaching the
        # descending neurons. The looming -> Giant Fiber pathway is dark here (T4/T5 silent:
        # the optic-lobe periphery is graded in life; see FINDINGS). ~880 pixels per eye.
        self.eye = Eye(cpu, a.data_dir) if not a.blind else None
        # brain regions for the activity view (superclass / class / type), per fly
        sc = cpu.sc.astype(str); cl = cpu.cls.astype(str); tyx = cpu.type.astype(str); side = cpu.side.astype(str)
        dev0 = self.sw.device
        def T(mask): return torch.as_tensor(np.flatnonzero(mask), device=dev0)
        ol = np.isin(sc, ["ol_sensory", "ol_intrinsic"])
        self.regions = {
            "eye_L": T(ol & (side == "L")), "eye_R": T(ol & (side == "R")),
            "vis_proj": T(sc == "visual_projection"),
            "antennal_lobe": T(np.isin(cl, ["ALPN", "ALLN", "olfactory"])),
            "mushroom_body": T(np.isin(cl, ["Kenyon_Cell", "MBON", "DAN"])),
            "lateral_horn": T(np.char.startswith(tyx, "LH")),
            "central_complex": T(cl == "CX"),
            "sez_taste": T(np.isin(cl, ["gustatory"]) | (sc == "cb_motor")),
            "central_other": T((sc == "cb_intrinsic") & ~np.isin(cl, ["ALPN", "ALLN", "Kenyon_Cell", "MBON", "DAN", "CX"]) & ~np.char.startswith(tyx, "LH")),
            "descending": T(sc == "descending_neuron"),
            "vnc": T(np.isin(sc, ["vnc_intrinsic", "vnc_motor", "ascending_neuron"])),
        }
        self.region_n = {k: max(int(v.numel()), 1) for k, v in self.regions.items()}
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
        # THE NOVEL-EFFECTOR EXPERIMENT (flyworld.py's GRAB, made real, for the whole
        # craft list). Every game verb a fly has no circuit for is bound to one descending
        # -neuron TYPE with no established role - the largest unassigned types, in a fixed
        # order, so the binding is reproducible. A verb is pressed when its type fires
        # above its own running baseline and the pack holds the verb's materials. The game
        # refuses anything else. Anything new in the pack is rewarded through PAM08.
        # The binding is an interface choice, labelled; what the brains do with it is the
        # experiment.
        import collections as _c
        dn_types = _c.Counter(nm.tolist())
        free = sorted([t for t, n in dn_types.items() if n >= 2 and t.startswith("DNg") and t not in ("DNg33",)],
                      key=lambda t: (-dn_types[t], t))
        self.pool_names = {verb: free[i] for i, verb in enumerate(VERBS)}
        self.pools = {verb: torch.as_tensor(dn[nm == t], device=dev) for verb, t in self.pool_names.items()}
        self.pool_stats = {i: {k: [0.0, 0.0, 0] for k in VERBS} for i in range(a.flies)}   # mean, M2, n per soul
        inherit_tribes = {int(x) for x in a.inherit_tribes.split(",") if x.strip() != ""}
        self.souls = [Soul(i, inherit=(i // a.per_tribe) in inherit_tribes, tribe=i // a.per_tribe)
                      for i in range(a.flies)]
        self.tribe_of_id = {}                 # player id -> tribe, learned from welcomes
        self.steps = int(round(a.tick_ms / self.sw.dt))
        os.makedirs(a.out_dir, exist_ok=True)
        self.logf = open(os.path.join(a.out_dir, "events.jsonl"), "a")
        self.state_path = os.path.join(a.out_dir, "state.json")
        self.deaths = []
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
        if self.eye is not None:
            night = (s.tick % (DAY_TICKS + NIGHT_TICKS)) >= DAY_TICKS
            ambient = 0.15 if night else 0.9
            scene = []
            for k, plist in pts.items():
                lum = 1.0 if k == "fire" else 0.1
                size = 1.2 if k in ("lieutenant", "beast", "kin", "stranger") else 1.0
                for (px, py) in sorted(plist, key=lambda q: math.hypot(q[0] - x, q[1] - y))[:24]:
                    d = math.hypot(px - x, py - y) / TILE
                    if d < 14: scene.append((math.atan2(py - y, px - x), d, size, lum))
            sw.see(f, self.eye.idx, self.eye.render(scene, s.heading, ambient))
        if left or right:
            sw.smell(f, {k: self.a.strength * v for k, v in left.items()},
                        {k: self.a.strength * v for k, v in right.items()})
        # crafting context: what is in reach and in the pack
        near_tree = near_rock = False
        for idx, tr in snap.get("visibleTrees", []):
            if tr.get("standing", True) is False: continue
            tx, ty_ = (idx % WORLD_W) * TILE + TILE // 2, (idx // WORLD_W) * TILE + TILE // 2
            if math.hypot(tx - x, ty_ - y) / TILE < 1.6: near_tree = True; break
        for idx, tile in snap.get("visibleTiles", []):
            if tile in (8, 9, 10):                       # Rock-ish tiles (world.ts Tile enum: rock outcrops)
                tx, ty_ = (idx % WORLD_W) * TILE + TILE // 2, (idx // WORLD_W) * TILE + TILE // 2
                if math.hypot(tx - x, ty_ - y) / TILE < 1.6: near_rock = True; break
        pack = dict(me.get("pack") or {})
        s.craft_ctx = {"tree_near": near_tree, "rock_near": near_rock, "pack": pack, "at_fire": bool(me.get("atFire"))}
        # contact chemosensation: the tile under the soul
        on_idx = (y // TILE) * WORLD_W + (x // TILE)
        tile_here = dict(snap.get("visibleTiles", [])).get(on_idx, 0)
        on_bush = tile_here == T_BUSH or ("bush" in nearest and nearest["bush"][0] < 0.8)
        on_water = tile_here == T_WATER or ("water" in nearest and nearest["water"][0] < 1.0)
        sw.drive_hz[f, self.g_all] = 0.0
        # A full fly stops. Crop distension is read by Piezo mechanosensory neurons and
        # ends the meal (Min et al. 2021, eLife; Piezo-null flies overeat until the crop
        # is grossly enlarged). The game clamps satiety at NEED_MAX but still strips the
        # bush, so an unstopped fly strips its tribe's food for nothing. Sugar drive fades
        # over the last quarter of the crop and is zero from 90 % (the stretch stop).
        full = float(np.clip((1.0 - hunger - 0.75) / 0.15, 0.0, 1.0))
        crop_stop = full >= 1.0
        if on_bush and not crop_stop:
            sw.drive_hz[f, self.g_sweet] = sw.p.max_rate_hz * (0.5 + 0.5 * hunger) * (1.0 - full)
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
        # anything new in the pack is a gain: materials are gathers, made things are crafts
        if s.prev_pack is not None:
            for k, v in pack.items():
                if isinstance(v, (int, float)) and v > s.prev_pack.get(k, 0):
                    sw.stimulate_type(f, "PAM08", 70.0); s.life["rewards"] += 1
                    if k in MADE and s.prev_pack.get(k, 0) == 0:
                        s.life["crafted"].append(k); self.note(s, "crafted", {"item": k})
                    elif k not in MADE:
                        s.life["gathered"] += 1; self.note(s, "gathered", {"material": k, "amount": v})
        s.prev_pack = pack
        return {"x": x, "y": y, "hunger": hunger, "thirst": thirst, "cold": cold, "crop_stop": crop_stop,
                "on_bush": on_bush, "on_water": on_water, "nearest": nearest}

    # -- one game tick for the whole population --------------------------------- #

    def step_all(self, ctx):
        """100 ms of brain for every soul with a fresh snapshot -> an INTENT per soul:
        continuous heading, speed 0..1, and whether to press the action button. The
        10 Hz sender turns intent into the game's {dx, dy} every tick."""
        sw = self.sw
        B = sw.n
        acc = {k: torch.zeros(B, device=sw.device) for k in ("L", "R", "leg", "feed", "esc")}
        reg = {k: torch.zeros(B, device=sw.device) for k in self.regions}
        pl = {k: torch.zeros(B, device=sw.device) for k in self.pools}
        for _ in range(self.steps):
            spk = sw.step()
            acc["L"] += spk[:, self.L].sum(1); acc["R"] += spk[:, self.R].sum(1)
            acc["leg"] += spk[:, self.leg].sum(1); acc["feed"] += spk[:, self.feed].sum(1)
            acc["esc"] += spk[:, self.escape].sum(1)
            for k, idx in self.regions.items(): reg[k] += spk[:, idx].sum(1)
            for k, idx in self.pools.items(): pl[k] += spk[:, idx].sum(1)
        out = {k: v.cpu().numpy() for k, v in acc.items()}
        pln = {k: v.cpu().numpy() for k, v in pl.items()}
        regn = {k: v.cpu().numpy() for k, v in reg.items()}
        secs = self.a.tick_ms / 1000.0
        for s in self.souls:
            s.regions = {k: float(regn[k][s.i] / self.region_n[k] / secs) for k in self.regions}   # Hz per cell
        for s in self.souls:
            c = ctx.get(s.i)
            if c is None: continue
            f = s.i
            L, R = out["L"][f] / max(len(self.L), 1), out["R"][f] / max(len(self.R), 1)
            tot = L + R
            frac = R / tot if tot > 1e-6 else 0.5
            s.ev_base = frac if s.ev_base is None else s.ev_base + (frac - s.ev_base) * 0.02
            evidence = float(np.clip((frac - s.ev_base) * 12.0, -1.0, 1.0))
            leg = out["leg"][f] * (150.0 / self.a.tick_ms)         # the box's per-150 ms scale
            esc = out["esc"][f] > 0
            if esc:
                s.life["escapes"] += 1
                thr = [c["nearest"][k] for k in ("lieutenant", "beast") if k in c["nearest"]]
                if thr:
                    d, px, py = min(thr)
                    s.heading = math.atan2(c["y"] - py, c["x"] - px)
            s.heading += evidence * 0.5
            # speed from leg motor drive: rest ~262 -> 0.34, odour ~490 -> 0.9, stop < 130
            speed = 0.0 if leg < 130.0 else min(1.0, (leg - 130.0) / 390.0)
            if esc: speed = 1.0
            feed_ok = out["feed"][f] * (150.0 / self.a.tick_ms) >= 60.0
            act = False
            if c["on_bush"] and feed_ok and c["crop_stop"]:
                # feeding drive on a distended crop: the fed-state valence of the brain's
                # own fructose sensor is aversive (Gr43a, Miyamoto et al. 2012, Cell) - a
                # mild punishment, half a hit, so overeating is taught against, not only blocked
                sw.stimulate_type(f, "PPL105", 35.0); s.life["overeat"] += 1
            elif c["on_bush"] and feed_ok:
                act = True; s.life["forage"] += 1; speed = 0.0
            elif c["on_water"] and c["thirst"] > 0.1:
                act = True; s.life["drink"] += 1; speed = 0.0                # IMPOSED
            # verb pools: a verb fires when its DN type is above its own running baseline
            # and the pack holds the materials; the action button also on a tree or rock
            verbs = []
            cc = getattr(s, "craft_ctx", None) or {}
            pack = cc.get("pack", {})
            hot = {}
            for k in self.pools:
                v = float(pln[k][f]); st = self.pool_stats[f][k]
                st[2] += 1; d = v - st[0]; st[0] += d / st[2]; st[1] += d * (v - st[0])
                sd = math.sqrt(st[1] / max(st[2] - 1, 1)) if st[2] > 2 else 1e9
                hot[k] = st[2] > 20 and v > st[0] + 1.5 * sd
            if hot.get("gather") and (cc.get("tree_near") or cc.get("rock_near")):
                verbs.append("action"); s.life["gather_tries"] += 1; act = True; speed = 0.0
            for k, (made, needs) in VERBS.items():
                if k in ("gather",) or not hot.get(k): continue
                if all(pack.get(m, 0) > 0 for m in needs):
                    verbs.append(k); s.life["craft_tries"] += 1
            s.intent = {"heading": s.heading, "speed": speed, "act": act, "evidence": evidence,
                        "leg": float(leg), "feed": float(out["feed"][f]), "escape": bool(esc),
                        "verbs": verbs, "hot": [k for k, h in hot.items() if h]}
            s.decisions += 1
            s.life["ticks"] += 1

    def tick_input(self, s):
        """Every game tick: dither the continuous heading across the game's eight
        directions so the path follows the heading instead of snapping to it."""
        it = s.intent
        extra = list(it.get("verbs", [])) if it else []
        if it: it["verbs"] = []                                   # each verb is pressed once per decision
        if it is None or it["speed"] <= 0.0:
            vs = (["action"] if (it and it["act"]) else []) + [v for v in extra if v != "action"]
            return {"t": "in", "dx": 0, "dy": 0, "verbs": vs, "slot": 0}
        if extra:
            return {"t": "in", "dx": 0, "dy": 0, "verbs": extra, "slot": 0}
        s.err_x += math.cos(it["heading"]) * it["speed"]
        s.err_y += math.sin(it["heading"]) * it["speed"]
        dx = dy = 0
        if abs(s.err_x) >= 0.5:
            dx = 1 if s.err_x > 0 else -1; s.err_x -= dx
        if abs(s.err_y) >= 0.5:
            dy = 1 if s.err_y > 0 else -1; s.err_y -= dy
        return {"t": "in", "dx": dx, "dy": dy, "verbs": [], "slot": 0}

    def write_state(self):
        wf = self.sw.weights_frac()
        souls = []
        for s in self.souls:
            me = None
            if s.snap and s.id is not None and s.id < len(s.snap.get("players", [])):
                me = s.snap["players"][s.id]
            souls.append({"i": s.i, "id": s.id, "name": "Fly-%d-%02d" % (s.tribe, s.i), "tribe": s.tribe,
                          "inherit": s.inherit, "lives": s.lives, "alive": bool(me and me.get("alive", True)),
                          "x": me["x"] if me else None, "y": me["y"] if me else None,
                          "needs": me.get("needs") if me else None, "health": me.get("health") if me else None,
                          "lineage": me.get("lineage") if me else None, "life": s.life, "decisions": s.decisions,
                          "weights": float(wf[s.i]), "intent": s.intent, "regions": getattr(s, "regions", None),
                          "crafted_ever": s.crafted_ever, "pack": (getattr(s, "craft_ctx", None) or {}).get("pack", {}),
                          "kin_d": float(np.mean(s.kin_d[-20:])) if s.kin_d else None,
                          "stranger_d": float(np.mean(s.str_d[-20:])) if s.str_d else None})
        st = {"ticks": self.game_ticks, "wall": time.time(), "souls": souls, "pool_names": self.pool_names,
              "deaths": self.deaths[-30:], "odours": self.odours}
        for path in [self.state_path] + ([self.a.state_copy] if self.a.state_copy else []):
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f)
            for _ in range(5):
                try:
                    os.replace(tmp, path); break
                except PermissionError:
                    time.sleep(0.05)

    # -- deaths ------------------------------------------------------------------ #

    def note(self, s, event, extra):
        rec = {"event": event, "soul": s.i, "tribe": s.tribe, "inherit": s.inherit, "game_tick": s.tick,
               "life_ticks": s.life["ticks"], **extra}
        self.logf.write(json.dumps(rec) + "\n"); self.logf.flush()
        if event == "crafted":
            s.crafted_ever.append({"item": extra["item"], "tick": s.tick, "life": s.lives + 1})
            log("  *** soul %d (Tribe-%d, %s) CRAFTED %s at tick %d ***" % (s.i, s.tribe, "inherit" if s.inherit else "naive", extra["item"], s.tick))

    def on_death(self, s: Soul, me):
        rec = {"event": "death", "soul": s.i, "tribe": s.tribe, "inherit": s.inherit, "lineage": me.get("lineage"),
               "kin_dist": float(np.mean(s.kin_d)) if s.kin_d else None,
               "stranger_dist": float(np.mean(s.str_d)) if s.str_d else None,
               "life_ticks": s.life["ticks"], "life": dict(s.life),
               "weights": float(self.sw.weights_frac()[s.i]), "game_tick": s.snap.get("snap", {}).get("tick", s.tick)}
        self.logf.write(json.dumps(rec) + "\n"); self.logf.flush()
        self.deaths.append(rec)
        log(f"  soul {s.i} ({'inherit' if s.inherit else 'naive'}) died after {s.life['ticks']} ticks: "
            f"forage {s.life['forage']} drink {s.life['drink']} hits {s.life['hits']} rewards {s.life['rewards']} "
            f"weights {100*rec['weights']:.2f}%")
        s.lives += 1
        s.life = {"born": time.time(), "ticks": 0, "forage": 0, "drink": 0, "hits": 0,
                  "escapes": 0, "rewards": 0, "punish": 0, "overeat": 0, "gathered": 0, "gather_tries": 0, "craft_tries": 0, "crafted": []}
        s.prev_pack = None
        s.prev = None; s.kin_d, s.str_d = [], []; s.intent = None
        self.sw.reset(flies=[s.i])
        if not s.inherit:
            self.sw.reset_weights(flies=[s.i])

    # -- network -------------------------------------------------------------- #

    async def soul_link(self, s):
        async with websockets.connect(self.a.url, max_size=None) as ws:
            s.ws = ws
            await ws.send(json.dumps({"t": "hello", "name": "Fly-%d-%02d" % (s.tribe, s.i), "kind": "agent",
                                      "family": "Tribe-%d" % s.tribe}))
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("t") == "welcome":
                    s.id = msg["id"]
                    self.tribe_of_id[s.id] = s.tribe
                    log("  soul %d joined as player %d (%s, Tribe-%d)" % (s.i, s.id, "inherit" if s.inherit else "naive", s.tribe))
                elif msg.get("t") == "state":
                    s.snap = msg.get("snap", msg)
                    s.tick = s.snap.get("tick", s.tick + 1)
                    self.game_ticks = max(self.game_ticks, s.tick)
                    players = s.snap.get("players", [])
                    me = players[s.id] if s.id is not None and s.id < len(players) else None
                    if me and not me.get("alive", True):
                        if not s.dead_sent:
                            self.on_death(s, me)
                            await ws.send(json.dumps({"t": "respawn"}))
                            s.dead_sent = True
                        continue
                    s.dead_sent = False
                    # REAL TIME: every game tick gets an input from the current intent
                    await ws.send(json.dumps(self.tick_input(s)))

    async def brain_loop(self):
        """Decide for every joined, living soul as often as the GPU allows - with 8
        flies about once per 1.4 s of wall time. The world does not wait."""
        t0 = time.time(); last_report = 0; last_state = 0
        loop = asyncio.get_event_loop()
        while True:
            ready = [s for s in self.souls if s.snap is not None and s.id is not None and not s.dead_sent]
            if not ready:
                await asyncio.sleep(0.1); continue
            ctx = {}
            for s in ready:
                c = self.sense(s)
                if c: ctx[s.i] = c
            await loop.run_in_executor(None, self.step_all, ctx)
            if time.time() - last_state > 0.5:
                last_state = time.time(); self.write_state()
            if time.time() - last_report > 30:
                last_report = time.time()
                alive = sum(1 for s in self.souls if s.id is not None and not s.dead_sent)
                wf = self.sw.weights_frac()
                dec = sum(s.decisions for s in self.souls) / max(len(ready), 1)
                log("  t=%7d game ticks (%5.0fs wall, %.0f decisions/soul)  alive %d/%d" % (self.game_ticks, time.time() - t0, dec, alive, len(ready)))
                for tr in sorted({s.tribe for s in self.souls}):
                    ss = [s for s in self.souls if s.tribe == tr]
                    kd = [np.mean(s.kin_d[-50:]) for s in ss if s.kin_d]; sd_ = [np.mean(s.str_d[-50:]) for s in ss if s.str_d]
                    log("    tribe %d (%s): forage %d drink %d hits %d deaths %d weights %.2f%%  nearest kin %.1f stranger %.1f tiles" % (
                        tr, "inherit" if ss[0].inherit else "naive", sum(s.life["forage"] for s in ss), sum(s.life["drink"] for s in ss),
                        sum(s.life["hits"] for s in ss), sum(s.lives for s in ss), 100 * np.mean([wf[s.i] for s in ss]),
                        np.mean(kd) if kd else float("nan"), np.mean(sd_) if sd_ else float("nan")))
                self.logf.write(json.dumps({"event": "summary", "ticks": self.game_ticks,
                                            "deaths": [s.lives for s in self.souls],
                                            "weights": wf.tolist()}) + "\n"); self.logf.flush()
            if self.a.max_ticks and self.game_ticks >= self.a.max_ticks:
                return
            await asyncio.sleep(0)

    async def run(self):
        links = [asyncio.create_task(self.soul_link(s)) for s in self.souls]
        log("  %d souls connecting to %s ..." % (len(self.souls), self.a.url))
        try:
            await self.brain_loop()
        finally:
            for t in links:
                t.cancel()


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
    ap.add_argument("--blind", action="store_true", help="no optic lobe input (faster)")
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--plume-tiles", type=float, default=12.0)
    ap.add_argument("--tick-ms", type=float, default=100.0, help="brain time per decision")
    ap.add_argument("--max-ticks", type=int, default=0)
    ap.add_argument("--out-dir", default="../results/verge")
    ap.add_argument("--state-copy", default="", help="also write the live state here (the Verge's flies.html reads flystate.json)")
    a = ap.parse_args()
    a.flies = a.tribes * a.per_tribe
    log(f"loading swarm of {a.flies} ({a.tribes} tribes x {a.per_tribe}) ...")
    vs = VergeSwarm(a)
    log(f"  device {vs.sw.device}; odours " + ", ".join(f"{k}={v}" for k, v in vs.odours.items()))
    asyncio.run(vs.run())


if __name__ == "__main__":
    main()
