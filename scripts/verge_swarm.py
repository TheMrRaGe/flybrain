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
    "give":            ("a gift", []),                    # hands the offered item to the nearest fly (bond: gave/fed)
    "cycleOffer":      ("a different offer", []),
}
# SCENT WORDS - the language experiment. Four emissions a fly can make, each bound to
# an unassigned descending-neuron type (like the craft verbs) and each smelled by the
# others on one olfactory receptor type no other odour in this world uses, chosen for a
# large Kenyon-cell footprint (odours3.json) so the listener's mushroom body represents
# it richly. No meaning is assigned anywhere. A word gets meaning only if the world
# pairs it with reward or punishment in a listener (KC->MBON plasticity), and gets
# used only if the speaker's brain drives it in some context. Flies cannot do this;
# it is an experiment with the fly's machinery, and labelled so.
WORDS = {"bzz": "ORN_DM1", "vrr": "ORN_DP1m", "tik": "ORN_DM2", "hum": "ORN_DC1"}
WORD_SECS = 20.0            # a word hangs in the air this long
# pack fields that are THINGS MADE (a tool, a garment, a container); materials gained
# count as gathers. Both are rewarded; only these count as crafts.
MADE = {"spear", "hammer", "knife", "axe", "pickaxe", "hoe", "cordage", "cloak", "bandage", "boots", "gloves",
        "backpack", "waterskin", "snare", "torch", "charcoal", "pot", "skep", "bar", "sword", "bow", "arrow",
        "driedMeat", "plank", "block", "cookedMeat", "ironKnife", "ironAxe", "ironPickaxe", "ironHoe"}


TILE, WORLD_W, WORLD_H = 1000, 576, 192
T_WATER, T_BUSH, T_BARE, T_FIRE = 3, 4, 5, 6
HOSTILE = {"wolf", "hedge-boar", "bog-lynx", "ash-hound", "fen-wraith"}
NEED_MAX = 1000.0


def log(m): print(m, flush=True)


class Soul:
    def __init__(self, i, inherit, tribe=0, sex="M", born_n=0, mother=None, father=None, gen=1):
        self.i, self.inherit, self.tribe = i, inherit, tribe
        self.born_n, self.gen = born_n, gen
        self.mother, self.father = mother, father          # names, or None for the founders
        self.mate = None; self.kids = []
        self.gravid_until = None; self.egg_w = None; self.recept = 0; self.refractory_until = 0
        self.dead = False
        self.kin_d, self.str_d = [], []      # distance to nearest kin / stranger, per tick
        self.id = None
        self.snap = None
        self.tick = -1
        self.heading = float(np.random.default_rng(i).uniform(0, 2 * math.pi))
        self.ev_base = None
        self.prev = None                     # previous (satiety, hydration, warmth, health)
        self.life = {"born": time.time(), "ticks": 0, "forage": 0, "drink": 0, "hits": 0,
                     "escapes": 0, "rewards": 0, "punish": 0, "overeat": 0, "gathered": 0, "gather_tries": 0, "craft_tries": 0, "crafted": [],
                     "alarms": 0, "songs": 0, "heard": 0, "touches": 0, "gave": 0, "words": 0, "heard_words": 0, "bumps": 0, "mates": 0}
        self.last_word = None
        self.crafted_ever = []
        self.prev_pack = None
        self.lives = 0
        self.dead_sent = False
        self.intent = None
        self.decisions = 0
        self.err_x = self.err_y = 0.0
        self.sex = sex
        self.name = "Fly-%d-%02d%s" % (tribe, born_n, "\u2642" if self.sex == "M" else "\u2640")
        self.x = self.y = None
        self.alarm_until = 0.0; self.alarm_pos = None
        self.flick_until = 0.0
        self.singing = False; self.court_target = None
        self.hears = False; self.touching = None; self.smells_alarm = False
        self.say_q = []; self.last_say = 0.0
        self.words_out = {}                   # word -> (expiry, x, y)
        self.heard = None                     # (word, speaker, dist0, decisions_left)
        self.heard_word = None; self.heard_from = None
        self.da_pool = 1.0; self.da_t = time.time()   # dopamine budget (see reward())

    def say(self, text):
        # one line per 6 s per fly, so the chat reads like a village and not a firehose
        if time.time() - self.last_say > 6.0:
            self.say_q.append(text); self.last_say = time.time()


class VergeSwarm:
    def __init__(self, a):
        self.a = a
        self.sw = FlySwarm(a.brain, n=a.capacity, params=Params(gain=1.0, kc_thresh_scale=a.kc_thresh),
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
        # fly-to-fly channels, each on the receptor the field has for it:
        #   kin       ORN_VA1v  Or47b  fly-derived methyl laurate/palmitoleate (Dweck et al. 2015)
        #   stranger  ORN_VA1d  Or88a  fly-derived methyl palmitate: "a fly, not one of mine" (labelled)
        #   male      ORN_DA1   Or67d  cVA, the male's volatile pheromone (Kurtovic et al. 2007)
        #   alarm     ORN_V     Gr21a/Gr63a  CO2, the Drosophila stress odorant (Suh et al. 2004)
        self.odours["kin"] = ["ORN_VA1v"]
        self.odours["stranger"] = ["ORN_VA1d"]
        self.odours["male"] = ["ORN_DA1"]
        self.odours["alarm"] = ["ORN_V"]
        for w, orn in WORDS.items(): self.odours[w] = [orn]
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
        for j, w in enumerate(WORDS): self.pool_names["say_" + w] = free[len(VERBS) + j]
        self.pools = {verb: torch.as_tensor(dn[nm == t], device=dev) for verb, t in self.pool_names.items()}
        self.pool_stats = {i: {k: [0.0, 0.0, 0] for k in self.pool_names} for i in range(a.capacity)}   # mean, M2, n per slot
        self.word_stats = {w: {"said": 0, "ctx": {"food": 0, "water": 0, "danger": 0, "stranger": 0, "kin": 0, "hungry": 0, "thirsty": 0},
                               "heard": 0, "approach": 0, "avoid": 0, "reward_after": 0, "punish_after": 0} for w in WORDS}
        # THE SOCIAL LAYER, from the connectome's own annotations (body-annotations feather):
        #   dimorphism 'male-specific'      1,258 cells a female does not have -> silenced in females
        #   receptorType putative_ppk23/25  leg pheromone neurons: 7,11-HD (female) / 7-T (male) on contact
        #   class mechanosensory_tactile, subclass leg   leg bristles: touch (Ramdya et al. 2015)
        #   JO-A / JO-B                     Johnston's organ, hears courtship song
        #   pC1 (P1 in the male), pIP10     courtship command and the song descending neuron
        # A female here is the male connectome minus its male-specific cells, labelled as such;
        # the accurate substitute is the female FlyWire brain (see FINDINGS).
        import pyarrow.feather as _feather
        ann = _feather.read_table(os.path.join(a.data_dir, "body-annotations-male-cns-v1.0-minconf-0.5.feather")).to_pandas()
        pos = {b: i for i, b in enumerate(cpu.bodyId.tolist())}
        def idx_of(mask):
            return np.array([pos[b] for b in ann.loc[mask, "bodyId"].tolist() if b in pos], int)
        dim = ann["dimorphism"].astype(str); rt = ann["receptorType"].astype(str); rs = ann["rootSide"].astype(str)
        acl = ann["class"].astype(str); asub = ann["subclass"].astype(str); aty = ann["type"].astype(str)
        self.male_only = idx_of(dim.isin(["male-specific", "potentially male-specific"]))
        self.ppk = {sd: idx_of(rt.str.startswith("putative_ppk") & (rs == sd)) for sd in "LR"}
        self.touch = {sd: idx_of((acl == "mechanosensory_tactile") & (asub == "leg") & (rs == sd)) for sd in "LR"}
        self.jo = idx_of(aty.str.match(r"JO-[AB]"))
        self.p1 = torch.as_tensor(np.flatnonzero(np.char.startswith(ty, "pC1")), device=dev)
        self.pip10 = torch.as_tensor(np.flatnonzero(ty == "pIP10"), device=dev)
        self.song_stats = {i: [0.0, 0.0, 0] for i in range(a.capacity)}
        log("  social layer: %d male-specific cells, ppk L/R %d/%d, leg bristles L/R %d/%d, JO %d, pC1 %d, pIP10 %d"
            % (len(self.male_only), len(self.ppk["L"]), len(self.ppk["R"]), len(self.touch["L"]), len(self.touch["R"]),
               len(self.jo), len(self.p1), len(self.pip10)))
        # THE POPULATION. No individual respawn. A generation starts with the founders
        # (per tribe: half male, half female), children are born of a courtship that the
        # female accepted, and when the last fly dies the next generation of founders
        # arrives. Tribe policy: in an "inherit" tribe a child's plastic KC->MBON weights
        # are the mean of its parents' (Lamarckian - memories are NOT inherited in real
        # flies; this is the experiment's lever, labelled); in a naive tribe a child is
        # born with the naive brain. Everything else about a child is the connectome.
        self.inherit_tribes = {int(x) for x in a.inherit_tribes.split(",") if x.strip() != ""}
        os.makedirs(a.out_dir, exist_ok=True)
        self.logf = open(os.path.join(a.out_dir, "events.jsonl"), "a")
        self.gen = 0; self.born = 0; self.dead_souls = []; self.links = []; self.game_ticks = 0
        self.souls = []
        self.new_generation()
        self.tribe_of_id = {}                 # player id -> tribe, learned from welcomes
        self.steps = int(round(a.tick_ms / self.sw.dt))
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
        s.x, s.y = x, y
        needs = me["needs"]
        now = time.time()
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
        others = [o for o in self.souls if o is not s and o.x is not None and o.id is not None and not o.dead_sent]
        for o in others:
            for w, (exp, wx, wy) in list(o.words_out.items()):
                if exp > now: pts[w].append((wx, wy))
            if o.sex == "M": pts["male"].append((o.x, o.y))                      # cVA drifts from every male
            if o.alarm_until > now and o.alarm_pos: pts["alarm"].append(o.alarm_pos)   # CO2 where a fly was hurt
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
            flick = {(o.x, o.y) for o in others if o.flick_until > now}
            for k, plist in pts.items():
                if k in ("male", "alarm") or k in WORDS: continue                # smelled, not seen
                lum = 1.0 if k == "fire" else 0.1
                size = 1.2 if k in ("lieutenant", "beast", "kin", "stranger") else 1.0
                for (px, py) in sorted(plist, key=lambda q: math.hypot(q[0] - x, q[1] - y))[:24]:
                    d = math.hypot(px - x, py - y) / TILE
                    if (px, py) in flick:                                          # wing flicks (Kacsoh et al. 2015):
                        size, lum = 1.8, 0.1 + 0.5 * (0.5 + 0.5 * math.sin(now * 31.4))   # spread wings, moving
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
        # contact: legs touching a neighbour (bristles on that side) and the neighbour's
        # cuticular pheromone on the leg ppk23/ppk25 neurons - 7,11-HD if she is female,
        # 7-tricosene if he is male (Thistle et al. 2012; Toda et al. 2012)
        s.touching = None
        near = sorted(((math.hypot(o.x - x, o.y - y) / TILE, o) for o in others), key=lambda q: q[0])
        if near and near[0][0] < 1.0:
            o = near[0][1]
            side = "R" if math.sin(math.atan2(o.y - y, o.x - x) - s.heading) > 0 else "L"
            sw.drive_hz[f, self.touch[side]] = 150.0
            sw.drive_hz[f, self.ppk[side]] = 120.0
            s.touching = o.name
        # hearing: a male singing within two tiles drives Johnston's organ (JO-A/B)
        s.hears = any(o.sex == "M" and o.singing and d < 2.0 for d, o in near)
        if s.hears: sw.drive_hz[f, self.jo] = 200.0
        s.smells_alarm = "alarm" in nearest and nearest["alarm"][0] < self.a.plume_tiles
        # listening: the nearest word in the air; a new one is an onset, then three
        # decisions decide whether the listener closed on the speaker or left
        heard = [(nearest[w][0], w) for w in WORDS if w in nearest and nearest[w][0] < self.a.plume_tiles]
        if heard:
            d0, w = min(heard)
            spk = None
            for o in others:
                if w in o.words_out and o.words_out[w][0] > now and (o.words_out[w][1], o.words_out[w][2]) == nearest[w][1:]:
                    spk = o; break
            if s.heard_word is None or s.heard_word[0] != w or now - s.heard_word[2] > WORD_SECS:
                s.heard_word = (w, spk.name if spk else "?", now); s.heard_from = spk
                s.heard = [w, spk, d0, 3]; s.life["heard_words"] += 1; self.word_stats[w]["heard"] += 1
                self.note(s, "heard", {"word": w, "from": spk.name if spk else None, "dist": d0})
                s.say("I hear %s: %s" % (spk.name if spk else "someone", w))
        elif s.heard_word and now - s.heard_word[2] > WORD_SECS:
            s.heard_word = None
        if s.heard and s.heard[3] > 0:
            s.heard[3] -= 1
            if s.heard[3] == 0 and s.heard[1] is not None and s.heard[1].x is not None:
                d1 = math.hypot(s.heard[1].x - x, s.heard[1].y - y) / TILE
                key = "approach" if d1 < s.heard[2] - 0.5 else "avoid" if d1 > s.heard[2] + 0.5 else None
                if abs(d1 - s.heard[2]) > 6.0: key = None                       # a respawn, not a walk
                if key:
                    self.word_stats[s.heard[0]][key] += 1
                    self.note(s, key, {"word": s.heard[0], "from": s.heard[1].name, "d0": s.heard[2], "d1": d1})
                    if key == "approach": s.say("Going to see what %s means." % s.heard[1].name)
        # hunger through its targets (Krashes 2009): MB-MP1 tonic when fed
        sw.quiet_dopamine(f)
        sw.ext[f, self.ppl101] = 8.0 * (1.0 - hunger)
        # the world teaches: reward on satiety/hydration gain, punishment on damage
        cur = (needs["satiety"], needs["hydration"], needs["warmth"], me.get("health", 0))
        if s.prev is not None:
            if cur[0] > s.prev[0] + 1: self.reward(s, f, need=1.0 - s.prev[0] / NEED_MAX)
            elif cur[1] > s.prev[1] + 1: self.reward(s, f, need=1.0 - s.prev[1] / NEED_MAX)
            if cur[3] < s.prev[3]:
                sw.stimulate_type(f, "PPL105", 70.0); s.life["punish"] += 1; s.life["hits"] += 1
                if s.heard_word and now - s.heard_word[2] < WORD_SECS: self.word_stats[s.heard_word[0]]["punish_after"] += 1
                # a hurt fly gives off CO2 that others avoid for minutes (Suh 2004) and
                # flicks its wings, which flies that can see it read as danger (Kacsoh 2015)
                s.alarm_until = now + 60.0; s.alarm_pos = (x, y); s.flick_until = now + 5.0
                s.life["alarms"] += 1
                s.say("Ow! I'm hurt over here - stay clear!")
        s.prev = cur
        # anything new in the pack is a gain: materials are gathers, made things are crafts
        if s.prev_pack is not None:
            for k, v in pack.items():
                if isinstance(v, (int, float)) and v > s.prev_pack.get(k, 0):
                    # a material is worth having until there is plenty of it: the first
                    # ten units are rewarded, more is not (soul 6 sat on a clay tile and
                    # dug 1,181 clay for 60 rewards in the first run). Made things always are.
                    if k in MADE or s.prev_pack.get(k, 0) < 10:
                        self.reward(s, f, need=1.0)
                    if k in MADE and s.prev_pack.get(k, 0) == 0:
                        s.life["crafted"].append(k); self.note(s, "crafted", {"item": k})
                    elif k not in MADE and k != "water":
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
        song = torch.zeros(B, device=sw.device); p1 = torch.zeros(B, device=sw.device)
        for _ in range(self.steps):
            spk = sw.step()
            acc["L"] += spk[:, self.L].sum(1); acc["R"] += spk[:, self.R].sum(1)
            acc["leg"] += spk[:, self.leg].sum(1); acc["feed"] += spk[:, self.feed].sum(1)
            acc["esc"] += spk[:, self.escape].sum(1)
            for k, idx in self.regions.items(): reg[k] += spk[:, idx].sum(1)
            for k, idx in self.pools.items(): pl[k] += spk[:, idx].sum(1)
            song += spk[:, self.pip10].sum(1); p1 += spk[:, self.p1].sum(1)
        out = {k: v.cpu().numpy() for k, v in acc.items()}
        pln = {k: v.cpu().numpy() for k, v in pl.items()}
        songn = (song + p1 / max(len(self.p1), 1) * 4.0).cpu().numpy()      # song command plus P1 drive
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
                s.flick_until = time.time() + 5.0
                s.say("Something's coming - watch out!")
                thr = [c["nearest"][k] for k in ("lieutenant", "beast") if k in c["nearest"]]
                if thr:
                    d, px, py = min(thr)
                    s.heading = math.atan2(c["y"] - py, c["x"] - px)
            s.heading += evidence * 0.5
            # a wall: the body did not move though the legs were driving. Real flies
            # feel that on the leg and head bristles and turn away; here the bristles on
            # both sides are driven for the next decision and the heading is turned
            # ~120 degrees plus noise (the reflex itself is an interface choice, labelled).
            # Without it three flies spent a run pressed into the map corner, singing.
            pi = s.intent
            if pi is not None and pi.get("speed", 0) > 0.3 and s.x is not None and getattr(s, "px", None) is not None:
                if math.hypot(s.x - s.px, s.y - s.py) < 0.05 * TILE:
                    s.heading += 2.1 + float(np.random.default_rng(s.decisions).uniform(-0.6, 0.6))
                    sw.drive_hz[f, self.touch["L"]] = 150.0; sw.drive_hz[f, self.touch["R"]] = 150.0
                    s.life["bumps"] += 1
            if s.x is not None:
                ex = s.x / TILE; ey = s.y / TILE
                outward = ((ex < 1.5 and math.cos(s.heading) < 0) or (ex > WORLD_W - 1.5 and math.cos(s.heading) > 0)
                           or (ey < 1.5 and math.sin(s.heading) < 0) or (ey > WORLD_H - 1.5 and math.sin(s.heading) > 0))
                if outward:                                                 # sliding along the edge counts as a wall too
                    s.heading = math.atan2(WORLD_H * TILE / 2 - s.y, WORLD_W * TILE / 2 - s.x) + float(np.random.default_rng(s.decisions).uniform(-0.8, 0.8))
                    sw.drive_hz[f, self.touch["L"]] = 150.0; sw.drive_hz[f, self.touch["R"]] = 150.0
                    s.life["bumps"] += 1
            s.px, s.py = s.x, s.y
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
            # courtship: a male whose song neurons fire above their own baseline, with a
            # female within two tiles, sings; she hears it on JO next decision
            near_f = [o for o in self.souls if o is not s and o.sex == "F" and o.x is not None and s.x is not None
                      and o.id is not None and not o.dead_sent and math.hypot(o.x - s.x, o.y - s.y) / TILE < 2.0]
            st = self.song_stats[f]; v = float(songn[f])
            st[2] += 1; d = v - st[0]; st[0] += d / st[2]; st[1] += d * (v - st[0])
            sd = math.sqrt(st[1] / max(st[2] - 1, 1)) if st[2] > 2 else 1e9
            was = s.singing
            s.singing = s.sex == "M" and st[2] > 20 and v > st[0] + 1.5 * sd and bool(near_f)
            s.court_target = near_f[0].name if s.singing else None
            if s.singing and not was:
                s.life["songs"] += 1; self.note(s, "song", {"to": s.court_target})
                s.say("Hey %s, come over here a minute." % s.court_target)
            if s.singing:
                # a courting male tracks the female (P1 -> LC10a visual pursuit, Ribeiro et
                # al. 2018). The optic-lobe pursuit path is dark here, so the tracking is
                # written out: face her, slow to a court. Labelled substitute.
                tgt = near_f[0]
                s.heading = math.atan2(tgt.y - s.y, tgt.x - s.x)
                d_t = math.hypot(tgt.x - s.x, tgt.y - s.y) / TILE
                speed = 0.0 if d_t < 0.6 else min(speed, 0.6)
            # a receptive female slows to the song (Coen et al. 2014). Her deciding
            # circuit (vpoDN) is absent from the male connectome, so the slowing is
            # written out for a female who is not gravid, not in refractory and fed:
            # the male side (the song) is the brain's; her side is the substitute.
            if s.sex == "F" and s.hears and s.gravid_until is None and s.tick > s.refractory_until and c["hunger"] < 0.7:
                speed = min(speed, 0.3)
            # acceptance. A receptive female slows and stays (Coen et al. 2014); the
            # female-specific circuit that decides this (vpoDN) is not in the male
            # connectome, so staying within a tile of a singing male for three decisions
            # at less than half speed IS the acceptance here, labelled as such
            if s.sex == "F" and not s.dead:
                suitor = next((o for o in self.souls if o.sex == "M" and o.singing and o.court_target == s.name
                               and o.x is not None and s.x is not None and math.hypot(o.x - s.x, o.y - s.y) / TILE < 1.0), None)
                s.recept = s.recept + 1 if (suitor and speed < 0.5) else 0
                if (s.recept >= 3 and s.gravid_until is None and s.tick > s.refractory_until
                        and c["hunger"] < 0.7 and suitor is not None):
                    self.mated(s, suitor)
            if s.hears and not getattr(s, "_heard", False):
                s.life["heard"] += 1; s.say("Someone's calling me.")
            s._heard = s.hears
            if s.touching and s.touching != getattr(s, "_touched", None):
                s.life["touches"] += 1; self.note(s, "touch", {"with": s.touching})
                s.say("Hi %s." % s.touching)
            s._touched = s.touching
            if s.smells_alarm and not getattr(s, "_alarmed", False):
                s.say("Trouble that way. I'm keeping my distance.")
            s._alarmed = s.smells_alarm
            # giving: the offered item to the nearest fly within reach
            me_pl = s.snap["players"][s.id] if s.snap and s.id is not None and s.id < len(s.snap.get("players", [])) else None
            offer = (me_pl or {}).get("offer", "wood")
            any_near = any(o is not s and o.x is not None and s.x is not None and o.id is not None and not o.dead_sent
                           and math.hypot(o.x - s.x, o.y - s.y) / TILE < 1.5 for o in self.souls)
            if hot.get("give") and any_near and pack.get(offer, 0) >= 1:
                verbs.append("give"); s.life["gave"] += 1; s.life["craft_tries"] += 1
                self.note(s, "gave", {"item": offer}); s.say("Here, take this %s." % offer)
            if hot.get("cycleOffer"):
                verbs.append("cycleOffer")
            # speaking: a word whose DN type fires above baseline goes into the air for
            # WORD_SECS; the context it was said in is the dictionary being written
            for w in WORDS:
                # words take a stronger burst than verbs (mean + 2.5 sd, ~1 in 200 decisions
                # by chance) so the air is not full of noise and a word is an event
                stw = self.pool_stats[f]["say_" + w]
                sdw = math.sqrt(stw[1] / max(stw[2] - 1, 1)) if stw[2] > 2 else 1e9
                loud = stw[2] > 20 and float(pln["say_" + w][f]) > stw[0] + 2.5 * sdw
                if loud and (w not in s.words_out or s.words_out[w][0] < time.time()) and s.x is not None:
                    s.words_out[w] = (time.time() + WORD_SECS, s.x, s.y)
                    c = ctx[f]; nr = c["nearest"]; ws = self.word_stats[w]; ws["said"] += 1
                    ctxs = {"food": "bush" in nr and nr["bush"][0] < 3, "water": "water" in nr and nr["water"][0] < 3,
                            "danger": s.alarm_until > time.time() or any(k in nr and nr[k][0] < 5 for k in ("lieutenant", "beast")),
                            "stranger": "stranger" in nr and nr["stranger"][0] < 4, "kin": "kin" in nr and nr["kin"][0] < 4,
                            "hungry": c["hunger"] > 0.5, "thirsty": c["thirst"] > 0.5}
                    for k2, v2 in ctxs.items():
                        if v2: ws["ctx"][k2] += 1
                    s.life["words"] += 1; s.last_word = (w, time.time())
                    self.note(s, "word", {"word": w, **ctxs}); s.say("%s!" % w)
            if hot.get("gather") and (cc.get("tree_near") or cc.get("rock_near")):
                verbs.append("action"); s.life["gather_tries"] += 1; act = True; speed = 0.0
            for k, (made, needs) in VERBS.items():
                if k in ("gather", "give", "cycleOffer") or not hot.get(k): continue
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
            souls.append({"i": s.i, "id": s.id, "name": s.name, "sex": s.sex, "tribe": s.tribe,
                          "social": {"singing": s.singing, "to": s.court_target, "hears": s.hears, "touching": s.touching,
                                     "alarm": s.alarm_until > time.time(), "flick": s.flick_until > time.time(),
                                     "smells_alarm": s.smells_alarm,
                                     "saying": [w for w, v in s.words_out.items() if v[0] > time.time()],
                                     "hearing": s.heard_word[0] if s.heard_word and time.time() - s.heard_word[2] < WORD_SECS else None,
                                     "hearing_from": s.heard_word[1] if s.heard_word and time.time() - s.heard_word[2] < WORD_SECS else None,
                                     "da_pool": round(s.da_pool, 2)},
                          "inherit": s.inherit, "lives": s.lives, "alive": bool(me and me.get("alive", True)),
                          "x": me["x"] if me else None, "y": me["y"] if me else None,
                          "needs": me.get("needs") if me else None, "health": me.get("health") if me else None,
                          "lineage": me.get("lineage") if me else None, "life": s.life, "decisions": s.decisions,
                          "weights": float(wf[s.i]), "intent": s.intent, "regions": getattr(s, "regions", None),
                          "crafted_ever": s.crafted_ever, "pack": (getattr(s, "craft_ctx", None) or {}).get("pack", {}),
                          "gen": s.gen, "mate": s.mate, "kids": s.kids, "mother": s.mother, "father": s.father,
                          "due": (s.gravid_until - s.tick) if s.gravid_until else None,
                          "kin_d": float(np.mean(s.kin_d[-20:])) if s.kin_d else None,
                          "stranger_d": float(np.mean(s.str_d[-20:])) if s.str_d else None})
        st = {"ticks": self.game_ticks, "wall": time.time(), "souls": souls, "pool_names": self.pool_names,
              "gen": self.gen, "born": self.born, "alive": len(self.souls),
              "graveyard": [{"name": d.name, "gen": d.gen, "tribe": d.tribe, "kids": d.kids, "mate": d.mate} for d in self.dead_souls[-16:]],
              "deaths": self.deaths[-30:], "odours": self.odours,
              "words": {w: {"orn": WORDS[w], "pool": self.pool_names["say_" + w], **self.word_stats[w]} for w in WORDS}}
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

    # -- population ---------------------------------------------------------- #

    def place(self, s):
        """Give a soul its brain slot: fresh dynamics, the right cells for its sex,
        fresh pool baselines. Weights are set by the caller (naive or inherited)."""
        self.sw.reset(flies=[s.i])
        if self.sw.silent is not None: self.sw.silent[s.i] = False
        if s.sex == "F": self.sw.silence(s.i, self.male_only)
        self.pool_stats[s.i] = {k: [0.0, 0.0, 0] for k in self.pool_names}
        self.song_stats[s.i] = [0.0, 0.0, 0]

    def free_slot(self):
        used = {s.i for s in self.souls}
        for i in range(self.a.capacity):
            if i not in used: return i
        return None

    def new_generation(self):
        self.gen += 1
        self.souls = []
        for i in range(self.a.flies):
            tribe = i // self.a.per_tribe
            s = Soul(i, inherit=tribe in self.inherit_tribes, tribe=tribe, sex="M" if i % 2 == 0 else "F",
                     born_n=self.born, gen=self.gen)
            self.born += 1
            self.sw.reset_weights(flies=[i]); self.place(s)
            self.souls.append(s)
        log("  === GENERATION %d: %d founders ===" % (self.gen, len(self.souls)))
        self.logf.write(json.dumps({"event": "generation", "gen": self.gen, "game_tick": self.game_ticks,
                                    "founders": [s.name for s in self.souls]}) + "\n"); self.logf.flush()
        return self.souls

    def mated(self, s, m):
        s.mate, m.mate = m.name, s.name
        s.gravid_until = s.tick + self.a.gestation
        s.egg_w = self.sw.w[m.i].clone()                    # the father's weights now, in case he dies first
        s.father_of_egg = m.name; s.recept = 0
        s.life["mates"] += 1; m.life["mates"] += 1
        self.note(s, "mated", {"with": m.name}); self.note(m, "mated", {"with": s.name})
        s.say("%s and I are together now." % m.name); m.say("%s said yes." % s.name)
        log("  ** %s accepted %s (child due at tick %d) **" % (s.name, m.name, s.gravid_until))

    def births(self):
        """Children due: a free brain slot, the mother's tribe, a coin for sex, and the
        tribe's inheritance policy for the weights."""
        for s in list(self.souls):
            if s.sex != "F" or s.gravid_until is None or s.tick < s.gravid_until or s.dead: continue
            slot = self.free_slot()
            if slot is None: continue                        # the world is full; try next decision
            rng = np.random.default_rng(self.born * 7919 + self.gen)
            c = Soul(slot, inherit=s.inherit, tribe=s.tribe, sex="M" if rng.random() < 0.5 else "F",
                     born_n=self.born, mother=s.name, father=s.father_of_egg, gen=self.gen)
            self.born += 1
            if s.inherit and s.egg_w is not None:
                self.sw.w[slot] = 0.5 * (self.sw.w[s.i] + s.egg_w)
            else:
                self.sw.reset_weights(flies=[slot])
            self.place(c)
            s.kids.append(c.name)
            for o in self.souls:
                if o.name == s.father_of_egg: o.kids.append(c.name)
            s.gravid_until = None; s.egg_w = None; s.refractory_until = s.tick + self.a.refractory
            self.souls.append(c)
            self.links.append(asyncio.get_event_loop().create_task(self.soul_link(c)))
            self.note(s, "birth", {"child": c.name, "father": c.father, "inherits": bool(s.inherit)})
            s.say("A child! %s." % c.name)
            log("  ** BIRTH: %s, child of %s and %s (%s) **" % (c.name, s.name, c.father, "inherits" if s.inherit else "naive"))

    def reward(self, s, f, need=1.0):
        """PAM08 for a good outcome, with two limits. (1) Appetitive reward needs
        hunger: a fed fly forms no sugar memory (Krashes et al. 2009), so the pulse
        scales with the need it met - full above 50 %, nothing when sated. (2) A
        budget: each pulse spends a quarter of a pool that refills over a minute, so
        a run of rewards tapers instead of piling up - the limit asked for, labelled
        as a modelling choice (dopamine release does adapt with repetition, but this
        number is ours). Punishment is not gated: pain is not state-dependent."""
        now = time.time()
        s.da_pool = min(1.0, s.da_pool + (now - s.da_t) / 60.0); s.da_t = now
        amp = 70.0 * min(1.0, need / 0.5) * s.da_pool
        if amp < 5.0: return False
        self.sw.stimulate_type(f, "PAM08", amp); s.da_pool = max(0.0, s.da_pool - 0.25)
        s.life["rewards"] += 1
        if s.heard_word and now - s.heard_word[2] < WORD_SECS: self.word_stats[s.heard_word[0]]["reward_after"] += 1
        return True

    def note(self, s, event, extra):
        rec = {"event": event, "soul": s.i, "tribe": s.tribe, "inherit": s.inherit, "game_tick": s.tick,
               "life_ticks": s.life["ticks"], **extra}
        self.logf.write(json.dumps(rec) + "\n"); self.logf.flush()
        if event == "crafted":
            s.crafted_ever.append({"item": extra["item"], "tick": s.tick, "life": s.lives + 1})
            s.say("Look - I made a %s!" % extra["item"])
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
        return

    # -- network -------------------------------------------------------------- #

    async def soul_link(self, s):
        async with websockets.connect(self.a.url, max_size=None) as ws:
            s.ws = ws
            await ws.send(json.dumps({"t": "hello", "name": s.name, "kind": "agent",
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
                            s.dead_sent = True
                            s.dead = True
                            if s in self.souls: self.souls.remove(s)
                            self.dead_souls.append(s)
                            for o in self.souls:
                                if o.mate == s.name: o.say("%s is gone." % s.name)
                        return
                    s.dead_sent = False
                    # REAL TIME: every game tick gets an input from the current intent
                    await ws.send(json.dumps(self.tick_input(s)))
                    while s.say_q:
                        await ws.send(json.dumps({"t": "say", "text": s.say_q.pop(0)}))

    async def brain_loop(self):
        """Decide for every joined, living soul as often as the GPU allows - with 8
        flies about once per 1.4 s of wall time. The world does not wait."""
        t0 = time.time(); last_report = 0; last_state = 0
        loop = asyncio.get_event_loop()
        while True:
            ready = [s for s in self.souls if s.snap is not None and s.id is not None and not s.dead_sent]
            if not ready:
                if getattr(self, "restart_requested", False):
                    await self.restart(); continue
                if not self.souls:
                    self.new_generation()
                    self.links += [loop.create_task(self.soul_link(s)) for s in self.souls]
                await asyncio.sleep(0.1); continue
            ctx = {}
            for s in ready:
                c = self.sense(s)
                if c: ctx[s.i] = c
            await loop.run_in_executor(None, self.step_all, ctx)
            if getattr(self, "restart_requested", False):
                await self.restart(); continue
            self.births()
            if not self.souls:
                log("  the last fly of generation %d has died" % self.gen)
                self.new_generation()
                self.links += [loop.create_task(self.soul_link(s)) for s in self.souls]
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

    async def control(self, reader, writer):
        """A one-verb HTTP control port for the spectator page: POST /restart starts
        the experiment over (every fly dies, generation 1 begins) without reloading
        the brain. Local only."""
        try:
            line = await reader.readline()
            while True:
                h = await reader.readline()
                if not h or h in (b"\r\n", b"\n"): break
            parts = line.split()
            path = parts[1].decode() if len(parts) > 1 else "/"
            if path.startswith("/restart"):
                self.restart_requested = True; body = b'{"ok":true,"restarting":true}'
            else:
                body = json.dumps({"ok": True, "gen": self.gen, "alive": len(self.souls), "born": self.born}).encode()
            head = ("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n"
                    "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\nContent-Length: %d" % len(body)) + "\r\n\r\n"
            writer.write(head.encode() + body)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    async def restart(self):
        log("  === RESTART requested from the page: generation 1 again ===")
        self.logf.write(json.dumps({"event": "restart", "game_tick": self.game_ticks, "gen": self.gen}) + "\n"); self.logf.flush()
        for s in list(self.souls):
            s.dead = True; s.dead_sent = True
            try:
                if getattr(s, "ws", None) is not None: await s.ws.close()
            except Exception:
                pass
        self.souls = []; self.dead_souls = []; self.deaths = []
        self.gen = 0; self.born = 0
        for w in self.word_stats:
            self.word_stats[w] = {"said": 0, "ctx": {k: 0 for k in self.word_stats[w]["ctx"]},
                                  "heard": 0, "approach": 0, "avoid": 0, "reward_after": 0, "punish_after": 0}
        self.new_generation()
        self.links += [asyncio.get_event_loop().create_task(self.soul_link(s)) for s in self.souls]
        self.restart_requested = False

    async def run(self):
        self.restart_requested = False
        self.links = [asyncio.create_task(self.soul_link(s)) for s in self.souls]
        log("  %d souls connecting to %s ..." % (len(self.souls), self.a.url))
        try:
            await asyncio.start_server(self.control, "127.0.0.1", self.a.control_port)
            log("  control port http://127.0.0.1:%d  (POST /restart)" % self.a.control_port)
        except Exception as e:
            log("  control port unavailable: %s" % e)
        try:
            await self.brain_loop()
        finally:
            for t in self.links:
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
    ap.add_argument("--capacity", type=int, default=16, help="brain slots on the GPU: the population cap")
    ap.add_argument("--control-port", type=int, default=8002, help="local HTTP port the page uses to restart the run")
    ap.add_argument("--gestation", type=int, default=1500, help="game ticks from acceptance to birth")
    ap.add_argument("--refractory", type=int, default=3000, help="game ticks before a mother can accept again")
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
