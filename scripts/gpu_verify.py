#!/usr/bin/env python3
"""
gpu_verify.py - is the swarm engine the same model as flysim.py, and how fast is it?

    python3 gpu_verify.py --flies 8

Same odour (designed CS+, bilateral 0.35 / 0.21), same regime (kc_thresh 1.0, core
0.2, MBON hold 0.85), plasticity off, 250 ms settle + 800 ms count. CPU: 3 seeds.
GPU: B flies. Populations compared: KC active cells, MBON spikes, DN spikes, leg
motor-neuron spikes. Pass = GPU mean inside the CPU seed range (or within 15%).
Then wall time per step for the swarm at several sizes.
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from flysim import FlyBrain, Params
from flysim_gpu import FlySwarm
from motormap import _load_neuromere


def leg_idx(b):
    ty = b.type.astype(str); sc = b.sc.astype(str)
    nm = _load_neuromere(b, "../data")
    motor = sc == "vnc_motor"
    wingish = (np.char.startswith(ty, "DLMn") | np.char.startswith(ty, "DVMn")
               | np.char.startswith(ty, "MNwm") | np.char.startswith(ty, "MNhm"))
    return np.flatnonzero(motor & np.isin(nm, ["T1", "T2", "T3"]) & ~wingish)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="../brain_mirrored.npz")
    ap.add_argument("--odours", default="../results/odours3.json")
    ap.add_argument("--flies", type=int, default=8)
    ap.add_argument("--sizes", default="1,8,32,64")
    ap.add_argument("--out", default="../results/gpu_verify.json")
    a = ap.parse_args()
    od = json.load(open(a.odours))
    P = Params(gain=1.0, kc_thresh_scale=1.0)

    # ---- CPU reference ---------------------------------------------------------
    cpu = FlyBrain(a.brain, P, seed=11)
    cpu.enable_plasticity(); cpu.enable_compartments(); cpu.plastic_on = False
    for d in cpu._da_by_type.values():
        d["w"] = np.where(d["w"] >= 0.2, d["w"], 0.0).astype(np.float32)
    cpu._odor_map["CS+"] = {t: 1.0 for t in od["CS+"]}
    leg = leg_idx(cpu); dn = cpu.pop["DN"]; mbon = cpu.pop["MBON"]; kc = cpu._kc
    ref = {"kc_active": [], "mbon": [], "dn": [], "leg": []}
    t0 = time.perf_counter()
    for s in (1000, 1097, 1194):
        cpu.rng = np.random.default_rng(s)
        cpu._noise_pool = cpu.rng.standard_normal(cpu._noise_pool.size, dtype=np.float32) * cpu.p.noise
        cpu._noise_off = 0
        cpu.reset(); cpu.drive_hz[:] = 0
        cpu.smell_bilateral({"CS+": 0.35}, {"CS+": 0.21})
        for t in cpu._da_by_type: cpu._ext[cpu._da_by_type[t]["cells"]] = 0.0
        for _ in range(250): cpu.step()
        tot = np.zeros(cpu.N)
        for _ in range(800): tot += cpu.step()
        ref["kc_active"].append(int((tot[kc] > 0).sum())); ref["mbon"].append(float(tot[mbon].sum()))
        ref["dn"].append(float(tot[dn].sum())); ref["leg"].append(float(tot[leg].sum()))
    cpu_ms = (time.perf_counter() - t0) / (3 * 1050) * 1000
    print(f"CPU (3 seeds): " + "  ".join(f"{k} {np.mean(v):.0f} [{min(v):.0f}-{max(v):.0f}]" for k, v in ref.items()) + f"   {cpu_ms:.2f} ms/step")
    del cpu

    # ---- GPU swarm ---------------------------------------------------------------
    sw = FlySwarm(a.brain, n=a.flies, params=P, seed=11, plasticity=False)
    dev = sw.device
    print(f"swarm on {dev}, {a.flies} flies, edges {sw.nnz:,}")
    sw.cpu._odor_map["CS+"] = {t: 1.0 for t in od["CS+"]}
    for f in range(a.flies):
        sw.smell(f, {"CS+": 0.35}, {"CS+": 0.21})
    groups = {"mbon": torch.as_tensor(mbon, device=dev), "dn": torch.as_tensor(dn, device=dev),
              "leg": torch.as_tensor(leg, device=dev)}
    kct = torch.as_tensor(kc, device=dev)
    for _ in range(250): sw.step()
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    kc_tot = torch.zeros(a.flies, len(kc), device=dev)
    out = {k: torch.zeros(a.flies, device=dev) for k in groups}
    for _ in range(800):
        s = sw.step()
        kc_tot += s[:, kct]
        for k, idx in groups.items(): out[k] += s[:, idx].sum(1)
    if dev.type == "cuda": torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) / 800 * 1000
    got = {"kc_active": (kc_tot > 0).sum(1).cpu().numpy(), **{k: v.cpu().numpy() for k, v in out.items()}}
    print(f"GPU ({a.flies} flies): " + "  ".join(f"{k} {np.mean(v):.0f} [{v.min():.0f}-{v.max():.0f}]" for k, v in got.items()) + f"   {gpu_ms:.2f} ms/step for all flies")
    verdict = {}
    for k in ref:
        lo, hi = min(ref[k]) * 0.85, max(ref[k]) * 1.15
        verdict[k] = bool(lo <= np.mean(got[k]) <= hi)
    print("agreement:", verdict)

    # ---- throughput -----------------------------------------------------------------
    timing = {}
    for n in [int(x) for x in a.sizes.split(",")]:
        try:
            s2 = FlySwarm(a.brain, n=n, params=P, seed=11, plasticity=True)
            for f in range(n): s2.smell(f, {"CS+": 0.35}, {"CS+": 0.21})
            for _ in range(50): s2.step()
            if dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200): s2.step()
            if dev.type == "cuda": torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 200 * 1000
            timing[n] = ms
            print(f"  {n:3d} flies, plasticity on: {ms:7.2f} ms/step  = {ms/n:6.3f} ms per fly-step  ({1000/ms*n/1000:.2f}x real time per fly)")
            del s2
            if dev.type == "cuda": torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  {n} flies: {str(e)[:80]}"); break
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"cpu": ref, "cpu_ms_per_step": cpu_ms, "gpu": {k: v.tolist() for k, v in got.items()},
               "gpu_ms_per_step": gpu_ms, "agreement": verdict, "timing": timing, "device": str(dev)},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
