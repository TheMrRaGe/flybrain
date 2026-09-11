#!/usr/bin/env python3
"""
flysim_gpu.py - many flies, one connectome, on the GPU.

    from flysim_gpu import FlySwarm
    swarm = FlySwarm("../brain_mirrored.npz", n=64, device="cuda")
    swarm.smell(fly=3, left={"CS+": 0.35}, right={"CS+": 0.21})
    spikes = swarm.step()            # [n, N] bool on the device

WHAT THIS IS
    The SAME model as flysim.py - same constants, same alpha synapse and 1.8 ms delay,
    same decisions 1-16 baked into the weight matrix by FlyBrain itself - stepped for
    B flies at once. The connectome is shared (one sparse matrix); only state is per
    fly, plus the 33,496 plastic KC->MBON weights, which are the one thing that
    differs between individuals and are kept as a [B, E] tensor.

WHY
    One brain with learning on costs ~0.5 s of wall per 150 ms tick on the CPU. A
    population of a hundred creatures in a world - the experiment the survival box
    and the Verge need - is impossible at that rate. On a GTX 1650 the propagation
    for all flies is one sparse-dense product per step.

VERIFIED against flysim.py (see gpu_verify.py): same odour, same regime, spike
counts in KC / MBON / DN / leg motor populations agree within seed-to-seed spread.
Noise and Poisson draws are not bit-identical (different generators), so the
comparison is statistical, as it must be.
"""
from __future__ import annotations

import numpy as np
import torch

from flysim import FlyBrain, Params


class FlySwarm:
    def __init__(self, path, n=8, params: Params | None = None, device=None, seed=0,
                 core=0.2, plasticity=True, vision=False):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n = n
        cpu = FlyBrain(path, params or Params(gain=1.0), seed=seed)
        self.cpu = cpu                         # kept for populations, receptor maps, odours
        self.p = p = cpu.p
        self.N = N = cpu.N
        self.dt = p.dt
        self.steps_delay = max(1, int(round(p.syn_delay_ms / p.dt)))
        torch.manual_seed(seed)
        dev = self.device

        # vision: the lamina hold and the dark baseline go into ext0 / the per-fly
        # visual drive (enable_vision, decision in flysim); the eye map is eye.Eye
        self.vision = vision
        if vision:
            cpu.enable_vision()
        self.vis = torch.as_tensor(cpu.pop["visual"], device=self.device, dtype=torch.long)
        self.vis_base = 0.0                      # dark = no histamine (corrected sign, see eye.py)
        # --- plastic edges (per fly) and the shared static matrix -----------------
        cpu.enable_plasticity()
        cpu.enable_compartments()
        if core > 0:
            for d in cpu._da_by_type.values():
                d["w"] = np.where(d["w"] >= core, d["w"], 0.0).astype(np.float32)
        self.plastic_on = plasticity
        E = len(cpu._plastic)
        self.E = E
        self.pl_pre = torch.as_tensor(cpu._plastic_pre, device=dev, dtype=torch.long)
        self.pl_post = torch.as_tensor(cpu._plastic_post, device=dev, dtype=torch.long)
        self.w0 = torch.as_tensor(cpu._w0, device=dev)                       # [E]
        self.w = self.w0.unsqueeze(0).repeat(n, 1).contiguous()              # [B, E]

        # static matrix: M[post, pre] with plastic edges removed. FlyBrain's M is CSR
        # (post rows); zero the plastic entries via the CSC layout it also holds.
        M = cpu.M.tocoo()
        pre_of_edge = np.repeat(np.arange(N), np.diff(cpu._out_ptr))
        plastic_mask = np.zeros(cpu._out_w.size, dtype=bool)
        plastic_mask[cpu._plastic] = True
        # drop the plastic (pre, post) pairs from the shared matrix, vectorised
        pl_key = pre_of_edge[plastic_mask].astype(np.int64) * N + cpu._out_tgt[plastic_mask].astype(np.int64)
        all_key = M.col.astype(np.int64) * N + M.row.astype(np.int64)
        keep = ~np.isin(all_key, pl_key)
        # EVENT-DRIVEN, like decision 10 on the CPU: out-edges indexed by PRESYNAPTIC
        # neuron (CSC of M), so a step touches only the edges of cells that fired.
        # A dense spike-matrix product was measured at 107 ms/step for 64 flies -
        # 466M multiply-adds of which 99.6% are by zero.
        pre_s, post_s, w_s = M.col[keep], M.row[keep], M.data[keep].astype(np.float32)
        order = np.argsort(pre_s, kind="stable")
        pre_s, post_s, w_s = pre_s[order], post_s[order], w_s[order]
        counts = np.bincount(pre_s, minlength=N)
        self.out_ptr = torch.as_tensor(np.concatenate([[0], np.cumsum(counts)]), device=dev, dtype=torch.long)
        self.out_deg = torch.as_tensor(counts, device=dev, dtype=torch.long)
        self.out_tgt = torch.as_tensor(post_s, device=dev, dtype=torch.long)
        self.out_w = torch.as_tensor(w_s, device=dev)
        self.nnz = int(counts.sum())

        # --- per-neuron constants ------------------------------------------------
        self.v_th = torch.as_tensor(cpu.v_th, device=dev)                    # [N]
        self.ext0 = torch.as_tensor(cpu._ext, device=dev)                    # [N] holds (MBON etc.)
        self.driven = torch.as_tensor(cpu.driven, device=dev)                # [N] bool
        self.driven_idx = torch.nonzero(self.driven).squeeze(1)
        self.kc = torch.as_tensor(cpu._kc, device=dev, dtype=torch.long)
        self.mbon = torch.as_tensor(cpu.pop["MBON"], device=dev, dtype=torch.long)
        # dopamine: per DAN type, cells and MBON weight vector (core cut applied).
        # Learning state is kept COMPACT: traces on the 4,064 KCs, dopamine on the
        # 97 MBONs - not on all 162k neurons (that loop was 60% of a step).
        kc_local = -np.ones(N, dtype=np.int64); kc_local[cpu._kc] = np.arange(len(cpu._kc))
        mb_local = -np.ones(N, dtype=np.int64); mb_local[cpu.pop["MBON"]] = np.arange(len(cpu.pop["MBON"]))
        self.pl_pre_k = torch.as_tensor(kc_local[cpu._plastic_pre], device=dev, dtype=torch.long)
        self.pl_post_m = torch.as_tensor(mb_local[cpu._plastic_post], device=dev, dtype=torch.long)
        self.n_kc, self.n_mbon = len(cpu._kc), len(cpu.pop["MBON"])
        self.da_types = {}
        for t, d in cpu._da_by_type.items():
            if (d["w"] > 0).any():
                self.da_types[t] = (torch.as_tensor(d["cells"], device=dev, dtype=torch.long),
                                    torch.as_tensor(d["w"][cpu.pop["MBON"]], device=dev))

        # --- per-fly state --------------------------------------------------------
        self.ext = self.ext0.unsqueeze(0).repeat(n, 1).contiguous()          # [B, N]
        self.drive_hz = torch.zeros(n, N, device=dev)
        self.clear_senses()
        self.reset()

    # -- state ---------------------------------------------------------------- #

    def reset(self, flies=None):
        B, N, dev = self.n, self.N, self.device
        sel = slice(None) if flies is None else flies
        if flies is None:
            self.v = torch.zeros(B, N, device=dev)
            self.g = torch.zeros(B, N, device=dev)
            self.refrac = torch.zeros(B, N, device=dev)
            self.last = torch.zeros(B, N, device=dev)
            self.dly = [torch.zeros(B, N, device=dev) for _ in range(self.steps_delay)]
            self.kc_trace = torch.zeros(B, self.n_kc, device=dev)
            self.da = torch.zeros(B, self.n_mbon, device=dev)
            self.da_base = torch.zeros(B, self.n_mbon, device=dev)
        else:
            for t in (self.v, self.g, self.refrac, self.last, self.kc_trace, self.da, *self.dly):
                t[sel] = 0.0

    def reset_weights(self, flies=None):
        if flies is None:
            self.w[:] = self.w0
        else:
            self.w[flies] = self.w0

    # -- senses (per fly) ------------------------------------------------------ #

    def clear_senses(self, fly=None):
        if fly is None:
            self.drive_hz.zero_()
            if self.vision: self.drive_hz[:, self.vis] = self.vis_base      # dark, not blind
        else:
            self.drive_hz[fly].zero_()
            if self.vision: self.drive_hz[fly, self.vis] = self.vis_base

    def see(self, fly, receptor_idx, hz):
        """Per-photoreceptor rates from eye.Eye.render (dark = baseline, light = less)."""
        self.drive_hz[fly, torch.as_tensor(receptor_idx, device=self.device)] = torch.as_tensor(
            np.asarray(hz, dtype=np.float32), device=self.device)

    def smell(self, fly, left: dict, right: dict):
        """Bilateral odour for one fly; strengths as in FlyBrain.smell_bilateral."""
        cpu = self.cpu
        olf = cpu.pop["olfactory"]
        row = self.drive_hz[fly]
        row[torch.as_tensor(olf, device=self.device)] = 0.0
        for sidekey, mix in (("L", left), ("R", right)):
            for name, strength in mix.items():
                if name not in cpu._odor_map:
                    cpu.define_odor(name)
                for rtype, st in cpu._odor_map[name].items():
                    grp = cpu._receptor_side[rtype]
                    idx = grp[sidekey]
                    if len(idx):
                        row[torch.as_tensor(idx, device=self.device)] += self.p.max_rate_hz * st * float(strength)
                    if len(grp["M"]):
                        row[torch.as_tensor(grp["M"], device=self.device)] += self.p.max_rate_hz * st * float(strength) * 0.5

    def drive(self, fly, neuron_idx, hz):
        self.drive_hz[fly, torch.as_tensor(neuron_idx, device=self.device)] = float(hz)

    def taste(self, fly, quality):
        cpu = self.cpu
        g = cpu.pop["gustatory"]
        ty = cpu.type[g].astype(str)
        pick = g[np.isin(ty, cpu.TASTE_SWEET if quality >= 0 else cpu.TASTE_BITTER)]
        self.drive_hz[fly, torch.as_tensor(g, device=self.device)] = 0.0
        self.drive_hz[fly, torch.as_tensor(pick, device=self.device)] = self.p.max_rate_hz * min(abs(float(quality)), 1.0)

    def stimulate_type(self, fly, dan_type, mv):
        cells, _ = self.da_types[dan_type]
        self.ext[fly, cells] = float(mv)

    def quiet_dopamine(self, fly=None):
        for cells, _ in self.da_types.values():
            if fly is None:
                self.ext[:, cells] = self.ext0[cells]
            else:
                self.ext[fly, cells] = self.ext0[cells]

    # -- one millisecond for every fly ---------------------------------------- #

    @torch.no_grad()
    def step(self):
        p, dt, B, N = self.p, self.dt, self.n, self.N
        arrived = self.dly.pop(0)
        self.dly.append(self.last)
        # synaptic input from the cells that fired steps_delay ago, all flies at once
        inc = torch.zeros(B, N, device=self.device)
        nz = arrived.nonzero(as_tuple=True)                                    # (fly, neuron)
        if nz[0].numel():
            fly, j = nz
            deg = self.out_deg[j]
            total = int(deg.sum())
            if total:
                starts = self.out_ptr[j]
                seg = torch.repeat_interleave(starts, deg)
                offs = torch.arange(total, device=self.device) - torch.repeat_interleave(
                    torch.cumsum(deg, 0) - deg, deg)
                e = seg + offs
                inc.index_put_((torch.repeat_interleave(fly, deg), self.out_tgt[e]), self.out_w[e], accumulate=True)
        pl = arrived[:, self.pl_pre] * self.w                                  # [B, E] per-fly plastic edges
        inc.index_add_(1, self.pl_post, pl)
        self.g += inc
        self.g -= self.g * (dt / p.tau_syn)
        syn = self.g * (dt / p.tau_m)
        noise = torch.randn(B, N, device=self.device) * p.noise if p.noise else 0.0
        dv = (-self.v / p.tau_m) * dt + syn + self.ext * (dt / p.tau_m) + noise
        free = self.refrac <= 0.0
        self.v = torch.where(free, self.v + dv, self.v)
        self.refrac = torch.where(free, self.refrac, self.refrac - dt)
        self.v.clamp_(min=-p.v_thresh)
        spk = (self.v >= self.v_th) & free
        # receptors are Poisson sources at their commanded rate
        hz = self.drive_hz[:, self.driven_idx]
        pois = (torch.rand(B, hz.shape[1], device=self.device) < hz * (dt / 1000.0)) & free[:, self.driven_idx]
        spk[:, self.driven_idx] = pois
        spkf = spk.float()
        self.v = torch.where(spk, torch.full_like(self.v, p.v_reset), self.v)
        self.refrac = torch.where(spk, torch.full_like(self.refrac, p.refractory), self.refrac)
        self.last = spkf
        if self.plastic_on:
            self._learn(spkf)
        return spk

    def _learn(self, spkf):
        p, dt = self.p, self.dt
        self.kc_trace *= (1.0 - dt / p.kc_trace_ms)
        self.kc_trace += spkf[:, self.kc] * (dt / p.kc_trace_ms)
        self.da *= (1.0 - dt / p.da_trace_ms)
        for cells, wvec in self.da_types.values():
            nspk = spkf[:, cells].sum(1, keepdim=True)                           # [B,1]
            if bool((nspk > 0).any()):
                self.da += wvec.unsqueeze(0) * nspk * (dt / p.da_trace_ms) / len(cells)
        self.da_base += (self.da - self.da_base) * (dt / p.da_baseline_ms)
        phasic = torch.clamp(self.da - self.da_base, min=0.0)
        kc = self.kc_trace[:, self.pl_pre_k] * p.kc_trace_scale                # [B,E]
        da = phasic[:, self.pl_post_m] * p.da_trace_scale
        self.w *= (1.0 - p.learn_rate * torch.tanh(kc * da))
        torch.maximum(self.w, self.w0 * p.min_weight_frac, out=self.w)

    def weights_frac(self):
        return (self.w / self.w0.clamp(min=1e-9)).mean(1).cpu().numpy()

    # -- convenience --------------------------------------------------------- #

    def pop(self, name):
        return torch.as_tensor(self.cpu.pop[name], device=self.device, dtype=torch.long)

    def run(self, ms, groups: dict):
        """Step ms and return {name: [B] spike counts} for index groups."""
        out = {k: torch.zeros(self.n, device=self.device) for k in groups}
        for _ in range(int(ms / self.dt)):
            s = self.step()
            for k, idx in groups.items():
                out[k] += s[:, idx].sum(1)
        return {k: v.cpu().numpy() for k, v in out.items()}
