#!/usr/bin/env python3
"""
flysim.py - a leaky integrate-and-fire simulation of a real fly connectome,
built to drive a creature rather than to reproduce a physiology experiment.

    from flysim import FlyBrain
    brain = FlyBrain("creature_net.npz")
    brain.smell({"food": 1.0})
    brain.set_drive("hunger", 0.8)
    out = brain.run(300)
    print(out.top_actions(5))

WHAT IS REAL HERE AND WHAT IS NOT
  Real   : which neuron connects to which, how many synapses, and whether each
           neuron excites or inhibits. All measured data.
  Model  : membrane dynamics, mV-per-synapse scaling, and the mapping from world
           events onto receptor neurons. Those are choices, made here, and the
           three below are the ones that matter.

DESIGN DECISION 1 - receptors are spike sources, not integrators.
    Injecting DC current into a receptor makes it a step function: silent below
    threshold, saturated above. Real receptors emit spike trains whose *rate*
    encodes intensity. Sensory neurons here fire as a Poisson process at a rate
    set by the stimulus, so a faint smell is genuinely faint.

DESIGN DECISION 2 - APL-style feedback inhibition on the mushroom body.
    A real mushroom body keeps ~5% of Kenyon cells active for any given odour, and
    it does that with the APL neuron: one giant cell that inhibits every KC in
    proportion to total KC activity. APL is a single neuron and does not survive
    a weight>=5 connectome threshold, so it is restored explicitly. Without it,
    KCs recruit each other and the sparse code collapses into a seizure.

DESIGN DECISION 3 - the subnetwork is 6:1 excitatory, so gain is scaled down.
    Carving out sense/learn/navigate/act keeps mostly cholinergic populations and
    leaves behind the central-brain interneurons that normally balance them.
    `gain` compensates. It is a fudge factor and is labelled as one.

Sign convention: glutamate is INHIBITORY in the fly (GluCl-alpha), opposite to
vertebrate cortex. Inverting it silently produces a dead or seizing network.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


# --------------------------------------------------------------------------- #
#  parameters
# --------------------------------------------------------------------------- #

@dataclass
class Params:
    """Membrane and synapse constants. Defaults follow published fly LIF work."""
    v_thresh: float = 7.0          # mV above rest; fly neurons sit ~-52, fire ~-45
    v_reset: float = 0.0           # mV, back to rest
    tau_m: float = 20.0            # ms, membrane time constant
    refractory: float = 2.2        # ms
    mv_per_synapse: float = 0.275  # W_syn: mV per synapse (Shiu et al. 2024, the
                                   # single free parameter of that model)
    tau_syn: float = 5.0           # ms, alpha-synapse conductance decay (Shiu et al.)
    syn_delay_ms: float = 1.8      # ms, spike -> membrane potential delay (Shiu et al.)
    gain: float = 1.0              # global synaptic scaling.
    #   DESIGN DECISION 6 - with alpha-synapses this should stay at 1.0.
    #   Earlier versions delivered each spike as an INSTANTANEOUS voltage jump of
    #   W_syn, which is roughly 4x too much: in the published model the conductance
    #   decays with tau_syn=5ms against a 20ms membrane constant, so a single spike
    #   contributes about a quarter of W_syn. Compensating for that with gain=0.15
    #   was fixing the right problem in the wrong place. Modelling the synapse
    #   properly means W_syn can be used at its published value.
    noise: float = 0.15            # mV per step of membrane jitter
    dt: float = 1.0                # ms; 1.0 is stable for tau_m=20
    apl_scale: float = 0.1         # gain on the REAL APL neuron's output synapses.
    #   DESIGN DECISION 12 - APL is in the connectome. Decision 2 said it did not
    #   survive the weight>=5 threshold and restored it as a surrogate; for this build
    #   that was wrong. Type 'APL', GABA, is present, driven by projection neurons and
    #   Kenyon cells, and MEASURED it delivers 7x more inhibition onto the KCs than
    #   they get excitation (E/I 0.14 for one glomerulus, 0.22 for eight). The
    #   surrogate was a duplicate of it. Two things are wrong with the raw neuron in
    #   an LIF: it is non-spiking and graded in life, so treating it as a cell that
    #   saturates at 250 Hz overestimates its output; and its synapse count onto KCs
    #   is enormous (thousands), so the per-synapse W_syn calibrated for ordinary
    #   spiking cells is not the right unit. apl_scale is that correction, chosen so
    #   ~5% of KCs respond to an odour (measured; see kc_sparsity.py).
    apl_w: float = 0.0             # SURROGATE APL, off. Kept for measurement only:
                                   # mV of inhibitory conductance onto every KC per
                                   # (KC spike / ms) of population activity
    apl_tau_ms: float = 20.0       # surrogate integration time constant
    #   The surrogate's earlier form compared the fraction of KCs spiking PER
    #   MILLISECOND against kc_target_sparsity=0.05: per step that fraction is
    #   0.08-0.8%, so it never engaged. MEASURED: results bit-identical at apl_gain
    #   200 / 600 / 1500 in every regime. Every sparsity result before this was
    #   threshold-only.
    apl_gain: float = 200.0        # DEAD - superseded by apl_w, kept so old calls parse
    apl_tau: float = 0.35          # DEAD - superseded by apl_tau_ms
    kc_target_sparsity: float = 0.05   # DEAD - see decision 12
    sign_override: tuple = (("lLN1", -1), ("lLN2", -1))
    #   ((type_prefix, sign), ...) applied to self.sign before the matrix is built.
    #   DESIGN DECISION 13 - the lLN1 and lLN2 antennal-lobe local neurons are
    #   INHIBITORY, whatever the transmitter table says. The table calls lLN1
    #   acetylcholine (42 of 59 cells) and splits lLN2 44 GABA / 40 acetylcholine;
    #   a cell type is transmitter-homogeneous, so a 50/50 split inside one type is
    #   the classifier saying it does not know. Both are reported GABAergic
    #   panglomerular local neurons (Chou et al. 2010; Schlegel et al. 2021), the
    #   lateral inhibition that gives the antennal lobe its gain control.
    #   MEASURED, as-is: one receptor type (ORN_DM6) drove PNs in 45 of 53 glomeruli
    #   above 5 Hz, twelve at saturation, and only 7% of uniglomerular PN spikes were
    #   in DM6. 96% of the excitatory drive onto an unstimulated glomerulus's PNs came
    #   from these LNs (lLN1_bc alone 45%). Two odours sharing no glomeruli then
    #   shared 55% of their Kenyon cells - the root of every non-specific conditioning
    #   result. With the 151 cells re-signed: 1 of 53 glomeruli hot, own share 1.00,
    #   the two odours share ZERO Kenyon cells. Pass sign_override=() to see the raw
    #   table's behaviour.
    mbon_hold_frac: float = 0.85   # tonic drive on MBONs as a fraction of threshold.
    #   DESIGN DECISION 14 - the output layer needs the tonic excitation it has in
    #   life. MEASURED with the antennal lobe and APL fixed: the principal MBONs of
    #   both taught compartments (MBON05/13/18/21) fire 0-4 spikes per 800 ms odour,
    #   each silenced by real inhibition (LHCENT, APL, MBON09) that in the animal is
    #   balanced by baseline drive the 0 Hz-rest model lacks. The cells that DID fire
    #   were strays receiving ~2% of their compartment's dopamine. Same family as the
    #   photoreceptor fix (lamina_hold_frac), one layer further out. Value measured by
    #   mbon_hold.py: 0.85 gives 3.4 Hz resting MBON rate over 33 types (biological
    #   5-20 Hz), MBON05/13 responding, 23 MBON types active to an odour (was 9).
    #   MBON18/21 stay silent at any hold (LHCENT -6.5k, MBON09 -5.5k vs 7 mV).
    #   0.0 reproduces the runs before 10 Sept 2026.
    bg_hold_frac: float = 0.0      # CANDIDATE DECISION 15 - tonic drive on EVERY central
    #   neuron as a fraction of threshold, replacing the 0 Hz-rest convention with a
    #   low-rate spontaneous regime. Excludes sensory spike sources and Kenyon cells
    #   (hyperpolarised in life; their sparsity depends on it). MBON and lamina holds
    #   override it where set. Measured by bg_hold.py before adoption; 0.0 = Shiu.
    bg_hold_exclude: tuple = ("Kenyon_Cell",)   # classes left at 0 Hz rest
    syn_sat_k: float = 0.0         # CANDIDATE - saturating synapse-count -> efficacy.
    #   0 = disabled (linear, current behaviour: efficacy = synapse_count * W_syn).
    #   >0: efficacy = synapse_count * k / (synapse_count + k), a Michaelis-Menten
    #   compression with half-max at k synapses. MOTIVATION: the dark MB->CX/LAL
    #   convergence layer is net-inhibited by single cell-pair connections of
    #   100-300 synapses (whole-connectome 90th pct is 22, 99th is 81), each
    #   delivering up to 300 x 0.275 = 82 mV per spike undamped. Real synaptic
    #   efficacy is known to saturate with release-site count / postsynaptic
    #   receptor density; linear scaling has no such ceiling. NOT ADOPTED - measure
    #   with bg_hold.py-style sweep before using. Small connections (the median is
    #   6 synapses) are left almost unchanged by design.
    kc_kc_scale: float = 1.0       # DIAGNOSTIC: scale KC->KC synapses. 1.0 is the
                                   # connectome. Measured: KC->KC excitatory weight is
                                   # 55% of the PN input to KCs, and 24% of KCs get more
                                   # excitation from other KCs than from PNs.
    max_rate_hz: float = 200.0     # ceiling for driven sensory neurons
    exact_noise: bool = False      # regenerate membrane noise every step (slower)
    # --- mushroom body plasticity (off unless enable_plasticity() is called) ---
    kc_trace_ms: float = 1200.0    # eligibility trace: how long a KC stays "tagged"
    da_trace_ms: float = 600.0     # dopamine signal decay at the synapse
    learn_rate: float = 0.0005     # depression per unit coincidence PER STEP.
    #   Applied every step, so it compounds ~800x per trial: 0.06 drove the
    #   depressed synapses to zero within two trials and wiped both odours.
    min_weight_frac: float = 0.2   # floor, so a synapse weakens rather than dies
    kc_normalise: bool = True      # equalise KC excitability (see decision 11)
    kc_trace_scale: float = 40.0   # brings a strongly-driven KC trace to ~1
    da_trace_scale: float = 20.0   # brings phasic dopamine to ~1
    da_baseline_ms: float = 4000.0 # slow average defining "resting" dopamine.
    #   PHASIC, NOT TONIC. Measured: PPL1 fires 941 spikes during an UNPAIRED odour
    #   presentation purely from network drive, and only 1,348 when punishment is
    #   delivered. Treating raw dopamine as the teaching signal therefore teaches both
    #   odours almost equally. Only the component ABOVE the running baseline carries
    #   information about what just happened - which is the same reason vertebrate
    #   reward-prediction-error signals are phasic bursts, not tonic level.
    # --- vision (off unless enable_vision() is called) ---
    photoreceptor_hz: float = 90.0  # TONIC baseline rate in the dark.
    #   Fly photoreceptors are HISTAMINERGIC and their output is INHIBITORY: all 4,107
    #   of them, all 29,469 out-edges, sign -1. They depolarise in the DARK and release
    #   histamine continuously; light REDUCES release and disinhibits the lamina. So a
    #   photoreceptor is not a light detector that fires - it is a tonic brake that
    #   light releases, and vision is the contrast around this baseline.
    lamina_hold_frac: float = 0.72  # where photoreceptor targets rest, as a fraction of
    #   threshold, once their tonic drive is balanced against baseline inhibition.
    #   Without this the whole optic lobe is a switch wired to nothing: measured,
    #   224,794 receptor spikes produced EXACTLY 0 downstream spikes.
    # --- short-term synaptic depression (off unless enable_std() is called) ---
    std_u: float = 0.08            # fraction of available resource released per spike
    std_tau_rec_ms: float = 480.0  # recovery time constant
    #   MEASURED, not chosen. Sweep on escape habituation, all-DN response over 20
    #   pulses (first-pulse spikes | last/first | novel/first):
    #       no STD      4913 | 2.267 | 2.251   sensitises, non-specific
    #       U=0.03       2345 | 1.383 | 0.626   still rising
    #       U=0.08       1139 | 0.376 | 1.023   habituates, novel fully recovers
    #       U=0.15        756 | 0.219 | 0.854   stronger, costs responsiveness
    #   U=0.08 is the cheapest setting where the novel stimulus comes back to naive
    #   (1.02x), which is what makes the decrement habituation rather than fatigue.
    #   STD must be GLOBAL: applied to sensory afferents only it made things WORSE
    #   (1.52x -> 3.40x), because the accumulation is in the recurrent central loops.
    kc_thresh_scale: float = 1.5   # Kenyon-cell threshold multiplier.
    #   DESIGN DECISION 5 - Kenyon cells need a HIGH threshold, not just inhibition.
    #   Sparse odour coding works because each KC samples ~5 glomeruli and only fires
    #   when several are active at once. Global APL inhibition alone shifts every KC
    #   down by the same amount, which preserves the ordering but not the selectivity.
    #   Raising the KC threshold is what forces coincidence detection.


@dataclass
class Result:
    """What came out of a run()."""
    spikes: np.ndarray             # (steps, N) bool
    dt: float
    names: np.ndarray
    dn_index: np.ndarray

    @property
    def rates(self) -> np.ndarray:
        ms = max(self.spikes.shape[0], 1) * self.dt
        return self.spikes.sum(0) / (ms / 1000.0)

    def top_actions(self, n: int = 10) -> list[tuple[str, float]]:
        """Most active descending neurons - the creature's chosen behaviour."""
        r = self.rates[self.dn_index]
        order = np.argsort(r)[::-1][:n]
        return [(str(self.names[self.dn_index][i]), round(float(r[i]), 1))
                for i in order if r[i] > 0]

    @property
    def mean_rate(self) -> float:
        return float(self.rates.mean())


# --------------------------------------------------------------------------- #
#  the brain
# --------------------------------------------------------------------------- #

class FlyBrain:
    """A connectome-backed spiking network with senses, drives and an action bus."""

    SENSORY_CLASSES = ("olfactory", "gustatory", "mechanosensory", "mechanosensory_tactile",
                       "mechanosensory_proprioceptive", "thermosensory", "hygrosensory",
                       "visual")

    def __init__(self, path: str = "creature_net.npz", params: Params | None = None,
                 seed: int | None = 0, balance_hemispheres: bool = True,
                 engine: str = "event"):
        d = np.load(path, allow_pickle=False)
        self._path = path
        self.bodyId = d["bodyId"]
        self.type = d["type"]
        self.cls = d["cls"]
        self.sc = d["sc"]
        self.nt = d["nt"]
        self.side = d["side"] if "side" in d.files else np.full(len(d["bodyId"]), "M")
        self.sign = d["sign"].astype(np.float32)
        self.N = len(self.bodyId)
        self.p = params or Params()
        self.n_sign_overridden = 0
        if self.p.sign_override:
            ty = self.type.astype(str)
            for prefix, s in self.p.sign_override:
                m = np.char.startswith(ty, prefix)
                self.n_sign_overridden += int(m.sum())
                self.sign[m] = float(s)
        self.rng = np.random.default_rng(seed)
        self.engine = engine        # "event" (default) or "dense" for verification

        # M[i, j] = signed mV delivered to i when j spikes.
        pre, post, w = d["pre"], d["post"], d["w"].astype(np.float32)

        # DESIGN DECISION 9 - rebalance the hemispheres before anything else.
        # This specimen's right side is more completely reconstructed than its left:
        # 9.7% more synaptic weight on 1.7% more neurons. A real fly is symmetric, so
        # that gap is tracing coverage, not biology - but it means an identical
        # stimulus to both sides drives the right harder, and EVERY left/right readout
        # inherits the bias. Uncorrected, the creature turned right on all 100 ticks
        # of a run. Matching receptor counts alone did not fix it; the imbalance is
        # spread through the whole connectome, so it is corrected at the weights.
        if balance_hemispheres:
            sd = d["side"] if "side" in d.files else np.full(self.N, "M")
            inL = float(w[sd[post] == "L"].sum())
            inR = float(w[sd[post] == "R"].sum())
            if inL > 0 and inR > 0:
                self.hemi_scale = inL / inR
                w = w * np.where(sd[post] == "R", self.hemi_scale, 1.0).astype(np.float32)
            else:
                self.hemi_scale = 1.0
        else:
            self.hemi_scale = 1.0
        if self.p.syn_sat_k > 0:
            w = w * self.p.syn_sat_k / (w + self.p.syn_sat_k)
        data = self.sign[pre] * w * self.p.mv_per_synapse * self.p.gain
        if self.p.kc_kc_scale != 1.0:
            iskc = self.cls == "Kenyon_Cell"
            data = np.where(iskc[pre] & iskc[post], data * self.p.kc_kc_scale, data)
        if self.p.apl_scale != 1.0:
            is_apl = self.type.astype(str) == "APL"
            self.n_apl = int(is_apl.sum())
            data = np.where(is_apl[pre], data * self.p.apl_scale, data)
        nz = data != 0.0                    # modulatory neurons carry no fast current
        self.M = csr_matrix((data[nz], (post[nz], pre[nz])),
                            shape=(self.N, self.N), dtype=np.float32)
        self.n_edges = int(nz.sum())

        # DESIGN DECISION 10 - propagate from spikes, not from every neuron.
        # At biological firing rates 590 of 162,517 neurons spike per millisecond -
        # 0.36% of the brain. A full sparse matrix-vector product propagates from all
        # of them anyway, doing 7.3M edge updates when only ~74,000 carry a spike:
        # 99x more work than necessary, almost all of it multiplying by zero.
        # Indexing the matrix by PREsynaptic neuron instead lets each step touch only
        # the out-edges of cells that actually fired. Same arithmetic, ~99x less of it.
        _csc = self.M.tocsc()
        self._out_ptr = _csc.indptr.astype(np.int64)      # slice per presynaptic neuron
        self._out_tgt = _csc.indices.astype(np.int32)     # postsynaptic targets
        self._out_w = _csc.data.astype(np.float32)        # signed weights

        self.pop = {
            "olfactory": np.where(self.cls == "olfactory")[0],
            "gustatory": np.where(self.cls == "gustatory")[0],
            "mechano": np.where(self.cls == "mechanosensory")[0],
            "thermo": np.where(self.cls == "thermosensory")[0],
            "ALPN": np.where(self.cls == "ALPN")[0],
            "KC": np.where(self.cls == "Kenyon_Cell")[0],
            "MBON": np.where(self.cls == "MBON")[0],
            "DAN": np.where(self.cls == "DAN")[0],
            "CX": np.where(self.cls == "CX")[0],
            "DN": np.where(self.sc == "descending_neuron")[0],
            "motor": np.where(self.sc == "cb_motor")[0],
            "endocrine": np.where(self.sc == "cb_endocrine")[0],
            # present only in a whole-CNS build
            "visual": np.where(self.cls == "visual")[0],
            "VPN": np.where(self.sc == "visual_projection")[0],
            "optic": np.where(np.char.startswith(self.sc.astype(str), "ol_"))[0],
            "tactile": np.where(self.cls == "mechanosensory_tactile")[0],
            "proprio": np.where(self.cls == "mechanosensory_proprioceptive")[0],
        }

        # receptors are spike sources (decision 1)
        self.driven = np.zeros(self.N, dtype=bool)
        for c in self.SENSORY_CLASSES:
            self.driven[self.cls == c] = True
        self.drive_hz = np.zeros(self.N, dtype=np.float32)

        self._ext = np.zeros(self.N, dtype=np.float32)   # tonic current (drives)
        self._odor_map: dict[str, np.ndarray] = {}
        self._drives: dict[str, float] = {}
        self._drive_targets: dict[str, np.ndarray] = {}
        self._kc = self.pop["KC"]
        self._driven_idx = np.flatnonzero(self.driven)
        pool = max(8 * self.N, 1 << 20)
        self._noise_pool = (self.rng.standard_normal(pool, dtype=np.float32)
                            * self.p.noise) if self.p.noise else None
        self._noise_off = 0
        self._noise_step = int(self.N * 0.61803) | 1     # stride, coprime-ish to pool
        self.v_th = np.full(self.N, self.p.v_thresh, dtype=np.float32)
        self.v_th[self._kc] *= self.p.kc_thresh_scale
        if self.p.bg_hold_frac:
            held = ~self.driven
            for c in self.p.bg_hold_exclude:
                held &= self.cls != c
            self._bg_held = np.flatnonzero(held)
            self._ext[held] = (self.p.bg_hold_frac * self.v_th[held]).astype(np.float32)
        if self.p.mbon_hold_frac:
            mb = self.pop["MBON"]
            self._ext[mb] = (self.p.mbon_hold_frac * self.v_th[mb]).astype(np.float32)

        # DESIGN DECISION 11 - per-Kenyon-cell threshold normalisation.
        # A uniform threshold means the KCs with the largest total excitatory input
        # always cross first, so the "sparse code" is selected by intrinsic
        # excitability rather than by odour identity. Measured: 41-61% of the cells
        # responding to one odour also responded to a completely disjoint one, and
        # narrowing the odours made it WORSE - the same easily-excited cells kept
        # winning. Scaling each KC's threshold by its own total excitatory drive makes
        # every cell equally hard to fire, so which cells win is decided by which
        # glomeruli are active. Real mushroom bodies homeostatically tune KC
        # excitability for the same reason.
        # Normalise by PROJECTION-NEURON input only. The first version used total
        # excitatory in-weight, which includes KC->KC synapses - 55% as much weight as
        # the PN input, and for 24% of cells more than it. That gave a cell a HIGH
        # threshold for being well connected to other KCs, which has nothing to do
        # with how many glomeruli it needs to see.
        if self.p.kc_normalise and len(self._kc):
            pre_of_edge = np.repeat(np.arange(self.N), np.diff(self._out_ptr))
            is_pn = np.zeros(self.N, dtype=bool)
            is_pn[self.cls == "ALPN"] = True
            pos = (self._out_w > 0) & is_pn[pre_of_edge]
            exc = np.zeros(self.N, dtype=np.float32)
            np.add.at(exc, self._out_tgt[pos], self._out_w[pos])
            drive = exc[self._kc]
            med = float(np.median(drive[drive > 0])) if (drive > 0).any() else 1.0
            scale = np.where(drive > 0, drive / med, 1.0)
            self.v_th[self._kc] *= np.clip(scale, 0.25, 4.0).astype(np.float32)

        # glomerular channels: group olfactory receptors by type (53 of them)
        olf = self.pop["olfactory"]
        self._receptor_index: dict[str, np.ndarray] = {}
        for rt in np.unique(self.type[olf]):
            self._receptor_index[str(rt)] = olf[self.type[olf] == rt]
        self.receptor_types = sorted(self._receptor_index)

        # DESIGN DECISION 7 - two antennae, or the creature cannot steer.
        # A single odour concentration driven into both sides carries no spatial
        # information: measured turn response to food on the left vs the right was
        # 0.0000, identical to four decimals. Real flies compare across antennae
        # (and across time). Splitting the receptors by soma side is what lets
        # descending-neuron asymmetry mean something.
        # DESIGN DECISION 8 - the two antennae must be MATCHED, not merely separate.
        # This specimen's right antennal nerve is more completely reconstructed than
        # its left: 1,343 right receptors against 883 left, and ORN_VA1v is 15 vs 73.
        # Real flies are symmetric; this is a tracing artefact. Left uncorrected, an
        # identical odour presented to both sides drives the right 52% harder, and the
        # creature is told to turn right on literally every tick. Each receptor type is
        # therefore trimmed to the same count on both sides.
        self._receptor_side: dict[str, dict[str, np.ndarray]] = {}
        self.receptor_trim = 0
        for rt, idx in self._receptor_index.items():
            sd = self.side[idx]
            l, r = idx[sd == "L"], idx[sd == "R"]
            k = min(len(l), len(r))
            self.receptor_trim += (len(l) - k) + (len(r) - k)
            self._receptor_side[rt] = {"L": l[:k], "R": r[:k],
                                       "M": idx[~np.isin(sd, ["L", "R"])]}

        self.reset()

    # -- state ------------------------------------------------------------- #

    def enable_plasticity(self) -> int:
        """
        Make the KC->MBON synapses learnable.

        The rule is the established one for Drosophila: coincidence of Kenyon cell
        activity and dopamine at a KC->MBON synapse DEPRESSES it. Depression, not
        potentiation - MBONs drive approach by default, so weakening the pathway for
        a punished odour is what produces avoidance.

        Which MBONs get depressed is not something to invent: it comes from the real
        DAN->MBON wiring. PAM (reward, 316 cells) contacts 47 output neurons, PPL1
        (punishment, 24 cells) contacts 80, and only 32 overlap. That compartment
        separation is what lets reward and punishment teach opposite lessons.

        Eligibility traces matter: dopamine arrives after the odour, so the Kenyon
        cells carry a decaying tag of recent activity for the dopamine to land on.
        """
        kc = self.cls == "Kenyon_Cell"
        mbon = self.cls == "MBON"
        # locate KC->MBON edges inside the CSC (out-edge) arrays
        pre_of_edge = np.repeat(np.arange(self.N), np.diff(self._out_ptr))
        sel = kc[pre_of_edge] & mbon[self._out_tgt]
        self._plastic = np.flatnonzero(sel)
        self._plastic_pre = pre_of_edge[self._plastic]
        self._plastic_post = self._out_tgt[self._plastic]
        self._w0 = self._out_w[self._plastic].copy()

        self._kc_trace = np.zeros(self.N, dtype=np.float32)
        self._da_at_mbon = np.zeros(self.N, dtype=np.float32)
        self._da_base = np.zeros(self.N, dtype=np.float32)

        dan = self.cls == "DAN"
        ty = self.type.astype(str)
        self.pop["PAM"] = np.flatnonzero(dan & np.char.startswith(ty, "PAM"))
        self.pop["PPL1"] = np.flatnonzero(dan & np.char.startswith(ty, "PPL"))
        # Dopamine reaches an MBON through the real DAN->MBON connectivity - but that
        # wiring is NOT in the simulation matrix. DANs are dopaminergic, so their sign
        # is 0 and their edges were correctly dropped from fast transmission: dopamine
        # modulates, it does not deliver current. The mapping therefore comes from the
        # raw edge list instead, which is exactly the right separation.
        raw = np.load(self._path, allow_pickle=False)
        rpre, rpost, rw = raw["pre"], raw["post"], raw["w"].astype(np.float32)
        self._da_w = {}
        for name in ("PAM", "PPL1"):
            src = self.pop[name]
            m = np.isin(rpre, src) & mbon[rpost]
            acc = np.zeros(self.N, dtype=np.float32)
            np.add.at(acc, rpost[m], rw[m])
            self._da_w[name] = acc / max(acc.max(), 1e-9)
        self.plastic_on = True
        return len(self._plastic)

    def learn(self) -> float:
        """One plasticity step. Returns mean fractional weight change so far."""
        if not getattr(self, "plastic_on", False):
            return 0.0
        p, dt = self.p, self.p.dt
        self._kc_trace *= (1.0 - dt / p.kc_trace_ms)
        self._kc_trace[self._kc] += self.last_spikes[self._kc] * (dt / p.kc_trace_ms)
        self._da_at_mbon *= (1.0 - dt / p.da_trace_ms)
        if getattr(self, "compartments_on", False):
            # per-TYPE dopamine: each compartment is taught by its own cells, so
            # driving one type depresses one compartment instead of all 80 MBONs
            for t, d in self._da_by_type.items():
                n = float(self.last_spikes[d["cells"]].sum())
                if n:
                    self._da_at_mbon += d["w"] * n * (dt / p.da_trace_ms) / len(d["cells"])
        else:
            for name in ("PAM", "PPL1"):
                src = self.pop[name]
                n = float(self.last_spikes[src].sum())
                if n:
                    self._da_at_mbon += self._da_w[name] * n * (dt / p.da_trace_ms) / max(len(self.pop[name]), 1)
        # phasic dopamine only: the surprise, not the standing level
        self._da_base += (self._da_at_mbon - self._da_base) * (dt / p.da_baseline_ms)
        phasic = np.maximum(self._da_at_mbon - self._da_base, 0.0)
        # SCALE THE TRACES. Both were normalised by their own time constants, so a
        # strongly-driven KC trace peaked at 0.024 and phasic dopamine at 0.052; their
        # product peaked at 0.00127 and tanh then shrank it further. Net depression was
        # 0.3% of naive after 12 trials - two orders of magnitude too small to move an
        # output. Normalising each trace to reach ~1 for a strongly-driven cell puts
        # the coincidence term in a range where learn_rate means what it says.
        kc = self._kc_trace[self._plastic_pre] * p.kc_trace_scale
        da = phasic[self._plastic_post] * p.da_trace_scale
        coincide = kc * da
        if coincide.any():
            self._out_w[self._plastic] *= (1.0 - p.learn_rate * np.tanh(coincide))
            np.maximum(self._out_w[self._plastic], self._w0 * p.min_weight_frac,
                       out=self._out_w[self._plastic])
        return float(np.mean(self._out_w[self._plastic] / np.maximum(self._w0, 1e-9)))

    def enable_vision(self, baseline_hz: float | None = None) -> dict:
        """
        Turn the optic lobe on. Returns what it did, measured.

        THE PROBLEM, MEASURED. Driving all 4,107 photoreceptors at 400 Hz produced
        224,794 receptor spikes and EXACTLY ZERO spikes in the visual projection
        neurons or anywhere downstream. Every visual receptor is histaminergic and all
        29,469 of their outgoing edges carry sign -1. Photoreceptor output is
        INHIBITORY, and against this model's correct 0 Hz resting rate an inhibitory
        input has nothing to suppress. 58% of the connectome contributed no spikes.

        THE FIX, WHICH IS THE ACTUAL BIOLOGY. Fly photoreceptors depolarise in the dark
        and release histamine tonically; light REDUCES release and disinhibits the
        lamina monopolar cells. Two things follow, and both are needed:

          1. Photoreceptors fire at a TONIC BASELINE and light drives them DOWN. The
             sign of the whole channel inverts: `see()` takes brightness, and brightness
             lowers the rate.
          2. The cells being inhibited must have their own depolarising drive, or there
             is still nothing to take away. Each photoreceptor target gets a tonic
             current sized so that, against the inhibition it receives at baseline, it
             rests at `lamina_hold_frac` of threshold.

        The steady state is exact rather than tuned. With g += w on arrival and
        g -= g*dt/tau_syn, a presynaptic rate r (spikes/ms) gives g_ss = W*r*tau_syn,
        and dv = (-v/tau_m)dt + g(dt/tau_m) + ext(dt/tau_m) settles at v = g + ext.
        So ext = v_hold - g_ss holds the cell exactly where we want it.
        """
        vis = self.pop["visual"]
        if len(vis) == 0:
            return {"receptors": 0, "targets": 0}
        p = self.p
        base = float(p.photoreceptor_hz if baseline_hz is None else baseline_hz)
        self._vis = vis
        self._vis_baseline = base

        # Total photoreceptor weight arriving at each neuron, read straight out of the
        # CSC the simulation actually uses. Re-deriving it from the npz would miss the
        # gain and the hemisphere rebalance and quietly hold the lamina in the wrong
        # place.
        inc = np.zeros(self.N, dtype=np.float64)
        for j in vis:
            a, b_ = self._out_ptr[j], self._out_ptr[j + 1]
            if b_ > a:
                np.add.at(inc, self._out_tgt[a:b_], self._out_w[a:b_])
        targets = np.flatnonzero(inc != 0.0)

        g_ss = inc[targets] * (base / 1000.0) * p.tau_syn      # mV at baseline
        v_hold = p.lamina_hold_frac * float(self.v_th[targets].mean())
        self._vis_targets = targets
        self._vis_ext = (v_hold - g_ss).astype(np.float32)
        self._ext[targets] = self._vis_ext
        self.drive_hz[vis] = base
        self.vision_on = True
        return {"receptors": int(len(vis)), "targets": int(len(targets)),
                "baseline_hz": base, "mean_hold_mv": float(v_hold),
                "mean_ext_mv": float(self._vis_ext.mean())}

    def see(self, brightness) -> None:
        """
        Show the eye something. `brightness` is 0 (dark) .. 1 (bright), a scalar or one
        value per visual receptor. Brightness LOWERS the firing rate, because that is
        what light does to a histaminergic photoreceptor.
        """
        if not getattr(self, "vision_on", False):
            self.enable_vision()
        b = np.clip(np.asarray(brightness, dtype=np.float32), 0.0, 1.0)
        self.drive_hz[self._vis] = self._vis_baseline * (1.0 - b)

    def enable_std(self, populations=("visual", "mechano", "olfactory", "gustatory",
                                      "VPN", "ALPN")) -> int:
        """
        Short-term synaptic depression on the listed presynaptic populations.

        The model had NO adaptation of any kind - no synaptic depression, no
        spike-frequency adaptation - so repeated identical stimulation could only
        accumulate. Measured: the descending population rose to 1.52x its first
        response over 20 pulses and the novel stimulus was matched, i.e. the escape
        pathway sensitised non-specifically instead of habituating.

        Depression is the standard mechanism for habituation and it is well documented
        at fly sensory synapses. One resource variable per presynaptic neuron: a spike
        consumes `std_u` of what is left, and it recovers with `std_tau_rec_ms`.
        """
        if populations is None:
            # every presynaptic neuron. Sensory-only depression was measured to make
            # habituation WORSE (all-DN response 1.52x -> 3.40x over 20 pulses): it
            # lowers the first pulse without touching the recurrent central loops that
            # are doing the accumulating.
            mask = np.ones(self.N, dtype=bool)
        else:
            mask = np.zeros(self.N, dtype=bool)
            for name in populations:
                idx = self.pop.get(name)
                if idx is not None and len(idx):
                    mask[idx] = True
        self._std_mask = mask
        self._std_x = np.ones(self.N, dtype=np.float32)
        self.std_on = True
        return int(mask.sum())

    def enable_compartments(self) -> dict:
        """
        Give each dopaminergic CELL TYPE its own map onto the MBONs.

        enable_plasticity() collapses all 24 PPL1 cells into ONE normalised scalar per
        MBON, so every MBON that PPL1 touches is depressed together and punishment
        cannot be selective about anything. Real mushroom-body learning is
        compartmentalised: individual PPL1 and PAM types innervate individual
        compartments, and only the Kenyon synapses in that compartment change.

        Those types are in the data and were being averaged away. This builds one
        weight vector per type, so driving a single type teaches a single compartment.
        Measured consequence of the old behaviour: punishing CS+ and punishing CS- both
        produced depression of the same sign (-0.040 vs -0.025, p=0.878 for reversal).
        """
        if not getattr(self, "plastic_on", False):
            self.enable_plasticity()
        raw = np.load(self._path, allow_pickle=False)
        rpre, rpost, rw = raw["pre"], raw["post"], raw["w"].astype(np.float32)
        mbon = self.cls == "MBON"
        ty = self.type.astype(str)
        self._da_by_type = {}
        for fam in ("PAM", "PPL1"):
            src = self.pop[fam]
            for t in sorted(set(ty[src])):
                cells = src[ty[src] == t]
                m = np.isin(rpre, cells) & mbon[rpost]
                if not m.any():
                    continue
                acc = np.zeros(self.N, dtype=np.float32)
                np.add.at(acc, rpost[m], rw[m])
                peak = float(acc.max())
                if peak <= 0:
                    continue
                self._da_by_type[t] = {"family": fam, "cells": cells,
                                       "w": acc / peak,
                                       "n_mbon": int((acc > 0).sum())}
        self.compartments_on = True
        return {t: {"family": d["family"], "cells": int(len(d["cells"])),
                    "mbons": d["n_mbon"]} for t, d in self._da_by_type.items()}

    def stimulate_type(self, dan_type: str, hz: float, mv: float | None = None) -> int:
        """
        Drive one dopaminergic type - one compartment's teaching signal.

        `mv` OVERRIDES the 14 mV ceiling that stimulate() applies, and it has to.
        MEASURED on PPL105 over 800 ms, spikes with and without an odour present:

            _ext 12.6 mV (stimulate's maximum)   83 spikes alone,   0 with CS+
            _ext 20 mV                          132 spikes alone,   2 with CS+
            _ext 45 mV                          228 spikes alone,  91 with CS+
            _ext 70 mV                          266 spikes alone, 218 with CS+

        Odour presentation drives enough inhibition onto the PPL1 cells to silence
        them completely at the strongest drive stimulate() can deliver - which is
        precisely when punishment is supposed to arrive. Any conditioning protocol
        built on stimulate() has been pairing the odour with NO dopamine at all.
        """
        d = self._da_by_type.get(dan_type)
        if d is None:
            raise KeyError(f"unknown DAN type {dan_type!r}; "
                           f"have {sorted(self._da_by_type)[:8]}...")
        if hz <= 0:
            self._ext[d["cells"]] = 0.0
        else:
            self._ext[d["cells"]] = (mv if mv is not None
                                     else 14.0 * min(1.0, hz / 200.0))
        return len(d["cells"])

    def stimulate(self, population: str, hz: float) -> None:
        """Drive a named population directly - used to deliver reward or punishment."""
        idx = self.pop[population]
        self._ext[idx] = 0.0 if hz <= 0 else 14.0 * min(1.0, hz / 200.0)

    def reset(self) -> None:
        self.v = np.zeros(self.N, dtype=np.float32)
        self.refrac = np.zeros(self.N, dtype=np.float32)
        self.last_spikes = np.zeros(self.N, dtype=np.float32)
        self.last_idx = np.zeros(0, dtype=np.int64)
        self.g = np.zeros(self.N, dtype=np.float32)      # alpha-synapse conductance
        # Traces MUST be cleared here. Leaving them meant dopamine from a punished
        # trial survived into the unpaired control trial and depressed that odour
        # too - which is exactly how a specific association becomes a global one.
        if hasattr(self, "_kc_trace"):
            self._kc_trace[:] = 0.0
            self._da_at_mbon[:] = 0.0
            # NB: _da_base deliberately survives reset - it is the slow expectation
            # of resting dopamine, and rebuilding it each trial would defeat it.
        n_dly = max(1, int(round(self.p.syn_delay_ms / self.p.dt)))
        # the delay line carries spike INDICES (~590 ints) rather than a full
        # 162,517-element vector, which also removes a flatnonzero scan per step
        self._dly = [np.zeros(0, dtype=np.int64) for _ in range(n_dly)]
        self._acc = np.zeros(self.N, dtype=np.float32)   # persistent accumulator
        self._apl = 0.0
        # short-term depression: one resource variable per PRESYNAPTIC neuron. The
        # delay line carries the release scale alongside the indices so a spike
        # delivers the resource it had WHEN IT FIRED, not 1.8 ms later after its own
        # decrement - otherwise every spike under-delivers by exactly one step of U.
        if getattr(self, "std_on", False):
            self._std_x[:] = 1.0
        self._dly_scale = [None for _ in range(n_dly)]

    # -- senses ------------------------------------------------------------ #

    def define_odor(self, name: str, n_channels: int = 20, seed: int | None = None) -> None:
        """
        Give a world object a smell.

        DESIGN DECISION 4 - odours are defined over receptor TYPES, not neurons.
        The antennal lobe is glomerular: every olfactory receptor neuron of one type
        converges on one glomerulus, and the projection neurons read glomeruli. Picking
        random individual receptors spreads a smell thinly across every glomerulus, and
        the first synapse averages it away - measured separation collapsed from +1.01 at
        the receptors to +0.02 one layer later. Choosing a handful of the 53 receptor
        types instead keeps the channel structure the brain is built to read.
        """
        rng = np.random.default_rng(seed if seed is not None else abs(hash(name)) % (2**32))
        types = self.receptor_types
        k = min(n_channels, len(types))
        chosen = rng.choice(len(types), size=k, replace=False)
        strengths = rng.uniform(0.4, 1.0, size=k)
        chan = {}
        for ci, st in zip(chosen, strengths):
            chan[types[ci]] = float(st)
        self._odor_map[name] = chan

    def smell(self, odors: dict[str, float]) -> None:
        """Present a mixture, e.g. {"food": 1.0, "rot": 0.3}. Replaces any prior smell."""
        for name in odors:
            if name not in self._odor_map:
                self.define_odor(name)
        self.drive_hz[self.pop["olfactory"]] = 0.0
        for name, strength in odors.items():
            for rtype, st in self._odor_map[name].items():
                idx = self._receptor_index[rtype]
                self.drive_hz[idx] += self.p.max_rate_hz * st * float(strength)

    def smell_bilateral(self, left: dict[str, float], right: dict[str, float]) -> None:
        """
        Present separate mixtures to the two antennae.

        This is what makes taxis possible: the left-right difference in receptor
        drive propagates to a left-right difference in descending-neuron firing,
        which the body reads as a turn.
        """
        for name in set(left) | set(right):
            if name not in self._odor_map:
                self.define_odor(name)
        self.drive_hz[self.pop["olfactory"]] = 0.0
        for sidekey, mix in (("L", left), ("R", right)):
            for name, strength in mix.items():
                for rtype, st in self._odor_map[name].items():
                    grp = self._receptor_side[rtype]
                    idx = grp[sidekey]
                    if len(idx):
                        self.drive_hz[idx] += self.p.max_rate_hz * st * float(strength)
                    if len(grp["M"]):    # midline receptors get the mean
                        self.drive_hz[grp["M"]] += (self.p.max_rate_hz * st
                                                    * float(strength) * 0.5)

    def taste(self, quality: float, n: int = 120) -> None:
        """quality > 0 sweet, < 0 bitter. Different receptor sets, as in the animal."""
        g = self.pop["gustatory"]
        if len(g) == 0:
            return
        pick = g[:min(n, len(g))] if quality >= 0 else g[-min(n, len(g)):]
        self.drive_hz[g] = 0.0
        self.drive_hz[pick] = self.p.max_rate_hz * min(abs(float(quality)), 1.0)

    # -- drives (the "needs" layer) ---------------------------------------- #

    def define_drive(self, name: str, population: str = "endocrine",
                     n: int | None = None, seed: int | None = None) -> None:
        """
        Bind a need to a slice of the neuromodulatory machinery.

        This is the honest version of a Sims motive bar. The need is not a number a
        utility function reads - it is tonic current injected into neurons wired into
        everything downstream. Hunger changes what the animal does because it changes
        the network's operating point.
        """
        rng = np.random.default_rng(seed if seed is not None else abs(hash(name)) % (2**32))
        src = self.pop[population]
        if len(src) == 0:
            raise ValueError(f"population {population!r} is empty")
        k = min(n or max(4, len(src) // 3), len(src))
        self._drive_targets[name] = rng.choice(src, size=k, replace=False)
        self._drives[name] = 0.0

    def set_drive(self, name: str, level: float, gain_mv: float = 9.0) -> None:
        """level in [0, 1]. 0 = sated, 1 = desperate."""
        if name not in self._drive_targets:
            self.define_drive(name)
        self._ext[self._drive_targets[name]] = gain_mv * float(np.clip(level, 0.0, 1.0))
        self._drives[name] = float(level)

    @property
    def drives(self) -> dict[str, float]:
        return dict(self._drives)

    # -- dynamics ---------------------------------------------------------- #

    def step(self) -> np.ndarray:
        p, dt = self.p, self.p.dt

        # alpha-synapse with transmission delay:
        #   g <- g + W  on arrival of a delayed spike, then  dg/dt = -g/tau_syn
        #   dv/dt = (g - v) / tau_m
        arrived = self._dly.pop(0)
        arrived_scale = self._dly_scale.pop(0)
        if getattr(self, "std_on", False):
            # release scale for the spikes about to enter the delay line
            self._dly_scale.append(self._std_x[self.last_idx].copy()
                                   if self.last_idx.size else None)
        else:
            self._dly_scale.append(None)
        self._dly.append(self.last_idx)
        if arrived.size:
            self.g += self._propagate(arrived, arrived_scale)
        self.g -= self.g * (dt / p.tau_syn)
        syn = self.g * (dt / p.tau_m)

        # APL surrogate: global feedback inhibition onto Kenyon cells (decisions 2, 12).
        # _apl is a smoothed KC population spike count per ms; apl_w converts it to an
        # inhibitory conductance (mV) that every KC sees, delivered like any synapse.
        if len(self._kc) and p.apl_w:
            syn[self._kc] -= p.apl_w * self._apl * (dt / p.tau_m)

        # Generating 162,517 Gaussians every step cost 2.25 ms - 35% of the whole
        # step, more than the synaptic propagation itself. Membrane jitter does not
        # need cryptographic independence, so it is drawn once into a pool and read
        # from a rolling offset. exact_noise=True restores per-step generation.
        if p.noise:
            if p.exact_noise:
                noise = self.rng.standard_normal(self.N, dtype=np.float32) * p.noise
            else:
                o = self._noise_off
                noise = self._noise_pool[o:o + self.N]
                self._noise_off = (o + self._noise_step) % (self._noise_pool.size - self.N)
        else:
            noise = np.float32(0.0)
        dv = (-self.v / p.tau_m) * dt + syn + self._ext * (dt / p.tau_m) + noise

        free = self.refrac <= 0.0
        self.v[free] += dv[free]
        self.refrac[~free] -= dt
        np.maximum(self.v, -p.v_thresh, out=self.v)

        spk = (self.v >= self.v_th) & free

        # receptors ignore membrane dynamics and fire at their commanded rate.
        # Drawing for all N was 0.82 ms/step to serve ~4,000 receptors; the draw is
        # restricted to neurons that are both driven and actually commanded to fire.
        di = self._driven_idx
        if di.size:
            hz = self.drive_hz[di]
            live = hz > 0
            if live.any():
                sub = di[live]
                pois = self.rng.random(sub.size) < (hz[live] * dt / 1000.0)
                spk[sub] = pois & free[sub]
                dead = di[~live]
                if dead.size:
                    spk[dead] = False
            else:
                spk[di] = False

        self.v[spk] = p.v_reset
        self.refrac[spk] = p.refractory
        self.last_spikes = spk.astype(np.float32)
        self.last_idx = np.flatnonzero(spk)

        # Tsodyks-Markram depression: a spike consumes U of the available resource,
        # which recovers exponentially. This is the mechanism the model was missing -
        # with no adaptation of any kind, repeated stimulation could only accumulate,
        # and the escape pathway SENSITISED +52% over 20 pulses instead of habituating.
        if getattr(self, "std_on", False):
            m = self._std_mask
            self._std_x[m] += (1.0 - self._std_x[m]) * (dt / p.std_tau_rec_ms)
            hit = self.last_idx[m[self.last_idx]] if self.last_idx.size else self.last_idx
            if hit.size:
                self._std_x[hit] *= (1.0 - p.std_u)

        if len(self._kc) and p.apl_w:
            # APL integrates KC population activity with its own time constant
            # rather than tracking it step to step - what the real (graded,
            # non-spiking) neuron does, and what stops the loop ringing.
            n_kc = float(spk[self._kc].sum()) / dt
            self._apl += (n_kc - self._apl) * (dt / p.apl_tau_ms)

        return spk

    def _propagate(self, spikes: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
        """
        Sum the out-edges of every neuron that fired. `spikes` is an INDEX array.

        Identical arithmetic to M.dot(spikes) - verified spike-for-spike against the
        dense path - but it only reads the edges that carry a spike this step.
        """
        if self.engine == "dense":
            v = np.zeros(self.N, dtype=np.float32)
            v[spikes] = 1.0 if scale is None else scale
            return self.M.dot(v)

        src = spikes
        starts = self._out_ptr[src]
        counts = self._out_ptr[src + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return np.zeros(self.N, dtype=np.float32)

        # ragged gather: turn per-neuron [start, start+count) slices into one index array
        ends = np.cumsum(counts)
        idx = np.arange(total, dtype=np.int64) - np.repeat(ends - counts, counts)
        idx += np.repeat(starts, counts)

        # accumulate into a persistent float32 buffer: np.bincount would allocate
        # and zero 1.3 MB of float64 every step, which cost more than the gather
        self._acc.fill(0.0)
        if scale is None:
            np.add.at(self._acc, self._out_tgt[idx], self._out_w[idx])
        else:
            np.add.at(self._acc, self._out_tgt[idx],
                      self._out_w[idx] * np.repeat(scale, counts))
        return self._acc

    def run(self, ms: float, record: bool = True) -> Result:
        steps = int(round(ms / self.p.dt))
        rec = np.zeros((steps, self.N), dtype=bool) if record else None
        for t in range(steps):
            spk = self.step()
            if record:
                rec[t] = spk
        return Result(rec if record else np.zeros((0, self.N), bool),
                      self.p.dt, self.type, self.pop["DN"])

    # -- sanity ------------------------------------------------------------ #

    def diagnose(self, ms: float = 400.0, stimulus: float = 1.0) -> dict:
        """
        The check that has to pass before anything is built on top.

        Two failure modes kill these models: silence (gain too low, or signs inverted
        so inhibition swamps everything) and seizure (everything at ceiling, carrying
        no information). Biological fly neurons idle in the 0.5-10 Hz range, and the
        mushroom body should hold roughly 5% of its Kenyon cells active.
        """
        self.reset()
        self.smell({"probe": stimulus})
        self.set_drive("probe_drive", 0.5)
        r = self.run(ms)
        rates = r.rates
        internal = ~self.driven
        kc_inst = float(r.spikes[:, self._kc].mean()) if r.spikes.size else 0.0
        out = {
            "mean_rate_hz": round(float(rates[internal].mean()), 2),
            "median_rate_hz": round(float(np.median(rates[internal])), 2),
            "max_rate_hz": round(float(rates.max()), 1),
            "silent_frac": round(float((rates[internal] == 0).mean()), 3),
            "kc_sparsity": round(kc_inst, 4),   # co-active fraction per ms; biological ~0.05
            "active_dns": int((rates[self.pop["DN"]] > 0).sum()),
            "dn_mean_hz": round(float(rates[self.pop["DN"]].mean()), 2),
        }
        m = out["mean_rate_hz"]
        out["verdict"] = ("DEAD - raise gain, or signs are inverted" if m < 0.05 else
                          "SEIZING - lower gain" if m > 100 else
                          "plausible")
        return out

    def population_rates(self, r: Result) -> dict[str, float]:
        rates = r.rates
        return {k: round(float(rates[v].mean()), 2) for k, v in self.pop.items() if len(v)}

    def __repr__(self) -> str:
        return (f"<FlyBrain {self.N:,} neurons, {self.n_edges:,} signed synapses, "
                f"dt={self.p.dt}ms, engine={self.engine}>")


if __name__ == "__main__":
    import sys
    b = FlyBrain(sys.argv[1] if len(sys.argv) > 1 else "creature_net.npz")
    print(b)
    for k, v in b.diagnose().items():
        print(f"  {k:<16} {v}")
