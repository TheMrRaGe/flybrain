# Fly connectome simulation — established facts

Read this before touching the code. Everything here was measured, not assumed.
Re-deriving any of it costs hours.

## Data

Male CNS Connectome v1.0 (HHMI Janelia FlyEM + Google Research), **CC-BY 4.0** —
commercial game use is permitted with attribution.

Three files matter, ~1.1 GB. The other ~22 GB of the public release (`syn-points`,
`syn-partners`, `tbar-neurotransmitters`) is per-synapse 3D coordinates: anatomy, not
dynamics. Nothing in the simulation reads them.

| file | contents |
|---|---|
| `connectome-weights` | 151,856,684 segment→segment rows; weight = synapse count |
| `body-annotations` | identity: `type`, `class`, `superclass`, `somaSide`, `rootSide` |
| `body-neurotransmitters` | `consensus_nt` → the only source of synapse **sign** |

Filtering to traced, typed neurons at weight ≥ 5 gives **162,517 neurons /
6,138,378 connections**. 58% of the brain is optic lobe.

## Traps that cost real time

- **Glutamate is INHIBITORY in the fly** (GluCl-α), opposite to vertebrate cortex.
  Inverting this silently produces a dead or seizing network.
- **Sensory neurons have `somaSide == "M"`.** All 2,635 olfactory, all 1,416
  gustatory, 4,078 of 4,107 visual. Their somas sit in the antenna, palp and retina —
  outside the imaged CNS volume. Laterality comes from **`rootSide`** instead.
  Without this the creature is spatially blind: measured turn response to a stimulus
  left vs right was 0.0000, identical to four decimals.
- **The right hemisphere is more completely traced**: 9.7% more synaptic weight on
  1.7% more neurons. Every left/right readout inherits that bias. Shiu et al. hit the
  same thing and sidestepped it by only ever stimulating one side.
- **Odours must be defined over the 53 receptor TYPES**, not random individual
  receptors. The antennal lobe is glomerular; random receptors smear across all
  glomeruli and the first synapse averages them away.
- `~some_python_bool` is `-1`, not `False`. Used in a numpy mask it silently turns a
  boolean mask into integer fancy-indexing.
- **The whole-DN-population steering readout is sign-inverted** relative to the real
  signal. It reports residual bias, not steering. See below.

## Model parameters — Shiu et al. 2024, Nature 634:210

All measured or published, none tuned by us:

```
V_rest      -52 mV      tau_m        20 ms      W_syn    0.275 mV  (their one free parameter)
V_threshold -45 mV      tau_syn       5 ms      dt        1.0 ms   (0.1 ms in the paper)
V_reset     -52 mV      syn_delay   1.8 ms      refractory 2.2 ms
```

**α-synapse dynamics and the 1.8 ms delay are not optional.** Delivering each spike as
an instantaneous voltage jump is ~4× too much (the conductance decays over 5 ms against
a 20 ms membrane constant), which forces a bogus gain fudge. With the synapse modelled
properly, `W_syn` works at its published value and `gain` stays at 1.0.

Three things the paper confirms that look like bugs:

1. Basal firing is **0 Hz by design** — "inhibitory connections to an inactive neuron
   have no effect." The network being silent at rest is correct, not broken.
2. **Absolute firing rates are not meaningful.** Only differences between conditions
   are. Always baseline-subtract.
3. Their robustness check shuffles connectivity while preserving the global weight
   distribution; only 1 of 100 shuffled networks reproduced the result.

## Circuit landmarks

| population | count | role |
|---|---|---|
| Kenyon cells | 4,064 | sparse odour code |
| MBON | 97 | learned valence readout |
| **PAM** | **316** | **dopamine REWARD channel** (2,595 → KC) |
| **PPL1** | **24** | **dopamine PUNISHMENT channel** (3,029 → KC) |
| KC→MBON | 33,496 | **the plastic synapses — where memory physically lives** |
| Central Complex | 2,950 | heading / path integration, 138,641 internal edges |
| Descending neurons | 1,310 | the action bus, 480 types |
| DNa02 | 1 per side | best-characterised turning neuron |
| DNp01 | 1 per side | Giant Fiber, escape |

Axo-axonic DN→DN connections reproduce Ceballos et al. 2026 closely: 1.23% of possible
pairs connected (paper ~1%), heterotypic 62% cholinergic (paper ~60%). **8,692
contralateral heterotypic DN→DN connections, 40% inhibitory** — a mutual-inhibition
winner-take-all circuit already present in the matrix.

## Steering — measured, 30 trials × 1000 ms per condition

Mirror-image stimuli (one antenna 1.0, the other 0.4), turn index = (R−L)/(R+L):

| readout | difference | d′ | p | spikes/trial |
|---|---|---|---|---|
| DNa02 (1/side) | +0.467 | 1.11 | 1.6e-4 | 1 |
| **DNa family** | **+0.221** | **4.21** | **3e-11** | **330** |
| all 1,310 DNs | −0.011 | −1.70 | 6e-7 | 11,853 |

**Use the DNa family.** DNa02 alone is right but fires ~1 spike/trial — too sparse to
read per-tick. The whole-population readout is significant *with the wrong sign*: it
tracks residual asymmetry, not steering. Averaging 1,310 neurons doesn't dilute the
signal, it inverts it.

Shuffle control not yet run.

## Performance

Event-driven propagation, verified **spike-for-spike identical** to the dense path.

- 2.8 ms/step (cloud container), ~2.2 ms/step on Heath's laptop → **0.45× real time**
- 590 of 162,517 neurons spike per ms (**0.36%**) — a dense matvec does 99× more work
  than necessary
- Predicted 99× speedup, got **4.1×**: event-driven fixes synaptic work, but
  per-neuron O(N) work doesn't shrink and now dominates
- Profiling surprises: Gaussian noise for all N cost 2.25 ms/step (35% of the step) —
  now read from a pool; Poisson draws were sampling all 162,517 neurons to serve
  ~4,000 receptors; `np.add.at` into a persistent buffer beats `np.bincount`, which
  allocates 1.3 MB of float64 per call

**Memory: 119 MB connectome (read-only, shareable) + 5.2 MB per creature.**
A swarm of 100 is 639 MB total — the connectome is shared, only state duplicates.

## Conditioning — NOT DEMONSTRATED, and now properly measured

> **PARTLY SUPERSEDED.** Those runs used `stimulate()` (which fires a DAN 0 times
> during an odour) and a parameter regime leaving only 33 of 4,064 Kenyon cells
> active. See *Conditioning after both fixes* below.

Earlier claims are retracted. Every conditioning number before `conditioning2.py` came
from ONE presentation of each odour, against a measured noise floor of mean −0.108,
sd 0.053. All of them (−0.012, −0.014, −0.059, +0.183, +0.179) sat inside it.

### The protocol that actually settles it

`conditioning2.py` — paired, three-armed. Discrimination index
`d = (CS+ − CS−)/(CS+ + CS−)` measured before and after training **using the same noise
seed**, so each seed is its own control:

| arm | plasticity | punished |
|---|---|---|
| punish CS+ | on | CS+ |
| punish CS− | on | CS− |
| no plastic | off | CS+ |

**With paired seeds the control arm's delta is exactly 0.0000** — the measurement is
perfectly reproducible when no weights change. All the earlier "noise" was from
comparing *unpaired* measurements. Use this design for anything measured from here on.

**The specificity test is the reversal arm.** If learning is associative, punishing CS−
instead of CS+ must flip the sign of the shift.

### Result

| arm | delta | sd |
|---|---|---|
| punish CS+ | −0.040 | 0.096 |
| punish CS− | −0.025 | 0.059 |
| no plastic | +0.000 | 0.000 |

**Same sign both ways** (p=0.878 for the reversal). Plasticity produces a small
non-specific depression that does not follow the contingency. Not distinguishable from
control (d′=−0.59, p=0.4).

### Why — a chain of four measured causes

1. **Depression was ~100× too weak.** The coincidence term peaked at 0.00127: both
   traces were normalised by their own time constants (KC ~0.024, phasic DA ~0.052),
   multiplied together, then compressed by tanh. Fixed with `kc_trace_scale` /
   `da_trace_scale`; one trial now depresses 4.5% of synapses by ~2%.
2. **Still non-specific.** With disjoint input channels, **41.4% of CS+ Kenyon cells
   also responded to CS−**. Depression hits shared cells, so punishing either odour
   lowers both responses.
3. **Narrowing the odours made it worse** (51.6% overlap at 5 glomeruli). The same
   easily-excited KCs win regardless of odour — the code was being selected by
   intrinsic excitability, not identity.
4. **Per-KC threshold normalisation** (scale each cell's threshold by its own total
   excitatory drive) brought overlap to **36.1%**. Real improvement, still too high.

### Next

Get KC overlap to ~15%. The remaining candidate is the one piece of antennal lobe
computation not yet implemented: **divisive normalisation of PN input**, a known
gain-control step that equalises total drive across odours. Until overlap comes down,
no learning rule can be odour-specific, because the codes themselves are not separable.

## Probe ladder — the overlap theory is refuted (`probe_ladder.py`)

60 paired trials, 5-fold held-out, plasticity off. Three readouts of the SAME Kenyon
activity, each adding one biological constraint:

| rung | constraint | held-out accuracy |
|---|---|---|
| 1 | unconstrained linear readout over 4,064 KCs | **100.0%** |
| 2 | restricted to the real KC→MBON wiring, weights non-negative | **100.0%** |
| 3 | same wiring, weights may only DECREASE (what the DA rule can express) | **93.3%** |

Rung 3 is a closed-form optimum, not an optimisation that might have got stuck: the
objective is linear in the per-cell depression factor, so the best possible setting is
at the box corners. It is a **ceiling**.

    depression-only ceiling   -0.1219 -> +0.0947   (shift +0.2166)
    measured learning shift                        -0.0400

**5.4x too small AND the wrong sign.** The codes are separable, the anatomy can express
the discrimination, and depression is a sufficient mechanism. KC overlap measured 72.2%
here — worse than the 36.1% the theory was built on — and the ceiling held anyway.

**Divisive normalisation of PN input is therefore NOT the next step.** The failure is in
credit assignment: which synapses the dopamine signal actually reaches. The previous
"Next" section above is superseded.

## Vision delivers exactly zero (`habituation.py`, docstring)

> **SUPERSEDED — FIXED.** See *FIXED: vision now works* below. Kept because the
> diagnosis is what the fix is built on.

Driving all 4,107 photoreceptors at 400 Hz:

    224,794 receptor spikes  ->  0 VPN spikes  ->  0 descending spikes

Not few. Zero. Mechanosensory drive at 180 Hz through the same machinery gives 10,324
descending spikes and 2 DNp01 spikes.

**Cause:** all 4,107 visual receptors are histaminergic and all 29,469 of their outgoing
edges carry sign −1. Photoreceptor output is INHIBITORY in the fly. Against the model's
0 Hz basal rate — which is correct and published — the visual system is a switch wired
to nothing: there is no activity for it to suppress. **58% of the connectome is optic
lobe and it contributes zero spikes.**

Same family as the glutamate and somaSide traps, one layer further out. Cannot be fixed
by driving harder. Vision needs a tonic baseline it can modulate DOWNWARD. Until then,
any visual experiment is measuring nothing, and `build_creature.py` dropping the optic
lobe costs nothing.

## Escape does not habituate — it sensitises (`habituation.py`)

> **SUPERSEDED — FIXED.** See *FIXED: escape habituates* below. This is the
> no-adaptation baseline the fix is measured against.

8 seeds x (20 identical mechanosensory pulses + 1 novel + 3 retest), 80 ms pulse /
220 ms gap, response normalised per seed to its own first pulse:

| population | last/first | novel/first | specific? |
|---|---|---|---|
| all DN (1,310) | **1.517** | 1.500 | no |
| DNp escape family | 1.069 | 1.035 | no |
| MBON | 1.272 | 1.197 | no |
| Giant Fiber (DNp01) | 0.6 spikes on pulse 1 — too sparse to normalise | | |

Response RISES 52% over 20 repetitions and the novel stimulus is matched, so nothing is
stimulus-specific. Predicted before running: there is no plastic synapse in this pathway,
no short-term synaptic depression and no spike-frequency adaptation, so residual
excitation has nowhere to go but up. **To get habituation, add short-term synaptic
depression** — a mechanism the model lacks entirely.

The Giant Fiber is unreadable for the same reason DNa02 is: ~1 spike per presentation.

## Embodied episode — starves without eating (`results/arena_run.jsonl`)

300 ticks requested, `--gain 1.0`, seed 3. Died at tick 188.

    events: {'died:starved': 1}      feeding events: 0      drinking events: 0

Seven food sources in the arena. Energy and water fall in straight lines because nothing
ever interrupted them. Turn command positive on 8 of 10 sampled ticks; speed never left
0.20–0.23. Steering is real (DNa family d'=4.21) and the action bus works — **chemotaxis
does not close the loop.**

Note `flyworld.py --gain` still defaults to 0.15, a leftover from before the alpha-synapse
fix; decision 6 says it should be 1.0.

## The teaching signal, measured (`teaching_signal.py`)

> **PARTLY INVALIDATED.** These numbers were taken with `stimulate()`, which cannot
> fire a DAN during an odour at all (see *TRAP* below). The tag-overlap result
> (cosine 0.846) stands; the dopamine ratios were measured against a punishment that
> largely never arrived.

probe_ladder.py ruled out the KC code, so the fault is credit assignment. Credit
assignment has two inputs and BOTH are broken. 6 repeats, 600 ms presentations,
punishment over the second half.

| condition | PPL1 spikes | peak phasic DA |
|---|---|---|
| CS- alone | 666 | 0.2389 |
| CS+ alone | 806 | **0.2914** |
| CS- + punish | 881 | 0.3326 |
| CS+ + punish | 1030 | 0.3839 |

**1. The teaching signal barely carries the contingency.** Punishment is only
**1.32x** the dopamine the odour drives on its own, and CS+ ALONE (0.291) produces
nearly as much dopamine as CS- WITH punishment (0.333). Odour identity moves dopamine
about as much as the reinforcement does.

**2. The eligibility tag is not odour-specific.** cosine(trace | CS+, trace | CS-) =
**0.846**; the top-200 tagged cells overlap **52.7%**.

Depression is therefore applied nearly equally, to nearly the same synapses, in every
condition. That is exactly the measured -0.040 / -0.025 same-sign result.

### The obvious parameter fix does NOT work — do not retry it

Shortening the dopamine baseline so it subtracts the odour-driven level, and shortening
the KC trace so the tag reflects only the current odour:

| da_baseline_ms | kc_trace_ms | DA ratio | tag cosine |
|---|---|---|---|
| 4000 (current) | 1200 | 1.29 | 0.848 |
| 600 | 1200 | **1.35** | 0.848 |
| 250 | 1200 | 1.23 | 0.848 |
| 250 | 400 | 1.19 | 0.849 |
| 120 | 300 | 1.06 | 0.848 |

A faster baseline makes it WORSE: punishment lasts 300 ms, so a 120 ms baseline tracks
the punishment and subtracts it away. And the tag cosine does not move at all, because
0.848 is a property of the Kenyon code, not of the trace time constant. 600 ms is a
marginal best and not worth chasing.

**What is left is structural, not parametric:**

1. **Compartment-specific dopamine.** `_da_w` is one normalised scalar per MBON over all
   24 PPL1 cells, so every MBON PPL1 touches is depressed together. Real mushroom-body
   learning is compartmentalised - individual PPL1 types innervate individual
   compartments. Those types are in the data and are currently averaged away.
2. **Drive PAM as the opposing channel.** Only PPL1 is ever driven, so everything is
   depression and nothing can move the other way. PAM reaches 47 MBONs, PPL1 80, only
   32 shared - that separation is the thing that lets reward and punishment teach
   opposite lessons, and it is unused.
3. **Per-odour dopamine baseline** as a DIAGNOSTIC only: subtract each odour's own
   unpunished dopamine response. Biologically dubious, but it settles whether the
   residual differential is large enough to be worth chasing at all.

### Trap recorded: how NOT to deliver punishment

`stimulate()` applies tonic current through `_ext`. Setting `drive_hz` on PPL1 does
NOTHING - `drive_hz` only reaches neurons in `SENSORY_CLASSES`, and DANs are not
sensory. A first version of `teaching_signal.py` did that and measured 829 vs 828 PPL1
spikes, appearing to prove punishment had no effect at all.

## FIXED: vision now works (`flysim.enable_vision()` / `see()`)

The optic lobe was dead because photoreceptor output is inhibitory and the network
rests at 0 Hz - an inhibitory input with nothing to inhibit. Both halves of the real
biology were missing:

1. Fly photoreceptors depolarise in the DARK and release histamine tonically; LIGHT
   REDUCES release and disinhibits the lamina. So they fire at a tonic baseline
   (`photoreceptor_hz`, 90 Hz) and `see(brightness)` drives them DOWN. The sign of the
   whole channel inverts.
2. The cells being inhibited need their own depolarising drive. Each of the 14,311
   photoreceptor targets gets a tonic current sized so that, against the inhibition it
   receives at baseline, it rests at `lamina_hold_frac` (0.72) of threshold.

The hold is exact, not tuned: g_ss = W*r*tau_syn and v settles at g + ext, so
ext = v_hold - g_ss.

**Measured, 300 ms per condition:**

| stimulus | visual | VPN | optic lobe | descending |
|---|---|---|---|---|
| before, 400 Hz drive | 224,794 | **0** | 0 | **0** |
| dark (baseline) | 86,937 | 1,860 | 113,242 | 0 |
| half-lit | 48,774 | 3,116 | 92,457 | 20 |
| full light | 0 | **4,540** | 66,216 | **777** |

Light now drives the brain monotonically and reaches the descending bus. 58% of the
connectome went from contributing zero spikes to being the largest live structure in
the model.

## FIXED: escape habituates (`flysim.enable_std()`)

The model had NO adaptation of any kind, so repeated stimulation could only accumulate.
Tsodyks-Markram short-term depression added: one resource variable per presynaptic
neuron, a spike consumes `std_u` of what is left, recovery `std_tau_rec_ms`.

**STD MUST BE GLOBAL.** Applied to sensory afferents only it made habituation WORSE -
all-DN response went from 1.52x to **3.40x** over 20 pulses. It lowers the first pulse
without touching the recurrent central loops, and the accumulation is central.

`std_u` was measured, not chosen (all-DN, 20 pulses, first-pulse spikes | last/first |
novel/first):

| U | first | last/first | novel/first | |
|---|---|---|---|---|
| none | 4913 | 2.267 | 2.251 | sensitises, non-specific |
| 0.03 | 2345 | 1.383 | 0.626 | still rising |
| **0.08** | **1139** | **0.376** | **1.023** | habituates, novel fully recovers |
| 0.15 | 756 | 0.219 | 0.854 | stronger, costs responsiveness |

0.08 is the cheapest setting where a novel stimulus returns to naive, which is what
makes the decrement habituation rather than fatigue.

**Confirmed, 8 seeds, U=0.08:**

| population | first | last/first | novel/first | retest | |
|---|---|---|---|---|---|
| all DN | 1,248 | **0.426** | 0.742 | 0.578 | SPECIFIC |
| DNp escape | 132 | **0.425** | 0.724 | 0.509 | SPECIFIC |

Decrement, stimulus specificity and partial dishabituation. Previously 1.517 and
non-specific. This is the first learning-like behaviour the model has produced.

## TRAP: `stimulate()` cannot fire a DAN during an odour

`stimulate()` converts hz to tonic current as `14.0 * min(1, hz/200)`, so **12.6 mV is
the strongest punishment it can ever deliver.** Measured on PPL105 over 800 ms:

| tonic drive | spikes, no odour | spikes, with CS+ |
|---|---|---|
| **12.6 mV (stimulate's ceiling)** | 83 | **0** |
| 20 mV | 132 | 2 |
| 45 mV | 228 | 91 |
| 70 mV | 266 | 218 |

Odour presentation drives enough inhibition onto the PPL1 cells to silence them
completely at the maximum stimulate() can deliver - which is exactly the moment
punishment is meant to arrive. **Every conditioning protocol built on `stimulate()`
has been pairing the odour with no dopamine at all.** `stimulate_type(..., mv=)` takes
an explicit drive and bypasses the ceiling.

The symptom that exposed it: two arms punishing OPPOSITE odours returned bit-identical
deltas (-0.0073, sd 0.0879, p=1.0000). Identical results from different conditions
means the condition was never applied.

## Individual limb and wing control (`motormap.py`)

The gait in `fly3d.py` and `desktop_fly.py` is a sine wave with fixed tripod phases -
animation. The real motor pool was never read, and it is right there:

`vnc_motor` holds **699 motor neurons**, labelled BY MUSCLE, with `somaNeuromere`
giving the segment and `exitNerve` confirming it:

    T1  173 MNs  prothoracic   front legs  (ProLN)
    T2  175 MNs  mesothoracic  mid legs + wings (MesoLN)
    T3  152 MNs  metathoracic  hind legs   (MetaLN)
    A1-A10       abdominal, not legs

Crossed with soma side that is **six legs, four joints each**, as named antagonist
pairs - and they line up one-for-one with the DesktopFly leg chain:

| joint | pulls one way | pulls the other |
|---|---|---|
| coxa (swing) | Sternal anterior rotator, Tergopleural/Pleural promotor | Sternal posterior rotator, Pleural remotor/abductor |
| trochanter (lift) | Tr flexor, Acc. tr flexor | Tr extensor, Sternotrochanter, Tergotr. |
| femur-tibia | Ti flexor, Acc. ti flexor | Ti extensor |
| tarsus | ltm, ltm1-tibia, ltm2-femur | Ta depressor |

Wings, per side: **5 downstroke (DLMn), 7 upstroke (DVMn), 16 steering** (b1-b3, hg,
iii, ps1, tp, MNwm).

A joint is read as the BALANCE of its antagonists, `(flex-ext)/(flex+ext)`, bounded in
[-1,1] - a difference normalised by the total, so a global gain change cannot
masquerade as a command. Same discipline as the DNa steering readout.

**Limitation, measured.** Driving mechanosensory at 180 Hz for 400 ms produced only
**781 motor spikes across 699 neurons** (~2.8 Hz each), and every joint balance
saturated at +-1.000 because one antagonist fired and the other did not. The map is
correct; the signal is too sparse to read per tick. Same problem as DNa02 and the Giant
Fiber, and it needs the same answer - integrate over a longer window, or drive harder.
Do not wire this to the legs and claim the connectome is walking until that is fixed.

## Endless running and respawn

- `fly3d.py --seconds 0` runs until Ctrl+C and writes what it has.
- `fly3d.py --respawn-every N` puts the body back on the floor every N seconds. A
  respawn resets the BODY, not the brain - traces, weights and adaptation survive,
  which is the point of having it.
- `web/flyroom.html` loops the recorded episode seamlessly with a lap counter and a
  running clock, plus Respawn and a Loop toggle. The episode is recorded, so an endless
  run is a loop, not new simulation - the page says so.

## Conditioning after both fixes (`conditioning3.py`)

Compartment-specific dopamine (PPL105 only, 25 MBONs read out), punishment at 70 mV,
and a LIVE mushroom body (kc_thresh 1.5, apl 200, noise 0.15). 8 paired seeds, 12
trials, learn_rate 0.02.

| arm | delta | weights after training |
|---|---|---|
| punish CS+ | **-0.1011** (sd 0.084) | 82.5% |
| punish CS- | **-0.0169** (sd 0.131) | 83.1% |
| no plasticity | +0.0000 (sd 0.0000) | 100.0% |

    vs control:  d' = -1.70   p = 0.0084     (was d' = -0.59, p = 0.4)
    reversal:    p = 0.1719   sign did NOT flip

**Real progress, still not associative.** The two arms are no longer identical - they
differ 6-fold - and the trained arm is now significantly different from control where
before it was not. But punishing CS- gives -0.017, near zero rather than positive, so
the shift does not reverse with the contingency.

**Why, and it follows from the earlier measurements.** The eligibility tags for the two
odours have cosine 0.846, so depressing "CS- preferring" cells hits CS+ cells too and
both responses fall together. probe_ladder's rung-3 optimum reached +0.217 by depressing
cells *where CS+ exceeds CS-* - a COMPARISON. A depression-only rule driven by
`trace x dopamine` never computes a difference; it only ever subtracts. With overlapping
tags and no opposing channel, the arms cannot separate in sign no matter how clean the
teaching signal is.

**The n=8 reversal test is also underpowered** - effect 0.084 against pooled sd ~0.11 is
d=0.76, roughly 25% power. Do not read p=0.17 as evidence of no difference.

### Next, and it is now specific

An opposing channel is needed so the unpunished odour teaches something too. The obvious
move - drive PAM on the unpunished odour and read the same MBONs - does NOT work here,
and the anatomy says why: **PAM10 shares only 4 of PPL105's 25 MBONs** (PAM08, PAM14,
PAM01 share none). Reward and punishment deliberately teach different compartments.

So the readout has to become differential across compartments, the way the fly actually
computes valence: punish CS+ through PPL105, reward CS- through PAM10, and read the
BALANCE between the two compartments rather than either alone.

## Why an embodied fly walks in circles — measured

The desktop and 3D builds both circled on first run. Two causes, both measured on the
DNa family, both now fixed in `desktop_fly.py` and `fly3d.py`.

**1. Receptors split by array index are not a left/right split.** Sensory somas sit
outside the imaged volume so `somaSide` is "M"; laterality is in `rootSide`, which is
what `side` carries. Splitting `pop["mechano"]` down the middle of the array gives
sets that are 68% / 71% impure. Split on `side` (828 L / 877 R).

**2. Structural bias is 16x the actual side signal.** `turn_raw = (R-L)/(R+L)*3` on
the DNa family:

| stimulus | turn_raw |
|---|---|
| symmetric | **−1.304** |
| left antenna only | −1.263 |
| right antenna only | −1.185 |

The bias is −1.3 and constant; the whole left-vs-right signal is **0.078**. The command
clips to −1.0 every tick and the fly turns at maximum rate forever. This is the same
trap decision 9 records for the hemispheres, arriving at the readout instead of the
weights.

**The fix is flyworld.py's `calibrate()`,** ported: measure the R-share under a
SYMMETRIC stimulus at several drive levels and interpolate. Whatever a symmetric
stimulus produces carries no information about the world, at any level, and that is the
zero point. After calibration, measured: **evidence is exactly 0.000 with no stimulus**,
and a disturbance on the left vs the right separates by ~0.48.

**Calibration alone is not enough, and this is the part worth remembering.** Even
perfectly zeroed, integrating a continuous turn signal produces circles, because a fly
does not steer continuously. It holds a course and turns in discrete body saccades —
about one a second walking, far faster in flight — and it stops and grooms constantly.
So the calibrated asymmetry is treated as **evidence** that biases a saccade generator,
with a walk / stop / groom / fly state machine over it.

**That generator and state machine are IMPOSED, not derived.** The connectome contains
the circuitry; these scripts do not read it out. Say so when showing the output. What
is genuinely the connectome's: turn evidence (calibrated DNa asymmetry), walking drive
(total descending rate), and escape takeoff (DNp01, the Giant Fiber, doing its real
job). Wingbeat, banking and leg placement are animation.

Resulting time budget over 60 s, seed 11: walk 41%, flight 29%, land 14%, groom 10%,
stop 6%, takeoff 1%.

## Visual report and desktop build

- `web/flybench.html` — published artifact, all four experiments plus the standing
  results, with the recorded episode replayed on a 3D body.
- `scripts/desktop_fly.py` — Windows desktop fly (Tk, keyed-out background). Brain on
  a worker thread at ~2.2 commands/s, UI at 60 fps, cursor drives mechanosensation.
- `scripts/fly3d.py` — a room (64 x 44 x 30) with five landable surfaces; walking,
  takeoff, flight and landing. Writes a frame-by-frame JSONL for the viewer.
- `web/flyroom.html` — published viewer for that episode, chase and orbit cameras.
- Body geometry in both is ported from DesktopFly-Linux (MIT, (c) 2026 Denis Shiryaev
  and contributors): segment dimensions, the six-leg table, tripod gait phases.
  Independent confirmation of our DN choices — that project maps DNa02/DNg13 to
  steering and DNp01/DNp10 to jump, and Fly64 (same MaleCNS data, drives Mario 64)
  maps the same cells.

## Files

- `build_creature.py` — raw feather → `.npz` (`--whole` for all 162,517)
- `mirror_connectome.py` — bilateral symmetry as a second measurement
- `flysim.py` — the LIF engine
- `flyworld.py` — survival sandbox
- `steering_protocol.py` — Shiu-style repeated-trial protocol
- `conditioning.py` — olfactory conditioning: CS+ paired with PPL1, CS- unpaired
- `conditioning2.py` / `conditioning3.py` — paired-seed reversal design; compartment-specific dopamine
- `conditioning4.py` — differential readout across compartments (PPL105 vs PAM08); `--swap`, `--odours`
- `odour_design.py` — per-glomerulus KC footprints → drive-matched, non-overlapping odour pair
- `kc_sparsity.py` — kc_thresh × apl_scale sweep: sparsity, overlap vs chance, single-glomerulus probe
- `al_selectivity.py` — glomerular identity at the PN level under LN sign variants (decision 13)

## Mirroring

A real fly is symmetric; this specimen's reconstruction is not. Every connection
reduces to (pre type, post type, crossing?), which a symmetric animal shows twice —
once from each side. Taking the better-resolved of the two brought R/L from **1.0968
to 1.0001** and recovered 4.7M connections. Of 1,066,731 canonical connections,
**511,340 were seen on only one side**; where both were seen they disagree by 33.6%
on average, which is a direct read on tracing noise.

Kenyon cells are deliberately excluded — their wiring is random per animal, so the
left mushroom body is not a measurement of the right one.

## Conditioning read out as a difference between compartments (`conditioning4.py`)

conditioning3's shift did not reverse with the contingency, and the reason was already
measured: a depression-only rule with overlapping tags only ever subtracts. The fly does
not compute valence inside one compartment. Punishment (PPL1) depresses Kenyon input to
APPROACH-driving MBONs, reward (PAM) depresses Kenyon input to AVOIDANCE-driving MBONs,
and behaviour is the balance. So the readout became `D = d_A - d_P`: the discrimination
index over the PPL105 compartment minus the same over the PAM08 compartment.

PAM08 chosen from the anatomy: the largest PAM type (50 cells -> 20 MBONs), **zero**
MBONs shared with PPL105, and its targets are MBON01/04/05/09/21/27/29 - the
gamma4/gamma5/beta'2 avoidance family - while PPL105's are the gamma1pedc/alpha approach
family. That is the real valence axis. Every arm drives one DAN type on each odour, so
the amount of dopamine is identical across arms.

8 paired seeds, 12 trials, 70 mV DAN drive, alphabetical odours (types [0:5] vs [5:10]):

| arm | dD | d_A | d_P |
|---|---|---|---|
| punish CS+ / reward CS- | **-0.665** (sd 0.127) | -0.024 | +0.640 |
| punish CS- / reward CS+ | **-0.218** (sd 0.202) | +0.091 | +0.309 |
| no plasticity | +0.000 | 0 | 0 |
| punish only | -0.215 | -0.101 | +0.114 |
| reward only | -0.527 | +0.096 | +0.623 |

    reversal:  d' = -2.65   MWU p = 0.0002   paired Wilcoxon p = 0.0078
    sign flipped: NO

Three things. The arms are now strongly separable (conditioning3: d'=0.76, p=0.17).
punish-only reproduces conditioning3's d_A to four decimals (-0.1011), so the pipeline is
consistent. And the contingency-following component is about **+-0.22**, riding on a
**-0.44 shift that ignores the contingency** - d_P rises in every arm, including
punish-only where PAM08 is never driven.

**The -0.44 is odour identity, measured two ways.** Swapping which glomeruli are CS+
mirrors the result exactly (+0.185 / +0.684); swapping presentation order changes
nothing (-0.684 / -0.185). Cause: `ORN_DA1` has 204 receptors - the cVA pheromone
glomerulus, enormous in a male - so odour X drove 43% more Kenyon activity and **744 of
Y's 842 cells were also X cells (88%)**. Y was nearly a subset of X. Depression paired
with EITHER odour removed more of Y's response than X's.

By the field's own standard (reciprocal odour groups averaged, Tully & Quinn 1985) this
is a learning index of ~0.22 at p=0.0002. By this project's stricter standard - the
sign must flip - it is not, and the reason it cannot is the next section.

## The Kenyon code was never sparse, and the antennal lobe was broadcasting

Designing a drive-matched odour pair (`odour_design.py`) exposed the real problem.
**Every single glomerulus, alone, fired 600-790 Kenyon cells (15-19% of 4,064)** -
more cells than a glomerulus is even wired to (median 5 of 53 glomeruli per KC, so
~10%). Two designed, disjoint 5-glomerulus odours still shared 852 of ~1,140 cells
(Jaccard 0.60; independent 27% sets give 0.16). The code was a fixed ~700-cell core
plus a fringe, whatever the odour. Three causes, in the order they were found:

**1. The APL surrogate never engaged - a bug.** It compared the fraction of KCs spiking
PER MILLISECOND (0.08-0.8%) against `kc_target_sparsity=0.05`, a per-odour number, and
inhibited only above it. Results were bit-identical at apl 200 / 600 / 1500 in every
regime. Every `apl` argument in every conditioning script has been a no-op, and every
sparsity result before today was threshold-only. Threshold alone gets sparsity but not
selectivity: at 5.6% active, Jaccard was still 0.378 against 0.026 for chance, and one
glomerulus fired 113 of an odour's 228 cells.

**2. KC->KC recurrence is NOT the cause - refuted.** KC->KC excitatory weight is 55% of
the PN input to KCs and 24% of KCs get more excitation from other KCs than from PNs, so
it was the obvious suspect. Ablating it (`kc_kc_scale=0`) changed nothing. The knob
stays at 1.0. (The threshold normalisation of decision 11 was also dividing by this
weight; it now uses PN input only.)

**3. The antennal lobe local neurons lLN1 and lLN2 are sign-flipped - decision 13.**
Stimulating ONE receptor type (ORN_DM6) drove PNs in **45 of 53 glomeruli** above
5 Hz, twelve at saturation; only **7%** of uniglomerular PN spikes were in DM6. 96% of
the excitatory drive onto an unstimulated glomerulus's PNs came from `ALLN`, lLN1_bc
alone 45%. The transmitter table calls lLN1 acetylcholine (42/59) and splits lLN2 44
GABA / 40 acetylcholine. A cell type is transmitter-homogeneous; a 50/50 split inside
one is the classifier not knowing. Both are reported GABAergic panglomerular LNs - the
lateral inhibition that gives the antennal lobe its gain control. With 151 cells
re-signed (`Params.sign_override`, now the default):

| | glomeruli >5 Hz | own share | KC cells shared |
|---|---|---|---|
| as-is | 45-47 / 53 | 0.07-0.25 | 138 (Jaccard 0.384, 12x chance) |
| **lLN1+lLN2 inhibitory** | **1 / 53 single, 6 / 53 odour** | **0.99-1.00** | **0** |

Same family as the glutamate and histamine traps: one transmitter label, one layer
further in. This is the root of the 0.846 tag cosine, the 41-72% overlaps, and every
non-specific conditioning result.

**4. The real APL is in the connectome and was 7x too strong - decision 12 rewritten.**
Decision 2 said APL did not survive the weight>=5 threshold. For this build it does
(type `APL`, GABA), driven by PNs and KCs, and with the antennal lobe fixed it was
delivering **-7.2M onto the KCs against +1.6M of PN excitation** - broad odours
suppressed themselves (8 glomeruli fired FEWER cells than 1). It is non-spiking and
graded in life; the LIF saturates it at 250 Hz. `apl_scale` scales its output; the
surrogate (`apl_w`) is off.

**The regime, measured (`kc_sparsity.py`, 8-glomerulus designed pair):**

| kc_thresh | apl_scale | % KC | Jaccard | chance | 1-glomerulus cells |
|---|---|---|---|---|---|
| 1.0 | 0.10 | 14.0 | 0.055 | 0.075 | 123 |
| **1.5** | **0.10** | **4.9** | **0.015** | **0.025** | **39 (odour: 188)** |
| 1.5 | 0.20 | 3.0 | 0.013 | 0.015 | 30 |
| 2.0 | 0.10 | 1.6 | 0.008 | 0.008 | 15 |

5% active, the two odours BELOW chance overlap (actively decorrelated), drive balanced
to 1%, and a lone glomerulus fires a fifth of what the odour does - coincidence
detection. These are now the engine defaults.

**Everything olfactory measured before 10 Sept 2026 was measured on the broadcasting
antennal lobe.** Steering (smell_bilateral), the probe ladder, teaching_signal and all
conditioning runs. The steering d'=4.21 and the ceiling arguments still hold as
measurements of that network; they need re-taking on this one.

## ASSOCIATIVE — conditioning on the fixed circuit (`conditioning4.py`, `results/conditioning4b*.json`)

Same protocol, same readout, same learning rule as this morning's run. The only
changes are upstream: the designed 8-glomerulus odour pair (`results/odours3.json`),
lLN1/lLN2 inhibitory, the real APL at `apl_scale 0.1`, kc_thresh 1.5. 8 paired seeds,
12 trials, 70 mV DAN drive, learn_rate 0.02.

| arm | dD | d_A | d_P |
|---|---|---|---|
| punish CS+ / reward CS- | **-0.783** (sd 0.333) | +0.174 | +0.957 |
| punish CS- / reward CS+ | **+0.843** (sd 0.149) | +0.074 | -0.768 |
| no plasticity | +0.000 (sd 0.000) | 0 | 0 |
| punish only | -0.088 (sd 0.264) | -0.024 | +0.064 |
| reward only | -0.465 (sd 0.336) | +0.491 | +0.956 |

    reversal:  d' = -6.29   MWU p = 0.0002   paired Wilcoxon p = 0.0078
    sign flipped: YES   8 of 8 seeds opposite in the two arms

Counterbalanced with the odours swapped: both **-0.847**, reversed **+0.687**,
d' = -7.15, again 8/8. The odour-identity bias is now +0.03 / -0.08 (was -0.44). The
counterbalanced learning index is **~0.79**; a real fly scores 0.3-0.8 after one
session.

Every arm receives the same dopamine - one DAN type per odour - so "dopamine depresses
everything" is controlled for, and the plasticity-off arm is exactly 0.0000, so nothing
but the weights moved. The shift follows the contingency and reverses with it.

What changed between "specific but not reversed" this morning and this was NOT the
learning rule, the compartments or the readout - all three were already in place.
It was the Kenyon code: two odours that shared 55% of their cells now share none.
probe_ladder's rung-3 ceiling argument was right that depression is sufficient; what
it could not see was that the tags it was given were 85% the same tag.

**The reward channel carries most of it.** reward-only through PAM08 (gamma4/5, beta'2)
moves D by -0.465; punish-only through PPL105 (gamma1pedc) by -0.088 - and d_P reaches
+0.96, meaning the avoidance MBONs' response to the rewarded odour is nearly abolished
at the 20% weight floor. Per-MBON dopamine is matched across the two (per-cell
normalisation in `learn()`, 2 cells vs 50 at the same 173 Hz), so the asymmetry is in
the compartments, not the teaching signal. Naive d_A is -0.38: CS- drives the PPL105
MBONs twice as hard as CS+ at rest, so there is less CS+ response there to depress.
Not yet diagnosed beyond that.

Cross-compartment effects are real and large: reward-only, which never touches PPL105,
moves d_A by +0.49. The MBONs feed back on one another (gamma1pedc>alpha/beta is the
canonical feedforward-inhibition MBON); the two compartments are not independent
readouts, and D should be read as the network's valence, not a sum of two parts.

### Next

1. **Close the loop.** Train a brain, then read the DNa steering asymmetry to CS+ vs
   CS- (`smell_bilateral`). If avoidance of the punished odour shows up on the action
   bus, that is a learned behaviour derived from the connectome end to end - and it is
   what `flyworld.py` needs to stop starving.
2. **Re-take the olfactory baselines on the fixed antennal lobe**: steering d',
   probe_ladder, teaching_signal. All were measured on a PN code with no glomerular
   identity.
3. Why is PPL105 punishment 5x weaker than PAM08 reward? Decompose the PPL105 MBONs'
   odour drive by presynaptic source; the compartment may be dominated by non-KC input.

## Qualification: the "compartments" are DAN target sets, and the cells that fire are strays

Investigating why PPL105 punishment is 5x weaker than PAM08 reward found something
that bounds the associative result above. `enable_compartments()` defines a DAN
type's compartment as every MBON it makes a direct synapse onto, with dopamine weight
normalised to its strongest target. That correctly finds the principal MBON -
PAM08 -> MBON05 (2,421 synapses, gamma4), PPL105 -> MBON13/MBON18 (680/510) - but the
readout then includes every stray target too, and **the strays are the cells that
fire**:

| compartment | principal MBONs (w >= 0.2) | spikes / 800 ms | stray carrying the readout | its w | spikes |
|---|---|---|---|---|---|
| PPL105 | MBON13, MBON18, MBON23 | **0-4** | MBON11 (12 synapses) | 0.018 | 34 |
| PAM08 | MBON05, MBON21 | **0-4** | MBON09 (115 synapses) | 0.026 | 40-57 |

So the differential readout that reversed was MBON11 (gamma1pedc) against MBON09
(gamma3beta'1), each receiving ~2% of its DAN type's peak dopamine. At learn_rate 0.02
compounding over 800 steps x 12 trials, 2% is still enough to drive their KC synapses
to the floor - which also says the learning rate is far too high for dopamine weight
to mean anything. The result stands as a phenomenon: two opposing DAN channels onto
two different MBON sets give a readout that follows the contingency and reverses.
It does NOT show compartment-specific learning in the anatomical sense; the labels
PPL105 / PAM08 should be read as "DAN target set", not "gamma1pedc compartment".

**The principal MBONs are silent for four different measured reasons** (CS+, seed
1000, apl_scale 0.1; E and I are spikes x weight over 800 ms):

| MBON | mean v / thr | E | I | what silences it |
|---|---|---|---|---|
| MBON13 (alpha'2) | 2.9-5.0 / 7.0 | 658-1291 | -85 | nothing - just under threshold, KC drive too weak |
| MBON18 (alpha2sc) | -7.0 / 7.0 | 943 | **-7164** | LHCENT1/2/3/9, GABA, driven straight from PNs (9,174 from ALPN) |
| MBON05 (gamma4>gamma1gamma2) | -3.9 to +0.8 / 7.0 | 2908-3749 | -3597 | **APL -3250** even at apl_scale 0.1; MBON09, MBON11 |
| MBON21 (gamma4gamma5) | -6.9 / 7.0 | 642-866 | -5769 | **MBON09 -5517** (glutamate, inhibitory) |

Every one of these has baseline firing and a robust odour response in the animal.
This is the optic-lobe problem in the output layer: the network rests at 0 Hz by
design, so a cell whose real inhibition is balanced by tonic excitation the model
does not have is simply off, and the winner-take-all among MBONs is decided by
whichever cell happens to be net-excited (MBON09 at E/I 2.7-5.6, MBON11-R at 2.4).
None of this is a sign error - LHCENT are GABAergic as labelled and MBON09 really is
glutamatergic - it is the missing tonic drive.

## Learned steering — NOT DEMONSTRATED at the action bus (`learned_steering.py`)

Train exactly as conditioning4, then read the DNa family under a lateralised odour
(near antenna 1.0, far 0.4): approach index `A(o) = T(o right) - T(o left)`, valence
`V = A(CS+) - A(CS-)`, paired seeds pre/post. Structural left/right bias cancels twice.
8 seeds, 12 trials, 423 DNa spikes per presentation.

| arm | DNa dV | MBON dD (same run) |
|---|---|---|
| punish CS+ / reward CS- | -0.039 (sd 0.159) | -0.767 |
| punish CS- / reward CS+ | -0.049 (sd 0.179) | +0.871 |
| no plasticity | +0.000 | +0.000 |

    reversal: MWU p = 0.96   d' = 0.06   opposite-sign seeds 4/8

The synapses learned in both directions; the steering bus did not move. Both arms show
the same small negative shift - a non-specific effect of depressing ~4% of KC->MBON
weight - and nothing that follows the contingency. Given the section above this is
the expected outcome: the learned change lives in MBON11 and MBON09, while the MBONs
with the real downstream footprint are silent and take no part.

### Next, in order

1. **Give the MBONs their tonic drive.** The output-layer analogue of
   `lamina_hold_frac`: a per-MBON tonic current sized so each cell rests at a measured
   fraction of threshold against its baseline inhibition. Measure principal-MBON odour
   responses before and after; the target is MBON05/13/18/21 responding, not just the
   strays.
2. **Restrict compartments to core members** (dopamine weight >= 0.2 of the type's
   peak) for both teaching and readout.
3. **Bring `learn_rate` down** until 2% dopamine weight no longer saturates; then
   compartment specificity can be tested rather than assumed.
4. Re-run conditioning4, then learned_steering. If the DNa readout still does not
   follow the contingency with the principal MBONs live, the next suspect is the
   MBON -> DN pathway itself, which has never been characterised here.

## Cross-reference against the literature (10 Sept 2026)

Do this periodically. Verdicts on what is new, checked against published work.

**NEW - the lLN1/lLN2 transmitter labels fail a dynamical test.** Eckstein et al. 2024
(Cell, the transmitter classifier) already flagged the population: the ALl1/ALv2
hemilineages "seem to break Dale's law and Lacin's law, with similar morphology types
predicted to express different transmitters", and "18-27% of antennal lobe local
neurons may be cholinergic, suggesting that lateral excitation is a more prominent
feature of antennal lobe processing than previously thought" - with no functional
validation. MaleCNS v1.0's paper was published 3 Sept 2026 with the same classifier
family (funkelab/synister_malecns; no accuracy figures published for it). Nobody has
published the consequence measured here: taken at face value, one glomerulus drives 45
of 53 to saturation (own share 0.07), which contradicts glomerulus-specific PN
responses and inhibition-dominated lateral interaction (Olsen & Wilson 2008; Bhandawat
et al. 2007 - CHECK these citations before quoting). Re-signing 151 cells restores
glomerular identity (own share 1.00). Worth a note to the FlyEM team now, and a short
methods note.

**NEW BUT NOT YET CLAIMABLE - associative conditioning in a whole-CNS LIF with the
measured DAN->MBON wiring.** Shiu et al. 2024 do not simulate olfaction, the mushroom
body or plasticity at all. Published MB learning models are mushroom-body-only
(Bennett et al. 2021; Springer & Nawrot 2021; Huang et al. 2024 Nature,
connectome-constrained MB model with voltage imaging). One FlyWire hobby repository
(lixiang1076/fly-brain) has KC->MBON dopamine learning with no controls, no reversal
test and no sparsity handling. Ours passes the reversal test with paired-seed controls
- but through stray MBONs (section above). Claimable once the principal MBONs are live
and it still reverses.

**INSTANCES OF A KNOWN LIMITATION - methods notes.** Shiu et al. state that "circuits
in which there is extensive basal inhibition, not captured by the model because of the
zero basal firing rate, may be poorly simulated." The photoreceptor-histamine result
(optic lobe delivers zero) and the silent principal MBONs are concrete instances; the
`lamina_hold_frac` construction is a practical fix others could reuse. The real APL
being 7x too strong as a spiking LIF cell (it is non-spiking in life) is a modelling
caveat not found elsewhere. KC->KC ablation having no effect is a small negative
result. DN->DN axo-axonic statistics matching Ceballos et al. 2026 is a replication.

Sources: Eckstein et al. 2024 https://pmc.ncbi.nlm.nih.gov/articles/PMC11106717/ ;
Shiu et al. 2024 https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/ ;
Schlegel et al. 2021 https://elifesciences.org/articles/66018 ;
MaleCNS https://male-cns.janelia.org/ ; https://github.com/funkelab/synister_malecns ;
Huang et al. 2024 https://www.nature.com/articles/s41586-024-07819-w ;
Li et al. 2020 MB connectome https://elifesciences.org/articles/62576

## FIXED: the output layer has tonic drive (`mbon_hold_frac`, decision 14; `mbon_hold.py`)

The principal MBONs were silent because the 0 Hz-rest model gives them no baseline
excitation to set real inhibition against. `mbon_hold_frac` x threshold of tonic
current on every MBON, the output-layer analogue of `lamina_hold_frac`. Swept, one
seed, 800 ms per odour:

| hold | resting MBON rate | MBON types active to odour | MBON13 | MBON05 | MBON18 / 21 | DNa to CS+ / CS- |
|---|---|---|---|---|---|---|
| 0 | 0 Hz | 9 | 4 | 0-4 | 0 | 419 / 151 |
| 0.7 | 0 Hz | 21 | 46 | 11-20 | 0 | 446 / 4 |
| **0.85** | **3.4 Hz, 33 types** | **23** | **50** | **15-25** | 0 | 494 / 8 |
| 0.95 | 7.7 Hz | 27 | 53 | 18-23 | 0 | 371 / 10 |

0.85 is now the default: a resting rate in the biological range (MBONs fire 5-20 Hz
spontaneously), the principal MBONs of both taught compartments responding, and both
core compartments giving a non-degenerate readout. **MBON18 and MBON21 stay silent at
any hold** - 7 mV of tonic current does not beat -6.5k from LHCENT or -5.5k from
MBON09. Recorded, not fixed.

**The hold changes what the odours do at the action bus.** DNa response to CS- fell
from 151 to ~5 spikes while CS+ stayed ~450: MBON output now shapes the descending
response strongly and odour-specifically. The pathway learned steering needs is live
where before the two odours barely differed at the DNs.

**Core compartments and learn rate.** conditioning4 now delivers dopamine only to
members at >= 0.2 of the type's peak (PPL105: MBON13/18/23, 6 cells; PAM08:
MBON05/21, 4 cells) and reads the same set. At `learn_rate 0.02` and even 0.005 the
readout saturates within 4 trials - the core MBONs sit near threshold, so a few
percent of depression silences them outright. The appetitive core saturates at every
rate tried (MBON05: 15-25 spikes per odour, ~10-15 Hz per cell, a biological rate
with no margin). The aversive core is graded at 0.0003 (-0.43 in 12 trials), which is
the new default. Saturation is the readout's dynamic range, not the learning.

## ASSOCIATIVE AND COMPARTMENT-SPECIFIC — conditioning with anatomical cores (`results/conditioning4c*.json`)

Same protocol as the morning's run; the circuit now has decisions 12-14 and dopamine
reaches only core compartment members (PPL105: MBON13/18/23; PAM08: MBON05/21).
8 paired seeds, 12 trials, 70 mV DAN drive, learn_rate 0.0003.

| arm | dD | d_A | d_P | weights (taught compartment) |
|---|---|---|---|---|
| punish CS+ / reward CS- | **-1.493** (sd 0.057) | -0.360 | +1.133 | 98% |
| punish CS- / reward CS+ | **+1.096** (sd 0.064) | +0.230 | -0.867 | 98% |
| no plasticity | +0.000 | 0 | 0 | 100% |
| punish only | -0.347 | **-0.325** | **+0.022** | A 98%, P 100% |
| reward only | -1.145 | **-0.012** | **+1.133** | A 100%, P 98% |

    reversal: 8/8 seeds flip, MWU p = 0.0002, paired Wilcoxon p = 0.0078
    odours swapped: -1.096 / +1.503, 8/8, same p
    decomposition: punish-only + reward-only = -1.492   vs both -1.493

Three things the morning's run could not claim:

1. **Each dopamine channel teaches only its own compartment.** PPL105 moves d_A by
   -0.33 and d_P by +0.02; PAM08 moves d_P by +1.13 and d_A by -0.01. That is the
   anatomical statement (Aso et al. 2014) measured rather than assumed, and it is the
   thing the stray-MBON readout could not show.
2. **The channels add exactly** (-1.492 vs -1.493). The two compartments are
   independent readouts. The earlier super-additivity was the strays talking to each
   other through MBON->MBON inhibition.
3. **Per-seed sd fell from ~0.3 to 0.06.** The core MBONs respond consistently, and
   2% mean depression in the taught compartment is enough to move the readout by more
   than one unit.

Caveat, unchanged: the appetitive core saturates - MBON05's CS- response goes to zero
(d_P at its bound) at any learn rate tried, because it responds at a biological
10-15 Hz with no margin above threshold. The aversive core is graded. The
contingency-independent component (+-0.20, mirrored by the swap) is that asymmetry.

## Learned steering, second attempt — still NOT DEMONSTRATED, and now it is clear why

`learned_steering.py` on the fully fixed circuit (`results/learned_steering2.json`):
both +0.21 (sd 0.61), reversed +0.08 (sd 0.51), p = 0.72; MBON dD -1.75 / +0.70 in the
same run. Worse variance than before: the MBON hold cut the DNa response to CS- to
~5 spikes, so the turn index for one odour is computed from nothing. The lateralised
DNa readout is not measurable for both odours on this circuit.

**Population test (`learned_dn.py`)** - per-DN change in (CS+ - CS-) response with
training, all 1,310 DNs, both arms, paired seeds:

    corr(dS_both, dS_reversed):   MBONs  -0.684     DNs  +0.599
    DNs whose change reverses with the contingency (|z|>3, >1 spike):  3 / 1310

Learning reverses across the whole MBON layer. What reaches the descending neurons
is the SAME in both arms - a non-specific consequence of training - and 3 of 1,310 at
|z|>3 is what chance gives. **The valence sign is lost between the MBONs and the
action bus.**

**Where the core MBONs' output goes (measured):**

| cell | fires on odour | direct -> DN | 2-hop -> DN | targets alive on odour | top targets |
|---|---|---|---|---|---|
| MBON13 L/R | 20-30 | 0% | 0.5% | 32-35% | CRE055, SIP015, LHPV5e1, FB5AB (CX) |
| MBON18 L/R | 0 | 0% | 0.3% | 35% | LHCENT1/9/6, LHPV5e1 |
| MBON23 L/R | 0 | 0% | 0.1% | 23-29% | LHCENT6, PAM10 |
| MBON05 R (L silent) | 15-25 | 0% | 0.9% | 50-52% | MBON30, LHPV7c1, CRE011, MBON11, PAM07 |
| MBON21 L/R | 0 | 0.7% | 3.0% | 27% | FB4R (CX), CRE100, MBON26, LAL159 |

No direct MBON -> DN output at all; 0.1-3% at two hops. The mushroom body reaches
behaviour through the central complex (FB4R, FB5AB), the LAL and the CRE/SIP
convergence neurons - the real anatomy - and **that layer is 50-77% silent** during an
odour. The learned signal is three cells wide (MBON13 L/R, MBON05-R), feeding a
mostly-dark layer, three or more synapses from the action bus.

### Next

The pattern is now unmistakable: photoreceptor targets (fixed), Kenyon cells (fixed),
MBONs (fixed), and now the MB -> CX/LAL convergence layer - every stage past the
first synapse is silent under the 0 Hz-rest convention until it is given the tonic
drive it has in life. Doing this one layer at a time is whack-a-mole.

**Decision 15 candidate: replace the 0 Hz rest with a low-rate spontaneous regime.**
A background tonic drive on every neuron, sized as a fraction of threshold, so
inhibition-dominated cells can be modulated downward and net-inhibited pathways carry
signal. It must be MEASURED before adoption: spontaneous rate distribution against
known values, stability (no seizure), and whether the established results (steering
d', habituation, vision, the conditioning result above) survive. Every earlier number
would then need re-taking. It is a session of its own and the right next one.

Cheaper first test: hold only the CX/LAL/CRE/SIP targets of the core MBONs and see
whether the learned DN signal appears. If it does, the global version is justified.

## Three fixes for the dark convergence layer, all refuted (`bg_hold.py`, `learned_dn.py`)

The MB -> CX/LAL/CRE layer that carries MBON output toward the descending neurons is
50-77% silent during an odour. Three mechanisms were tried; none opens it.

**1. Global tonic drive (`bg_hold_frac`, candidate decision 15) - NOT ADOPTED.**
Tonic current on every central neuron except sensory sources and Kenyon cells. It is
stable and plausible at rest (0.5-0.7: 1.3-1.9 Hz/neuron brain-wide, MBONs 12-13 Hz,
DNs 7-10 Hz, no ramping over 1.6 s, KC sparsity and overlap intact) - but the core
MBONs' targets go from 20% alive to only 24-29%, and the learned signal at the DNs
gets WORSE:

| bg_hold | corr(dS_both, dS_reversed) MBONs | DNs |
|---|---|---|
| 0 | -0.684 | +0.599 |
| 0.5 | -0.531 | +0.655 |
| 0.7 | +0.334 | +0.981 |

At 0.7 the MBON layer itself loses its reversal: spontaneous activity swamps the
learned change. Default stays 0.0. The parameter is kept for measurement.

**2. Short-term depression, global - INCOMPATIBLE with sustained odour.** With
`enable_std()` as parameterised for habituation (U 0.08, tau 480 ms), the Kenyon code
collapses to 1 active cell: at 200 Hz receptor drive the olfactory afferents settle at
x = 1/(1 + U r tau) = 11% of their strength before the test window opens. This is why
every conditioning script has STD off, and it means the habituation and olfactory
results are currently measured on different engines.

**3. Short-term depression, central only (sensory, ALPN, KC excluded) - refuted.**
Targets alive 16-17%; DNs differing between odours fall from 254 to 46. Incidentally
it frees the appetitive core (MBON09 -> MBON05/21 inhibition depresses; P core 15 ->
162 spikes) - a hint that MBON09's dominance is part of the problem - but the
convergence layer stays dark.

**Why they all fail, measured.** Every top target of MBON13 and MBON05 is net-inhibited
during the odour by 1.5-5x (LHPV5e1: E 7,790 / I -11,655 from CRE050, LHCENT2, AstA1;
LHCENT4: E 6,468 / I -32,478; LHPV10d1: I -11,349 from mALB3; CRE077: I -7,536 from
oviIN). The odour drives an inhibitory wave through the whole convergence zone, and a
tonic hold of a few mV cannot beat a net -5 mV/ms. This is not a dark cell to fix;
**it is the E/I balance of the central brain under a uniform 0.275 mV per synapse and
transmitter-label signs** - the regime Shiu et al. said the model would simulate
poorly, and the fourth stage in a row to hit it.

### Next - the structural candidates, in order of how much they would change

1. **A sublinear synapse-count -> efficacy mapping.** A 371-synapse connection
   delivers 102 mV of conductance per spike here; real efficacy saturates with synapse
   count. This would shrink the massive inhibitory connections (CRE050 -5,324, LHCENT9
   -7,464) more than the many small excitatory ones. Needs the whole ladder re-run.
2. **Glutamate sign outside the antennal lobe.** GluCl-alpha inhibition is
   established in the AL; the central brain also has excitatory glutamate receptors.
   Shiu et al. tested both globally. LHCENT4, MBON09, MBON05, MBON30 are all
   glutamatergic and all sit on this pathway.
3. **MBON09's dominance.** It fires 40-57 spikes to any odour and inhibits MBON05,
   MBON21, MBON30 and MBON11 by thousands. Real MBON-gamma3beta'1 is not a
   winner-take-all hub. Whether that is (1), (2) or the transmitter label is testable.

The synaptic conditioning result stands. Reaching behaviour is blocked at a
well-characterised place.

## Fourth fix refuted: synapse-count saturation (`syn_sat_k`) — and glutamate-sign flip ruled out without running it

**Global excitatory glutamate is not a live candidate.** Shiu et al. 2024 already ran
this experiment brainwide: switching their default (inhibitory) to excitatory
"eliminates" a confirmed result (bitter/Ir94e sign) and raises the optogenetic
false-positive rate from 1% to 16%. They state plainly: "it is not feasible to predict
whether a particular glutamatergic connection is excitatory or inhibitory at a
brainwide scale." A blanket flip is worse by their own measurement; a targeted,
cell-type-specific flip (the lLN1/lLN2 precedent) would need per-type evidence of
excitatory glutamate receptor expression that does not exist for MBON05/MBON30/LHCENT4
specifically - central-brain iGluRs are documented (AMPA/NMDA-like, Frontiers 2020;
mGluRs in KCs), but that motivates neither sign for these particular cells. Not
attempted.

**Synapse-count saturation, tested and refuted.** `syn_sat_k`: efficacy =
synapse_count * k/(synapse_count + k), motivated by the top offenders in the dark
layer being 100-300 synapse individual connections (whole-connectome 90th percentile
is 22, 99th is 81) delivering up to ~82 mV/spike undamped. Swept k = 30, 60, 120, 300,
600 against the standard diagnostics (bg_hold=0 baseline: KC active 190-226, MBON
targets alive 16-20%, 254 DNs differing >5 spikes between odours):

| k | KC active (CS+) | MBON-target alive | DNs differing |
|---|---|---|---|
| none (linear) | 190 | 20% | 254 |
| 600 | 153 | 14% | 213 |
| 300 | 123 | 11% | 190 |
| 120 | 62 | 6% | 33 |
| 60 | 14 | 5% | 11 |
| 30 | 2 | 4% | 1 |

**Monotonically worse at every k tried, never better.** The compression is not
selective for the pathological large inhibitory connections; it hits the strong
excitatory connections the sparse code and the whole network's drive depend on just
as hard, and the network quiets globally rather than rebalancing. Rest DN rate drops
to 0 Hz at every k >= 30 (was 3.0 Hz at k=0). Not adopted; parameter kept, default
0 (disabled).

### Status

Four candidates for the dark MB->CX/LAL layer are now refuted (background hold,
global STD, central-only STD, synapse saturation) and one (glutamate sign) is ruled
out by the reference paper's own experiment. None is a quick parameter fix. What is
left is architectural, not parametric - the two live candidates are:

1. **A real graded/non-spiking treatment for the AL-and-lateral-horn interneuron
   population** (the way APL is already handled as a rate, not a spiking LIF cell -
   decision 12). LHCENT, CRE, SIP, mALB cells might be the same category: local,
   possibly non-spiking, and mis-modelled as ordinary spiking neurons drives their
   inhibition far past what the real graded cell would deliver at saturation.
2. **Per-hemilineage neurotransmitter re-evaluation** at the ambiguous types Eckstein
   et al. flagged, the way lLN1/lLN2 was - but for LHCENT/CRE/mALB types, which would
   need the same 45-of-53-glomeruli-style functional test as decision 13, one type at
   a time, and there is no shortcut to it.

Both are substantially larger investigations than anything tried today and deserve a
session of their own with a full re-validation plan, not a bolt-on parameter.

**Recommendation:** treat the compartment-specific associative conditioning result
(measured this session) as complete and reportable on its own terms - synaptic
learning that is associative, reversible, and anatomically compartment-specific in a
whole-CNS connectome model, with the finding that it does not (yet) reach descending
behaviour clearly diagnosed and localised. Reaching behaviour is now a separate,
scoped research question, not a bug to patch.

## The MB -> DN pathway is live in a quiet brain and swamped by the odour (`mbon_to_dn.py`)

Instead of opening intermediate layers, ask directly: drive one MBON type at a
physiological rate and count what moves on the action bus. 20 mV tonic, 800 ms, paired
seed against the same condition with the MBON quiet.

**No odour** - the pathway is strong, specific and signed:

| MBON driven | rate | DN total change | DNs moved (>3 spikes) | biggest |
|---|---|---|---|---|
| MBON05 (gamma4, glut) | 82 Hz | **-2,146** | 231 | DNg33 -152, DNg70 -82, MDN -45 |
| MBON18 (alpha2sc) | 29 Hz | **-3,703** | 273 | DNg33 -308, MDN -101 |
| MBON21 (gamma4gamma5, ACh) | 51 Hz | **+831** | 163 | DNg33 +67, DNg02 +42, DNp31 +30 |
| MBON13 (alpha'2) | 54 Hz | +337 | 81 | DNg33 +68 |
| MBON09 (gamma3beta'1, GABA) | 67 Hz | -2,577 | 274 | DNge143 +111, MDN -81 |

Individual MBON types move hundreds of DNs by thousands of spikes, with opposite
signs (MBON05 and MBON21 - the two appetitive-core cells - push DNg33 in opposite
directions), and reach MDN, the moonwalker backward-walking command neuron.

**With CS+ at full strength** the same drives move +202, +107, -311 - ten to twenty
times less - and MBON18/MBON21 cannot fire at all under 20 mV. The odour-evoked state
shuts the convergence layer during exactly the window the learning is read in.

**Why: the central brain is inhibition-dominated, and the odour pathway is
all-or-none at the DNs.** Spike counts, central neurons (not sensory, KC or DN):

| odour strength | ORN Hz | own-PN Hz | LH spikes | central E | central I | I/E | DN spikes |
|---|---|---|---|---|---|---|---|
| 0 (MBON hold only) | 0 | 0 | 123 | 13,111 | 25,493 | **1.94** | 4,259 |
| 0.1 | 19 | 38 | 1,524 | 17,312 | 30,213 | 1.75 | 4,231 |
| 0.2 | 35 | 88 | 4,450 | 29,290 | 51,668 | 1.76 | **6,732** |
| 0.5 | 77 | 165 | 10,456 | 34,382 | 56,710 | 1.65 | 6,390 |
| 1.0 | 124 | 204 | 14,087 | 38,810 | 62,937 | 1.62 | 6,264 |

The chain is graded from receptors through the lateral horn, then the DN population
jumps to ~6,500 at strength 0.2 and stays there. And the central brain fires
**1.6-1.9 inhibitory spikes per excitatory spike** at rest and at every intensity:
inhibitory cells run ~3x the rate of excitatory ones on average. That is the dark
convergence layer, and it is why none of the four bolt-on fixes could open it.

**The practical route: run the test where the MB is not swamped.** Strength 1.0 is
200 Hz on every receptor of 8 glomeruli, which pins PNs at 204 Hz (their refractory
ceiling); real sustained PN rates are 50-130 Hz. At strength 0.35 (PN 130 Hz) with
kc_thresh 1.0:

    KC 264 / 358 active (6.5-8.8%), Jaccard 0.035 vs 0.040 chance
    core A 46 / 54, core P 36 / 56  - both cores respond to both odours, with margin
    MBON05 drive -> DNs with odour present: +992 spikes across 172 DNs  (was +202)

The mushroom body can move the action bus in this regime. `--strength` and
`--kc-thresh` now thread through conditioning4 / learned_dn / learned_steering.

## LEARNING REACHES THE ACTION BUS (`learned_dn.py`, kc_thresh 1.0, strength 0.35)

Same training, same paired-seed population test as before, at the moderate-odour
regime. 8 seeds, 12 trials.

    corr(dS_both, dS_reversed):   MBONs  -0.803     DNs  -0.801
    DNs whose (CS+ - CS-) response change reverses with the contingency
        (|z|>3, >1 spike):  17 / 1,310           (full-strength regime: 3, i.e. chance)

The learned change at the descending neurons now reverses with the contingency
across the population, as strongly as it does at the MBONs. The cells that carry it
come in **bilateral pairs with matching sign**:

| DN | side | dS both | dS reversed | z | naive S |
|---|---|---|---|---|---|
| DNd02 | R / L | -6.0 / -4.5 | +5.4 / +5.1 | -10.5 / -5.8 | +7 / +8 |
| DNbe007 | L / R | -15.3 / -16.5 | +11.8 / +12.6 | -6.4 / -5.4 | +7 / +13 |
| DNge069 | R / L | -7.5 / -7.1 | +4.9 / +4.3 | -5.2 / -4.4 | +3 / +4 |
| DNg33 | L / R | **+79.9 / +80.3** | **-51.1 / -50.6** | +3.1 / +3.1 | +65 / +65 |
| DNp56 | R / L | -2.1 / -2.4 | +0.9 / +0.3 | -3.4 / -3.2 | -1 / -1 |
| DNge143 | L / R | +3.1 / +2.9 | -3.4 / -3.3 | +3.5 / +3.1 | -1 / -1 |

The two hemispheres are independent readouts, so paired agreement is an internal
replication. DNg33 - the cell MBON05 and MBON21 pushed in opposite directions in the
influence map - moves by +80 / -51 spikes per 800 ms, the largest learned change on
the bus. DNbe007 and DNd02 shift by 5-16 spikes on a naive baseline of 7-13, i.e.
their odour preference roughly inverts.

What changed is not the circuit or the rule: it is that the test is run where the
odour does not saturate the pathway. At strength 1.0 the DN population sits at its
ceiling and MBON output cannot move it; at 0.35 (PN 130 Hz, within the biological
sustained range) it can, and the learned signal comes through.

**Counterbalanced (`learned_dn_mod_swap.json`).** Odours swapped, same regime:
corr(dS_both, dS_reversed) MBONs -0.792, **DNs -0.617**, 59 DNs reversing at |z|>3.
The top cells are the same ones with the same sign relative to the contingency -
DNd02 L/R (z -5.8 / -5.1), DNge069 L/R (-5.3 / -5.1), DNp56 - so they follow WHICH
ODOUR WAS PUNISHED, not which odour it was. Two odour assignments, two hemispheres,
paired seeds, plasticity-off control at exactly zero: the learned change at these
descending neurons is associative by the same standard the synaptic result met.

The DNa steering family is not where it lives (learned_steering3: both -0.07, sd
0.17). The learned valence is carried by DNd02, DNge069, DNbe007, DNg33 and DNp56.
What those neurons command in the animal is the next question, and it is a question
about the literature, not the model.

## What the learned DNs command — traced to muscles in the connectome

`learned_steering3`: DNa family both -0.074, reversed -0.069 (sd 0.16): the DNa
steering family does not carry the learned signal; the population test already said
where it lives. Tracing those DNs through the VNC to the 699 muscle-labelled motor
neurons (`motormap.py`'s map), direct and two-hop, signed:

| DN | nt | direct motor targets | two hops |
|---|---|---|---|
| DNbe007 | ACh | T3 MNhl62 (+18), T1 coxa promotor (+9), wing steering ps1 / MNwm35 / MNwm36 / hg4 | T2 trochanter flexor (+12), T1 promotor (+12), T2 tibia extensor (+8) |
| DNge069 | glutamate (inh.) | **T2 trochanter flexor (-40)**, T2/T1 sternal anterior rotator (-11, -5) | **TTMn +15 (jump muscle)**, T2 sternotrochanter +10 |
| DNg33 | ACh | abdominal MNad03 A3-A5 (+107), MNad25/22 | wing power DLMn (+20), DVMn (+21) |
| DNp56 | ACh | none | T1 coxa promotor (+10), T1 tibia extensor (+5) |
| DNd02 | **unclear** | **none - sign 0, no output in the model** | - |

Leg, wing and abdominal motor DNs. DNge069 suppresses mid-leg lift and primes the
jump muscle; DNbe007 drives hind-leg and wing-steering motor neurons. DNd02 is the
largest-z learned cell and cannot transmit: its consensus transmitter is "unclear"
(the literature has it co-releasing glutamate and tyramine; Cande et al. 2018 has it
driving slow locomotion), so the model silences it. Another transmitter-label
consequence, this time on the output side.

For comparison DNa02 (the steering neuron) goes direct to coxa rotators in all three
legs (+92 / +87 / +64) - a different motor pool, untouched by the learning.

## IT LEARNS — the change reaches the motor neurons (`learned_motor.json`)

`learned_dn.py` extended to the 699 muscle-labelled VNC motor neurons, same run,
same paired-seed design, kc_thresh 1.0, strength 0.35, 8 seeds, 12 trials:

    corr(dS_both, dS_reversed):   MBONs -0.803    DNs -0.801    MOTOR NEURONS -0.916
    motor neurons spiking during odour: 425 / 699;  reversing at |z|>3: 33

The strongest reversal of any layer is at the muscles. The cells are a coherent,
bilateral motor pattern:

| motor neuron | muscle | dS both (punish CS+) | dS reversed | naive S |
|---|---|---|---|---|
| DVMn 2a,b  L / R / R / L | wing upstroke (power) | +39.0 / +41.8 / +42.0 / +46.1 | -25.1 / -25.5 / -26.4 / -27.9 | +48 |
| DLMn c-f  R | wing downstroke (power) | +57.8 | -33.6 | +34 |
| MNhm03  R / L | haltere | +8.4 / +3.5 | -9.6 / -7.8 | +14 / +10 |
| MNad03 R, MNad42 R, MNad09 L | abdominal | +10.6 / +5.4 / +0.8 | -7.4 / -9.0 / -1.1 | +16 / +24 / 0 |
| Tr flexor R, Sternal post. rotator R | leg | +1.5 / +6.3 | -3.0 / -4.4 | +3 / +10 |

After punishing CS+, the punished odour evokes MORE wing-power, haltere and
abdominal motor-neuron drive relative to the rewarded odour; reverse the contingency
and every one of these flips sign. Wing power muscles + halteres + abdomen is the
flight motor. This is consistent with a takeoff / escape response to the punished
odour - which is what aversive olfactory conditioning produces in the animal - and
it is stated as consistent-with, not as a demonstrated behaviour: there is no body
here, only the command to the muscles.

The pathway is traceable link by link, each measured today:

    KC -> MBON05 / MBON21   (the plastic synapses, PAM08 compartment, d_P reverses)
        -> DNg33            (influence map: MBON05 -152, MBON21 +67 on DNg33;
                             learned change +80 / -51, bilateral)
        -> DLMn / DVMn      (DNg33 two-hop weight onto wing power motor neurons)

**What made the difference was not the circuit or the rule - both were finished this
morning.** It was running the test where the odour does not saturate the descending
population (strength 0.35, PN ~130 Hz, in the sustained biological range) instead of
at 200 Hz on every receptor, where the DNs sit at a ceiling that MBON output cannot
move. The four "fixes" refuted this afternoon were all attempts to move the ceiling;
the answer was to stop driving the network into it.

Counterbalanced at the DN level (odours swapped: DNs -0.617, same cells); motor-level
swap in `learned_motor_swap.json`.

**Counterbalanced at the muscles (`learned_motor_swap.json`).** Odours swapped:
MBONs -0.792, DNs -0.617, **motor neurons -0.628**, 30 reversing at |z|>3. Same sign
relative to the contingency, same category - the punished odour drives more wing
and leg motor output - with the particular muscles shifting from wing power
(DLMn/DVMn) to wing steering (b1 L/R, b2 L/R, tp1) and leg rotators (sternal
posterior rotator, pleural remotor, Tr flexor) under the other assignment. The
category and the sign follow the contingency; which muscles is odour identity.

That completes the standard the synaptic result was held to, at every layer the
model can read: two odour assignments, two hemispheres, paired seeds,
plasticity-off control at exactly zero, sign reversal with the contingency -
at the MBONs, at the descending neurons, and at the motor neurons.

## The survival box, and the taste / hunger layer researched before building on it

`flybox.py` (in progress): a continuously running arena - food, water, heat hazards,
shelter, day/night; energy, water, warmth; death replaces the body and by default the
brain persists; eating drives PAM08 and hazards drive PPL105 with plasticity on, so
the world is the teacher. First smoke run exposed three weakly modelled layers, and
each was researched rather than patched:

**Gustatory identity - decision 16.** `taste()` split sweet/bitter by ARRAY INDEX. The
"sweet" 120 drove the proboscis motor neurons 0 times; the "bitter" 120 drove them 435
spikes/tick. Types are named by body part (LB labellum, LgLG/LgAG leg, WG wing, PhG
pharynx), not modality, and the dataset's `subclass`/`receptorType` carry none. The
published mapping (taste-feeding connectome, bioRxiv 2025.08.25.671814 / Cell 2026,
built on MaleCNS): **LB1a-d bitter (Gr33a), LB3a water (ppk28), LB3b low salt
(Ir56b), LB3b-c sugar (Gr64f), LB3d aversive heavy-metal (Ir47a)**; LB2 and LB4 are
novel/unassigned. Leg GRNs: LgAG ascend to the brain (feeding initiation), LgLG stay
in the thoracic ganglia (locomotion suppression; Thoma et al. 2016).

**The model's taste pathway is not modality-selective (`grn_screen`, `lglg_screen`).**
Driving each type alone against the feeding motor neurons (MN9/10/11/12/MNx01, the
pathway Shiu et al. validated): LB3d (aversive) 1,633 spikes/600 ms, LB4a/b, LB2a,
PhG1c/8/9, taste pegs 1,600-1,800; sugar LB3c **564**; bitter LB1a-d 0; water LB3a 41.
In a walking context (mechano 60 Hz) the feeding motor neurons sit at 1,590 with no
taste at all. Bitter suppression of sweet: -12%. Walking suppression by any gustatory
type: at most -23% (LB3b). So "the fly is feeding" cannot yet be read from the
proboscis motor neurons selectively, and "sugar stops walking" is weak. Both are
recorded as limits of the SEZ under this parameterisation, not worked around.

**Hunger cannot enter through the identified neurons - transmitter labels again.**
The hunger/satiety cells are in the dataset by name and their directions are
sourced: IPC (16; insulin + DSK, active when fed, drop with starvation), DH44 (6;
internal nutrient sensors), AstA1 (2; satiety), NPFL1 (2; hunger, gates appetitive
memory - Krashes 2009), LK (12; rises with starvation), Hugin-RG (4). But **IPC, DH44,
LK, NPFL1 and Hugin all carry `consensus_nt = unclear`, sign 0, and have zero
out-edges in the simulation** - like DNd02. Only AstA1 (GABA) transmits. The
`define_drive("hunger", "endocrine")` hack drives a population that cannot speak.
Peptidergic modulation is outside what a fast-synapse LIF represents; a sourced
substitute has to model the peptide's *targets* (e.g. NPF -> PPL1-gamma1pedc gating,
insulin -> sugar-GRN sensitivity via DopEcR, Inagaki 2012), which is the next piece
of research, not a parameter.

### Next (resume here)
1. Finish `flybox.py` with the sourced gustatory sets; read feeding from LB3c-driven
   proboscis motor activity with the mechano confound measured, or find the
   sugar-selective second-order neurons (G2N-1, Zorro, Usnea... Sterne et al. 2021)
   and read those.
2. Hunger: implement through targets, not peptide cells - NPF gating of the PPL1
   channel and sugar-GRN gain, each with its source.
3. Then the box experiment: lifespan across lives, inherit vs naive, time near
   hazards vs food.

**Sources verified (10 Sept, evening).** Taste pegs: `dorsal_tpGRN` / `claw_tpGRN`
match Gr5a-, Ir60d-, Ir56d- and Gr64e-GAL4 (Cell 2026 version of the taste-feeding
connectome) - Gr5a and Gr64e are sugar receptors, so the taste pegs join the sweet
set and their 1,600-1,800 feeding-motor drive in the screen was right. With the
mechanosensory confound removed (the box drives none): **sugar set -> 260 proboscis
motor spikes/tick, rest 4, bitter 0** - selective. Pharyngeal PhG types are not
mapped to modality in the open-access texts; left out (they fire after ingestion).

Hunger targets, verified: PPL101 = PPL1-gamma1pedc = MB-MP1; "dopaminergic PPL1
MB-MP1 neurons are inhibited by NPF in hungry flies, allowing the retrieval of
appetitive memories ... stimulating them suppresses performance in hungry flies"
(Krashes et al. 2009, Cell) -> tonic drive on PPL101 proportional to satiety.
"Starvation increases dopaminergic release onto Gr5a sugar-sensing neurons, and
DopEcR is required in these sensory neurons ... a single dopaminergic neuron
(TH-VUM) is likely the source" (Inagaki et al. 2012, Cell) -> sugar GRN gain
0.5 + 0.5 x hunger. Hugin-AstA as a central energy sensor regulating sweet
sensation (eLife 2025) is consistent with AstA1 being the one hunger cell that
transmits here.

## Field survey, 10 Sept 2026 (see cross-reference section for the papers)

Open-source models on FlyWire / MaleCNS: fly-brain-minecraft (MaleCNS, Shiu LIF, KC
gain 0.25 as a sparsity hack, **"antennal lobe saturates under odor"** - they paint
visual signals onto LC4/LPLC2 and use a reflex layer for food; no plasticity),
desktop-fly (817 stars; 668-neuron FlyWire circuit + MaleCNS VNC; same DN map as
ours; no learning; honest that tests do not show agreement with real locomotion),
snedea/flybrain (browser LIF, no validation), eonsystemspbc/fly-brain (Shiu LIF on
six GPU backends, no senses), lixiang1076/fly-brain (KC->MBON "dopamine learning",
no controls or reversal), Loihi 2 neuromorphic (arXiv 2508.16792), FlyGM (trained
GNN controller, arXiv 2602.17997), Fly64 (MaleCNS -> Mario 64, same DN choices).

None has controlled associative learning; none reads motor neurons; all assign
glutamate inhibitory and drop "unclear" cells, so all share the silent peptidergic
/ DNd02 problem. The antennal-lobe saturation is independently reported, which is
evidence for the FlyEM note that it is the transmitter table, not us.

**Water -> proboscis is dark (`flybox.py`, 10 Sept evening).** ppk28 water GRNs (LB3a,
17 cells, ACh) at 200 Hz: 6 proboscis motor spikes/tick, i.e. rest; the sugar set gives
454 and saturates by 100 Hz. Their second-order targets fire strongly (GNG229 241,
GNG175 230, DNg67 167 spikes/800 ms; E/I healthy) but **GNG229 and GNG175 are labelled
GABAergic** and DNg67 descends to the VNC, so there is no excitatory route to MN9/11/12.
The ISNs (4 cells, the thirst/hunger integrators of Jourjine et al. 2016) driven at
30 mV also give nothing. Either those two labels are wrong - the same pattern as
lLN1/lLN2, DNd02 and the peptide cells - or the water PER path lies below the
5-synapse threshold. Unresolved. **In the box, drinking is therefore IMPOSED** (uptake
on contact while stationary, gated by thirst) and labelled as such in the code;
eating is derived from the proboscis motor neurons. First life (box1): ate 69 ticks,
drank 106, 20 hazard contacts costing 0.6 of its energy - hazards are what kills it,
which is what the learning has to fix.

## The swarm engine: 100 flies on a GTX 1650, faster than real time (`flysim_gpu.py`)

Same model - the weight matrix is built by `FlyBrain` with every decision baked in,
then stepped for B flies on the GPU with state tensors [B, N]. Only the 33,496
plastic KC->MBON weights are per fly ([B, E]); the connectome is shared.
**Verified statistically** against the CPU engine (`gpu_verify.py`, designed CS+,
bilateral 0.35/0.21, 800 ms; CPU 3 seeds vs 8 GPU flies):

| population | CPU mean [range] | GPU mean [range] |
|---|---|---|
| KC active cells | 158 [151-164] | 166 [153-179] |
| MBON spikes | 621 [609-630] | 619 [582-670] |
| DN spikes | 6,030 [5,750-6,318] | 5,824 [5,332-6,554] |
| leg motor spikes | 2,714 [2,561-2,867] | 2,692 [2,501-3,089] |

Two things made it fast, both measured. (1) A dense spike-matrix product costs
107 ms/step for 64 flies - 466M multiply-adds, 99.6% by zero. Event-driven
propagation (decision 10, batched: gather the out-edges of every (fly, neuron) that
fired, one index_put) halves the no-learning cost. (2) With learning on the
dopamine loop was 60% of a step because it ran over [B, 162k] for every DAN type;
dopamine exists at 97 MBONs and traces at 4,064 KCs, so the learning state is kept
there. Result, plasticity ON:

| flies | ms/step (all) | ms per fly-step | real-time factor per fly |
|---|---|---|---|
| 8 | 14.2 | 1.78 | 0.56x |
| 64 | 44.5 | 0.70 | 1.44x |
| **100** | **64.5** | **0.65** | **1.55x** |

CPU engine: ~3.5-4 ms per fly-step. Hardware: GTX 1650 Max-Q (4 GB), i5-10300H.
Noise and Poisson draws use different generators, so the comparison is statistical
by necessity; the conditioning result should be reproduced on the swarm before it is
used for a claim (not yet done).

**Box, first survivor.** With derived feeding and thirst-gated drinking a naive brain
survived 20 min of brain time (1,001 feeding ticks, 30 drinks, 86 hazard contacts,
weights 96%): no death, so no inherit-vs-naive reading from that pair of runs. The
lineage experiment needs a world that kills - which is what the Verge's Lieutenant
is for.

## Fly tribes in the Verge: sight, the whole craft list as novel effectors, and a full stop (`verge_swarm.py`, `eye.py`)

**Vision, and its limits.** `eye.py` builds a compound eye from the L1 cartridge
positions in the connectome (1,348 photoreceptor inputs with azimuth/elevation), and
`FlySwarm(vision=True)` drives them with a rendered luminance scene (rate proportional
to brightness - an earlier draft had the sign backwards and was corrected). What the
flies get from it is honest and small: bright/dark blobs at a bearing. The T4/T5
looming pathway is dark in the whole-brain model (graded periphery, the same
convergence-layer problem as the DNs), so a Lieutenant is a dark blob, not a threat,
until PPL105 has paired it with a hit.

**The novel-effector experiment, made real for the whole craft list.** The Verge has 44
verbs a fly has no circuit for (make spear/hammer/knife/axe/…, saw planks, dress
blocks, smelt, build fire/wall/door/bed/well/…, gather). Each is bound to one
gnathal descending-neuron TYPE with no established role in the field, the 44
largest DNg/DNge types with at least two cells, in size order so the binding is
reproducible (gather -> DNg08, eat -> DNg07, cook -> DNg106, build -> DNge091, spear ->
DNge094, …; the full map is written to `state.json` as `pool_names`). A verb is
pressed when its type fires above its own running baseline (Welford mean + 1.5 sd,
after 20 decisions) AND the pack holds the game's own material cost for that verb
(the game enforces it anyway; this just makes the attempt count meaningful). Every
new thing in the pack fires PAM08; a made thing is logged as `crafted` and kept
across lives. This is an interface choice, labelled; whether the brains ever chain
gather -> spear, or gather -> hammer -> mason bench -> blocks -> wall, is the
question. Nothing in the connectome says DNg08 means "gather"; if the association
forms, it formed through PAM08 -> KC->MBON plasticity -> DN, the same path the
odour conditioning traced.

**Overeating, and the fix from the biology.** The first run showed the flies
stripping bushes while full: the game clamps satiety at NEED_MAX but still
harvests the bush, so a fed fly with sugar GRNs at half gain kept pressing the
action button and starving its own tribe. Two published mechanisms now stop it:
(1) Piezo-expressing mechanosensory neurons on the crop end the meal when it is
distended (Min et al. 2021, eLife; Piezo-null flies overeat until the crop is
grossly enlarged) - sugar drive fades over the last quarter of satiety and is zero
from 90 %; (2) the brain's own fructose sensor Gr43a is appetitive when hungry and
aversive when fed (Miyamoto et al. 2012, Cell) - feeding-motor drive on a bush with
a full crop gives a half-strength PPL105 pulse, so overeating is taught against,
not only blocked. Drinking was already thirst-gated at 90 % (ISN osmosensing,
Jourjine et al. 2016, is the analogue; drinking itself remains IMPOSED).

**Spectator page** (`xaya/prototypes/stage-b/flies.html` + `flies.js`, served by the
game): the game's own renderer over the union of what all flies currently see,
one card per fly with a live regional-activity brain, flash feedback, View all /
double-click-to-follow, and a persistent gold ⚒ badge + banner (localStorage) so a
craft that happens while nobody is watching is still on screen later.

## First craft, and the social layer (`verge_swarm.py`, 11 Sept 2026)

**First complete gather -> craft chain.** Fly-1-07 (Tribe 1, naive, first life, decision
470, game tick 1,385,063) made a hammer: wood and stone picked up with the action
button, then the `makeHammer` pool - DNg03, 12 cells, no function assigned in the
field - fired above its own baseline with the materials present and the game
accepted the press (15 attempts, 1 accepted). One hammer is not learning; the
per-fly `craft x/y` rate over lineages is the measure. A hammer unlocks every
`build*` verb.

**Inherit vs naive, first honest reading: no difference yet.** 109 deaths pooled
across runs: inherit tribe mean life 135 decisions / 8.3 forages, naive 178 / 12.2;
no trend in successive life lengths. Reasons, in order: total KC->MBON change is
~0.5 % after 55 lives (the punishment context is noisy); what kills them is the
Lieutenant, which reaches the brain only as a dark blob through the weakest path
in the model (T4/T5 dark), so the mushroom body has little to associate with the
hit; lives are short enough that a naive fly learns as much per life. The test
that would settle it: a smellable threat through the AL, matched runs, hours.

**The social layer**, every channel on the receptor the field has for it, from the
connectome's own annotations (`body-annotations` feather: `dimorphism`,
`receptorType`, `class/subclass`, `rootSide`):

| channel | emitter | receiver in the connectome | basis |
|---|---|---|---|
| kin scent | every fly | ORN_VA1v (Or47b) | fly-derived methyl laurate, Dweck et al. 2015 |
| stranger scent | other tribe | ORN_VA1d (Or88a) | fly-derived methyl palmitate; "a fly, not mine" is a labelled interface choice |
| male pheromone | every male, volatile | ORN_DA1 (Or67d), cVA | Kurtovic et al. 2007 |
| alarm | a fly that was just hit, 60 s | ORN_V (Gr21a/Gr63a), CO2 | Drosophila stress odorant, Suh et al. 2004 |
| contact pheromone | touching fly, by its sex | putative_ppk23 / ppk25 leg neurons (269 / 257 cells), by side | 7,11-HD / 7-tricosene, Thistle et al. 2012, Toda et al. 2012 |
| touch | adjacent fly | leg tactile bristles (213 cells), by side | Ramdya et al. 2015 |
| wing flick | a fly hit or escaping, 5 s | the compound eye: a larger, brighter, varying blob | Kacsoh et al. 2015 (needs vision + MB in the receiver) |
| song | a male whose pC1/pIP10 fire above baseline with a female within 2 tiles | JO-A / JO-B (Johnston's organ) | pIP10 is the song descending neuron |
| gift | `give` / `cycleOffer` verb pools | the game's bond system ("gave" / "fed") | interface |

**Sex.** Souls alternate male/female within a tribe. A male is the MaleCNS brain.
A female is the MaleCNS brain with its 1,258 male-specific cells (annotation
`dimorphism` = male-specific or potentially male-specific: P1/pC1 fru+ types,
mAL_m*, pIP10, SMP703m, …) silenced per fly (`FlySwarm.silence`). That is a
labelled substitute, not a female: the sexually-dimorphic cells that exist in
both sexes with different wiring (771 annotated) are left male, and the
female-specific circuits (female pC1 subtypes, vpoDN, the receptivity path) are
absent. The accurate substitute is the female FlyWire brain (FAFB, Dorkenwald et
al. 2024), which needs its own build (different naming, no VNC); flagged as the
next model upgrade. Everything a "female" does here is what a male brain minus
male-specific cells does with female pheromone emission.

**Chat.** Each fly speaks its social events into the game's chat as an agent, rate
limited to one line per 6 s: hit (alarm + wing flick), escape (wing flick),
song ("*sings to Fly-0-03♀* (pIP10)"), hearing, touch, alarm smelled, gift, and
crafts ("I made a hammer!"). The spectator page shows the same on each card and
flashes pink on song/touch/gift.

## Scent words and a dopamine limit (`verge_swarm.py`, 11 Sept 2026)

**Is there a fly language?** Not in the field: the repertoire (cVA, 7,11-HD, CO2,
song, wing flicks) is innate and syntax-free; a male raised alone sings normally.
What this brain can support is a signal acquiring meaning in a *listener* through
the one plastic site it has (KC->MBON, dopamine-gated), which we have shown reaches
the DNs. That is the experiment, labelled as an experiment with the fly's machinery
and not as fly biology.

**Design.** Four words (bzz, vrr, tik, hum). Emission: each bound to an unassigned
DN type (DNge087/092/093/113), pressed when the type bursts above its own baseline
by 2.5 sd (~1 in 200 decisions by chance); a word hangs in the air for 20 s where
it was said. Reception: each word is one olfactory receptor type no other odour in
this world uses (ORN_DM1, DP1m, DM2, DC1 - the largest Kenyon-cell footprints in
`odours3.json`), smelled bilaterally like any plume. No meaning is assigned. The
dictionary is written by the world: every emission logs its context (food, water,
danger, stranger, kin near; hungry, thirsty), every hearing logs whether the
listener closed on the speaker or left within three decisions, and whether a reward
or a hit followed within the word's window. Song and other social channels have
words alongside them in the chat.

**Expectation, stated before the data.** Listener-side meaning (a word's MBON
response diverging with its history) should appear if a word is reliably paired
with reward or punishment, as odours were. Speaker-side use (a word said at food
more often than baseline) has no mechanism behind it - nothing reinforces
emitting - so if it appears it is a finding, and if it does not, the model has
listeners who understand a vocabulary nobody speaks on purpose.

**The dopamine limit.** Flies in the box and the Verge sat on rewards. Two limits,
labelled: (1) need-gating - appetitive reward needs hunger (Krashes et al. 2009:
sated flies form no sugar memory), so a PAM08 pulse scales with the need it met,
full above 50 %, zero when sated; (2) a budget - each pulse spends a quarter of a
pool that refills over a minute, so runs of reward taper (dopamine release does
adapt with repetition; the number is ours). Punishment is not gated. First minutes:
the budget works (Fly-0-00 at 7 % after four rewards), and the clay/bush loops are
gone.

## Generations: courtship, acceptance, children, inheritance as the lever (11 Sept 2026)

No individual respawn any more. A generation starts with eight founders (per tribe
two males, two females); when the last fly dies the next generation's founders
arrive. A male sings when pC1/pIP10 burst with a female within two tiles; the
female accepts by staying within a tile of him for three decisions at under half
speed (a receptive female slows and stays - Coen et al. 2014; the female-specific
decision circuit, vpoDN, is not in the male connectome, so the behaviour IS the
acceptance here, labelled). A child is due 1,500 game ticks later, joins the
mother's tribe with a coin-flip sex, and needs a free brain slot (capacity 16).

Inheritance is the tribe policy and the experiment's lever: Tribe 0 children take
the mean of their parents' plastic KC->MBON weights (the father's snapshotted at
mating); Tribe 1 children are born naive. Memories are not inherited in real flies;
this Lamarckian rule is the tool, labelled as such. The question is whether the
inheriting lineage outlasts the naive one over generations, measured by births,
deaths and generation length per tribe in `events.jsonl`.

Craft tally so far: three hammers (Fly-1-07 twice across lives, Fly-1-04), all
Tribe 1, all in the old respawn regime; one gift (wood) by Fly-1-07.

## The hive mind (11 Sept 2026)

`--hive tribe|all` (default in `run_verge.ps1`; `-Solo` for individual brains).
Every fly keeps its own brain, senses and body; the plastic KC->MBON weights - the
one place this brain learns - are a single shared vector per tribe. Each fly's
dopamine-tagged change is applied to the shared weights (the product of the
members' multiplicative factors, `FlySwarm.set_hive`), and every member reads the
same weights back next step. Children are born into the hive's memory; the memory
outlives a generation. Verified at boot: every Tribe 0 fly at 0.244 %, every
Tribe 1 fly at 0.230 % memory change after one minute. Nothing in fly biology
does this; it is the collective-learning comparison condition, labelled. Separate
memory files per mode (`results/verge_hive/memory.pt`, `results/verge/memory.pt`).

Also this session: memory across sessions (`memory.pt`, restored on start; `--fresh`
or the page's Restart wipes it), tribe memory seeding the next generation's founders,
all tribes inherit by default, words bound by a 60-decision activity survey
(bzz->DNge129, vrr->DNg70, tik->DNg56, hum->DNg98 in the first run), the crop stop
made a threshold after the taper stopped all feeding, the courting male tracking the
female and the receptive female slowing, an edge reflex, a page Restart with
confirmation via a local control port, and Telegram summaries.
