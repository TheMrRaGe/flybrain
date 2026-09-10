# flybrain

A leaky integrate-and-fire simulation of the complete *Drosophila* male CNS connectome
(162,517 neurons, 6.1M connections), built to drive a creature and to test what the
wiring can and cannot do: steering, habituation, vision, and olfactory conditioning
with the measured dopamine-to-mushroom-body wiring.

**Read [FINDINGS.md](FINDINGS.md) first.** It is the record of everything measured,
every trap, and every design decision, in the order they were found. Nothing in it
was assumed.

## Data

Male CNS Connectome v1.0 (HHMI Janelia FlyEM + Google Research), CC-BY 4.0.
Download the three files from https://male-cns.janelia.org/download/ into `data/`:

    connectome-weights-male-cns-v1.0-minconf-0.5.feather
    body-annotations-male-cns-v1.0-minconf-0.5.feather
    body-neurotransmitters-male-cns-v1.0.feather

then

    cd scripts
    python3 build_creature.py --whole          # -> ../brain_whole.npz
    python3 mirror_connectome.py               # -> ../brain_mirrored.npz

Requires Python 3.9+, numpy, scipy, pyarrow.

## Layout

- `scripts/flysim.py` — the engine. Design decisions are numbered in its docstrings.
- `scripts/conditioning4.py` — olfactory conditioning, differential compartment readout
- `scripts/odour_design.py`, `kc_sparsity.py`, `al_selectivity.py` — the input-code diagnostics
- `scripts/learned_steering.py` — does learning reach the descending neurons
- `scripts/steering_protocol.py`, `habituation.py`, `motormap.py`, `teaching_signal.py`, `probe_ladder.py`
- `scripts/flyworld.py`, `fly3d.py`, `desktop_fly.py` — embodiment
- `web/` — published reports and the 3D episode viewer
- `results/` — JSON/log output of every run cited in FINDINGS.md

Model constants follow Shiu et al. 2024, *Nature* 634:210. Body geometry in the
embodied builds is ported from DesktopFly-Linux (MIT, (c) 2026 Denis Shiryaev and
contributors).
