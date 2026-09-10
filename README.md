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

- `scripts/flysim.py` - the engine. Design decisions are numbered in its docstrings.
- `scripts/conditioning4.py` - olfactory conditioning, differential compartment readout
- `scripts/odour_design.py`, `kc_sparsity.py`, `al_selectivity.py`, `mbon_hold.py` -
  the input- and output-code diagnostics
- `scripts/learned_steering.py` - does learning reach the descending neurons
- `scripts/steering_protocol.py`, `habituation.py`, `motormap.py`, `teaching_signal.py`,
  `probe_ladder.py`
- `scripts/flyworld.py`, `fly3d.py`, `desktop_fly.py` - embodiment
- `web/` - published reports and the 3D episode viewer
- `results/` - JSON/log output of every run cited in FINDINGS.md
- `notes/` - drafts of correspondence

## Credits and references

This project is built on other people's measurements. Where a design decision rests
on a published result, FINDINGS.md and the docstrings in `flysim.py` say which.

**Data**
- Male CNS Connectome v1.0 - HHMI Janelia FlyEM Project Team, the Cambridge
  Drosophila Connectomics Group (MRC LMB), and Google Research Connectomics.
  CC-BY 4.0. https://male-cns.janelia.org/
- Neurotransmitter predictions - Eckstein, Bates, et al. 2024, *Cell*,
  "Neurotransmitter classification from electron microscopy images at synaptic
  sites in Drosophila melanogaster" (method); funkelab/synister_malecns (this dataset).

**Model**
- Neuron and synapse constants, the 0 Hz basal-rate convention, the alpha-synapse and
  delay, and the shuffle control - Shiu et al. 2024, *Nature* 634:210,
  "A Drosophila computational brain model reveals sensorimotor processing", and
  philshiu/Drosophila_brain_model.
- Glutamate as inhibitory (GluCl-alpha) - Liu & Wilson 2013.

**Biology the design decisions depend on**
- Compartmentalised mushroom-body learning, DAN/MBON valence organisation -
  Aso et al. 2014 (*eLife*); Owald et al. 2015 (*Neuron*); Hige et al. 2015.
- Mushroom-body connectome, DAN->MBON and MBON->MBON wiring - Li et al. 2020, *eLife*.
- Sparse, decorrelated Kenyon-cell coding and the APL feedback - Lin et al. 2014,
  *Nature Neuroscience*; Turner et al. 2008; Honegger et al. 2011.
- Antennal-lobe lateral inhibition and gain control - Olsen & Wilson 2008, *Nature*;
  Bhandawat et al. 2007; GABAergic identity of the lateral-lineage local neurons -
  Chou et al. 2010; Okada et al. 2009. Antennal-lobe cell types - Schlegel et al.
  2021, *eLife*.
- Photoreceptor histaminergic inhibition - Hardie 1989; Stuart et al. 2007.
- Short-term synaptic depression - Tsodyks & Markram 1997.
- Descending-neuron steering (DNa02) - Rayshubskiy et al. 2020; Giant Fiber escape -
  Allen et al. 2006; DN->DN axo-axonic organisation - Ceballos et al. 2026.
- Olfactory conditioning protocol and the reciprocal-odour learning index -
  Tully & Quinn 1985.

**Code and models ported or independently confirming choices**
- Body geometry, leg chain and tripod gait phases in `fly3d.py` / `desktop_fly.py` -
  DesktopFly-Linux (MIT, (c) 2026 Denis Shiryaev and contributors).
- Fly64 (same MaleCNS data driving Mario 64) - independently maps DNa02/DNg13 to
  steering and DNp01/DNp10 to jump, as here.

**Authorship**
- Simulation design, experiments and the written record: Mr. Moon (TheMrRaGe).
- Code written with Claude (Anthropic); each commit carries a co-author line.

Anything cited from memory in FINDINGS.md is marked CHECK until verified.
