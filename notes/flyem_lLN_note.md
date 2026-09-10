# Draft: note to the FlyEM / MaleCNS team on lLN1 / lLN2 transmitter predictions

*For review before sending. Suggested channel: GitHub issue on the MaleCNS or
synister_malecns repository, or email to the FlyEM project team contact listed at
male-cns.janelia.org. Fill in the repository URL at the bottom once pushed.*

---

**Subject:** MaleCNS v1.0 consensus_nt for lLN1 / lLN2 is inconsistent with antennal-lobe physiology under simulation

Hello,

Thank you for the v1.0 release. I am building a leaky integrate-and-fire model of the
full male CNS from the v1.0 files (connectome-weights, body-annotations,
body-neurotransmitters; traced, typed neurons at weight >= 5; 162,517 neurons), with
synapse sign taken from `consensus_nt` and model constants from Shiu et al. 2024. I
found something in the antennal lobe that I think is worth flagging, because it is a
functional test of the transmitter predictions rather than a morphological one.

**The predictions.** In `body-neurotransmitters-male-cns-v1.0.feather`:

- `lLN1*` (59 cells): 42 acetylcholine, 15 GABA, 2 unclear
- `lLN2*` (92 cells): 44 GABA, 40 acetylcholine, 8 unclear

Eckstein et al. 2024 noted that the ALl1/ALv2 hemilineages "seem to break Dale's law
... with similar morphology types predicted to express different transmitters" and
suggested 18-27% of antennal lobe LNs may be cholinergic. The 44/40 split inside lLN2
looks like that same ambiguity.

**The functional consequence, measured.** Taking the predictions at face value and
driving a single receptor type (ORN_DM6, Poisson at 200 Hz for 600 ms):

- projection neurons in **45 of 53 glomeruli** exceeded 5 Hz, twelve at saturation
- only **7%** of uniglomerular PN spikes were in DM6
- 96% of the excitatory drive onto an unstimulated glomerulus's PNs came from ALLNs,
  `lLN1_bc` alone 45%

That is, the antennal lobe loses glomerular identity entirely: any odour looks like
every odour. Downstream, two odours sharing no glomeruli shared 55% of their Kenyon
cells.

Treating lLN1 and lLN2 as GABAergic (151 cells re-signed, nothing else changed):

- **1 of 53** glomeruli above 5 Hz for a single receptor type; own-glomerulus share
  of PN spikes **1.00** (0.99 for a 5-glomerulus odour, with 0.4 Hz of lateral spill)
- the two odours shared **0** Kenyon cells

Re-signing *all* cholinergic-labelled ALLNs gave the same result, so the effect is
carried by lLN1/lLN2 specifically. This matches the physiology I know of
(inhibition-dominated lateral interaction, glomerulus-specific PN output; Olsen &
Wilson 2008, Bhandawat et al. 2007) and the reported GABAergic identity of these
lines (Chou et al. 2010).

I realise a point-neuron model is a blunt instrument and that the classifier may be
seeing real co-transmission. But a single glomerulus driving 45 others to saturation
is not a fly antennal lobe, and the prediction's own within-type inconsistency points
the same way. It may be worth a caveat in the release notes for anyone using
`consensus_nt` as synapse sign, and I would be glad to hear if you have ground truth
for these types.

Code, the measurement scripts (`scripts/al_selectivity.py`), and the full record are
at: **https://github.com/TheMrRaGe/flybrain**

Best regards,
<name>
