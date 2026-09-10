#!/usr/bin/env python3
"""
build_creature.py - carve a creature-sized network out of the male CNS connectome.

Every pipeline decision is printed as it is taken.

The full CNS is 162,517 neurons. A creature that senses, learns, navigates and acts
does not need all of it - it needs chemosensation, the mushroom body, the central
complex, the neuromodulatory drive machinery, and the descending action bus.
That is what this keeps.

Input   : the three .feather files from https://male-cns.janelia.org/download/ (CC-BY 4.0)
Output  : creature_net.npz

    bodyId[N] type[N] cls[N] sc[N] nt[N]      identity
    side[N]                                   soma side L/R/M - bilateral steering needs it
    sign[N]                                   +1 excitatory / -1 inhibitory / 0 modulatory
    pre[E] post[E] w[E]                       edges, as indices into the node table
"""
from __future__ import annotations
import argparse, gc
import numpy as np, pyarrow as pa, pyarrow.feather as pf

# Fly fast-transmission signs. ACh excites. GABA inhibits. Glutamate is inhibitory in
# the fly via the GluCl-alpha channel - the opposite of vertebrate cortex, and the single
# easiest thing to get wrong here. Histamine inhibits (photoreceptor -> LMC). Monoamines
# carry no fast sign: they modulate, and the simulator treats them separately.
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
        "dopamine": 0, "octopamine": 0, "serotonin": 0, "unclear": 0, "unknown": 0}

# What a creature needs, by the dataset's own labels.
KEEP_CLASS = {
    "olfactory", "gustatory", "thermosensory", "hygrosensory", "mechanosensory",  # senses
    "ALPN", "ALLN", "ALIN", "ALON", "SEZPN",                                      # relays
    "Kenyon_Cell", "MBON", "DAN",                                                 # learning
    "CX",                                                                         # navigation
}
KEEP_SUPERCLASS = {
    "descending_neuron",   # brain -> body. the action bus, 480 named types
    "cb_motor",            # direct motor output
    "ascending_neuron",    # body -> brain. proprioceptive feedback
    "cb_endocrine",        # internal state / drive signalling
    # cb_intrinsic is NOT optional. Learned valence reaches the action bus almost
    # entirely through it: MBON->DN direct is 151 connections, MBON->cb_intrinsic is
    # 8,904 and cb_intrinsic->DN is 96,423. Cutting it severs behaviour from learning.
    # It also carries the inhibition that balances the network (E:I 1.65:1 internally,
    # against 6.17:1 without it), which removes the need for a gain fudge factor.
    "cb_intrinsic",
}

def _laterality(ann):
    """
    Left/right for every neuron, which bilateral steering depends on.

    somaSide alone is not enough: all 2,635 olfactory receptors, 1,416 gustatory and
    4,078 of 4,107 visual neurons are marked "M", because sensory somas sit in the
    antenna, palp and retina - outside the imaged CNS volume. Their side is instead
    recoverable from rootSide, where the axon enters the brain. Without this the
    creature is spatially blind: measured turn response to a stimulus on the left
    versus the right was 0.0000, identical to four decimals.
    """
    soma = ann["somaSide"].fillna("").astype(str)
    root = ann["rootSide"].fillna("").astype(str)
    side = soma.where(soma.isin(["L", "R"]), root)
    return side.where(side.isin(["L", "R"]), "M").to_numpy().astype(str)


def log(m): print(m, flush=True)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dir with the annotation/neurotransmitter feathers")
    ap.add_argument("--weights", required=True, help="path to connectome-weights feather")
    ap.add_argument("--out", default="creature_net.npz")
    ap.add_argument("--min-weight", type=int, default=5,
                    help="drop connections below N synapses (default 5)")
    ap.add_argument("--whole", action="store_true",
                    help="keep the ENTIRE CNS - no circuit selection at all. Slower, but "
                         "no relay can be accidentally severed and the E:I balance is the "
                         "animal's own rather than an artefact of what was kept.")
    a = ap.parse_args()

    log("[1/4] neurons")
    ann = pf.read_table(f"{a.data}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                        columns=["bodyId", "type", "class", "superclass", "somaSide", "rootSide",
                                 "status"]).to_pandas()
    log(f"      {len(ann):,} segments in the annotation table")
    ann = ann[(ann["status"] == "Traced") & ann["type"].notna()]
    log(f"      {len(ann):,} traced, typed neurons (orphans, glia and debris dropped)")

    cls = ann["class"].fillna("").astype(str)
    sc  = ann["superclass"].fillna("").astype(str)
    if a.whole:
        ann = ann.sort_values("bodyId").reset_index(drop=True)
        log(f"      {len(ann):,} kept - WHOLE CNS, no circuit selection")
    else:
        keep = cls.isin(KEEP_CLASS) | sc.isin(KEEP_SUPERCLASS)
        ann = ann[keep].sort_values("bodyId").reset_index(drop=True)
        log(f"      {len(ann):,} kept for the creature "
            f"(sense + learn + navigate + act; optic lobe and the rest of the central brain dropped)")

    ids = ann["bodyId"].to_numpy(dtype="int64")

    log("[2/4] signs")
    nt = pf.read_table(f"{a.data}/body-neurotransmitters-male-cns-v1.0.feather",
                       columns=["body", "consensus_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]
    ntv  = nt.reindex(ids).fillna("unknown").to_numpy().astype(str)
    sign = np.array([SIGN.get(x, 0) for x in ntv], dtype="int8")
    log(f"      excitatory {int((sign > 0).sum()):,}  "
        f"inhibitory {int((sign < 0).sum()):,}  "
        f"modulatory/unknown {int((sign == 0).sum()):,}")

    log(f"[3/4] edges  (streaming, weight >= {a.min_weight})")
    def to_index(x: np.ndarray) -> np.ndarray:
        """bodyId -> row in the node table, -1 when the neuron is not in the subnetwork."""
        i = np.searchsorted(ids, x)
        np.clip(i, 0, len(ids) - 1, out=i)
        return np.where(ids[i] == x, i, -1).astype("int32")

    rdr = pa.ipc.open_file(pa.memory_map(a.weights))
    P, Q, W, raw = [], [], [], 0
    for b in range(rdr.num_record_batches):
        bt = rdr.get_batch(b)
        w = bt.column("weight").to_numpy(zero_copy_only=False)
        raw += len(w)
        k = w >= a.min_weight
        if k.any():
            pre  = to_index(bt.column("body_pre").to_numpy(zero_copy_only=False)[k].astype("int64"))
            post = to_index(bt.column("body_post").to_numpy(zero_copy_only=False)[k].astype("int64"))
            ok = (pre >= 0) & (post >= 0)
            if ok.any():
                P.append(pre[ok]); Q.append(post[ok]); W.append(w[k][ok].astype("int32"))
            del pre, post, ok
        del bt, w, k
        if b % 25 == 0:
            gc.collect()

    pre  = np.concatenate(P) if P else np.zeros(0, "int32")
    post = np.concatenate(Q) if Q else np.zeros(0, "int32")
    w    = np.concatenate(W) if W else np.zeros(0, "int32")
    del P, Q, W; gc.collect()
    log(f"      {raw:,} raw edges -> {len(pre):,} internal to the creature network")

    log(f"[4/4] writing {a.out}")
    np.savez_compressed(
        a.out,
        bodyId=ids,
        type=ann["type"].to_numpy().astype(str),
        cls=ann["class"].fillna("").to_numpy().astype(str),
        sc=ann["superclass"].fillna("").to_numpy().astype(str),
        side=_laterality(ann),
        nt=ntv, sign=sign, pre=pre, post=post, w=w,
    )

    c = ann["class"].fillna("").astype(str)
    s = ann["superclass"].fillna("").astype(str)
    log("      populations:")
    for name, n in [
        ("chemosensory in", int(c.isin(["olfactory", "gustatory"]).sum())),
        ("mushroom body  ", int(c.isin(["Kenyon_Cell", "MBON", "DAN"]).sum())),
        ("central complex", int((c == "CX").sum())),
        ("endocrine drive", int((s == "cb_endocrine").sum())),
        ("descending out ", int((s == "descending_neuron").sum())),
        ("visual proj.  ", int((s == "visual_projection").sum())),
        ("optic lobe    ", int(s.str.startswith("ol_").sum())),
    ]:
        log(f"        {name}  {n:>6,}")

if __name__ == "__main__":
    main()
