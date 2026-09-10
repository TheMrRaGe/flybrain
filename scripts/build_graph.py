#!/usr/bin/env python3
"""
build_graph.py - turn the raw male CNS connectome into a simulation-ready graph.

Pipeline decisions are printed as they are made.

Input  (raw, from https://male-cns.janelia.org/download/ , CC-BY 4.0):
  body-annotations-*.feather      neuron identity: type / class / superclass
  body-neurotransmitters-*.feather consensus neurotransmitter -> synapse SIGN
  connectome-weights-*.feather     152M segment->segment edges, weight = synapse count

Output: flybrain_graph.npz
  bodyId[N] type[N] superclass[N] nt[N] sign[N]   node table
  pre[E] post[E] w[E]                             edges as *indices into the node table*
"""
import argparse, gc, sys
import numpy as np, pyarrow as pa, pyarrow.feather as pf

# Fly synapse sign. ACh excites; GABA and glutamate (via GluCl-alpha) inhibit;
# histamine inhibits (photoreceptor -> LMC). Modulators carry no fast sign, so 0.
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
        "dopamine": 0, "octopamine": 0, "serotonin": 0, "unclear": 0, "unknown": 0}

def log(m): print(m, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir holding the three .feather files")
    ap.add_argument("--weights", required=True, help="path to connectome-weights feather")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-weight", type=int, default=5,
                    help="drop connections with fewer than N synapses (default 5)")
    ap.add_argument("--drop-optic-lobe", action="store_true",
                    help="drop ol_* neurons: 58%% of the brain, all vision")
    a = ap.parse_args()

    log(f"[1/5] neurons: reading annotations")
    ann = pf.read_table(f"{a.data}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                        columns=["bodyId","type","superclass","status"]).to_pandas()
    n0 = len(ann)
    ann = ann[(ann["status"] == "Traced") & ann["type"].notna()]
    log(f"      {n0:,} segments -> {len(ann):,} traced, typed neurons "
        f"(dropped orphans, glia, untyped fragments)")

    if a.drop_optic_lobe:
        m = ann["superclass"].astype(str).str.startswith("ol_")
        log(f"      --drop-optic-lobe: removing {m.sum():,} ol_* neurons "
            f"({100*m.mean():.0f}% of the brain)")
        ann = ann[~m]

    ann = ann.sort_values("bodyId").reset_index(drop=True)
    ids = ann["bodyId"].to_numpy(dtype="int64")
    log(f"      node table: {len(ids):,} neurons")

    log(f"[2/5] signs: mapping consensus neurotransmitter -> +1/-1/0")
    nt = pf.read_table(f"{a.data}/body-neurotransmitters-male-cns-v1.0.feather",
                       columns=["body","consensus_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]
    ntv  = nt.reindex(ids).fillna("unknown").to_numpy().astype(str)
    sign = np.array([SIGN.get(x, 0) for x in ntv], dtype="int8")
    log(f"      excitatory {int((sign>0).sum()):,} | inhibitory {int((sign<0).sum()):,} "
        f"| modulatory/unknown {int((sign==0).sum()):,}")

    log(f"[3/5] edges: streaming {a.weights} in record batches (RAM-bounded)")
    def to_index(x):                      # bodyId -> row in the node table, -1 if absent
        i = np.searchsorted(ids, x)
        np.clip(i, 0, len(ids)-1, out=i)
        return np.where(ids[i] == x, i, -1).astype("int32")

    rdr = pa.ipc.open_file(pa.memory_map(a.weights))
    P, Q, W, raw = [], [], [], 0
    for b in range(rdr.num_record_batches):
        bt = rdr.get_batch(b)
        w = bt.column("weight").to_numpy(zero_copy_only=False)
        raw += len(w)
        k = w >= a.min_weight                        # cheap filter first
        if not k.any():
            del bt, w; continue
        pre  = to_index(bt.column("body_pre").to_numpy(zero_copy_only=False)[k].astype("int64"))
        post = to_index(bt.column("body_post").to_numpy(zero_copy_only=False)[k].astype("int64"))
        ok = (pre >= 0) & (post >= 0)
        P.append(pre[ok]); Q.append(post[ok]); W.append(w[k][ok].astype("int32"))
        del bt, w, k, pre, post, ok
        if b % 20 == 0: gc.collect()

    pre = np.concatenate(P); post = np.concatenate(Q); w = np.concatenate(W)
    del P, Q, W; gc.collect()
    log(f"      {raw:,} raw edges -> {len(pre):,} kept "
        f"(weight>={a.min_weight}, both endpoints real neurons)")

    log(f"[4/5] writing {a.out}")
    np.savez(a.out, bodyId=ids, type=ann["type"].to_numpy().astype(str),
             superclass=ann["superclass"].to_numpy().astype(str),
             nt=ntv, sign=sign, pre=pre, post=post, w=w)

    log(f"[5/5] done")
    sc = ann["superclass"].astype(str)
    log(f"      sensory in : {int(sc.str.contains('sensory').sum()):,} neurons")
    log(f"      descending : {int((sc=='descending_neuron').sum()):,} neurons "
        f"({ann.loc[sc=='descending_neuron','type'].nunique():,} distinct types) = action bus")
    log(f"      motor out  : {int(sc.str.contains('motor').sum()):,} neurons")

if __name__ == "__main__":
    main()
