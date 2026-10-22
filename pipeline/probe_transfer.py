"""Narration-probe readout: how much of each arm's (and the ceiling's) hack suppression survives when
only the narration is rewritten (pipeline/rewrite_narration.py; commands and terminal output unchanged).

Per model M and variant v (original / sanitized / surgery / french), over the same records:
  gap_v(M)      = mean NLL_v(M) - mean NLL_v(noint)    (RETAIN-only, final checkpoint)
  kept_v(M)     = gap_v(M) / gap_original(M)           1 = suppression unaffected by the rewrite,
                                                       0 = it vanishes without the original wording
  transfer_v(M) = gap_v(M) / gap_v(ceiling)            pipeline/transfer.py's score on rewritten copies
Records: the arm's strict test set (data/routing/<arm>.test.json) for condition arms; every probed hack
for the references. F arms compare against F_noint / F_ceiling on reasoned-format copies. 95% CIs
bootstrap over records.

Usage: python probe_transfer.py <dir with losses_<arm>.jsonl and probe_losses_<arm>.jsonl> [--boot 2000]
"""
import argparse, json, random
from pathlib import Path

from transfer import REPO, final_retain

VARIANTS = ("sanitized", "surgery", "french")


def mean(d, ks):
    return sum(d[k] for k in ks) / len(ks)


def readout(M, N, C, keys, boot, rng):
    """M, N, C: {variant: {base key: nll}} for the model, noint and ceiling."""
    keys = [k for k in keys if all(k in d[v] for d in (M, N, C) for v in ("original",) + VARIANTS)]
    if len(keys) < 5:
        return None

    def stats(ks):
        gap = {v: mean(M[v], ks) - mean(N[v], ks) for v in M}
        cgap = {v: mean(C[v], ks) - mean(N[v], ks) for v in M}
        return {v: (gap[v], gap[v] / gap["original"], gap[v] / cgap[v]) for v in M}
    est = stats(keys)
    bs = [stats([rng.choice(keys) for _ in keys]) for _ in range(boot)]
    out = {"n": len(keys)}
    for v in est:
        ci = lambda i: sorted(b[v][i] for b in bs)
        kept, tr = ci(1), ci(2)
        out[v] = {"gap": est[v][0], "kept": est[v][1], "kept_ci": [kept[int(.025 * boot)], kept[int(.975 * boot)]],
                  "transfer": est[v][2], "transfer_ci": [tr[int(.025 * boot)], tr[int(.975 * boot)]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("losses", type=Path)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rng = random.Random(0)
    L = {}
    for p in a.losses.glob("probe_losses_*.jsonl"):
        arm = p.stem.removeprefix("probe_losses_")
        orig = a.losses / f"losses_{arm}.jsonl"
        if not orig.exists():
            continue
        probe = final_retain(p)
        L[arm] = {"original": final_retain(orig)}
        for v in VARIANTS:
            L[arm][v] = {k.rsplit("#", 1)[0]: x for k, x in probe.items() if k.endswith("#" + v)}
    probed = sorted(L.get("noint", {}).get("sanitized", {}))
    out = {}
    print("| model | records | gap, original | " + " | ".join(f"{v}: kept [95% CI] / transfer" for v in VARIANTS) + " |")
    print("|---|--:|--:|" + "---|" * len(VARIANTS))
    for arm in sorted(L):
        fmt = arm.startswith("F_")
        refs = ("F_noint", "F_ceiling") if fmt else ("noint", "ceiling_20pct")
        if arm == refs[0] or not all(r in L for r in refs):
            continue
        tf = REPO / f"data/routing/{arm}.test.json"
        keys = json.loads(tf.read_text()) if arm != refs[1] and tf.exists() else probed
        r = readout(L[arm], L[refs[0]], L[refs[1]], keys, a.boot, rng)
        out[arm] = r
        if not r:
            print(f"| {arm} | - |"); continue
        cells = [f"{r[v]['kept']:.2f} [{r[v]['kept_ci'][0]:.2f}, {r[v]['kept_ci'][1]:.2f}] / {r[v]['transfer']:.2f}"
                 for v in VARIANTS]
        print(f"| {arm} | {r['n']} | {r['original']['gap']:+.3f} | " + " | ".join(cells) + " |")
    (a.losses / "probe_transfer.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
