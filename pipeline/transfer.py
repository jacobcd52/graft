"""Loss-based transfer scores for run 2's condition arms.

For each arm, over the loss-eval records in its strict test set (data/routing/<arm>.test.json):
  transfer = (arm - noint) / (ceiling - noint)
where each term is the mean end-of-training RETAIN-only NLL on those records (no intervention: the
same half-adapter removal). 0 = no more suppression than doing nothing, 1 = as much as the 20%
stratified ceiling. The same is reported on the arm's own routed records for reference, and on its
held-out-task test hacks (never trained on; data/routing/<arm>.test_heldout.json from
labeling/heldout_tests.py), from the full held-out pass (heldout_losses_<arm>.jsonl,
pod/heldout_loss.sh) when available, else from the thin coverage in the main loss-eval set. 95% CIs
bootstrap over records (paired: the same records in every model).

Format arms (F_*) compare against F_noint / F_ceiling; F_reasoned's test records are the
commands-only versions ("#commands") of its never-routed commands-format hacks. Narration arms
(SURG_* / FREN_*, pipeline/narration_pools.py) likewise, with "#surgery" / "#french" copies.

Usage: python transfer.py <dir with losses_<arm>.jsonl> [--boot 2000] > transfer.md
"""
import argparse, json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FAMILIES = {"F_": "#commands", "SURG_": "#surgery", "FREN_": "#french"}  # arm prefix -> test-copy suffix


def final_retain(path):
    rows = [json.loads(l) for l in open(path)]
    last = max(r["step"] for r in rows)
    return {r["key"]: r["nll"] for r in rows if r["step"] == last and r["mode"] == "retain_only"}


def score(arm, noint, ceil, keys, boot, rng):
    keys = [k for k in keys if k in arm and k in noint and k in ceil]
    if len(keys) < 5:
        return None

    def t(ks):
        a = sum(arm[k] for k in ks) / len(ks); n = sum(noint[k] for k in ks) / len(ks)
        c = sum(ceil[k] for k in ks) / len(ks)
        return (a - n) / (c - n) if c != n else float("nan"), a - n, c - n
    est, d_arm, d_ceil = t(keys)
    bs = sorted(t([rng.choice(keys) for _ in keys])[0] for _ in range(boot))
    return {"n": len(keys), "transfer": est, "lo": bs[int(0.025 * boot)], "hi": bs[int(0.975 * boot)],
            "arm_minus_noint": d_arm, "ceil_minus_noint": d_ceil}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("losses", type=Path)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rng = random.Random(0)
    L = {p.stem.removeprefix("losses_"): final_retain(p) for p in a.losses.glob("losses_*.jsonl")}
    # full held-out-task pass (pod/heldout_loss.sh), when present for the arm and its references
    HO = {p.stem.removeprefix("heldout_losses_"): final_retain(p) for p in a.losses.glob("heldout_losses_*.jsonl")}
    out = {}
    print("| arm | routed: transfer [95% CI] (n) | test: transfer [95% CI] (n) | test: arm - noint | test: ceiling - noint "
          "| held-out-task test: transfer [95% CI] (n) |")
    print("|---|---|---|--:|--:|---|")
    for test_file in sorted((REPO / "data/routing").glob("*.test.json")):
        arm = test_file.name.removesuffix(".test.json")
        if arm not in L:
            continue
        pre = next((p for p in FAMILIES if arm.startswith(p)), "")
        fmt = bool(pre)
        refs = (pre + "noint", pre + "ceiling") if fmt else ("noint", "ceiling_20pct")
        noint, ceil = L.get(refs[0]), L.get(refs[1])
        if not noint or not ceil:
            continue
        routed = json.loads((REPO / f"data/routing/{arm}.json").read_text())
        test = json.loads(test_file.read_text())
        if fmt:
            test = [k + FAMILIES[pre] for k in test]
        hf = REPO / f"data/routing/{arm}.test_heldout.json"
        ho = json.loads(hf.read_text()) if hf.exists() else []
        if fmt:
            ho = [k + FAMILIES[pre] for k in ho]
        r = score(L[arm], noint, ceil, routed, a.boot, rng)
        s = score(L[arm], noint, ceil, test, a.boot, rng)
        if arm in HO and all(r in HO for r in refs):
            h = score(HO[arm], HO[refs[0]], HO[refs[1]], ho, a.boot, rng)
        else:
            h = score(L[arm], noint, ceil, ho, a.boot, rng)
        out[arm] = {"routed": r, "test": s, "heldout_test": h}
        f = lambda x: f"{x['transfer']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}] ({x['n']})" if x else "-"
        print(f"| {arm} | {f(r)} | {f(s)} | {s['arm_minus_noint']:+.3f} | {s['ceil_minus_noint']:+.3f} | {f(h)} |"
              if s else f"| {arm} | {f(r)} | - | | | {f(h)} |")
    (a.losses / "transfer.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
