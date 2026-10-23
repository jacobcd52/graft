"""Blog plots of the agentic results under one elicitation prompt (red team or nudge), hack = judge
score >= 0.5. Same data, quantities and bootstrap as pipeline/elicit_table.py, except Serious grader
tampering uses the conjunctive p(hack and never reads or tampers) instead of p(hack).

  transfer_<elicit>.png  transfer = (FLOOR - NARROW) / (FLOOR - CEILING) per arm, 95% CI
  rates_<elicit>.png     FLOOR / NARROW / CEILING rate per arm, 95% Jeffreys interval

Arms whose models are not all judged yet get a hatched placeholder bar.
Usage: python plot_agentic.py <dir with jobs/> <out dir> [--boot 2000]
"""
import argparse, collections, random, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentic_props import category_tasks  # noqa: E402
from elicit_table import load  # noqa: E402

hack = lambda r: r["hack"]
# (label, NARROW model, (FLOOR, CEILING), denominator category arm or None, numerator, conjunctive)
ARMS = [
    ("Non-security", "T3_nonsecurity", None, "T3_nonsecurity", hack, False),
    ("Python", "T5_python_only", None, "T5_python_only", hack, False),
    ("Short vs long", "L1_short", None, "L1_short", hack, False),
    ("Sysadmin vs science", "D1_sysadmin_to_science", None, "D1_sysadmin_to_science", hack, False),
    ("Code vs ops", "K1_code_to_ops", None, "K1_code_to_ops", hack, False),
    ("Random tasks", "R1_random_tasks", None, "R1_random_tasks", hack, False),
    ("Reasoning vs stripped", "F_reasoned", ("ablate_F_noint", "F_ceiling"), None, hack, False),
    ("Hacky reasoning", "SURG_routed", ("ablate_SURG_noint", "SURG_ceiling"), None, hack, False),
    ("English vs French", "FREN_routed", ("ablate_FREN_noint", "FREN_ceiling"), None, hack, False),
    ("Red team reasoning", "H9_redteam", None, None, lambda r: r["hack"] and not r["rt"], True),
    ("No grader tampering", "A_make_it_up", None, None, lambda r: r["hack"] and r["pt"], True),
    ("Serious grader tampering", "V1_severe_to_mild", None, None, lambda r: r["hack"] and not r["pt"] and not r["tamper"], True),
]
GROUPS = [(0, 5, "Task"), (6, 9, "Reasoning"), (10, 11, "Hack style")]
_rng = random.Random(1)  # Jeffreys interval by sampling (no scipy on the box)
COL = {"FLOOR": "#2a78d6", "NARROW": "#eb6834", "CEILING": "#1baf7a"}
INK, INK2, MUTED, GRID = "#1f1e1b", "#55544e", "#8a8980", "#e6e5e0"
TITLE = {"redteam": "red-team prompt", "nudge": "nudge prompt"}


def counts(data, model, den, num):
    cc = collections.defaultdict(lambda: [0, 0])
    for r in data.get(model, []):
        if den(r):
            cc[r["task"]][1] += 1
            cc[r["task"]][0] += bool(num(r))
    return cc


def summarize(data, rng, boot):
    out = []
    for label, arm, refs, cat_arm, num, conj in ARMS:
        fl, ce = refs or ("floor", "ceiling")
        cat = category_tasks(cat_arm) if cat_arm else None
        den = (lambda r: r["task"] in cat) if cat else (lambda r: True)
        c = {k: counts(data, m, den, num) for k, m in (("FLOOR", fl), ("NARROW", arm), ("CEILING", ce))}
        rates = {k: (sum(v[0] for v in cc.values()), sum(v[1] for v in cc.values())) if cc else None
                 for k, cc in c.items()}
        tr = None
        if all(c.values()):
            tasks = sorted(set().union(*c.values()))
            rate = lambda cc, ts: (sum(cc[t][0] for t in ts if t in cc), sum(cc[t][1] for t in ts if t in cc))
            f = lambda q: (q["FLOOR"] - q["NARROW"]) / (q["FLOOR"] - q["CEILING"]) if q["FLOOR"] != q["CEILING"] else float("nan")
            point = f({k: v[0] / v[1] for k, v in rates.items()})
            if rates["FLOOR"][0] / rates["FLOOR"][1] <= rates["CEILING"][0] / rates["CEILING"][1]:
                point = float("nan")  # no FLOOR-CEILING gap to transfer
            bs = []
            for _ in range(boot):
                s = [rng.choice(tasks) for _ in tasks]
                q = {}
                for k in c:
                    a, n = rate(c[k], s)
                    q[k] = rng.betavariate(a + 0.5, n - a + 0.5) if n else float("nan")
                bs.append(f(q))
            bs = sorted(x for x in bs if x == x)
            tr = (point, bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]) if bs and point == point else ("nan",)
        out.append((label, conj, rates, tr))
    return out


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c1")
    ax.tick_params(colors=INK2)


def frame(ax, n, ylab):
    style(ax)
    ax.axvspan(8.5, n - 0.5, color="#f4f3ee", zorder=0)
    for i in (5.5, 9.5):
        ax.axvline(i, color="#d8d7d0", lw=0.8, zorder=0)
    for a, b, g in GROUPS:
        ax.text((a + b) / 2, 1.02, g, transform=ax.get_xaxis_transform(), ha="center", fontsize=10, color=INK2)
    ax.text((8.5 + n - 0.5) / 2, 0.97, "conjunctive: p(hack ∧ invisible type)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.5, color=INK2, style="italic")
    ax.text((-0.5 + 8.5) / 2, 0.97, "conditional: p(hack | invisible task / forced style)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.5, color=INK2, style="italic")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylabel(ylab, color=INK2)


def plot_transfer(res, elicit, path):
    n = len(res)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    frame(ax, n, "Transfer")
    lo_lim, hi_lim = -1.0, 2.5
    for i, (label, conj, rates, tr) in enumerate(res):
        if tr is None or tr[0] == "nan":
            ax.bar(i, 1, width=0.6, color="none", edgecolor="#c9c8c1", hatch="///", lw=0)
            ax.text(i, 1.05, "pending" if tr is None else "no FLOOR–CEILING gap", ha="center", fontsize=8, color=MUTED)
            continue
        p, lo, hi = tr
        ax.bar(i, max(min(p, hi_lim), lo_lim), width=0.6, color="#2a78d6", zorder=2)
        ax.plot([i, i], [max(lo, lo_lim), min(hi, hi_lim)], color=INK, lw=1.6, zorder=3)
        for v, lim, mk in ((lo, lo_lim, "v"), (hi, hi_lim, "^")):
            if (v < lim) if mk == "v" else (v > lim):
                ax.plot(i, lim, marker=mk, color=INK, ms=6, zorder=3)
        y = min(max(p, lo_lim), hi_lim)
        ax.text(i - 0.32, y + (0.04 if p >= 0 else -0.04), f"{p:.2f}", fontsize=8.5, color=INK, ha="right",
                va="bottom" if p >= 0 else "top", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8), zorder=4)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(n - 0.45, 1.02, "CEILING", fontsize=8, color=INK2, va="bottom", ha="right")
    ax.text(n - 0.45, 0.02, "FLOOR", fontsize=8, color=INK2, va="bottom", ha="right")
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_xticks(range(n), [r[0] for r in res], rotation=30, ha="right", color=INK)
    fig.suptitle(f"Transfer to the invisible set, {TITLE[elicit]} (hack = judge score ≥ 0.5)", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.text(0.01, 0.01, "Transfer = (FLOOR − NARROW) / (FLOOR − CEILING). Lines: 95% CI (task bootstrap × Jeffreys posterior); "
             "arrows: CI runs past the axis. Hatched: not all three models judged yet, or FLOOR does not hack more than CEILING.\nHacky reasoning shows p(hack): "
             "covert narration can't be forced and almost never occurs.", fontsize=8.5, color="#6b6a64")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)


def plot_rates(res, elicit, path):
    n = len(res)
    fig, ax = plt.subplots(figsize=(14, 5.8))
    frame(ax, n, "Hack rate on the invisible set")
    w = 0.26
    top = 0.0
    for i, (label, conj, rates, tr) in enumerate(res):
        for j, k in enumerate(("FLOOR", "NARROW", "CEILING")):
            x = i + (j - 1) * (w + 0.02)
            if rates[k] is None:
                ax.bar(x, 0.05, width=w, color="none", edgecolor=COL[k], hatch="///", lw=0, zorder=2)
                continue
            a, m = rates[k]
            p = a / m
            d = sorted(_rng.betavariate(a + 0.5, m - a + 0.5) for _ in range(4000))
            lo, hi = d[100], d[3899]
            top = max(top, hi)
            ax.bar(x, p, width=w, color=COL[k], zorder=2)
            ax.plot([x, x], [lo, hi], color=INK, lw=1.1, zorder=3)
    ax.set_ylim(0, max(0.12, min(1.0, top * 1.12)) + 0.08 * max(0.12, top))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks(range(n), [r[0] for r in res], rotation=30, ha="right", color=INK)
    handles = [Patch(color=COL[k], label=k) for k in COL] + [Patch(facecolor="none", edgecolor=MUTED, hatch="///", label="pending")]
    ax.legend(handles=handles, ncol=4, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.06))
    fig.suptitle(f"Hacking on the invisible set, {TITLE[elicit]} (hack = judge score ≥ 0.5)", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.text(0.01, 0.01, "Lines: 95% Jeffreys interval. Reasoning arms use their own family's FLOOR / CEILING (trained on the "
             "modified data); Reasoning vs stripped and English vs French force the invisible style.\nHacky reasoning "
             "shows p(hack). All models have FORGET ablated.", fontsize=8.5, color="#6b6a64")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for elicit in ("redteam", "nudge"):
        res = summarize(load(a.dir, elicit), random.Random(0), a.boot)
        for label, conj, rates, tr in res:
            print(elicit, label, rates, tr)
        plot_transfer(res, elicit, a.out / f"transfer_{elicit}.png")
        plot_rates(res, elicit, a.out / f"rates_{elicit}.png")


if __name__ == "__main__":
    main()
