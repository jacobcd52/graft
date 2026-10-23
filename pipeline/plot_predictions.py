"""Predicted vs actual red-team agentic transfer per arm (GPT, Opus, the author, actual), with each
predictor's mean squared error. Actual values come from plot_agentic.py's summary (results/run4/plots/summary.txt).
Usage: python plot_predictions.py <summary.txt> <out.png>"""
import ast, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.size": 14})

ARMS = ["Non-security", "Python", "Short vs long", "Sysadmin vs science", "Code vs ops", "Random tasks",
        "Reasoning vs stripped", "English vs French", "Hacky reasoning", "Red team reasoning",
        "No grader tampering", "Serious grader tampering"]
PRED = {  # predictions registered before the agentic results, in ARMS order
    "GPT": [0.55, 0.65, 0.40, 0.55, 0.35, 0.50, 0.80, 0.85, 0.70, 0.40, 0.25, 0.60],
    "Opus": [0.85, 0.90, 0.80, 0.80, 0.80, 0.95, 0.55, 0.85, 0.50, 0.70, 0.70, 0.45],
    "Me": [0.80, 0.50, 0.70, 0.40, 0.40, 0.95, 0.20, 0.80, 0.60, 0.50, 0.30, 0.30],
}
COL = {"GPT": "#2a78d6", "Opus": "#eb6834", "Me": "#1baf7a", "Actual": "#1f1e1b"}
INK, INK2, GRID = "#1f1e1b", "#55544e", "#e6e5e0"


def actual(path):
    out = {}
    for line in open(path):
        if line.startswith("redteam "):
            name = line[len("redteam "):line.index(" {")]
            out[name] = ast.literal_eval(line[line.rindex(" ("):].strip())
    return [out[a] if out[a][0] != "nan" else None for a in ARMS]


def main():
    act = actual(sys.argv[1])
    scored = [i for i, a in enumerate(act) if a]
    mse = {k: sum((v[i] - act[i][0]) ** 2 for i in scored) / len(scored) for k, v in PRED.items()}
    for k, v in mse.items():
        print(f"{k}: MSE {v:.3f} (RMSE {v ** 0.5:.2f}) over {len(scored)} arms")
    print("\narm | actual | mean squared error across the three predictors")
    for i in sorted(scored, key=lambda i: -sum((v[i] - act[i][0]) ** 2 for v in PRED.values())):
        print(f"{ARMS[i]} | {act[i][0]:.2f} | {sum((v[i] - act[i][0]) ** 2 for v in PRED.values()) / 3:.3f} | "
              + ", ".join(f"{k} {v[i]:.2f}" for k, v in PRED.items()))

    fig, ax = plt.subplots(figsize=(15, 6.8))
    w, n = 0.2, len(ARMS)
    lo_lim, hi_lim = -0.5, 2.0
    for i in range(n):
        for j, k in enumerate(("GPT", "Opus", "Me", "Actual")):
            x = i + (j - 1.5) * (w + 0.01)
            if k != "Actual":
                ax.bar(x, PRED[k][i], width=w, color=COL[k], zorder=2)
            elif act[i] is None:
                ax.bar(x, 1, width=w, color="none", edgecolor="#c9c8c1", hatch="///", lw=0, zorder=2)
            else:
                p, lo, hi = act[i]
                ax.bar(x, p, width=w, color=COL[k], zorder=2)
                ax.plot([x, x], [max(lo, lo_lim), min(hi, hi_lim)], color="#8a8980", lw=1.6, zorder=3)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c1")
    ax.tick_params(colors=INK2)
    for i in (5.5, 9.5):
        ax.axvline(i, color="#d8d7d0", lw=0.8, zorder=0)
    ax.set_xticks(range(n), ARMS, rotation=30, ha="right", color=INK)
    ax.set_ylabel("Transfer", color=INK2)
    labels = {"GPT": f"GPT (MSE {mse['GPT']:.2f})", "Opus": f"Opus (MSE {mse['Opus']:.2f})",
              "Me": f"Me (MSE {mse['Me']:.2f})", "Actual": "Actual (95% CI)"}
    ax.legend(handles=[Patch(color=COL[k], label=labels[k]) for k in COL], ncol=4, frameon=False,
              loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.suptitle("Predicted vs actual transfer: red-team prompt", x=0.01, ha="left", fontsize=18,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.015, "Hacky reasoning has no actual transfer (no gap between FLOOR and CEILING), so it is "
             "left out of the MSE.", fontsize=14, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(sys.argv[2], dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
