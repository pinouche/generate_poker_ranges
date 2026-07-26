#!/usr/bin/env python3
"""Turn an arena run into figures.

Reads whatever ``arena.py`` left in ``results/`` and writes one PNG per question:

    01_win_rate.png        did the advisor beat the bots, and by how much
    02_cumulative.png      where that number came from, deal by deal
    03_advice_source.png   which decisions the charts and solves actually answered
    04_by_position.png     the same win rate split by seat
    05_action_profile.png  what the advisor did with its turn
    06_execution_gap.png   how much was lost between the advice and the table

Usage:
    python3 plots.py                       # every run in results/
    python3 plots.py --tags main no_solves
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.ticker import FuncFormatter          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# --- palette ---------------------------------------------------------------
# The validated light-mode categorical order (blue, aqua, orange, violet): worst
# adjacent CVD dE 9.2, worst adjacent normal-vision dE 24.0 on the #fcfcfb surface.
# Aqua sits below 3:1 against that surface, so every mark it fills carries a direct
# label -- the relief rule, not an optional nicety.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7")
POSITIVE, NEGATIVE = "#0ca30c", "#d03b3b"

SOURCE_COLOR = {
    "preflop_chart": SERIES[0],
    "postflop_solve": SERIES[1],
    "heuristic": SERIES[2],
    "preflop_equity": SERIES[3],
    "advisor_error": NEGATIVE,
}
SOURCE_LABEL = {
    "preflop_chart": "preflop chart",
    "postflop_solve": "postflop solve",
    "heuristic": "equity heuristic (fallback)",
    "preflop_equity": "equity, preflop (fallback)",
    "advisor_error": "advisor raised an error",
}
STREETS = ("preflop", "flop", "turn", "river")

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.dpi": 140,
})


def style(ax, xgrid=False, ygrid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", visible=xgrid or ygrid,
            color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    return ax


def title(fig, headline: str, subtitle: str, footnote: str = ""):
    """Headline, subtitle and footnote at fixed *inch* offsets from the figure edge.

    Fraction-of-height offsets collide on a short figure and float away on a tall one --
    these charts range from 2in to 5in, so the spacing has to be absolute.  Returns the
    tight_layout rect that leaves room for what was drawn.
    """
    height = fig.get_figheight()
    fig.text(0.012, 1 - 0.30 / height, headline, ha="left", va="center", fontsize=14,
             color=INK, fontweight="semibold")
    fig.text(0.012, 1 - 0.58 / height, subtitle, ha="left", va="center", fontsize=9.5,
             color=INK_2)
    bottom = 0.0
    if footnote:
        fig.text(0.012, 0.16 / height, footnote, ha="left", va="center", fontsize=8,
                 color=MUTED)
        bottom = 0.34 / height
    return (0, bottom, 1, 1 - 0.80 / height)


# --- loading ---------------------------------------------------------------

class Run:
    def __init__(self, tag: str):
        self.tag = tag
        with open(os.path.join(RESULTS, f"summary_{tag}.json")) as f:
            self.summary = json.load(f)
        self.hands = _read_csv(os.path.join(RESULTS, f"hands_{tag}.csv"))
        self.decisions = _read_csv(os.path.join(RESULTS, f"decisions_{tag}.csv"))
        self.big_blind = self.summary["meta"]["big_blind"]

    @property
    def label(self) -> str:
        return ("advisor with solves" if self.summary["meta"]["solves"]
                else "advisor, solves hidden")

    def deal_scores(self) -> List[float]:
        by_deal = defaultdict(list)
        for h in self.hands:
            by_deal[int(h["deal"])].append(float(h["advisor_chips"]))
        return [statistics.fmean(v) for _, v in sorted(by_deal.items())]


def _read_csv(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_runs(tags: Sequence[str]) -> List[Run]:
    if not tags:
        tags = sorted(os.path.basename(p)[len("summary_"):-len(".json")]
                      for p in glob.glob(os.path.join(RESULTS, "summary_*.json")))
    return [Run(tag) for tag in tags]


# --- 01: the headline ------------------------------------------------------

def interval(values: Sequence[float], big_blind: int) -> dict:
    per_100 = [v / big_blind * 100.0 for v in values]
    n = len(per_100)
    mean = statistics.fmean(per_100) if n else 0.0
    half = (1.96 * statistics.stdev(per_100) / (n ** 0.5)) if n > 1 else float("inf")
    return {"bb_per_100": mean, "ci_half_width": half, "deals": n,
            "significant": bool(n > 1 and abs(mean) > half)}


def paired_difference(runs: Sequence[Run]):
    """What the solves are worth, measured on the same decks rather than by eye.

    Two runs seeded alike see identical decks deal for deal, so the difference can be
    taken per deal.  That cancels the shared card luck a second time and is far tighter
    than asking whether two separate intervals happen to overlap.
    """
    keyed = {run.summary["meta"]["solves"]: run for run in runs}
    if set(keyed) != {True, False}:
        return None
    with_solves, without = keyed[True], keyed[False]
    if (with_solves.summary["meta"]["seed"] != without.summary["meta"]["seed"]
            or with_solves.summary["meta"]["villain_name"]
            != without.summary["meta"]["villain_name"]):
        return None
    a, b = with_solves.deal_scores(), without.deal_scores()
    if len(a) != len(b):
        return None
    return interval([x - y for x, y in zip(a, b)], with_solves.big_blind)


def plot_win_rate(runs: Sequence[Run], path: str):
    rows = [(run.label, run.summary["advisor"], ("beats the bots", "loses to them"))
            for run in runs]
    difference = paired_difference(runs)
    if difference:
        rows.append(("what the solves are worth\n(same decks, paired)", difference,
                     ("the solves help", "the solves hurt")))

    fig, ax = plt.subplots(figsize=(9, 1.5 + 0.95 * len(rows)))
    style(ax, xgrid=True)

    labels = []
    for i, (name, stats, verdicts) in enumerate(reversed(rows)):
        mean, half = stats["bb_per_100"], stats["ci_half_width"]
        colour = POSITIVE if mean > 0 else NEGATIVE
        ax.errorbar(mean, i, xerr=half, fmt="o", color=colour, ecolor=AXIS,
                    elinewidth=2, capsize=5, capthick=2, markersize=11,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        verdict = ("too close to call" if not stats["significant"]
                   else verdicts[0] if mean > 0 else verdicts[1])
        ax.annotate(f"{mean:+.0f} ± {half:.0f} bb/100   ({verdict})",
                    (mean, i), xytext=(0, 15), textcoords="offset points",
                    ha="center", fontsize=10, color=INK, fontweight="semibold")
        labels.append(f"{name}\n{stats['deals']} deals")

    ax.axvline(0, color=AXIS, linewidth=1.5, zorder=1)
    ax.annotate("break even", (0, len(rows) - 0.42), xytext=(6, 0),
                textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(rows)), labels, fontsize=10, color=INK_2)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlabel("big blinds won per 100 hands  (95% confidence interval)")
    ax.tick_params(axis="y", length=0)

    rect = title(fig, "How the advisor did against two heuristic bots",
                 "Three-handed, 100bb deep, every deal played from all three seats so "
                 "card luck cancels.",
                 "Positive = the advisor takes chips off the bots. The interval is the "
                 "range the true win rate plausibly lies in.")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


# --- 02: where the number came from ---------------------------------------

def plot_cumulative(runs: Sequence[Run], path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    style(ax, ygrid=True)

    for run, colour in zip(runs, SERIES):
        scores = run.deal_scores()
        running, total = [], 0.0
        for s in scores:
            total += s / run.big_blind
            running.append(total)
        ax.plot(range(1, len(running) + 1), running, color=colour, linewidth=2,
                label=run.label, solid_capstyle="round")
        if running:
            ax.annotate(f"{running[-1]:+.0f}bb", (len(running), running[-1]),
                        xytext=(8, 0), textcoords="offset points", fontsize=10,
                        color=colour, fontweight="semibold", va="center")

    ax.axhline(0, color=AXIS, linewidth=1.5)
    ax.set_xlabel("duplicate deals played")
    ax.set_ylabel("cumulative big blinds won by the advisor")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}"))
    ax.margins(x=0.06)
    # Above the plot, not inside it: the cumulative line reaches the top left corner
    # about as often as not.
    legend = ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
                       ncol=2, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK_2)

    rect = title(fig, "The result, deal by deal",
                 "A rising line is the advisor winning. A line that wanders around zero "
                 "is a coin flip, however far it drifts.",
                 "Each deal is one deck played three times, once from each seat; the "
                 "plotted step is the advisor's average over those three.")
    fig.tight_layout(rect=(0, rect[1], 1, rect[3] - 0.04))
    fig.savefig(path)
    plt.close(fig)


# --- 03: what actually answered ------------------------------------------

def plot_sources(run: Run, path: str):
    counts: Dict[str, Counter] = {s: Counter() for s in STREETS}
    for d in run.decisions:
        counts[d["street"]][d["source"]] += 1
    streets = [s for s in STREETS if sum(counts[s].values())]

    fig, ax = plt.subplots(figsize=(9, 1.4 + 0.85 * len(streets)))
    style(ax, xgrid=True)
    order = [s for s in SOURCE_COLOR if any(counts[st][s] for st in streets)]

    for row, street in enumerate(reversed(streets)):
        total = sum(counts[street].values())
        left = 0.0
        for source in order:
            n = counts[street][source]
            if not n:
                continue
            share = n / total
            ax.barh(row, share, left=left, height=0.5,
                    color=SOURCE_COLOR[source], edgecolor=SURFACE, linewidth=2,
                    zorder=3)
            # Every segment is labelled: aqua is below 3:1 on this surface, and a
            # share is unreadable off an axis anyway.
            if share > 0.07:
                ax.text(left + share / 2, row, f"{share:.0%}", ha="center",
                        va="center", fontsize=9.5, color="white",
                        fontweight="semibold", zorder=4)
            left += share
        ax.text(-0.012, row, f"{street}\n{total:,} decisions", ha="right",
                va="center", fontsize=10, color=INK_2, linespacing=1.4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SOURCE_COLOR[s]) for s in order]
    grand = sum(sum(counts[st].values()) for st in streets)
    # A source too rare to draw still gets a legend row, but with its count, so the
    # swatch is not a promise of a segment nobody can find.
    labels = [SOURCE_LABEL[s] + (
        f" — {sum(counts[st][s] for st in streets):,} in total"
        if sum(counts[st][s] for st in streets) / grand < 0.02 else "")
        for s in order]
    legend = ax.legend(handles, labels, frameon=False,
                       ncol=min(2, len(order)), fontsize=9.5,
                       loc="lower left", bbox_to_anchor=(0, 1.0))
    for text in legend.get_texts():
        text.set_color(INK_2)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(streets) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("share of the advisor's decisions on that street")

    rect = title(fig, "Which of the advisor's answers came from solved data",
                 "Blue and green are chart and solve lookups. Orange is the equity "
                 "heuristic standing in where they had nothing.",
                 "The river is never solved (it is dumped nowhere), and a flop with all "
                 "three players still in is outside the heads-up solves.")
    fig.tight_layout(rect=(0.14, rect[1], 1, rect[3] - 0.08))
    fig.savefig(path)
    plt.close(fig)


# --- 04: by position -------------------------------------------------------

def plot_positions(runs: Sequence[Run], path: str):
    positions = ("BTN", "SB", "BB")
    fig, ax = plt.subplots(figsize=(9, 5))
    style(ax, ygrid=True)

    width = 0.7 / len(runs)
    for i, (run, colour) in enumerate(zip(runs, SERIES)):
        stats = [run.summary["advisor_by_position"][p] for p in positions]
        xs = [j + (i - (len(runs) - 1) / 2) * width for j in range(len(positions))]
        ax.bar(xs, [s["bb_per_100"] for s in stats], width=width * 0.88,
               color=colour, label=run.label, zorder=3)
        ax.errorbar(xs, [s["bb_per_100"] for s in stats],
                    yerr=[s["ci_half_width"] for s in stats], fmt="none",
                    ecolor=INK_2, elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)
        for x, s in zip(xs, stats):
            # At the tip of the bar, not the tip of the whisker: the whiskers here run
            # off the bottom of the axes and the label would land on the tick text.
            value = s["bb_per_100"]
            # Beside the bar's outer edge, not over its centre: the whisker runs
            # vertically through the centre and a centred label lands on it.
            ax.annotate(f"{value:+.0f}", (x + width * 0.44, value),
                        xytext=(3, 7 if value >= 0 else -14),
                        textcoords="offset points", ha="left", fontsize=9.5,
                        color=INK_2, fontweight="semibold")

    ax.axhline(0, color=AXIS, linewidth=1.5)
    ax.set_xticks(range(len(positions)),
                  ["button\n(acts last postflop)", "small blind\n(acts first postflop)",
                   "big blind"], fontsize=10, color=INK_2)
    ax.set_ylabel("big blinds won per 100 hands")
    ax.tick_params(axis="x", length=0)
    if len(runs) > 1:
        legend = ax.legend(frameon=False, fontsize=10, loc="upper right")
        for text in legend.get_texts():
            text.set_color(INK_2)

    rect = title(fig, "Win rate by seat",
                 "Scored on independent hands rather than paired deals, so the intervals "
                 "are wide -- read the sign, not the number.",
                 "The big blind posts money blind every hand, so a negative number there "
                 "is normal poker, not a broken strategy.")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


# --- 05: what the advisor did ---------------------------------------------

KIND_ORDER = ("FOLD", "CHECK", "CALL", "BET", "RAISE", "ALLIN")
KIND_COLOR = {"FOLD": "#c3c2b7", "CHECK": "#898781", "CALL": SERIES[0],
              "BET": SERIES[2], "RAISE": SERIES[3], "ALLIN": NEGATIVE}


def plot_action_profile(run: Run, path: str):
    counts: Dict[str, Counter] = {s: Counter() for s in STREETS}
    for d in run.decisions:
        counts[d["street"]][d["kind"]] += 1
    streets = [s for s in STREETS if sum(counts[s].values())]

    fig, ax = plt.subplots(figsize=(9, 1.4 + 0.85 * len(streets)))
    style(ax, xgrid=True)
    kinds = [k for k in KIND_ORDER if any(counts[s][k] for s in streets)]

    for row, street in enumerate(reversed(streets)):
        total = sum(counts[street].values())
        left = 0.0
        for kind in kinds:
            share = counts[street][kind] / total
            if not share:
                continue
            ax.barh(row, share, left=left, height=0.5, color=KIND_COLOR[kind],
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            if share > 0.07:
                ax.text(left + share / 2, row, f"{share:.0%}", ha="center",
                        va="center", fontsize=9.5, color="white",
                        fontweight="semibold", zorder=4)
            left += share
        ax.text(-0.012, row, f"{street}\n{total:,} decisions", ha="right",
                va="center", fontsize=10, color=INK_2, linespacing=1.4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR[k]) for k in kinds]
    legend = ax.legend(handles, [k.lower() for k in kinds], frameon=False,
                       ncol=len(kinds), fontsize=9.5, loc="lower left",
                       bbox_to_anchor=(0, 1.0))
    for text in legend.get_texts():
        text.set_color(INK_2)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(streets) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("share of the advisor's decisions on that street")

    rect = title(fig, "What the advisor did with its turn",
                 "The action it played, by street. Preflop folding is the charts being "
                 "selective, not the bot being timid.",
                 "One row per street; a hand contributes a decision to a street each "
                 "time the action came back around to the advisor.")
    fig.tight_layout(rect=(0.14, rect[1], 1, rect[3] - 0.08))
    fig.savefig(path)
    plt.close(fig)


# --- 06: the execution gap ------------------------------------------------

WARNING_LABEL = {
    "board_snapped": "board not in the 186-flop subset",
    "no_solve": "no solve covers this spot",
    "random_hand_equity": "equity measured vs random hands",
    "pot_mismatch": "pot does not match the solved line",
    "never_bluffs": "heuristic never bluffs",
    "limped_pot": "limped pot (charts have no limp branch)",
    "sizing_snapped": "villain's bet is not in the solved tree",
    "hand_resettled": "hand collides with the solved board",
    "turn_resettled": "turn card resettled onto the solved board",
    "close_decision": "close decision",
    "ambiguous_flop_line": "flop line ambiguous from the pot",
    "iso_approximation": "iso-raise range approximated",
    "incomplete_options": "chart frequencies do not sum to 1",
    "stack_mismatch": "stack is not the charts' 100bb",
    "no_preflop_chart": "no preflop chart at all",
    "hu_handwritten_ranges": "heads-up solve, hand-written ranges",
    "weak_limper_assumption": "solve assumes a weak limper",
    "push_fold_deep": "push/fold table used deep",
}


def plot_execution_gap(run: Run, path: str):
    snaps = [float(d["snap_bb"]) for d in run.decisions
             if d["kind"] in ("BET", "RAISE") and d["snap_bb"] != ""]
    # `random_hand_equity` is emitted on every heuristic answer, so it is an exact
    # duplicate of `no_solve` and would plot as a second identical bar.
    warnings = Counter(tag for d in run.decisions
                       for tag in d["warnings"].split("|")
                       if tag and tag != "random_hand_equity")
    top = warnings.most_common(10)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 1.35]})
    style(left, ygrid=True)
    style(right, xgrid=True)

    # Equal-width buckets rather than a log axis: unequal bins on a log scale make the
    # first bar's area meaningless, and the whole point of this panel is the comparison
    # between "played exactly" and the tail.
    edges = [(0.0, 0.05, "exact"), (0.05, 0.5, "<0.5bb"), (0.5, 1.0, "0.5-1bb"),
             (1.0, 2.0, "1-2bb"), (2.0, 4.0, "2-4bb"), (4.0, float("inf"), ">4bb")]
    counts = [sum(1 for s in snaps if lo <= s < hi) for lo, hi, _ in edges]
    total_snaps = max(1, len(snaps))
    left.bar(range(len(edges)), counts, width=0.68, color=SERIES[0], zorder=3)
    for i, n in enumerate(counts):
        if n:
            # Below half a percent the share rounds to "0%", which reads as "none";
            # show the count instead.
            label = f"{n / total_snaps:.0%}" if n / total_snaps >= 0.005 else f"{n}"
            left.annotate(label, (i, n), xytext=(0, 5), textcoords="offset points",
                          ha="center", fontsize=9, color=INK_2)
    left.set_xticks(range(len(edges)), [label for _, _, label in edges], fontsize=9)
    left.tick_params(axis="x", length=0)
    left.set_ylim(0, max(counts) * 1.16)
    left.set_xlabel("distance from the advised bet size")
    left.set_ylabel("bets and raises")
    left.set_title(f"{counts[0] / total_snaps:.0%} of bets went in at exactly the\n"
                   f"size the advisor asked for  ({total_snaps:,} bets)",
                   fontsize=10.5, color=INK_2, loc="left", pad=10)

    labels = [WARNING_LABEL.get(tag, tag) for tag, _ in top]
    values = [n for _, n in top]
    total = len(run.decisions)
    right.barh(range(len(top)), values, height=0.62, color=SERIES[2], zorder=3)
    for i, n in enumerate(values):
        right.annotate(f"{n:,}  ({n / total:.0%})", (n, i), xytext=(6, 0),
                       textcoords="offset points", va="center", fontsize=9,
                       color=INK_2)
    right.set_yticks(range(len(top)), labels, fontsize=9.5, color=INK_2)
    right.invert_yaxis()
    right.tick_params(axis="y", length=0)
    right.spines["left"].set_visible(False)
    right.set_xlim(0, max(values) * 1.28 if values else 1)
    right.set_xlabel(f"decisions carrying that caveat (of {total:,})")
    right.set_title("Every answer says how far to trust it", fontsize=10.5,
                    color=INK_2, loc="left", pad=10)

    rect = title(fig, "The gap between the advice and the table",
                 "Left: the advisor answers in exact chips, the table offers a menu of "
                 "sizes. Right: the caveats it attached to its own answers.",
                 "The big gaps are not a coarse menu: they are advice the table cannot "
                 "legally take -- a raise below the minimum, or a bet smaller than a "
                 "quarter of a pot that already dwarfs the stack.")
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


# --- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tags", nargs="*", default=[],
                        help="run tags to plot (default: everything in results/)")
    parser.add_argument("--out", default=RESULTS)
    args = parser.parse_args()

    runs = load_runs(args.tags)
    if not runs:
        raise SystemExit(f"no runs found in {RESULTS}; run arena.py first")
    # The solve-backed run leads every figure that shows one run.
    runs.sort(key=lambda r: not r.summary["meta"]["solves"])
    os.makedirs(args.out, exist_ok=True)

    written = []
    for name, fn in (
        ("01_win_rate.png", lambda p: plot_win_rate(runs, p)),
        ("02_cumulative.png", lambda p: plot_cumulative(runs, p)),
        ("03_advice_source.png", lambda p: plot_sources(runs[0], p)),
        ("04_by_position.png", lambda p: plot_positions(runs, p)),
        ("05_action_profile.png", lambda p: plot_action_profile(runs[0], p)),
        ("06_execution_gap.png", lambda p: plot_execution_gap(runs[0], p)),
    ):
        path = os.path.join(args.out, name)
        fn(path)
        written.append(path)

    print(f"{len(written)} figures written to {args.out}")
    for path in written:
        print(f"  {os.path.basename(path)}")


if __name__ == "__main__":
    main()
