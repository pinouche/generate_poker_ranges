#!/usr/bin/env python3
"""Play the advisor against two heuristic bots, three-handed, full hands.

The table is the ``poker_self_play`` engine (a sibling repo: real no-limit rules, side
pots, all-in EV runouts).  One seat is the advisor from ``resources/python`` driven
through ``advisor_bridge``; the other two are copies of that project's equity-threshold
``HeuristicAgent``.  Hands run preflop to showdown, and every hero decision is a live
question to the charts and the solves.

Two things are done to keep the comparison honest:

* **Stacks reset to 100bb every hand.**  The charts are a 100bb game; letting stacks
  carry over would drift the table to depths the pack explicitly warns it cannot answer,
  and the win rate would then be measuring the warning rather than the strategy.
* **Every deck is played three times, once with the advisor in each chair.**  The button
  is worth real money three-handed, and one 100bb hand has a standard deviation around
  35bb, so scoring independent hands would need ~100,000 of them to see anything.  See
  ``run_match``.

Usage
-----
    python3 arena.py --hands 300
    python3 arena.py --hands 900 --villain tight_aggressive --mode max
    python3 arena.py --hands 60 --no-solves        # skip the 96GB of solves
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
# .../<projects>/generate_poker_ranges/resources/python/sandbox -> .../<projects>
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ENGINE = os.environ.get("POKER_ENGINE",
                        os.path.join(os.path.dirname(REPO_ROOT), "poker_self_play"))

for path in (HERE, ENGINE):
    if path not in sys.path:
        sys.path.insert(0, path)

if not os.path.isdir(ENGINE):
    raise SystemExit(
        f"the poker engine is not where this expects it: {ENGINE}\n"
        f"clone poker_self_play next to generate_poker_ranges, or set ENGINE by hand.")

from advisor_bridge import AdvisorAgent  # noqa: E402

from config import Config, EnvConfig, ObsConfig      # noqa: E402
from environment.poker_env import PokerEnv           # noqa: E402
from evaluation.heuristic_agent import (             # noqa: E402
    HeuristicAgent,
    loose_passive,
    tight_aggressive,
)
from environment.state import action_space_for       # noqa: E402

POSITIONS = ("BTN", "SB", "BB")
RESULTS = os.path.join(HERE, "results")

VILLAINS = {
    "heuristic": lambda seed, space: HeuristicAgent(seed=seed, action_space=space),
    "tight_aggressive": lambda seed, space: tight_aggressive(seed=seed, action_space=space),
    "loose_passive": lambda seed, space: loose_passive(seed=seed, action_space=space),
}

# A finer bet abstraction than the engine's default three-and-three.
#
# The advisor answers in continuous chips -- a 2.5bb open, an 11bb 3bet, a 33% flop stab
# -- and every one of those has to land on a menu entry.  With the default menu (2x/3x/4x
# raises) a 2.5bb open is 25% away from the nearest option it can actually play, which
# would make the results a measurement of the abstraction rather than of the strategy.
# These sizings cover the ones the charts and the solves actually use; what is left over
# is reported as `snap_bb` per decision.
BET_FRACTIONS = (0.25, 0.33, 0.50, 0.66, 0.75, 1.00, 1.50)
RAISE_MULTIPLIERS = (2.0, 2.2, 2.5, 3.0, 3.5, 4.4, 6.0)


def build_config(args) -> Config:
    big_blind = args.big_blind
    env = EnvConfig(
        num_players=3,
        small_blind=big_blind // 2,
        big_blind=big_blind,
        starting_stack=int(round(args.stack_bb * big_blind)),
        bet_fractions=BET_FRACTIONS,
        raise_multipliers=RAISE_MULTIPLIERS,
        # The button is pinned to seat 0 by run_match (that is what makes the three
        # replays of a deal the same deal); the advisor gets every position by moving
        # seats instead, so there is nothing left for the engine to rotate.
        rotate_dealer=False,
        # Settle all-in pots on the expectation over the remaining boards instead of one
        # random runout.  Identical in expectation, far less variance -- which is the
        # difference between a readable win rate at 600 hands and one at 6,000.
        all_in_ev_runout=not args.random_runouts,
    )
    return Config(env=env, obs=ObsConfig())


# ---------------------------------------------------------------------------
# One hand.
# ---------------------------------------------------------------------------

def play_hand(env: PokerEnv, agents: Sequence[object], rng: random.Random,
              dealer: Optional[int] = None) -> dict:
    """Deal and play a hand to the end, returning what happened."""
    env.reset(dealer=dealer)
    decisions = 0
    while not env.is_terminal:
        seat = env.to_act
        observation = env.get_observation(seat)
        mask = observation["legal_action_mask"]
        if mask.sum() <= 0:
            raise RuntimeError(f"seat {seat} has no legal action")
        choice = agents[seat].act(observation, None, mask, rng)
        if not mask[choice.action]:
            raise RuntimeError(f"{agents[seat].name} chose an illegal action")
        env.step(choice.action)
        decisions += 1

    final = env.state
    return {
        "dealer": final.dealer,
        "street_reached": int(final.street),
        "showdown": bool(final.went_to_showdown),
        "board": " ".join(str(c) for c in final.board),
        "chip_deltas": env.chip_deltas(),
        "decisions": decisions,
        "hole": {seat: " ".join(str(c) for c in p.hole)
                 for seat, p in enumerate(final.players)},
        "positions": {seat: POSITIONS[final.position_of(seat)]
                      for seat in range(final.num_players)},
    }


# ---------------------------------------------------------------------------
# The match.
# ---------------------------------------------------------------------------

def reseed(agent, seed: int) -> None:
    """Put an agent's own RNG on a known stream.

    Both the advisor (sampling a mixed strategy) and the heuristic bots (a bluff
    frequency) draw from ``self.rng``, which ``reset()`` rebuilds from ``self.seed``.
    """
    agent.seed = seed
    agent.reset()


def run_match(cfg: Config, advisor: AdvisorAgent, villains: Sequence[object],
              hands: int, seed: int, verbose_every: int = 60) -> List[dict]:
    """Duplicate deals: every deck is played three times, advisor in each seat.

    Independent hands are hopeless here.  A 100bb no-limit hand has a standard deviation
    of ~35bb, so a 95% interval of +/-20bb/100 needs on the order of 100,000 hands, and
    each hand costs a chart lookup and sometimes an 86MB solve.  Replaying the same deck
    with the advisor rotated through all three chairs removes the "who was dealt aces"
    term entirely: three identical strategies would score exactly zero on every deal, so
    what survives is the difference between the strategies and nothing else.

    The button stays on seat 0 for all three replays, which is what makes them the same
    deal; rotating the advisor is what gives it every position.
    """
    rng_seed = seed
    deals = max(1, hands // cfg.env.num_players)
    records: List[dict] = []
    started = time.time()
    hand_id = 0

    for deal in range(deals):
        deal_seed = seed * 7919 + deal
        for advisor_seat in range(cfg.env.num_players):
            table = list(villains)
            table.insert(advisor_seat, advisor)
            # A fresh env per replay, seeded by the deal: the deck order is a function of
            # the seed alone (cards come off it in a fixed order), so all three replays
            # see the same hands and the same board.
            env = PokerEnv(cfg.env, cfg.obs, seed=deal_seed)
            # Same random draws in all three replays -- that is what makes them the same
            # deal -- but a different stream per deal.  Reseeding to a constant instead
            # would make every hand's first mixed decision take the same coin flip.
            reseed(advisor, deal_seed * 3)
            for i, villain in enumerate(villains):
                reseed(villain, deal_seed * 3 + 1 + i)
            advisor.bind(env, hand_id)
            result = play_hand(env, table, random.Random(rng_seed * 31 + hand_id),
                               dealer=0)
            deltas = result["chip_deltas"]
            records.append({
                "hand": hand_id,
                "deal": deal,
                "advisor_seat": advisor_seat,
                "advisor_position": result["positions"][advisor_seat],
                "dealer_seat": result["dealer"],
                "street_reached": result["street_reached"],
                "showdown": result["showdown"],
                "board": result["board"],
                "advisor_hole": result["hole"][advisor_seat],
                "advisor_chips": deltas[advisor_seat],
                "villain_chips": sum(d for s, d in enumerate(deltas)
                                     if s != advisor_seat),
                "decisions": result["decisions"],
                **{f"seat{s}_chips": d for s, d in enumerate(deltas)},
            })
            hand_id += 1
        if verbose_every and hand_id % verbose_every == 0:
            elapsed = time.time() - started
            print(f"  {hand_id:5d} hands  {elapsed:6.1f}s "
                  f"({elapsed / hand_id:.2f}s/hand)", flush=True)
    return records


# ---------------------------------------------------------------------------
# Summarising.
# ---------------------------------------------------------------------------

def bb_per_100(chips: Sequence[float], big_blind: int) -> Dict[str, float]:
    """Mean win rate with a 95% normal-approximation interval, in bb/100 hands."""
    per_100 = [c / big_blind * 100.0 for c in chips]
    n = len(per_100)
    mean = statistics.fmean(per_100) if n else 0.0
    half = (1.96 * statistics.stdev(per_100) / (n ** 0.5)) if n > 1 else float("inf")
    return {
        "hands": n,
        "bb_per_100": mean,
        "ci_half_width": half,
        "significant": bool(n > 1 and abs(mean) > half),
    }


def deal_scores(records: Sequence[dict]) -> List[float]:
    """The advisor's chip result per deal, averaged over the three seats it played it in.

    This is the paired number: the same cards, the same board, only the seat differs.
    """
    by_deal: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        by_deal[r["deal"]].append(r["advisor_chips"])
    return [statistics.fmean(v) for _, v in sorted(by_deal.items())]


def summarise(records: Sequence[dict], decisions: Sequence[dict],
              cfg: Config, meta: dict) -> dict:
    big_blind = cfg.env.big_blind
    advisor_chips = [r["advisor_chips"] for r in records]
    # Three-handed poker is zero sum, so the two villains together are the mirror image
    # of the advisor; per-villain-seat is the fair per-agent number.
    villain_chips = [d for r in records for s, d in
                     ((0, r["seat0_chips"]), (1, r["seat1_chips"]), (2, r["seat2_chips"]))
                     if s != r["advisor_seat"]]

    by_position = {
        position: bb_per_100([r["advisor_chips"] for r in records
                              if r["advisor_position"] == position], big_blind)
        for position in POSITIONS
    }

    sources = Counter(d["source"] for d in decisions)
    by_street = defaultdict(Counter)
    for d in decisions:
        by_street[d["street"]][d["source"]] += 1

    kinds = Counter(d["kind"] for d in decisions)
    warnings = Counter(tag for d in decisions
                       for tag in d["warnings"].split("|") if tag)
    snaps = [d["snap_bb"] for d in decisions if d["kind"] in ("BET", "RAISE")]
    substituted = sum(1 for d in decisions if d["substituted"])
    mixed = sum(1 for d in decisions if not d["pure"])

    paired = bb_per_100(deal_scores(records), big_blind)
    paired["deals"] = paired.pop("hands")

    return {
        "meta": meta,
        # The headline. `advisor_unpaired` is the same result scored on independent
        # hands, kept only to show how much the pairing bought.
        "advisor": paired,
        "advisor_unpaired": bb_per_100(advisor_chips, big_blind),
        "villain_per_seat": bb_per_100(villain_chips, big_blind),
        "advisor_by_position": by_position,
        "showdown_rate": statistics.fmean([float(r["showdown"]) for r in records]),
        "street_reached": dict(Counter(
            ("preflop", "flop", "turn", "river", "showdown")[r["street_reached"]]
            for r in records)),
        "decisions": {
            "total": len(decisions),
            "by_source": dict(sources),
            "by_source_share": {k: v / max(1, len(decisions))
                                for k, v in sources.items()},
            "by_street": {street: dict(counts) for street, counts in by_street.items()},
            "by_kind": dict(kinds),
            "mixed_node_share": mixed / max(1, len(decisions)),
            "substituted_kind": substituted,
            "sizing_snap_bb": {
                "n": len(snaps),
                "mean": statistics.fmean(snaps) if snaps else 0.0,
                "median": statistics.median(snaps) if snaps else 0.0,
                "max": max(snaps) if snaps else 0.0,
            },
        },
        "warnings": dict(warnings),
    }


def write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(summary: dict) -> str:
    advisor, unpaired = summary["advisor"], summary["advisor_unpaired"]
    lines = [
        "",
        f"{summary['meta']['hands']} hands over {advisor['deals']} duplicate deals, "
        f"{summary['meta']['stack_bb']:g}bb deep, advisor in every seat",
        "",
        f"  {'scoring':<22}{'n':>7}{'bb/100':>10}{'+/- 95%':>10}   significant",
        f"  {'advisor, paired':<22}{advisor['deals']:>7}"
        f"{advisor['bb_per_100']:>10.1f}{advisor['ci_half_width']:>10.1f}"
        f"   {'yes' if advisor['significant'] else 'no'}",
        f"  {'advisor, independent':<22}{unpaired['hands']:>7}"
        f"{unpaired['bb_per_100']:>10.1f}{unpaired['ci_half_width']:>10.1f}"
        f"   {'yes' if unpaired['significant'] else 'no'}",
        "",
        f"  (paired is the headline: the same deal played from all three seats, so card "
        f"luck cancels.\n   A positive number is chips taken off the "
        f"{summary['meta']['villain_name']} bots.)",
        "",
        "  advisor by position (independent hands, so wide):",
    ]
    for position, stats in summary["advisor_by_position"].items():
        lines.append(f"    {position:<20}{stats['hands']:>7}{stats['bb_per_100']:>10.1f}"
                     f"{stats['ci_half_width']:>10.1f}")

    decisions = summary["decisions"]
    lines += ["", f"  {decisions['total']} advisor decisions, answered by:"]
    for source, count in sorted(decisions["by_source"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    {source:<20}{count:>7}  {count / decisions['total']:>6.1%}")
    snap = decisions["sizing_snap_bb"]
    lines += [
        "",
        f"  mixed nodes: {decisions['mixed_node_share']:.1%} of decisions",
        f"  sizing snap: mean {snap['mean']:.2f}bb, median {snap['median']:.2f}bb, "
        f"max {snap['max']:.2f}bb over {snap['n']} bets/raises",
        f"  kind substituted (engine could not offer it): "
        f"{decisions['substituted_kind']}",
        "",
        "  warnings raised (a decision may raise several):",
    ]
    for tag, count in sorted(summary["warnings"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    {tag:<26}{count:>7}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hands", type=int, default=300,
                        help="hands to play, split over three seats (default 300)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--villain", default="heuristic", choices=sorted(VILLAINS),
                        help="which heuristic bot fills the other two seats")
    parser.add_argument("--mode", default="sample", choices=("sample", "max"),
                        help="'sample' plays the chart's mixed frequencies, 'max' always "
                             "takes its most common action (default sample)")
    parser.add_argument("--stack-bb", type=float, default=100.0,
                        help="starting stack in big blinds; the charts are 100bb")
    parser.add_argument("--big-blind", type=int, default=20)
    parser.add_argument("--random-runouts", action="store_true",
                        help="deal one random board on an all-in instead of paying the "
                             "expectation over every completion (noisier, not wrong)")
    parser.add_argument("--no-solves", action="store_true",
                        help="hide the postflop solves, so postflop falls back to the "
                             "equity heuristic -- a fast run, and the control for how "
                             "much the solves are worth")
    parser.add_argument("--out", default=RESULTS, help="results directory")
    parser.add_argument("--tag", default="", help="suffix for the output filenames")
    args = parser.parse_args()

    if args.no_solves:
        # Point the solve lookup at nothing.  Set on the module rather than in the
        # environment because SOLVE_BASE is read at import time and the bridge -- and
        # with it postflop_advisor -- is already imported by now.
        import postflop_advisor
        postflop_advisor.SOLVE_BASE = os.path.join(args.out, ".no-solves")

    # One crash log per run, cleared first: a log that mixes two runs cannot be counted,
    # and being countable against the run summary is its whole use.
    import advisor_bridge
    advisor_bridge.CRASH_LOG = os.path.join(
        args.out, f"advisor_crashes{'_' + args.tag if args.tag else ''}.jsonl")
    if os.path.isfile(advisor_bridge.CRASH_LOG):
        os.remove(advisor_bridge.CRASH_LOG)

    cfg = build_config(args)
    space = action_space_for(cfg.env)
    advisor = AdvisorAgent(name=f"advisor[{args.mode}]", mode=args.mode, seed=args.seed)
    villains = [VILLAINS[args.villain](args.seed + 1 + i, space) for i in range(2)]

    print(f"advisor ({args.mode}) vs 2x {args.villain}, {args.hands} hands, "
          f"{args.stack_bb:g}bb deep")
    started = time.time()
    records = run_match(cfg, advisor, villains, args.hands, args.seed)
    elapsed = time.time() - started

    meta = {
        "hands": len(records),
        "seed": args.seed,
        "villain_name": args.villain,
        "advisor_name": advisor.name,
        "mode": args.mode,
        "stack_bb": args.stack_bb,
        "big_blind": cfg.env.big_blind,
        "solves": not args.no_solves,
        "all_in_ev_runout": cfg.env.all_in_ev_runout,
        "bet_fractions": list(BET_FRACTIONS),
        "raise_multipliers": list(RAISE_MULTIPLIERS),
        "seconds": round(elapsed, 1),
    }
    summary = summarise(records, advisor.log, cfg, meta)

    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    write_csv(os.path.join(args.out, f"hands{tag}.csv"), records)
    write_csv(os.path.join(args.out, f"decisions{tag}.csv"), advisor.log)
    with open(os.path.join(args.out, f"summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(report(summary))
    print(f"\nwritten to {args.out}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
