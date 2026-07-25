#!/usr/bin/env python
"""Play any two bots against each other, straight from the command line.

The environment is three-handed (``config.py`` -> ``EnvConfig.num_players``),
so a "head to head" is really the hero in one seat against two copies of the
villain, which is how ``evaluation/evaluate.py`` runs its own suites.  Pass
``--third`` if you want three different bots at the table instead.

Examples
--------
    python sandbox/play_bots.py tight_aggressive calling_station
    python sandbox/play_bots.py heuristic random --hands 3000
    python sandbox/play_bots.py tight_aggressive loose_passive --paired --deals 1000
    python sandbox/play_bots.py checkpoint:checkpoints/latest.pt tight_aggressive
    python sandbox/play_bots.py heuristic random --third calling_station

Everything under sandbox/ is gitignored, so edit this file freely.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable no matter where this is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from environment.state import action_space_for
from evaluation.evaluate import (
    bb_per_100_interval,
    duplicate_deal_scores,
    evaluate_match,
    format_results,
    make_network_agent,
)
from evaluation.heuristic_agent import HeuristicAgent, loose_passive, tight_aggressive
from evaluation.random_agent import (
    AlwaysFoldAgent,
    CallingStationAgent,
    RandomAgent,
)

# --- bot registry ----------------------------------------------------------
# Each builder takes (seed, action_space) and returns an Agent.  The heuristic
# bots need the action space so their sizing preferences line up with whatever
# bet abstraction the config declares.
BOTS = {
    "random": lambda seed, space: RandomAgent(),
    "calling_station": lambda seed, space: CallingStationAgent(),
    "always_fold": lambda seed, space: AlwaysFoldAgent(),
    "heuristic": lambda seed, space: HeuristicAgent(seed=seed, action_space=space),
    "tight_aggressive": lambda seed, space: tight_aggressive(seed=seed, action_space=space),
    "loose_passive": lambda seed, space: loose_passive(seed=seed, action_space=space),
}

CHECKPOINT_PREFIX = "checkpoint:"


def bot_names() -> str:
    return ", ".join(sorted(BOTS)) + f", or {CHECKPOINT_PREFIX}<path/to/x.pt>"


def load_config(specs, device: str) -> Config:
    """Use a checkpoint's own config if one is in play, else the defaults.

    A trained network only makes sense under the observation encoding and bet
    abstraction it was trained with, so the checkpoint's config wins.
    """
    for spec in specs:
        if spec and spec.startswith(CHECKPOINT_PREFIX):
            from model.network import load_checkpoint

            _, cfg, _ = load_checkpoint(spec[len(CHECKPOINT_PREFIX) :], device=device)
            return cfg
    return Config()


def build_bot(spec: str, cfg: Config, seed: int, device: str):
    """Turn a CLI name into an Agent."""
    if spec.startswith(CHECKPOINT_PREFIX):
        from model.network import load_checkpoint

        path = spec[len(CHECKPOINT_PREFIX) :]
        network, _, _ = load_checkpoint(path, device=device)
        name = os.path.basename(path).removesuffix(".pt")
        # temperature 0 = deterministic (argmax) play, the standard eval setting.
        return make_network_agent(network, cfg, temperature=0.0, device=device, name=name)

    if spec not in BOTS:
        raise SystemExit(f"unknown bot {spec!r}. available: {bot_names()}")
    return BOTS[spec](seed, action_space_for(cfg.env))


# --- run modes -------------------------------------------------------------
def run_match(hero, villain, third, cfg, hands: int, seed: int) -> None:
    """Straight match, hands split evenly across all three seat rotations."""
    table = [hero, villain, third if third is not None else villain]
    results = {f"{hero.name} vs {villain.name}": evaluate_match(
        table, cfg, num_hands=hands, seed=seed
    )}
    print(format_results(results))


def run_paired(hero, villain, third, cfg, deals: int, seed: int) -> None:
    """Duplicate-deal scoring: same decks, hero rotated through every seat.

    Cancels most of the card-luck variance, so a real edge shows up in far
    fewer hands than an independent-hands bb/100 needs.
    """
    opponents = [villain, third if third is not None else villain]
    deal_seeds = list(range(seed, seed + deals))

    hero_scores = duplicate_deal_scores(hero, opponents, cfg, deal_seeds)
    hero_ci = bb_per_100_interval(hero_scores, cfg.env.big_blind)

    # Same deals with the roles swapped, so both numbers are directly comparable.
    villain_scores = duplicate_deal_scores(
        villain, [hero, third if third is not None else hero], cfg, deal_seeds
    )
    villain_ci = bb_per_100_interval(villain_scores, cfg.env.big_blind)

    print(f"\nduplicate deals: {hero_ci['deals']} decks x 3 seat rotations each")
    print(f"  {'agent':<20}{'bb/100':>10}{'+/- 95%':>10}   significant")
    for agent, ci in ((hero, hero_ci), (villain, villain_ci)):
        print(
            f"  {agent.name:<20}{ci['bb_per_100']:>10.1f}{ci['ci_half_width']:>10.1f}"
            f"   {'yes' if ci['significant'] else 'no'}"
        )
    print("\n(each row is that bot in one seat against two copies of the other)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play two bots against each other.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"bots: {bot_names()}",
    )
    parser.add_argument("hero", help="bot in the hero seat")
    parser.add_argument("villain", help="bot filling the other two seats")
    parser.add_argument("--third", default=None, help="different bot for the third seat")
    parser.add_argument("--hands", type=int, default=900, help="hands to play (default 900)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="torch device for checkpoints")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="duplicate-deal scoring with confidence intervals instead of a plain match",
    )
    parser.add_argument(
        "--deals", type=int, default=500, help="deals for --paired (default 500)"
    )
    args = parser.parse_args()

    cfg = load_config([args.hero, args.villain, args.third], args.device)
    # Distinct seeds so two instances of the same bot are not lockstep clones.
    hero = build_bot(args.hero, cfg, args.seed, args.device)
    villain = build_bot(args.villain, cfg, args.seed + 1, args.device)
    third = build_bot(args.third, cfg, args.seed + 2, args.device) if args.third else None

    if args.paired:
        run_paired(hero, villain, third, cfg, args.deals, args.seed + 1)
    else:
        run_match(hero, villain, third, cfg, args.hands, args.seed)


if __name__ == "__main__":
    main()
