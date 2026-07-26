#!/usr/bin/env python3
"""Regression check for turn nodes the solver never dumped a strategy for.

This used to be a bug report: ``postflop_advisor.advise`` raised ``KeyError: 'strategy'``
on roughly a third of turn nodes. It is now fixed (``dumped_suit_map`` redirects the turn
onto an interchangeable suit that *was* dumped, with a ``has_strategy`` guard behind it),
so this script exits 0 while that holds and 1 if it regresses.

The stubs themselves are still in the data and always will be -- they are how the solver
writes an isomorphic duplicate, not a mistake -- so the counts below stay the same.

`set_dump_rounds 2` writes the flop and the turn, but the solver resolves some turn
cards by **suit isomorphism** and writes a stub in their place: a node with
``node_type``, ``player``, ``actions`` and ``childrens``, and no ``strategy``.  On a
monotone club flop the whole spade suit is stubbed this way.

``advise`` guards the wrong thing.  It checks that the ``dealcards`` entry exists --

    root = (chance.get('dealcards') or {}).get(card_str(turn_card))
    if not root:
        raise Unsupported(...)

-- and it does exist, stub and all.  Then ``strat = node['strategy']`` blows up, which
over HTTP is a 500 rather than the 422 the module is careful to produce everywhere else.

This walks one solve, counts the stubs, and then puts a real game state through the
public entry point to show the exception.

    python3 reproduce_turn_bug.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import postflop_advisor as postflop  # noqa: E402

SCENARIO, BOARD = "BTN_vs_BB", "Qc_8c_4c"


def count_stubs(node, on_turn=False, seen=None):
    """(turn action nodes, how many of them carry no strategy)."""
    total = missing = 0
    if not isinstance(node, dict):
        return 0, 0
    if node.get("node_type") == "action_node" and on_turn:
        total, missing = 1, int("strategy" not in node)
    for key in ("childrens", "dealcards"):
        for child in (node.get(key) or {}).values():
            a, b = count_stubs(child, on_turn or key == "dealcards")
            total, missing = total + a, missing + b
    return total, missing


def card(rank, suit):
    return {"rank": rank, "suit": suit}


def main():
    path = os.path.join(postflop.SOLVE_BASE, SCENARIO, f"{BOARD}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"no solve at {path} -- this repro needs the solves on disk")

    print(f"solve: {SCENARIO}/{BOARD}.json")
    data = json.load(open(path))
    total, missing = count_stubs(data)
    print(f"  turn action nodes: {total:,}")
    print(f"  without a 'strategy' key: {missing:,}  ({missing / total:.1%})")

    line, chance, _ = postflop.chance_lines(data)[0]
    dumped, stubbed = [], []
    for turn, sub in (chance.get("dealcards") or {}).items():
        (dumped if "strategy" in sub else stubbed).append(turn)
    print(f"\n  after the flop line {postflop.seat_line(line, postflop.SCENARIOS[0])}:")
    print(f"    {len(dumped)} turn cards have a strategy")
    print(f"    {len(stubbed)} do not: {' '.join(sorted(stubbed))}")
    print("    (the flop's own three cards, plus every spade -- the isomorphic suit)")

    # A real spot on one of those turns: BTN opens 2.5bb, SB folds, BB calls, flop is
    # bet 2bb and called, and the turn is a spade.
    state = {
        "hero": {"cards": [card("T", "c"), card("9", "h")],
                 "stack": 1910.0, "bet": 0.0, "active": True},
        "villain_left": {"cards": [], "stack": 1990.0, "bet": 0.0, "active": False},
        "villain_right": {"cards": [], "stack": 1910.0, "bet": 0.0, "active": True},
        "board": [card("Q", "c"), card("8", "c"), card("4", "c"), card("A", "s")],
        "pot": 180.0, "small_blind": 10.0, "big_blind": 20.0,
        "dealer": "hero", "street": "turn", "showdown": False,
    }

    print("\n  putting a turn spot on As through postflop_advisor.advise():")
    try:
        result = postflop.advise(state)
    except KeyError as e:
        print(f"    KeyError: {e}   <- REGRESSED: a stub reached node['strategy'], "
              f"which is a 500, not a 422")
        return 1
    except postflop.Unsupported as e:
        print(f"    Unsupported: {e}")
        print("    <- a clean 422, but the redirect should have answered this one")
        return 1

    action, weight = result["strategy"][0]
    print(f"    answered: {action} at {weight:.1%}")
    redirect = next((w for w in result["warnings"] if w.startswith("Turn suit")), None)
    print(f"    {redirect}" if redirect else
          "    <- answered without a redirect; the solve dumped this suit after all")

    # The redirect is only sound if it lands on a suit that plays the same: As, Ad and Ah
    # are all off-suit aces on a monotone club board, while Ac completes the flush and
    # must NOT match.
    strategies = {}
    for suit in "sdhc":
        probe = json.loads(json.dumps(state))
        probe["board"][3] = card("A", suit)
        strategies[suit] = dict(postflop.advise(probe)["strategy"])
    offsuit = [strategies[s].get("CHECK", 0.0) for s in "sdh"]
    print(f"\n  CHECK frequency by turn suit: " + "  ".join(
        f"A{s} {strategies[s].get('CHECK', 0.0):.3f}" for s in "sdhc"))
    if max(offsuit) - min(offsuit) > 0.01:
        print("    <- the three off-suit aces disagree; the redirect is not isomorphic")
        return 1
    if abs(strategies["c"].get("CHECK", 0.0) - offsuit[0]) < 0.01:
        print("    <- the club ace matches the off-suits; the flush suit got flattened")
        return 1
    print("    the three off-suit aces agree and the club (flush suit) does not: correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
