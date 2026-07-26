#!/usr/bin/env python3
"""Checks on the one thing in here that can be silently wrong: the state conversion.

A mis-seated table does not crash -- it reads a real chart for the wrong seat and plays
it confidently, which is the failure the advisor itself calls "worse than not advising".
So the invariant is checked against the engine rather than assumed:

    the seat the advisor derives  ==  the seat the engine says that player is in

plus the money conventions (`bet` is per-street, `pot` includes the live bets) and the
guarantee that whatever comes back is legal at the table.

    python3 test_bridge.py
"""

from __future__ import annotations

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.environ.get("POKER_ENGINE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))),
    "poker_self_play"))
for path in (HERE, os.path.dirname(HERE), ENGINE):
    if path not in sys.path:
        sys.path.insert(0, path)

from advisor_bridge import AdvisorAgent, advisor_state  # noqa: E402
from arena import BET_FRACTIONS, RAISE_MULTIPLIERS, play_hand  # noqa: E402
from config import EnvConfig, ObsConfig  # noqa: E402
from environment.poker_env import PokerEnv  # noqa: E402
from evaluation.heuristic_agent import HeuristicAgent  # noqa: E402
from preflop_advisor import seats_of  # noqa: E402
from environment.state import action_space_for  # noqa: E402

POSITIONS = ("BTN", "SB", "BB")
CFG = EnvConfig(num_players=3, small_blind=10, big_blind=20, starting_stack=2000,
                bet_fractions=BET_FRACTIONS, raise_multipliers=RAISE_MULTIPLIERS)

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


def test_seat_mapping():
    """Every (dealer, hero) pair must agree with the engine on who sits where."""
    for dealer in range(3):
        for hero in range(3):
            env = PokerEnv(CFG, ObsConfig(), seed=dealer * 3 + hero)
            state = env.reset(dealer=dealer)
            request = advisor_state(state, hero, "preflop")
            seats = seats_of(request)
            for offset in range(3):
                seat = (hero + offset) % 3
                role = ("hero", "villain_left", "villain_right")[offset]
                engine = POSITIONS[state.position_of(seat)]
                check(seats[role] == engine,
                      f"dealer={dealer} hero={hero}: advisor calls {role} "
                      f"{seats[role]}, engine says seat {seat} is {engine}")


def test_blind_posts_line_up():
    """The advisor re-derives the blinds from the bets and rejects a mismatch. If our
    clockwise walk were inverted this is what would catch it, so run it deliberately."""
    for dealer in range(3):
        for hero in range(3):
            env = PokerEnv(CFG, ObsConfig(), seed=99 + dealer)
            state = env.reset(dealer=dealer)
            request = advisor_state(state, hero, "preflop")
            seats = seats_of(request)
            posted = {seats[p]: request[p]["bet"]
                      for p in ("hero", "villain_left", "villain_right")}
            check(posted["SB"] == state.small_blind,
                  f"dealer={dealer} hero={hero}: SB seat posted {posted['SB']}")
            check(posted["BB"] == state.big_blind,
                  f"dealer={dealer} hero={hero}: BB seat posted {posted['BB']}")
            check(posted["BTN"] == 0,
                  f"dealer={dealer} hero={hero}: button posted {posted['BTN']}")


def test_money_conventions():
    """`pot` includes the live bets; `bet` is this street's commitment, not the hand's."""
    env = PokerEnv(CFG, ObsConfig(), seed=7)
    state = env.reset(dealer=0)
    request = advisor_state(state, 0, "preflop")
    check(request["pot"] == 30, f"preflop pot should be 30, got {request['pot']}")

    space = action_space_for(CFG)
    # Button raises, then the blinds call, then look at the flop.
    env.step(space.raise_ids[0])
    while env.state.street == 0 and not env.is_terminal:
        legal = env.legal_actions()
        env.step(2 if legal.is_legal(2) else legal.legal_ids()[0])
    if int(env.state.street) >= 1:
        flop = advisor_state(env.state, 0, "flop")
        check(all(flop[p]["bet"] == 0 for p in
                  ("hero", "villain_left", "villain_right")),
              "street bets should be zero at the start of the flop")
        check(flop["pot"] == sum(p.contributed for p in env.state.players),
              "flop pot should be everything contributed so far")


def test_hero_cards_only():
    """The advisor is handed no villain hole cards, ever."""
    env = PokerEnv(CFG, ObsConfig(), seed=3)
    state = env.reset(dealer=1)
    request = advisor_state(state, 2, "preflop")
    check(len(request["hero"]["cards"]) == 2, "hero should have two cards")
    check(request["villain_left"]["cards"] == [] and
          request["villain_right"]["cards"] == [], "villains must be face down")


def test_every_answer_is_playable():
    """Over a few hundred real decisions, the chosen action is always legal."""
    space = action_space_for(CFG)
    advisor = AdvisorAgent(seed=5)
    villains = [HeuristicAgent(seed=s, action_space=space) for s in (11, 12)]
    rng = random.Random(0)
    decisions = 0
    for hand in range(25):
        env = PokerEnv(CFG, ObsConfig(), seed=1000 + hand)
        advisor.bind(env, hand)
        table = [advisor, *villains]
        play_hand(env, table, rng, dealer=hand % 3)
    for entry in advisor.log:
        decisions += 1
        check(entry["played"] in space.names,
              f"played {entry['played']!r} is not an action")
    check(decisions > 50, f"only {decisions} decisions exercised")
    # play_hand itself raises on an illegal action, so reaching here is the guarantee.


def main():
    for test in (test_seat_mapping, test_blind_posts_line_up, test_money_conventions,
                 test_hero_cards_only, test_every_answer_is_playable):
        before = len(failures)
        test()
        status = "ok" if len(failures) == before else f"{len(failures) - before} FAILED"
        print(f"  {test.__name__:<32} {status}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for message in failures[:20]:
            print(f"  - {message}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
