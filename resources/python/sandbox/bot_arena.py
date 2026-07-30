#!/usr/bin/env python
"""Play the advisor against simple opponent profiles, full hands, three-handed.

arena.py already does this, but it imports the RL project (config, environment,
evaluation) that lives outside this repo, so it cannot run here. This is a self-contained
substitute: a small no-limit engine plus a handful of rule-based opponents, driving the
real advisor through advisor_bridge.consult -- the same charts, solves and heuristic the
HTTP endpoint serves.

What it models, and what it does not:

  * Three seats, everyone reset to the same stack every hand, so an all-in is always for
    the same amount and SIDE POTS CANNOT ARISE. That removes the single most bug-prone
    part of a poker engine at the cost of never testing short stacks.
  * Bets are a fraction of the pot, raises a multiple of the last bet, four bets per
    street maximum. A real table has a continuum; the advisor's requested size is snapped
    to the nearest legal one and the gap is recorded.
  * No rake. Real results at low stakes would be worse.

Duplicate deals: every deal is played three times, with the advisor in each seat and the
same cards on the table each time. Card luck largely cancels between the three, which is
what makes a few thousand hands say anything at all -- independent hands need ~100k.

Usage:
    python3 resources/python/sandbox/bot_arena.py --deals 400
    python3 resources/python/sandbox/bot_arena.py --deals 400 --profile station
"""

from __future__ import annotations

import argparse
import bisect
import functools
import itertools
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import heuristic_advisor as heuristic          # noqa: E402
from advisor_bridge import consult, pick_option  # noqa: E402

BB = 2.0
SB = 1.0
START_STACK = 200.0          # 100bb, matching the charts' depth
SEATS = ("hero", "villain_left", "villain_right")
POT_BETS = (0.5, 0.75, 1.0)  # bet sizes offered, as a fraction of the pot
RAISE_TO = (2.5, 3.5)        # raise sizes offered, as a multiple of the current bet
MAX_RAISES = 4               # bets+raises per street, so the tree cannot run away


# ---------------------------------------------------------------------------
# Cards. The evaluator's (rank value, suit) tuples throughout; json only at the boundary.
# ---------------------------------------------------------------------------

def card_json(card):
    value, suit = card
    return {'rank': heuristic.RANK_CHAR[value], 'suit': suit}


def hand_strength(hole, board):
    """A crude 0-1 strength, good enough to give the profiles distinct taste in hands.

    Postflop it is the made-hand category (high card through straight flush) with a
    little kicker on top. Preflop there is no board, so it is a hand-shape score: pairs
    high, then high cards, with something for suited and connected. Neither is an equity
    calculation -- these are opponents, and giving them the advisor's own evaluator would
    make them better than the players they are meant to imitate.
    """
    if not board:
        (v1, s1), (v2, s2) = hole
        hi, lo = max(v1, v2), min(v1, v2)
        if hi == lo:
            return 0.55 + hi / 40.0
        gap = min(hi - lo - 1, 4)
        return min(0.54, (hi / 14.0) * 0.34 + (0.08 if s1 == s2 else 0.0)
                   + (0.06 - 0.015 * gap) + lo / 100.0)
    category, *kickers = heuristic.eval7(list(hole) + list(board))
    return min(1.0, category / 8.0 + (kickers[0] if kickers else 0) / 400.0)


# ---------------------------------------------------------------------------
# The opponents. Each takes the table view and returns (kind, chips).
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def preflop_ranking():
    """Every hand class ordered by equity against one random hand, best first.

    Built here rather than read from qb_ranges on purpose: an opponent that consults the
    same charts as the advisor is not an independent test of them. Equity vs a random
    hand is a crude way to rank starting hands -- it undervalues suited connectors, which
    play well rather than showing down well -- but it owes the advisor nothing.
    """
    seen, rows = set(), []
    for a, b in itertools.combinations(heuristic.FULL_DECK, 2):
        (v1, s1), (v2, s2) = a, b
        hi, lo = max(v1, v2), min(v1, v2)
        name = (heuristic.RANK_CHAR[hi] + heuristic.RANK_CHAR[lo] if hi == lo else
                f"{heuristic.RANK_CHAR[hi]}{heuristic.RANK_CHAR[lo]}"
                f"{'s' if s1 == s2 else 'o'}")
        if name in seen:
            continue
        seen.add(name)
        rows.append((heuristic.equity([a, b], [], 1), name))
    rows.sort(reverse=True)
    return {name: i / len(rows) for i, (_, name) in enumerate(rows)}


def hand_class_of(hole):
    (v1, s1), (v2, s2) = hole
    hi, lo = max(v1, v2), min(v1, v2)
    if hi == lo:
        return heuristic.RANK_CHAR[hi] * 2
    return (f"{heuristic.RANK_CHAR[hi]}{heuristic.RANK_CHAR[lo]}"
            f"{'s' if s1 == s2 else 'o'}")


@functools.lru_cache(maxsize=4096)
def board_values(board):
    """Sorted made-hand values of every two-card holding on this board."""
    deck = [c for c in heuristic.FULL_DECK if c not in set(board)]
    return sorted(heuristic.eval7(list(h) + list(board))
                  for h in itertools.combinations(deck, 2))


def percentile(hole, board):
    """How strong this hand is as a fraction of everything it could have been. 1 is best.

    Preflop that is a lookup; postflop it is where the made hand ranks among all holdings
    on this board. Percentile is what a pot-odds decision needs -- "top 50%" is a range,
    "0.54" is not.
    """
    if not board:
        return 1.0 - preflop_ranking()[hand_class_of(hole)]
    values = board_values(tuple(board))
    mine = heuristic.eval7(list(hole) + list(board))
    return bisect.bisect_left(values, mine) / len(values)


class Defender:
    """Defends by pot odds and fights back after the flop.

    The other profiles decide with a fixed threshold on hand strength, which makes them
    fold far too much: a big blind getting 3.7-to-1 should continue with about half its
    hands, not with the top fifth. This one converts the price it is being laid into a
    fraction of its range and defends exactly that, then plays a polarised game postflop
    -- value at the top, bluffs from the bottom, the middle checking.

    It exists to answer one question: how much of the advisor's edge over `tag` was real,
    and how much was `tag` folding its blinds away.
    """
    name = 'defender'
    blurb = "defends by pot odds, bluffs and raises postflop"

    # Fraction of range defended is 1 - price * this. At 27% (a 2.5x open into the BB)
    # that is a little over half the hands, which is roughly where solvers put it.
    DEFEND_SLOPE = 1.8
    OPEN = {'BTN': 0.48, 'SB': 0.38, 'BB': 0.40}   # first-in raising ranges
    VALUE_RAISE = 0.88      # top of range raises for value
    BLUFF_RAISE = 0.10      # bottom of range raises as a bluff
    VALUE_BET = 0.62
    BLUFF_BET = 0.16

    def act(self, view, rng):
        pct = percentile(view['hole'], view['board'])
        pot, to_call, stack = view['pot'], view['to_call'], view['stack']
        preflop = not view['board']

        if to_call > 0:
            price = to_call / (pot + to_call)
            keep = max(0.06, 1.0 - price * self.DEFEND_SLOPE)
            if view['can_raise'] and stack > to_call and (
                    pct >= self.VALUE_RAISE or
                    (pct <= self.BLUFF_RAISE and rng.random() < 0.35)):
                return ('RAISE', view['raise_sizes'][0])
            return ('CALL', min(to_call, stack)) if pct >= 1.0 - keep else ('FOLD', 0.0)

        if view['can_raise'] and stack > 0:
            if preflop:
                # First in: open the position's range, everything else folds its turn out.
                if pct >= 1.0 - self.OPEN.get(view['position'], 0.4):
                    return ('BET', min(view['raise_sizes'][0], stack))
                return ('CHECK', 0.0)
            if pct >= self.VALUE_BET or pct <= self.BLUFF_BET:
                return ('BET', min(pot * 0.7, stack))
        return ('CHECK', 0.0)


class Profile:
    """A rule-based opponent: a calling threshold, a betting threshold, a bluff rate."""

    def __init__(self, name, call_at, bet_at, bluff, size=0.75, blurb=""):
        self.name, self.blurb = name, blurb
        self.call_at, self.bet_at, self.bluff, self.size = call_at, bet_at, bluff, size

    def act(self, view, rng):
        strength = hand_strength(view['hole'], view['board'])
        pot, to_call, stack = view['pot'], view['to_call'], view['stack']
        if to_call > 0:
            odds = to_call / (pot + to_call)
            # Raise the hands that are clearly good, call the ones that are priced in.
            if strength >= self.bet_at + 0.12 and view['can_raise'] and stack > to_call:
                return ('RAISE', view['raise_sizes'][0])
            if strength >= self.call_at + odds * 0.5:
                return ('CALL', min(to_call, stack))
            if rng.random() < self.bluff * 0.3 and view['can_raise']:
                return ('RAISE', view['raise_sizes'][0])
            return ('FOLD', 0.0)
        if view['can_raise'] and stack > 0:
            if strength >= self.bet_at or rng.random() < self.bluff:
                return ('BET', min(pot * self.size, stack))
        return ('CHECK', 0.0)


PROFILES = {
    'nit':      Profile('nit', 0.62, 0.72, 0.00, 0.6,
                        "folds everything but premiums, never bluffs"),
    'station':  Profile('station', 0.00, 0.80, 0.00, 0.5,
                        "calls almost anything, almost never raises"),
    'tag':      Profile('tag', 0.50, 0.58, 0.08, 0.7,
                        "tight and aggressive: good hands, bet them"),
    'lag':      Profile('lag', 0.34, 0.45, 0.25, 0.85,
                        "loose and aggressive: wide range, lots of pressure"),
    'maniac':   Profile('maniac', 0.12, 0.20, 0.55, 1.0,
                        "raises and bets almost regardless of cards"),
    'rock':     Profile('rock', 0.55, 0.90, 0.00, 0.5,
                        "calls a fair bit, bets only the nuts"),
    'defender': Defender(),
}


class FoldBot:
    """Control. Folds at the first opportunity, even when checking would be free.

    Folding a free check is not something a player would do, and that is the point: it
    makes the hand end immediately, so the advisor's result is exactly the dead money it
    picks up and nothing that happens after the flop can muddy it.
    """
    name, blurb = 'always-fold', "control: gives up the blinds, never sees a flop"

    def act(self, view, rng):
        return ('FOLD', 0.0)


class AdvisorBot:
    """The system under test: preflop charts, postflop solves, the equity heuristic."""
    name, blurb = 'advisor', "charts + solves + heuristic, via advisor_bridge.consult"

    def __init__(self, mode='sample', seed=0):
        self.mode, self.rng = mode, random.Random(seed)
        self.sources = {}

    def act(self, view, rng):
        advice = consult(view['advisor_state'])
        self.sources[advice.source] = self.sources.get(advice.source, 0) + 1
        option = pick_option(advice.options, self.rng, self.mode)
        return self.snap(option, view)

    def snap(self, option, view):
        """The advisor names a size; the table only offers a few. Take the nearest."""
        to_call, stack, pot = view['to_call'], view['stack'], view['pot']
        kind = option.kind
        if kind == 'FOLD':
            return ('FOLD', 0.0) if to_call > 0 else ('CHECK', 0.0)
        if kind in ('CHECK',):
            return ('CHECK', 0.0) if to_call <= 0 else ('FOLD', 0.0)
        if kind == 'CALL':
            return ('CALL', min(to_call, stack)) if to_call > 0 else ('CHECK', 0.0)
        # A raise or an all-in: snap the requested total to the nearest legal size.
        sizes = view['raise_sizes']
        if not sizes or stack <= to_call:
            return ('CALL', min(to_call, stack)) if to_call > 0 else ('CHECK', 0.0)
        want = option.chips if option.chips else (pot * heuristic.BET_FRACTION + to_call)
        best = min(sizes, key=lambda s: abs(s - want))
        return ('RAISE' if to_call > 0 else 'BET', best)


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------

class Hand:
    def __init__(self, holes, board, button, bots, rng):
        self.holes, self.board_full, self.button = holes, board, button
        self.bots, self.rng = bots, rng
        self.stacks = [START_STACK] * 3
        self.bets = [0.0] * 3
        self.committed = [0.0] * 3
        self.folded = [False] * 3
        self.street = 0                      # 0 preflop, 1 flop, 2 turn, 3 river
        self.pot = 0.0

    # -- helpers ----------------------------------------------------------
    def live(self):
        return [i for i in range(3) if not self.folded[i]]

    def board(self):
        return self.board_full[:(0, 3, 4, 5)[self.street]]

    def post(self, seat, amount):
        amount = min(amount, self.stacks[seat])
        self.stacks[seat] -= amount
        self.bets[seat] += amount
        self.committed[seat] += amount
        self.pot += amount

    def advisor_state(self, seat):
        """The table as the advisor's json, from `seat`'s point of view."""
        order = [seat, (seat + 1) % 3, (seat + 2) % 3]
        players = {}
        for role, s in zip(SEATS, order):
            players[role] = {
                'cards': [card_json(c) for c in self.holes[s]] if s == seat else [],
                'stack': self.stacks[s], 'bet': self.bets[s],
                'active': not self.folded[s],
            }
        return {**players,
                'board': [card_json(c) for c in self.board()],
                'pot': self.pot, 'small_blind': SB, 'big_blind': BB,
                'dealer': SEATS[order.index(self.button)],
                'street': ('preflop', 'flop', 'turn', 'river')[self.street]}

    def view(self, seat, raises_made):
        to_call = max(self.bets) - self.bets[seat]
        can_raise = raises_made < MAX_RAISES and self.stacks[seat] > to_call
        sizes = []
        if can_raise:
            if to_call > 0:
                sizes = [min(max(self.bets) * m, self.bets[seat] + self.stacks[seat])
                         for m in RAISE_TO]
            else:
                sizes = [min(self.pot * f, self.stacks[seat]) for f in POT_BETS]
            sizes = sorted({round(s, 2) for s in sizes
                            if s > to_call + 1e-9 and s <= self.bets[seat] + self.stacks[seat]})
        return {'hole': self.holes[seat], 'board': self.board(), 'pot': self.pot,
                'to_call': to_call, 'stack': self.stacks[seat],
                'can_raise': bool(sizes), 'raise_sizes': sizes,
                'position': ('BTN', 'SB', 'BB')[(seat - self.button) % 3],
                'advisor_state': self.advisor_state(seat)}

    # -- betting ----------------------------------------------------------
    def betting_round(self, first):
        raises_made = 1 if self.street == 0 else 0     # the big blind is a bet
        acted = set()
        seat = first
        guard = 0
        while guard < 60:
            guard += 1
            active = [i for i in self.live() if self.stacks[i] > 0]
            if len(self.live()) < 2:
                return
            to_call = max(self.bets) - self.bets[seat]
            done = (seat in acted and to_call <= 1e-9)
            if not self.folded[seat] and self.stacks[seat] > 0 and not done:
                kind, chips = self.bots[seat].act(self.view(seat, raises_made), self.rng)
                if kind == 'FOLD':
                    self.folded[seat] = True
                elif kind in ('BET', 'RAISE'):
                    self.post(seat, chips - self.bets[seat])
                    raises_made += 1
                    acted = {seat}          # a raise reopens the action for everyone else
                elif kind == 'CALL':
                    self.post(seat, to_call)
                acted.add(seat)
            else:
                acted.add(seat)
            if all(i in acted for i in self.live()) and \
                    len({round(self.bets[i], 6) for i in self.live()
                         if self.stacks[i] > 0}) <= 1:
                return
            if not active:
                return
            seat = (seat + 1) % 3

    def play(self):
        self.post((self.button + 1) % 3, SB)
        self.post((self.button + 2) % 3, BB)
        first = self.button                       # 3-handed, the button acts first preflop
        for street in range(4):
            self.street = street
            if len(self.live()) < 2:
                break
            if street > 0:
                self.bets = [0.0] * 3
                first = (self.button + 1) % 3
                if len([i for i in self.live() if self.stacks[i] > 0]) < 2:
                    continue                      # everyone all-in: run it out, no betting
            self.betting_round(first)
        return self.settle()

    def settle(self):
        live = self.live()
        if len(live) == 1:
            winners = live
        else:
            board = self.board_full
            best = {i: heuristic.eval7(list(self.holes[i]) + list(board)) for i in live}
            top = max(best.values())
            winners = [i for i in live if best[i] == top]
        share = self.pot / len(winners)
        result = [-self.committed[i] for i in range(3)]
        for w in winners:
            result[w] += share
        return result


def play_deal(holes, board, button, bots, rng):
    return Hand(holes, board, button, bots, rng).play()


# ---------------------------------------------------------------------------
# Running a match.
# ---------------------------------------------------------------------------

def run(profile_name, deals, seed=0, mode='sample', hero_name='advisor'):
    """Hero vs two copies of one profile, with hero rotated through all seats.

    `hero_name` is 'advisor' or one of the profiles. Running a plain profile as hero is
    the baseline the advisor has to beat: a win rate only means something next to what a
    dozen lines of if-statements score on the same deals.
    """
    rng = random.Random(seed)
    advisor = (AdvisorBot(mode=mode, seed=seed) if hero_name == 'advisor'
               else PROFILES[hero_name])
    if not hasattr(advisor, 'sources'):
        advisor.sources = {}
    villain = PROFILES[profile_name] if profile_name in PROFILES else FoldBot()

    per_hand, chips_checked = [], True
    for deal in range(deals):
        cards = rng.sample(heuristic.FULL_DECK, 11)
        holes = [tuple(cards[0:2]), tuple(cards[2:4]), tuple(cards[4:6])]
        board = tuple(cards[6:11])
        button = deal % 3
        for advisor_seat in range(3):
            bots = [villain] * 3
            bots[advisor_seat] = advisor
            result = play_deal(holes, board, button, bots, random.Random(seed + deal))
            if abs(sum(result)) > 1e-6:
                chips_checked = False
            per_hand.append(result[advisor_seat])
    return per_hand, advisor, chips_checked


def self_play_check(deals, seed=0):
    """Three identical deterministic bots must sum to exactly zero across the rotations.

    Each deal is replayed with the bot under test in each of the three seats. If all three
    seats play the same deterministic strategy, the three replays ARE the same hand, so
    the three results are that hand's three payoffs and they have to cancel. Any pot
    misallocation, any chip conjured or lost in the betting, shows up here as a non-zero
    total -- a much sharper test than "the numbers look plausible".
    """
    rng = random.Random(seed)
    worst = 0.0
    for deal in range(deals):
        cards = rng.sample(heuristic.FULL_DECK, 11)
        holes = [tuple(cards[0:2]), tuple(cards[2:4]), tuple(cards[4:6])]
        board = tuple(cards[6:11])
        bot = PROFILES['tag']                     # stateless and deterministic
        total = 0.0
        for seat in range(3):
            result = play_deal(holes, board, deal % 3, [bot] * 3, random.Random(7))
            total += result[seat]
        worst = max(worst, abs(total))
    return worst


def bb_per_100(results):
    n = len(results)
    mean = statistics.fmean(results) / BB * 100
    if n < 2:
        return mean, float('inf')
    sd = statistics.stdev(results) / BB * 100
    return mean, 1.96 * sd / (n ** 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--deals', type=int, default=300,
                        help="deals; each is played 3x, once per advisor seat")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--profile', default=None, help="just one profile")
    parser.add_argument('--mode', default='sample', choices=('sample', 'max'))
    parser.add_argument('--hero', default='advisor',
                        help="'advisor' (default) or a profile name, to get a baseline")
    args = parser.parse_args()

    drift = self_play_check(min(args.deals, 200), args.seed)
    print(f"\nengine check: worst chip drift over identical-bot rotations {drift:.2e} "
          f"({'ok' if drift < 1e-6 else 'BROKEN -- results below are meaningless'})")

    names = [args.profile] if args.profile else list(PROFILES) + ['always-fold']
    print(f"\n'{args.hero}' vs each profile -- {args.deals} deals x 3 seats = "
          f"{args.deals * 3} hands each, 100bb, no rake\n")
    print(f"{'opponent':<12}{'bb/100':>9}{'95% CI':>9}{'sig':>5}   "
          f"{'chart':>6}{'solve':>6}{'heur':>6}   profile")
    for name in names:
        results, advisor, ok = run(name, args.deals, args.seed, args.mode, args.hero)
        mean, ci = bb_per_100(results)
        blurb = (PROFILES[name].blurb if name in PROFILES else FoldBot.blurb)
        total = max(1, sum(advisor.sources.values()))
        pct = lambda *keys: sum(advisor.sources.get(k, 0) for k in keys) / total
        flag = 'yes' if abs(mean) > ci else 'no'
        print(f"{name:<12}{mean:>+9.1f}{ci:>9.1f}{flag:>5}   "
              f"{pct('preflop_chart', 'headsup_table'):>6.0%}{pct('postflop_solve'):>6.0%}"
              f"{pct('heuristic', 'advisor_error', 'preflop_equity'):>6.0%}   {blurb}"
              f"{'' if ok else '   CHIPS LOST'}")


if __name__ == '__main__':
    main()
