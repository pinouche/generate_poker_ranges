"""Fallback postflop advice for spots the solves cannot answer: equity vs pot odds.

The solved trees are heads-up, flop and turn, five preflop scenarios. Everything real
that falls outside them -- a multiway pot, a river, heads-up play after the third player
busted, a scenario whose solve files are not on disk -- used to be a 422. It lands here
instead, answered from first principles:

  1. Hero's equity is estimated by Monte Carlo: deal every live villain a random hand,
     run the board out, count hero's share of the pot. Random hands overstate hero's
     equity (ranges that bet and call are stronger than random), so the thresholds
     below carry a margin and every answer says so.
  2. That equity is compared with the pot odds (facing a bet) or with the multiway
     baseline 1/players plus a margin (when hero could check).

This is bedrock poker arithmetic, not a strategy: it never bluffs, never balances, and
knows nothing about position or ranges. It exists so a fine question outside the solves
gets a sane, clearly-flagged answer instead of no answer.

Usage:
    python3 resources/python/heuristic_advisor.py state.json
    cat state.json | python3 resources/python/heuristic_advisor.py
"""

import argparse
import json
import random
import sys

from hu_advisor import VILLAINS, busted
from postflop_advisor import RANK_VAL, SUITS, big_blind, card_str, parse_card
from preflop_advisor import TABLE_ORDER, Unsupported, seats_of

# Cards are (rank value, suit) throughout the evaluator; parse_card's (rank char, suit)
# is converted at the boundary in advise().
FULL_DECK = [(v, s) for v in RANK_VAL.values() for s in SUITS]

TRIALS = 800            # +/- ~1.7% at 50% equity; plenty for threshold decisions
CALL_MARGIN = 0.05      # equity above pot odds before calling: the random-villain tax
VALUE_BET_MARGIN = 0.20  # equity above the multiway baseline before betting
RAISE_EQUITY = 0.75     # raise only when hero very likely holds the best hand
BET_FRACTION = 2 / 3
MARGINAL = 0.04         # within this of a threshold, say the decision is close


# ---------------------------------------------------------------------------
# Hand evaluation: one pass over 7 cards, no per-combo enumeration.
# ---------------------------------------------------------------------------

def straight_high(vals):
    """Highest straight top card in a set of rank values, 0 if none. The wheel counts."""
    for high in range(14, 4, -1):
        need = {14, 5, 4, 3, 2} if high == 5 else {high - i for i in range(5)}
        if need <= vals:
            return high
    return 0


def eval7(cards):
    """Best 5-card value of 7 cards as a comparable tuple: (category, tiebreakers...)."""
    counts, by_suit = {}, {}
    for v, s in cards:
        counts[v] = counts.get(v, 0) + 1
        by_suit.setdefault(s, []).append(v)

    flush = next((sorted(vs, reverse=True) for vs in by_suit.values() if len(vs) >= 5),
                 None)
    if flush:
        sf = straight_high(set(flush))
        if sf:
            return (8, sf)

    groups = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    if groups[0][1] == 4:
        quad = groups[0][0]
        return (7, quad, max(v for v in counts if v != quad))
    if groups[0][1] == 3 and groups[1][1] >= 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return (5, *flush[:5])
    straight = straight_high(set(counts))
    if straight:
        return (4, straight)
    if groups[0][1] == 3:
        trips = groups[0][0]
        return (3, trips, *sorted((v for v in counts if v != trips), reverse=True)[:2])
    if groups[0][1] == 2 and groups[1][1] == 2:
        hi, lo = groups[0][0], groups[1][0]
        return (2, hi, lo, max(v for v in counts if v not in (hi, lo)))
    if groups[0][1] == 2:
        pair = groups[0][0]
        return (1, pair, *sorted((v for v in counts if v != pair), reverse=True)[:3])
    return (0, *sorted(counts, reverse=True)[:5])


def equity(hero, board, n_villains, trials=TRIALS):
    """Hero's Monte Carlo pot share vs n random hands, board run out to the river.

    Seeded from the spot so the same request always gets the same answer -- an advisor
    that changes its mind on a refresh reads as broken.
    """
    known = hero + board
    if len(set(known)) != len(known):
        raise Unsupported("duplicate cards between hero's hand and the board")
    deck = [c for c in FULL_DECK if c not in set(known)]
    to_come = 5 - len(board)
    rng = random.Random(f"{sorted(known)}|{n_villains}")

    share = 0.0
    for _ in range(trials):
        draw = rng.sample(deck, 2 * n_villains + to_come)
        runout = board + draw[2 * n_villains:]
        hero_val = eval7(hero + runout)
        best, holders = None, 0
        for i in range(n_villains):
            val = eval7(draw[2 * i:2 * i + 2] + runout)
            if best is None or val > best:
                best, holders = val, 1
            elif val == best:
                holders += 1
        if hero_val > best:
            share += 1.0
        elif hero_val == best:
            share += 1.0 / (holders + 1)
    return share / trials


# ---------------------------------------------------------------------------
# Turning equity into an answer.
# ---------------------------------------------------------------------------

def seat_labels(state):
    """Player -> seat name, for the answer's header.

    Three-handed the dealer fixes everything, same as the chart advisors. With a busted
    villain the game is true heads-up and the dealer IS the small blind, so the
    3-handed clockwise walk would mislabel both players.
    """
    if any(busted(state.get(v)) for v in VILLAINS):
        villain = next(v for v in VILLAINS if not busted(state.get(v)))
        dealer = state.get('dealer')
        if dealer == 'hero':
            return {'hero': 'SB', villain: 'BB'}
        if dealer == villain:
            return {'hero': 'BB', villain: 'SB'}
        raise Unsupported(f"the dealer ({dealer}) has busted; the button cannot be on "
                          f"an empty seat")
    return seats_of(state)


def option(action, kind, chips, frequency):
    return {'action': action, 'kind': kind, 'chips': chips, 'frequency': frequency}


def in_bb(chips, bb):
    return f"{chips / bb:.1f}".rstrip('0').rstrip('.') + 'bb'


def advise(state, bb_override=None, reason=None):
    street = str(state.get('street', '')).lower()
    if street not in ('flop', 'turn', 'river'):
        raise Unsupported(f"the heuristic answers flop/turn/river, not {street!r}")
    if not state['hero'].get('active', True):
        raise Unsupported('hero has folded; nothing to advise')

    villains = [p for p in TABLE_ORDER[1:]
                if state.get(p) and state[p].get('active', True) and not busted(state[p])]
    if not villains:
        raise Unsupported('no live villain; the hand is over')

    bb = big_blind(state, bb_override)
    board = [parse_card(c) for c in state.get('board', [])]
    want = {'flop': 3, 'turn': 4, 'river': 5}[street]
    if len(board) != want:
        raise Unsupported(f"{street} needs {want} board cards, got {len(board)}")
    if len(state['hero'].get('cards', [])) != 2:
        raise Unsupported(f"hero has {len(state['hero'].get('cards', []))} cards, need 2")
    hero_cards = [parse_card(c) for c in state['hero']['cards']]

    to_vals = lambda cards: [(RANK_VAL[r], s) for r, s in cards]
    n = len(villains)
    eq = equity(to_vals(hero_cards), to_vals(board), n)

    pot = float(state.get('pot', 0))
    hero_bet = float(state['hero'].get('bet', 0))
    stack = float(state['hero'].get('stack', 0))
    top_bet = max(float(state[v].get('bet', 0)) for v in villains)
    to_call = max(0.0, top_bet - hero_bet)

    seats = seat_labels(state)
    hand = ''.join(card_str(c) for c in hero_cards)
    board_str = ''.join(card_str(c) for c in board)

    warnings = []
    if reason:
        warnings.append(f"No solve covers this spot ({reason}); the answer below is a "
                        f"Monte Carlo heuristic, not a solved strategy.")
    warnings.append(
        f"Equity is measured against {n} random hand{'s' if n > 1 else ''}. Real "
        f"betting ranges are stronger than random, so treat the equity as an upper "
        f"bound; the thresholds already discount it a little.")

    if to_call > 1e-9:
        need = to_call / (pot + to_call)
        call_chips = min(to_call, stack)
        raise_to = min(3 * top_bet, hero_bet + stack)
        raise_allin = raise_to >= hero_bet + stack - 1e-9
        raise_opt = option(
            f"{'ALL-IN' if raise_allin else 'RAISE to'} {raise_to:g} "
            f"({in_bb(raise_to, bb)})",
            'ALLIN' if raise_allin else 'RAISE', raise_to, 0.0)
        call_opt = option(f"CALL {call_chips:g}", 'CALL', call_chips, 0.0)
        fold_opt = option('FOLD', 'FOLD', None, 0.0)

        if eq >= RAISE_EQUITY and stack > to_call + 1e-9:
            picked, rest = raise_opt, [call_opt, fold_opt]
        elif eq >= need + CALL_MARGIN:
            picked, rest = call_opt, [fold_opt, raise_opt]
        else:
            picked, rest = fold_opt, [call_opt, raise_opt]
        if abs(eq - (need + CALL_MARGIN)) < MARGINAL:
            warnings.append(
                f"Close decision: equity {eq:.0%} vs pot odds {need:.0%} "
                f"(plus the margin) is within a few points of the threshold.")
        facing = f"facing {to_call:g} to call into {pot:g}"
    else:
        baseline = 1.0 / (n + 1)
        bet_chips = min(max(round(BET_FRACTION * pot), bb), stack)
        bet_allin = bet_chips >= stack - 1e-9
        bet_opt = option(
            f"{'ALL-IN' if bet_allin else 'BET'} {bet_chips:g} ({in_bb(bet_chips, bb)})",
            'ALLIN' if bet_allin else 'BET', bet_chips, 0.0)
        check_opt = option('CHECK', 'CHECK', None, 0.0)

        if eq >= baseline + VALUE_BET_MARGIN:
            picked, rest = bet_opt, [check_opt]
        else:
            picked, rest = check_opt, [bet_opt]
            warnings.append(
                "The heuristic never bluffs: it only bets when its equity is well above "
                "the table average, so CHECK here does not rule out a good bluff.")
        if abs(eq - (baseline + VALUE_BET_MARGIN)) < MARGINAL:
            warnings.append(
                f"Close decision: equity {eq:.0%} is within a few points of the "
                f"{baseline + VALUE_BET_MARGIN:.0%} betting threshold.")
        facing = f"no bet to call, pot {pot:g}"

    picked['frequency'] = 1.0
    line = (f"{street} {board_str}: {facing} -- hero equity ~{eq:.0%} "
            f"vs {n} random hand{'s' if n > 1 else ''}")

    return {
        'seats': seats, 'hero_seat': seats['hero'], 'hand': hand, 'bb': bb,
        'street': street, 'equity': eq, 'n_villains': n, 'pot': pot,
        'to_call': to_call, 'action_so_far': line,
        'options': [picked, *rest], 'warnings': warnings,
    }


def report(r):
    seat_str = ', '.join(f"{p.replace('villain_', '')}={seat}"
                         for p, seat in r['seats'].items())
    print(f"Table    : {seat_str}   (bb {r['bb']:g})")
    print(f"Hero     : {r['hand']} in the {r['hero_seat']}")
    print(f"Action   : {r['action_so_far']}")

    print()
    for o in r['options']:
        bar = '#' * int(round(o['frequency'] * 40))
        print(f"  {o['action']:24} {o['frequency'] * 100:5.1f}%  {bar}")

    print(f"\n=> {r['options'][0]['action']}   (heuristic)")
    for warning in r['warnings']:
        print(f"\n! {warning}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('state', nargs='?', help="game-state json; omit to read stdin")
    parser.add_argument('--bb', type=float, help="big blind in chips, if not in the state")
    args = parser.parse_args()

    raw = open(args.state).read() if args.state else sys.stdin.read()
    state = json.loads(raw)

    try:
        report(advise(state, args.bb))
    except Unsupported as e:
        print(f"No heuristic for this spot: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
