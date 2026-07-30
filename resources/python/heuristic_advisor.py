"""Fallback postflop advice for spots the solves cannot answer: equity vs pot odds.

The solved trees are heads-up, flop and turn, five preflop scenarios. Everything real
that falls outside them -- a multiway pot, a river, heads-up play after the third player
busted, a scenario whose solve files are not on disk -- used to be a 422. It lands here
instead, answered from first principles:

  1. Hero's equity vs random hands. From the turn on this is counted exactly -- the
     space left is small enough to walk (see exact_turn, exact_river). On the flop it is
     a Monte Carlo estimate instead: deal every live villain a random hand, run the board
     out, count hero's share of the pot. Random hands overstate hero's equity (ranges
     that bet and call are stronger than random), so the thresholds below carry a margin
     and every answer says so.
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
import functools
import itertools
import json
import os
import random
import sys

from hu_advisor import VILLAINS, busted
from postflop_advisor import RANK_VAL, SUITS, big_blind, card_str, parse_card
from preflop_advisor import CHART_BASE, TABLE_ORDER, Unsupported, seats_of

# Cards are (rank value, suit) throughout the evaluator; parse_card's (rank char, suit)
# is converted at the boundary in advise().
FULL_DECK = [(v, s) for v in RANK_VAL.values() for s in SUITS]
RANK_CHAR = {v: r for r, v in RANK_VAL.items()}

TRIALS = 800            # +/- ~1.7% at 50% equity; plenty for threshold decisions
# Equity depends only on the cards and the villain count, and one turn is often asked
# about several times -- hero bets, gets raised, decides again -- at ~0.14s each. Bounded
# so a server left running does not hold every spot it has ever been shown; a few thousand
# entries covers a session's worth of hands and costs a few hundred KB.
EQUITY_CACHE = 4096
CALL_MARGIN = 0.05      # equity above pot odds before calling: the random-villain tax
VALUE_BET_MARGIN = 0.20  # equity above the multiway baseline before betting
RAISE_EQUITY = 0.75     # raise only when hero very likely holds the best hand
BET_FRACTION = 2 / 3
MARGINAL = 0.04         # within this of a threshold, say the decision is close

# PROTOTYPE. Equity against uniformly random hands is the heuristic's largest error by
# far -- measured 3 to 8 points against a real range, where the sampling it replaces is
# 1.7 -- and, worse, it has no fixed sign. A chart's callers have both folded their trash
# and 3-bet their premiums, so the range is capped as well as narrowed: hero's hands that
# beat only junk lose value (JTs on a missed board, -4 to -6), while hands with showdown
# value often GAIN it because the premiums that crushed them are not in the range (ace
# high on a low board, +8). Which way a made hand moves depends on how the range hits the
# board. A single margin cannot correct a bias that changes direction, which is what
# CALL_MARGIN is currently asked to do.
#
# Weighting villain hands by a real preflop range fixes the model rather than the margin.
# Off by default because it changes every postflop answer and because CONTINUING_RANGE is
# a coarse stand-in (see below); set USE_RANGE_WEIGHTS to turn it on.
USE_RANGE_WEIGHTS = True
# The qb_ranges pack has a file per preflop node, but naming the right one needs the
# preflop line -- exactly what a heuristic called on the flop does not have. This one
# range stands in for "a villain who continued preflop". It is 37% of hand classes, and
# being roughly right about that beats being exactly wrong about "any two cards".
CONTINUING_RANGE = os.path.join('BB', 'BTN_2.5bb_BB_Call.txt')

# What a bet says about the range making it. A player firing into a pot is not doing it
# with everything: they bet the hands that want money in, plus some that cannot win a
# showdown, and check the middle -- and the bigger the bet, the more polarised that split.
# (top share betting for value, bottom share bluffing), by bet size relative to the pot.
BET_SHAPES = {'small': (0.45, 0.10), 'medium': (0.32, 0.14), 'large': (0.22, 0.20)}
SMALL_BET = 0.4          # of pot; at or under this, the 'small' shape
LARGE_BET = 0.8          # over this, the 'large' shape
MIDDLE_BET_FREQ = 0.15   # the middle of a range still bets sometimes, so never zero it


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


@functools.lru_cache(maxsize=16)
def load_range(name):
    """A qb_ranges chart as {hand class: weight}: one line of 'AA:1.0,AKs:0.75,...'.

    The same files preflop_advisor reads, used the other way round -- not "what should
    hero do with this hand" but "which hands is the villain here holding at all".
    """
    path = os.path.join(CHART_BASE, name)
    if not os.path.isfile(path):
        raise Unsupported(f"no range chart at {path}")
    with open(path) as f:
        text = f.read().strip()
    try:
        return {k.strip(): float(v) for k, v in
                (pair.split(':') for pair in text.split(',') if pair.strip())}
    except ValueError as bad:
        raise Unsupported(f"malformed range chart {name}: {bad}")


def combo_weight(weights, hand, cache):
    """How much of the villain's range this exact two-card combo is.

    Charts are written per hand class, so every combo of a class carries that class's
    weight; the cache is because a turn asks this ~45k times for 169 distinct answers.
    """
    (v1, s1), (v2, s2) = hand
    hi, lo = (v1, v2) if v1 >= v2 else (v2, v1)
    key = (hi, lo, s1 == s2)
    weight = cache.get(key)
    if weight is None:
        name = (RANK_CHAR[hi] + RANK_CHAR[lo] if hi == lo else
                f"{RANK_CHAR[hi]}{RANK_CHAR[lo]}{'s' if key[2] else 'o'}")
        weight = cache[key] = weights.get(name, 0.0)
    return weight


def bet_shape(to_call, street_pot):
    """Which BET_SHAPES entry a bet of this size implies, by fraction of the pot bet into."""
    fraction = to_call / street_pot if street_pot > 1e-9 else LARGE_BET + 1
    return 'small' if fraction <= SMALL_BET else 'medium' if fraction <= LARGE_BET else 'large'


def narrow_by_bet(hands, board, shape):
    """Re-weight (hand, weight) pairs by what a bet of this size says about them.

    Ranked by how the hand stands RIGHT NOW, on the board the bet was made into: the top
    of the range bets for value, the bottom bets as a bluff, and the middle mostly checks.
    Nothing is zeroed -- a range with a hole in it is a worse model than a soft one, and
    the middle does keep betting sometimes.

    Ranking by made strength puts draws near the bottom, so they come out weighted as
    bluffs. That is roughly where a semi-bluff belongs, but it is luck rather than design:
    this model knows nothing about equity that has not arrived yet.
    """
    value_share, bluff_share = BET_SHAPES[shape]
    ranked = sorted(hands, key=lambda hw: eval7(list(hw[0]) + board), reverse=True)
    total = sum(weight for _, weight in ranked)
    if total <= 0:
        raise Unsupported("the villain's range is empty on this board")

    narrowed, below = {}, 0.0
    for hand, weight in ranked:
        share = below / total          # how much of the range is stronger than this hand
        below += weight
        if share < value_share or share > 1.0 - bluff_share:
            narrowed[hand] = weight
        else:
            narrowed[hand] = weight * MIDDLE_BET_FREQ
    return narrowed


def build_combo_weights(deck, board, class_weights, shape=None):
    """{two-card combo: weight} for one villain, judged on the board as it stands now.

    Built once per spot rather than once per runout. The villain bet what they had on
    THIS street, so the narrowing has to be judged on this board -- re-ranking their
    range against every possible river would credit them with cards they had not seen.
    """
    cache, hands = {}, []
    for hand in itertools.combinations(deck, 2):
        weight = 1.0 if class_weights is None else combo_weight(class_weights, hand, cache)
        if weight > 0.0:
            hands.append((hand, weight))
    if not hands:
        raise Unsupported("no hand in the villain's range survives the cards on show")
    return narrow_by_bet(hands, board, shape) if shape else dict(hands)


def weighted_stats(hands):
    """(total weight, total squared weight, and both again per card) for (hand, weight)s."""
    total = squares = 0.0
    per_card, squares_per_card = {}, {}
    for hand, weight in hands:
        total += weight
        squares += weight * weight
        for card in hand:
            per_card[card] = per_card.get(card, 0.0) + weight
            squares_per_card[card] = squares_per_card.get(card, 0.0) + weight * weight
    return total, squares, per_card, squares_per_card


def pairs_within(hands):
    """Sum of w1*w2 over unordered pairs of DISJOINT hands drawn from one set.

    The unweighted form of this is C(n,2) - sum of C(d_c,2): all pairs, less the pairs
    sharing a card, counted per card because two distinct two-card hands can share at
    most one. Weights change what is being summed, not that identity -- occurrence counts
    become weight sums, and the C(d,2) terms become (S_c^2 - Q_c)/2.
    """
    total, squares, per_card, squares_per_card = weighted_stats(hands)
    sharing = sum(s * s - squares_per_card[c] for c, s in per_card.items())
    return (total * total - squares - sharing) / 2


def pairs_across(left, right):
    """Same sum for one hand from each set. The two sets must share no hand."""
    left_total, _, left_cards, _ = weighted_stats(left)
    right_total, _, right_cards, _ = weighted_stats(right)
    return left_total * right_total - sum(w * right_cards.get(c, 0.0)
                                          for c, w in left_cards.items())


def exact_river(hero, board, n_villains, deck, combos=None):
    """Hero's exact pot share vs n villain hands on a complete board. n is 1 or 2.

    A river has nothing left to deal, so the villains' hands ARE the sample space --
    C(45,2)=990 of them -- and hero's own seven-card value is a constant, evaluated once
    rather than once per trial. That makes counting the whole space cheaper than the 800
    trials it replaces, so there is no accuracy/cost trade to make here.

    Two villains would be ~450k disjoint pairs to walk, which is the one slow way to do
    this. It is not needed: two DISTINCT two-card hands overlap iff they share exactly
    one card (sharing both would make them the same hand), so the pair sums follow from
    per-card totals alone (see pairs_within). That keeps card removal between the
    villains exact at O(deck) arithmetic.

    `combos` is {two-card combo: weight} from build_combo_weights -- how much of the
    villain's range each exact hand is, after their preflop range and their bet have both
    been taken into account. None weights every hand equally: the vs-random-hands model.
    """
    hero_val = eval7(hero + board)
    live, beat, tied = [], [], []   # in range; of those, hands hero beats / ties
    for hand in itertools.combinations(deck, 2):
        weight = 1.0 if combos is None else combos.get(hand, 0.0)
        if weight <= 0.0:
            continue
        live.append((hand, weight))
        val = eval7(list(hand) + board)
        if hero_val > val:
            beat.append((hand, weight))
        elif hero_val == val:
            tied.append((hand, weight))

    if not live:
        raise Unsupported("no hand in the villain's range survives the cards on show")

    if n_villains == 1:
        total = sum(w for _, w in live)
        return (sum(w for _, w in beat) + sum(w for _, w in tied) / 2) / total

    # Hero takes the whole pot beating both, half tying one, a third tying both.
    return (pairs_within(beat)
            + pairs_across(tied, beat) / 2
            + pairs_within(tied) / 3) / pairs_within(live)


def exact_turn(hero, board, n_villains, deck, combos=None):
    """Hero's exact pot share vs n villain hands with one card to come. n is 1 or 2.

    Every river card leaves an exact river problem behind it, so the turn is 46 of those
    rather than a space of its own. That matters most for two villains: walking it
    directly is 20.5M hand-pair-and-river combinations, but exact_river counts its pairs
    instead of enumerating them, so the second villain costs nothing over the first.

    Conditioning on the river this way is not an approximation. The unknown cards are
    exchangeable, so each is the river with probability 1/len(deck) and the villains are
    uniform over what is left -- the same joint distribution the sampling draws from.
    (One float per river card is summed here, so the result sits within a few ulps of the
    true rational rather than exactly on it, as exact_river's single division does.)

    Unlike the river, this is a real trade: ~0.15s per call against the +/-1.7% the 800
    trials would leave. Worth it here because the turn is where a call most often sits
    near its pot odds; the flop stays sampled because it is ~1.07M combinations even
    heads-up, seconds of eval7 rather than a fraction of one.
    """
    return sum(exact_river(hero, board + [river], n_villains,
                           [c for c in deck if c != river], combos)
               for river in deck) / len(deck)


def equity(hero, board, n_villains, trials=TRIALS, villain_range=None, bet_shape=None):
    """Hero's pot share vs n villain hands, board run out to the river.

    Exact from the turn on (exact_turn, exact_river); Monte Carlo on the flop and
    preflop, where the space is too big to count.

    `villain_range` NAMES a qb_ranges chart and `bet_shape` names a BET_SHAPES entry --
    names rather than the loaded range or the built weights, so the memo below still has
    something hashable to key on. Both only reach the streets that enumerate: the sampler
    would have to draw from the range instead of the deck, which is a larger change than
    this. A flop ignores them, and advise() says so.

    The sampling is seeded from the spot so the same request always gets the same answer
    -- an advisor that changes its mind on a refresh reads as broken.
    """
    known = hero + board
    if len(set(known)) != len(known):
        raise Unsupported("duplicate cards between hero's hand and the board")
    # Sorted so a board that arrives in a different order is the same cache entry; the
    # answer never depended on the order anyway.
    return _equity(tuple(sorted(hero)), tuple(sorted(board)), n_villains, trials,
                   villain_range, bet_shape)


@functools.lru_cache(maxsize=EQUITY_CACHE)
def _equity(hero, board, n_villains, trials, villain_range, bet_shape):
    """equity() past the argument check, memoised. Cards must arrive hashable."""
    hero, board = list(hero), list(board)
    deck = [c for c in FULL_DECK if c not in set(hero + board)]
    to_come = 5 - len(board)
    combos = None
    if (villain_range or bet_shape) and to_come <= 1 and n_villains in (1, 2):
        combos = build_combo_weights(
            deck, board, load_range(villain_range) if villain_range else None, bet_shape)

    # Three villains cannot happen at a three-handed table, so the pair counting these
    # two share covers every turn and river; anything else falls through to sampling.
    if n_villains in (1, 2):
        if to_come == 0:
            return exact_river(hero, board, n_villains, deck, combos)
        if to_come == 1:
            return exact_turn(hero, board, n_villains, deck, combos)

    # Same seed the uncached version used: hero and board arrive sorted separately, and
    # sorting their concatenation is what it always keyed on.
    rng = random.Random(f"{sorted(hero + board)}|{n_villains}")

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

    pot = float(state.get('pot', 0))
    hero_bet = float(state['hero'].get('bet', 0))
    stack = float(state['hero'].get('stack', 0))
    top_bet = max(float(state[v].get('bet', 0)) for v in villains)
    to_call = max(0.0, top_bet - hero_bet)

    to_vals = lambda cards: [(RANK_VAL[r], s) for r, s in cards]
    n = len(villains)
    # The range only reaches the streets that enumerate, so a flop is vs random either
    # way; asking for it there would claim an opponent model the answer does not have.
    weighted = USE_RANGE_WEIGHTS and len(board) >= 4 and n <= 2
    # Only a bet narrows the range. No bet is NOT the same as a check: hero may simply be
    # first to act, and the state does not say whether the villains have had their turn.
    # Reading "no chips in front" as "checked" would cap a range that has not acted.
    shape = None
    if weighted and top_bet > 1e-9:
        street_pot = pot - hero_bet - sum(float(state[v].get('bet', 0)) for v in villains)
        shape = bet_shape(top_bet, street_pot)
    eq = equity(to_vals(hero_cards), to_vals(board), n,
                villain_range=CONTINUING_RANGE if weighted else None, bet_shape=shape)

    seats = seat_labels(state)
    hand = ''.join(card_str(c) for c in hero_cards)
    board_str = ''.join(card_str(c) for c in board)

    warnings = []
    if reason:
        warnings.append(f"No solve covers this spot ({reason}); the answer below is an "
                        f"equity-vs-pot-odds heuristic, not a solved strategy.")
    # The turn and river count every villain hand, so the only error left there is the
    # random-hand model itself; the flop carries sampling error on top of it, and saying
    # which is which is the difference between a number worth acting on near a threshold
    # and one that is not.
    exact = len(board) >= 4 and n <= 2
    measured = 'measured exactly' if exact else 'estimated by Monte Carlo'
    if weighted:
        drawn = (f"{CONTINUING_RANGE}, narrowed to the {shape} bet they just made"
                 if shape else CONTINUING_RANGE)
        warnings.append(
            f"Equity is {measured} against {n} hand{'s' if n > 1 else ''} drawn from "
            f"{drawn} -- a stand-in for a villain who continued preflop, not this "
            f"villain's actual line.")
        if not shape:
            warnings.append(
                "Nobody has bet, so the range is not narrowed by any action on this "
                "street: a villain with no chips in front may have checked or may simply "
                "not have acted yet, and the state does not say which.")
        elif n > 1:
            warnings.append(
                f"Both villains are modelled as holding the same {shape}-bet range, "
                f"though only the bettor made that bet -- the caller's range is treated "
                f"as stronger than it is.")
    else:
        warnings.append(
            f"Equity is {measured} against {n} random hand{'s' if n > 1 else ''}. A real "
            f"range moves this by more than the arithmetic below can see, and not always "
            f"the same way: a hand that beats only junk is worth less than this says, "
            f"while one with showdown value is often worth more.")

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
    # Name the opponent model in the one line that gets logged per hand: "42% vs random"
    # and "42% vs a range that just bet big" are different claims about the same number.
    if not weighted:
        against = f"{n} random hand{'s' if n > 1 else ''}"
    elif shape:
        against = f"{n} hand{'s' if n > 1 else ''} from a {shape}-bet range"
    else:
        against = f"{n} hand{'s' if n > 1 else ''} from a preflop-continuing range"
    line = f"{street} {board_str}: {facing} -- hero equity ~{eq:.0%} vs {against}"

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
