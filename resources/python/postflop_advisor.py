"""Read a live postflop game state and answer: what does the solved tree say I should do?

The solves in resources/outputs/programmatic are one json per (scenario, flop): the five
scenario folders are every heads-up postflop spot a 3-handed game reaches, and each json
is the full flop tree plus, under each chance node's 'dealcards', the turn trees. The
river is solved but not dumped (22GB per board), so this answers flop and turn only.

Turning a snapshot into a node in that tree takes four inferences, each checked and
warned about rather than assumed:

  1. THE SCENARIO. The two live seats leave at most two candidates (single-raised or
     3bet pot), and the pot at the start of the street -- which is fixed by the preflop
     line -- picks between them.
  2. THE BOARD. Solves cover 186 canonical flops. A real flop is matched by suit
     isomorphism (an exact strategic match); failing that, the nearest solved flop with
     the same suit and rank pattern stands in, by the same feature metric that built the
     subset. Hero's cards and the turn card are translated through the same suit map.
  3. THE STREET SO FAR. Like preflop, the state is a snapshot, not a history: the two
     bet amounts in front of the players are matched against every partial line in the
     tree that ends with hero to act, and the closest sizing wins.
  4. THE FLOP LINE (turn only). Flop bets are gone from the snapshot, but not from the
     pot: the money each player put in on the flop is (turn pot - flop pot) / 2, and the
     flop lines that end in a call or check-check at that price are the candidates.
     When two lines cost the same (bet-call vs check-bet-call), that is a real
     ambiguity and the answer says so.

Usage:
    python3 resources/python/postflop_advisor.py state.json
    cat state.json | python3 resources/python/postflop_advisor.py
"""

import argparse
import functools
import json
import os
import re
import sys
import time

from preflop_advisor import TABLE_ORDER, Unsupported, seats_of

# Resolved from this file, not the working directory, same as CHART_BASE in the preflop
# advisor. SOLVE_BASE overrides it (e.g. a container with the solves volume-mounted).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOLVE_BASE = os.environ.get(
    'SOLVE_BASE', os.path.join(REPO_ROOT, 'resources/outputs/programmatic'))

RANK_ORDER = '23456789TJQKA'
RANK_VAL = {r: i + 2 for i, r in enumerate(RANK_ORDER)}
SUITS = 'cdhs'

# Solver player ids: player 1 acts first postflop (out of position), player 0 last.
OOP, IP = 1, 0

# The five solved spots, mirroring generate_charts.py. `pot` and `stack` are the bb at
# the START of the flop and are what identifies a scenario from a snapshot: the two live
# seats narrow it to two candidates at most, and the pot separates those by a mile
# (5.5 vs 22.5). `oop`/`ip` come from postflop seat order (SB, BB, BTN), not from who
# raised -- the SB is OOP in every pot it plays.
SCENARIOS = [
    {'name': 'BTN_vs_BB', 'oop': 'BB', 'ip': 'BTN', 'pot': 5.5, 'stack': 97.5,
     'preflop': 'BTN opens 2.5bb, SB folds, BB calls'},
    {'name': 'SB_vs_BB', 'oop': 'SB', 'ip': 'BB', 'pot': 6.0, 'stack': 97.0,
     'preflop': 'BTN folds, SB opens 3.0bb, BB calls'},
    {'name': 'BB_vs_BTN_3bet', 'oop': 'BB', 'ip': 'BTN', 'pot': 22.5, 'stack': 89.0,
     'preflop': 'BTN opens 2.5bb, SB folds, BB 3bets to 11bb, BTN calls'},
    {'name': 'SB_vs_BTN_3bet', 'oop': 'SB', 'ip': 'BTN', 'pot': 23.0, 'stack': 89.0,
     'preflop': 'BTN opens 2.5bb, SB 3bets to 11bb, BB folds, BTN calls'},
    {'name': 'BB_vs_SB_3bet', 'oop': 'SB', 'ip': 'BB', 'pot': 18.0, 'stack': 91.0,
     'preflop': 'BTN folds, SB opens 3.0bb, BB 3bets to 9bb, SB calls'},
]

# The same five fields for true heads-up, after the third player busted. Kept in a
# separate list because the seat pair {SB, BB} appears in BOTH sets and means different
# things: three-handed the SB acts first postflop, heads-up the SB IS the button and acts
# last. Merging them would let a heads-up spot match SB_vs_BB, which is the exact
# position-reversal error this whole path exists to avoid. `scenarios_for` picks the list
# by table size; nothing else should reach past it.
#
# Every heads-up pot has the BB out of position and the BTN in position, so there is only
# one geometry and the pot alone separates the four.
HU_SCENARIOS = [
    {'name': 'HU_limped', 'oop': 'BB', 'ip': 'BTN', 'pot': 2.0, 'stack': 99.0,
     'preflop': 'heads-up: BTN limps, BB checks'},
    {'name': 'HU_BTN_vs_BB', 'oop': 'BB', 'ip': 'BTN', 'pot': 5.0, 'stack': 97.5,
     'preflop': 'heads-up: BTN opens 2.5bb, BB calls'},
    {'name': 'HU_limp_raised', 'oop': 'BB', 'ip': 'BTN', 'pot': 8.0, 'stack': 96.0,
     'preflop': 'heads-up: BTN limps, BB raises to 4bb, BTN calls'},
    {'name': 'HU_BB_3bet', 'oop': 'BB', 'ip': 'BTN', 'pot': 20.0, 'stack': 90.0,
     'preflop': 'heads-up: BTN opens 2.5bb, BB 3bets to 10bb, BTN calls'},
]

# Solves built on hand-written ranges rather than solver output, so every answer they
# produce carries this. See resources/python/build_hu_ranges.py.
HU_RANGE_WARNING = (
    "Heads-up solve: its preflop ranges are hand-written approximations, not solver "
    "output. The strategy below is exact for those ranges, which is not the same as "
    "being right.")
HU_LIMP_WARNING = (
    "Limped-pot solve: it assumes a weak limper (limps wide, raises only premiums). "
    "Against someone who limps a balanced range it will have you attacking too hard.")


def all_scenarios():
    return SCENARIOS + HU_SCENARIOS

CARD_RE = re.compile(r'^[AKQJT98765432][cdhs]$')


def parse_card(c):
    """Accept {'rank': 'K', 'suit': 's'}, 'Ks', 'ks' or '10h' -> ('K', 's')."""
    if isinstance(c, dict):
        r, s = str(c.get('rank', '')), str(c.get('suit', ''))
    else:
        r, s = str(c)[:-1], str(c)[-1:]
    r = {'10': 'T'}.get(r, r.upper())
    s = s.lower()
    if not CARD_RE.match(r + s):
        raise Unsupported(f"unreadable card {c!r}")
    return (r, s)


def card_str(card):
    return card[0] + card[1]


def big_blind(state, override=None):
    """Postflop the blinds are long since swallowed by the pot, so unlike preflop there
    is nothing to infer them from -- the state has to say, or the caller has to."""
    if override:
        return float(override)
    if state.get('big_blind'):
        return float(state['big_blind'])
    raise Unsupported(
        'big_blind missing from the state and not inferable postflop; pass --bb')


# ---------------------------------------------------------------------------
# Matching a real board to a solved one.
#
# suit_pattern / rank_pattern / features mirror build_flop_subset.py exactly (not
# imported: that module pulls in sklearn, which the API image does not carry). They must
# stay in lockstep -- the nearest-representative pick below is only faithful to the
# subset if it measures distance the same way the clustering did.
# ---------------------------------------------------------------------------

HIGH_CARD_WEIGHT = 10.0


def suit_pattern(cards):
    return {1: 'monotone', 2: 'two-tone', 3: 'rainbow'}[len({s for _, s in cards})]


def rank_pattern(cards):
    counts = {}
    for r, _ in cards:
        counts[r] = counts.get(r, 0) + 1
    return {3: 'trips', 2: 'paired', 1: 'unpaired'}[max(counts.values())]


def features(cards):
    vals = sorted((RANK_VAL[r] for r, _ in cards), reverse=True)
    hi, mid, lo = vals
    return [HIGH_CARD_WEIGHT * hi / 14.0, mid / 14.0, lo / 14.0,
            (hi - mid) / 12.0, (mid - lo) / 12.0]


@functools.lru_cache(maxsize=None)
def solved_boards(scenario):
    """Every solved flop of one scenario: {frozenset of cards: stem} plus the parsed list."""
    folder = os.path.join(SOLVE_BASE, scenario)
    if not os.path.isdir(folder):
        raise Unsupported(f"no solves on disk for scenario {scenario}")
    index, boards = {}, []
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        tokens = stem.split('_')
        if ext != '.json' or len(tokens) != 3 or not all(CARD_RE.match(t) for t in tokens):
            continue
        cards = tuple(parse_card(t) for t in tokens)
        index[frozenset(cards)] = stem
        boards.append((stem, cards))
    if not boards:
        raise Unsupported(f"scenario folder {scenario} holds no solves")
    return index, boards


def suit_roles(cards):
    """The board's suits ordered by how much they matter: most cards first, then
    highest card, so the (backdoor) flush suit of one board maps onto the flush suit
    of the other. Suits not on the board follow in fixed order."""
    by_suit = {}
    for r, s in cards:
        by_suit.setdefault(s, []).append(RANK_VAL[r])
    ranked = sorted(by_suit, key=lambda s: (-len(by_suit[s]), -max(by_suit[s]), s))
    return ranked + [s for s in SUITS if s not in ranked]


def match_board(flop, scenario):
    """Find the solved flop that stands for this real one.

    Exact first: two flops are the same strategic problem iff one is a suit relabelling
    of the other, so try all 24 relabellings against the solved set. Otherwise the
    nearest solved flop in the same stratum stands in -- same suit pattern and rank
    pattern, closest rank texture -- which is exactly the approximation the subset was
    built to make well. Returns (stem, suit map real->solved, exact).
    """
    from itertools import permutations
    index, boards = solved_boards(scenario)

    for perm in permutations(SUITS):
        smap = dict(zip(SUITS, perm))
        key = frozenset((r, smap[s]) for r, s in flop)
        if key in index:
            return index[key], smap, True

    sp, rp = suit_pattern(flop), rank_pattern(flop)
    candidates = [(stem, cards) for stem, cards in boards
                  if suit_pattern(cards) == sp and rank_pattern(cards) == rp]
    if not candidates:
        raise Unsupported(
            f"no solved flop shares this board's texture ({sp}, {rp}) in {scenario}")
    mine = features(flop)
    stem, cards = min(candidates, key=lambda sc: sum(
        (a - b) ** 2 for a, b in zip(mine, features(sc[1]))))
    smap = dict(zip(suit_roles(flop), suit_roles(cards)))
    return stem, smap, False


def place_card(card, smap, taken):
    """Map one card into the solved board's world, dodging cards already there.

    Collisions only happen on inexact board matches (the solved flop holds a card the
    suit map would hand us); the substitute keeps the rank and takes any free suit.
    Returns (card, moved) so the caller can warn.
    """
    r, s = card
    want = (r, smap[s])
    if want not in taken:
        return want, False
    for alt in SUITS:
        if (r, alt) not in taken:
            return (r, alt), True
    raise Unsupported(f"no room for {card_str(card)} on the solved board")


def map_hero(cards, smap, taken):
    """Hero's two cards in the solved board's world.

    What must survive the translation is the relationship to the board -- suitedness
    between the two cards, and which of them lives in the board's flush suit -- not the
    suit letters themselves. Prefer the straight suit map, but on a collision take the
    placement that agrees with it in the most places. Returns (cards, moved).
    """
    (r1, s1), (r2, s2) = cards
    suited = s1 == s2
    best = None
    for a in SUITS:
        for b in SUITS:
            if (a == b) != suited:
                continue
            c1, c2 = (r1, a), (r2, b)
            if c1 in taken or c2 in taken or c1 == c2:
                continue
            score = (a == smap[s1]) + (b == smap[s2])
            if best is None or score > best[0]:
                best = (score, c1, c2)
    if best is None:
        raise Unsupported(f"cannot place {card_str(cards[0])}{card_str(cards[1])} "
                          f"on the solved board")
    return (best[1], best[2]), best[0] < 2


# ---------------------------------------------------------------------------
# The tree.
# ---------------------------------------------------------------------------

# A parsed solve is ~1GB of Python objects, so hold the last two: while playing a hand
# you ask about the same board twice (flop, then turn), and a turn request that has to
# disambiguate two candidate scenarios touches two files.
@functools.lru_cache(maxsize=2)
def load_solve(path):
    with open(path) as f:
        return json.load(f)


def action_amount(action):
    """'BET 8.000000' -> 8.0: the actor's total commitment on this street, in bb."""
    return float(action.split()[1])


def street_nodes(root):
    """Every decision node of one street, with the line to it and both commitments.

    Only CHECK/BET/RAISE edges are followed: a call or a fold ends the street, so no
    node hero could be asked about lies beyond one. Commitments are per-street totals,
    which is also what a bet/raise action's number means and what the table's `bet`
    fields hold -- the three compare directly.
    """
    found = []

    def rec(node, line, commits):
        if node.get('node_type') != 'action_node':
            return
        found.append((line, node, dict(commits)))
        player = node['player']
        for action, child in (node.get('childrens') or {}).items():
            verb = action.split()[0]
            if verb == 'CHECK':
                rec(child, line + [(player, action)], commits)
            elif verb in ('BET', 'RAISE'):
                rec(child, line + [(player, action)],
                    {**commits, player: action_amount(action)})

    rec(root, [], {OOP: 0.0, IP: 0.0})
    return found


def chance_lines(root):
    """Every completed flop line, as (line, chance node, what each player put in).

    These are the ways a flop can end without ending the hand: a call, or check-check.
    An all-in call that skips straight to showdown has no chance node and is dropped --
    there is no turn decision to advise on down that line anyway.
    """
    found = []

    def rec(node, line, commits):
        kind = node.get('node_type')
        if kind == 'chance_node':
            found.append((line, node, max(commits.values())))
            return
        if kind != 'action_node':
            return
        player = node['player']
        for action, child in (node.get('childrens') or {}).items():
            verb = action.split()[0]
            if verb == 'CHECK':
                rec(child, line + [(player, action)], commits)
            elif verb in ('BET', 'RAISE'):
                rec(child, line + [(player, action)],
                    {**commits, player: action_amount(action)})
            elif verb == 'CALL':
                rec(child, line + [(player, action)],
                    {**commits, player: max(commits.values())})

    rec(root, [], {OOP: 0.0, IP: 0.0})
    return found


# ---------------------------------------------------------------------------
# Presenting an action.
# ---------------------------------------------------------------------------

def kind_of(action):
    verb = action.split()[0]
    return verb if verb in ('CHECK', 'BET', 'RAISE', 'CALL', 'FOLD') else 'RAISE'


def chips_of(action, bb, to_call=None):
    verb = kind_of(action)
    if verb in ('BET', 'RAISE'):
        return action_amount(action) * bb
    return to_call if verb == 'CALL' else None


def describe(action, bb, to_call=None):
    """'BET 8.000000' with bb=2 -> 'BET 16 (8bb)': chips first, like the preflop
    advisor, because chips are what you can type into the table."""
    verb = kind_of(action)
    if verb in ('CHECK', 'FOLD'):
        return verb
    if verb == 'CALL':
        return 'CALL' + (f" {to_call:g}" if to_call else '')
    size = action_amount(action)
    to = 'RAISE to' if verb == 'RAISE' else 'BET'
    return f"{to} {size * bb:g} ({size:g}bb)"


def seat_line(line, scenario):
    """[(1, 'CHECK'), (0, 'BET 2.000000')] -> [('BB', 'CHECK'), ('BTN', 'BET 2.000000')]."""
    seat = {OOP: scenario['oop'], IP: scenario['ip']}
    return [(seat[p], a) for p, a in line]


def action_line(r):
    """The whole hand so far as one readable string, shared by the CLI and the API."""
    bb = r['bb']
    parts = [r['preflop']]
    flop = ' '.join(r['flop_cards'])
    if r['street'] == 'turn':
        played = ' -> '.join(f"{s} {describe(a, bb)}" for s, a in r['flop_line'])
        parts.append(f"flop {flop}: {played}")
        street, board = 'turn', r['turn_card']
    else:
        street, board = 'flop', flop
    played = ' -> '.join(f"{s} {describe(a, bb)}" for s, a in r['line'])
    parts.append(f"{street} {board}: {played or 'hero first to act'}")
    return ' | '.join(parts)


# ---------------------------------------------------------------------------
# Hand memory.
#
# The client sends the full table state at every hero decision, so by the turn the
# server has already SEEN the flop betting live -- remembering it beats re-inferring it
# from the pot, and kills the bet-call vs check-bet-call ambiguity, which a snapshot
# alone genuinely cannot resolve. The flop cards themselves identify the hand: the same
# dealer and the same three cards within the TTL is the same hand for any practical
# purpose. Memory is an overlay only -- a turn request with no matching entry (server
# restarted, flop request never arrived, pot that contradicts the memory) falls back to
# the stateless inference, so nothing depends on it.
# ---------------------------------------------------------------------------

_MEMORY = {}
_MEMORY_TTL = 3600.0   # a live hand is minutes; an hour is generous
_MEMORY_MAX = 64


def _hand_key(state, flop, bb):
    return (state.get('dealer'), frozenset(card_str(c) for c in flop), bb)


def _remember(key, entry):
    now = time.time()
    for k in [k for k, v in _MEMORY.items() if now - v['time'] > _MEMORY_TTL]:
        del _MEMORY[k]
    while len(_MEMORY) >= _MEMORY_MAX:
        del _MEMORY[min(_MEMORY, key=lambda k: _MEMORY[k]['time'])]
    _MEMORY[key] = {**entry, 'time': now}


def _recall(key):
    entry = _MEMORY.get(key)
    if entry and time.time() - entry['time'] <= _MEMORY_TTL:
        return entry
    return None


# ---------------------------------------------------------------------------
# The lookup itself.
# ---------------------------------------------------------------------------

def hero_and_villain(state):
    """Postflop the solves are heads-up, so exactly one villain must still be live."""
    if not state['hero'].get('active', True):
        raise Unsupported('hero has folded; nothing to advise')
    live = [p for p in TABLE_ORDER[1:] if state[p].get('active', True)]
    if len(live) == 2:
        raise Unsupported('three players saw the flop; the solves are heads-up only')
    if not live:
        raise Unsupported('both villains folded; the hand is over')
    return live[0]


def hu_seats(state, villain):
    """Seats for true heads-up: the dealer posts the small blind AND holds the button.

    seats_of() cannot be used here -- it walks three chairs from the dealer and would
    hand the busted seat a position. Heads-up there are two seats, and the dealer's is
    both the SB preflop and the BTN postflop. We name it BTN because these solves are
    postflop and the whole point is who acts last.
    """
    dealer = state['dealer']
    if dealer == 'hero':
        return {'hero': 'BTN', villain: 'BB'}
    if dealer == villain:
        return {'hero': 'BB', villain: 'BTN'}
    raise Unsupported(f"the dealer ({dealer}) has busted; the button cannot be on an "
                      f"empty seat")


def scenarios_for(state, villain):
    """(seats, candidate scenarios) for this table -- heads-up or three-handed.

    The two lists are never mixed: see the note on HU_SCENARIOS for why that would be a
    correctness bug rather than just noise.
    """
    import hu_advisor
    if hu_advisor.is_heads_up(state):
        seats = hu_seats(state, villain)
        return seats, HU_SCENARIOS
    seats = seats_of(state)
    pair = {seats['hero'], seats[villain]}
    return seats, [s for s in SCENARIOS if {s['oop'], s['ip']} == pair]


def scenario_warnings(scenario):
    """The caveats a scenario's own provenance forces onto every answer from it."""
    if not scenario['name'].startswith('HU_'):
        return []
    warnings = [HU_RANGE_WARNING]
    if 'limp' in scenario['name']:
        warnings.append(HU_LIMP_WARNING)
    return warnings


def match_street_node(root, hero_player, hero_bet, vill_bet):
    """The tree node hero is standing at, from the two bets in front of the players.

    Structure first, size second: a player who has not put chips in must map to a line
    where they have not (zero is not a sizing to snap to), and among the lines with the
    right shape the closest sizings win, exactly like the preflop chart snap.
    """
    vill_player = IP if hero_player == OOP else OOP
    candidates = []
    for line, node, commits in street_nodes(root):
        if node['player'] != hero_player:
            continue
        if (commits[hero_player] < 1e-9) != (hero_bet < 1e-9):
            continue
        if (commits[vill_player] < 1e-9) != (vill_bet < 1e-9):
            continue
        gap = abs(commits[hero_player] - hero_bet) + abs(commits[vill_player] - vill_bet)
        candidates.append((gap, len(line), line, node))
    if not candidates:
        raise Unsupported(
            f"no node in the solve matches bets of {hero_bet:g}bb (hero) and "
            f"{vill_bet:g}bb (villain) with hero to act on this street")
    gap, _, line, node = min(candidates, key=lambda c: (c[0], c[1]))
    return line, node, gap


def find_turn_root(state, candidates, flop, turn_pot, remembered=None):
    """Pick the scenario AND the flop line that put this much money in the middle.

    With a remembered flop request the line was watched, not guessed: hero stood at a
    known node when it last acted, so the real flop line extends that prefix, and the
    turn pot only has to pick among its completions (hero's own action plus at most a
    closing call) -- usually exactly one. Without memory, both the scenario and the
    line come from the pot alone: each candidate scenario fixes the flop-start pot,
    each of its flop lines adds a known amount, and the observed turn pot has to be
    explained by the pair. Returns the best match, any other lines the pot explains
    equally well (bet-call and check-bet-call cost the same and are genuinely
    indistinguishable from a snapshot), and whether the line was observed.
    """
    if remembered:
        scenario = next(s for s in all_scenarios() if s['name'] == remembered['scenario'])
        need = (turn_pot - scenario['pot']) / 2
        board = (remembered['stem'], remembered['smap'], remembered['exact'])
        data = load_solve(os.path.join(SOLVE_BASE, scenario['name'],
                                       remembered['stem'] + '.json'))
        prefix = remembered['line']
        matches = [(2 * abs(invested - need), len(line), scenario, board,
                    line, node, invested)
                   for line, node, invested in chance_lines(data)
                   if line[:len(prefix)] == prefix]
        if matches:
            best = min(matches, key=lambda e: (e[0], e[1]))
            if best[0] <= 2.0:
                err, _, scenario, board, line, node, invested = best
                return scenario, board, line, node, invested, err, [], True
        # The pot contradicts what we watched (or the hand ended another way and this
        # is a new one under a recycled key): distrust the memory, infer from scratch.

    entries = []
    for scenario in candidates:
        need = (turn_pot - scenario['pot']) / 2
        if need < -0.75:   # the pot cannot shrink; this scenario is out
            continue
        stem, smap, exact = match_board(flop, scenario['name'])
        data = load_solve(os.path.join(SOLVE_BASE, scenario['name'], stem + '.json'))
        for line, node, invested in chance_lines(data):
            err = 2 * abs(invested - need)
            entries.append((err, len(line), scenario, (stem, smap, exact),
                            line, node, invested))
    if not entries:
        raise Unsupported(
            f"a turn pot of {turn_pot:g}bb is smaller than any candidate scenario's "
            f"flop pot; the preflop action matches no solve")
    entries.sort(key=lambda e: (e[0], e[1]))
    best = entries[0]
    ambiguous = [e for e in entries[1:]
                 if e[0] - best[0] < 0.5 and (e[2]['name'], e[4]) != (best[2]['name'], best[4])]
    err, _, scenario, board, line, node, invested = best
    return scenario, board, line, node, invested, err, ambiguous, False


def advise(state, bb_override=None):
    street = str(state.get('street', '')).lower()
    if street == 'preflop':
        raise Unsupported('preflop is the chart lookup, not the solves; '
                          'use preflop_advisor')
    if street == 'river':
        raise Unsupported('the river is solved but not dumped (22GB per board); '
                          'the solves stop at the turn')
    if street not in ('flop', 'turn'):
        raise Unsupported(f"unknown street {state.get('street')!r}")

    villain = hero_and_villain(state)
    seats, candidates = scenarios_for(state, villain)
    hero_seat, vill_seat = seats['hero'], seats[villain]
    if not candidates:   # cannot happen at either table size, but fail loudly not crash
        raise Unsupported(f"no scenario covers {hero_seat} vs {vill_seat}")

    bb = big_blind(state, bb_override)
    board = [parse_card(c) for c in state.get('board', [])]
    if len(board) != (3 if street == 'flop' else 4):
        raise Unsupported(f"{street} needs {3 if street == 'flop' else 4} board cards, "
                          f"got {len(board)}")
    if len(state['hero'].get('cards', [])) != 2:
        raise Unsupported(f"hero has {len(state['hero'].get('cards', []))} cards, need 2")
    hero_cards = [parse_card(c) for c in state['hero']['cards']]
    flop = board[:3]

    hero_bet = float(state['hero'].get('bet', 0)) / bb
    vill_bet = float(state[villain].get('bet', 0)) / bb
    # The pot field includes the live bets (the preflop convention: blinds posted,
    # pot 30 at 10/20), so the street-start pot -- the quantity the scenarios and the
    # tree are keyed by -- is what is left after backing them out.
    street_pot = float(state.get('pot', 0)) / bb - hero_bet - vill_bet

    warnings = []
    flop_line, invested, turn_card = [], 0.0, None
    key = _hand_key(state, flop, bb)
    observed = False

    if street == 'flop':
        scenario = min(candidates, key=lambda s: abs(street_pot - s['pot']))
        stem, smap, exact = match_board(flop, scenario['name'])
        root = load_solve(os.path.join(SOLVE_BASE, scenario['name'], stem + '.json'))
    else:
        remembered = _recall(key)
        scenario, (stem, smap, exact), raw_flop_line, chance, invested, _, ties, observed = \
            find_turn_root(state, candidates, flop, street_pot, remembered)
        if remembered and not observed:
            warnings.append(
                "Hero's flop requests were seen but do not line up with this turn pot; "
                "the flop line below is inferred from the pot instead.")
        flop_line = seat_line(raw_flop_line, scenario)
        if ties:
            lines = '; '.join(
                ' -> '.join(f"{s} {describe(a, bb)}" for s, a in seat_line(t[4], t[2]))
                for t in ties)
            warnings.append(
                f"The flop line is ambiguous: the pot also matches [{lines}]. Different "
                f"lines reach different turn strategies; showing the shortest match.")
        turn_card, moved = place_card(board[3], smap, set(parse_card(t) for t in
                                                          stem.split('_')))
        if moved:
            warnings.append(f"Turn card resettled: {card_str(board[3])} had no clean "
                            f"suit on the solved board, playing it as {card_str(turn_card)}.")
        root = (chance.get('dealcards') or {}).get(card_str(turn_card))
        if not root:
            raise Unsupported(f"the solve has no turn subtree for {card_str(turn_card)}")

    # Where the solve came from qualifies the whole answer rather than one detail of it,
    # so it goes ahead of the snap/mismatch notes below.
    warnings = scenario_warnings(scenario) + warnings

    solved_flop = tuple(parse_card(t) for t in stem.split('_'))
    if not exact:
        warnings.append(
            f"Board snapped: no solve is suit-isomorphic to {' '.join(map(card_str, flop))}; "
            f"the nearest same-texture solve {' '.join(map(card_str, solved_flop))} stands in.")
    expected_pot = scenario['pot'] + 2 * invested
    if abs(street_pot - expected_pot) > 1.0:
        warnings.append(
            f"Pot mismatch: the table's {street}-start pot is {street_pot:g}bb but the "
            f"matched line makes it {expected_pot:g}bb -- the earlier sizings differ "
            f"from the solve's.")

    taken = set(solved_flop) | ({turn_card} if turn_card else set())
    hero_solved, moved = map_hero(hero_cards, smap, taken)
    if moved:
        warnings.append(
            f"Hand resettled: {''.join(map(card_str, hero_cards))} collides with the "
            f"solved board, playing it as {''.join(map(card_str, hero_solved))} -- "
            f"suit-dependent details (flush draws) may not carry over.")

    hero_player = OOP if hero_seat == scenario['oop'] else IP
    raw_line, node, snap = match_street_node(root, hero_player, hero_bet, vill_bet)
    if street == 'flop':
        # Every hero decision on the flop lands here, so the stored line is always the
        # latest (longest) one -- exactly the prefix the turn request wants.
        _remember(key, {'scenario': scenario['name'], 'stem': stem, 'smap': smap,
                        'exact': exact, 'line': raw_line})
    if snap > 0.5:
        warnings.append(f"Sizing snapped: the solved line's bets are {snap:.1f}bb away "
                        f"from what is actually in front of the players.")

    combo = card_str(hero_solved[0]) + card_str(hero_solved[1])
    strat = node['strategy']
    probs = (strat['strategy'].get(combo)
             or strat['strategy'].get(card_str(hero_solved[1]) + card_str(hero_solved[0])))
    if probs is None:
        raise Unsupported(
            f"the solve's {hero_seat} range never reaches this node holding {combo}; "
            f"the chart would have folded (or never taken this line with) this hand")

    strategy = sorted(zip(strat['actions'], (float(p) for p in probs)),
                      key=lambda x: -x[1])

    # The stack the solve assumed at this node vs the stack hero actually has behind.
    stack_bb = (float(state['hero']['stack']) + hero_bet * bb) / bb
    solved_stack = scenario['stack'] - invested
    if abs(stack_bb - solved_stack) > 0.15 * solved_stack:
        warnings.append(
            f"Stack mismatch: hero has {stack_bb:.0f}bb behind at the start of the "
            f"{street} but the solve assumes {solved_stack:.1f}bb. The SPR drives the "
            f"whole strategy -- treat this as a rough guide only.")

    return {
        'seats': seats, 'hero_seat': hero_seat, 'villain_seat': vill_seat,
        'position': 'OOP' if hero_player == OOP else 'IP',
        'hand': ''.join(map(card_str, hero_cards)),
        'hand_solved': combo, 'bb': bb, 'street': street,
        'scenario': scenario['name'], 'preflop': scenario['preflop'],
        'flop_cards': [card_str(c) for c in flop],
        'turn_card': card_str(board[3]) if street == 'turn' else None,
        'solve_board': stem, 'board_exact': exact,
        'flop_line': flop_line, 'line': seat_line(raw_line, scenario),
        'flop_line_source': 'observed' if observed else 'inferred',
        'strategy': strategy, 'snap_error': snap,
        'pot': float(state.get('pot', 0)),
        'to_call': float(state[villain].get('bet', 0)) - float(state['hero'].get('bet', 0)),
        'stack_bb': stack_bb, 'warnings': warnings,
    }


def report(state, r):
    bb, call = r['bb'], r['to_call']
    seat_str = ', '.join(f"{p.replace('villain_', '')}={r['seats'][p]}" for p in TABLE_ORDER)
    solved = r['solve_board'].replace('_', ' ')
    print(f"Table    : {seat_str}   (bb {bb:g})")
    print(f"Scenario : {r['scenario']} -- {r['preflop']}")
    board = ' '.join(r['flop_cards']) + (f"  {r['turn_card']}" if r['turn_card'] else '')
    match = 'exact suit-isomorph' if r['board_exact'] else 'nearest texture'
    print(f"Board    : {board}   [solved as {solved}, {match}]")
    print(f"Hero     : {r['hand']} in the {r['hero_seat']} ({r['position']}, "
          f"{r['stack_bb']:.0f}bb behind)   [solved as {r['hand_solved']}]")
    line_note = '' if r['street'] == 'flop' else f"   [flop line {r['flop_line_source']}]"
    print(f"Action   : {action_line(r)}{line_note}")

    print()
    for action, weight in r['strategy']:
        bar = '#' * int(round(weight * 40))
        print(f"  {describe(action, bb, call):24} {weight * 100:5.1f}%  {bar}")

    best_action, best_weight = r['strategy'][0]
    print()
    if best_weight >= 0.999:
        print(f"=> {describe(best_action, bb, call)}   (pure, 100%)")
    else:
        mix = ', '.join(f"{describe(a, bb, call)} {w * 100:.0f}%"
                        for a, w in r['strategy'] if w > 0.005)
        print(f"=> mixed: {mix}")
        print(f"   most of the time: {describe(best_action, bb, call)}")

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
        report(state, advise(state, args.bb))
    except Unsupported as e:
        print(f"No solve for this spot: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
