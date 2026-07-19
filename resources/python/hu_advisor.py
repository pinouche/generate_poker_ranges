"""Read a live game state and answer the heads-up preflop spot from Nash push/fold tables.

When the third player busts, the game becomes true heads-up and none of the 3-handed
material applies: the dealer now posts the small blind, acts first preflop and LAST
postflop -- the SB_vs_BB solve has that position reversed, and the qb_ranges charts are
built for a game with another player left to act. What we have for this game instead is
the HoldemResources Nash equilibrium of the jam-or-fold game:

    holdemresources_hu_push.csv   the SB's shove range, one row per effective stack
    holdemresources_hu_call.csv   the BB's call-a-shove range, same layout

Rows run 1.00bb to 200.00bb in 0.05bb steps; cells are frequencies, almost all 0 or 1
with fractions at the Nash boundary. Jam-or-fold is only near-optimal short, roughly
15bb effective and below. Deeper, the tables are still an equilibrium of the RESTRICTED
game -- unexploitable as long as you never do anything but jam or fold -- but a
raise-based strategy earns more, so deep spots get an answer with a warning attached.

Covered nodes: the SB first in (jam or fold) and the BB facing a jam (call or fold).
That is the whole game tree of jam-or-fold poker. Everything else -- limps, raises
smaller than all-in, any postflop street -- is Unsupported, the same way a missing
chart is in the 3-handed pack.

Usage:
    python3 resources/python/hu_advisor.py state.json
    cat state.json | python3 resources/python/hu_advisor.py
    python3 resources/python/hu_advisor.py --hand Q7o --stack 8 --seat SB
"""

import argparse
import bisect
import json
import os
import sys

from preflop_advisor import Unsupported, describe, hand_class

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HU_RANGE_BASE = os.environ.get(
    'HU_RANGE_BASE', os.path.join(REPO_ROOT, 'ranges/heads_up_ranges'))
PUSH_TABLE = 'holdemresources_hu_push.csv'
CALL_TABLE = 'holdemresources_hu_call.csv'

# Above this, jam-or-fold stops being how the game should be played at all; the answer
# is still returned (it remains unexploitable within jam-or-fold) but flagged.
PUSH_FOLD_MAX_BB = 15.0

VILLAINS = ['villain_left', 'villain_right']

_tables = {}


def _table(name):
    """(sorted stacks, rows aligned with them, path). Loaded once, ~4k rows x 169 hands."""
    if name not in _tables:
        path = os.path.join(HU_RANGE_BASE, name)
        if not os.path.isfile(path):
            raise Unsupported(f"no heads-up table at {path}")
        with open(path) as f:
            hands = [h.strip() for h in f.readline().split(',')[1:]]
            stacks, rows = [], []
            for line in f:
                cells = [c.strip() for c in line.split(',')]
                if len(cells) != len(hands) + 1:
                    raise Unsupported(f"malformed row in {name}: {line[:40]!r}")
                stacks.append(float(cells[0]))
                rows.append(dict(zip(hands, map(float, cells[1:]))))
        _tables[name] = (stacks, rows, path)
    return _tables[name]


def frequency(name, eff_bb, hand):
    """The table's frequency for a hand at the nearest listed stack.

    Nearest rather than interpolated: the rows are 0.05bb apart, finer than anyone
    measures a live stack, and interpolating 0/1 cells would invent mixes the
    equilibrium does not have.
    """
    stacks, rows, path = _table(name)
    i = bisect.bisect_left(stacks, eff_bb)
    if i >= len(stacks):
        i = len(stacks) - 1
    elif i > 0 and eff_bb - stacks[i - 1] < stacks[i] - eff_bb:
        i -= 1
    if hand not in rows[i]:
        raise Unsupported(f"hand {hand} missing from {name}")
    return rows[i][hand], stacks[i], path


def busted(p):
    """Out of the game entirely, as opposed to folded (keeps a stack) or all-in (has a bet).

    `... or 0` rather than a .get default: a busted seat is often sent with stack=null,
    and the key being present means .get returns that None, which would crash the compare."""
    return p is None or (not p.get('active', True)
                         and (p.get('stack') or 0) < 1e-9 and (p.get('bet') or 0) < 1e-9)


def is_heads_up(state):
    """True when a villain seat is empty -- the busted-player table, not a folded hand."""
    return any(busted(state.get(v)) for v in VILLAINS)


def advise(state, bb_override=None):
    live = [v for v in VILLAINS if not busted(state.get(v))]
    if len(live) != 1:
        raise Unsupported("both villains have busted; hero is the only player left")
    villain = live[0]

    if state.get('street') != 'preflop':
        raise Unsupported(
            "heads-up postflop: the 3-handed solves do not apply (the blind-vs-blind one "
            "even has the postflop position reversed), and jam-or-fold ends the hand preflop")

    dealer = state['dealer']
    if dealer == 'hero':
        hero_seat = 'SB'
    elif dealer == villain:
        hero_seat = 'BB'
    else:
        raise Unsupported(f"the dealer ({dealer}) has busted; the button cannot be on an "
                          f"empty seat")
    seats = {'hero': hero_seat, villain: 'BB' if hero_seat == 'SB' else 'SB'}

    hand = hand_class(state['hero']['cards'])
    hero_bet, vill_bet = state['hero']['bet'], state[villain]['bet']

    bb = float(bb_override or state.get('big_blind') or 0)
    if not bb:
        sb_bet, bb_bet = (hero_bet, vill_bet) if hero_seat == 'SB' else (vill_bet, hero_bet)
        if bb_bet > 0 and abs(sb_bet - bb_bet / 2) < 1e-6:
            bb = float(bb_bet)
        else:
            raise Unsupported(
                f"cannot infer the big blind from bets {sb_bet:g}/{bb_bet:g}; pass --bb")
    sb = float(state.get('small_blind') or bb / 2)

    # The tables are indexed by the effective stack at the start of the hand, blinds
    # included -- chips in front plus chips already posted, smaller player's total.
    eff_bb = min(state['hero']['stack'] + hero_bet,
                 state[villain]['stack'] + vill_bet) / bb

    if hero_seat == 'SB':
        if vill_bet > bb + 1e-9:
            raise Unsupported("the BB has raised before the SB acted; that is not a legal "
                              "heads-up preflop order")
        if hero_bet > sb + 1e-9:
            raise Unsupported(f"hero has already put in {hero_bet / bb:.1f}bb; the tables "
                              f"only answer the unopened small blind")
        table_name, act, prior = PUSH_TABLE, 'AllIn', []
    else:
        if not state[villain].get('active', True):
            raise Unsupported("the SB folded; the pot is hero's, nothing to decide")
        if vill_bet <= sb + 1e-9:
            raise Unsupported("the SB has not acted yet; it is not hero's turn")
        if vill_bet <= bb + 1e-9:
            raise Unsupported("the SB limped; the tables are jam-or-fold and have no "
                              "limp branch")
        if state[villain]['stack'] > 1e-9:
            raise Unsupported(f"the SB raised to {vill_bet / bb:.1f}bb without being "
                              f"all-in; the tables only answer a jam")
        table_name, act, prior = CALL_TABLE, 'Call', [('SB', 'AllIn')]

    freq, row_stack, path = frequency(table_name, eff_bb, hand)
    strategy = sorted([(act, freq, path), ('FOLD', 1.0 - freq, path)], key=lambda x: -x[1])

    warnings = []
    if eff_bb > PUSH_FOLD_MAX_BB:
        warnings.append(
            f"Effective stack is {eff_bb:.0f}bb: jam-or-fold Nash is only near-optimal "
            f"below ~{PUSH_FOLD_MAX_BB:.0f}bb. This answer cannot be exploited as long as "
            f"you stick to jam-or-fold, but a raise-based strategy would earn more here.")
    if abs(row_stack - eff_bb) > 0.05 + 1e-9:
        warnings.append(f"Stack snapped: the tables stop at {row_stack:g}bb and the "
                        f"effective stack is {eff_bb:.1f}bb.")

    return {
        'seats': seats, 'hero_seat': hero_seat, 'hand': hand, 'bb': bb, 'sb': sb,
        'prior': prior, 'strategy': strategy, 'eff_bb': eff_bb, 'row_stack': row_stack,
        'stack_bb': (state['hero']['stack'] + hero_bet) / bb,
        'to_call': max(hero_bet, vill_bet) - hero_bet, 'warnings': warnings,
    }


def report(state, r):
    bb, call = r['bb'], r['to_call']
    villain = next(p for p in r['seats'] if p != 'hero')
    print(f"Table    : heads-up, hero={r['hero_seat']} "
          f"{villain.replace('villain_', '')}={r['seats'][villain]}   "
          f"(blinds {r['sb']:g}/{bb:g})")
    print(f"Hero     : {r['hand']} in the {r['hero_seat']}, effective {r['eff_bb']:.1f}bb "
          f"(table row {r['row_stack']:g}bb)")
    line = ' -> '.join(f"{s} {describe(a, bb)}" for s, a in r['prior']) or 'first to act'
    print(f"Action   : {line}")

    print()
    for action, weight, _ in r['strategy']:
        bar = '#' * int(round(weight * 40))
        print(f"  {describe(action, bb, call):24} {weight * 100:5.1f}%  {bar}")

    best_action, best_weight, _ = r['strategy'][0]
    print()
    if 1e-9 < best_weight < 0.999:
        mix = ', '.join(f"{describe(a, bb, call)} {w * 100:.0f}%"
                        for a, w, _ in r['strategy'] if w > 0)
        print(f"=> mixed: {mix}")
    else:
        print(f"=> {describe(best_action, bb, call)}   (pure, 100%)")

    for warning in r['warnings']:
        print(f"\n! {warning}")


def quick_state(hand, stack_bb, seat):
    """A synthetic table for --hand/--stack/--seat lookups: blinds 0.5/1, chips = bb.

    The BB spot only exists facing a jam, so the villain SB is put all-in for the
    effective stack -- exactly the node the call table answers.
    """
    suits = ('s', 's') if hand.endswith('s') else ('s', 'h')
    cards = [{'rank': hand[0], 'suit': suits[0]}, {'rank': hand[1], 'suit': suits[1]}]
    hero = {'cards': cards, 'stack': 0.0, 'bet': 0.0, 'active': True}
    vill = {'cards': [], 'stack': 0.0, 'bet': 0.0, 'active': True}
    if seat == 'SB':
        dealer, (hero['bet'], vill['bet']) = 'hero', (0.5, 1.0)
    else:
        dealer, (hero['bet'], vill['bet']) = 'villain_left', (1.0, float(stack_bb))
    hero['stack'] = stack_bb - hero['bet']
    vill['stack'] = stack_bb - vill['bet']
    return {'street': 'preflop', 'dealer': dealer, 'big_blind': 1.0, 'small_blind': 0.5,
            'hero': hero, 'villain_left': vill, 'villain_right': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('state', nargs='?', help="game-state json; omit to read stdin")
    parser.add_argument('--bb', type=float, help="big blind, if it cannot be inferred")
    parser.add_argument('--hand', help="quick lookup: hand class, e.g. Q7o, A5s, TT")
    parser.add_argument('--stack', type=float, help="quick lookup: effective stack in bb")
    parser.add_argument('--seat', choices=['SB', 'BB'], help="quick lookup: hero's seat")
    args = parser.parse_args()

    if args.hand:
        if args.stack is None or not args.seat:
            parser.error("--hand needs --stack and --seat")
        state = quick_state(args.hand.strip(), args.stack, args.seat)
    else:
        raw = open(args.state).read() if args.state else sys.stdin.read()
        state = json.loads(raw)

    try:
        report(state, advise(state, args.bb))
    except Unsupported as e:
        print(f"No table for this spot: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
