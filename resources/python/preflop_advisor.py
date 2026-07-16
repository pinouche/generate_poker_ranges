"""Read a live game state and answer: what does the preflop chart say I should do?

The qb_ranges pack is a full preflop tree encoded in filenames -- 'BTN_2.5bb_BB_11.0bb.txt'
is "BTN opens 2.5bb, BB 3-bets to 11bb", and the file holds the range of the last player
named. The files that share a prefix and differ only in the final action ARE the option set
at that node, and a hand's weights across them sum to 1. So answering "what do I do" is:
reconstruct the action so far, find the files matching it, and read the hand out of each.

An option with no file is an option the chart does not offer: there is no
SB/BTN_2.5bb_SB_Call.txt, because the SB never flats a button open here -- 3-bet or fold.

Usage:
    python3 resources/python/preflop_advisor.py state.json
    cat state.json | python3 resources/python/preflop_advisor.py
"""

import argparse
import json
import os
import re
import sys

# Resolved from this file, not the working directory: the API serves from wherever it is
# started and inside a container, and a chart path that only works when you happen to be
# in the repo root would fail there. CHART_BASE overrides it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHART_BASE = os.environ.get(
    'CHART_BASE', os.path.join(REPO_ROOT, 'ranges/qb_ranges/100bb 2.5x 500rake'))
CHART_DEPTH_BB = 100.0   # the pack is 100bb deep; a shallower table is a different game
RANKS = 'AKQJT98765432'

# Three-handed, so the seats are the three that always exist. Play moves clockwise from
# the dealer: the player on the dealer's left posts the small blind, the next the big.
SEATS_FROM_DEALER = ['BTN', 'SB', 'BB']
TABLE_ORDER = ['hero', 'villain_left', 'villain_right']
# Preflop the blinds act last, so the button opens the action three-handed.
PREFLOP_ORDER = ['BTN', 'SB', 'BB']


class Unsupported(Exception):
    """The spot is real but this chart pack has nothing for it."""


def seats_of(state):
    """Map each player to a seat, starting from the dealer and moving clockwise."""
    dealer = state['dealer']
    if dealer not in TABLE_ORDER:
        raise Unsupported(f"unknown dealer {dealer!r}")
    start = TABLE_ORDER.index(dealer)
    return {TABLE_ORDER[(start + i) % 3]: seat
            for i, seat in enumerate(SEATS_FROM_DEALER)}


def hand_class(cards):
    """[{'rank':'A','suit':'h'}, {'rank':'Q','suit':'s'}] -> 'AQo'. Charts are hand-class."""
    if len(cards) != 2:
        raise Unsupported(f"hero has {len(cards)} cards, need 2")
    (r1, s1), (r2, s2) = [(c['rank'].upper(), c['suit'].lower()) for c in cards]
    if r1 not in RANKS or r2 not in RANKS:
        raise Unsupported(f"bad ranks {r1}{r2}")
    hi, lo = sorted([r1, r2], key=RANKS.index)
    if hi == lo:
        return hi + lo
    return f"{hi}{lo}{'s' if s1 == s2 else 'o'}"


def big_blind(state, seats, override=None):
    """The big blind, which every size in the charts is measured against.

    Take it from the state when it is there. Falling back on the posted bets only works
    while the blinds are still the blinds -- once someone raises, their bet says nothing
    about the level -- so if it is absent and unrecoverable, ask rather than guess: a wrong
    bb rescales every raise and would silently point at the wrong chart.
    """
    if override:
        return float(override)
    if state.get('big_blind'):
        return float(state['big_blind'])
    bets = {seats[p]: state[p]['bet'] for p in TABLE_ORDER}
    sb, bb = bets['SB'], bets['BB']
    if bb > 0 and abs(sb - bb / 2) < 1e-6:
        return float(bb)
    raise Unsupported(
        f"cannot infer the big blind from bets SB={sb} BB={bb}; pass --bb")


def check_seating(state, seats, sb, bb):
    """Cross-check the seats against who actually posted the blinds.

    Seats come from the dealer plus clockwise order, which assumes villain_left is the
    player on hero's left. Now that the state names the blinds we can verify that instead
    of trusting it: whoever we call the SB must have posted the small blind. If they did
    not, the seating is inverted and every chart we pick after this is the wrong one, so
    stop -- advising confidently from the wrong seat is worse than not advising.

    Only checkable before anyone raises; a player who has already acted has bet over their
    blind, and we cannot tell what they posted. A blind sitting at MORE than their post is
    fine (the SB completing to the bb is a limp, handled later); less is impossible in a
    legal hand, so it can only mean the seats are wrong. Folded players are skipped: some
    tables zero a folded blind's bet, and dead chips prove nothing about seating.
    """
    posted = {seats[p]: state[p]['bet'] for p in TABLE_ORDER if state[p].get('active', True)}
    if any(v > bb + 1e-9 for v in posted.values()):
        return   # someone has raised; blinds are no longer visible in the bets
    if posted.get('SB', sb) + 1e-9 < sb or posted.get('BB', bb) + 1e-9 < bb:
        raise Unsupported(
            f"seating does not match the blinds: with {state['dealer']} on the button, "
            f"the SB seat bet {posted.get('SB', 0):g} and the BB seat bet "
            f"{posted.get('BB', 0):g}, but the blinds are {sb:g}/{bb:g}. "
            f"Is villain_left really on hero's left?")


def actions_so_far(state, seats, bb):
    """Reconstruct the preflop action from the chips in front of each player.

    We get a snapshot, not a history, so this is an inference: anyone betting more than the
    big blind has raised, and raises can only ever increase, so sorting the raisers by bet
    size recovers the order they acted in. A player matching the top bet without having
    raised has called. Posting a blind is not an action and does not appear in chart names.

    This cannot see a limp-then-fold or a call that was later raised over, but neither can
    the charts -- they have no limp branches at all -- so nothing is lost that was there.
    """
    live = [(seats[p], state[p]) for p in TABLE_ORDER if state[p].get('active', True)]
    top = max(pl['bet'] for _, pl in live)

    raisers = sorted([(seat, pl) for seat, pl in live
                      if pl['bet'] > bb + 1e-9], key=lambda x: x[1]['bet'])
    if not raisers:
        if top > bb + 1e-9:
            raise Unsupported("a bet exceeds the blind but no raiser found")
        # A non-BB player sitting at exactly the big blind has voluntarily completed to it
        # without raising -- a limp. The charts are raise-or-fold and have no limp branches,
        # so this is a real spot we cannot answer, not the fold-round-to-hero case below.
        # (The BB is naturally at bb from posting, so only the others betray a limp.)
        if any(seat != 'BB' and pl['bet'] > bb - 1e-9 for seat, pl in live):
            raise Unsupported(
                "limped pot: a player completed to the big blind rather than raising, and "
                "these charts have no limp branches")
        return []   # folded round to the button: nobody has voluntarily put money in

    # A raiser with nothing behind has shoved, and the charts name that node 'AllIn'
    # rather than by its size -- the size is whatever the stack happened to be.
    seq = [(seat, 'AllIn' if pl['stack'] < 1e-9 else f"{pl['bet'] / bb:.1f}bb")
           for seat, pl in raisers]

    # Anyone at the top bet who is not the last raiser has called it. Earlier raisers who
    # then called a re-raise are visible the same way: their bet was topped up to match.
    last_raiser, last_bet = raisers[-1][0], raisers[-1][1]['bet']
    for seat, pl in live:
        if seat == last_raiser:
            continue
        if abs(pl['bet'] - last_bet) < 1e-9:
            seq.append((seat, 'Call'))
    return seq


def parse_stem(stem):
    """'CO_2.5bb_BTN_Call_SB_13.0bb' -> [('CO','2.5bb'), ('BTN','Call'), ('SB','13.0bb')]."""
    tokens = stem.split('_')
    if len(tokens) % 2:
        return []
    return [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]


def load_charts(seat):
    """Every chart for one seat, as (action sequence, path). Includes vs_3bet/4bet/5bet."""
    charts = []
    root_dir = os.path.join(CHART_BASE, seat)
    if not os.path.isdir(root_dir):
        raise Unsupported(f"no chart folder for seat {seat}")
    for root, _, files in os.walk(root_dir):
        for f in files:
            if not f.endswith('.txt'):
                continue
            seq = parse_stem(os.path.splitext(f)[0])
            if seq:
                charts.append((seq, os.path.join(root, f)))
    return charts


def snapshot_line(seq):
    """A chart line as the bets in front of the players would show it.

    The snapshot keeps only each player's final state: a re-raise overwrites the same
    player's earlier raise (the chart stem 'BTN_2.5bb_SB_11.0bb_BTN_24.0bb' leaves BTN's
    chips at 24bb, the 2.5bb is gone), and a fold leaves nothing visible at all. Collapsing
    the chart lines this way makes them comparable with actions_so_far's reconstruction --
    including hero's own earlier raises, which are as much a part of the line as anyone's.
    """
    last = {}
    for i, (seat, _) in enumerate(seq):
        last[seat] = i
    return [(seat, act) for i, (seat, act) in enumerate(seq)
            if last[seat] == i and act != 'FOLD']


def is_raise(action):
    return action.endswith('bb')


def size_of(action):
    return float(action[:-2]) if is_raise(action) else None


def same_shape(chart_seq, actual_seq):
    """Same players acting in the same order, doing the same kind of thing.

    Sizes are compared later: the chart offers one canonical open (2.5bb) and the table
    will not have used exactly that, so shape has to match first and size snaps after.
    """
    if len(chart_seq) != len(actual_seq):
        return False
    for (cs, ca), (as_, aa) in zip(chart_seq, actual_seq):
        if cs != as_:
            return False
        if is_raise(ca) != is_raise(aa):
            return False
        if not is_raise(ca) and ca != aa:
            return False
    return True


def size_distance(chart_seq, actual_seq):
    """How far the chart's sizings are from what actually happened, in bb."""
    return sum(abs(size_of(ca) - size_of(aa))
               for (_, ca), (_, aa) in zip(chart_seq, actual_seq) if is_raise(ca))


def read_weight(path, hand):
    for entry in open(path).read().strip().split(','):
        h, _, w = entry.partition(':')
        if h == hand:
            return float(w) if w else 1.0
    return 0.0


def describe(action, bb=None, to_call=None):
    """Name an action, in chips too when we know the blind -- 'RAISE to 2.5bb' is not
    something you can type into a table, '50' is."""
    if action == 'FOLD':
        return 'FOLD'
    if action == 'Call':
        return 'CALL' + (f" {to_call:g}" if to_call else '')
    if action == 'AllIn':
        return 'ALL-IN'
    if bb:
        return f"RAISE to {size_of(action) * bb:g} ({action})"
    return f"RAISE to {action}"


def advise(state, bb_override=None):
    if state.get('street') != 'preflop':
        raise Unsupported(f"street is {state.get('street')!r}; this only reads preflop charts")

    seats = seats_of(state)
    hero_seat = seats['hero']
    hand = hand_class(state['hero']['cards'])
    bb = big_blind(state, seats, bb_override)
    # `or` rather than a .get default: the API sends small_blind explicitly as null
    # when the table omits it, and float(None) is a crash, not a fallback.
    sb = float(state.get('small_blind') or bb / 2)
    check_seating(state, seats, sb, bb)
    prior = actions_so_far(state, seats, bb)

    charts = load_charts(hero_seat)
    # The option set: every chart whose line -- seen the way a snapshot shows it, each
    # player's final action only -- is exactly the action so far plus one more action,
    # taken by hero. Hero's earlier raises stay in: facing a 3-bet, the chart stem
    # starts with hero's own open.
    options = [(seq, snapshot_line(seq[:-1]), path) for seq, path in charts
               if seq[-1][0] == hero_seat]
    options = [(seq, vis, path) for seq, vis, path in options if same_shape(vis, prior)]
    if not options:
        raise Unsupported(
            f"no chart covers {hero_seat} facing "
            f"{' '.join(f'{s} {a}' for s, a in prior) or 'no action'}")

    # Several sizings of the same line may match (e.g. a 2.5bb and a 3bb open); take the
    # chart whose sizes are closest to the table's, and report how far off it was.
    best = min(size_distance(vis, prior) for _, vis, _ in options)
    options = [(seq, vis, path) for seq, vis, path in options
               if size_distance(vis, prior) == best]

    strategy = sorted(((seq[-1][1], read_weight(path, hand), path)
                       for seq, _, path in options),
                      key=lambda x: -x[1])
    top = max(state[p]['bet'] for p in TABLE_ORDER if state[p].get('active', True))
    stack_bb = state['hero']['stack'] / bb

    # Anything that makes the answer less trustworthy than it looks. Built here rather than
    # at print time so the API hands back the same caveats the CLI shows.
    warnings = []
    if best > 1e-9:
        warnings.append(
            f"Sizing snapped: the chart line is {best:.1f}bb away from what was actually bet.")
    if abs(stack_bb - CHART_DEPTH_BB) > 0.15 * CHART_DEPTH_BB:
        warnings.append(
            f"Stack mismatch: hero is {stack_bb:.0f}bb deep but the charts are "
            f"{CHART_DEPTH_BB:.0f}bb. Short stacks play a different game (fewer flats, more "
            f"jams) -- treat this as a rough guide only.")
    if abs(sum(w for _, w, _ in strategy) - 1.0) > 0.02:
        warnings.append("The option set may be incomplete: the frequencies do not sum to 1.")

    return {
        'seats': seats, 'hero_seat': hero_seat, 'hand': hand, 'bb': bb, 'sb': sb,
        'prior': prior, 'strategy': strategy, 'snap_error': best,
        'stack_bb': stack_bb, 'chart_line': options[0][0][:-1],
        'to_call': top - state['hero']['bet'], 'warnings': warnings,
    }


def report(state, r):
    bb, call = r['bb'], r['to_call']
    seat_str = ', '.join(f"{p.replace('villain_', '')}={r['seats'][p]}" for p in TABLE_ORDER)
    print(f"Table    : {seat_str}   (blinds {r['sb']:g}/{bb:g})")
    print(f"Hero     : {r['hand']} in the {r['hero_seat']}, {r['stack_bb']:.0f}bb deep")
    line = ' -> '.join(f"{s} {describe(a, bb)}" for s, a in r['prior']) or 'folds to hero'
    print(f"Action   : {line}")

    if r['prior']:
        chart_line = ' -> '.join(f"{s} {describe(a, bb)}" for s, a in r['chart_line'])
        snap = f"  [chart line: {chart_line}]"
        if r['snap_error'] > 1e-9:
            snap += f"  -- snapped, chart sizing is {r['snap_error']:.1f}bb off the table's"
        print(f"Chart    :{snap}")

    print()
    for action, weight, _ in r['strategy']:
        bar = '#' * int(round(weight * 40))
        print(f"  {describe(action, bb, call):24} {weight * 100:5.1f}%  {bar}")

    best_action, best_weight, _ = r['strategy'][0]
    print()
    if best_weight <= 1e-9:
        print(f"=> {describe(best_action, bb, call)} -- every option is 0%: the chart does "
              f"not play {r['hand']} here at all")
    elif best_weight >= 0.999:
        print(f"=> {describe(best_action, bb, call)}   (pure, 100%)")
    else:
        mix = ', '.join(f"{describe(a, bb, call)} {w * 100:.0f}%"
                        for a, w, _ in r['strategy'] if w > 0)
        print(f"=> mixed: {mix}")
        print(f"   most of the time: {describe(best_action, bb, call)}")

    for warning in r['warnings']:
        print(f"\n! {warning}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('state', nargs='?', help="game-state json; omit to read stdin")
    parser.add_argument('--bb', type=float, help="big blind, if it cannot be inferred")
    args = parser.parse_args()

    raw = open(args.state).read() if args.state else sys.stdin.read()
    state = json.loads(raw)

    try:
        report(state, advise(state, args.bb))
    except Unsupported as e:
        print(f"No chart for this spot: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
