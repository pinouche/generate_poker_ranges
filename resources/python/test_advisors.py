"""Feed hand-built table snapshots through both advisors and check the answers make sense.

Every test constructs the same json the table would POST to /advise (chips, not bb:
blinds 1/2, 100bb = 200-chip stacks) and asserts poker sanity rather than exact solver
frequencies: premiums never fold, trash folds, frequencies sum to 1, the seats come out
where the dealer button says they must.

Run with output visible (each test prints the spot and the recommendation it got):

    pytest resources/python/test_advisors.py -v -s
"""

import itertools
import os
import sys
import time
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heuristic_advisor as heuristic
import hu_advisor as hu
import postflop_advisor as postflop
from preflop_advisor import Unsupported, advise as preflop_advise

BB = 2.0        # chips; blinds 1/2, so 100bb deep = 200 chips
STACK = 200.0


def card(c):
    """'Ah' -> {'rank': 'A', 'suit': 'h'}, the table's card encoding."""
    return {'rank': c[0], 'suit': c[1]}


def player(cards=None, stack=STACK, bet=0.0, active=True):
    return {'cards': [card(c) for c in (cards or [])],
            'stack': stack, 'bet': bet, 'active': active}


def table(dealer, street, hero, left, right, board=(), pot=0.0):
    return {'dealer': dealer, 'street': street, 'big_blind': BB, 'small_blind': BB / 2,
            'pot': pot, 'board': [card(c) for c in board],
            'hero': hero, 'villain_left': left, 'villain_right': right}


def weights(r):
    """{action: frequency} out of an advise() result, either advisor."""
    return {a: w for a, w, *_ in r['strategy']}


def top_action(r):
    return r['strategy'][0][0]


def check_distribution(r):
    w = weights(r)
    assert all(-1e-9 <= v <= 1 + 1e-9 for v in w.values()), f"weight out of [0,1]: {w}"
    assert abs(sum(w.values()) - 1.0) < 0.02, f"frequencies do not sum to 1: {w}"


def show(label, r):
    line = ' -> '.join(f"{s} {a}" for s, a in r.get('prior', r.get('line', []))) or 'first in'
    mix = ', '.join(f"{a} {w * 100:.0f}%" for a, w, *_ in r['strategy'] if w > 0.005)
    print(f"\n  {label:52} [{line}]  =>  {mix}")


# ---------------------------------------------------------------------------
# Preflop: unopened pots.
# ---------------------------------------------------------------------------

def btn_first_in(hand):
    """Hero on the button, blinds posted, nobody has acted."""
    return table('hero', 'preflop',
                 hero=player(hand),
                 left=player(bet=1),      # SB
                 right=player(bet=2))     # BB


def test_btn_opens_premium():
    r = preflop_advise(btn_first_in(['Ah', 'Ks']))
    show('BTN first in with AKo', r)
    check_distribution(r)
    assert top_action(r) == '2.5bb', "AKo must open from the button"
    assert weights(r)['2.5bb'] > 0.99


def test_btn_folds_trash():
    r = preflop_advise(btn_first_in(['7h', '2s']))
    show('BTN first in with 72o', r)
    check_distribution(r)
    assert top_action(r) == 'FOLD', "72o is the worst hand in poker; the button folds it"
    assert weights(r)['FOLD'] > 0.9


def test_sb_opens_after_btn_folds():
    # BTN folded; hero is next to the dealer's left, so dealer must be villain_right.
    state = table('villain_right', 'preflop',
                  hero=player(['Ah', 'Qh'], bet=1),          # SB
                  left=player(bet=2),                        # BB
                  right=player(bet=0, active=False))         # BTN, folded
    r = preflop_advise(state)
    show('SB first in with AQs (BTN folded)', r)
    check_distribution(r)
    assert r['hero_seat'] == 'SB'
    assert top_action(r) != 'FOLD', "AQs never folds first-in from the SB"


# ---------------------------------------------------------------------------
# Preflop: facing an open.
# ---------------------------------------------------------------------------

def bb_facing_btn_open(hand, open_chips=5.0):
    """Hero in the BB; BTN (villain_left) opened, SB folded."""
    return table('villain_left', 'preflop',
                 hero=player(hand, bet=2),                       # BB
                 left=player(bet=open_chips),                    # BTN, opener
                 right=player(bet=1, active=False))              # SB, folded


def test_bb_defends_suited_connector():
    # This 500-rake pack 3-bets 87s most of the time rather than calling (rake punishes
    # flatting); what must hold either way is that 87s continues.
    r = preflop_advise(bb_facing_btn_open(['8h', '7h']))
    show('BB with 87s vs BTN 2.5bb open', r)
    check_distribution(r)
    w = weights(r)
    assert r['prior'] == [('BTN', '2.5bb')]
    assert w.get('FOLD', 0) < 0.15, "87s defends the BB against a 2.5x open"
    assert top_action(r) in ('Call', '11.0bb')


def test_bb_folds_junk():
    r = preflop_advise(bb_facing_btn_open(['7c', '2d']))
    show('BB with 72o vs BTN 2.5bb open', r)
    check_distribution(r)
    assert top_action(r) == 'FOLD'
    assert weights(r)['FOLD'] > 0.9


def test_bb_3bets_premium():
    r = preflop_advise(bb_facing_btn_open(['Kd', 'Kc']))
    show('BB with KK vs BTN 2.5bb open', r)
    check_distribution(r)
    w = weights(r)
    assert w.get('FOLD', 0) < 0.01, "KK never folds to a single raise"
    raise_weight = sum(v for a, v in w.items() if a.endswith('bb'))
    assert raise_weight > 0.5, f"KK 3-bets the BB most of the time, got {w}"


def test_bb_open_size_snaps():
    # The table opened to 6 (3bb); the chart only has a 2.5bb line. It should still
    # answer, and say it snapped.
    r = preflop_advise(bb_facing_btn_open(['8h', '7h'], open_chips=6.0))
    show('BB with 87s vs BTN 3bb open (snapped)', r)
    check_distribution(r)
    assert r['snap_error'] > 0
    assert any('snapped' in w.lower() for w in r['warnings'])


def test_sb_is_3bet_or_fold():
    # Hero in the SB facing a BTN open: this pack has no SB flat, so the option set
    # must be exactly raise-or-fold.
    state = table('villain_right', 'preflop',
                  hero=player(['Ad', 'Qd'], bet=1),              # SB
                  left=player(bet=2),                            # BB
                  right=player(bet=5))                           # BTN, opener
    r = preflop_advise(state)
    show('SB with AQs vs BTN 2.5bb open', r)
    check_distribution(r)
    actions = set(weights(r))
    assert 'Call' not in actions, "the SB never flats a BTN open in this pack"
    assert top_action(r) == '11.0bb', "AQs 3-bets when the SB cannot flat"


# ---------------------------------------------------------------------------
# Preflop: facing a 3-bet / 4-bet (hero's own earlier raise is part of the line).
# ---------------------------------------------------------------------------

def btn_facing_sb_3bet(hand):
    """Hero opened the button to 5 (2.5bb), SB 3-bet to 22 (11bb), BB folded."""
    return table('hero', 'preflop',
                 hero=player(hand, bet=5, stack=195),
                 left=player(bet=22, stack=178),                 # SB, 3-bettor
                 right=player(bet=2, active=False))              # BB, folded


def test_btn_vs_3bet_option_set():
    r = preflop_advise(btn_facing_sb_3bet(['Ah', 'As']))
    show('BTN with AA vs SB 3-bet to 11bb', r)
    check_distribution(r)
    w = weights(r)
    assert set(w) == {'24.0bb', 'Call', 'FOLD'}, f"BTN vs 3-bet is 4bet/call/fold, got {set(w)}"
    assert w['FOLD'] < 0.01, "AA never folds to a 3-bet"
    assert w['24.0bb'] > 0.3, f"AA 4-bets a healthy share of the time, got {w}"


def test_btn_vs_3bet_folds_weak_open():
    r = preflop_advise(btn_facing_sb_3bet(['Jh', '6h']))
    show('BTN with J6s vs SB 3-bet to 11bb', r)
    check_distribution(r)
    assert top_action(r) == 'FOLD', "J6s opens the button but folds to a 3-bet"


def test_sb_vs_4bet():
    # Hero 3-bet from the SB to 22 (11bb), BTN 4-bet to 48 (24bb), BB long folded.
    # Hero's own 3-bet (and the BTN's superseded open) must be reconstructed.
    state = table('villain_right', 'preflop',
                  hero=player(['As', 'Ac'], bet=22, stack=178),  # SB
                  left=player(bet=2, active=False),              # BB, folded
                  right=player(bet=48, stack=152))               # BTN, 4-bettor
    r = preflop_advise(state)
    show('SB with AA vs BTN 4-bet to 24bb', r)
    check_distribution(r)
    w = weights(r)
    assert w.get('FOLD', 0) < 0.01, "AA never folds to a 4-bet"
    assert w.get('AllIn', 0) + w.get('Call', 0) > 0.99


def test_btn_calls_5bet_shove_with_aces():
    # Hero opened, BB 3-bet to 22, hero 4-bet to 48, BB shoved the rest (bet 200,
    # nothing behind). A shove is the chart's 'AllIn' node whatever its size.
    state = table('hero', 'preflop',
                  hero=player(['As', 'Ac'], bet=48, stack=152),
                  left=player(bet=1, active=False),              # SB, folded
                  right=player(bet=200, stack=0))                # BB, shoved
    r = preflop_advise(state)
    show('BTN with AA vs BB 5-bet shove', r)
    check_distribution(r)
    w = weights(r)
    assert set(w) == {'Call', 'FOLD'}
    assert w['Call'] > 0.95, f"AA snap-calls a 5-bet shove, got {w}"


# ---------------------------------------------------------------------------
# Preflop: plumbing -- seats, blinds, unsupported spots.
# ---------------------------------------------------------------------------

def test_seats_follow_the_dealer():
    # Play moves clockwise from the dealer: the player on the dealer's left posts the
    # small blind. villain_left sits on hero's left, so the rotation is fixed.
    from preflop_advisor import seats_of
    for dealer, want in [
            ('hero', {'hero': 'BTN', 'villain_left': 'SB', 'villain_right': 'BB'}),
            ('villain_left', {'villain_left': 'BTN', 'villain_right': 'SB', 'hero': 'BB'}),
            ('villain_right', {'villain_right': 'BTN', 'hero': 'SB', 'villain_left': 'BB'})]:
        assert seats_of({'dealer': dealer}) == want


def test_big_blind_inferred_from_posts():
    state = btn_first_in(['Ah', 'Ks'])
    state['big_blind'] = None
    state['small_blind'] = None
    r = preflop_advise(state)
    assert r['bb'] == 2.0, "with untouched blinds of 1/2 the bb is inferable"


def test_swapped_seating_is_caught():
    # villain_left posted the BB and villain_right the SB -- the opposite of what
    # dealer='hero' implies. The advisor must refuse rather than read the wrong chart.
    state = table('hero', 'preflop',
                  hero=player(['Ah', 'Ks']),
                  left=player(bet=2),      # says SB seat, posted 2
                  right=player(bet=1))     # says BB seat, posted 1
    with pytest.raises(Unsupported, match='seating'):
        preflop_advise(state)


def test_bb_checks_a_limp():
    # SB completes to 2 instead of raising or folding. The pack has no limp branch, but
    # hero closes the action in the BB and can check for free, so we recommend that rather
    # than going silent -- flagged as a heuristic.
    state = table('villain_left', 'preflop',
                  hero=player(['Ah', 'Ks'], bet=2),              # BB
                  left=player(bet=0, active=False),              # BTN folded
                  right=player(bet=2))                           # SB limped
    r = preflop_advise(state)
    show('BB facing a SB limp with AKo', r)
    check_distribution(r)
    assert r['hero_seat'] == 'BB'
    assert top_action(r) == 'Check', "the BB can always take a free flop against a limp"
    assert any('heuristic' in w.lower() for w in r['warnings'])


def test_sb_iso_raises_a_limp():
    # BTN limps; hero is the SB with the BB still behind. No limp branch, so we fall back
    # to hero's raise-first-in range as the iso range and size up 1bb per limper (-> 4bb).
    state = table('villain_right', 'preflop',
                  hero=player(['Ah', 'Ks'], bet=1),              # SB
                  left=player(bet=2),                            # BB
                  right=player(bet=2))                           # BTN limped
    r = preflop_advise(state)
    show('SB facing a BTN limp with AKo', r)
    check_distribution(r)
    assert r['hero_seat'] == 'SB'
    assert top_action(r) == '4.0bb', "AKo isolates a limper (3bb open + 1bb per limper)"
    assert any('heuristic' in w.lower() for w in r['warnings'])


def test_sb_folds_trash_to_a_limp():
    state = table('villain_right', 'preflop',
                  hero=player(['7h', '2s'], bet=1),              # SB
                  left=player(bet=2),                            # BB
                  right=player(bet=2))                           # BTN limped
    r = preflop_advise(state)
    show('SB facing a BTN limp with 72o', r)
    check_distribution(r)
    assert top_action(r) == 'FOLD', "72o folds even to a limp"


def test_short_stack_warns():
    state = btn_first_in(['Ah', 'Ks'])
    state['hero']['stack'] = 50   # 25bb deep against 100bb charts
    r = preflop_advise(state)
    assert any('Stack mismatch' in w for w in r['warnings'])


# ---------------------------------------------------------------------------
# Postflop: flop. BTN_vs_BB single-raised pot (2.5bb open, BB call), flop-start pot
# 5.5bb = 11 chips; Ad Kc 3c is in the solved subset, so the board match is exact.
# ---------------------------------------------------------------------------

FLOP = ('Ad', 'Kc', '3c')
FLOP_POT = 11.0        # 5.5bb
POST_STACK = 195.0     # 97.5bb behind after the preflop


def btn_on_flop(hand, hero_bet=0.0, vill_bet=0.0, board=FLOP, pot=None):
    """Hero is the BTN (in position); BB (villain_right) is the live player."""
    return table('hero', 'flop',
                 hero=player(hand, stack=POST_STACK - hero_bet, bet=hero_bet),
                 left=player(bet=0, active=False),               # SB folded preflop
                 right=player(stack=POST_STACK - vill_bet, bet=vill_bet),
                 board=board,
                 pot=FLOP_POT + hero_bet + vill_bet if pot is None else pot)


def bb_on_flop(hand, hero_bet=0.0, vill_bet=0.0, board=FLOP, pot=None):
    """Hero is the BB (out of position); BTN (villain_left) is the live player."""
    return table('villain_left', 'flop',
                 hero=player(hand, stack=POST_STACK - hero_bet, bet=hero_bet),
                 left=player(stack=POST_STACK - vill_bet, bet=vill_bet),
                 right=player(bet=0, active=False),              # SB folded preflop
                 board=board,
                 pot=FLOP_POT + hero_bet + vill_bet if pot is None else pot)


@pytest.fixture(autouse=True)
def fresh_hand_memory():
    postflop._MEMORY.clear()


def test_flop_ip_bets_top_set():
    r = postflop.advise(btn_on_flop(['Ah', 'As']))
    show('BTN with AA (top set) on AK3tt, checked to', r)
    check_distribution(r)
    assert r['scenario'] == 'BTN_vs_BB'
    assert r['board_exact'], "Ad Kc 3c is in the solved subset"
    assert r['line'] == [('BB', 'CHECK')], "both bets zero with hero IP means BB checked"
    w = weights(r)
    bet = sum(v for a, v in w.items() if a.startswith('BET'))
    assert bet > 0.5, f"top set bets for value when checked to, got {w}"


def test_flop_oop_checks_junk():
    r = postflop.advise(bb_on_flop(['9h', '8h']))
    show('BB with 98s (air) on AK3tt, first to act', r)
    check_distribution(r)
    assert r['position'] == 'OOP'
    assert r['line'] == [], "hero is first to act"
    assert weights(r).get('CHECK', 0) > 0.7, "the BB checks air on an ace-high board"


def test_flop_facing_bet_set_never_folds():
    # BB checked, BTN bet half pot (2.75bb = 5.5 chips), hero holds bottom set.
    r = postflop.advise(bb_on_flop(['3h', '3s'], vill_bet=5.5))
    show('BB with 33 (set) on AK3tt vs half-pot bet', r)
    check_distribution(r)
    w = weights(r)
    assert w.get('FOLD', 0) < 0.05, f"a set never folds to one flop bet, got {w}"
    assert 'FOLD' in w and any(a.startswith(('CALL', 'RAISE')) for a in w), \
        "facing a bet the options are fold/call/raise"


def test_flop_facing_bet_air_mostly_folds():
    r = postflop.advise(bb_on_flop(['9h', '8h'], vill_bet=5.5))
    show('BB with 98s (air) on AK3tt vs half-pot bet', r)
    check_distribution(r)
    assert weights(r).get('FOLD', 0) > 0.5, "no pair, no draw folds to a bet most of the time"


def test_flop_board_suit_isomorphism():
    # Ah Kd 3d is Ad Kc 3c with the suits relabelled: strategically identical, so the
    # match must be exact, not a texture approximation.
    r = postflop.advise(btn_on_flop(['Ac', 'As'], board=('Ah', 'Kd', '3d')))
    show('BTN with AA on AhKd3d (isomorph of solve)', r)
    check_distribution(r)
    assert r['board_exact']
    assert r['solve_board'] == 'Ad_Kc_3c'
    assert not any('snapped' in w.lower() for w in r['warnings'])


def test_flop_board_snaps_to_nearest_texture():
    # AK4 two-tone is not in the subset; the advisor should stand in the nearest
    # same-texture solve and say so.
    r = postflop.advise(btn_on_flop(['Ah', 'As'], board=('Ad', 'Kc', '4c')))
    show('BTN with AA on AdKc4c (not solved, snaps)', r)
    check_distribution(r)
    assert not r['board_exact']
    assert any('Board snapped' in w for w in r['warnings'])


# ---------------------------------------------------------------------------
# Postflop: turn.
# ---------------------------------------------------------------------------

def test_turn_after_watched_flop():
    # The same hand asks about the flop first (hero BB, first to act), then the turn
    # after it went check-check. The server watched the flop, so the line is observed.
    postflop.advise(bb_on_flop(['3h', '3s']))
    turn = bb_on_flop(['3h', '3s'])
    turn['street'] = 'turn'
    turn['board'].append(card('8s'))
    r = postflop.advise(turn)
    show('BB with 33, AK3tt flop went check-check, turn 8s', r)
    check_distribution(r)
    assert r['street'] == 'turn'
    assert r['flop_line_source'] == 'observed'
    assert [a for _, a in r['flop_line']] == ['CHECK', 'CHECK']
    w = weights(r)
    assert sum(v for a, v in w.items() if a.startswith('BET')) > 0.3, \
        f"a set bets the turn plenty after it checked through, got {w}"


def test_turn_line_inferred_from_pot():
    # No flop request was seen (server restart). The turn pot carries the flop betting:
    # 5.5bb start + 2bb bet + 2bb call = 9.5bb = 19 chips.
    turn = bb_on_flop(['3h', '3s'], pot=19.0)
    turn['street'] = 'turn'
    turn['board'].append(card('8s'))
    r = postflop.advise(turn)
    show('BB with 33, turn 8s, flop line inferred from pot', r)
    check_distribution(r)
    assert r['flop_line_source'] == 'inferred'
    invested = {a for _, a in r['flop_line']}
    assert any(a.startswith('BET') for a in invested) and any(
        a.startswith('CALL') for a in invested), \
        "a 9.5bb turn pot means 2bb went in each on the flop: a bet and a call"


# ---------------------------------------------------------------------------
# Postflop: turn cards the solver resolved by suit isomorphism.
#
# `set_dump_rounds 2` writes a subtree under all 52 `dealcards` keys, but the solver
# resolves some turn cards by isomorphism and writes a STUB in their place: node_type,
# player, actions, childrens, and no strategy. A stub is shaped like a tree, so it
# survives `if not root` and used to reach `node['strategy']` as a KeyError -- a 500
# rather than a 422, on ~30% of turn nodes. dumped_suit_map redirects the turn onto an
# interchangeable suit that was dumped; has_strategy is the backstop.
#
# Qc 8c 4c is monotone and in the subset, and every spade turn on it is a stub.
# ---------------------------------------------------------------------------

MONOTONE = ('Qc', '8c', '4c')


def btn_on_turn(hand, turn, board=MONOTONE, flop_bet=2.0):
    """Hero IP on the turn, after a flop bet and call of `flop_bet` bb each."""
    state = btn_on_flop(hand, board=board,
                        pot=FLOP_POT + 2 * flop_bet * BB)
    state['street'] = 'turn'
    state['board'].append(card(turn))
    return state


def redirected_to(r):
    """The suit the turn was actually played as, out of the redirect warning."""
    import re
    for w in r['warnings']:
        found = re.search(r'played as ([AKQJT98765432][cdhs])', w)
        if found:
            return found.group(1)
    return None


def test_turn_on_an_isomorphic_suit_is_answered():
    r = postflop.advise(btn_on_turn(['Tc', '9h'], 'As'))
    show('BTN on Qc8c4c, turn As (a suit the solve stubbed)', r)
    check_distribution(r)
    assert redirected_to(r), f"the redirect should say so, got {r['warnings']}"


def test_turn_suit_redirect_lands_on_the_same_spot():
    """The redirect must be a relabelling, not an approximation.

    Hero is given Tc 9h -- a club and a heart -- so that exchanging spades for diamonds
    fixes hero's cards AND the monotone club board. Under that relabelling a spade turn
    and a diamond turn are literally the same node, so the two answers have to be
    identical, not merely close. The club ace is a fourth club and must differ.
    """
    answers = {suit: weights(postflop.advise(btn_on_turn(['Tc', '9h'], 'A' + suit)))
               for suit in 'sdc'}
    print("\n  CHECK by turn suit on Qc8c4c, hero Tc9h: " +
          '  '.join(f"A{s} {w.get('CHECK', 0.0):.4f}" for s, w in answers.items()))
    assert answers['s'] == answers['d'], (
        "As and Ad are one node under the s<->d relabelling that fixes this board and "
        f"this hand; got {answers['s']} vs {answers['d']}")
    assert abs(answers['c'].get('CHECK', 0.0) - answers['d'].get('CHECK', 0.0)) > 0.01, \
        f"the club ace completes the flush texture and must differ, got {answers}"


def test_hero_suits_follow_the_turn_redirect():
    """Moving the turn's suit without moving hero's would destroy the one relationship
    that matters most on a turn card: whether hero's hand shares its suit."""
    r = postflop.advise(btn_on_turn(['9s', '7s'], 'Ks'))
    show('BTN with 9s7s on Qc8c4c, turn Ks (redirected)', r)
    turn = redirected_to(r)
    assert turn, f"this turn should have been redirected, got {r['warnings']}"
    solved = r['hand_solved']
    assert solved[1] == solved[3], f"hero's two cards stay suited, got {solved}"
    assert solved[1] == turn[1], (
        f"hero held the turn's suit, so the translation must keep that: hero is "
        f"{solved} on a {turn} turn")


def test_has_strategy_rejects_a_stub():
    real = {'node_type': 'action_node', 'player': 0, 'actions': ['CHECK'],
            'strategy': {'actions': ['CHECK'], 'strategy': {}}}
    stub = {'node_type': 'action_node', 'player': 0, 'actions': ['CHECK'],
            'childrens': {}}
    assert postflop.has_strategy(real)
    assert not postflop.has_strategy(stub)
    assert not postflop.has_strategy(None)


def test_dumped_suit_map_leaves_a_board_suit_alone():
    """A suit the flop holds carries the flush relationship and can never be swapped."""
    flop = [postflop.parse_card(c) for c in ('Qc', '8c', '4c')]
    identity = {s: s for s in postflop.SUITS}
    chance = {'dealcards': {}}   # nothing dumped at all
    smap, changed = postflop.dumped_suit_map(('A', 'c'), identity, chance, set(flop))
    assert not changed and smap == identity, "clubs are on the board; not interchangeable"


# ---------------------------------------------------------------------------
# Postflop: spots the solves cannot answer must refuse, not guess.
# ---------------------------------------------------------------------------

def test_river_unsupported():
    state = btn_on_flop(['Ah', 'As'], board=('Ad', 'Kc', '3c'))
    state['street'] = 'river'
    with pytest.raises(Unsupported, match='river'):
        postflop.advise(state)


def test_multiway_flop_unsupported():
    state = btn_on_flop(['Ah', 'As'])
    state['villain_left']['active'] = True   # three players saw the flop
    with pytest.raises(Unsupported, match='heads-up'):
        postflop.advise(state)


# ---------------------------------------------------------------------------
# Heads-up: the third player busted, so the Nash push/fold tables take over.
# ---------------------------------------------------------------------------

def busted_seat():
    """An empty chair: no cards, no chips, no bet -- unlike a fold, which keeps a stack."""
    return player(stack=0.0, bet=0.0, active=False)


def hu_sb_first_in(hand, stack_bb=10.0):
    """Hero on the button (= the SB heads-up), blinds posted, nobody has acted."""
    return table('hero', 'preflop',
                 hero=player(hand, stack=stack_bb * BB - 1, bet=1),   # SB posted
                 left=player(stack=stack_bb * BB - 2, bet=2),         # BB posted
                 right=busted_seat())


def hu_bb_facing_jam(hand, stack_bb=10.0):
    """Hero in the BB; the SB (villain_left, on the button) has jammed."""
    return table('villain_left', 'preflop',
                 hero=player(hand, stack=stack_bb * BB - 2, bet=2),
                 left=player(stack=0.0, bet=stack_bb * BB),           # all-in
                 right=busted_seat())


def test_heads_up_detected_only_on_a_busted_seat():
    assert hu.is_heads_up(hu_sb_first_in(['Ah', 'Ad']))
    assert not hu.is_heads_up(btn_first_in(['Ah', 'Ad'])), "three live players"
    folded = bb_facing_btn_open(['Ah', 'Ad'])   # SB folded but still has chips
    assert not hu.is_heads_up(folded), "a folded player has not busted"


def test_hu_sb_jams_aces_at_any_depth():
    for stack_bb in (3.0, 10.0, 100.0):
        r = hu.advise(hu_sb_first_in(['Ah', 'Ad'], stack_bb))
        assert top_action(r) == 'AllIn', f"AA always jams at {stack_bb}bb"
        assert weights(r)['AllIn'] > 0.99
    show('HU SB first in with AA, 10bb', hu.advise(hu_sb_first_in(['Ah', 'Ad'])))


def test_hu_sb_stack_depth_moves_the_line():
    # Q7o is a Nash push at 10bb and a fold at 20bb; 72o pushes at neither.
    short = hu.advise(hu_sb_first_in(['Qh', '7s'], stack_bb=10.0))
    show('HU SB first in with Q7o, 10bb', short)
    check_distribution(short)
    assert top_action(short) == 'AllIn'

    deep = hu.advise(hu_sb_first_in(['Qh', '7s'], stack_bb=20.0))
    show('HU SB first in with Q7o, 20bb', deep)
    assert top_action(deep) == 'FOLD'

    assert top_action(hu.advise(hu_sb_first_in(['7h', '2s']))) == 'FOLD'


def test_hu_bb_calls_jam_with_premium_folds_trash():
    r = hu.advise(hu_bb_facing_jam(['Ah', 'Ad']))
    show('HU BB vs 10bb jam with AA', r)
    check_distribution(r)
    assert r['hero_seat'] == 'BB'
    assert top_action(r) == 'Call' and weights(r)['Call'] > 0.99

    r = hu.advise(hu_bb_facing_jam(['7h', '2s']))
    show('HU BB vs 10bb jam with 72o', r)
    assert top_action(r) == 'FOLD'


def test_hu_deep_stack_answers_with_a_warning():
    r = hu.advise(hu_sb_first_in(['Ah', 'Ad'], stack_bb=100.0))
    assert any('jam-or-fold' in w for w in r['warnings']), \
        "100bb is far beyond push/fold territory; the answer must say so"


def test_hu_effective_stack_is_the_short_stack():
    # Hero covers a 5bb villain: the 5bb table row decides, and Q7o pushes there.
    state = hu_sb_first_in(['Qh', '7s'], stack_bb=50.0)
    state['villain_left'] = player(stack=5 * BB - 2, bet=2)
    r = hu.advise(state)
    show('HU SB with Q7o, 50bb vs a 5bb stack', r)
    assert abs(r['eff_bb'] - 5.0) < 1e-9
    assert top_action(r) == 'AllIn'


def test_hu_normal_raise_unsupported():
    # The SB min-raises with chips behind: the tables only answer a jam.
    state = hu_bb_facing_jam(['Ah', 'Ad'])
    state['villain_left'] = player(stack=16, bet=4)
    with pytest.raises(Unsupported, match='all-in'):
        hu.advise(state)


def test_hu_postflop_unsupported():
    state = hu_sb_first_in(['Ah', 'Ad'])
    state['street'] = 'flop'
    with pytest.raises(Unsupported, match='postflop'):
        hu.advise(state)


def hu_bb_facing_limp(hand, stack_bb=10.0):
    """Hero in the BB; the SB (on the button) completed to the big blind."""
    return table('villain_left', 'preflop',
                 hero=player(hand, stack=stack_bb * BB - 2, bet=2),
                 left=player(stack=stack_bb * BB - 2, bet=2),         # limped
                 right=busted_seat())


def test_hu_bb_vs_limp_checks_rather_than_refusing():
    r = hu.advise(hu_bb_facing_limp(['7h', '2s']))
    show('HU BB vs a 10bb limp with 72o', r)
    assert r['hero_seat'] == 'BB'
    assert top_action(r) == 'Check', "checking is free and closes the action"
    assert r['to_call'] == 0.0
    assert any('limp branch' in w for w in r['warnings']), \
        "the answer is a floor, not a solved node, and must say so"


def test_hu_bb_vs_limp_flags_jam_candidates_only_from_the_push_range():
    strong = hu.advise(hu_bb_facing_limp(['Ah', 'Ad']))
    assert any('candidate to jam' in w for w in strong['warnings']), \
        "AA is in the first-in jam range at 10bb"
    assert any('upper bound' in w for w in strong['warnings']), \
        "the push range is drawn against folding, not against a free flop"
    assert top_action(strong) == 'Check', "the floor stands; the jam is only flagged"

    weak = hu.advise(hu_bb_facing_limp(['7h', '2s']))
    assert not any('candidate to jam' in w for w in weak['warnings'])


def test_hu_bb_vs_limp_deep_skips_the_push_range_entirely():
    r = hu.advise(hu_bb_facing_limp(['Ah', 'Ad'], stack_bb=60.0))
    assert not any('candidate to jam' in w for w in r['warnings']), \
        "60bb is nowhere near jam-or-fold; the push table has nothing to say"
    assert any('past jam-or-fold territory' in w for w in r['warnings'])


def test_hu_sb_all_in_under_a_big_blind_is_not_read_as_a_limp():
    # A 1bb SB posts 0.5 and jams the other 0.5: total bet is one blind, the same size
    # as a limp. Hero's posted blind covers it either way, so there is no decision --
    # and the reason must say that rather than blaming a missing limp branch.
    state = hu_bb_facing_limp(['Ah', 'Ad'])
    state['villain_left'] = player(stack=0.0, bet=BB)
    with pytest.raises(Unsupported, match='nothing to decide'):
        hu.advise(state)


# ---------------------------------------------------------------------------
# The HTTP endpoint: same spots, through api.py's models and conversion.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient
    from api import app
    return TestClient(app)


def test_api_preflop(client):
    resp = client.post('/advise', json=btn_first_in(['Ah', 'Ks']))
    assert resp.status_code == 200, resp.text
    a = resp.json()
    print(f"\n  API: BTN first in with AKo  =>  {a['recommendation']['action']}")
    assert a['hand'] == 'AKo'
    assert a['hero_cards'] == 'A♥ K♠', "the chart class is AKo; the cards say which AKo"
    assert a['recommendation']['kind'] == 'RAISE'
    assert a['recommendation']['chips'] == 5.0, "2.5bb at a 2-chip blind is 5 chips"
    assert a['pure'] is True


def test_api_postflop(client):
    resp = client.post('/advise', json=bb_on_flop(['3h', '3s'], vill_bet=5.5))
    assert resp.status_code == 200, resp.text
    a = resp.json()
    print(f"\n  API: BB with 33 vs flop bet  =>  {a['recommendation']['action']}")
    assert a['recommendation']['kind'] in ('CALL', 'RAISE'), "a set never folds here"
    kinds = {o['kind'] for o in a['options']}
    assert 'FOLD' in kinds, "facing a bet, folding is at least on the option list"


def test_api_river_falls_back_to_heuristic(client):
    # The solves stop at the turn, but a real river with a full board still gets an
    # answer -- the equity heuristic -- flagged as such in the warnings.
    state = btn_on_flop(['Ah', 'As'], board=FLOP + ('7h', '2d'), pot=FLOP_POT)
    state['street'] = 'river'
    resp = client.post('/advise', json=state)
    assert resp.status_code == 200, resp.text
    a = resp.json()
    print(f"\n  API: river with AA (heuristic)  =>  {a['recommendation']['action']}")
    assert a['recommendation']['kind'] in ('BET', 'ALLIN'), "top set on a blank river bets"
    assert a['hero_cards'] == 'A♥ A♠', "the fallback answer carries the cards too"
    assert any('heuristic' in w for w in a['warnings'])


def test_api_river_with_short_board_still_422(client):
    # A 'river' with three board cards is malformed for the heuristic too, so the
    # original reason (the solves stop at the turn) is the one reported.
    state = btn_on_flop(['Ah', 'As'])
    state['street'] = 'river'
    resp = client.post('/advise', json=state)
    assert resp.status_code == 422
    assert 'river' in resp.json()['detail']


def test_api_heads_up(client):
    state = hu_sb_first_in(['Ah', 'Ad'])
    resp = client.post('/advise', json=state)
    assert resp.status_code == 200, resp.text
    a = resp.json()
    print(f"\n  API: HU SB first in with AA, 10bb  =>  {a['recommendation']['action']}")
    assert a['hero_seat'] == 'SB'
    assert a['recommendation']['kind'] == 'ALLIN'
    assert a['pure'] is True

    # A table that drops the busted seat from the json entirely works the same way.
    del state['villain_right']
    resp = client.post('/advise', json=state)
    assert resp.status_code == 200, resp.text
    assert resp.json()['recommendation']['kind'] == 'ALLIN'


def test_api_heads_up_postflop_falls_back_to_heuristic(client):
    # The push/fold tables end the hand preflop; a heads-up flop that somehow happens
    # anyway is answered by the equity heuristic rather than 422'd.
    state = hu_sb_first_in(['Ah', 'Ad'])
    state['street'] = 'flop'
    state['board'] = [card(c) for c in ('As', 'Kc', '3c')]
    state['pot'] = 4.0
    state['hero']['bet'] = state['villain_left']['bet'] = 0.0
    resp = client.post('/advise', json=state)
    assert resp.status_code == 200, resp.text
    a = resp.json()
    print(f"\n  API: HU flop with AA (heuristic)  =>  {a['recommendation']['action']}")
    assert a['hero_seat'] == 'SB', "heads-up the dealer is the small blind"
    assert a['recommendation']['kind'] in ('BET', 'ALLIN'), "top set bets"
    assert any('heuristic' in w for w in a['warnings'])

    # Without board cards the heuristic cannot answer either, so the reason from the
    # solve lookup survives as the 422 detail.
    state['board'] = []
    resp = client.post('/advise', json=state)
    assert resp.status_code == 422
    assert 'board cards' in resp.json()['detail']


def test_hu_postflop_never_matches_a_three_handed_scenario():
    """The whole point of a separate heads-up scenario list.

    Heads-up the seats are named BTN and BB and the BTN acts last. Three-handed the seat
    pair {SB, BB} also exists but the SB acts FIRST postflop, so a heads-up spot that
    matched SB_vs_BB would be handed a solve with position reversed -- confidently wrong
    advice, which is worse than no advice.
    """
    state = hu_sb_first_in(['Ah', 'Ad'])
    state['street'] = 'flop'
    state['board'] = ['7h', '2c', '9d']
    seats, candidates = postflop.scenarios_for(state, 'villain_left')

    assert seats['hero'] == 'BTN', "heads-up the dealer holds the button postflop"
    assert seats['villain_left'] == 'BB'
    assert candidates, "heads-up spots must have scenarios to match"
    assert all(s['name'].startswith('HU_') for s in candidates), \
        f"three-handed solve leaked into a heads-up spot: {[s['name'] for s in candidates]}"
    assert all(s['oop'] == 'BB' and s['ip'] == 'BTN' for s in candidates), \
        "every heads-up pot has the BB out of position and the button in position"


def test_three_handed_still_picks_three_handed_scenarios():
    state = btn_on_flop(['Ah', 'As'])
    seats, candidates = postflop.scenarios_for(state, 'villain_right')
    assert candidates and not any(s['name'].startswith('HU_') for s in candidates), \
        "a live third player must never reach the heads-up solves"


def test_hu_scenario_answers_carry_their_provenance():
    """These solves rest on hand-written ranges; no answer from them may hide that."""
    for scenario in postflop.HU_SCENARIOS:
        warnings = postflop.scenario_warnings(scenario)
        assert any('hand-written' in w for w in warnings), scenario['name']
        if 'limp' in scenario['name']:
            assert any('weak limper' in w for w in warnings), scenario['name']
    for scenario in postflop.SCENARIOS:
        assert postflop.scenario_warnings(scenario) == [], \
            "solver-derived scenarios must not inherit the heads-up caveat"


# ---------------------------------------------------------------------------
# The heuristic fallback: multiway pots and other spots outside the solves.
# ---------------------------------------------------------------------------

def multiway_flop(hand, board=FLOP, hero_bet=0.0, bets=(0.0, 0.0), pot=6.0):
    """All three players saw the flop -- the spot the heads-up solves refuse."""
    return table('hero', 'flop',
                 hero=player(hand, bet=hero_bet),
                 left=player(bet=bets[0]),
                 right=player(bet=bets[1]),
                 board=board,
                 pot=pot + hero_bet + sum(bets))


def test_api_multiway_flop_falls_back_to_heuristic(client):
    resp = client.post('/advise', json=multiway_flop(['Ah', 'As']))
    assert resp.status_code == 200, resp.text
    a = resp.json()
    print(f"\n  API: multiway flop with AA (heuristic)  =>  {a['recommendation']['action']}")
    assert a['recommendation']['kind'] in ('BET', 'ALLIN'), "top set bets multiway too"
    assert any('heuristic' in w for w in a['warnings'])
    assert a['recommendation']['frequency'] == 1.0
    kinds = [o['kind'] for o in a['options']]
    assert 'CHECK' in kinds, "with no bet to face, checking is on the option list"


def test_heuristic_multiway_air_checks_and_folds():
    r = heuristic.advise(multiway_flop(['7h', '2s'], board=('Ad', 'Kc', '9c')))
    assert r['options'][0]['kind'] == 'CHECK', "72o with no pair checks"

    r = heuristic.advise(multiway_flop(['7h', '2s'], board=('Ad', 'Kc', '9c'),
                                       bets=(8.0, 8.0)))
    assert r['options'][0]['kind'] == 'FOLD', "72o facing a bet multiway folds"


def test_heuristic_strong_hand_continues_facing_a_bet():
    r = heuristic.advise(multiway_flop(['Ah', 'As'], bets=(8.0, 8.0)))
    assert r['options'][0]['kind'] in ('CALL', 'RAISE', 'ALLIN'), \
        "top set never folds to a flop bet"


def test_heuristic_good_pot_odds_beat_weak_equity():
    # A tiny bet into a big pot needs almost no equity: even a weak draw calls.
    r = heuristic.advise(multiway_flop(['6h', '5h'], board=('Ah', 'Kh', '9c'),
                                       bets=(1.0, 1.0), pot=100.0))
    assert r['options'][0]['kind'] == 'CALL', "a flush draw calls 1 into 100"


def test_heuristic_is_deterministic():
    state = multiway_flop(['Ah', 'As'])
    assert heuristic.advise(state)['equity'] == heuristic.advise(state)['equity'], \
        "the same spot must always get the same answer"


# The river is counted, not sampled: nothing is left to deal, so the villains' hands are
# the whole sample space and walking all 990 of them costs less than the 800 trials it
# replaces. These pin the exactness, since a wrong count would still look plausible.
# ---------------------------------------------------------------------------

def river(hand, board, n_villains=1):
    """A complete board, as the values equity() works in: (rank value, suit)."""
    to_vals = lambda cards: [(heuristic.RANK_VAL[r], s)
                             for r, s in map(heuristic.parse_card, cards)]
    return to_vals(hand), to_vals(board), n_villains


def test_river_equity_is_exact_not_sampled():
    # The board is a royal flush, so every hand alive plays it and chops: 1/2 heads-up,
    # 1/3 three-handed. Monte Carlo lands near these; only counting hits them exactly.
    hero, board, _ = river(['2c', '3d'], ['As', 'Ks', 'Qs', 'Js', 'Ts'])
    assert heuristic.equity(hero, board, 1) == 0.5, "everyone plays the board, hero chops"
    assert heuristic.equity(hero, board, 2) == 1 / 3, "three-way chop, exactly a third"

    # The nuts is 1.0 and drawing dead is 0.0 with no sampling slack either side.
    hero, board, _ = river(['As', 'Ks'], ['2s', '5s', '9s', 'Jd', '4c'])
    assert heuristic.equity(hero, board, 1) == 1.0, "the nut flush beats every hand"
    assert heuristic.equity(hero, board, 2) == 1.0


def test_river_two_villain_counting_matches_the_pair_walk():
    # exact_river never enumerates the ~450k disjoint pairs -- it counts them from
    # per-card occurrences. This is that shortcut checked against the walk it replaces,
    # on a tie-heavy board where the thirds are easiest to get wrong.
    hero, board, _ = river(['2c', '3d'], ['As', 'Ad', 'Ah', 'Ks', 'Kd'])
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]
    hero_val = heuristic.eval7(hero + board)
    hands = [(set(h), heuristic.eval7(list(h) + board))
             for h in itertools.combinations(deck, 2)]

    share, total = Fraction(0), 0
    for i, (s1, v1) in enumerate(hands):
        for s2, v2 in hands[i + 1:]:
            if s1 & s2:
                continue
            total += 1
            best = max(v1, v2)
            if hero_val > best:
                share += 1
            elif hero_val == best:
                share += Fraction(1, 1 + (v1 == hero_val) + (v2 == hero_val))

    # Exact rationals, not a tolerance: the counting identity is either right or it is not.
    assert heuristic.exact_river(hero, board, 2, deck) == float(share / total)


def test_turn_equity_is_exact_not_sampled():
    # A royal flush on the turn: no river card and no two hole cards beat it, so the
    # answer is exactly 1, both heads-up and three-handed. Sampling only approaches it.
    hero, board, _ = river(['As', 'Ks'], ['Qs', 'Js', 'Ts', '2h'])
    assert heuristic.equity(hero, board, 1) == 1.0, "a turned royal cannot be beaten"
    assert heuristic.equity(hero, board, 2) == 1.0


def test_turn_equity_matches_the_full_runout_walk():
    # The turn is counted by conditioning on each river card. This is that decomposition
    # against the walk it stands in for: every (villain hand, river) combination.
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js'])
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]

    share, total = Fraction(0), 0
    for card_ in deck:
        runout = board + [card_]
        hero_val = heuristic.eval7(hero + runout)
        for hand in itertools.combinations([c for c in deck if c != card_], 2):
            val = heuristic.eval7(list(hand) + runout)
            total += 1
            share += 1 if hero_val > val else Fraction(1, 2) if hero_val == val else 0

    # Not bit-equality: the turn sums one float per river card, so it lands within a few
    # ulps of the rational rather than on it. The river, one integer division, is exact.
    assert abs(heuristic.exact_turn(hero, board, 1, deck) - float(share / total)) < 1e-12


def test_turn_two_villains_does_not_walk_the_pair_space():
    # Three-handed the turn is 20.5M hand-pair-and-river combinations; counting the pairs
    # rather than enumerating them is what keeps it near the heads-up cost. Walking it
    # takes ~5s, so this both times out a regression and pins the answer's shape.
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js'])
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]

    start = time.perf_counter()
    two = heuristic.exact_turn(hero, board, 2, deck)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"the turn took {elapsed:.1f}s -- the pair space is being walked"
    one = heuristic.exact_turn(hero, board, 1, deck)
    assert 0.0 < two < one, "beating two random hands is strictly harder than beating one"


def test_river_equity_beats_sampling_on_cost():
    # The point of counting the river is that it is not a trade -- it is exact AND
    # cheaper, because hero's value is one eval instead of one per trial.
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js', '4c'])
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]

    start = time.perf_counter()
    heuristic.exact_river(hero, board, 1, deck)
    exact_s = time.perf_counter() - start

    start = time.perf_counter()
    heuristic.equity(hero, board[:3], 1)          # a flop, so the same call samples
    sampled_s = time.perf_counter() - start

    assert exact_s < 4 * sampled_s, \
        f"counting the river took {exact_s:.3f}s vs {sampled_s:.3f}s to sample a flop"


def test_equity_cache_makes_a_repeated_spot_free():
    # One turn is often asked about more than once -- hero bets, gets raised, decides
    # again -- and the answer cannot have changed. The exact turn costs ~0.14s, so the
    # repeat has to come out of the cache rather than be recomputed.
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js'])
    heuristic._equity.cache_clear()

    start = time.perf_counter()
    first = heuristic.equity(hero, board, 2)
    cold = time.perf_counter() - start

    start = time.perf_counter()
    again = heuristic.equity(hero, board, 2)
    warm = time.perf_counter() - start

    assert again == first, "the same spot must give back the identical float"
    assert warm < cold / 100, f"repeat took {warm * 1000:.1f}ms vs {cold * 1000:.1f}ms cold"


def test_equity_cache_keys_on_the_spot_not_the_call():
    # A board that arrives in a different order is the same spot, and must not pay twice.
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js'])
    heuristic._equity.cache_clear()
    assert heuristic.equity(hero, board, 2) == heuristic.equity(hero, board[::-1], 2)
    assert heuristic._equity.cache_info().currsize == 1, "reordering made a second entry"

    # But the villain count and the board itself are part of the spot.
    assert heuristic.equity(hero, board, 1) != heuristic.equity(hero, board, 2)
    assert heuristic.equity(hero, board[:3], 1) != heuristic.equity(hero, board, 1)


def test_equity_rejects_duplicate_cards_before_caching():
    # The guard sits outside the memo, so a malformed spot raises every time rather than
    # being answered once and then remembered.
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js'])
    for _ in range(2):
        with pytest.raises(Unsupported, match='duplicate'):
            heuristic.equity(hero, board + [hero[0]], 1)


# Range-weighted equity: the villain holds hands from a real preflop chart rather than any
# two cards, narrowed further by the bet they just made. The counting identity is the same
# one the unweighted path uses, with occurrence counts replaced by weight sums, so what
# the tests have to pin is that it degenerates correctly and moves the answer the right way.
# ---------------------------------------------------------------------------

ALL_HANDS = ({r + r: 1.0 for r in 'AKQJT98765432'} |
             {f"{a}{b}{sfx}": 1.0 for a in 'AKQJT98765432' for b in 'AKQJT98765432'
              for sfx in ('s', 'o')})


def combos_for(deck, board, class_weights, shape=None):
    return heuristic.build_combo_weights(deck, board, class_weights, shape)


def test_range_weighting_degenerates_to_uniform():
    # A range that holds every hand at full weight IS the vs-random-hands model, so the
    # weighted path has to land on the unweighted answer exactly, not merely close.
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js', '4c'])
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]
    every = combos_for(deck, board, ALL_HANDS)
    for n in (1, 2):
        assert (heuristic.exact_river(hero, board, n, deck, every) ==
                heuristic.exact_river(hero, board, n, deck))


def test_combo_weights_survive_the_turn_reslicing_the_deck():
    # Combos are keyed on the tuples combinations() yields, and exact_turn re-slices the
    # deck once per river card. If that changed a tuple's order the lookups would miss and
    # those hands would silently drop to weight 0, so this pins the ordering rather than
    # trusting it: an all-ones range must still reproduce the uniform answer exactly.
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js'])
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]
    every = combos_for(deck, board, ALL_HANDS)
    for n in (1, 2):
        assert (heuristic.exact_turn(hero, board, n, deck, every) ==
                heuristic.exact_turn(hero, board, n, deck))

    for card_ in deck:
        for hand in itertools.combinations([c for c in deck if c != card_], 2):
            assert hand in every, f"{hand} went missing once {card_} was dealt"


def test_range_weighted_river_matches_the_weighted_walk():
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js', '4c'])
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]
    weights = heuristic.load_range(heuristic.CONTINUING_RANGE)
    combos = combos_for(deck, board, weights)
    hero_val = heuristic.eval7(hero + board)

    num = den = 0.0
    for hand, w in combos.items():
        val = heuristic.eval7(list(hand) + board)
        den += w
        num += w * (1.0 if hero_val > val else 0.5 if hero_val == val else 0.0)

    assert abs(heuristic.exact_river(hero, board, 1, deck, combos) - num / den) < 1e-12


def shift_vs_range(hand, board_cards, shape=None):
    """(equity vs random, equity vs the continuing range) for one river spot."""
    weights = heuristic.load_range(heuristic.CONTINUING_RANGE)
    hero, board, _ = river(hand, board_cards)
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]
    return (heuristic.exact_river(hero, board, 1, deck),
            heuristic.exact_river(hero, board, 1, deck,
                                  combos_for(deck, board, weights, shape)))


def test_range_weighting_costs_a_hand_with_no_showdown_value():
    # The one direction that holds on every board: a hand that beats only junk is worth
    # less once the junk is gone. JTs missed all three of these.
    for board_cards in (['Ac', 'Kd', '3s', '7h', '2c'],
                        ['Kc', '8d', '3s', '7h', '2c'],
                        ['9c', '6d', '3s', '7h', '2c']):
        random_eq, range_eq = shift_vs_range(['Jh', 'Td'], board_cards)
        assert range_eq < random_eq, f"JTs should lose value vs a range on {board_cards}"


def test_range_weighting_does_not_shift_every_hand_the_same_way():
    # The reason this has to be a model change and not a bigger CALL_MARGIN: the
    # correction has no fixed sign. A busted hand loses equity against a range while a
    # made hand gains it, because the chart's callers both folded their trash AND 3-bet
    # their premiums -- the range is capped as well as narrowed.
    busted_random, busted_range = shift_vs_range(['Jh', 'Td'], ['9c', '6d', '3s', '7h', '2c'])
    made_random, made_range = shift_vs_range(['Ah', 'Kd'], ['9c', '6d', '3s', '7h', '2c'])

    assert busted_range < busted_random, "a hand beating only junk loses when junk leaves"
    assert made_range > made_random, "ace high gains against a range that 3-bet its aces"
    assert (made_range - made_random) * (busted_range - busted_random) < 0, \
        "the two shifts must have opposite signs; one margin cannot correct both"


def test_range_weighting_is_on_by_default():
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js', '4c'])
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]
    assert heuristic.USE_RANGE_WEIGHTS is True
    assert (heuristic.equity(hero, board, 1, villain_range=heuristic.CONTINUING_RANGE) !=
            heuristic.exact_river(hero, board, 1, deck)), "weighting must change something"


def test_bet_shape_reads_the_size():
    assert heuristic.bet_shape(2.0, 10.0) == 'small'      # 20% pot
    assert heuristic.bet_shape(6.0, 10.0) == 'medium'     # 60% pot
    assert heuristic.bet_shape(15.0, 10.0) == 'large'     # 150% pot
    assert heuristic.bet_shape(5.0, 0.0) == 'large', "a bet into nothing is not a small one"


def test_a_bet_polarises_the_range_it_narrows():
    # The middle of the range is the part that stops betting, so a bet should leave the
    # top and the bottom intact and thin out what is in between.
    hero, board, _ = river(['As', 'Ks'], ['2c', '7d', '9h', 'Js', '4c'])
    deck = [c for c in heuristic.FULL_DECK if c not in set(hero + board)]
    weights = heuristic.load_range(heuristic.CONTINUING_RANGE)

    wide = combos_for(deck, board, weights)
    narrow = combos_for(deck, board, weights, 'large')
    assert set(narrow) == set(wide), "narrowing re-weights hands, it never removes them"

    ranked = sorted(wide, key=lambda h: heuristic.eval7(list(h) + board), reverse=True)
    best, middle, worst = ranked[0], ranked[len(ranked) // 2], ranked[-1]
    assert narrow[best] == wide[best], "the top of the range bets for value"
    assert narrow[worst] == wide[worst], "the bottom bets as a bluff"
    assert narrow[middle] < wide[middle], "the middle is what checks"
    assert narrow[middle] > 0, "and it is thinned, not deleted"


def test_a_big_bet_hurts_a_bluff_catcher_more_than_a_small_one():
    # The payoff of reading bet size at all: a hand that only beats bluffs is worth less
    # against a big bet than a small one, because a big bet is the more polarised range.
    board_cards = ['Kc', '8d', '3s', '7h', '2c']
    _, small = shift_vs_range(['9h', '9d'], board_cards, 'small')
    _, large = shift_vs_range(['9h', '9d'], board_cards, 'large')
    assert large < small, "an underpair does worse against the more polarised bet"


def test_no_bet_leaves_the_range_unnarrowed():
    # A villain with no chips in front may have checked or may not have acted yet, and
    # the state does not say which -- so advise() must not read it as a check.
    state = multiway_flop(['Ah', 'As'], board=('Kc', '8d', '3s', '7h'))
    state['street'] = 'turn'
    r = heuristic.advise(state, reason='measuring')
    assert any('not narrowed by any action' in w for w in r['warnings'])


def test_load_range_rejects_a_chart_that_is_not_there():
    with pytest.raises(Unsupported, match='no range chart'):
        heuristic.load_range(os.path.join('BB', 'not_a_real_node.txt'))


def test_heuristic_rejects_what_it_cannot_answer():
    state = multiway_flop(['Ah', 'As'])
    state['street'] = 'preflop'
    with pytest.raises(Unsupported):
        heuristic.advise(state)

    state = multiway_flop(['Ah', 'As'])
    state['hero']['active'] = False
    with pytest.raises(Unsupported, match='folded'):
        heuristic.advise(state)


# A busted seat sent as stack=null instead of 0 (some parsers report it that way).
# It must read as busted, not crash the comparison in busted() nor the API's stack format.
# ---------------------------------------------------------------------------

def test_null_stack_busted_seat_is_heads_up():
    state = hu_sb_first_in(['Ah', 'Ad'])
    state['villain_right']['stack'] = None   # busted, reported as null rather than 0
    assert hu.busted(state['villain_right']), "no chips behind is busted, even as null"
    assert hu.is_heads_up(state)
    # And it still produces the same jam a zero-stack busted seat does.
    assert top_action(hu.advise(state)) == 'AllIn'


def test_api_accepts_null_stack_busted_seat():
    from fastapi.testclient import TestClient

    import api

    state = hu_sb_first_in(['Ah', 'Ad'])
    state['villain_right']['stack'] = None

    # Previously a 422: Player.stack was a required, non-nullable float.
    api.GameState.model_validate(state)
    # The VERBOSE log render formats every stack; a null one must not crash it.
    assert 'busted' in api.render_request(api.GameState.model_validate(state))

    resp = TestClient(api.app).post('/advise', json=state)
    assert resp.status_code == 200, resp.text
    assert resp.json()['recommendation']['kind'] == 'ALLIN'
