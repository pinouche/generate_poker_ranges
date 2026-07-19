"""Feed hand-built table snapshots through both advisors and check the answers make sense.

Every test constructs the same json the table would POST to /advise (chips, not bb:
blinds 1/2, 100bb = 200-chip stacks) and asserts poker sanity rather than exact solver
frequencies: premiums never fold, trash folds, frequencies sum to 1, the seats come out
where the dealer button says they must.

Run with output visible (each test prints the spot and the recommendation it got):

    pytest resources/python/test_advisors.py -v -s
"""

import os
import sys

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

    # Without board cards the heuristic cannot answer either, so the original hu
    # reason survives as the 422 detail.
    state['board'] = []
    resp = client.post('/advise', json=state)
    assert resp.status_code == 422
    assert 'postflop' in resp.json()['detail']


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
