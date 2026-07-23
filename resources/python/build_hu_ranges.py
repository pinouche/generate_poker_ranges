"""Build approximate heads-up preflop ranges for the 100bb postflop solves.

WHAT THESE ARE, AND ARE NOT
---------------------------
These are hand-written approximations of heads-up 100bb preflop play, not solver
output. Everything else under ranges/ came out of a solver; these did not. They exist
because nothing in the repo could supply a heads-up raising range -- the six-max and
three-handed packs are the wrong game (a heads-up button opens ~85% where a three-handed
button opens ~50%), and holdemresources_hu_*.csv are jam-or-fold frequency tables with
no raise branch to extract.

They are built from a tier spec rather than typed out as 169 numbers so that the shape
of each range is legible, checkable and easy to correct: read the spec, disagree with a
line, change it, re-run. `--stats` prints the combo-weighted width of every range, which
is the number to sanity-check them against.

Consequence to keep in mind: solves built on these inherit their error. The solver will
compute an exact equilibrium FOR THESE RANGES, and its confidence says nothing about
whether the ranges are right. Every scenario folder gets a note saying so.

THE LIMPED POTS ARE AN OPPONENT MODEL, NOT AN EQUILIBRIUM
---------------------------------------------------------
Solved heads-up play does limp some hands, but the limped pots here are not that. They
model the common weak-player pattern -- limp most hands, raise only premiums -- because
that is who actually limps at a live table. The button's limping range is therefore wide
and CAPPED (no strong hands in it), which hands the big blind a range advantage the
solver will exploit. Against an opponent who limps a balanced range, these solves are
wrong in a specific direction: they will have you attacking too hard.

Usage:
    python3 resources/python/build_hu_ranges.py           # write the files
    python3 resources/python/build_hu_ranges.py --stats   # widths only, write nothing
"""

import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_BASE = os.path.join(REPO_ROOT, 'ranges/heads_up_ranges/100bb')
# The canonical 169-hand order and naming. Read from a real range file rather than
# regenerated here: the solver is fed these strings verbatim, and a hand order or a
# naming convention that drifts from the existing pack is a silent corruption.
ORDER_SOURCE = os.path.join(
    REPO_ROOT, 'ranges/qb_ranges/100bb 2.5x 500rake/BTN/BTN_2.5bb.txt')

RANKS = 'AKQJT98765432'


def hand_order():
    with open(ORDER_SOURCE) as f:
        return [cell.split(':')[0] for cell in f.read().strip().split(',')]


def combos(hand):
    """6 for a pair, 4 suited, 12 offsuit -- the weights that make a range's width mean
    something. A range of 'every offsuit hand' is far more of the deck than 'every
    suited hand' despite being the same number of cells."""
    if hand[0] == hand[1]:
        return 6
    return 4 if hand.endswith('s') else 12


def expand(spec):
    """'77+' -> every pair 77 and up. 'A2s+' -> A2s..AKs. 'AJo+' -> AJo,AQo,AKo.
    Anything without a trailing + is one hand, written exactly as the pack writes it."""
    if not spec.endswith('+'):
        return [spec]
    base = spec[:-1]
    if len(base) == 2 and base[0] == base[1]:
        return [r + r for r in RANKS[:RANKS.index(base[0]) + 1]]
    hi, lo, suit = base[0], base[1], base[2]
    i_hi, i_lo = RANKS.index(hi), RANKS.index(lo)
    return [hi + RANKS[j] + suit for j in range(i_lo, i_hi, -1)]


def build(*groups):
    """Merge (weight, 'spec,spec,...') groups into a hand -> weight map. Later groups
    win, so a broad group can be stated first and then carved into."""
    out = {}
    for weight, specs in groups:
        for spec in specs.replace(' ', '').split(','):
            if spec:
                for hand in expand(spec):
                    out[hand] = weight
    return out


def everything_except(*groups):
    """Weight 1.0 on all 169 hands, then apply the groups on top. The natural way to
    write a heads-up button, which raises most of the deck and folds a short list."""
    return build((1.0, ','.join(hand_order())), *groups)


# ---------------------------------------------------------------------------
# The ranges. Percentages in the comments are combo-weighted (--stats prints them).
# ---------------------------------------------------------------------------

# The junk a heads-up button still folds at 2.5x: low, disconnected and offsuit. Suited
# hands all continue -- at this price nothing suited is a fold heads-up.
BTN_JUNK = ('32o,42o,52o,62o,72o,82o,92o,T2o,'
            '43o,53o,63o,73o,83o,54o,64o,74o')

# The big blind's 3bet. Linear at the top, with suited wheel aces and suited connectors
# as the bluffs -- hands that make the best of being called and having to play out of
# position. ATo is a partial: too good to fold, not good enough to 3bet every time.
BB_3BET_SPEC = ('88+,AJs+,AQo+,KQs,KJs,KTs,A2s,A3s,A4s,A5s,A6s,'
                'K9s,Q9s,QTs,QJs,J9s,JTs,T9s,98s,87s,76s,65s,54s')

# What the big blind will not defend at all: offsuit hands too weak to call 1.5 into 4
# out of position. Everything suited defends, as does every pair.
BB_FOLD = ('32o,42o,52o,62o,72o,82o,92o,T2o,J2o,Q2o,K2o,'
           '43o,53o,63o,73o,83o,93o,T3o,J3o,Q3o,K3o,'
           '54o,64o,74o,84o,T4o,J4o,'
           '85o,95o,T5o,J5o')

RANGES = {
    # --- the button (small blind), which is IN position on every heads-up street ---

    # Open 2.5x. ~85%: everything but the junk list.
    'BTN/BTN_2.5bb.txt': everything_except((0.0, BTN_JUNK)),

    # Facing the big blind's 3bet to 10bb, the button calls 5.5 into 16 (needs ~34%).
    # AA/KK are absent because they 4bet; QQ and AKs are partials for the same reason.
    # This is the flop range for the 3bet pot, so it must be the CALLING range only.
    'BTN/BTN_2.5bb_BB_10.0bb_BTN_Call.txt': build(
        (1.0, '22+,A2s+,K2s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s,43s,'
              'ATo+,KJo+,QJo'),
        (0.5, 'QQ,AKs'),
        (0.0, 'AA,KK'),
    ),

    # Limping: the weak-player model. Everything playable EXCEPT the premiums, which
    # this opponent raises. Capped by construction -- that is the whole point of it.
    'BTN/BTN_Limp.txt': everything_except(
        (0.0, BTN_JUNK),
        (0.0, 'TT+,AQs+,AKo'),
    ),

    # That limper facing a 4bb iso-raise calls with most of what it limped, folding the
    # weakest offsuit holdings. Passive players call too much here, which is why this is
    # wider than a defensible defending range.
    'BTN/BTN_Limp_BB_4.0bb_BTN_Call.txt': build(
        (1.0, '22+,A2s+,K2s+,Q4s+,J6s+,T6s+,96s+,86s+,75s+,65s,54s,'
              'A7o+,K9o+,Q9o+,J9o+,T9o'),
        (0.0, 'TT+,AQs+,AKo'),
    ),

    # --- the big blind, which is OUT of position on every heads-up street ---

    # Calling the 2.5x open: defend everything but the fold list, minus what 3bets.
    'BB/BTN_2.5bb_BB_Call.txt': build(
        (1.0, ','.join(hand_order())),
        (0.0, BB_FOLD),
        (0.0, BB_3BET_SPEC),
        (0.5, 'ATo'),
    ),

    'BB/BTN_2.5bb_BB_10.0bb.txt': build(
        (1.0, BB_3BET_SPEC),
        (0.5, 'ATo'),
    ),

    # Iso-raising the limp to 4bb: value plus the suited hands that flop well. ~19%.
    'BB/BTN_Limp_BB_4.0bb.txt': build(
        (1.0, '77+,ATs+,KJs+,QJs,AJo+,KQo'),
        (1.0, 'A2s,A3s,A4s,A5s,K9s,KTs,Q9s,QTs,J9s,JTs,T9s,98s,87s,76s'),
    ),

    # Checking behind the limp: everything that did not raise. Nothing folds -- it is
    # free. This is the range the limped-pot solve gives the big blind.
    'BB/BTN_Limp_BB_Check.txt': build(
        (1.0, ','.join(hand_order())),
        (0.0, '77+,ATs+,KJs+,QJs,AJo+,KQo'),
        (0.0, 'A2s,A3s,A4s,A5s,K9s,KTs,Q9s,QTs,J9s,JTs,T9s,98s,87s,76s'),
    ),
}


def width(hands):
    """Combo-weighted share of the deck, as a percentage."""
    total = sum(combos(h) for h in hand_order())
    return 100.0 * sum(w * combos(h) for h, w in hands.items()) / total


def serialise(hands):
    order = hand_order()
    unknown = set(hands) - set(order)
    if unknown:
        raise SystemExit(f"not real hand names: {sorted(unknown)}")
    return ','.join(f"{h}:{hands.get(h, 0.0):g}" for h in order)


NOTE = """Approximate heads-up preflop ranges, 100bb.

HAND-WRITTEN, NOT SOLVED. Every other range pack in this repo is solver output; this one
is not. It exists because no heads-up raising range was available -- the six-max and
three-handed packs are a different game, and holdemresources_hu_*.csv are jam-or-fold
tables with no raise branch.

Regenerate with:  python3 resources/python/build_hu_ranges.py
Edit the tier spec in that file, not these .txt files -- they are overwritten.

The limped-pot ranges model a weak, passive limper (limps wide, raises only premiums),
not equilibrium play. The button's limping range is capped on purpose. Solves built on
it will attack a balanced limper too aggressively.

Widths (combo-weighted):
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stats', action='store_true',
                        help="print range widths without writing anything")
    args = parser.parse_args()

    lines = []
    for rel, hands in sorted(RANGES.items()):
        lines.append(f"  {rel:45} {width(hands):5.1f}%")
    print('\n'.join(lines))

    if args.stats:
        return

    for rel, hands in RANGES.items():
        path = os.path.join(OUT_BASE, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(serialise(hands))

    with open(os.path.join(OUT_BASE, 'README.txt'), 'w') as f:
        f.write(NOTE + '\n'.join(lines) + '\n')
    print(f"\nWrote {len(RANGES)} ranges to {OUT_BASE}")


if __name__ == '__main__':
    main()
