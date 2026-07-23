Approximate heads-up preflop ranges, 100bb.

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
  BB/BTN_2.5bb_BB_10.0bb.txt                     12.4%
  BB/BTN_2.5bb_BB_Call.txt                       59.6%
  BB/BTN_Limp_BB_4.0bb.txt                       13.6%
  BB/BTN_Limp_BB_Check.txt                       86.4%
  BTN/BTN_2.5bb.txt                              85.5%
  BTN/BTN_2.5bb_BB_10.0bb_BTN_Call.txt           24.5%
  BTN/BTN_Limp.txt                               81.7%
  BTN/BTN_Limp_BB_4.0bb_BTN_Call.txt             32.3%
