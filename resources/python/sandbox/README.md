# Advisor vs. heuristic bots

Plays the advisor in `resources/python` against **two** copies of the heuristic bot from
the sibling `poker_self_play` repo, three-handed, full hands preflop to river.

```sh
python3 arena.py --hands 3000 --tag main          # play
python3 plots.py                                  # draw everything in results/
```

## What is actually playing

| seat | bot |
|---|---|
| one seat | the advisor: `preflop_advisor` charts, `postflop_advisor` solves, `heuristic_advisor` where neither answers |
| the other two | `poker_self_play`'s `HeuristicAgent` — Monte-Carlo equity against random hands vs. fixed thresholds |

The engine (rules, dealing, side pots, showdowns) is `poker_self_play`, expected as a
sibling directory; override with `POKER_ENGINE=/path/to/poker_self_play`.

`advisor_bridge.py` is the whole distance between the two: it renders the engine's
`GameState` as the game-state json the live table would POST to `/advise`, routes it the
way `api.advise_endpoint` does, samples the chart's mixed frequencies, and maps the
answer ("RAISE to 50") onto the nearest sizing the engine offers. The advisor only ever
sees public state plus its own hole cards.

## How it is scored

Independent hands are useless here — one 100bb hand has a standard deviation around
35bb, so a readable win rate would need ~100,000 of them. Instead every deck is played
**three times, once with the advisor in each seat**, and the advisor's score for that
deal is its average over the three. Three identical strategies score exactly zero on
every deal, so what survives is the difference between the strategies rather than who
was dealt aces. `--hands 3000` is 1000 such deals.

Stacks reset to 100bb every hand, because that is the depth the charts are built for.

## What came out (12,000 hands / 4,000 deals, seed 1)

| | bb/100 | 95% interval | significant |
|---|---|---|---|
| advisor, solves on disk | **+22.8** | ± 42.5 | no |
| advisor, solves hidden | +13.0 | ± 41.2 | no |
| what the solves are worth (paired, same decks) | +9.8 | ± 31.7 | no |

**Nothing here is distinguishable from break-even.** The honest reading is a bound: the
advisor's edge over these bots is somewhere between −20 and +65 bb/100, so it is not
being beaten, and it is not crushing them either. At 1,000 deals the solves looked worth
+45 bb/100; at 4,000 that fell to +9.8, which is what a noise effect does.

Two things explain the size of it more than strategy does:

* **Only 25% of decisions are a solve lookup.** Preflop is always a chart (40% of
  decisions), but the river is never solved and a three-way flop is outside the heads-up
  solves, so 35% of the advisor's decisions come from `heuristic_advisor` — *the same
  equity-vs-pot-odds rule the opponents use*. On those it has no edge by construction.
* **The bots limp constantly**, so 15% of decisions are in a limped pot the charts have
  no branch for, and 11% carry a "pot does not match the solved line" warning because
  the preflop action never matched a solved scenario's pot.

Beating them by more would need an exploitative strategy, which is not what a GTO chart
pack is for.

## Options worth knowing

```
--villain heuristic|tight_aggressive|loose_passive   which bot fills the other two seats
--mode sample|max        play the chart's mixed frequencies, or always its modal action
--no-solves              hide the 96GB of solves, so postflop falls back to the
                         equity heuristic -- the control for what the solves are worth
--stack-bb 100           starting stack; the charts are 100bb and warn loudly otherwise
--random-runouts         deal one board on an all-in instead of paying the expectation
```

## What lands in `results/`

| file | holds |
|---|---|
| `hands_<tag>.csv` | one row per hand: deal, seat, position, board, chips won |
| `decisions_<tag>.csv` | one row per advisor decision: what it was asked, which source answered, what it advised, what got played, how far apart those were, which caveats came back |
| `summary_<tag>.json` | the aggregates the plots are drawn from |
| `advisor_crashes_<tag>.jsonl` | game states the advisor raised an unexpected exception on; cleared at the start of each run so its length matches that run's `advisor_error` count |
| `01..06_*.png` | the figures |

## A bug this turned up, since fixed

`postflop_advisor.advise` did `strat = node['strategy']`, and **30.8% of turn action
nodes in a solve have no `strategy` key** — the solver writes a stub subtree (`actions`,
`childrens`, `player`, no strategy) for turn cards it resolved by suit isomorphism. On
`Qc 8c 4c` every one of the 13 spade turns is such a stub. The guard above only checked
that the `dealcards` entry *exists*, and it does, so the `KeyError` escaped as a 500
rather than the 422 the module produces everywhere else. It hit 57 of 30,320 decisions
in the first full run (0.2%), every one on a turn.

Fixed in `postflop_advisor.py`:

* `has_strategy(node)` — asks whether a subtree holds a strategy or is only shaped like
  one.
* `dumped_suit_map(turn, smap, chance, solved_flop)` — steers the turn onto a suit that
  *was* dumped. A suit appearing nowhere on the flop is interchangeable with any other
  suit appearing nowhere on the flop; a suit the flop holds carries the flush
  relationship and is never swapped. The swap is applied to the **suit map**, not the
  card, so it carries through to hero's cards: `9s7s` on a `Ks` turn becomes `9d7d` on
  `Kd`, keeping "hero holds the turn's suit" intact.
* the missing-`strategy` check is kept as a backstop, raising `Unsupported` so the API
  falls back cleanly instead of erroring.

Sampled across 12 solved boards in 3 scenarios, every turn card is now answerable and
the backstop never fires. Verified isomorphic rather than merely non-crashing: with hero
holding `Tc9h` on `Qc8c4c`, `As` and `Ad` return *identical* frequencies (0.6105), while
the club ace correctly differs (0.5698).

```sh
python3 reproduce_turn_bug.py                                   # exits 0; 1 on regression
python3 -m pytest resources/python/test_advisors.py -q          # 5 new tests, 63 total
```
