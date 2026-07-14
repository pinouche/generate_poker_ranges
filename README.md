# generate_poker_ranges

Precomputed GTO postflop strategies for single-raised pots, solved with
[TexasSolver](https://github.com/bupticybee/TexasSolver) (`console_solver`) and rendered
as 13x13 range heatmaps.

The goal is a **lookup table you can use while playing**: given a board and a betting
line, read off the strategy without solving anything at the table.

## Run options

Per-solve cost is measured on `As Ad Ks`, 16 threads, `set_dump_rounds 2`, 150 iterations.

| option | scenarios | solves | time | disk | betting tree |
|---|---|---|---|---|---|
| **1** | BTN vs BB | 186 | **~9 h** | ~15 GB | trimmed — single 75% bet on turn & river |
| **2** | BTN, CO, SB vs BB | 558 | **~27 h** | ~43 GB | trimmed — single 75% bet on turn & river |
| **3** | BTN, CO, SB vs BB | 558 | **~58 h** | ~72 GB | full — 50%/100% bets + raise on every street |

Options 1 and 2 use the trimmed tree: **174 s** and **79 MB** per solve. Option 3 keeps the
original tree: **375 s** and **128 MB** per solve. Trimming is 2.2x faster and 38% smaller;
what it costs is turn/river bet-size granularity — you can no longer represent a choice
between a polarized 100%-pot bet and a 50% one.

All options dump the **flop and turn** (`set_dump_rounds 2`). All are resumable: a board
whose `.json` already exists is skipped, so you can stop and restart, or add a scenario
later without re-solving what is already on disk.

```sh
python3 resources/python/generate_charts.py BTN_vs_BB          # option 1
python3 resources/python/generate_charts.py                    # option 2 (all scenarios)
```

## Scenarios

All three are single-raised pots where the BB defends by calling. **IP is whoever acts
last after the flop, which is not always the preflop raiser** — postflop action opens with
the first player left of the button, so the BTN and CO have position on a BB caller, but
the SB does not. The SB is the one raiser who plays the hand out of position.

| scenario | preflop | OOP (acts first) | IP (has position) | pot | stack |
|---|---|---|---|---|---|
| `BTN_vs_BB` | BTN opens 2.5bb, BB calls | BB | BTN | 5.5bb | 97.5bb |
| `CO_vs_BB` | CO opens 2.5bb, BB calls | BB | CO | 5.5bb | 97.5bb |
| `SB_vs_BB` | SB opens 3.0bb, BB calls | **SB** | **BB** | 6.0bb | 97.0bb |

Pot and stack differ per scenario and are not interchangeable: a 2.5bb open called by the
BB leaves `2.5 + 2.5 + the SB's dead 0.5 = 5.5bb` in the middle with 97.5bb behind, while
an SB open has no dead money (`3 + 3 = 6bb`, 97bb behind). Pot and stack fix the SPR, and
every bet size in the tree is a fraction of the pot.

## Flop subset

Solves run against `resources/text/flop_subset.txt` — 186 flops, rebuilt from the 1755
strategically distinct flops by `build_flop_subset.py`.

**Do not use `resources/text/184_flops.txt`.** It contains no unpaired rainbow flops at all
and is 40% monotone, against ~40% and ~5% in a real deck, so roughly 40% of the boards you
actually get dealt have no representative in it. Suit isomorphism can relabel `As Jh 9c` to
`Ah Js 9d`, but it can never turn a rainbow board into a two-tone one.

| suit texture | `184_flops.txt` | `flop_subset.txt` | actually dealt |
|---|---|---|---|
| two-tone | 59% | 55.1% | 55.1% |
| monotone | **40%** | 5.2% | 5.2% |
| rainbow | **1%** | 39.8% | 39.8% |

The rebuild maps all 22,100 dealable flops onto their canonical form to get each one's true
frequency, stratifies by the classes a representative may never cross (suit pattern x rank
pattern), and clusters within each stratum. Weights sum to 1.0. Residual approximation: a
real board has a **1.7%** chance of drawing a representative with a different high card, and
a mean total rank error of **1.47** across the three cards.

## Output

Solves are filed under **one directory per scenario**, so a result is never ambiguous about
the preflop action it came from:

```
resources/outputs/programmatic/
  BTN_vs_BB/
    scenario.txt        <- the preflop action, pot, stack, SPR and both range files
    9h_6d_4c.json       <- the solve
    9h_6d_4c.txt        <- the config that produced it
    ...
  CO_vs_BB/
  SB_vs_BB/
```

`scenario.txt` is written on every run and is the answer to "what *is* this spot": the
solver config records only a pot, a stack and two anonymous range strings — nothing about
how the hand got there. The board is only recorded in the filename; the json does not
contain it.

A `dump_rounds 2` json holds ~5,000 decision nodes. The flop is reached through `childrens`
(keyed by action, e.g. `BET 2.000000`); the turn hangs off chance nodes under a separate
**`dealcards`** key (keyed by the card that falls). Following only `childrens` stops dead at
the flop.

The river is solved but not written. Dumping it too (`set_dump_rounds 3`) costs **22 GB for a
single board**, since it stores all ~47 turns x ~46 rivers.

## Heatmaps

```sh
python3 resources/python/generate_heatmaps.py                          # every solve on disk
python3 resources/python/generate_heatmaps.py --scenario BTN_vs_BB     # one scenario
python3 resources/python/generate_heatmaps.py path/to/9h_6d_4c.json    # one solve
python3 resources/python/generate_heatmaps.py path/to/9h_6d_4c.json --full   # + turn nodes
```

Renders one image per decision node into `resources/outputs/heatmaps/{scenario}/{board}/`,
mirroring the solve layout, numbered in the order the hand is played (`0000_root`,
`0001_BET2`, `0002_BET2-RAISE8`, …). One board yields ~14 flop nodes: the root plus every
bet/raise/check line that can follow.

Every image is titled with its scenario, its board, and **which seat is acting** — e.g.
`OOP (BB) to act` on a `BTN_vs_BB` root. The seat is derived from postflop acting order, not
from who raised, which is what keeps `SB_vs_BB` honest: there the raiser is OOP and the BB
has position.

Batch runs are **flop-only**. A `dump_rounds 2` json holds ~5,000 decision nodes and nearly
all of them are turn nodes, sitting under one of the ~47 cards that can fall — rendering them
for every board would be millions of images. `--full` expands the turn too, but only for a
specific json; it refuses to run against the whole batch.

Cells are colored by action (green check, aqua call, blue fold, red ramp for bet/raise sized
light to dark) and stacked in proportion, never blended. Faded cells are hands the preflop
range only holds part of the time; the opacity is that frequency. Ranges are read from the
`.txt` config next to the json, per player — the two players alternate down the tree, so a
single range file is wrong at half the nodes.

## Preflop advisor (API)

Preflop needs no solve — it is a lookup in the charts. `preflop_advisor.py` takes a live
game state and reads the chart that covers it; `api.py` is the same thing behind HTTP.

### Docker

```sh
docker build -t preflop-advisor .
docker run -d --name preflop -p 8000:8000 preflop-advisor

curl -X POST localhost:8000/advise -H 'Content-Type: application/json' -d @state.json
docker rm -f preflop                                   # stop it
```

The image (~277 MB) carries only the advisor and `ranges/qb_ranges/`. `.dockerignore` keeps
`resources/outputs/` out of the build context — the solves are tens of GB and the API never
reads them. Interactive docs are at `localhost:8000/docs`, health at `/health`.

### Without Docker

```sh
uvicorn api:app --app-dir resources/python --reload    # same API on :8000
python3 resources/python/preflop_advisor.py state.json # or straight from the CLI
```

### The request and the answer

`POST /advise` takes the table's game-state json verbatim (extra fields are ignored, so the
schema can grow without breaking it):

```json
{ "hero": {"cards": [{"rank":"A","suit":"h"}, {"rank":"Q","suit":"s"}], "stack":300, "bet":0, "active":true},
  "villain_left":  {"cards": [], "stack":290, "bet":10, "active":true},
  "villain_right": {"cards": [], "stack":280, "bet":20, "active":true},
  "board": [], "pot":30, "small_blind":10, "big_blind":20,
  "dealer":"hero", "street":"preflop", "showdown":false }
```

```json
{ "hand": "AQo", "hero_seat": "BTN", "action_so_far": "folds to hero",
  "options": [ {"action":"RAISE to 50 (2.5bb)", "kind":"RAISE", "chips":50.0, "frequency":1.0},
               {"action":"FOLD", "kind":"FOLD", "chips":null, "frequency":0.0} ],
  "recommendation": {"action":"RAISE to 50 (2.5bb)", "kind":"RAISE", "chips":50.0, "frequency":1.0},
  "pure": true,
  "warnings": ["Stack mismatch: hero is 15bb deep but the charts are 100bb. ..."] }
```

`chips` is the number to act on, `kind` is machine-readable, and mixed strategies come back
as frequencies with `pure: false` rather than being rounded to a single action.

**A spot the charts do not cover is a 422, not a 500** — a postflop street, a limped pot, a
seating the pack has no chart for. That is a real answer ("no chart here"), and the client
has to be able to tell it apart from the server being broken.

### How a game state becomes a chart

The chart filenames *are* the preflop tree, so the lookup is a tree walk:

1. **Seats** come from `dealer` plus clockwise order (`villain_left` is on hero's left).
   Once the blinds are known this is *checked*, not assumed: whoever we call the SB must
   have posted the small blind, or the request is rejected. Advising confidently from an
   inverted seating would be worse than not advising.
2. **The action so far** is inferred from the chips in front of each player — the state is a
   snapshot, not a history. Anyone above the big blind has raised, and raises only ever
   increase, so sorting raisers by bet size recovers the order they acted in.
3. **The node** is the set of charts whose line is that action plus one more action by hero.
   Those files are the complete option set and a hand's weights across them sum to 1.
   **A missing file is an option that does not exist** — there is no
   `SB/BTN_2.5bb_SB_Call.txt`, because the SB never flats a button open here: 3-bet or fold.
4. **Sizes snap** to the nearest chart line (a 3bb open reads off the 2.5bb chart) and the
   response says how far it had to reach.

The charts are **100bb**. A materially different stack gets a loud warning rather than a
quiet wrong answer: at 15bb the game is largely jam-or-fold and the pack's sizings are
simply not the right ones, even when its choice of hand happens to agree.

## Known solver quirks

- **`set_use_isomorphism 1` segfaults on some textures** (exit 139). `As,Ad,Ah` is one; it
  solves fine with isomorphism off. `generate_charts.py` retries a crashed board with
  isomorphism disabled rather than leaving a hole in the table. The speedup is worth ~1.5x,
  which is why it is not simply disabled everywhere.
- **The solver exits 0 having written nothing** when this happens, so the output file — not
  the return code — is the only trustworthy success signal.
- **Ranges are hand-class only.** Feeding a combo (`AhKs`) aborts with
  `range str AhKs len not valid`. This is why turn strategy must be *dumped* rather than
  re-solved from a range that reached the turn: that range is irreducibly combo-specific.
  On `As Ad Ks` facing a 5bb bet, `9s8s` calls 98% while `9c8c`, `9d8d` and `9h8h` fold 100%
  — a backdoor spade draw — and no hand-class range can express "98s, but only the spade one".
- **Preflop needs no solve.** TexasSolver is postflop-only; preflop strategy is a lookup in
  the `ranges/qb_ranges/` charts.
