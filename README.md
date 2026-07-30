# generate_poker_ranges

Precomputed GTO postflop strategies for **3-handed** (BTN / SB / BB) 100bb cash, solved
with [TexasSolver](https://github.com/bupticybee/TexasSolver) (`console_solver`) and
rendered as 13x13 range heatmaps.

The goal is a **lookup table you can use while playing**: given a board and a betting
line, read off the strategy without solving anything at the table.

## Scenarios

The five below are **every heads-up postflop spot a 3-handed game reaches**: two
single-raised pots and three 3bet pots.

**IP is whoever acts last after the flop, and that is decided by seat order (SB, then BB,
then BTN) — not by who raised.** So the SB is out of position in all three of the pots it
plays, whether it was the aggressor or the caller.

| scenario | preflop | OOP (acts first) | IP (has position) | pot | stack | SPR |
|---|---|---|---|---|---|---|
| `BTN_vs_BB` | BTN opens 2.5bb, SB folds, BB calls | BB | BTN | 5.5bb | 97.5bb | 17.7 |
| `SB_vs_BB` | BTN folds, SB opens 3.0bb, BB calls | **SB** | **BB** | 6.0bb | 97.0bb | 16.2 |
| `BB_vs_BTN_3bet` | BTN opens 2.5bb, SB folds, BB 3bets to 11bb, BTN calls | BB | BTN | 22.5bb | 89.0bb | 4.0 |
| `SB_vs_BTN_3bet` | BTN opens 2.5bb, SB 3bets to 11bb, BB folds, BTN calls | **SB** | BTN | 23.0bb | 89.0bb | 3.9 |
| `BB_vs_SB_3bet` | BTN folds, SB opens 3.0bb, BB 3bets to 9bb, SB calls | **SB** | **BB** | 18.0bb | 91.0bb | 5.1 |

**There is no BTN-open, SB-call pot.** Facing a BTN open the SB is out of position with the
BB still to act, and the ranges give it only a 3bet or a fold — so the BTN and the SB meet
after the flop only in a 3bet pot. That is why `SB_vs_BTN_3bet` is here and a
`BTN_vs_SB` single-raised pot is not.

Pot and stack are per-scenario and are **not** interchangeable. Each is (money in the
middle) / (100bb minus what each player put in), and the dead blind money matters: a 2.5bb
BTN open called by the BB leaves `2.5 + 2.5 + the SB's dead 0.5 = 5.5bb`, while an SB open
has no dead money at all (`3 + 3 = 6bb`). Pot and stack fix the SPR, and every bet size in
the tree is a fraction of the pot.

## Run options

Per-solve cost is measured on `As Ad Ks`, 16 threads, `set_dump_rounds 2`, 150 iterations.
All 186 flops in the subset are run per scenario.

| option | scenarios | solves | time | disk |
|---|---|---|---|---|
| **1** | `BTN_vs_BB` only | 186 | **~9 h** | ~15 GB |
| **2** | both single-raised pots | 372 | **~18 h** | ~29 GB |
| **3** | all five (adds the 3bet pots) | 930 | ~18 h **+ the 3bet pots** | ~29 GB + |

The single-raised pots cost **174 s** and **79 MB** per solve on the trimmed betting tree
(single 75% bet on turn and river). The full tree — 50%/100% bets plus a raise on every
street — costs 375 s and 128 MB instead; trimming is 2.2x faster and 38% smaller, and what
it buys back is turn/river bet-size granularity, so you can no longer represent a choice
between a polarized 100%-pot bet and a 50% one.

**The 3bet pots have not been timed yet**, but they run at SPR ~4-5 rather than ~17, which
makes their trees far smaller — expect them to be materially cheaper per solve than the
numbers above, not equal to them. They are also the place where the trimmed tree costs
least, since there is little room left to manoeuvre postflop.

All options dump the **flop and turn** (`set_dump_rounds 2`) and are resumable: a board
whose `.json` already exists is skipped, so you can stop and restart, or add a scenario
later without re-solving what is already on disk.

```sh
python3 resources/python/generate_charts.py BTN_vs_BB               # option 1
python3 resources/python/generate_charts.py BTN_vs_BB SB_vs_BB      # option 2
python3 resources/python/generate_charts.py                         # option 3 (everything)
```

## Ranges

The charts in `ranges/qb_ranges/100bb 2.5x 500rake` are 6-max solves, but the nodes used
here transfer: "folded to the BTN" in 6-max has the **same two players left to act** (SB,
BB) at the same blinds and stacks as a BTN open 3-handed, so it is the same game. The one
discrepancy is the bunching effect — the players who folded in the 6-max tree were holding
disproportionately weak cards, so the remaining deck is not quite the one a 3-handed table
draws from. It is a small distortion and it washes out almost entirely postflop, but if you
have true 3-handed charts, prefer them.

These ranges also assume **no ante**. With an ante every pot in the table above is wrong.

## Heads-up (when a player busts)

Once the third player is out, the game is true heads-up and **nothing above applies**: the
dealer now posts the small blind, acts first preflop and **last** postflop — the `SB_vs_BB`
scenario has that postflop position reversed, and the qb_ranges charts assume another
player left to act. The heads-up game is answered instead by the HoldemResources Nash
equilibrium of the **jam-or-fold** game, in `ranges/heads_up_ranges/`:

| file | holds |
|---|---|
| `holdemresources_hu_push.csv` | the SB's shove range, one row per effective stack |
| `holdemresources_hu_call.csv` | the BB's call-a-shove range, same layout |

Rows run 1–200bb in 0.05bb steps; cells are frequencies, almost all 0 or 1 with fractions
at the Nash boundary. The advisor covers the two nodes that make up jam-or-fold poker —
the SB first in, and the BB facing a jam — and refuses everything else (limps, raises
smaller than all-in, postflop) the same way a missing chart does.

**Trust it at ~15bb effective and below.** Deeper, the tables remain unexploitable *as
long as you only ever jam or fold*, but a raise-based strategy earns more; the advisor
still answers there, with a warning saying exactly that.

The API routes to these tables automatically when a villain seat is busted (zero stack,
zero bet, inactive) or absent from the request. For a quick lookup at the table:

```sh
python3 resources/python/hu_advisor.py --hand Q7o --stack 8 --seat SB    # => ALL-IN
python3 resources/python/hu_advisor.py --hand A9o --stack 12 --seat BB   # => CALL
```

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

## Advisor (API)

One endpoint answers every street. Preflop is a lookup in the charts
(`preflop_advisor.py`); the flop and the turn are read out of the solved trees above
(`postflop_advisor.py`). `api.py` is both behind HTTP — the `street` field routes the
request. A postflop spot the solves cannot answer — a multiway pot, a river (solved but
not dumped), heads-up after a bust, solves missing from disk — falls back to an
equity-vs-pot-odds heuristic (`heuristic_advisor.py`) and says so in `warnings`.

### Docker

```sh
docker build -t preflop-advisor .
docker run -d --name preflop -p 8000:8000 \
    -v "$PWD/resources/outputs:/app/resources/outputs:ro" preflop-advisor

curl -X POST localhost:8000/advise -H 'Content-Type: application/json' -d @state.json
docker rm -f preflop                                   # stop it
```

The image (~277 MB) carries only the advisor and `ranges/qb_ranges/`; the solves are tens
of GB, so `.dockerignore` keeps `resources/outputs/` out of the build context and the
volume mount above is how the container reads them. Without the mount it still answers
everything, but every postflop answer is the heuristic, not a solve. Interactive docs are
at `localhost:8000/docs`, health at `/health`.

By default the log is one block per hand — the action-so-far line and the option rows,
nothing else:

```
A♠ Q♥  BTN opens 2.5bb, SB folds, BB calls | flop Kh 6c 6h: hero first to act
CHECK          86.1%  <- recommended
BET 40 (2bb)   13.9%
```

Hero's cards lead the block: they are what marks where one hand ends and the next begins.

Add `-e ADVISOR_VERBOSE=1` to `docker run` for the full picture — the incoming game
state, the `warnings`, and uvicorn's per-request access line — when a spot looks wrong and
you need to see why.

### Without Docker

```sh
uvicorn api:app --app-dir resources/python --reload      # same API on :8000
python3 resources/python/preflop_advisor.py state.json   # preflop, straight from the CLI
python3 resources/python/postflop_advisor.py state.json  # flop/turn, same json
python3 resources/python/heuristic_advisor.py state.json # the fallback, same json
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
{ "hand": "AQo", "hero_cards": "A♥ Q♠", "hero_seat": "BTN", "action_so_far": "folds to hero",
  "options": [ {"action":"RAISE to 50 (2.5bb)", "kind":"RAISE", "chips":50.0, "frequency":1.0},
               {"action":"FOLD", "kind":"FOLD", "chips":null, "frequency":0.0} ],
  "recommendation": {"action":"RAISE to 50 (2.5bb)", "kind":"RAISE", "chips":50.0, "frequency":1.0},
  "pure": true,
  "warnings": ["Stack mismatch: hero is 15bb deep but the charts are 100bb. ..."] }
```

`chips` is the number to act on, `kind` is machine-readable, and mixed strategies come back
as frequencies with `pure: false` rather than being rounded to a single action. `hand` is
the chart class (the same for every hand dealt from that cell); `hero_cards` is the actual
holding with suit symbols, which is what identifies the hand you are in.

Postflop the answer has the same shape, with two more `kind`s (`CHECK`, `BET`), and the
`action_so_far` names the whole line the state implies, e.g.
`"BTN opens 2.5bb, SB folds, BB calls | flop Kh 5d 2d: BB CHECK -> BTN BET 4 (2bb)"`.

**A spot the data does not cover still gets an answer when a sane one exists.** A
preflop limp is answered heuristically (iso-raise off the open range, or a free check in
the BB). Postflop, anything the solves refuse — a multiway flop, a river, heads-up after
a bust, a hand the solved range never holds at that node — falls back to
`heuristic_advisor.py`: Monte Carlo equity against random villain hands, compared with
the pot odds (facing a bet) or a value-bet threshold (checked to). It never bluffs and
knows nothing about ranges, and every such answer carries a warning naming the reason the
solve could not answer, so solve-backed advice and arithmetic are distinguishable at a
glance. **Only a spot with no sane answer at all is a 422, not a 500** — hero folded,
malformed cards, an unknown street. That is a real answer ("no chart here"), and the
client has to be able to tell it apart from the server being broken.

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

### How a postflop state becomes a node in a solve

The same snapshot-not-history problem, solved by four inferences, each one checked and
warned about rather than assumed:

1. **The scenario.** The two live seats leave at most two candidates (the single-raised
   and the 3bet pot between those seats), and the pot at the start of the street — fixed
   by the preflop line, 5.5bb vs 22.5bb — picks between them.
2. **The board.** Solves cover the 186-flop subset. A real flop is matched by **suit
   isomorphism** (all 24 relabellings against the solved set — an exact strategic match);
   failing that, the nearest solved flop with the same suit and rank pattern stands in,
   measured with the same weighted rank features `build_flop_subset.py` clustered with.
   Hero's cards and the turn card are translated through the same suit map, so a flush
   draw is still a flush draw on the solved board.
3. **The street so far.** The two `bet` amounts are matched against every partial line
   in the tree that ends with hero to act — structure first (who has put chips in), then
   closest sizing, exactly like the preflop snap.
4. **The flop line (turn only).** The client sends the full state at every hero
   decision, so the server has already *watched* the flop betting: it remembers each
   flop request (keyed by dealer + the three flop cards — the same hand for any
   practical purpose), and the turn request only has to complete that observed prefix,
   which the turn pot does uniquely. Memory is an overlay, never a dependency: with no
   matching entry (server restarted, flop request never arrived) or a pot that
   contradicts it, the line is inferred from the pot instead — what each player put in
   on the flop is `(turn pot − flop pot) / 2` — and then bet-call and check-bet-call
   can cost the same. That is a real ambiguity, and the response warns which line it
   picked rather than pretending to know. The two lines reach genuinely different
   strategies (a checked-through range is a different range), which is why observing
   beats inferring.

A solve is ~100MB of json and takes ~1.5s to parse; the advisor keeps the last two
parsed solves in memory (~1GB each), so the turn question about a flop you already asked
about is instant. Everything else (stack/SPR mismatch, snapped sizings, snapped boards,
a resettled suit) comes back in `warnings`.

**Docker note:** the image deliberately excludes `resources/outputs/` (tens of GB), so
none of the above happens unless the solves are mounted in
(`-v "$PWD/resources/outputs:/app/resources/outputs:ro"`, as in the run command up top).
Without the mount every postflop request falls back to the equity heuristic — the log
warning `no solves on disk for scenario ...` on a heads-up flop means the mount is
missing.

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
