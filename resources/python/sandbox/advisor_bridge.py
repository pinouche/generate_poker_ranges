"""Drive the advisor (``resources/python``) as a bot inside the poker_self_play engine.

The advisor answers questions about a *table*: it takes the game-state json the live
table would POST to ``/advise`` and reads back a strategy.  The engine in
``../../../../poker_self_play`` deals hands and expects an agent that picks an id out of
a discrete action space.  This module is the whole distance between those two:

  1. **State.**  ``advisor_state`` renders an engine ``GameState`` as the advisor's json,
     from the acting seat's point of view.  The advisor's three player keys are
     positional -- ``villain_left`` is the player on hero's left -- so the mapping is a
     clockwise walk from hero, and the dealer is named by whichever key lands on it.
  2. **Routing.**  ``consult`` mirrors ``api.advise_endpoint`` exactly: heads-up tables
     first, then preflop charts / postflop solves by street, then the equity heuristic
     for anything they refuse.  It deliberately does not import ``api`` -- that would
     drag in FastAPI and pydantic for a function that is fifteen lines of if/else -- but
     it must not drift from it either, which is what ``test_bridge.py`` checks.
  3. **Action.**  The advisor answers in chips ("RAISE to 50"); the engine offers a fixed
     menu of sizings.  ``choose_action_id`` takes the legal menu entry closest to what
     was asked for and records how far it had to move, so the abstraction's distortion
     is measured rather than hidden.

Everything the advisor gets here is public plus its own hole cards -- the same
information the heuristic bots read out of ``observation["meta"]``.
"""

from __future__ import annotations

import os
import random
import sys
from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

ADVISOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ADVISOR_DIR not in sys.path:
    sys.path.insert(0, ADVISOR_DIR)

import heuristic_advisor as heuristic          # noqa: E402
import hu_advisor as hu                        # noqa: E402
import postflop_advisor as postflop            # noqa: E402
import preflop_advisor as preflop              # noqa: E402
from preflop_advisor import Unsupported        # noqa: E402

# Seat keys in clockwise order from hero, which is what the advisor's TABLE_ORDER means.
SEAT_ROLES = ("hero", "villain_left", "villain_right")
STREETS = ("preflop", "flop", "turn", "river")

#: Where an answer came from.  The first three are chart/solve lookups, the last two are
#: arithmetic standing in for a spot the data does not cover.
SOURCES = ("preflop_chart", "postflop_solve", "headsup_table", "heuristic",
           "preflop_equity", "advisor_error")

Decision = namedtuple("Decision", "action policy value")


# ---------------------------------------------------------------------------
# Engine state -> advisor json.
# ---------------------------------------------------------------------------

def advisor_state(state, hero_seat: int, street: str) -> dict:
    """The engine's ``GameState`` as the json the table would POST to /advise.

    ``bet`` is the seat's commitment *on the current street*, which is what both the
    preflop reconstruction and the postflop node match read; ``pot`` is everything in the
    middle including those live bets, the convention the advisor documents (blinds
    posted, 10/20, pot 30).
    """
    n = state.num_players
    out = {}
    for offset in range(n):
        seat = (hero_seat + offset) % n
        player = state.players[seat]
        out[SEAT_ROLES[offset]] = {
            # Only hero's cards: the advisor must not see what it could not see live.
            "cards": [c.to_dict() for c in player.hole] if seat == hero_seat else [],
            "stack": float(player.stack),
            "bet": float(player.street_bet),
            "active": not player.folded,
        }
    out.update(
        board=[c.to_dict() for c in state.board],
        pot=float(state.pot),
        small_blind=float(state.small_blind),
        big_blind=float(state.big_blind),
        dealer=SEAT_ROLES[(state.dealer - hero_seat) % n],
        street=street,
        showdown=False,
    )
    return out


# ---------------------------------------------------------------------------
# Advice, normalised to (kind, chips, frequency).
#
# `chips` is always the number to act on: for BET/RAISE the resulting TOTAL street bet,
# for CALL the amount to put in, None for CHECK/FOLD/ALLIN.  The three advisors already
# agree on that; this only makes it explicit.
# ---------------------------------------------------------------------------

Option = namedtuple("Option", "kind chips frequency label")


def _preflop_kind(action: str) -> str:
    return {"FOLD": "FOLD", "Check": "CHECK", "Call": "CALL",
            "Limp": "CALL", "AllIn": "ALLIN"}.get(action, "RAISE")


def _preflop_options(r: dict) -> List[Option]:
    bb, to_call = r["bb"], r["to_call"]
    options = []
    for action, weight, _ in r["strategy"]:
        kind = _preflop_kind(action)
        chips = (preflop.size_of(action) * bb if kind == "RAISE"
                 else to_call if kind == "CALL" else None)
        options.append(Option(kind, chips, float(weight),
                              preflop.describe(action, bb, to_call)))
    return options


def _postflop_options(r: dict) -> List[Option]:
    bb, to_call = r["bb"], r["to_call"]
    return [Option(postflop.kind_of(a), postflop.chips_of(a, bb, to_call), float(w),
                   postflop.describe(a, bb, to_call))
            for a, w in r["strategy"]]


def _heuristic_options(r: dict) -> List[Option]:
    return [Option(o["kind"], o["chips"], float(o["frequency"]), o["action"])
            for o in r["options"]]


def preflop_equity_advice(state: dict) -> dict:
    """Last resort: a preflop spot no chart covers, answered by equity vs pot odds.

    ``heuristic_advisor`` refuses preflop by design -- it is the *postflop* fallback, and
    a preflop chart pack that cannot answer a preflop spot is a gap worth seeing rather
    than papering over.  A simulation still has to put a chip in, so the same arithmetic
    it uses postflop is applied here, on a bare board, and flagged with its own source so
    these hands can be counted and excluded.  It is not part of the advisor.
    """
    hero = state["hero"]
    villains = [p for p in SEAT_ROLES[1:]
                if state.get(p) and state[p].get("active", True)]
    if not villains:
        raise Unsupported("no live villain preflop")

    cards = [postflop.parse_card(c) for c in hero.get("cards", [])]
    if len(cards) != 2:
        raise Unsupported(f"hero has {len(cards)} cards, need 2")
    hole = [(postflop.RANK_VAL[r], s) for r, s in cards]
    equity = heuristic.equity(hole, [], len(villains))

    bb = float(state["big_blind"])
    pot, stack = float(state["pot"]), float(hero["stack"])
    hero_bet = float(hero["bet"])
    top = max(float(state[v]["bet"]) for v in villains)
    to_call = max(0.0, top - hero_bet)

    if to_call > 1e-9:
        need = to_call / (pot + to_call)
        call = Option("CALL", min(to_call, stack), 0.0, f"CALL {min(to_call, stack):g}")
        fold = Option("FOLD", None, 0.0, "FOLD")
        raise_to = min(3 * top, hero_bet + stack)
        aggressive = Option("RAISE", raise_to, 0.0, f"RAISE to {raise_to:g}")
        if equity >= heuristic.RAISE_EQUITY and stack > to_call:
            ordered = [aggressive, call, fold]
        elif equity >= need + heuristic.CALL_MARGIN:
            ordered = [call, fold, aggressive]
        else:
            ordered = [fold, call, aggressive]
    else:
        open_to = min(max(2.5 * bb, hero_bet + bb), hero_bet + stack)
        aggressive = Option("RAISE", open_to, 0.0, f"RAISE to {open_to:g}")
        check = Option("CHECK", None, 0.0, "CHECK")
        threshold = 1.0 / (len(villains) + 1) + heuristic.VALUE_BET_MARGIN
        ordered = [aggressive, check] if equity >= threshold else [check, aggressive]

    ordered = [ordered[0]._replace(frequency=1.0)] + list(ordered[1:])
    return {
        "options": ordered,
        "action_so_far": f"preflop: hero equity ~{equity:.0%} vs "
                         f"{len(villains)} random hand(s)",
        "warnings": ["No preflop chart covers this spot; answered by equity vs pot "
                     "odds, which is NOT the advisor."],
    }


@dataclass
class Advice:
    """One answer, whatever produced it."""

    source: str
    options: List[Option]
    warnings: List[str] = field(default_factory=list)
    action_so_far: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def pure(self) -> bool:
        return bool(self.options) and self.options[0].frequency >= 0.999


def consult(state: dict) -> Advice:
    """Route a state to whatever can answer it -- the same order ``api.advise_endpoint``
    uses, so the bot is playing the advisor the HTTP client would get."""
    try:
        if hu.is_heads_up(state):
            if state["street"] == "preflop":
                r = hu.advise(state)
                return Advice("headsup_table", _preflop_options(r), r["warnings"],
                              _line(r), _detail(r))
            r = postflop.advise(state)
            return Advice("postflop_solve", _postflop_options(r), r["warnings"],
                          postflop.action_line(r), _detail(r))
        if state["street"] == "preflop":
            r = preflop.advise(state)
            return Advice("preflop_chart", _preflop_options(r), r["warnings"],
                          _line(r), _detail(r))
        r = postflop.advise(state)
        return Advice("postflop_solve", _postflop_options(r), r["warnings"],
                      postflop.action_line(r), _detail(r))
    except Unsupported as reason:
        return _fallback(state, str(reason), "heuristic")
    except Exception as bug:
        # Not a spot the charts decline to answer -- the advisor raised something it does
        # not raise on purpose, which over HTTP would be a 500.  A match cannot stop for
        # it, so the state is written out for diagnosis and the hand carries on under the
        # heuristic, counted separately so these never pass for advice.
        _record_crash(state, bug)
        return _fallback(state, f"{type(bug).__name__}: {bug}", "advisor_error")


def _fallback(state: dict, reason: str, source: str) -> Advice:
    try:
        r = heuristic.advise(state, reason=reason)
        return Advice(source, _heuristic_options(r), r["warnings"],
                      r["action_so_far"], _detail(r))
    except Unsupported:
        # Preflop is the only street the heuristic refuses outright.
        r = preflop_equity_advice(state)
        return Advice("preflop_equity" if source == "heuristic" else source,
                      r["options"], [reason] + r["warnings"], r["action_so_far"], {})


#: Where unexpected advisor exceptions get written.  ``arena.py`` repoints this per run
#: so two runs never share a log -- the count in one is the whole point of it.
CRASH_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "results", "advisor_crashes.jsonl")


def _record_crash(state: dict, bug: Exception) -> None:
    import json
    import traceback

    os.makedirs(os.path.dirname(CRASH_LOG), exist_ok=True)
    with open(CRASH_LOG, "a") as f:
        f.write(json.dumps({
            "error": f"{type(bug).__name__}: {bug}",
            "traceback": traceback.format_exc().splitlines()[-6:],
            "state": state,
        }) + "\n")


def _line(r: dict) -> str:
    """The preflop 'action so far', the way api.preflop_advice renders it."""
    bb = r["bb"]
    return " -> ".join(f"{s} {preflop.describe(a, bb)}" for s, a in r["prior"]) \
        or "folds to hero"


def _detail(r: dict) -> dict:
    """The few chart/solve fields worth keeping per decision."""
    return {k: r[k] for k in ("hand", "hero_seat", "scenario", "equity", "snap_error",
                              "board_exact", "flop_line_source") if k in r}


# ---------------------------------------------------------------------------
# Advice -> a legal action id.
# ---------------------------------------------------------------------------

def pick_option(options: Sequence[Option], rng: random.Random, mode: str) -> Option:
    """Which option to play.

    ``sample`` draws from the chart's own frequencies, which is how a mixed strategy is
    meant to be played -- always taking the most common action turns a balanced range
    into an exploitable one.  ``max`` takes the modal action instead.

    Frequencies that sum to zero mean the chart never holds this hand at this node (it
    would have folded earlier).  There is nothing to sample, so take the cheapest way
    out rather than the arbitrary first row.
    """
    total = sum(o.frequency for o in options)
    if total <= 1e-9:
        for kind in ("FOLD", "CHECK", "CALL"):
            for option in options:
                if option.kind == kind:
                    return option
        return options[0]
    if mode == "max":
        return max(options, key=lambda o: o.frequency)
    return rng.choices(options, weights=[o.frequency for o in options])[0]


#: What to do when the engine cannot offer what was asked for, in order of preference.
#: A raise that cannot be made is still a strong hand, so it calls; a fold with no bet
#: to fold to is a check.
FALLBACKS = {
    "FOLD": ("FOLD", "CHECK", "CALL"),
    "CHECK": ("CHECK", "CALL", "FOLD"),
    "CALL": ("CALL", "CHECK", "FOLD"),
    "BET": ("BET", "CALL", "CHECK", "FOLD"),
    "RAISE": ("RAISE", "CALL", "CHECK", "FOLD"),
    "ALLIN": ("ALLIN", "RAISE", "BET", "CALL", "CHECK", "FOLD"),
}


def choose_action_id(option: Option, legal, space, big_blind: float
                     ) -> Tuple[int, float, bool]:
    """(action id, sizing error in bb, whether the kind had to be substituted).

    Sizings snap to the nearest thing the engine's bet abstraction can express, the same
    way the advisor itself snaps a table's bet to the nearest chart line.  The error is
    returned rather than swallowed: it is the price of playing a continuous-sizing
    strategy through a discrete menu, and it belongs in the results.
    """
    to_amounts = legal.to_amounts
    aggressive = [i for i in space.aggressive_ids if legal.is_legal(i)]
    from environment.state import CALL, CHECK, FOLD

    def ids_for(kind: str) -> List[int]:
        if kind == "FOLD":
            return [FOLD] if legal.is_legal(FOLD) else []
        if kind == "CHECK":
            return [CHECK] if legal.is_legal(CHECK) else []
        if kind == "CALL":
            return [CALL] if legal.is_legal(CALL) else []
        if kind == "ALLIN":
            return [space.all_in] if legal.is_legal(space.all_in) else []
        return aggressive          # BET and RAISE are the same menu to the engine

    for step, kind in enumerate(FALLBACKS[option.kind]):
        ids = ids_for(kind)
        if not ids:
            continue
        substituted = step > 0
        if kind in ("BET", "RAISE") and option.chips is not None:
            best = min(ids, key=lambda i: abs(to_amounts[i] - option.chips))
            return best, abs(to_amounts[best] - option.chips) / big_blind, substituted
        return ids[0], 0.0, substituted

    # Unreachable in a legal spot, but never crash a match over it.
    return legal.legal_ids()[0], 0.0, True


# ---------------------------------------------------------------------------
# Warnings, bucketed so they can be counted.
# ---------------------------------------------------------------------------

WARNING_TAGS = (
    ("stack_mismatch", "Stack mismatch"),
    ("sizing_snapped", "Sizing snapped"),
    ("board_snapped", "Board snapped"),
    ("hand_resettled", "Hand resettled"),
    ("turn_resettled", "Turn card resettled"),
    ("pot_mismatch", "Pot mismatch"),
    ("limped_pot", "imped pot"),
    ("hu_handwritten_ranges", "preflop ranges are hand-written"),
    ("weak_limper_assumption", "assumes a weak limper"),
    ("no_solve", "No solve covers this spot"),
    ("no_preflop_chart", "No preflop chart covers"),
    ("random_hand_equity", "Equity is measured against"),
    ("never_bluffs", "never bluffs"),
    ("close_decision", "Close decision"),
    ("ambiguous_flop_line", "flop line is ambiguous"),
    ("incomplete_options", "frequencies do not sum to 1"),
    ("iso_approximation", "Iso range approximated"),
    ("push_fold_deep", "jam-or-fold"),
)


def tag_warnings(warnings: Sequence[str]) -> List[str]:
    tags = []
    for warning in warnings:
        for tag, needle in WARNING_TAGS:
            if needle in warning and tag not in tags:
                tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# The agent.
# ---------------------------------------------------------------------------

class AdvisorAgent:
    """The advisor, playing.

    It reads the engine state directly (``bind``) rather than the network-facing
    observation, because the advisor's question is about the whole table -- who has how
    much in front of them, where the button is -- and the observation is canonicalised
    for a network and drops most of that.  Only hero's own hole cards are ever read.
    """

    def __init__(self, name: str = "advisor", mode: str = "sample", seed: int = 0):
        self.name = name
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.env = None
        self.log: List[dict] = []
        self.hand_id = 0

    def bind(self, env, hand_id: int = 0) -> None:
        self.env = env
        self.hand_id = hand_id

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def act(self, observation, flat_observation, legal_mask, rng) -> Decision:
        import numpy as np

        state = self.env.state
        seat = observation["meta"]["self_seat"]
        street = STREETS[int(state.street)]
        request = advisor_state(state, seat, street)

        advice = consult(request)
        option = pick_option(advice.options, self.rng, self.mode)

        legal = self.env.legal_actions(seat)
        space = self.env.action_space
        action, snap, substituted = choose_action_id(
            option, legal, space, float(state.big_blind))

        self.log.append({
            "hand": self.hand_id,
            "seat": seat,
            "position": ("BTN", "SB", "BB")[state.position_of(seat)],
            "street": street,
            "source": advice.source,
            "hand_class": advice.detail.get("hand", ""),
            "scenario": advice.detail.get("scenario", ""),
            "advised": option.label,
            "kind": option.kind,
            "frequency": round(option.frequency, 4),
            "pure": advice.pure,
            "num_options": len(advice.options),
            "requested_chips": option.chips,
            "played": space.name(action),
            "played_chips": legal.to_amounts.get(action),
            "snap_bb": round(snap, 3),
            "substituted": substituted,
            "pot": float(state.pot),
            "to_call": float(state.to_call(seat)),
            "warnings": "|".join(tag_warnings(advice.warnings)),
            "action_so_far": advice.action_so_far,
        })

        policy = np.zeros(len(legal_mask), dtype=np.float32)
        policy[action] = 1.0
        return Decision(action=action, policy=policy, value=0.0)
