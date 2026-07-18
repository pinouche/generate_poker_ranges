"""HTTP wrapper around the preflop chart lookup and the postflop solve lookup.

    POST /advise   <- the table's game-state json, verbatim
                   -> the strategy for hero's hand at that node
                      (preflop: the qb_ranges charts; flop/turn: the solves)

The lookups live in preflop_advisor, postflop_advisor and hu_advisor; this module only
converts between JSON and those functions, so the endpoint and the CLIs can never
disagree about what the charts and solves say. The street field routes the request,
except that a busted third player routes to the heads-up push/fold tables first.

Run:
    uvicorn api:app --app-dir resources/python --reload
"""

import logging
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import heuristic_advisor as heuristic
import hu_advisor as hu
import postflop_advisor as postflop
from preflop_advisor import Unsupported, advise, describe, size_of

app = FastAPI(
    title="Poker advisor",
    description="Maps a live game state onto the qb_ranges preflop charts "
                "(street=preflop), the solved postflop trees (street=flop/turn), or "
                "the heads-up Nash push/fold tables when a third player has busted.",
    version="1.2.0",
)

# Piggyback on uvicorn's handlers so these lines land wherever the server's own log does.
log = logging.getLogger("uvicorn.error").getChild("advise")


def timestamp_logs(datefmt: str = "%H:%M:%S") -> None:
    """Prefix uvicorn's log lines with the time of day.

    Uvicorn's default format has no timestamp, which makes a log of hands impossible to
    line up with the table. It applies its own dictConfig before importing this module, so
    the formatters we need to amend already exist -- we edit them in place rather than
    fight for ownership of the logging config.
    """
    # "uvicorn" carries the handler that uvicorn.error propagates to; access has its own.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(name).handlers:
            fmt = handler.formatter
            if fmt and fmt._fmt and "%(asctime)s" not in fmt._fmt:
                handler.setFormatter(type(fmt)(f"%(asctime)s {fmt._fmt}", datefmt))

    # Anything but uvicorn (pytest, `python -m fastapi dev`, a bare import) leaves the
    # logger bare, and a chart lookup that logs nowhere is worse than an ugly log line.
    if not log.hasHandlers():
        logging.basicConfig(level=logging.INFO, datefmt=datefmt,
                            format="%(asctime)s %(levelname)s: %(message)s")


timestamp_logs()


class Card(BaseModel):
    rank: str
    suit: str


class Player(BaseModel):
    # The table sends fields we do not use (equity_pct, and whatever it grows next), and a
    # new one should not turn into a 422 on a spot we could have answered.
    model_config = ConfigDict(extra="allow")

    cards: list[Card] = []
    stack: float
    bet: float = 0
    active: bool = True


class GameState(BaseModel):
    model_config = ConfigDict(extra="allow")

    hero: Player
    # A villain may be absent: the table drops (or zeroes out) a player who busted, and
    # the hand is then heads-up rather than malformed.
    villain_left: Player | None = None
    villain_right: Player | None = None
    board: list[Any] = []
    pot: float = 0
    small_blind: float | None = None
    big_blind: float | None = None
    dealer: Literal["hero", "villain_left", "villain_right"]
    street: str
    showdown: bool = False


class Option(BaseModel):
    action: str = Field(description="human-readable, e.g. 'RAISE to 50 (2.5bb)'")
    kind: Literal["RAISE", "CALL", "FOLD", "ALLIN", "CHECK", "BET"]
    chips: float | None = Field(None, description="total bet to make, in chips")
    frequency: float = Field(description="how often the chart takes this action, 0-1")


class Advice(BaseModel):
    hand: str
    hero_seat: str
    seats: dict[str, str]
    action_so_far: str
    options: list[Option]
    recommendation: Option
    pure: bool = Field(description="true if the chart takes one action 100% of the time")
    warnings: list[str] = Field(
        description="reasons to trust this less than it looks, e.g. a stack-depth mismatch")


def kind_of(action: str) -> str:
    return {"FOLD": "FOLD", "Check": "CHECK", "Call": "CALL",
            "AllIn": "ALLIN"}.get(action, "RAISE")


def to_option(action: str, weight: float, bb: float, to_call: float) -> Option:
    kind = kind_of(action)
    chips = size_of(action) * bb if kind == "RAISE" else (to_call if kind == "CALL" else None)
    return Option(action=describe(action, bb, to_call), kind=kind,
                  chips=chips, frequency=round(weight, 4))


# Log lines are read by a human watching a live table, so they are laid out as a block
# rather than dumped as JSON: the spot on top, the chart's answer under it.
INDENT = "\n" + " " * 10


def render_request(state: GameState) -> str:
    seats = (("hero", state.hero), ("villain_left", state.villain_left),
             ("villain_right", state.villain_right))
    rows = [f"{name:<14}{cards_of(p):<8}{p.stack:>8,.0f}{p.bet:>8,.0f}"
            f"{'' if p.active else '   folded'}" if p is not None else
            f"{name:<14}{'--':<8}{'':>8}{'':>8}   busted"
            for name, p in seats]
    header = f"{'':<14}{'cards':<8}{'stack':>8}{'bet':>8}"
    board = "".join(f"{c['rank']}{c['suit']}" if isinstance(c, dict) else str(c)
                    for c in state.board)
    return INDENT.join([f"POST /advise <- {state.street}"
                        f"{' ' + board if board else ''}, pot {state.pot:,.0f}, "
                        f"dealer {state.dealer}", header, *rows])


def cards_of(p: Player) -> str:
    return "".join(f"{c.rank}{c.suit}" for c in p.cards) or "--"


def render_advice(a: Advice) -> str:
    others = ", ".join(f"{name.replace('villain_', '')} {seat}"
                       for name, seat in a.seats.items() if name != "hero")
    width = max(len(o.action) for o in a.options)
    rows = [f"{o.action:<{width}}  {o.frequency:>6.1%}"
            f"{'  <- recommended' if o is a.recommendation else ''}"
            for o in a.options]
    warnings = [f"! {w}" for w in a.warnings]
    return INDENT.join([f"POST /advise -> 200  {a.hand} as {a.hero_seat} ({others})",
                        a.action_so_far, *rows, *warnings])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def preflop_advice(state: GameState, advise_fn=advise) -> Advice:
    r = advise_fn(state.model_dump())

    bb, to_call = r["bb"], r["to_call"]
    options = [to_option(a, w, bb, to_call) for a, w, _ in r["strategy"]]
    line = " -> ".join(f"{s} {describe(a, bb)}" for s, a in r["prior"]) or "folds to hero"

    return Advice(
        hand=r["hand"],
        hero_seat=r["hero_seat"],
        seats=r["seats"],
        action_so_far=line,
        options=options,
        recommendation=options[0],
        pure=options[0].frequency >= 0.999,
        warnings=r["warnings"],
    )


def postflop_advice(state: GameState) -> Advice:
    r = postflop.advise(state.model_dump())

    bb, to_call = r["bb"], r["to_call"]
    options = [Option(action=postflop.describe(a, bb, to_call),
                      kind=postflop.kind_of(a),
                      chips=postflop.chips_of(a, bb, to_call),
                      frequency=round(w, 4))
               for a, w in r["strategy"]]

    return Advice(
        hand=r["hand"],
        hero_seat=r["hero_seat"],
        seats=r["seats"],
        action_so_far=postflop.action_line(r),
        options=options,
        recommendation=options[0],
        pure=options[0].frequency >= 0.999,
        warnings=r["warnings"],
    )


def heuristic_advice(state: GameState, reason: str) -> Advice:
    r = heuristic.advise(state.model_dump(), reason=reason)
    options = [Option(**o) for o in r["options"]]
    return Advice(
        hand=r["hand"],
        hero_seat=r["hero_seat"],
        seats=r["seats"],
        action_so_far=r["action_so_far"],
        options=options,
        recommendation=options[0],
        pure=options[0].frequency >= 0.999,
        warnings=r["warnings"],
    )


@app.post("/advise", response_model=Advice)
def advise_endpoint(state: GameState) -> Advice:
    """Read the chart or the solve for whatever spot this state describes.

    Preflop is a chart lookup; the flop and the turn are read out of the solved trees.
    A postflop spot the solves cannot answer (a multiway flop, a river, heads-up after
    a bust, solves missing from disk) falls back to the equity-vs-pot-odds heuristic,
    flagged as such in the warnings -- like a preflop limp, a practical answer beats a
    422. Only a spot with no sane answer at all (hero folded, malformed cards, unknown
    street) is a 422, not a 500: it is a fine question with no answer here, and the
    client needs to tell that apart from us being broken.
    """
    log.info(render_request(state))

    try:
        if hu.is_heads_up(state.model_dump()):
            # A busted third player makes this true heads-up: none of the 3-handed
            # material applies, so route every street to the push/fold tables --
            # hu.advise itself rejects postflop, with the reason why.
            advice = preflop_advice(state, hu.advise)
        elif state.street == "preflop":
            advice = preflop_advice(state)
        else:
            advice = postflop_advice(state)
    except Unsupported as e:
        try:
            advice = heuristic_advice(state, str(e))
        except Unsupported:
            # The heuristic could not answer either; the original reason is the one
            # that names the spot, so it is the one worth reporting.
            log.info("POST /advise -> 422  no chart for this spot: %s", e)
            raise HTTPException(status_code=422, detail=f"No chart for this spot: {e}")

    log.info(render_advice(advice))
    return advice
