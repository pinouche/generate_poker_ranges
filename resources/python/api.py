"""HTTP wrapper around the preflop chart lookup.

    POST /advise   <- the table's game-state json, verbatim
                   -> the chart's strategy for hero's hand at that node

The chart reading lives in preflop_advisor; this module only converts between JSON and
that function, so the endpoint and the CLI can never disagree about what the charts say.

Run:
    uvicorn api:app --app-dir resources/python --reload
"""

import json
import logging
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from preflop_advisor import Unsupported, advise, describe, size_of

app = FastAPI(
    title="Preflop advisor",
    description="Maps a live game state onto the qb_ranges preflop charts.",
    version="1.0.0",
)

# Piggyback on uvicorn's handlers so these lines land wherever the server's own log does.
log = logging.getLogger("uvicorn.error").getChild("advise")


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
    villain_left: Player
    villain_right: Player
    board: list[Any] = []
    pot: float = 0
    small_blind: float | None = None
    big_blind: float | None = None
    dealer: Literal["hero", "villain_left", "villain_right"]
    street: str
    showdown: bool = False


class Option(BaseModel):
    action: str = Field(description="human-readable, e.g. 'RAISE to 50 (2.5bb)'")
    kind: Literal["RAISE", "CALL", "FOLD", "ALLIN"]
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
    return {"FOLD": "FOLD", "Call": "CALL", "AllIn": "ALLIN"}.get(action, "RAISE")


def to_option(action: str, weight: float, bb: float, to_call: float) -> Option:
    kind = kind_of(action)
    chips = size_of(action) * bb if kind == "RAISE" else (to_call if kind == "CALL" else None)
    return Option(action=describe(action, bb, to_call), kind=kind,
                  chips=chips, frequency=round(weight, 4))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/advise", response_model=Advice)
def advise_endpoint(state: GameState) -> Advice:
    """Read the chart for whatever spot this state describes.

    A spot the charts do not cover is a 422, not a 500: it is a fine question with no
    answer here (a postflop street, a limped pot, an unsupported seating), and the client
    needs to tell those apart from us being broken.
    """
    payload = state.model_dump()
    log.info("POST /advise <- %s", json.dumps(payload, default=str))

    try:
        r = advise(payload)
    except Unsupported as e:
        log.info("POST /advise -> 422 No chart for this spot: %s", e)
        raise HTTPException(status_code=422, detail=f"No chart for this spot: {e}")

    bb, to_call = r["bb"], r["to_call"]
    options = [to_option(a, w, bb, to_call) for a, w, _ in r["strategy"]]
    line = " -> ".join(f"{s} {describe(a, bb)}" for s, a in r["prior"]) or "folds to hero"

    advice = Advice(
        hand=r["hand"],
        hero_seat=r["hero_seat"],
        seats=r["seats"],
        action_so_far=line,
        options=options,
        recommendation=options[0],
        pure=options[0].frequency >= 0.999,
        warnings=r["warnings"],
    )
    log.info("POST /advise -> 200 %s", advice.model_dump_json())
    return advice
