#!/usr/bin/env python3
"""Play the ``/advise`` endpoint against Slumbot, alternating seats, and score it.

``slumbot.py`` deals the hands and counts the money; this module is the only thing
between its wire protocol and the endpoint in ``api.py``:

  1. **Replay.**  Slumbot sends the whole hand as one string (``b200c/kb200``).  The
     endpoint wants a table -- who has how much in front of them, what is in the middle,
     whose button.  ``replay`` walks the string and rebuilds that, which is the one piece
     of code here that can be silently wrong, so it is checked against Slumbot's own
     ``winnings`` on every hand that ends in a fold (see ``accounting_error``).
  2. **Ask.**  ``AdvisorPolicy.consult`` calls ``api.advise_endpoint`` -- the real handler,
     in process by default, or over HTTP with ``--url`` -- so what plays here is what a
     live table would be told.  Nothing re-implements the routing.
  3. **Answer.**  The endpoint replies in chips ("RAISE to 250"); Slumbot wants ``b250``.
     Sizings are clamped to what is legal (min-raise, all-in) and the distance moved is
     recorded per decision rather than swallowed.

**A 422 is part of the result, not an error to hide.**  Heads-up preflop the endpoint
routes to the jam-or-fold tables, and every node off that tree -- the big blind facing a
raise that is not a shove, the small blind facing a 3bet -- is a 422 with no advice at
all.  A match cannot fold to a 422 (folding is a strategy, and a bad one), so those spots
are answered by ``advisor_bridge.preflop_equity_advice``, exactly as the arena does, and
counted separately under ``preflop_equity`` so they are never mistaken for the advisor.

**The seat alternates** because ``SlumbotClient`` carries its session token; with 1,000
hands that is 500 from each seat, reported apart as well as together.

Usage
-----
    python3 slumbot_advisor.py --hands 1000
    python3 slumbot_advisor.py --hands 100 --policy fold      # harness calibration
    python3 slumbot_advisor.py --hands 50 --url http://127.0.0.1:8000/advise
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ADVISOR_DIR = os.path.dirname(HERE)
for path in (HERE, ADVISOR_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import advisor_bridge as bridge                                        # noqa: E402
import api                                                             # noqa: E402
from advisor_bridge import FALLBACKS, Option, pick_option, tag_warnings  # noqa: E402
from slumbot import (BIG_BLIND, SMALL_BLIND, STACK, HandResult,        # noqa: E402
                     HandState, SessionSummary, by_seat, call_policy,
                     fold_policy, play_session, summarise)

STREETS = ("preflop", "flop", "turn", "river")
BOARD_CARDS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
HERO, BOT = "hero", "bot"

#: One action.  ``bX`` first so it is not read as a bare ``b``.
TOKEN = re.compile(r"b\d+|[kcf]")


# ---------------------------------------------------------------------------
# Slumbot's action string -> a table.
# ---------------------------------------------------------------------------

@dataclass
class Spot:
    """The table as the endpoint needs to see it, rebuilt from the action string.

    ``bet`` fields are this street's commitment and ``paid`` fields are the whole hand's,
    which is the same split the advisor uses (``pot`` includes the live bets).
    """

    street: str
    hero_bet: float
    villain_bet: float
    hero_paid: float
    villain_paid: float
    pot: float
    to_call: float
    min_raise_to: float
    all_in_to: float
    hero_is_button: bool
    folded: Optional[str] = None

    @property
    def hero_stack(self) -> float:
        return STACK - self.hero_paid

    @property
    def villain_stack(self) -> float:
        return STACK - self.villain_paid

    @property
    def can_raise(self) -> bool:
        """A raise needs chips behind the current bet; against a shove there are none."""
        return self.all_in_to > max(self.hero_bet, self.villain_bet) + 1e-9


def replay(action: str, client_pos: int) -> Spot:
    """Rebuild the table from the hand's action string.

    Two conventions decide everything and both were verified against the live API rather
    than assumed: ``bX`` sets the actor's total commitment **on this street** to X (so a
    preflop ``b200`` from the small blind is 150 more, not 200), and the actor alternates
    on every token -- heads-up, any action passes the turn or ends the street.  The button
    posts the small blind and acts first preflop; the big blind acts first on every later
    street, which is why the two first-actors differ below.
    """
    hero_button = client_pos == 1
    paid = {HERO: 0.0, BOT: 0.0}          # settled streets only
    bet = {HERO: 0.0, BOT: 0.0}
    folded = None
    min_increment = float(BIG_BLIND)
    streets = action.split("/")

    for index, tokens in enumerate(streets):
        button, other = (HERO, BOT) if hero_button else (BOT, HERO)
        if index == 0:
            bet = {button: float(SMALL_BLIND), other: float(BIG_BLIND)}
            actor = button
        else:
            bet = {HERO: 0.0, BOT: 0.0}
            actor = other
        # A raise must be at least as large as the last one, and never less than a big
        # blind -- which is also the smallest opening bet on a fresh street.
        min_increment = float(BIG_BLIND)

        for token in TOKEN.findall(tokens):
            if token == "f":
                folded = actor
            elif token == "c":
                bet[actor] = min(max(bet.values()), STACK - paid[actor])
            elif token.startswith("b"):
                to = float(token[1:])
                min_increment = max(min_increment, to - max(bet.values()))
                bet[actor] = to
            actor = BOT if actor == HERO else HERO

        if index < len(streets) - 1:
            for player in paid:
                paid[player] += bet[player]

    top = max(bet.values())
    hero_paid, villain_paid = paid[HERO] + bet[HERO], paid[BOT] + bet[BOT]
    return Spot(
        street=STREETS[len(streets) - 1],
        hero_bet=bet[HERO],
        villain_bet=bet[BOT],
        hero_paid=hero_paid,
        villain_paid=villain_paid,
        pot=hero_paid + villain_paid,
        to_call=max(0.0, top - bet[HERO]),
        min_raise_to=top + min_increment,
        all_in_to=float(STACK) - paid[HERO],
        hero_is_button=hero_button,
        folded=folded,
    )


def accounting_error(result: HandResult) -> Optional[str]:
    """Check the replay against the only ground truth Slumbot gives us: the money.

    A hand that ends in a fold has a known result -- the folder loses exactly what they
    put in -- so if the replay's contributions disagree with ``winnings``, the state the
    endpoint was shown was wrong, and every decision in the run is suspect.  Showdowns
    say nothing here (the pot is split by cards, not by the line), so they are skipped.
    """
    spot = replay(result.action, result.client_pos)
    if spot.folded == HERO and result.winnings != -spot.hero_paid:
        return (f"hand {result.hand}: we folded having paid {spot.hero_paid:g}, "
                f"slumbot says {result.winnings}")
    if spot.folded == BOT and result.winnings != spot.villain_paid:
        return (f"hand {result.hand}: bot folded having paid {spot.villain_paid:g}, "
                f"slumbot says {result.winnings}")
    return None


# ---------------------------------------------------------------------------
# A table -> the /advise request.
# ---------------------------------------------------------------------------

def card(text: str) -> Dict[str, str]:
    """``'Ah'`` -> ``{'rank': 'A', 'suit': 'h'}``, the shape the endpoint's Card takes."""
    return {"rank": text[0].upper(), "suit": text[1].lower()}


#: The third chair, busted.  This is what makes the endpoint read the table as true
#: heads-up (``hu_advisor.is_heads_up``): a seat with no chips and no action, which is how
#: a real table encodes a player who has been knocked out.  It is sent as a seat rather
#: than dropped because the postflop lookup walks all three keys.
BUSTED = {"cards": [], "stack": 0.0, "bet": 0.0, "active": False}


def advise_request(hand: HandState, spot: Spot) -> dict:
    """The game-state json for this spot, from hero's point of view."""
    board = [card(c) for c in hand.board[:BOARD_CARDS[spot.street]]]
    if len(board) != BOARD_CARDS[spot.street]:
        raise RuntimeError(f"{spot.street} needs {BOARD_CARDS[spot.street]} board cards, "
                           f"slumbot sent {list(hand.board)}")
    return {
        "hero": {"cards": [card(c) for c in hand.hole_cards],
                 "stack": spot.hero_stack, "bet": spot.hero_bet, "active": True},
        "villain_left": {"cards": [], "stack": spot.villain_stack,
                         "bet": spot.villain_bet, "active": True},
        "villain_right": dict(BUSTED),
        "board": board,
        "pot": spot.pot,
        "small_blind": float(SMALL_BLIND),
        "big_blind": float(BIG_BLIND),
        # Heads-up the dealer posts the small blind and acts first preflop, which is
        # exactly Slumbot's ``client_pos == 1``.
        "dealer": "hero" if spot.hero_is_button else "villain_left",
        "street": spot.street,
        "showdown": False,
    }


# ---------------------------------------------------------------------------
# The /advise answer -> a Slumbot action.
# ---------------------------------------------------------------------------

def to_action(option: Option, spot: Spot) -> Tuple[str, float, bool]:
    """(the ``incr`` string, sizing moved in bb, whether the kind had to be substituted).

    Slumbot's grammar is narrower than the advisor's vocabulary: ``c`` is strictly a call
    and is rejected when there is nothing to call, ``f`` needs a bet to fold to, and a
    raise has to clear the minimum and stop at the stack.  Where the ask cannot be played
    the substitution follows ``advisor_bridge.FALLBACKS`` -- the same order the engine
    arena uses -- and how far a sizing had to move is returned, not hidden.
    """
    facing = spot.to_call > 1e-9
    for step, kind in enumerate(FALLBACKS[option.kind]):
        substituted = step > 0
        if kind == "FOLD" and facing:
            return "f", 0.0, substituted
        if kind == "CHECK" and not facing:
            return "k", 0.0, substituted
        if kind == "CALL" and facing:
            return "c", 0.0, substituted
        if kind in ("BET", "RAISE", "ALLIN") and spot.can_raise:
            wanted = spot.all_in_to if kind == "ALLIN" else option.chips
            if wanted is None:
                continue
            # Below the minimum raise is not a legal size; above the stack does not exist.
            amount = min(max(float(wanted), spot.min_raise_to), spot.all_in_to)
            amount = int(round(amount))
            return f"b{amount}", abs(amount - float(wanted)) / BIG_BLIND, substituted
    # Nothing in the fallback chain was legal, which should not happen at a live node.
    return ("c" if facing else "k"), 0.0, True


# ---------------------------------------------------------------------------
# Asking the endpoint.
# ---------------------------------------------------------------------------

class Unanswered(Exception):
    """The endpoint returned 422: a fine question it has no chart for."""


def source_of(street: str, warnings: Sequence[str]) -> str:
    """Which lookup behind the endpoint actually answered.

    The response does not say, but the routing is deterministic and the heuristic labels
    itself: heads-up preflop is the jam-or-fold tables, postflop is the solves, and
    anything the solves decline carries the heuristic's own "No solve covers this spot".
    """
    if any(w.startswith("No solve covers this spot") for w in warnings):
        return "heuristic"
    return "headsup_table" if street == "preflop" else "postflop_solve"


def _http_advise(url: str, request: dict) -> dict:
    import urllib.error
    import urllib.request

    body = json.dumps(request).encode()
    try:
        post = urllib.request.Request(url, data=body,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(post, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        if error.code == 422:
            raise Unanswered(error.read().decode()[:200]) from None
        raise


class AdvisorPolicy:
    """The endpoint, playing Slumbot.

    ``mode='sample'`` plays the chart's own frequencies, which is how a mixed strategy is
    meant to be played; ``max`` always takes its most common action.
    """

    def __init__(self, mode: str = "sample", seed: int = 0, url: Optional[str] = None):
        self.mode = mode
        self.rng = random.Random(seed)
        self.url = url
        self.log: List[dict] = []
        self.hand_id = 0

    # -- the two ways to reach /advise ------------------------------------
    def consult(self, request: dict, street: str) -> Tuple[str, List[Option], List[str], str, str]:
        try:
            if self.url:
                answer = _http_advise(self.url, request)
                options = [Option(o["kind"], o["chips"], float(o["frequency"]), o["action"])
                           for o in answer["options"]]
                warnings, label, line = (answer["warnings"], answer["hand"],
                                         answer["action_so_far"])
            else:
                advice = api.advise_endpoint(api.GameState(**request))
                options = [Option(o.kind, o.chips, float(o.frequency), o.action)
                           for o in advice.options]
                warnings, label, line = advice.warnings, advice.hand, advice.action_so_far
            return source_of(street, warnings), options, warnings, label, line
        except Unanswered as reason:
            return self._fallback(request, str(reason))
        except Exception as error:                       # noqa: BLE001
            from fastapi import HTTPException
            if isinstance(error, HTTPException) and error.status_code == 422:
                return self._fallback(request, str(error.detail))
            raise

    def _fallback(self, request: dict, reason: str) -> Tuple[str, List[Option], List[str], str, str]:
        """No chart for the spot.  Put a chip in anyway, and say so.

        The jam-or-fold tables cover two preflop nodes and refuse the rest; a match still
        has to act.  This is the arena's own last resort -- equity against a random hand
        versus the pot odds -- flagged with its own source so these decisions can be
        counted out of any claim about the advisor.
        """
        answer = bridge.preflop_equity_advice(request)
        return ("preflop_equity", list(answer["options"]),
                [f"422: {reason}"] + answer["warnings"], "", answer["action_so_far"])

    # -- the policy Slumbot calls -----------------------------------------
    def __call__(self, hand: HandState) -> str:
        spot = replay(hand.action, hand.client_pos)
        request = advise_request(hand, spot)
        started = time.perf_counter()
        source, options, warnings, label, line = self.consult(request, spot.street)
        option = pick_option(options, self.rng, self.mode)
        incr, snap, substituted = to_action(option, spot)

        self.log.append({
            "hand": self.hand_id,
            "seat": "button" if spot.hero_is_button else "big_blind",
            "street": spot.street,
            "source": source,
            "hole": " ".join(hand.hole_cards),
            "hand_class": label,
            "pot": spot.pot,
            "to_call": spot.to_call,
            "advised": option.label,
            "kind": option.kind,
            "frequency": round(option.frequency, 4),
            "pure": bool(options) and options[0].frequency >= 0.999,
            "requested_chips": option.chips,
            "played": incr,
            "snap_bb": round(snap, 3),
            "substituted": substituted,
            "warnings": "|".join(tag_warnings(warnings)),
            "action_so_far": line,
            "seconds": round(time.perf_counter() - started, 3),
        })
        return incr


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarise_run(results: Sequence[HandResult], decisions: Sequence[dict],
                  overall: SessionSummary, meta: dict, mismatches: Sequence[str]) -> dict:
    sources = Counter(d["source"] for d in decisions)
    streets = Counter(d["street"] for d in decisions)
    snaps = [d["snap_bb"] for d in decisions if d["kind"] in ("BET", "RAISE", "ALLIN")]
    return {
        "meta": meta,
        "overall": overall.to_dict(),
        "by_seat": by_seat(results),
        "hands_reaching": dict(Counter(
            replay(r.action, r.client_pos).street for r in results)),
        "showdowns": sum(1 for r in results if r.bot_hole_cards),
        "decisions": {
            "total": len(decisions),
            "by_source": dict(sources),
            "by_source_share": {k: v / max(1, len(decisions)) for k, v in sources.items()},
            "by_street": dict(streets),
            "by_kind": dict(Counter(d["kind"] for d in decisions)),
            "substituted_kind": sum(1 for d in decisions if d["substituted"]),
            "sizing_snap_bb": {
                "n": len(snaps),
                "mean": sum(snaps) / len(snaps) if snaps else 0.0,
                "max": max(snaps) if snaps else 0.0,
            },
            "mean_seconds": (sum(d["seconds"] for d in decisions) / len(decisions)
                             if decisions else 0.0),
        },
        "warnings": dict(Counter(tag for d in decisions
                                 for tag in d["warnings"].split("|") if tag)),
        "accounting_mismatches": list(mismatches),
    }


def report(summary: dict) -> str:
    overall, seats = summary["overall"], summary["by_seat"]
    meta = summary["meta"]
    lines = [
        "",
        f"{overall['hands']} hands vs Slumbot, {STACK // BIG_BLIND}bb deep "
        f"({SMALL_BLIND}/{BIG_BLIND}), seat alternating -- {meta['policy']}",
        "",
        f"  {'seat':<14}{'hands':>7}{'chips':>10}{'mbb/g':>10}{'95% CI':>22}",
    ]
    rows = [("both seats", overall)] + [(name.replace("_", " "), seats[name])
                                        for name in ("button", "big_blind") if name in seats]
    for name, stats in rows:
        low, high = stats["ci95_mbb"]
        lines.append(f"  {name:<14}{stats['hands']:>7}{stats['total_chips']:>10,}"
                     f"{stats['mbb_per_game']:>10.0f}{f'[{low:,.0f}, {high:,.0f}]':>22}")

    mismatches = summary["accounting_mismatches"]
    verdict = "ok on every folded hand" if not mismatches \
        else f"{len(mismatches)} MISMATCHES"
    lines += [
        "",
        f"  state-replay check: {verdict}",
        "  hands reached: " + ", ".join(f"{k} {v}" for k, v in
                                        summary["hands_reaching"].items()),
    ]

    decisions = summary["decisions"]
    if decisions["total"]:
        lines += ["", f"  {decisions['total']} decisions, answered by:"]
        for source, count in sorted(decisions["by_source"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {source:<18}{count:>7}  {count / decisions['total']:>6.1%}")
        snap = decisions["sizing_snap_bb"]
        lines += [
            f"  sizing snap: mean {snap['mean']:.2f}bb, max {snap['max']:.2f}bb over "
            f"{snap['n']} bets/raises",
            f"  kind substituted (illegal at slumbot): {decisions['substituted_kind']}",
            "",
            "  warnings raised (a decision may raise several):",
        ]
        for tag, count in sorted(summary["warnings"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {tag:<26}{count:>7}")
    return "\n".join(lines)


def load_run(out: str, tag: str) -> Tuple[List[HandResult], List[dict]]:
    """The hands and decisions a finished run left on disk."""
    results = []
    with open(os.path.join(out, f"slumbot_hands_{tag}.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            results.append(HandResult(
                hand=row["hand"], client_pos=row["client_pos"],
                hole_cards=tuple(row["hole_cards"]), board=tuple(row["board"]),
                action=row["action"], winnings=row["winnings"],
                bot_hole_cards=tuple(row["bot_hole_cards"]) or None,
                seconds=row["seconds"]))
    path = os.path.join(out, f"slumbot_decisions_{tag}.csv")
    decisions = []
    if os.path.isfile(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["snap_bb"] = float(row["snap_bb"])
                row["seconds"] = float(row["seconds"])
                row["substituted"] = row["substituted"] == "True"
                decisions.append(row)
    return results, decisions


def merge(out: str, tags: Sequence[str], tag: str) -> dict:
    """Score several sessions as one experiment.

    A long session is played as a few concurrent ones -- each carries its own Slumbot
    token, so each alternates its own seats and the hands stay independent.  Pooling the
    per-hand winnings is therefore exactly the same estimator as one long run; the wall
    clock is the only thing that changes.
    """
    results: List[HandResult] = []
    decisions: List[dict] = []
    for source in tags:
        run_results, run_decisions = load_run(out, source)
        results.extend(run_results)
        decisions.extend(run_decisions)
    seconds = sum(r.seconds for r in results)
    overall = summarise([r.winnings for r in results], seconds)
    mismatches = [m for m in (accounting_error(r) for r in results) if m]
    meta = {"merged_from": list(tags), "hands_played": overall.hands,
            "policy": "/advise (sample, in process)", "stack": STACK,
            "blinds": [SMALL_BLIND, BIG_BLIND], "seconds": round(seconds, 1)}
    summary = summarise_run(results, decisions, overall, meta, mismatches)
    with open(os.path.join(out, f"slumbot_summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

POLICIES = {"advisor": None, "fold": fold_policy, "call": call_policy}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--policy", default="advisor", choices=sorted(POLICIES),
                        help="'advisor' plays /advise; 'fold' and 'call' are the "
                             "calibration opponents with known win rates")
    parser.add_argument("--mode", default="sample", choices=("sample", "max"),
                        help="'sample' plays the chart's mixed frequencies (default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--url", default=None,
                        help="POST to a running server instead of calling the handler "
                             "in process, e.g. http://127.0.0.1:8000/advise")
    parser.add_argument("--out", default=os.path.join(HERE, "results"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--every", type=int, default=25, help="progress line frequency")
    parser.add_argument("--merge", nargs="+", metavar="TAG",
                        help="score finished runs as one experiment instead of playing")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.merge:
        summary = merge(args.out, args.merge, args.tag or "merged")
        print(report(summary))
        for line in summary["accounting_mismatches"][:5]:
            print(f"    ! {line}")
        return 0
    tag = f"_{args.tag}" if args.tag else ""
    hands_path = os.path.join(args.out, f"slumbot_hands{tag}.jsonl")
    if os.path.exists(hands_path):
        os.remove(hands_path)          # a log that mixes two runs cannot be counted
    bridge.CRASH_LOG = os.path.join(args.out, f"slumbot_advisor_crashes{tag}.jsonl")
    if os.path.exists(bridge.CRASH_LOG):
        os.remove(bridge.CRASH_LOG)

    policy = (AdvisorPolicy(mode=args.mode, seed=args.seed, url=args.url)
              if args.policy == "advisor" else POLICIES[args.policy])
    results: List[HandResult] = []
    mismatches: List[str] = []

    def on_hand(result: HandResult, running: SessionSummary) -> None:
        results.append(result)
        error = accounting_error(result)
        if error:
            mismatches.append(error)
        if isinstance(policy, AdvisorPolicy):
            policy.hand_id = result.hand + 1
        if args.every and running.hands % args.every == 0:
            low, high = running.ci95_mbb
            print(f"  {running.hands:5d} hands  {running.mbb_per_game:>8.0f} mbb/g  "
                  f"[{low:,.0f}, {high:,.0f}]  {running.seconds:6.0f}s", flush=True)

    label = (f"/advise ({args.mode}{', over http' if args.url else ', in process'})"
             if args.policy == "advisor" else f"{args.policy}_policy")
    print(f"{label} vs slumbot, {args.hands} hands, alternating seats")
    summary_path = os.path.join(args.out, f"slumbot_summary{tag}.json")
    started = time.time()
    try:
        overall = play_session(args.hands, policy, log_path=hands_path, on_hand=on_hand)
    except KeyboardInterrupt:
        overall = summarise([r.winnings for r in results], time.time() - started)
        print("\ninterrupted; scoring the hands that finished")

    meta = {
        "hands_requested": args.hands,
        "hands_played": overall.hands,
        "policy": label,
        "mode": args.mode,
        "seed": args.seed,
        "stack": STACK,
        "blinds": [SMALL_BLIND, BIG_BLIND],
        "endpoint": args.url or "api.advise_endpoint (in process)",
        "seconds": round(time.time() - started, 1),
    }
    decisions = policy.log if isinstance(policy, AdvisorPolicy) else []
    summary = summarise_run(results, decisions, overall, meta, mismatches)

    write_csv(os.path.join(args.out, f"slumbot_decisions{tag}.csv"), decisions)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(report(summary))
    for line in mismatches[:5]:
        print(f"    ! {line}")
    print(f"\nwritten to {args.out}  ({meta['seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
