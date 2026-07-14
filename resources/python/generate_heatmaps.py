import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Rectangle

# Standard poker grid order
RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
RANK_IDX = {r: i for i, r in enumerate(RANKS)}

# Cells with no combos in the strategy (blocked by the board, or outside the range)
EMPTY_COLOR = '#e8e7e3'

# Passive actions keep the conventional poker semantics.
ACTION_COLORS = {
    'CHECK': '#008300',   # green
    'CALL':  '#1baf7a',   # aqua
    'FOLD':  '#2a78d6',   # blue
}
# Bet/raise size is a magnitude, so it gets a sequential red ramp: light = small, dark = large.
BET_RAMP = ['#f0908f', '#e34948', '#a11f1e']


def get_hand_name(r1, r2):
    if r1 == r2:
        return f"{RANKS[r1]}{RANKS[r2]}"
    elif r1 < r2:
        return f"{RANKS[r1]}{RANKS[r2]}s"
    else:
        return f"{RANKS[r2]}{RANKS[r1]}o"


def total_combos(r1, r2, board=()):
    """How many combos a cell can actually hold once the board is removed from the deck.

    Nominally 6 pairs / 4 suited / 12 offsuit, but a board card kills every combo
    that uses it. Dividing by the nominal count instead of this would draw a cell
    as partially filled when in fact every dealable combo is present.
    """
    board = set(board)
    suits = 'cdhs'
    hi, lo = RANKS[min(r1, r2)], RANKS[max(r1, r2)]
    if r1 == r2:
        live = [hi + s for s in suits if hi + s not in board]
        return len(live) * (len(live) - 1) // 2
    if r1 < r2:  # suited
        return sum(1 for s in suits
                   if hi + s not in board and lo + s not in board)
    return sum(1 for a in suits for b in suits  # offsuit
               if a != b and hi + a not in board and lo + b not in board)


def parse_combo(key):
    """Map a solver combo key ('AcKd', '2d2c') to its grid cell (row, col).

    The solver writes the higher-ranked card first and does not use a fixed suit
    order, so parse the key rather than trying to regenerate it: suited hands sit
    in the upper triangle, offsuit in the lower, pairs on the diagonal.
    """
    if len(key) != 4:
        return None
    hi, hi_suit, lo, lo_suit = key[0], key[1], key[2], key[3]
    if hi not in RANK_IDX or lo not in RANK_IDX:
        return None
    a, b = RANK_IDX[hi], RANK_IDX[lo]
    if a > b:  # defensive: normalise so `a` is always the higher rank (lower index)
        a, b = b, a
    if a == b:
        return a, b                      # pair -> diagonal
    if hi_suit == lo_suit:
        return a, b                      # suited -> upper triangle
    return b, a                          # offsuit -> lower triangle


def board_from_filename(json_path):
    """Read the board off the filename, e.g. 'BTN_vs_BB_As_Ad_Ks.json' -> ['As','Ad','Ks'].

    The JSON itself does not record the board, so the filename is the only
    authoritative source; take the trailing run of card-shaped tokens.
    """
    stem = os.path.splitext(os.path.basename(json_path))[0]
    board = []
    for token in reversed(stem.split('_')):
        if re.fullmatch(r'[AKQJT98765432][cdhs]', token):
            board.insert(0, token)
        else:
            break
    return board if 3 <= len(board) <= 5 else []


def infer_board(strategy_dict):
    """Fallback: the board cards are the ones no combo in the range can contain.

    A hand is dealt from the 52-card deck minus the board, so a card absent from
    every combo is almost certainly on the board. Only trust this when the count
    looks like a real street; a narrow enough range can omit a card innocently.
    """
    used = set()
    for key in strategy_dict:
        used.add(key[0:2])
        used.add(key[2:4])
    missing = sorted({r + s for r in RANKS for s in 'cdhs'} - used)
    return missing if 3 <= len(missing) <= 5 else []


def build_action_colors(actions):
    """Assign a color per action, ordering bet/raise sizes light->dark by size."""
    colors = {}
    sizes = []
    for action in actions:
        if 'BET' in action or 'RAISE' in action:
            m = re.search(r'[\d.]+', action)
            sizes.append((float(m.group()) if m else 0.0, action))
        else:
            colors[action] = ACTION_COLORS.get(action, '#4a3aa7')

    for rank, (_, action) in enumerate(sorted(sizes)):
        if len(sizes) == 1:
            colors[action] = BET_RAMP[1]
        else:
            # spread the available sizes across the ramp
            pos = rank / (len(sizes) - 1) * (len(BET_RAMP) - 1)
            colors[action] = BET_RAMP[int(round(pos))]
    return colors


def ranges_from_config(json_path):
    """Pull both players' ranges out of the solve config sitting next to the json.

    The two players alternate through the tree, so a single range file would be
    wrong at half the nodes. The config records both, keyed by position:
    player 1 acts first on the flop and is therefore OOP, player 0 is IP.
    """
    config = os.path.splitext(json_path)[0] + '.txt'
    if not os.path.exists(config):
        return {}
    with open(config, 'r') as f:
        text = f.read()

    ranges = {}
    for player, key in ((1, 'set_range_oop'), (0, 'set_range_ip')):
        m = re.search(rf'^{key}\s+(.+)$', text, re.MULTILINE)
        if m:
            ranges[player] = parse_range_text(m.group(1))
    return ranges


def parse_range_text(text):
    """'AA:0.0,A2s:1.0,ATs:0.13,22' -> hand -> weight. No ':' means full frequency."""
    weights = {}
    for token in re.split(r'[,\s]+', text.strip()):
        if not token:
            continue
        hand, _, weight = token.partition(':')
        weights[hand] = float(weight) if weight else 1.0
    return weights


def parse_range_file(range_path):
    """Read a solver range file ('AA:0.0,A2s:1.0,ATs:0.13,22,...') into hand -> weight.

    A hand with no ':' is played at full frequency. The json omits weight-0 hands
    entirely, but keeps every combo of a fractional hand, so without these weights
    a 0.13-frequency hand would count as much as a 1.0-frequency one.
    """
    with open(range_path, 'r') as f:
        return parse_range_text(f.read())


def walk_to(data, path):
    """Descend the game tree by a list of action names, e.g. ['CHECK', 'BET 2.000000'].

    Action names must match the json exactly; on a miss we print the legal actions
    rather than raising, since the names carry solver-formatted bet sizes.
    """
    node = data
    for step in path:
        children = node.get('childrens') or {}
        if step not in children:
            print(f"  no action {step!r} here; legal: {list(children)}")
            return None
        node = children[step]
    return node


def describe_path(path):
    return ' > '.join(path) if path else 'root'


def slug_action(action):
    """'BET 2.000000' -> 'BET2', 'RAISE 19.000000' -> 'RAISE19', 'CHECK' -> 'CHECK'."""
    if action.startswith('@'):   # a dealt card, not a player action
        return action[1:]
    m = re.match(r'(\w+)\s+([\d.]+)', action)
    if not m:
        return action
    size = f"{float(m.group(2)):g}".replace('.', 'p')
    return f"{m.group(1)}{size}"


def slug_path(path):
    return '-'.join(slug_action(a) for a in path) if path else 'root'


def iter_decision_nodes(node, path=()):
    """Every node in the tree that holds a strategy, with the action path to reach it.

    One solve contains a decision node for both players at every point in the
    betting line, not just the opening one, so a single json is many heatmaps.

    Two kinds of edge lead out of a node: player actions under 'childrens', and --
    at a chance node, if the solve dumped more than one round -- the card that
    falls, under 'dealcards'. Following only 'childrens' stops dead at the flop.

    'dealcards' lists all 52 cards, including the three already on the flop, which
    cannot be dealt again. Those come through as action nodes holding zero combos, so
    require a non-empty strategy rather than merely a present one -- otherwise every
    board renders a fistful of blank grids for turn cards that are physically impossible.
    """
    strategy = node.get('strategy') or {}
    if strategy.get('strategy'):
        yield path, node
    for name, child in (node.get('childrens') or {}).items():
        yield from iter_decision_nodes(child, path + (name,))
    for card, child in (node.get('dealcards') or {}).items():
        if child:
            yield from iter_decision_nodes(child, path + (f'@{card}',))


def generate_heatmap(json_path, output_image, path=(), range_file=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    node = walk_to(data, list(path))
    if node is None:
        return
    if not node.get('strategy'):
        kind = node.get('node_type')
        print(f"No strategy at {describe_path(list(path))} (node_type={kind})")
        return
    weights = parse_range_file(range_file) if range_file else {}
    render_node(node, json_path, output_image, path, weights)


def render_node(node, json_path, output_image, path=(), range_weights=None):
    # hand -> how often the acting player's range holds it at all; empty = unweighted
    range_weights = range_weights or {}

    strategy_dict = node['strategy']['strategy']
    actions = node['strategy']['actions']
    player = node.get('player')
    action_colors = build_action_colors(actions)

    # The flop comes from the name; any later street was dealt to us on the way down.
    board = board_from_filename(json_path) + [s[1:] for s in path if s.startswith('@')]
    dealt = infer_board(strategy_dict)
    if board and dealt and set(board) != set(dealt):
        # The board in the name should be exactly the cards no combo can hold. If it
        # isn't, the json has been paired with the wrong filename.
        print(f"  WARNING: filename says board {board}, but combos imply {dealt}")
    board = board or dealt

    grid_size = 13
    # weight[i][j][k] = summed probability of action k over the combos present in that cell
    weights = np.zeros((grid_size, grid_size, len(actions)))
    present = np.zeros((grid_size, grid_size))
    # rw[i][j] = how often the range holds this hand at all (1.0 without a range file)
    rw = np.ones((grid_size, grid_size))
    if range_weights:
        for i in range(grid_size):
            for j in range(grid_size):
                rw[i, j] = range_weights.get(get_hand_name(i, j), 1.0)

    for key, probs in strategy_dict.items():
        cell = parse_combo(key)
        if cell is None:
            continue
        i, j = cell
        weights[i, j] += np.asarray(probs, dtype=float)
        present[i, j] += 1

    fig, ax = plt.subplots(figsize=(11, 9))

    for i in range(grid_size):
        for j in range(grid_size):
            n_total = total_combos(i, j, board)
            # Fraction of the DEALABLE combos this range actually holds, so a cell
            # whose every possible combo is in the range draws full height even when
            # the board has eaten some of its suits.
            filled = present[i, j] / n_total if n_total else 0.0

            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=EMPTY_COLOR,
                                   edgecolor='white', linewidth=2))

            if present[i, j] > 0:
                # Average across the combos actually present, then scale to `filled`
                # so a partially-blocked cell reads as partially filled.
                shares = weights[i, j] / present[i, j] * filled
                # A hand the range only plays sometimes is drawn faint, so it cannot be
                # mistaken for a full-frequency hand. Floored so it stays distinct from
                # the "not in range" background.
                alpha = max(rw[i, j], 0.18) if range_weights else 1.0
                y = i
                for k, action in enumerate(actions):
                    h = shares[k]
                    if h <= 0:
                        continue
                    ax.add_patch(Rectangle((j, y), 1, h,
                                           facecolor=action_colors[action],
                                           edgecolor='none', alpha=alpha))
                    y += h
                # redraw the cell border on top of the stack
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor='none',
                                       edgecolor='white', linewidth=2))

            ax.text(j + 0.5, i + 0.5, get_hand_name(i, j),
                    ha='center', va='center', fontsize=8, color='#0b0b0b',
                    path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)  # keep AA in the top-left
    ax.set_xticks(np.arange(grid_size) + 0.5)
    ax.set_yticks(np.arange(grid_size) + 0.5)
    ax.set_xticklabels(RANKS)
    ax.set_yticklabels(RANKS)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal')

    # Label each action with the share of the range that takes it, so an action that
    # is legal but never used (or a barely-used one) is obvious rather than invisible.
    # Weighted by range frequency: a hand the range only holds 13% of the time should
    # not sway the mix as much as one it always holds.
    mix = (weights * rw[:, :, None]).sum(axis=(0, 1)) / max((present * rw).sum(), 1)
    handles = [Rectangle((0, 0), 1, 1, facecolor=action_colors[a],
                         label=f"{a}  ({mix[k] * 100:.1f}%)")
               for k, a in enumerate(actions)]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=EMPTY_COLOR,
                             label='not in range'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=False)

    n_combos = int(present.sum())
    # Player 1 acts first on the flop, i.e. is out of position; player 0 is in position.
    seat = {1: 'OOP', 0: 'IP'}.get(player, f'player {player}')
    subtitle = f"{describe_path(list(path))}  ·  {seat} to act  ·  {n_combos} combos"
    if board:
        subtitle = f"board {' '.join(board)}  ·  {subtitle}"
    if range_weights:
        # The opacity channel carries meaning, so it has to be spelled out.
        subtitle += "\nfaded = hand played only part of the time (opacity = range frequency)"
    ax.set_title(f"{os.path.basename(json_path)}\n{subtitle}", loc='left', fontsize=11)

    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_image} ({n_combos} combos matched)")


def heatmap_tree(json_path, output_dir, flop_only=True):
    """Render the decision nodes of one solve, into its own directory.

    A dump_rounds 2 solve holds ~5000 decision nodes, nearly all of them turn nodes
    sitting under one of the ~47 cards that can fall. Rendering all of them for every
    board would be millions of images, and you only ever want the handful on the line
    you are actually in -- so batch runs stop at the flop and the turn is rendered on
    demand. Pass flop_only=False for the full tree of a single board.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    ranges = ranges_from_config(json_path)
    if not ranges:
        print(f"  no config next to {os.path.basename(json_path)}; drawing unweighted")

    stem = os.path.splitext(os.path.basename(json_path))[0]
    board_dir = os.path.join(output_dir, stem)
    os.makedirs(board_dir, exist_ok=True)

    count = 0
    for path, node in iter_decision_nodes(data):
        if flop_only and any(step.startswith('@') for step in path):
            continue
        # Number the images we actually write, so they sort in the order the hand is
        # played. Numbering by position in the whole tree would leave gaps of hundreds
        # once the turn nodes are skipped, and sort wrong.
        name = f"{count:04d}_{slug_path(path)}.png"
        render_node(node, json_path, os.path.join(board_dir, name), path,
                    ranges.get(node.get('player'), {}))
        count += 1
    print(f"{stem}: {count} decision nodes -> {board_dir}")


def batch_heatmaps(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".json"):
            heatmap_tree(os.path.join(input_dir, file), output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render 13x13 strategy heatmaps from solver output.")
    parser.add_argument('json', nargs='*',
                        help="solve(s) to render; default is every json solved so far")
    parser.add_argument('--full', action='store_true',
                        help="also render turn nodes (~5000 images for one board -- "
                             "only sensible for a single solve)")
    parser.add_argument('--out', default="resources/outputs/heatmaps")
    args = parser.parse_args()

    if args.json:
        for path in args.json:
            heatmap_tree(path, args.out, flop_only=not args.full)
    elif args.full:
        parser.error("--full needs a specific json; the whole batch would be millions of images")
    else:
        batch_heatmaps("resources/outputs/programmatic", args.out)
