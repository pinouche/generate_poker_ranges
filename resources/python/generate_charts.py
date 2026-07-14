import subprocess
import os
import json
import glob
import time

def load_flops(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: Flop list {file_path} not found. Using default 10 flops.")
        return [
            ("1.0", "As,Kd,7h"), ("1.0", "Qs,9d,4h"), ("1.0", "Js,Ts,9h"), # High / Connected
            ("1.0", "8s,7h,2d"), ("1.0", "6s,4d,2h"),           # Low
            ("1.0", "Ks,7s,2s"), ("1.0", "9s,8s,4s"),           # Monotone
            ("1.0", "Ts,Tc,2d"), ("1.0", "5s,5d,5h")            # Paired / Trips
        ]
    flops = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                parts = line.split(",", 1)
                weight = parts[0].strip()
                board = parts[1].strip()
                flops.append((weight, board))
            else:
                flops.append(("1.0", line))
    return flops

def create_solver_input(pot, stack, board, range_ip, range_oop, output_file, script_name,
                        use_isomorphism=1, dump_rounds=2):
    # Betting abstraction is deliberately narrow. At SPR ~17 the node count (and
    # therefore memory) grows multiplicatively across streets, and the river tree is
    # replicated across every runout, so each extra size is very expensive. Two sizes
    # per street with a single raise size keeps a solve inside RAM.
    #
    # dump_rounds is what we can afford to *keep*, not what gets solved: the solver
    # always solves to the river, but writing the river back out is ~22GB per board
    # (measured), vs ~120MB for flop+turn. So we keep two streets and re-derive the
    # river when we need it.
    #
    # use_isomorphism is a ~1.5x speedup that segfaults on some board textures
    # (As,Ad,Ah is the first flop in the 184 list and dies on it), hence the retry
    # in process_clusters.
    formatted_board = ",".join([board[i:i+2] for i in range(0, len(board), 2)]) if "," not in board else board

    script_content = f"""set_pot {pot}
set_effective_stack {stack}
set_board {formatted_board}
set_range_ip {range_ip}
set_range_oop {range_oop}
set_bet_sizes oop,flop,bet,33
set_bet_sizes oop,flop,raise,60
set_bet_sizes ip,flop,bet,33,75
set_bet_sizes ip,flop,raise,60
set_bet_sizes oop,turn,bet,75
set_bet_sizes oop,turn,raise,60
set_bet_sizes ip,turn,bet,75
set_bet_sizes ip,turn,raise,60
set_bet_sizes oop,river,bet,75
set_bet_sizes oop,river,raise,60
set_bet_sizes ip,river,bet,75
set_bet_sizes ip,river,raise,60
set_allin_threshold 0.67
build_tree
set_thread_num 16
set_accuracy 0.5
set_max_iteration 150
set_print_interval 10
set_use_isomorphism {use_isomorphism}
start_solve
set_dump_rounds {dump_rounds}
dump_result {output_file}
"""
    with open(script_name, 'w') as f:
        f.write(script_content)

def run_solver(script_name, output_file):
    # Log to a file in the project root for easier monitoring
    log_file = "solver_log.txt"
    with open(log_file, "a") as log:
        log.write(f"\n--- Starting Solve: {script_name} ---\n")
        # The solver writes straight to the fd, so flush our own buffer first or the
        # markers land after the output they are meant to bracket.
        log.flush()

        start = time.time()
        proc = subprocess.run(['./console_solver', '--input_file', script_name],
                              stdout=log, stderr=log)
        elapsed = time.time() - start

        # The solver can exit 0 having produced nothing (some board textures abort
        # during the dump), so the output file is the only trustworthy success signal.
        ok = proc.returncode == 0 and os.path.exists(output_file)
        status = "OK" if ok else f"FAILED (rc={proc.returncode}, output written={os.path.exists(output_file)})"
        log.write(f"--- Finished in {elapsed:.1f}s: {status} ---\n")
        log.flush()

    return elapsed, ok

def load_range(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


def write_scenario_note(scenario_dir, scenario, range_base):
    """Drop a scenario.txt in the folder saying which preflop action these solves are.

    The solver config records only the pot, the stack and two anonymous range strings --
    nothing about how the hand got here. Six months from now the folder name alone will
    not tell you whether SB_vs_BB has the SB in or out of position, so write it down
    next to the results rather than leaving it to be reconstructed from the pot size.
    """
    # The range file lives under the folder of the seat that holds it, e.g. BB/... is
    # the BB's range, so the seat is just the first path component -- no need to guess
    # it from the scenario name, which does not say who ended up out of position.
    oop_seat = scenario["oop"].split("/")[0]
    ip_seat = scenario["ip"].split("/")[0]
    note = f"""{scenario['name']}

Preflop:  {scenario['preflop']}
Flop pot: {scenario['pot']}bb
Stacks:   {scenario['stack']}bb behind (SPR {scenario['stack'] / scenario['pot']:.1f})

OOP (acts first on the flop): {oop_seat}
  range: {scenario['oop']}
IP  (acts last on the flop):  {ip_seat}
  range: {scenario['ip']}

Ranges are from {range_base}.
Each {{board}}.json here is one solve; the {{board}}.txt beside it is the config that produced it.
"""
    with open(os.path.join(scenario_dir, "scenario.txt"), 'w') as f:
        f.write(note)

def process_clusters(only=None):
    range_base = "ranges/qb_ranges/100bb 2.5x 500rake"
    output_base = "resources/outputs/programmatic"
    # NOT resources/text/184_flops.txt -- that file has no unpaired rainbow flops at
    # all and is 40% monotone, so ~40% of real boards have no representative in it.
    # flop_subset.txt is rebuilt from all_1755.txt by build_flop_subset.py, stratified
    # so its texture mix matches the deck's.
    flop_list_path = "resources/text/flop_subset.txt"

    os.makedirs(output_base, exist_ok=True)

    flops = load_flops(flop_list_path)
    print(f"Loaded {len(flops)} flops for simulation.")

    # Single-raised pots where the BB defends by calling.
    #
    # `ip` is whoever acts LAST after the flop, which is not always the preflop raiser:
    # postflop action opens with the first player left of the button, so against a BB
    # caller the BTN and CO have position -- but the SB does not. The SB is the one
    # raiser here who plays the whole hand out of position, with the BB behind them.
    #
    # Pot and stack are per-scenario because they differ. A 2.5bb open called by the BB
    # leaves 2.5 + 2.5 + the SB's dead 0.5 = 5.5bb in the middle and 97.5bb behind;
    # an SB open has no dead money and is 3 + 3 = 6bb with 97bb behind. Pot and stack
    # fix the SPR, and every bet size in the tree is a fraction of the pot, so getting
    # these wrong tilts the entire solve.
    scenarios = [
        {"name": "BTN_vs_BB",
         "preflop": "folds to BTN, BTN opens 2.5bb, SB folds, BB calls",
         "ip":  "BTN/BTN_2.5bb.txt",
         "oop": "BB/BTN_2.5bb_BB_Call.txt",
         "pot": 5.5, "stack": 97.5},
        {"name": "CO_vs_BB",
         "preflop": "folds to CO, CO opens 2.5bb, BTN and SB fold, BB calls",
         "ip":  "CO/CO_2.5bb.txt",
         "oop": "BB/CO_2.5bb_BB_Call.txt",
         "pot": 5.5, "stack": 97.5},
        {"name": "SB_vs_BB",
         "preflop": "folds to SB, SB opens 3.0bb, BB calls",
         "ip":  "BB/SB_3.0bb_BB_Call.txt",   # the BB has position on the SB
         "oop": "SB/SB_3.0bb.txt",
         "pot": 6.0, "stack": 97.0},
    ]

    if only:
        scenarios = [s for s in scenarios if s["name"] in only]
        if not scenarios:
            raise SystemExit(f"no scenario matching {only}")

    # Each scenario gets its own directory, so a solve is never ambiguous about the
    # preflop action it came from: resources/outputs/programmatic/BTN_vs_BB/9h_6d_4c.json.
    def out_path(s_name, board):
        return os.path.join(output_base, s_name, f"{board.replace(',', '_')}.json")

    solve_times = []
    failures = []
    retried = []
    # Boards that already have a result are skipped below, so they must not count
    # towards the ETA -- on a resumed run that would inflate it by the work we did last time.
    total_solves = sum(1 for s in scenarios for _, board in flops
                       if not os.path.exists(out_path(s["name"], board)))
    print(f"Scenarios: {', '.join(s['name'] for s in scenarios)}  ({total_solves} solves to run)")

    for scenario in scenarios:
        s_name = scenario["name"]
        print(f"\nProcessing Scenario: {s_name}  ({scenario['preflop']})")

        ip_path = os.path.join(range_base, scenario["ip"])
        oop_path = os.path.join(range_base, scenario["oop"])

        if not os.path.exists(ip_path) or not os.path.exists(oop_path):
            print(f"  Missing range files for {s_name}, skipping.")
            continue

        range_ip = load_range(ip_path)
        range_oop = load_range(oop_path)

        scenario_dir = os.path.join(output_base, s_name)
        os.makedirs(scenario_dir, exist_ok=True)
        write_scenario_note(scenario_dir, scenario, range_base)

        for i, (weight, board) in enumerate(flops):
            board_slug = board.replace(",", "_")
            out_file = out_path(s_name, board)
            script_file = os.path.join(scenario_dir, f"{board_slug}.txt")

            # Skip if already solved to allow resuming
            if os.path.exists(out_file):
                continue

            print(f"  [{i+1}/{len(flops)}] Solving {board} (weight {weight})...", flush=True)

            pot = scenario["pot"]
            stack = scenario["stack"]

            # Isomorphism is worth ~1.5x, so try it first; but it segfaults outright on
            # some textures, and a board we cannot solve is a hole in the lookup table
            # exactly when we are sitting at one. Losing the speedup beats losing the board.
            create_solver_input(pot, stack, board, range_ip, range_oop, out_file, script_file)
            elapsed, ok = run_solver(script_file, out_file)

            if not ok:
                print(f"      crashed after {elapsed:.0f}s; retrying without isomorphism",
                      flush=True)
                create_solver_input(pot, stack, board, range_ip, range_oop, out_file,
                                    script_file, use_isomorphism=0)
                retry_elapsed, ok = run_solver(script_file, out_file)
                elapsed += retry_elapsed
                if ok:
                    retried.append((s_name, board))

            if not ok:
                failures.append((s_name, board))
                print(f"      FAILED after {elapsed:.0f}s - no result written", flush=True)
                continue

            solve_times.append(elapsed)
            mean = sum(solve_times) / len(solve_times)
            remaining = total_solves - len(solve_times) - len(failures)
            print(f"      done in {elapsed:.0f}s (avg {mean:.0f}s, "
                  f"~{remaining * mean / 3600:.1f}h left for {remaining} solves)", flush=True)

    print(f"\nAll clusters processed. Results are in {output_base}")
    if retried:
        print(f"\n{len(retried)} solve(s) needed the no-isomorphism retry (still complete):")
        for s_name, board in retried:
            print(f"  {s_name}  {board}")
    if failures:
        print(f"\n{len(failures)} solve(s) FAILED and produced no output:")
        for s_name, board in failures:
            print(f"  {s_name}  {board}")

if __name__ == "__main__":
    # e.g. `python3 resources/python/generate_charts.py BTN_vs_BB` to run one scenario;
    # no argument runs all of them. Already-solved boards are skipped, so adding a
    # scenario later never re-solves the ones already on disk.
    import sys
    process_clusters(only=sys.argv[1:] or None)
