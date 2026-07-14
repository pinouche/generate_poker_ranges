"""Build a frequency-weighted flop subset from the 1755 strategically distinct flops.

The shipped resources/text/184_flops.txt cannot be used: it contains no unpaired
rainbow flops at all and is 40% monotone, against ~40% / ~5% in a real deck, so
~40% of the boards you actually get dealt have no representative in it.

What this produces instead:

  * every one of the 22100 dealable flops is mapped to its suit-isomorphic
    canonical form, which gives each of the 1755 its true frequency (a monotone
    flop is genuinely rare, a rainbow one common -- the 1755 are equivalence
    classes, not equiprobable outcomes);
  * flops are stratified by the features no clustering may blur -- suit pattern
    (rainbow/two-tone/monotone) and rank pattern (unpaired/paired/trips) -- with
    each stratum's share of the subset set by its real frequency. Suits cannot be
    relabelled across these classes, so a rainbow board can never be served by a
    two-tone representative;
  * inside a stratum, k-means over the rank features picks representatives that
    span the texture space, and each representative carries the summed frequency
    of the flops it stands for.

Weights therefore sum to 1.0 and the subset's texture mix matches the deck's.
"""

import os
from collections import Counter, defaultdict
from itertools import combinations, permutations

import numpy as np
from sklearn.cluster import KMeans

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
RANK_VAL = {r: i + 2 for i, r in enumerate(RANKS)}

ALL_1755 = 'resources/text/all_1755.txt'
OUT_FILE = 'resources/text/flop_subset.txt'
TARGET = 184
SEED = 0


def cards_of(flop):
    """'4c3c2c' -> [('4','c'), ('3','c'), ('2','c')]"""
    return [(flop[i], flop[i + 1]) for i in range(0, 6, 2)]


def canonical(cards):
    """Suit-isomorphic canonical form: the same key for every relabelling of suits.

    Two flops are the same strategic problem exactly when one can be turned into
    the other by renaming suits, so take the smallest key over all 24 renamings.
    """
    best = None
    for perm in permutations(SUITS):
        mapping = dict(zip(SUITS, perm))
        relabelled = sorted(((r, mapping[s]) for r, s in cards),
                            key=lambda c: (-RANK_VAL[c[0]], c[1]))
        key = ''.join(r + s for r, s in relabelled)
        if best is None or key < best:
            best = key
    return best


def suit_pattern(cards):
    return {1: 'monotone', 2: 'two-tone', 3: 'rainbow'}[len({s for _, s in cards})]


def rank_pattern(cards):
    counts = sorted(Counter(r for r, _ in cards).values(), reverse=True)
    return {3: 'trips', 2: 'paired', 1: 'unpaired'}[counts[0]]


# The top card decides who holds the range advantage, so it is the feature a
# representative can least afford to get wrong -- being handed king-high strategy on
# an ace-high board is a much worse error than being off by a rank on the low cards.
# Left at parity with the other features, a real board has an 11.6% chance of drawing
# a representative with a different high card; at 10.0 that is 1.7%, and the whole cost
# is that the mean total rank error creeps from 1.34 to 1.47 across the three cards.
HIGH_CARD_WEIGHT = 10.0


def features(cards):
    """Rank texture only -- suit and rank patterns are handled by stratifying.

    Connectedness is what the gaps carry: 987 is a very different board from 962
    even though both are nine-high.
    """
    vals = sorted((RANK_VAL[r] for r, _ in cards), reverse=True)
    hi, mid, lo = vals
    return [HIGH_CARD_WEIGHT * hi / 14.0, mid / 14.0, lo / 14.0,
            (hi - mid) / 12.0, (mid - lo) / 12.0]


def true_frequencies():
    """Map all 22100 dealable flops onto canonical forms and count each one."""
    deck = [r + s for r in RANKS for s in SUITS]
    freq = Counter()
    for combo in combinations(deck, 3):
        freq[canonical([(c[0], c[1]) for c in combo])] += 1
    return freq


def main():
    flops = [line.strip() for line in open(ALL_1755) if line.strip()]
    freq = true_frequencies()

    # Every canonical class must be one of the 1755 and vice versa; if that fails,
    # the isomorphism logic is wrong and every weight below it is meaningless.
    keyed = {canonical(cards_of(f)): f for f in flops}
    assert len(keyed) == len(flops), 'all_1755 contains isomorphic duplicates'
    assert set(keyed) == set(freq), 'canonical classes do not match all_1755'
    total = sum(freq.values())
    assert total == 22100, total

    # Group by the classes a representative may never cross.
    strata = defaultdict(list)
    for key, flop in keyed.items():
        cards = cards_of(flop)
        strata[(suit_pattern(cards), rank_pattern(cards))].append((flop, freq[key]))

    # Hand out the subset's slots in proportion to how often each stratum really
    # occurs, so the texture mix is right by construction rather than by luck.
    shares = {k: sum(w for _, w in v) / total for k, v in strata.items()}
    alloc = {k: max(1, round(TARGET * s)) for k, s in shares.items()}

    rows = []
    mismatched = 0.0
    rank_error = []
    for stratum, members in sorted(strata.items()):
        k = min(alloc[stratum], len(members))
        X = np.array([features(cards_of(f)) for f, _ in members])
        w = np.array([weight for _, weight in members], dtype=float)

        # Every flop within one stratum has the same multiplicity (all rainbow
        # unpaired flops stand for 24 real flops, all monotone ones for 4), so a
        # sample_weight here would be a no-op -- except that it perturbs the
        # k-means++ draw and lands on a different local optimum. Frequency is
        # already carried by `alloc` above; leave the clustering unweighted.
        if k == len(members):
            labels = np.arange(len(members))
        else:
            labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)

        for c in range(k):
            idx = np.flatnonzero(labels == c)
            if idx.size == 0:
                continue
            centre = X[idx].mean(axis=0)
            pick = idx[np.argmin(((X[idx] - centre) ** 2).sum(axis=1))]
            rep = members[pick][0]
            rows.append((float(w[idx].sum() / total), rep, stratum))

            # A subset is an approximation, so measure it: how often does a real board
            # get handed a representative with the wrong high card, and how far off are
            # the ranks? Weighted by real frequency -- distorting a common board matters
            # more than distorting a rare one.
            rep_v = sorted((RANK_VAL[r] for r, _ in cards_of(rep)), reverse=True)
            for i in idx:
                flop_v = sorted((RANK_VAL[r] for r, _ in cards_of(members[i][0])),
                                reverse=True)
                p = w[i] / total
                if flop_v[0] != rep_v[0]:
                    mismatched += p
                rank_error.append((p, sum(abs(a - b) for a, b in zip(flop_v, rep_v))))

    rows.sort(key=lambda r: -r[0])
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        for weight, flop, _ in rows:
            board = ','.join(flop[i:i + 2] for i in range(0, 6, 2))
            f.write(f"{weight:.6f},{board}\n")

    mean_err = sum(p * e for p, e in rank_error) / sum(p for p, _ in rank_error)
    print(f"wrote {len(rows)} flops to {OUT_FILE}  (weights sum to {sum(r[0] for r in rows):.6f})")
    print("\nhow lossy the subset is, weighted by how often each board really comes:")
    print(f"  chance your board's representative has a different high card: {mismatched * 100:.1f}%")
    print(f"  mean total rank error across the three cards:                 {mean_err:.2f}")
    print(f"\n{'stratum':28s} {'subset':>7s} {'subset wt':>10s} {'true wt':>9s}")
    got = Counter()
    for weight, _, stratum in rows:
        got[stratum] += weight
    for stratum in sorted(strata):
        n = sum(1 for r in rows if r[2] == stratum)
        print(f"  {stratum[0]+'/'+stratum[1]:26s} {n:7d} {got[stratum]:9.3f} {shares[stratum]:9.3f}")


if __name__ == '__main__':
    main()
