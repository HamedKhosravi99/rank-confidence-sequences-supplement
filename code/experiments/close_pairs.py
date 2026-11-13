"""Monitoring on close real pairs: adjacent models of each benchmark's ranking (Tables 2 and 4).

A retrospective stress test conditional on the fixed score matrix, not an operating policy and
not the all-395 error rate. All 395 models are sorted by their mean on the whole benchmark
(stable tie-break by model index). Two kinds of panel are preselected before any item order is
drawn:

  pairs     all 394 adjacent pairs of the ranking;
  windows   non-overlapping contiguous windows of 3, 5 and 8 models.

Each item order is shared by every panel of a benchmark, as on a real leaderboard.

Fixed-sample rank sets (the paper's construction, as in Table 9): Holm over the w(w-1) one-sided
paired tests, closed under transitivity. The paired z-test follows real_leaderboard._fixed_n_at
(statistic 0 when the sample variance is 0). The exact one-sided McNemar (sign) test is used on
binary benchmarks only, since on [0, 1] scores it targets a median. For a pair, Holm's first step
is at alpha/2, so a winner is claimed when |z| >= 1.96 or when the losing count reaches the
exact binomial critical value at alpha/2.

Ours: the paper's pairwise wealth (mixture bet, finite-benchmark offset) computed for all panel
pairs at once. Pairs are certified by the shortcut, which with two models certifies once the
wealth reaches 2/alpha and there coincides with e-Bonferroni (the exact test can certify earlier,
since it also eliminates the tie ordering from both wealths together). Windows use the shortcut of
rcs.certify. Ours is checked at the same looks as the fixed-sample sets, and pairs also after
every item.

Error events: a pair errs if a winner is claimed against the full-benchmark order (either
direction for a tied pair); a window errs if any false dominance is certified. Look schedules:
one look at half; 5, 10, 20, 50 and 100 equally spaced looks; and 10 or 20 looks spaced
geometrically from 1% to 100% (early-heavy) or mirrored (late-heavy). The 100-look schedule also
records each look alone (the single-look control at every look time). Windows use the 100-look
schedule and the half look only.

Usage:  python -m experiments.close_pairs --smoke
        python -m experiments.close_pairs --orders 200 --workers 3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.real_leaderboard import BENCHMARKS, load
from rcs.baselines import holm_dominance
from rcs.certify import ShortcutCertifier, transitive_closure
from rcs.wealth import DEFAULT_DELTA, MIXTURE_GRID

ALPHA = 0.05
Z_CRIT = stats.norm.isf(ALPHA / 2)  # Holm's first step for a pair: alpha/2
WINDOW_SIZES = (3, 5, 8)
TESTS = ("z", "mcnemar", "ours")


def schedules(n: int) -> dict[str, list[int]]:
    out = {"half": [int(round(0.5 * n))]}
    for k in (5, 10, 20, 50, 100):
        out[f"equal{k}"] = sorted({max(2, int(round(i * n / k))) for i in range(1, k + 1)})
    for k in (10, 20):
        geo = [0.01 ** ((k - i) / (k - 1)) for i in range(1, k + 1)]
        out[f"early{k}"] = sorted({max(2, int(round(u * n))) for u in geo})
        out[f"late{k}"] = sorted({max(2, min(n, int(round((1.01 - u) * n)))) for u in geo})
    return out


def mcnemar_crit(dmax: int, level: float) -> np.ndarray:
    """crit[d] = smallest w with P(Bin(d, 1/2) >= w) <= level (d + 1 if none), checked exactly."""
    d = np.arange(dmax + 1)
    crit = (stats.binom.isf(level, d, 0.5) + 1).astype(np.int64)
    crit[0] = 1
    ok = stats.binom.sf(crit - 1, d, 0.5) <= level
    below = stats.binom.sf(crit - 2, d, 0.5) > level
    assert np.all((ok | (crit == d + 1))[1:]) and np.all(below[1:]), "McNemar critical values"
    return crit


def batched_closure(d: np.ndarray) -> np.ndarray:
    """rcs.certify.transitive_closure applied to each (w, w) slice of a (W, w, w) array."""
    d = d.copy()
    for k in range(d.shape[1]):
        d |= d[:, :, k, None] & d[:, None, k, :]
    return d


def batched_holm(p: np.ndarray, alpha: float) -> np.ndarray:
    """rcs.baselines.holm_dominance applied to each (w, w) slice of a (W, w, w) array."""
    n_win, w, _ = p.shape
    off = ~np.eye(w, dtype=bool)
    flat = p[:, off]
    k = flat.shape[1]
    order = np.argsort(flat, axis=1, kind="stable")
    passed = np.take_along_axis(flat, order, axis=1) <= alpha / (k - np.arange(k))
    n_reject = np.where(passed.all(axis=1), k, np.argmin(passed, axis=1))
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(k), order.shape), axis=1)
    d = np.zeros((n_win, w, w), dtype=bool)
    d[:, off] = rank < n_reject[:, None]
    return d


def batched_shortcut_hits(lw: np.ndarray, alpha: float) -> np.ndarray:
    """New certifications of rcs.certify.ShortcutCertifier.update for each (w, w) slice, before
    the union with earlier ones and the closure."""
    n_win, w, _ = lw.shape
    cap = np.log(w * (w - 1) / alpha) + 1e-9
    wealth = np.exp(np.minimum(lw, cap))
    idx = np.arange(w)
    wealth[:, idx, idx] = 0.0
    through = np.minimum(wealth[:, :, :, None], wealth[:, None, :, :]).sum(axis=2)
    hit = wealth + through >= w * (w - 1) / alpha
    hit[:, idx, idx] = False
    return hit


def panels(theta: np.ndarray, total: np.ndarray):
    """Adjacent pairs and windows in the stable ranking; unordered pairs listed once."""
    order = np.argsort(-total, kind="stable")
    pairs = [(int(a), int(b)) for a, b in zip(order[:-1], order[1:])]
    windows = {w: [order[i * w:(i + 1) * w].astype(int) for i in range(len(order) // w)] for w in WINDOW_SIZES}
    unordered = {p: i for i, p in enumerate(pairs)}
    for w, wins in windows.items():
        for win in wins:
            for i in range(w):
                for j in range(i + 1, w):
                    key = (int(win[i]), int(win[j]))
                    if key not in unordered and (key[1], key[0]) not in unordered:
                        unordered[key] = len(unordered)
    return order, pairs, windows, list(unordered.keys())


class ManyPairWealth:
    """The paper's mixture wealth with the finite-benchmark offset, for many unordered pairs at
    once, in both directions. Column c of the result is pair c in its listed direction; the
    reverse direction is kept separately. Same update as rcs.wealth.PairwiseWealth."""

    def __init__(self, n_pairs: int, population: int, delta: float = DEFAULT_DELTA) -> None:
        self.grid = np.asarray(MIXTURE_GRID)[:, None, None]
        self.g = np.zeros((len(MIXTURE_GRID), 2, n_pairs))  # grid log-wealths, directions (+, -)
        self.s = np.zeros((2, n_pairs))
        self.n, self.t, self.delta = population, 0, delta

    def update(self, z: np.ndarray) -> np.ndarray:
        self.t += 1
        zz = np.stack([z, -z])
        b = np.clip(-self.s / (self.n - self.t + 1), -1.0 + self.delta, 1.0)
        y = (zz - b) / (1.0 + b)
        self.g += np.log1p(self.grid * y[None])
        self.s += zz
        top = self.g.max(axis=0)
        return top + np.log(np.exp(self.g - top).mean(axis=0))  # log of the mixture average


def one_order(args: tuple) -> dict:
    index, seed, benchmark, with_windows = args
    x_full, _ = load(benchmark)
    n, m = x_full.shape
    binary = bool(np.all((x_full == 0) | (x_full == 1)))
    total = x_full.sum(axis=0)
    if binary:
        total = np.rint(total)
    order, pairs, windows, unordered = panels(x_full.mean(axis=0), total)
    ua = np.array([p[0] for p in unordered])
    ub = np.array([p[1] for p in unordered])
    n_adj = len(pairs)  # the first n_adj unordered pairs are the adjacent pairs, (higher, lower)
    gap = total[ua[:n_adj]] - total[ub[:n_adj]]  # >= 0
    tied = gap == 0

    sch = schedules(n)
    look_list = sorted(set(t for v in sch.values() for t in v))
    look_index = {t: i for i, t in enumerate(look_list)}
    n_looks = len(look_list)
    crit = mcnemar_crit(n, ALPHA / 2) if binary else None

    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]

    # per-look claims for adjacent pairs: up = higher model claimed better, dn = lower claimed better
    up = {k: np.zeros((n_looks, n_adj), dtype=bool) for k in TESTS}
    dn = {k: np.zeros((n_looks, n_adj), dtype=bool) for k in TESTS}
    ours_item_dn = np.zeros(n_adj, dtype=bool)
    ours_item_up = np.zeros(n_adj, dtype=bool)
    thr_pair = math.log(2 / ALPHA)

    wealth = ManyPairWealth(len(unordered), n)
    s1 = np.zeros(len(unordered))
    s2 = np.zeros(len(unordered))
    wins = np.zeros(len(unordered))
    losses = np.zeros(len(unordered))
    lw_at = {}  # log-wealth (2, n_pairs) at the 100-look schedule, for the windows
    sums_at = {}
    w_looks = set(sch["equal100"]) | set(sch["half"])
    for t in range(1, n + 1):
        row = x[t - 1]
        z = row[ua] - row[ub]
        lw = wealth.update(z)
        s1 += z
        s2 += z * z
        if binary:
            wins += z > 0
            losses += z < 0
        ours_item_up |= lw[0, :n_adj] >= thr_pair
        ours_item_dn |= lw[1, :n_adj] >= thr_pair
        if t in look_index:
            i = look_index[t]
            mean = s1[:n_adj] / t
            var = np.maximum(s2[:n_adj] - t * mean**2, 0.0) / max(t - 1, 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                stat = np.where(var > 0, math.sqrt(t) * mean / np.sqrt(var), 0.0)
            up["z"][i] = stat >= Z_CRIT
            dn["z"][i] = stat <= -Z_CRIT
            if binary:
                d = (wins[:n_adj] + losses[:n_adj]).astype(int)
                up["mcnemar"][i] = (d > 0) & (wins[:n_adj] >= crit[d])
                dn["mcnemar"][i] = (d > 0) & (losses[:n_adj] >= crit[d])
            up["ours"][i] = lw[0, :n_adj] >= thr_pair
            dn["ours"][i] = lw[1, :n_adj] >= thr_pair
            if with_windows and t in w_looks:
                lw_at[t] = lw.copy()
                sums_at[t] = (s1.copy(), s2.copy(), wins.copy(), losses.copy())

    # pair errors per schedule: wrong direction ever (tied pairs: either direction)
    out = {"order": index, "seed": int(seed), "benchmark": benchmark, "pairs": {}, "single_look": {}}
    for name, looks in sch.items():
        idx = [look_index[t] for t in looks]
        res = {}
        for k in TESTS:
            if k == "mcnemar" and not binary:
                continue
            ever_up = up[k][idx].any(axis=0)
            ever_dn = dn[k][idx].any(axis=0)
            res[k] = np.flatnonzero(ever_dn | (tied & ever_up)).tolist()  # pairs that erred
        out["pairs"][name] = res
    out["pairs"]["every_item"] = {"ours": np.flatnonzero(ours_item_dn | (tied & ours_item_up)).tolist()}
    # single-look control at each look of the 100-look schedule: number of erring pairs per look
    for k in TESTS:
        if k == "mcnemar" and not binary:
            continue
        err = dn[k] | (tied[None] & up[k])
        out["single_look"][k] = {str(t): np.flatnonzero(err[look_index[t]]).tolist() for t in sch["equal100"]}

    if with_windows:
        out["windows"] = window_errors(x_full, total, windows, unordered, lw_at, sums_at, sch, binary, crit, n)
    return out


def _window_index(win: np.ndarray, key: dict) -> tuple[np.ndarray, np.ndarray]:
    """Column of each ordered pair (j, l) of a window in the unordered-pair list, and whether it
    is stored reversed (1) or as listed (0)."""
    w = len(win)
    col = np.zeros((w, w), dtype=int)
    sign = np.zeros((w, w), dtype=int)
    for i in range(w):
        for j in range(w):
            if i != j:
                a, b = int(win[i]), int(win[j])
                if (a, b) in key:
                    col[i, j], sign[i, j] = key[(a, b)], 0
                else:
                    col[i, j], sign[i, j] = key[(b, a)], 1
    return col, sign


def window_errors(x_full, total, windows, unordered, lw_at, sums_at, sch, binary, crit, n) -> dict:
    """Per window size and schedule: indices of windows with a false dominance, per test.
    Batched over the windows of one size; window_errors_reference is the per-window version."""
    key = {p: i for i, p in enumerate(unordered)}
    res = {}
    for w, wins in windows.items():
        res[str(w)] = {}
        idx = [_window_index(win, key) for win in wins]
        col = np.stack([c for c, _ in idx])
        sign = np.stack([s for _, s in idx])
        th = np.stack([total[win] for win in wins])
        false_pair = (th[:, :, None] <= th[:, None, :]) & ~np.eye(w, dtype=bool)[None]
        sg = np.where(sign == 0, 1.0, -1.0)
        diag = np.arange(w)
        for sname in ("equal100", "half"):
            ever = {k: np.zeros(len(wins), dtype=bool) for k in TESTS if binary or k != "mcnemar"}
            d_ours = np.zeros((len(wins), w, w), dtype=bool)
            for t in sch[sname]:
                lw = lw_at[t][sign, col]
                lw[:, diag, diag] = -np.inf
                d_ours = batched_closure(d_ours | batched_shortcut_hits(lw, ALPHA))
                ever["ours"] |= d_ours[:, diag, diag].any(axis=1) | (d_ours & false_pair).any(axis=(1, 2))
                s1, s2, wn, ls = sums_at[t]
                mean = sg * s1[col] / t
                var = np.maximum(s2[col] - t * mean**2, 0.0) / max(t - 1, 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    stat = np.where(var > 0, math.sqrt(t) * mean / np.sqrt(var), 0.0)
                p = stats.norm.sf(stat)
                p[:, diag, diag] = 1.0
                d = batched_closure(batched_holm(p, ALPHA))
                ever["z"] |= (d & false_pair).any(axis=(1, 2))
                if binary:
                    k_win = np.where(sign == 0, wn[col], ls[col])
                    dd = wn[col] + ls[col]
                    p = np.where(dd > 0, stats.binom.sf(k_win - 1, dd, 0.5), 1.0)
                    p[:, diag, diag] = 1.0
                    d = batched_closure(batched_holm(p, ALPHA))
                    ever["mcnemar"] |= (d & false_pair).any(axis=(1, 2))
            res[str(w)][sname] = {k: np.flatnonzero(v).tolist() for k, v in ever.items()}
    return res


def window_errors_reference(x_full, total, windows, unordered, lw_at, sums_at, sch, binary, crit, n) -> dict:
    """Per window size and schedule: indices of windows with a false dominance, per test."""
    key = {p: i for i, p in enumerate(unordered)}
    res = {}
    for w, wins in windows.items():
        res[str(w)] = {}
        for sname in ("equal100", "half"):
            looks = sch[sname]
            bad = {k: [] for k in TESTS if binary or k != "mcnemar"}
            for wi, win in enumerate(wins):
                th = total[win]
                false_pair = (th[:, None] <= th[None, :]) & ~np.eye(w, dtype=bool)
                # column and direction of each ordered pair (j, l): evidence that j beats l
                col = np.zeros((w, w), dtype=int)
                sign = np.zeros((w, w), dtype=int)
                for i in range(w):
                    for j in range(w):
                        if i == j:
                            continue
                        a, b = int(win[i]), int(win[j])
                        if (a, b) in key:
                            col[i, j], sign[i, j] = key[(a, b)], 0
                        else:
                            col[i, j], sign[i, j] = key[(b, a)], 1
                cert = ShortcutCertifier(w, ALPHA)
                ever = {k: False for k in bad}
                for t in looks:
                    lw = lw_at[t][sign, col]
                    np.fill_diagonal(lw, -np.inf)
                    d_ours = cert.update(lw)
                    ever["ours"] = ever["ours"] or cert.failed or bool((d_ours & false_pair).any())
                    s1, s2, wn, ls = sums_at[t]
                    sg = np.where(sign == 0, 1.0, -1.0)
                    mean = sg * s1[col] / t
                    var = np.maximum(s2[col] - t * mean**2, 0.0) / max(t - 1, 1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        stat = np.where(var > 0, math.sqrt(t) * mean / np.sqrt(var), 0.0)
                    p = stats.norm.sf(stat)
                    np.fill_diagonal(p, 1.0)
                    d = transitive_closure(holm_dominance(p, ALPHA))
                    ever["z"] = ever["z"] or bool((d & false_pair).any())
                    if binary:
                        k_win = np.where(sign == 0, wn[col], ls[col])
                        dd = (wn[col] + ls[col]).astype(int)
                        p = np.where(dd > 0, stats.binom.sf(k_win - 1, dd, 0.5), 1.0)
                        np.fill_diagonal(p, 1.0)
                        d = transitive_closure(holm_dominance(p, ALPHA))
                        ever["mcnemar"] = ever["mcnemar"] or bool((d & false_pair).any())
                for k in bad:
                    if ever[k]:
                        bad[k].append(wi)
            res[str(w)][sname] = bad
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--no-windows", action="store_true")
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("../results/close_pairs"))
    args = ap.parse_args()
    if args.smoke:
        args.orders, args.benchmarks = 2, ["truthfulqa", "arc"]

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    for b in args.benchmarks:
        t0 = time.time()
        seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(b)]).generate_state(args.start + args.orders)
        jobs = [(args.start + i, int(seeds[args.start + i]), b, not args.no_windows) for i in range(args.orders)]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as pool:
                runs = list(pool.map(one_order, jobs, chunksize=max(1, args.orders // (4 * args.workers))))
        else:
            runs = [one_order(j) for j in jobs]
        x_full, models = load(b)
        total = x_full.sum(axis=0)
        binary = bool(np.all((x_full == 0) | (x_full == 1)))
        if binary:
            total = np.rint(total)
        order, pairs, windows, _ = panels(x_full.mean(axis=0), total)
        meta = {"pairs": [[models[a], models[c]] for a, c in pairs],
                "gap": [float(total[a] - total[c]) for a, c in pairs],
                "identical": [bool(np.array_equal(x_full[:, a], x_full[:, c])) for a, c in pairs],
                "discordant": [float((x_full[:, a] != x_full[:, c]).sum()) for a, c in pairs],
                "windows": {str(w): len(v) for w, v in windows.items()},
                "binary": binary, "n_items": int(x_full.shape[0]), "schedules": schedules(x_full.shape[0])}
        tag = "smoke" if args.smoke else f"orders{args.start}-{args.start + args.orders - 1}"
        path = args.out / f"pairs_{b}_{tag}.json"
        path.write_text(json.dumps({"config": config, "meta": meta, "runs": runs}))
        print(f"{b}: {len(runs)} orders in {time.time() - t0:.0f}s -> {path}", flush=True)
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
