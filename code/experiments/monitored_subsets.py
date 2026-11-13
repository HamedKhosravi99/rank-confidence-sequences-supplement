"""Holm on the paper's real subsets under different look schedules (Table 3).

For the subsets of Section 5 E2 and Table 9 (the 20 highest-scoring models, the 8 highest-scoring,
and the spread subset of 20), this counts the item orders in which Holm makes a false dominance
claim, with the paper's paired z-test and with the exact McNemar sign test, under

  * looks every 1% of the benchmark starting at 1% (the first look), 5%, 10% or 25%;
  * one pre-planned look at 1%, 5%, 10%, 25% or 50%.

A claim "j above l" is false when the full-benchmark means do not support it (theta_j <= theta_l),
the error event of the paper. Claims about exactly tied pairs are also counted apart, so that
errors among non-tied models (theta_j < theta_l) can be read on their own. McNemar's sign test
targets a median on [0, 1] scores, so its TruthfulQA results are not valid (Table 9, footnote b).

Seeds, item orders, look grid and the Holm construction (with transitive closure) are those of
experiments/real_leaderboard.py, so orders 0-49 are the paper's orders, and their error rates must
equal the saved Table 9 run. Ours, monitored at every look with the paper's method, runs on the
first --ours-orders orders.

Usage:  python -m experiments.monitored_subsets --orders 2 --ours-orders 1 --benchmarks truthfulqa   (smoke)
        python -m experiments.monitored_subsets --orders 1000 --ours-orders 1000 --workers 6
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.real_leaderboard import (BENCHMARKS, FRACTIONS, SUBSETS, _fixed_n_at, choose,
                                          leaderboard_score, load)
from rcs import RankConfidenceSequence
from rcs.certify import transitive_closure

STARTS = (0.01, 0.05, 0.10, 0.25, 0.50)
SCHEDULES = [f"every 1% from {f}" for f in STARTS[:-1]] + [f"one look at {f}" for f in STARTS]
TESTS = {"z": "fixed_z", "mcnemar": "fixed_n"}


def look_grid(n: int, every_frac: float) -> tuple[np.ndarray, dict[int, float], dict[float, int]]:
    """The paper's looks (every 1% plus the checkpoints), and the look used for each start."""
    every = max(1, int(round(every_frac * n)))
    looks = np.unique(np.concatenate([np.arange(every, n + 1, every), [n]]))
    check = {int(round(f * n)) if f < 1 else n: f for f in FRACTIONS}
    looks = np.unique(np.concatenate([looks, list(check)]))
    at = {f: (every if f == 0.01 else int(round(f * n))) for f in STARTS}
    assert all(t in set(int(u) for u in looks) for t in at.values())
    return looks, check, at


def one_order(args: tuple) -> dict:
    index, seed, benchmark, subset, alpha, every_frac, run_ours = args
    scores, _ = load(benchmark)
    x_full = scores[:, choose(subset, leaderboard_score())]
    n, m = x_full.shape
    theta = x_full.mean(axis=0)
    off = ~np.eye(m, dtype=bool)
    false_pair = (theta[:, None] <= theta[None, :]) & off
    reversed_pair = theta[:, None] < theta[None, :]  # false and not tied
    rng = np.random.default_rng(seed)
    x = x_full[rng.permutation(n)]
    looks, check, at = look_grid(n, every_frac)
    rec = {"order": index, "seed": int(seed), "benchmark": benchmark, "subset": subset, "n_items": n,
           "n_models": m, "n_looks": len(looks), "start_items": {str(f): t for f, t in at.items()},
           "tied_pairs": int(((theta[:, None] == theta[None, :]) & off).sum() // 2)}

    fixed = _fixed_n_at(x, looks, alpha)
    for test, key in TESTS.items():
        wrong, wrong_untied = [], []
        for i, t in enumerate(looks):
            d = transitive_closure(fixed[key][i])
            if (d & false_pair).any():
                wrong.append(int(t))
            if (d & reversed_pair).any():
                wrong_untied.append(int(t))
        rec[test] = {"wrong": wrong, "wrong_untied": wrong_untied}

    if run_ours:
        method = "exact" if m <= 8 else "shortcut"  # as in the paper
        seq = RankConfidenceSequence(m, alpha, population_size=n, method=method)
        look_set = set(int(t) for t in looks)
        wrong, wrong_untied = [], []
        for t in range(n):
            seq.observe(x[t])
            if t + 1 in look_set:
                rep = seq.report()
                if rep.failed or (rep.dominance & false_pair).any():
                    wrong.append(t + 1)
                if rep.failed or (rep.dominance & reversed_pair).any():
                    wrong_untied.append(t + 1)
        rec["ours"] = {"wrong": wrong, "wrong_untied": wrong_untied}
    return rec


def schedule_events(wrong: list[int], start_items: dict[str, int]) -> dict[str, bool]:
    out = {}
    for f in STARTS:
        t0 = start_items[str(f)]
        if f < 0.50:
            out[f"every 1% from {f}"] = any(t >= t0 for t in wrong)
        out[f"one look at {f}"] = t0 in wrong
    return out


def summarize(runs: list[dict]) -> dict:
    table = {}
    keys = sorted({(r["benchmark"], r["subset"]) for r in runs},
                  key=lambda k: (SUBSETS.index(k[1]), BENCHMARKS.index(k[0])))
    for b, s in keys:
        rs = [r for r in runs if r["benchmark"] == b and r["subset"] == s]
        row = {"orders": len(rs), "tied_pairs": rs[0]["tied_pairs"], "n_items": rs[0]["n_items"],
               "start_items": rs[0]["start_items"]}
        for test in list(TESTS) + ["ours"]:
            rt = [r for r in rs if test in r]
            if not rt:
                continue
            row[test] = {"orders": len(rt)}
            for ev in ("wrong", "wrong_untied"):
                counts = dict.fromkeys(SCHEDULES, 0)
                for r in rt:
                    for k, v in schedule_events(r[test][ev], r["start_items"]).items():
                        counts[k] += int(v)
                row[test][ev] = counts
        table[f"{b}/{s}"] = row
    return table


def check_against_paper(runs: list[dict], saved: Path) -> list[str]:
    """Orders 0-49 must reproduce the saved Table 9 run (real_leaderboard_orders50.json)."""
    paper = json.loads(saved.read_text())["results"]
    problems = []
    for key, agg in paper.items():
        b, s = key.split("/")
        rs = sorted((r for r in runs if r["benchmark"] == b and r["subset"] == s and r["order"] < 50),
                    key=lambda r: r["order"])
        if len(rs) < 50:
            continue
        for test, name in TESTS.items():
            ever = float(np.mean([bool(r[test]["wrong"]) for r in rs]))
            if abs(ever - agg[name]["1.0"]["ever_wrong_so_far"]) > 1e-12:
                problems.append(f"{key} {test} ever wrong {ever} vs {agg[name]['1.0']['ever_wrong_so_far']}")
            for f in FRACTIONS:
                t = int(round(f * rs[0]["n_items"])) if f < 1 else rs[0]["n_items"]
                at = float(np.mean([t in r[test]["wrong"] for r in rs]))
                if abs(at - agg[name][str(f)]["wrong_at_this_look"]) > 1e-12:
                    problems.append(f"{key} {test} wrong at {f}: {at} vs {agg[name][str(f)]['wrong_at_this_look']}")
        if all("ours" in r for r in rs):
            ever = float(np.mean([bool(r["ours"]["wrong"]) for r in rs]))
            if abs(ever - agg["ours_ever_wrong"]) > 1e-12:
                problems.append(f"{key} ours ever wrong {ever} vs {agg['ours_ever_wrong']}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=int, default=1000)
    ap.add_argument("--ours-orders", type=int, default=0, help="run ours on the first this many orders")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--every-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--paper", type=Path, default=Path("../results/real_leaderboard/real_leaderboard_orders50.json"))
    ap.add_argument("--out", type=Path, default=Path("../results/monitored_subsets"))
    args = ap.parse_args()

    started = time.time()
    jobs = []
    for b in args.benchmarks:
        for s in args.subsets:
            seeds = np.random.SeedSequence([args.seed, BENCHMARKS.index(b), SUBSETS.index(s)]).generate_state(args.orders)
            jobs += [(i, int(seeds[i]), b, s, args.alpha, args.every_frac, i < args.ours_orders)
                     for i in range(args.orders)]
    jobs.sort(key=lambda j: (j[0], j[2], j[3]))  # round-robin, so a cutoff leaves every setting even
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"panels_orders0-{args.orders - 1}.json"
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    runs = []

    def save(final: bool = False) -> None:
        out = {"config": config, "runs": runs, "seconds": time.time() - started}
        if final:
            out["table"] = summarize(runs)
            out["paper_check"] = check_against_paper(runs, args.paper)
        path.write_text(json.dumps(out))

    with ProcessPoolExecutor(max(1, args.workers)) as pool:
        futures = [pool.submit(one_order, j) for j in jobs]
        for fut in as_completed(futures):
            runs.append(fut.result())
            if len(runs) % 500 == 0:
                print(f"{len(runs)} of {len(jobs)} orders done, {time.time() - started:.0f}s", flush=True)
                save()
    save(final=True)
    out = json.loads(path.read_text())
    problems = out["paper_check"]
    print("paper check (orders 0-49 against the saved Table 9 run):", "all equal" if not problems else problems[:10])
    for key, row in out["table"].items():
        for test in list(TESTS) + ["ours"]:
            if test not in row:
                continue
            w, u = row[test]["wrong"], row[test]["wrong_untied"]
            print(f"{key:<20}{test:<8} of {row[test]['orders']}: " + "  ".join(
                f"{k.replace('every 1% from ', 'from ').replace('one look at ', 'at ')}={w[k]}/{u[k]}" for k in SCHEDULES))
    print("wrote", path, f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
