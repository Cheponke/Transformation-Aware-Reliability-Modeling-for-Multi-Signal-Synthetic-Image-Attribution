"""Class-balanced sampling and stratified 70/15/15 splits."""
import csv
import os
import random
from collections import defaultdict

from attribution.data.datasets import Record


def sample_per_class(records: list[Record], n: int, seed: int, balance_sources: bool = True) -> list[Record]:
    """Draw up to n records per class.

    With balance_sources, each class's n is split evenly across the source
    datasets; a dataset with too few images gives all it has and the shortfall
    is shared among the larger ones. Otherwise the draw is from the pooled class.
    """
    rng = random.Random(seed)
    by_label = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_label[r.label][r.source if balance_sources else ""].append(r)
    out = []
    for label in sorted(by_label):
        groups = sorted(by_label[label].items(), key=lambda kv: (len(kv[1]), kv[0]))
        remaining = n
        for i, (_, items) in enumerate(groups):
            quota = remaining // (len(groups) - i)
            take = min(quota, len(items))
            out.extend(rng.sample(items, take))
            remaining -= take
    return out


def stratified_split(
    records: list[Record], ratios: dict[str, float], seed: int
) -> dict[str, list[Record]]:
    by_label = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
    rng = random.Random(seed)
    splits = {name: [] for name in ratios}
    names = list(ratios)
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        start = 0
        for i, name in enumerate(names):
            end = len(items) if i == len(names) - 1 else start + round(ratios[name] * len(items))
            splits[name].extend(items[start:end])
            start = end
    return splits


def k_folds(records: list[Record], k: int, seed: int) -> list[list[Record]]:
    """Stratified k-fold partition; returns k disjoint folds."""
    by_label = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        for i, r in enumerate(items):
            folds[i % k].append(r)
    return folds


def deduplicate(records: list[Record], hashes: list[str]) -> tuple[list[Record], int, int]:
    """Keep one copy of each byte-identical image so no image can land in two splits.

    Copies that disagree on the label are dropped entirely. Returns the kept
    records, the number of records removed, and how many of the duplicated
    images had conflicting labels.
    """
    by_hash: dict[str, list[Record]] = defaultdict(list)
    for r, h in zip(records, hashes):
        by_hash[h].append(r)
    kept, removed, conflicts = [], 0, 0
    for copies in by_hash.values():
        if len({r.label for r in copies}) > 1:
            conflicts += 1
            removed += len(copies)
            continue
        kept.append(copies[0])
        removed += len(copies) - 1
    return kept, removed, conflicts


def read_split(path) -> list[Record]:
    with open(path) as f:
        return [Record(r["path"], int(r["label"]), r["source"]) for r in csv.DictReader(f)]


def leaking_partners(pairs: list[dict], keep_order: tuple[str, ...] = ("test", "val", "train")) -> dict[str, set[str]]:
    """For each cross-split near-duplicate pair, the copy to drop: the one in the lower-priority split.

    With the default order the test copy is always kept (so the test set is
    unchanged), a val copy is dropped only when its partner is in test, and a
    train copy is always dropped. Returns {split: paths to drop}.
    """
    rank = {name: i for i, name in enumerate(keep_order)}
    drop: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        a, b = pair["split_a"], pair["split_b"]
        if a == b:
            continue
        loser = "a" if rank[a] > rank[b] else "b"
        drop[pair[f"split_{loser}"]].add(os.path.normpath(pair[f"path_{loser}"]))
    return dict(drop)


def _groups_by_majority_label(records: list[Record], groups: list[int]) -> dict[int, dict]:
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    out = {}
    for g, idx in members.items():
        n_ai = sum(records[i].label for i in idx)
        out[g] = {"idx": idx, "label": int(n_ai * 2 > len(idx)), "size": len(idx)}
    return out


def grouped_k_folds(records: list[Record], groups: list[int], k: int, seed: int) -> list[int]:
    """Fold number (0..k-1) for every record: whole groups go to one fold, classes stay balanced.

    Groups (e.g. from near_duplicate_groups) are placed largest first, each into
    the fold that currently has the fewest images of the group's majority label,
    so near-duplicates never end up on both sides of a train/test boundary.
    """
    if len(records) != len(groups):
        raise ValueError("records and groups must have the same length")
    if k < 2:
        raise ValueError("k must be at least 2")
    info = _groups_by_majority_label(records, groups)
    order = list(info)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -info[g]["size"])  # stable: random among equal sizes
    counts = [[0, 0] for _ in range(k)]
    fold_of = [0] * len(records)
    for g in order:
        lab = info[g]["label"]
        f = min(range(k), key=lambda j: (counts[j][lab], sum(counts[j]), j))
        for i in info[g]["idx"]:
            fold_of[i] = f
            counts[f][records[i].label] += 1
    return fold_of


def grouped_holdout(records: list[Record], groups: list[int], fraction: float, seed: int) -> tuple[list[Record], list[Record]]:
    """Split records into (rest, holdout) with about `fraction` of each class held out, never splitting a group."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    info = _groups_by_majority_label(records, groups)
    totals = [sum(r.label == lab for r in records) for lab in (0, 1)]
    target = [round(fraction * t) for t in totals]
    order = list(info)
    random.Random(seed).shuffle(order)
    held = [0, 0]
    rest, holdout = [], []
    for g in order:
        lab = info[g]["label"]
        take = held[lab] + info[g]["size"] <= target[lab]
        for i in info[g]["idx"]:
            (holdout if take else rest).append(records[i])
        if take:
            held[lab] += info[g]["size"]
    return rest, holdout
