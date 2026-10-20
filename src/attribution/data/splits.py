"""Class-balanced sampling and stratified 70/15/15 splits."""
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
