"""Document-level train/test splits. Indices refer to positions in the corpus list."""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from trustedtext.bench.data import AI, HUMAN


@dataclass(frozen=True)
class Split:
    name: str
    train: np.ndarray
    test: np.ndarray


def stratified_kfold(labels: Sequence[int], n_splits: int, seed: int) -> List[Split]:
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        Split(f"fold{i}", train, test)
        for i, (train, test) in enumerate(skf.split(np.zeros(len(labels)), labels))
    ]


def leave_one_group_out(
    labels: Sequence[int],
    groups: Sequence[str],
    side: str,
    other_test_fraction: float,
    seed: int,
    min_group_size: int = 1,
) -> List[Split]:
    """
    Hold out every document of one group (e.g. one AI generator when
    side="ai", or one human outlet when side="human") for testing.

    The test set also needs the opposite class, so a random
    `other_test_fraction` of the opposite-class documents is held out with it.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    held_label = {"ai": AI, "human": HUMAN}[side]
    rng = np.random.default_rng(seed)

    held_side = np.flatnonzero(labels == held_label)
    other_side = np.flatnonzero(labels != held_label)
    n_other_test = max(1, round(other_test_fraction * len(other_side)))

    splits = []
    for group in sorted(set(groups[held_side])):
        in_group = held_side[groups[held_side] == group]
        if len(in_group) < min_group_size or len(in_group) == len(held_side):
            continue
        other_test = rng.choice(other_side, size=n_other_test, replace=False)
        test = np.sort(np.concatenate([in_group, other_test]))
        train = np.setdiff1d(np.arange(len(labels)), test)
        splits.append(Split(f"heldout={group}", train, test))
    return splits


def holdout(n_train: int, n_total: int) -> List[Split]:
    """Train on the first n_train documents, test on the rest."""
    return [Split("test", np.arange(n_train), np.arange(n_train, n_total))]
