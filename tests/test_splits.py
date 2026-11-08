import numpy as np

from trustedtext.bench.splits import holdout, leave_one_group_out, stratified_kfold

LABELS = np.array([0] * 12 + [1] * 10)
GROUPS = np.array(["reuters"] * 6 + ["scmp"] * 6 + ["gpt"] * 4 + ["claude"] * 4 + ["seed"] * 2)


def test_kfold_partitions_documents_without_overlap():
    splits = stratified_kfold(LABELS, n_splits=4, seed=0)
    tested = np.concatenate([s.test for s in splits])
    assert sorted(tested) == list(range(len(LABELS)))
    for s in splits:
        assert not set(s.train) & set(s.test)
        assert set(LABELS[s.test]) == {0, 1}


def test_kfold_depends_on_seed():
    a = stratified_kfold(LABELS, 4, seed=0)[0].test
    b = stratified_kfold(LABELS, 4, seed=1)[0].test
    assert not np.array_equal(a, b)


def test_leave_one_generator_out_holds_out_whole_group():
    splits = leave_one_group_out(LABELS, GROUPS, "ai", 0.25, seed=0, min_group_size=3)
    assert [s.name for s in splits] == ["heldout=claude", "heldout=gpt"]
    for s in splits:
        held = s.name.split("=")[1]
        assert held not in set(GROUPS[s.train])
        assert set(np.flatnonzero(GROUPS == held)) <= set(s.test)
        assert not set(s.train) & set(s.test)
        # Opposite class is present so AUROC is defined
        assert (LABELS[s.test] == 0).sum() == 3


def test_leave_one_outlet_out():
    splits = leave_one_group_out(LABELS, GROUPS, "human", 0.2, seed=0)
    assert [s.name for s in splits] == ["heldout=reuters", "heldout=scmp"]
    for s in splits:
        assert set(LABELS[s.test]) == {0, 1}


def test_holdout():
    (split,) = holdout(5, 8)
    assert list(split.train) == [0, 1, 2, 3, 4]
    assert list(split.test) == [5, 6, 7]
