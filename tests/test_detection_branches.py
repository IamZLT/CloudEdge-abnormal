from scripts.bench_detection_branches import split_normal_indices, stratified_indices


def test_stratified_indices_balances_binary_labels():
    labels = [0] * 20 + [1] * 5
    chosen = stratified_indices(labels, 8, seed=42)
    selected = [labels[index] for index in chosen]
    assert len(chosen) == 8
    assert selected.count(0) == 4
    assert selected.count(1) == 4


def test_stratified_indices_full_split():
    assert stratified_indices([0, 1, 1], None, seed=42) == [0, 1, 2]


def test_train_normal_split_is_disjoint_and_normal_only():
    labels = [0] * 10 + [1] * 4
    bank, calibration = split_normal_indices(labels, maximum=8, calibration_fraction=0.25, seed=42)
    assert len(bank) == 6
    assert len(calibration) == 2
    assert set(bank).isdisjoint(calibration)
    assert all(labels[index] == 0 for index in bank + calibration)
