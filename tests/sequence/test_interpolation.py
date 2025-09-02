import numpy as np
import pytest
from pie.data_structs import Structure
from pie.interpolation import (
    find_crit_lambdas,
    compute_edit_distance,
    find_interpolated_sequences,
    find_anchors,
)

# Minimal amino acid alphabet for test
PMPNN_ALPHABET = list("ACD")  # reduce alphabet size for toy test


@pytest.fixture
def simple_structures():
    # Toy distributions for 3 residues and 3 amino acids
    prob_1 = np.array([
        [1, 0, 0],
        [0.5, 0.5, 0],
        [0, 1, 0]
    ])
    prob_2 = np.array([
        [0, 1, 0],
        [0.2, 0.3, 0.5],
        [0, 0, 1]
    ])
    s1 = Structure(identity="template_1",
                   structure_path="dummy1.pdb",
                   sequence="AAA",
                   prob_dist=np.pad(prob_1, ((0,0),(0,18))))  # pad to 21
    s2 = Structure(identity="template_2",
                   structure_path="dummy2.pdb",
                   sequence="AAA",
                   prob_dist=np.pad(prob_2, ((0,0),(0,18))))
    return s1, s2


def test_find_crit_lambdas(simple_structures):
    s1, s2 = simple_structures
    lambdas = find_crit_lambdas(s1, s2)
    assert np.all((lambdas >= 0) & (lambdas <= 1))
    assert np.all(np.diff(lambdas) >= 0)  # sorted


def test_compute_edit_distance():
    seq_dict = {
        "AAA": 0.1,
        "AAB": 0.2,
        "ABB": 0.3,
        "ABD": 0.4,
    }
    result = compute_edit_distance(seq_dict, min_edit=1)
    assert "AAA" in result
    assert all(isinstance(val, tuple) for val in result.values())
    assert result["AAA"][1] == 0
    # Check that distances are respected
    for seq, (val, d) in result.items():
        if seq != "AAA":
            assert d == 1


def test_find_interpolated_sequences(simple_structures):
    s1, s2 = simple_structures
    lambdas = np.linspace(0, 1, 5)
    seqs = find_interpolated_sequences(lambdas, s1, s2, min_edit=1)
    assert isinstance(seqs, dict)
    assert all(isinstance(v, tuple) for v in seqs.values())
    for seq, (l, d) in seqs.items():
        assert isinstance(seq, str)
        assert 0 <= l <= 1
        assert d >= 0


def test_find_anchors_round1(simple_structures):
    s1, s2 = simple_structures
    # Fake GlobalTracker with round 0
    class DummyRound:
        def __init__(self, p1, p2):
            self.parent_1 = p1
            self.parent_2 = p2

    class DummyGlobal:
        rounds = [[DummyRound(s1, s2)]]

    anchors = find_anchors(1, DummyGlobal)
    assert anchors == [(s1, s2)]


def test_find_anchors_round2(simple_structures):
    s1, s2 = simple_structures

    class DummyStruct:
        def __init__(self, identity):
            self.identity = identity

    class DummyRound:
        def __init__(self, direction):
            self.parent_1 = s1
            self.parent_2 = s2
            self.direction = direction
            self.edit_distances = [1, 3, 2]
            self.generated_structures = [
                DummyStruct("s0"),
                DummyStruct("s1"),
                DummyStruct("s2"),
            ]

    class DummyGlobal:
        rounds = [
            [DummyRound("A")],  # round 0
            [DummyRound("A")],  # round 1
        ]

    anchors = find_anchors(2, DummyGlobal)
    assert len(anchors) == 2
    assert all(len(pair) == 2 for pair in anchors)


def test_find_anchors_round3(simple_structures):
    s1, s2 = simple_structures

    class DummyStruct:
        def __init__(self, identity):
            self.identity = identity

    class DummyRound:
        def __init__(self, direction, p1, p2):
            self.parent_1 = p1
            self.parent_2 = p2
            self.direction = direction
            self.edit_distances = [1, 2]
            self.generated_structures = [
                DummyStruct(f"{direction}_s0"),
                DummyStruct(f"{direction}_s1"),
            ]

    class DummyGlobal:
        rounds = [
            [DummyRound("A", s1, s2)],            # round 0
            [DummyRound("A", s1, s2)],            # round 1
            (DummyRound("A", s1, s2), DummyRound("B", s1, s2)),  # round 2 (index = 2 → round 3)
        ]

    anchors = find_anchors(3, DummyGlobal)

    # Should return two pairs because round 3 has A and B
    assert len(anchors) == 2
    assert all(len(pair) == 2 for pair in anchors)

    # Check direction-specific anchor rule:
    # - "A" → parent_1 is anchor
    # - "B" → parent_2 is anchor
    found_A = any(pair[0] == s1 for pair in anchors)
    found_B = any(pair[0] == s2 for pair in anchors)
    assert found_A and found_B