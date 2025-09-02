import numpy as np
from Levenshtein import distance as edit_distance 
from Levenshtein import editops
from pie.data_structs import Structure


def find_crit_lambdas(template_1: Structure, template_2: Structure):
    '''Find critical values of lambda where the sequence would change sequence.

    Use formula lambda = (pij^B - pik^B) / [(pik^A - pik^B) - (pij^A - p^ij^B)] 
    for all (j,k) pairs at each position i

    Returns sorted lambda values such that lambda is in [0, 1].
    '''
    prob_1 = template_1.prob_dist
    prob_2 = template_2.prob_dist

    numerator = prob_2[:, :, None] - prob_2[:, None, :]
    diff = prob_1 - prob_2
    denominator = diff[:, :, None] - diff[:, None, :]
    l = numerator / denominator
    lambda_crit = np.sort(l[np.where(np.logical_and(l >= 0, l <= 1))])
    lambda_crit_inter = (np.asarray([0] + list(lambda_crit)[:-1]) + lambda_crit) / 2 # Use values between critical points

    return lambda_crit_inter


def compute_edit_distance(seqs, min_edit):
    """
    Given an ordered dict {sequence: value}, compute edit distance to the previous sequence.
    Returns a new dict {sequence: (lambda, edit_distance)}, 
    excluding entries with edit distance < min_edit.
    """
    new_dict = dict() # Python 3.9+ means this is ordered
    prev_seq = None
    prev_val = None

    for seq, val in seq_dict.items():
        if prev_seq is None:
            # First entry: keep with edit_distance=0
            new_dict[seq] = (val, 0)
        else:
            d = edit_distance(prev_seq, seq)
            if d >= min_edit:
                new_dict[seq] = (val, d)
        prev_seq, prev_val = seq, val

    return new_dict


def find_interpolated_sequences(lambda_crit_inter: np.ndarray, template_1: Structure, template_2: Structure, min_edit: int = 1):
    '''Use critical lambda values and probability distributions to build a nonredundant 
    set of protein sequences.

    Returns a dictionary where key is the sequence value and val is the tuple (lambda value, edit distance).
    The edit distance corresponds to the distance between two "adjacent" sequences.
    '''
    prob_1 = template_1.prob_dist
    prob_2 = template_2.prob_dist

    probs_all = (lambda_crit_inter * prob_1[:, :, None] + (1-lambda_crit_inter)*prob_2[:, :, None]).transpose(2, 0, 1)

    seqs = dict()
    alphabet = np.asarray(PMPNN_ALPHABET)
    for p, l in zip(probs_all, lambda_crit_inter):
        argmax = np.argmax(p, axis=1)
        seq = ''.join(alphabet[argmax])
        seqs[seq] = l

    pruned_seqs = compute_edit_distance(seqs, min_edit)

    return pruned_seqs


def find_anchors(num_round, global_tracker):
    '''
    Define anchors for each round. The following rules are followed:

    - Round 1: only two templates, use them as anchors.
    - Round 2: previous round only has one "direction". Repeat intermediate anchor.
    - Round n>2: define two sets of anchors.
        First set: first template provided by user + maximum edit distance step from previous round
        Second set: second template provided by user + maximum edit distance step from previous round
    '''
    round_0 = global_tracker.rounds[0][0]

    if num_round == 1:
        return [(round_0.parent_1, round_0.parent_2)]
    elif num_round == 2:
        prev_round = global_tracker.rounds[num_round-1][0]
        anchor_idx = np.argmax(prev_round.edit_distances) + 1
        common_anchor =  prev_round.generated_structures[anchor_idx]
        return [(round_0.parent_1, common_anchor), (round_0.parent_2, common_anchor)]       
    else:
        # First set of anchors
        prev_round_A = global_tracker.rounds[num_round-1][0]
        assert prev_round_A.direction == "A"
        anchor_idx = np.argmax(prev_round_A.edit_distances) + 1
        anchor_set_A = (round_0.parent_1, prev_round_A.generated_structures[anchor_idx])
        
        # Second set of anchors
        prev_round_B = global_tracker.rounds[num_round-1][1]
        assert prev_round_B.direction == "B"
        anchor_idx = np.argmax(prev_round_B.edit_distances) + 1
        anchor_set_B = (round_0.parent_2, prev_round_B.generated_structures[anchor_idx])
        return [anchor_set_A, anchor_set_B]

