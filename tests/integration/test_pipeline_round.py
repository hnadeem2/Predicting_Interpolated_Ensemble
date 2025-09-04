import os
import pytest
from pathlib import Path
import numpy as np

from pie.pipeline.core import load_templates, run_round_master
from pie.data_structs import GlobalTracker, Round


@pytest.mark.slow
def test_pipeline_round():
    """
    Integration test for one round of the full pipeline.
    """
    # Set up templates (PDB files in tests/data/)
    template_paths = [
        Path("tests/data/4IH4.pdb"),
        Path("tests/data/5HZG.pdb")
    ]
    chain_ids = ["A", "A"]
    ref_seq = np.loadtxt("tests/data/ref_seq.txt", dtype=str).item()
    aln_file = Path("tests/data/aln.fa")

    # Minimal args object
    class Args:
        ref_seq = ""
        output_dir = "scratch/"
        pmpnn_path = "/opt/ProteinMPNN"
        pmpnn_script = "pie/pmpnn.sh"
        boltz_script = "pie/boltz.sh"
        device = "gpu"
        min_edit_dist = 1
        msa_mode = "empty"

    args = Args()
    args.ref_seq = ref_seq

    pmpnn_kwargs = {
        "output_dir": "scratch/pmpnn",
        "mpnn_path": "/opt/ProteinMPNN",
        "pmpnn_script": "pie/pmpnn.sh",
        "seed": 42,
        "temp": 0.1,
        "batch_size": 1,
    }

    # Run template loader
    structures = load_templates(template_paths, ref_seq, chain_ids, aln_file, **pmpnn_kwargs)

    # Initialize global tracker and add initial templates as round 0
    global_tracker = GlobalTracker()
    initial_round = Round(round_num=0, direction="A", parent_1=structures[0], parent_2=structures[1])
    initial_round.generated_structures = structures
    global_tracker.rounds.append((initial_round,))

    # Run pipeline round (num_round = 1 now, since 0 was initialization)
    run_round_master(
        num_round=1,
        global_tracker=global_tracker,
        args=args
    )

    # Assertions
    assert len(global_tracker.rounds) >= 2  # initial + new round(s)

    new_rounds = global_tracker.rounds[1:]
    for rnd in new_rounds:
        for rnd_dir in rnd:
            for s in rnd_dir.generated_structures:
                assert isinstance(s.prob_dist, np.ndarray)
                assert s.prob_dist.shape[1] == 21
                assert len(s.sequence.replace("-", "")) == s.prob_dist.shape[0]