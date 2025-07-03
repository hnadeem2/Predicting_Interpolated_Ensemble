import pytest
from pathlib import Path
import numpy as np

from pie.pipeline.core import load_templates, run_round_master


def test_pipeline_round(tmp_path):
    """
    Integration test for one round of the full pipeline.
    Requires two sample templates and working ProteinMPNN and Boltz scripts.
    """

    # Set up templates (PDB files in tests/data/)
    template_paths = [
        Path("tests/data/4IH4.pdb"),
        Path("tests/data/5HZG.pdb")
    ]
    chain_ids = ["A", "A"]
    ref_seq = np.loadtxt("tests/data/ref_seq.txt", dtype=str).item() # Applies for AtD14

    # Define a minimal args object
    class Args:
        output_dir = "/scratch/"
        pmpnn_path = "/opt/ProteinMPNN/"  # Replace with ProteinMPNN path
        pmpnn_script = "pie/pmpnn.sh"
        boltz_script = "pie/boltz.sh"
        device = "gpu"
        interpolation_steps = 2

    # Define extra arguments for PMPNN
    pmpnn_kwargs = {
        "output_dir": "scratch/",
        "mpnn_path": "/opt/ProteinMPNN/",
        "pmpnn_script": "pie/pmpnn.sh",
        "seed": 42,
        "temp": 0.1,
        "batch_size": 1,
    }

    # Run the template loader
    structures = load_templates(template_paths, ref_seq, chain_ids, **pmpnn_kwargs)

    # Run the actual pipeline round
    structures, fape_matrix, path = run_round_master(
        num_round=0,
        structures=structures,
        cached_dist_mat=None,
        args=Args
    )

    # --- Assertions ---
    # There should be 2 original + 2 interpolated structures
    assert len(structures) == 4
    for s in structures:
        assert isinstance(s.prob_dist, np.ndarray)
        assert s.prob_dist.shape[1] == 21
        assert len(s.sequence) == s.prob_dist.shape[0]
