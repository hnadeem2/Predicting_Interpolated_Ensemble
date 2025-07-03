import os
import pytest
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock

from pie.pipeline.core import load_templates, run_round_master


@pytest.fixture(autouse=True)
def mock_save_boltz_input(monkeypatch):
    """
    Automatically patch save_boltz_input in pie.pipeline.core to avoid real file I/O.
    """
    def dummy_save_boltz_input(seqs, args, num_round):
        base_dir = os.path.join(args.output_dir, f"round_{num_round}", "boltz", "input")
        os.makedirs(base_dir, exist_ok=True)
        fasta_paths = []
        for i, seq in enumerate(seqs):
            fasta_path = os.path.join(base_dir, f"struct_{i}.fa")
            with open(fasta_path, "w") as f:
                f.write(">A|protein|empty\n")
                f.write(seq + "\n")
            fasta_paths.append(fasta_path)
        return fasta_paths

    # patch in the pipeline module, since run_round_master imported it there
    monkeypatch.setattr("pie.pipeline.core.save_boltz_input", dummy_save_boltz_input)


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
        pmpnn_path = "/opt/ProteinMPNN/"
        pmpnn_script = "pie/pmpnn.sh"
        boltz_script = "pie/boltz.sh"
        device = "gpu"
        interpolation_steps = 2

    args = Args()
    args.ref_seq = ref_seq

    pmpnn_kwargs = {
        "output_dir": "scratch/pmpnn",
        "mpnn_path": "/opt/ProteinMPNN/",
        "pmpnn_script": "pie/pmpnn.sh",
        "seed": 42,
        "temp": 0.1,
        "batch_size": 1,
    }

    # Run template loader
    structures = load_templates(template_paths, ref_seq, chain_ids, aln_file, **pmpnn_kwargs)

    # Run pipeline round
    structures, fape_matrix, path = run_round_master(
        num_round=0,
        structures=structures,
        cached_dist_mat=None,
        args=args
    )

    # Assertions
    assert len(structures) == 4
    for s in structures:
        assert isinstance(s.prob_dist, np.ndarray)
        assert s.prob_dist.shape[1] == 21
        assert len(s.sequence) == s.prob_dist.shape[0]
