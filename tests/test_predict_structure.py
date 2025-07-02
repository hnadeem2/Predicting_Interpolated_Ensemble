import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pie.predict_structure import save_boltz_input, run_boltz

# ----------------------------
# Tests for save_boltz_input
# ----------------------------

def test_save_boltz_input(tmp_path):
    class Args:
        output_dir = tmp_path

    seqs = ["MKTAYIAKQRQISFVKSHFS", "GAVLILALLVLQAVALVAVAV"]
    num_round = 2

    fasta_paths = save_boltz_input(seqs, Args(), num_round)

    # Check correct number of files written
    assert len(fasta_paths) == 2

    # Check file paths
    for i, path in enumerate(fasta_paths):
        expected_path = tmp_path / f"round_{num_round}" / "boltz" / "input" / f"struct_{i}.fa"
        assert Path(path) == expected_path
        assert os.path.exists(path)

        # Check contents
        with open(path, "r") as f:
            lines = f.read().splitlines()
            assert lines[0] == ">A|protein|"
            assert lines[1] == seqs[i]

# -------------------------
# Tests for run_boltz
# -------------------------

@patch("pie.predict_structure.subprocess.run")
@patch("os.path.exists")
@patch("pie.predict_structure.Path.glob")
def test_run_boltz_success(mock_glob, mock_exists, mock_subproc, tmp_path):
    # Setup mock input FASTA files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    fasta_files = [input_dir / f"struct_{i}.fa" for i in range(2)]
    mock_glob.side_effect = lambda pattern: fasta_files if "*.fa" in pattern else []

    # Simulate all expected PDBs exist
    def fake_exists(path):
        return str(path).endswith("_model_0.pdb") or str(path).endswith(".fa")
    mock_exists.side_effect = fake_exists

    output_dir = tmp_path / "output"
    boltz_script = "/fake/run_boltz.sh"

    expected_pdbs = [
        os.path.join(output_dir, "predictions", f"struct_{i}", f"struct_{i}_model_0.pdb")
        for i in range(2)
    ]

    result = run_boltz(
        input_dir=input_dir,
        output_dir=output_dir,
        boltz_script=boltz_script,
        accelerator='gpu',
        recycling_steps=2,
        output_format='pdb',
        diffusion_samples=1,
        preprocessing_threads=1
    )

    # Check subprocess call
    mock_subproc.assert_called_once()
    assert result == expected_pdbs

@patch("pie.predict_structure.Path.glob", return_value=[])
def test_run_boltz_no_fasta(mock_glob):
    with pytest.raises(FileNotFoundError, match="No FASTA files found"):
        run_boltz(
            input_dir="/fake/input",
            output_dir="/fake/output",
            boltz_script="/fake/run_boltz.sh"
        )

@patch("pie.predict_structure.subprocess.run")
@patch("pie.predict_structure.Path.glob")
@patch("os.path.exists", return_value=False)
def test_run_boltz_missing_outputs(mock_exists, mock_glob, mock_run, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    fasta_files = [input_dir / "struct_0.fa"]
    mock_glob.side_effect = lambda pattern: fasta_files if "*.fa" in pattern else []

    with pytest.raises(FileNotFoundError, match="Missing predicted PDB files"):
        run_boltz(
            input_dir=input_dir,
            output_dir="/fake/output",
            boltz_script="/fake/script.sh"
        )
