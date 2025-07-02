import pytest
from unittest.mock import patch, MagicMock
from pie.predict_sequence import run_pmpnn

@patch("pie.predict_sequence.subprocess.run")
@patch("os.path.exists")
def test_run_pmpnn(mock_exists, mock_subprocess):
    # Simulate the shell script running without error
    mock_subprocess.return_value = None

    # Pretend the expected output files exist
    mock_exists.return_value = True

    fasta, prob = run_pmpnn(
        pdb_path="/tmp/example.pdb",
        output_dir="/tmp/output",
        mpnn_path="/tmp/mpnn",
        pmpnn_script="/tmp/run_pmpnn.sh",
        seed=42,
        temp=0.5,
        batch_size=4,
    )

    # Check subprocess was called with the correct arguments
    mock_subprocess.assert_called_once_with([
        "bash",
        "/tmp/run_pmpnn.sh",
        "/tmp/example.pdb",
        "/tmp/output",
        "0.5",
        "42",
        "4",
        "/tmp/mpnn"
    ], check=True)

    # Check returned paths
    assert fasta == "/tmp/output/seqs/example.fa"
    assert prob == "/tmp/output/probs/example.npz"