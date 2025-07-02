import os
import tempfile
import subprocess
import shutil
from pathlib import Path
import pytest
from pie.sequence.predict_sequence import run_pmpnn


@pytest.fixture
def mock_pmpnn_script(tmp_path):
    """
    Creates a mock pmpnn.sh that simulates writing outputs to the expected location.
    """
    script_path = tmp_path / "mock_pmpnn.sh"
    script_path.write_text("""#!/bin/bash
mkdir -p "$2/seqs"
mkdir -p "$2/probs"
touch "$2/seqs/$(basename $1 .pdb).fa"
touch "$2/probs/$(basename $1 .pdb).npz"
""")
    os.chmod(script_path, 0o755)
    return script_path


def test_run_pmpnn_creates_expected_outputs(mock_pmpnn_script, tmp_path):
    pdb_path = tmp_path / "test_struct.pdb"
    pdb_path.write_text("HEADER    MOCK PDB\n")

    output_dir = tmp_path / "output"
    mpnn_path = tmp_path / "ProteinMPNN_dummy"

    fasta_file, prob_file = run_pmpnn(
        pdb_path=str(pdb_path),
        output_dir=str(output_dir),
        mpnn_path=str(mpnn_path),
        pmpnn_script=str(mock_pmpnn_script),
        pdb_path_chains="A"
    )

    assert os.path.exists(fasta_file), "FASTA output file was not created"
    assert os.path.exists(prob_file), "NPZ output file was not created"
    assert "test_struct.fa" in os.path.basename(fasta_file)
    assert "test_struct.npz" in os.path.basename(prob_file)


def test_run_pmpnn_raises_if_outputs_missing(tmp_path, mock_pmpnn_script):
    pdb_path = tmp_path / "bad_struct.pdb"
    pdb_path.write_text("HEADER    MOCK PDB\n")
    output_dir = tmp_path / "missing_output"
    mpnn_path = tmp_path / "ProteinMPNN_dummy"

    # Write a broken script that does NOT generate the expected outputs
    broken_script = tmp_path / "broken_pmpnn.sh"
    broken_script.write_text("#!/bin/bash\necho 'broken script'\n")
    os.chmod(broken_script, 0o755)

    with pytest.raises(FileNotFoundError, match="Expected PMPNN outputs"):
        run_pmpnn(
            pdb_path=str(pdb_path),
            output_dir=str(output_dir),
            mpnn_path=str(mpnn_path),
            pmpnn_script=str(broken_script),
            pdb_path_chains="A"
        )
