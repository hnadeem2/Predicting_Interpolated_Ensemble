import os
import subprocess
from pathlib import Path
from pie.structure.predict_structure import save_boltz_input, run_boltz


def test_save_boltz_input_no_ligands(tmp_path):
    """Test save_boltz_input without ligands (old behavior)."""
    class Args:
        output_dir = tmp_path
        msa_mode = "empty"
        ligands = None
        ligands_str = None  # No ligands

    seqs = ["ACDE", "FGHI"]
    paths = save_boltz_input(seqs, Args(), num_round=1, direction="A")

    assert len(paths) == 2
    for i, path in enumerate(paths):
        path = Path(path)
        assert path.exists(), f"Expected file {path} does not exist."
        content = path.read_text()
        assert content.startswith(">A|protein|empty")
        assert seqs[i] in content
        # No ligands appended
        assert len(content.strip().splitlines()) == 2  # header + sequence


def test_save_boltz_input_with_ligands(tmp_path):
    """Test save_boltz_input with ligands appended."""
    class Args:
        output_dir = tmp_path
        msa_mode = "empty"
        ligands = Path("a/path/")
        ligands_str = ">B|ccd\nGLC\n>C|smiles\nC1=CC=CC=C1\n"

    seqs = ["ACDE", "FGHI"]
    paths = save_boltz_input(seqs, Args(), num_round=1, direction="A")

    assert len(paths) == 2
    for i, path in enumerate(paths):
        path = Path(path)
        assert path.exists()
        content = path.read_text()
        assert content.startswith(">A|protein|empty")
        assert seqs[i] in content
        # Ligand content should appear
        assert ">B|ccd" in content
        assert "GLC" in content
        assert ">C|smiles" in content
        assert "C1=CC=CC=C1" in content


def test_run_boltz(monkeypatch, tmp_path):
    # Setup mock FASTA input
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    pred_dir = output_dir / "boltz_results_input" / "predictions"
    input_dir.mkdir()
    pred_dir.mkdir(parents=True)

    # Create mock FASTA
    fasta1 = input_dir / "struct_0.fa"
    fasta1.write_text(">A\nACDE\n")

    # Create expected output PDB file
    out_pdb = pred_dir / "struct_0" / "struct_0_model_0.pdb"
    out_pdb.parent.mkdir(parents=True)
    out_pdb.write_text("MOCK PDB CONTENT")

    # Mock subprocess.run to prevent actual script execution
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)

    results = run_boltz(
        input_dir=input_dir,
        output_dir=output_dir,
        boltz_script="fake_boltz.sh",
        accelerator="cpu"
    )

    assert len(results) == 1
    assert Path(results[0]).exists()
    assert results[0].endswith("struct_0_model_0.pdb")
