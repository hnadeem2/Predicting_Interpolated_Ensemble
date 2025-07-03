import os
import subprocess
from pathlib import Path
from pie.structure.predict_structure import save_boltz_input, run_boltz


def test_save_boltz_input(tmp_path):
    class Args:
        output_dir = tmp_path

    seqs = ["ACDE", "FGHI"]
    paths = save_boltz_input(seqs, Args(), num_round=1)

    assert len(paths) == 2
    for i, path in enumerate(paths):
        path = Path(path)
        assert path.exists(), f"Expected file {path} does not exist."
        content = path.read_text()
        assert content.startswith(">A|protein|")
        assert seqs[i] in content


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
