import csv
from pathlib import Path


def write_round_summary(structures, args, num_round, path=None):
    output_dir = Path(args.output_dir) / f"round_{num_round}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write summary of all structures
    summary_path = output_dir / "structures.csv"
    with open(summary_path, mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "identity",
            "sequence_length",
            "structure_path",
            "parent_identities",
            "parent_weights",
            "sequence"
        ])
        for s in structures:
            parent_ids = ",".join(p.identity for p in s.parents) if s.parents else ""
            weights = ",".join(f"{w:.3f}" for w in s.parent_weights) if s.parent_weights else ""
            writer.writerow([
                s.identity,
                len(s.sequence),
                str(s.structure_path),
                parent_ids,
                weights,
                s.sequence
            ])

    # 2. If a path is given, write an explanation of it
    if path is not None:
        path_summary_path = output_dir / "path.csv"
        with open(path_summary_path, mode="w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "step",
                "identity",
                "parent_1",
                "parent_2",
                "weight_1",
                "weight_2",
                "structure_path",
                "sequence"
            ])
            for step, idx in enumerate(path):
                s = structures[idx]
                if s.parents and s.parent_weights and len(s.parents) == 2 and len(s.parent_weights) == 2:
                    p1, p2 = s.parents
                    w1, w2 = s.parent_weights
                    p1_id = p1.identity
                    p2_id = p2.identity
                    w1_fmt = f"{w1:.3f}"
                    w2_fmt = f"{w2:.3f}"
                else:
                    p1_id = p2_id = w1_fmt = w2_fmt = ""
                writer.writerow([
                    step,
                    s.identity,
                    p1_id,
                    p2_id,
                    w1_fmt,
                    w2_fmt,
                    str(s.structure_path),
                    s.sequence
                ])