import json
from pathlib import Path
from pie.data_structs import GlobalTracker


def write_summary(global_tracker: GlobalTracker, args):
    log_path = Path(args.output_dir) / "summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not global_tracker.rounds:
        return

    log_data = []
    for round_tuple in global_tracker.rounds:
        for round_obj in round_tuple:
            entry = {
                "__comment__": f"Round {round_obj.round_num} (Direction: {round_obj.direction})",
                "round_num": round_obj.round_num,
                "direction": round_obj.direction,
                "parents": [
                    {
                        "identity": parent.identity,
                        "sequence_length": len(parent.sequence.replace("-", "")),
                        "structure_path": str(parent.structure_path)
                    }
                    for parent in (round_obj.parent_1, round_obj.parent_2)
                ],
                "sequences": [
                    {
                        "sequence": seq,
                        "edit_distance": (
                            round_obj.edit_distances[idx]
                            if round_obj.edit_distances else None
                        ),
                        "weight": (
                            round_obj.weights[idx]
                            if round_obj.weights else None
                        )
                    }
                    for idx, seq in enumerate(round_obj.sequences or [])
                ],
                "generated_structures": [
                    {
                        "identity": struct.identity,
                        "sequence_length": len(struct.sequence.replace("-", "")),
                        "structure_path": str(struct.structure_path),
                        "parents": [
                            {"identity": p.identity, "weight": w}
                            for p, w in zip(struct.parents or [], struct.parent_weights or [])
                        ]
                    }
                    for struct in (round_obj.generated_structures or [])
                ]
            }
            log_data.append(entry)

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)



# def write_round_summary(structures, args, num_round, path=None):
#     output_dir = Path(args.output_dir) / f"round_{num_round}"
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 1. Write summary of all structures
#     summary_path = output_dir / "structures.csv"
#     with open(summary_path, mode="w", newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([
#             "identity",
#             "sequence_length",
#             "structure_path",
#             "parent_identities",
#             "parent_weights",
#             "sequence"
#         ])
#         for s in structures:
#             parent_ids = ",".join(p.identity for p in s.parents) if s.parents else ""
#             weights = ",".join(f"{w:.3f}" for w in s.parent_weights) if s.parent_weights else ""
#             writer.writerow([
#                 s.identity,
#                 len(s.sequence),
#                 str(s.structure_path),
#                 parent_ids,
#                 weights,
#                 s.sequence
#             ])

#     # 2. If a path is given, write an explanation of it
#     if path is not None:
#         path_summary_path = output_dir / "path.csv"
#         with open(path_summary_path, mode="w", newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([
#                 "step",
#                 "identity",
#                 "parent_1",
#                 "parent_2",
#                 "weight_1",
#                 "weight_2",
#                 "structure_path",
#                 "sequence"
#             ])
#             for step, idx in enumerate(path):
#                 s = structures[idx]
#                 if s.parents and s.parent_weights and len(s.parents) == 2 and len(s.parent_weights) == 2:
#                     p1, p2 = s.parents
#                     w1, w2 = s.parent_weights
#                     p1_id = p1.identity
#                     p2_id = p2.identity
#                     w1_fmt = f"{w1:.3f}"
#                     w2_fmt = f"{w2:.3f}"
#                 else:
#                     p1_id = p2_id = w1_fmt = w2_fmt = ""
#                 writer.writerow([
#                     step,
#                     s.identity,
#                     p1_id,
#                     p2_id,
#                     w1_fmt,
#                     w2_fmt,
#                     str(s.structure_path),
#                     s.sequence
#                 ])