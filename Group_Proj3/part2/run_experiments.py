"""Run the approved PH2 suite in separate processes, then build its report."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from lib.points import STRATEGIES
from lib.records import checkpoint_path, file_hash, load_record, save_run_record, validate_reference
from train import add_common_arguments, validate_arguments


def build_plan(args):
    root = Path(args.output_dir).resolve()
    reference = root / f"full_{args.model}_bce.pth.json"
    plan = [{"name": "fully_supervised", "supervision": "full", "record": str(reference),
             "source_record": args.reference_record}]
    configurations = [("uniform", value) for value in args.click_counts]
    for strategy in args.strategies:
        if (strategy, args.strategy_clicks) not in configurations:
            configurations.append((strategy, args.strategy_clicks))
    for strategy, count in configurations:
        name = f"points_{strategy}_{count:02d}p_{count:02d}n"
        plan.append({"name": name, "supervision": "points", "strategy": strategy,
                     "positive_clicks": count, "negative_clicks": count,
                     "record": str(root / f"{name}.pth.json")})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output-dir", required=True, type=Path, help="New run directory; must not exist.")
    parser.add_argument("--reference-record", help="Reuse a matching completed Part 1/P2 fully supervised BCE record.")
    parser.add_argument("--click-counts", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--strategy-clicks", type=int, default=10)
    parser.add_argument("--click-seed", type=int, default=42)
    parser.add_argument("--margin", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true", help="Print the plan without creating files or training.")
    args = parser.parse_args()
    try:
        validate_arguments(args)
        if not args.click_counts or min(args.click_counts) < 1 or args.strategy_clicks < 1:
            raise ValueError("Click budgets must be positive")
        if len(set(args.click_counts)) != len(args.click_counts):
            raise ValueError("Click-count configurations must be unique")
        if len(set(args.strategies)) != len(args.strategies) or args.click_seed < 0 or args.margin < 1:
            raise ValueError("Strategies must be unique, click seed nonnegative and margin positive")
    except ValueError as error:
        parser.error(str(error))
    args.output_dir = args.output_dir.resolve()
    if args.data_root:
        args.data_root = str(Path(args.data_root).resolve())
    if args.reference_record:
        args.reference_record = str(Path(args.reference_record).resolve())
        validate_reference(load_record(args.reference_record), {**vars(args), "dataset": "PH2", "loss": "bce"})
    plan = build_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; refusing CPU execution of the HPC suite")
    # mkdir(exist_ok=False) also rejects existing directories and occupied paths.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "config": {key: str(value) if isinstance(value, Path) else value
                                               for key, value in vars(args).items()},
                "experiments": plan, "completed": []}
    manifest_path = args.output_dir / "experiments.json"
    save_run_record(manifest_path, manifest, create=True)
    folder = Path(__file__).resolve().parent
    common = []
    for name in ("model", "size", "batch_size", "epochs", "lr", "split_seed", "seed", "num_workers", "num_threads"):
        common += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if args.data_root:
        common += ["--data-root", args.data_root]
    if args.require_cuda:
        common.append("--require-cuda")
    for experiment in plan:
        try:
            record_path = Path(experiment["record"])
            if experiment["supervision"] == "full" and args.reference_record:
                print(f"Reusing fully supervised reference {record_path}", flush=True)
                source = load_record(args.reference_record)
                weights = checkpoint_path(source, args.reference_record)
                if source.get("checkpoint_sha256", file_hash(weights)) != file_hash(weights):
                    raise ValueError("Reference checkpoint checksum differs from its record")
                with record_path.open("xb") as handle:
                    handle.write(Path(args.reference_record).read_bytes())
                with record_path.with_suffix("").open("xb") as handle:
                    handle.write(weights.read_bytes())
            else:
                command = [sys.executable, "-B", "-u", str(folder / "train.py"), *common,
                           "--supervision", experiment["supervision"], "--checkpoint", str(record_path.with_suffix(""))]
                if experiment["supervision"] == "points":
                    command += ["--positive-clicks", str(experiment["positive_clicks"]),
                                "--negative-clicks", str(experiment["negative_clicks"]),
                                "--strategy", experiment["strategy"], "--margin", str(args.margin),
                                "--click-seed", str(args.click_seed), "--reference-record", plan[0]["record"]]
                print(f"Starting {experiment['name']}", flush=True)
                subprocess.run(command, cwd=folder, check=True)
            load_record(record_path)
            experiment["record_sha256"] = file_hash(record_path)
            manifest["completed"].append(experiment["name"])
            save_run_record(manifest_path, manifest)
        except Exception:
            manifest["status"] = "failed"
            manifest["failed_experiment"] = experiment["name"]
            save_run_record(manifest_path, manifest)
            raise
    manifest["status"] = "training_completed"
    save_run_record(manifest_path, manifest)
    try:
        subprocess.run([sys.executable, "-B", "-u", str(folder / "plot_results.py"),
                        "--run-dir", str(args.output_dir)], cwd=folder, check=True)
    except Exception:
        manifest["status"] = "report_failed"
        save_run_record(manifest_path, manifest)
        raise
    manifest["status"] = "completed"
    save_run_record(manifest_path, manifest)
    print(f"Completed suite: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
