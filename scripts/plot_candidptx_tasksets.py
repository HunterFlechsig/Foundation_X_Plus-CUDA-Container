#!/usr/bin/env python3
"""Plot CANDID-PTX task sets (a)–(g) and summarize their best scores.

Reads each run's export_csvFile.csv. One figure per task set and task:
student is blue, teacher is orange, the y-axis is 0–100, and the x-axis is
the epoch. A solid curve connects focused scores. A dashed curve connects the
score after every epoch. When every epoch is focused, or none are, each model
is drawn once.

The table stores the best score of each task on any epoch, for the student and
the teacher. A tie keeps the earlier epoch.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_SETS = ("a", "b", "c", "d", "e", "f", "g")
TASKS = ("CLS", "LOC", "SEG")
MODELS = ("Student", "Teacher")

TASK_LABEL = {
    "CLS": "Classification",
    "LOC": "Localization",
    "SEG": "Segmentation",
}
METRIC_COLUMN = {"CLS": "AUC", "LOC": "mAP50", "SEG": "DICE"}
METRIC_LABEL = {"CLS": "AUC", "LOC": "mAP@50", "SEG": "Dice"}
MODEL_COLOR = {"Student": "#1f77b4", "Teacher": "#ff7f0e"}

REQUIRED_COLUMNS = (
    "Epoch",
    "Dataset",
    "Task-Train",
    "Model",
    "Task-Test",
    "AUC",
    "mAP50",
    "DICE",
)

TABLE_FIELDS = (
    "task_set",
    "student_auc",
    "student_auc_epoch",
    "teacher_auc",
    "teacher_auc_epoch",
    "student_map50",
    "student_map50_epoch",
    "teacher_map50",
    "teacher_map50_epoch",
    "student_dice",
    "student_dice_epoch",
    "teacher_dice",
    "teacher_dice_epoch",
)

TABLE_METRIC = (
    ("CLS", "auc"),
    ("LOC", "map50"),
    ("SEG", "dice"),
)


@dataclass(frozen=True)
class ScorePoint:
    epoch: int
    value: float
    focused: bool


def default_root() -> Path:
    scratch = os.environ.get("SCRATCH") or f"/scratch/{os.environ.get('USER', 'user')}"
    return Path(scratch) / "FoundationX" / "candidptx_tasksets"


def task_named(text: str) -> str | None:
    """Return CLS, LOC, or SEG when the cell names that task."""
    folded = text.upper()
    if "SEGMENTATION" in folded:
        return "SEG"
    if "LOCALIZATION" in folded:
        return "LOC"
    if "CLASSIFICATION" in folded:
        return "CLS"
    return None


def parse_number(value: str) -> float | None:
    text = value.strip()
    if text in {"", "-", "None", "nan", "NaN"}:
        return None
    return float(text)


def format_score(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def collapse_epochs(points: list[ScorePoint]) -> list[ScorePoint]:
    """Keep the last row written for an epoch. A resumed run appends later rows."""
    by_epoch: dict[int, ScorePoint] = {}
    for point in points:
        by_epoch[point.epoch] = point
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def load_run(csv_path: Path) -> dict[str, dict[str, list[ScorePoint]]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")
        grouped: dict[str, dict[str, list[ScorePoint]]] = {
            task: {model: [] for model in MODELS} for task in TASKS
        }
        for row_number, row in enumerate(reader, start=2):
            dataset = (row.get("Dataset") or "").strip()
            if dataset.upper() != "CANDID-PTX":
                continue
            model = (row.get("Model") or "").strip().capitalize()
            if model not in MODELS:
                continue
            evaluated = task_named(row.get("Task-Test") or "")
            trained = task_named(row.get("Task-Train") or "")
            if evaluated is None:
                continue
            if trained is None:
                raise ValueError(
                    f"{csv_path}:{row_number} Task-Train does not name a task: "
                    f"{row.get('Task-Train')!r}"
                )
            value = parse_number(row.get(METRIC_COLUMN[evaluated]) or "")
            if value is None:
                continue
            try:
                epoch = int(float(row["Epoch"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{row_number} has no epoch") from exc
            grouped[evaluated][model].append(
                ScorePoint(epoch=epoch, value=value, focused=trained == evaluated)
            )
    return {
        task: {model: collapse_epochs(points) for model, points in models.items()}
        for task, models in grouped.items()
    }


def best_point(points: list[ScorePoint]) -> ScorePoint | None:
    """Highest score. The earlier epoch wins a tie."""
    chosen: ScorePoint | None = None
    for point in points:
        if chosen is None or point.value > chosen.value or (
            point.value == chosen.value and point.epoch < chosen.epoch
        ):
            chosen = point
    return chosen


def curve_mode(points: list[ScorePoint]) -> str:
    focused = any(point.focused for point in points)
    unfocused = any(not point.focused for point in points)
    if focused and unfocused:
        return "both"
    if focused:
        return "focused"
    return "unfocused"


def _plot_model(axis, model: str, points: list[ScorePoint]) -> None:
    if not points:
        return
    color = MODEL_COLOR[model]
    ordered = sorted(points, key=lambda point: point.epoch)
    mode = curve_mode(ordered)
    if mode == "both":
        focused = [point for point in ordered if point.focused]
        axis.plot(
            [point.epoch for point in focused],
            [point.value for point in focused],
            color=color,
            linestyle="-",
            marker="o",
            markersize=3,
            zorder=3,
            label=f"{model} focused",
        )
        axis.plot(
            [point.epoch for point in ordered],
            [point.value for point in ordered],
            color=color,
            linestyle="--",
            zorder=2,
            label=f"{model} all epochs",
        )
        return
    axis.plot(
        [point.epoch for point in ordered],
        [point.value for point in ordered],
        color=color,
        linestyle="-" if mode == "focused" else "--",
        marker="o" if mode == "focused" else None,
        markersize=3,
        label=model,
    )


def plot_task(
    task_set: str,
    task: str,
    series: dict[str, list[ScorePoint]],
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for model in MODELS:
        _plot_model(axis, model, series.get(model, []))
    axis.set_title(f"Task set ({task_set}) {TASK_LABEL[task]}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(METRIC_LABEL[task])
    axis.set_ylim(0, 100)
    axis.grid(True, alpha=0.3)
    if axis.lines:
        axis.legend(fontsize=8)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def table_rows(runs: dict[str, dict[str, dict[str, list[ScorePoint]]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task_set in TASK_SETS:
        if task_set not in runs:
            continue
        row = {field: "" for field in TABLE_FIELDS}
        row["task_set"] = task_set
        for task, metric_name in TABLE_METRIC:
            for model in MODELS:
                chosen = best_point(runs[task_set][task][model])
                prefix = f"{model.lower()}_{metric_name}"
                if chosen is None:
                    continue
                row[prefix] = format_score(chosen.value)
                row[f"{prefix}_epoch"] = str(chosen.epoch)
        rows.append(row)
    return rows


def write_table(rows: list[dict[str, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def warn_outside_scale(task_set: str, runs_for_set: dict[str, dict[str, list[ScorePoint]]]) -> None:
    for task, models in runs_for_set.items():
        for model, points in models.items():
            for point in points:
                if point.value < 0 or point.value > 100:
                    print(
                        f"Warning: task set {task_set} {TASK_LABEL[task]} {model} "
                        f"epoch {point.epoch} is {point.value}, outside 0–100",
                        file=sys.stderr,
                    )


def compare(root: Path, output: Path) -> list[dict[str, str]]:
    found: dict[str, dict[str, dict[str, list[ScorePoint]]]] = {}
    missing: list[str] = []
    for task_set in TASK_SETS:
        csv_path = root / task_set / "export_csvFile.csv"
        if not csv_path.is_file():
            missing.append(task_set)
            continue
        found[task_set] = load_run(csv_path)
        warn_outside_scale(task_set, found[task_set])
        for task in TASKS:
            figure_path = output / f"{task_set}_{TASK_LABEL[task].lower()}.png"
            plot_task(task_set, task, found[task_set][task], figure_path)
            print(f"Wrote {figure_path}")
    if missing:
        print(
            "Missing export_csvFile.csv for task sets: " + ", ".join(missing),
            file=sys.stderr,
        )
    if not found:
        raise FileNotFoundError(f"No task-set CSVs under {root}")
    rows = table_rows(found)
    table_path = output / "best_scores.csv"
    write_table(rows, table_path)
    print(f"Wrote {table_path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root(),
        help="Directory with task-set folders a–g (default: $SCRATCH/FoundationX/candidptx_tasksets)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("candidptx_taskset_comparison"),
        help="Directory for the figures and best_scores.csv",
    )
    args = parser.parse_args(argv)
    try:
        compare(args.root, args.output)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
