#!/usr/bin/env python3
"""Checks for the CANDID-PTX task-set comparison."""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_candidptx_tasksets as plot


HEADER = "Epoch,Dataset,Task-Train,Model,Task-Test,AUC,mAP40,mAP50,mAP50_95,DICE"


def score_row(epoch, trained, model, evaluated, **metrics):
    values = {
        "AUC": "-",
        "mAP40": "-",
        "mAP50": "-",
        "mAP50_95": "-",
        "DICE": "-",
    }
    values.update(metrics)
    return (
        f"{epoch},CANDID-PTX,{trained},{model},{evaluated},"
        f"{values['AUC']},{values['mAP40']},{values['mAP50']},{values['mAP50_95']},{values['DICE']}"
    )


def write_csv(directory: Path, lines: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "export_csvFile.csv").write_text("\n".join([HEADER, *lines]) + "\n")


class LoadRunTest(unittest.TestCase):
    def test_lockrelease_name_marks_the_trained_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_csv(
                path,
                [
                    score_row(1, "Classification_CANDIDPTX_B_Train", "Student", "Classification_CANDID-PTX", AUC="80"),
                    score_row(1, "Classification_CANDIDPTX_B_Train", "Student", "Localization_CANDID-PTX", mAP40="99", mAP50="10"),
                    score_row(1, "Localization_CANDIDPTX_A_Train", "Teacher", "Localization_CANDID-PTX", mAP50="40"),
                    score_row(1, "Localization_CANDIDPTX_A_Train", "Teacher", "Segmentation_CANDID-PTX", DICE="7"),
                ],
            )
            loaded = plot.load_run(path / "export_csvFile.csv")

        self.assertTrue(loaded["CLS"]["Student"][0].focused)
        self.assertEqual(loaded["CLS"]["Student"][0].value, 80)
        self.assertFalse(loaded["LOC"]["Student"][0].focused)
        self.assertEqual(loaded["LOC"]["Student"][0].value, 10)
        self.assertTrue(loaded["LOC"]["Teacher"][0].focused)
        self.assertFalse(loaded["SEG"]["Teacher"][0].focused)

    def test_later_duplicate_epoch_replaces_the_earlier_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_csv(
                path,
                [
                    score_row(2, "Segmentation_CANDIDPTX_B_Train", "Student", "Segmentation_CANDID-PTX", DICE="11"),
                    score_row(2, "Segmentation_CANDIDPTX_B_Train", "Student", "Segmentation_CANDID-PTX", DICE="22"),
                ],
            )
            loaded = plot.load_run(path / "export_csvFile.csv")
        self.assertEqual(loaded["SEG"]["Student"], [plot.ScorePoint(2, 22, True)])

    def test_trainer_writes_the_trained_task_under_the_dataset_header(self):
        """The header says Dataset, Task-Train. Each appended row is task, dataset."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_csv(
                path,
                [
                    "1,Classification_CANDIDPTX_B_Train,CANDID-PTX,Student,Classification_CANDID-PTX,80,-,-,-,-",
                    "1,Classification_CANDIDPTX_B_Train,CANDID-PTX,Teacher,Classification_CANDID-PTX,70,-,-,-,-",
                    "1,Classification_CANDIDPTX_B_Train,CANDID-PTX,Student,Localization_CANDID-PTX,-,-,12,-,-",
                ],
            )
            loaded = plot.load_run(path / "export_csvFile.csv")
        self.assertEqual(loaded["CLS"]["Student"], [plot.ScorePoint(1, 80, True)])
        self.assertEqual(loaded["CLS"]["Teacher"], [plot.ScorePoint(1, 70, True)])
        self.assertEqual(loaded["LOC"]["Student"], [plot.ScorePoint(1, 12, False)])

    def test_blank_task_train_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_csv(
                path,
                [score_row(1, "None", "Student", "Classification_CANDID-PTX", AUC="1")],
            )
            with self.assertRaises(ValueError):
                plot.load_run(path / "export_csvFile.csv")


class BestScoreTest(unittest.TestCase):
    def test_earlier_epoch_wins_a_tie(self):
        points = [
            plot.ScorePoint(3, 40, True),
            plot.ScorePoint(1, 40, True),
            plot.ScorePoint(2, 10, False),
        ]
        self.assertEqual(plot.best_point(points), plot.ScorePoint(1, 40, True))

    def test_unfocused_epoch_can_be_the_best(self):
        points = [
            plot.ScorePoint(1, 80, True),
            plot.ScorePoint(2, 90, False),
        ]
        self.assertEqual(plot.best_point(points).epoch, 2)
        self.assertEqual(plot.best_point(points).value, 90)


class PlotAndTableTest(unittest.TestCase):
    def test_curves_and_table_follow_the_task_set_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            output = Path(tmp) / "out"
            write_run_a(root / "a")
            write_run_d(root / "d")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rows = plot.compare(root, output)

            loaded_a = plot.load_run(root / "a" / "export_csvFile.csv")
            loaded_d = plot.load_run(root / "d" / "export_csvFile.csv")
            _figure_a, axis_a = plot.plt.subplots()
            plot._plot_model(axis_a, "Student", loaded_a["CLS"]["Student"])
            plot._plot_model(axis_a, "Teacher", loaded_a["CLS"]["Teacher"])
            _figure_d, axis_d = plot.plt.subplots()
            plot._plot_model(axis_d, "Student", loaded_d["CLS"]["Student"])
            _figure_loc, axis_loc = plot.plt.subplots()
            plot._plot_model(axis_loc, "Student", loaded_a["LOC"]["Student"])

            self.assertEqual([line.get_linestyle() for line in axis_a.lines], ["-", "-"])
            self.assertEqual([line.get_color() for line in axis_a.lines], ["#1f77b4", "#ff7f0e"])

            student_lines = {line.get_label(): line for line in axis_d.lines}
            self.assertEqual(list(student_lines["Student focused"].get_ydata()), [80.0])
            self.assertEqual(list(student_lines["Student all epochs"].get_ydata()), [80.0, 90.0])
            self.assertEqual(student_lines["Student all epochs"].get_linestyle(), "--")
            self.assertEqual([line.get_linestyle() for line in axis_loc.lines], ["--"])

            plot.plt.close("all")

            self.assertEqual([row["task_set"] for row in rows], ["a", "d"])
            by_set = {row["task_set"]: row for row in rows}
            self.assertEqual(by_set["a"]["student_auc"], "30")
            self.assertEqual(by_set["a"]["student_auc_epoch"], "2")
            self.assertEqual(by_set["a"]["teacher_auc"], "40")
            self.assertEqual(by_set["a"]["teacher_auc_epoch"], "1")
            self.assertEqual(by_set["a"]["student_map50"], "9")
            self.assertEqual(by_set["d"]["student_auc"], "90")
            self.assertEqual(by_set["d"]["student_auc_epoch"], "2")
            self.assertEqual(by_set["d"]["teacher_map50"], "15")
            self.assertEqual(by_set["d"]["student_dice"], "4")

            self.assertEqual(len(list(output.glob("*.png"))), 6)
            with (output / "best_scores.csv").open(newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written[1]["student_auc"], "90")
            self.assertIn("b", stderr.getvalue())
            self.assertIn("g", stderr.getvalue())

    def test_ylim_is_zero_to_one_hundred(self):
        recorded = {}
        original = plot.plt.subplots

        def capture(*args, **kwargs):
            figure, axis = original(*args, **kwargs)
            recorded["axis"] = axis
            return figure, axis

        plot.plt.subplots = capture
        try:
            with tempfile.TemporaryDirectory() as tmp:
                plot.plot_task(
                    "a",
                    "CLS",
                    {"Student": [plot.ScorePoint(1, 25, True)], "Teacher": []},
                    Path(tmp) / "a_classification.png",
                )
        finally:
            plot.plt.subplots = original
        self.assertEqual(recorded["axis"].get_ylim(), (0.0, 100.0))


def write_run_a(directory: Path) -> None:
    """Classification every epoch. Localization and segmentation are only scored."""
    lines = []
    student_auc = {1: "10", 2: "30"}
    teacher_auc = {1: "40", 2: "40"}
    for epoch in (1, 2):
        trained = "Classification_CANDIDPTX_B_Train"
        lines.extend(
            [
                score_row(epoch, trained, "Student", "Classification_CANDID-PTX", AUC=student_auc[epoch]),
                score_row(epoch, trained, "Teacher", "Classification_CANDID-PTX", AUC=teacher_auc[epoch]),
                score_row(epoch, trained, "Student", "Localization_CANDID-PTX", mAP50="5" if epoch == 1 else "9"),
                score_row(epoch, trained, "Teacher", "Localization_CANDID-PTX", mAP50="6"),
                score_row(epoch, trained, "Student", "Segmentation_CANDID-PTX", DICE="1"),
                score_row(epoch, trained, "Teacher", "Segmentation_CANDID-PTX", DICE="2"),
            ]
        )
    write_csv(directory, lines)


def write_run_d(directory: Path) -> None:
    """Epoch 1 trains classification. Epoch 2 trains localization."""
    lines = [
        score_row(1, "Classification_CANDIDPTX_B_Train", "Student", "Classification_CANDID-PTX", AUC="80"),
        score_row(1, "Classification_CANDIDPTX_B_Train", "Teacher", "Classification_CANDID-PTX", AUC="70"),
        score_row(1, "Classification_CANDIDPTX_B_Train", "Student", "Localization_CANDID-PTX", mAP50="3"),
        score_row(1, "Classification_CANDIDPTX_B_Train", "Teacher", "Localization_CANDID-PTX", mAP50="4"),
        score_row(1, "Classification_CANDIDPTX_B_Train", "Student", "Segmentation_CANDID-PTX", DICE="1"),
        score_row(1, "Classification_CANDIDPTX_B_Train", "Teacher", "Segmentation_CANDID-PTX", DICE="2"),
        score_row(2, "Localization_CANDIDPTX_B_Train", "Student", "Classification_CANDID-PTX", AUC="90"),
        score_row(2, "Localization_CANDIDPTX_B_Train", "Teacher", "Classification_CANDID-PTX", AUC="60"),
        score_row(2, "Localization_CANDIDPTX_B_Train", "Student", "Localization_CANDID-PTX", mAP50="12"),
        score_row(2, "Localization_CANDIDPTX_B_Train", "Teacher", "Localization_CANDID-PTX", mAP50="15"),
        score_row(2, "Localization_CANDIDPTX_B_Train", "Student", "Segmentation_CANDID-PTX", DICE="4"),
        score_row(2, "Localization_CANDIDPTX_B_Train", "Teacher", "Segmentation_CANDID-PTX", DICE="3"),
    ]
    write_csv(directory, lines)


if __name__ == "__main__":
    unittest.main()
