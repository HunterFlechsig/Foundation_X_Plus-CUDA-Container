#!/usr/bin/env python3
"""Report paths the CANDID-PTX test run needs that are not on disk.

Checks the launcher files, the dataset-location YAML that launcher selects,
and the images named by the split files foundation_x5_pretraining opens.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts-apptainer" / "run_apptainer_candidptx.sh"
IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.split(" #", 1)[0].strip().strip("'\"")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            node: dict = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = value
    return root


def runner_assignments(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key.replace("_", "").isalnum() or "$" in value or " " in value:
            continue
        values[key] = value.strip().strip("'\"")
    return values


def exists(path: Path) -> bool:
    return path.exists()


def resolve_image(path: Path, dicom: bool = False) -> Path | None:
    if path.exists():
        return path
    if dicom:
        return None
    base = path.with_suffix("") if path.suffix else path
    for ext in IMAGE_EXTS:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


class Report:
    def __init__(self) -> None:
        self.present = 0
        self.missing: list[str] = []
        self._seen: set[str] = set()

    def add(self, path: Path) -> bool:
        key = str(path)
        if key in self._seen:
            return path.exists()
        self._seen.add(key)
        if path.exists():
            self.present += 1
            return True
        self.missing.append(key)
        return False


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.strip() for line in handle if line.strip()]


def check_joined(report: Report, root: Path, names: list[str], dicom: bool = False) -> None:
    if not report.add(root):
        return
    for name in names:
        candidate = Path(name) if os.path.isabs(name) else root / name
        key = str(candidate)
        if key in report._seen:
            continue
        report._seen.add(key)
        if resolve_image(candidate, dicom=dicom) is None:
            report.missing.append(key)
        else:
            report.present += 1


def first_field(line: str, sep: str | None = None) -> str:
    if sep is None:
        return line.split()[0]
    return line.split(sep)[0]


def check_list(report: Report, image_root: Path, split_file: Path, dicom: bool = False, sep: str | None = None, suffix: str = "") -> None:
    if not report.add(split_file):
        report.add(image_root)
        return
    names = [first_field(line, sep) + suffix for line in read_lines(split_file)]
    check_joined(report, image_root, names, dicom=dicom)


def check_csv_column(report: Report, image_root: Path, split_file: Path) -> None:
    if not report.add(split_file):
        report.add(image_root)
        return
    names = []
    for index, line in enumerate(read_lines(split_file)):
        if index == 0:
            continue
        column = line.split(",")[0].strip().strip('"')
        if column:
            names.append(column)
    check_joined(report, image_root, names)


def check_coco(report: Report, image_root: Path, ann_file: Path) -> None:
    if not report.add(ann_file):
        report.add(image_root)
        return
    with ann_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    names = [item["file_name"] for item in payload.get("images", []) if item.get("file_name")]
    check_joined(report, image_root, names)


def check_chestxdet_cls(report: Report, image_root: Path, split: str) -> None:
    ann_file = image_root / f"ChestX_det_{split}_NAD_v2.json"
    image_dir = image_root / split
    if not report.add(ann_file):
        report.add(image_dir)
        return
    with ann_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    names = [item["file_name"] for item in payload.get("images", []) if item.get("file_name")]
    check_joined(report, image_dir, names)


def check_dir_nonempty(report: Report, path: Path) -> None:
    if not report.add(path):
        return
    if path.is_dir() and not any(path.iterdir()):
        report.missing.append(f"{path} (empty directory)")


def main() -> int:
    yml_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    runner_text = RUNNER.read_text(encoding="utf-8")
    assigned = runner_assignments(runner_text)
    yml_path = yml_arg or (ROOT / assigned.get("DATASET_LOCATIONS_YML", "config/dataset_locations_jliang12.yml"))
    if not yml_path.is_absolute():
        yml_path = ROOT / yml_path

    print(f"Checking paths required by {RUNNER.relative_to(ROOT)}")
    print(f"Location file: {yml_path}")

    report = Report()
    for relative in (
        "scripts-apptainer/run_apptainer_candidptx.sh",
        "cuda-apptainer.sh",
        "apptainer-cuda.sif",
        "main_Consolidated.py",
        assigned.get("CONFIGFILE", "config/DINO/DINO_4scale_swinBASE.py"),
    ):
        path = Path(relative)
        report.add(path if path.is_absolute() else ROOT / path)
    report.add(yml_path)
    if assigned.get("backbone_dir"):
        report.add(Path(assigned["backbone_dir"]))
    if assigned.get("coco_path"):
        report.add(Path(assigned["coco_path"]))

    if not yml_path.is_file():
        _finish(report)
        return 1

    cfg = parse_simple_yaml(yml_path.read_text(encoding="utf-8"))
    classification = cfg.get("classification", {})
    localization = cfg.get("localization", {})
    segmentation = cfg.get("segmentation", {})

    def cls(name: str) -> dict:
        return classification.get(name, {})

    def loc(name: str) -> dict:
        return localization.get(name, {})

    def seg(name: str) -> dict:
        return segmentation.get(name, {})

    nih = cls("nih_chestxray14")
    check_list(report, Path(nih["images"]), Path(nih["splits"]["train"]))
    check_list(report, Path(nih["images"]), Path(nih["splits"]["test"]))

    chexpert = cls("chexpert")
    check_csv_column(report, Path(chexpert["images"]), Path(chexpert["splits"]["train"]))
    check_csv_column(report, Path(chexpert["images"]), Path(chexpert["splits"]["test"]))

    mimic = cls("mimic_cxr")
    check_csv_column(report, Path(mimic["images"]), Path(mimic["splits"]["train"]))
    check_csv_column(report, Path(mimic["images"]), Path(mimic["splits"]["test"]))

    vindr = cls("vindr_cxr")
    check_list(report, Path(vindr["images"]), Path(vindr["splits"]["train"]), suffix=".jpeg")
    check_list(report, Path(vindr["images"]), Path(vindr["splits"]["test"]), suffix=".jpeg")

    shenzhen = cls("shenzhen")
    check_list(report, Path(shenzhen["images"]), Path(shenzhen["splits"]["train"]), sep=",")
    check_list(report, Path(shenzhen["images"]), Path(shenzhen["splits"]["test"]), sep=",")

    tbx = cls("tbx11k")
    check_list(report, Path(tbx["images"]), Path(tbx["splits"]["train"]))
    check_list(report, Path(tbx["images"]), Path(tbx["splits"]["test"]))

    node = cls("node21")
    check_list(report, Path(node["images"]), Path(node["splits"]["train"]))
    check_list(report, Path(node["images"]), Path(node["splits"]["test"]))

    chest = cls("chestx_det")
    check_chestxdet_cls(report, Path(chest["images"]), "train")
    check_chestxdet_cls(report, Path(chest["images"]), "test")

    rsna = cls("rsna_pneumonia")
    check_list(report, Path(rsna["images"]), Path(rsna["splits"]["train"]))
    check_list(report, Path(rsna["images"]), Path(rsna["splits"]["test"]))

    siim = cls("siim_acr_ptx")
    siim_root = Path(siim["images"])
    check_list(report, siim_root / "train_jpeg", Path(siim["splits"]["train"]), suffix=".dcm.jpeg")
    check_list(report, siim_root / "test_jpeg", Path(siim["splits"]["test"]), suffix=".dcm.jpeg")

    candid = cls("candid_ptx")
    candid_images = Path(candid["images_dicom"] if str(candid.get("use_dicom", "")).lower() == "true" else candid["images"])
    check_list(report, candid_images, Path(candid["splits"]["train"]), dicom=True)
    check_list(report, candid_images, Path(candid["splits"]["test"]), dicom=True)

    tbx_loc = loc("tbx11k")
    check_coco(report, Path(tbx_loc["images"]), Path(tbx_loc["splits"]["train_json"]))
    check_coco(report, Path(tbx_loc["images"]), Path(tbx_loc["splits"]["test_json"]))

    node_loc = loc("node21")
    check_coco(report, Path(node_loc["images"]), Path(node_loc["splits"]["train_json"]))
    check_coco(report, Path(node_loc["images"]), Path(node_loc["splits"]["test_json"]))

    chest_loc = loc("chestx_det")
    check_coco(report, Path(chest_loc["images_train"]), Path(chest_loc["splits"]["train_json"]))
    check_coco(report, Path(chest_loc["images_test"]), Path(chest_loc["splits"]["test_json"]))

    rsna_loc = loc("rsna_pneumonia")
    check_coco(report, Path(rsna_loc["images"]), Path(rsna_loc["splits"]["train_json"]))
    check_coco(report, Path(rsna_loc["images"]), Path(rsna_loc["splits"]["val_json"]))

    siim_loc = loc("siim_acr_ptx")
    check_coco(report, Path(siim_loc["images_train"]), Path(siim_loc["splits"]["train_json"]))
    check_coco(report, Path(siim_loc["images_val"]), Path(siim_loc["splits"]["val_json"]))

    candid_loc = loc("candid_ptx")
    check_coco(report, Path(candid_loc["images"]), Path(candid_loc["splits"]["train_json"]))
    check_coco(report, Path(candid_loc["images"]), Path(candid_loc["splits"]["val_json"]))

    candid_seg = seg("candid_ptx")
    candid_seg_images = Path(candid_seg["images"])
    check_list(report, candid_seg_images, Path(candid_seg["train_list"]), dicom=True, sep=",")
    check_list(report, candid_seg_images, Path(candid_seg["test_list"]), dicom=True, sep=",")

    chest_seg = seg("chestx_det")
    for key in ("images_train", "masks_train", "images_test", "masks_test"):
        check_dir_nonempty(report, Path(chest_seg[key]))

    siim_seg = seg("siim_acr_ptx")
    check_list(report, Path(siim_seg["images_train"]), Path(siim_seg["train_list"]), sep=",")
    # The mask path is the second CSV field, joined to the same image directory.
    for image_key, list_key in (("images_train", "train_list"), ("images_test", "test_list")):
        split_file = Path(siim_seg[list_key])
        image_root = Path(siim_seg[image_key])
        if split_file.is_file() and image_root.exists():
            masks = [line.split(",")[1].strip() for line in read_lines(split_file) if "," in line]
            check_joined(report, image_root, masks)

    _finish(report)
    return 1 if report.missing else 0


def _finish(report: Report) -> None:
    out_path = Path.cwd() / "missing_dataset_paths.txt"
    for path in report.missing:
        print(f"MISSING {path}")
    out_path.write_text("\n".join(report.missing) + ("\n" if report.missing else ""), encoding="utf-8")
    print(f"Present: {report.present}")
    print(f"Missing: {len(report.missing)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyError as exc:
        print(f"Location file is missing key {exc}", file=sys.stderr)
        raise SystemExit(2)
