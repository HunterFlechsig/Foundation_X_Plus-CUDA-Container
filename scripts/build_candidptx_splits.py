#!/usr/bin/env python3
"""Build CANDID-PTX classification and segmentation lists from the localization JSON.

The train, validation, and test studies are the images already listed in
bbox_annotation_pneumothorax/CANDID_PTX_{train,valid,test}_1.json. A study is
class 1 when EncodedPixels is not -1, and class 0 when it is. The same string
is the segmentation mask. Studies in a JSON file with no reports row are
skipped and counted. Patient indexes that appear in more than one JSON file
are reported and left where they are.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path("/scratch/hflechsi/CANDID-PTX")
JSON_NAMES = {
    "train": "CANDID_PTX_train_1.json",
    "val": "CANDID_PTX_valid_1.json",
    "test": "CANDID_PTX_test_1.json",
}


def patient_index(filename):
    return filename.split("_", 1)[0]


def sop_from_file_name(file_name):
    name = Path(file_name).name
    if name.endswith(".png"):
        name = name[: -len(".png")]
    return name


def load_reports(path):
    reports = {}
    duplicates = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sop = (row.get("SOPInstanceUID") or "").strip()
            if not sop:
                continue
            if sop in reports:
                duplicates += 1
                continue
            reports[sop] = {
                "filename": (row.get("filename") or "").strip(),
                "encoded": (row.get("EncodedPixels") or "").strip(),
            }
    return reports, duplicates


def image_sops(json_path):
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return [sop_from_file_name(image["file_name"]) for image in payload["images"]]


def classification_label(encoded):
    if encoded == "-1":
        return 0
    if encoded == "":
        return None
    return 1


def build(root):
    root = Path(root)
    reports, duplicate_rows = load_reports(root / "Pneumothorax_reports.csv")
    json_dir = root / "bbox_annotation_pneumothorax"
    out_dir = root / "data_files_splits" / "candid_ptx"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_sops = {name: image_sops(json_dir / filename) for name, filename in JSON_NAMES.items()}
    patients = defaultdict(set)
    skipped = defaultdict(list)
    blank = defaultdict(list)
    written = {}

    cls_names = {
        "train": "CANDIDPTX_cls_train.txt",
        "val": "CANDIDPTX_cls_val.txt",
        "test": "CANDIDPTX_cls_test.txt",
    }
    seg_names = {"train": "train.txt", "test": "test.txt"}

    for split, sops in split_sops.items():
        cls_lines = []
        seg_lines = []
        positives = 0
        for sop in sops:
            row = reports.get(sop)
            if row is None:
                skipped[split].append(sop)
                continue
            label = classification_label(row["encoded"])
            if label is None:
                blank[split].append(sop)
                continue
            if row["filename"]:
                patients[patient_index(row["filename"])].add(split)
            cls_lines.append(f"{sop} {label}\n")
            seg_lines.append(f"{sop},{row['encoded']}\n")
            positives += label
        (out_dir / cls_names[split]).write_text("".join(cls_lines), encoding="utf-8")
        if split in seg_names:
            (out_dir / seg_names[split]).write_text("".join(seg_lines), encoding="utf-8")
        written[split] = {"studies": len(cls_lines), "positives": positives}

    sop_splits = defaultdict(set)
    for split, sops in split_sops.items():
        for sop in sops:
            sop_splits[sop].add(split)
    repeated_sops = {sop: splits for sop, splits in sop_splits.items() if len(splits) > 1}
    overlap = {patient: splits for patient, splits in patients.items() if len(splits) > 1}

    overlap_lines = [f"{patient}\t{','.join(sorted(splits))}\n" for patient, splits in sorted(overlap.items())]
    (out_dir / "patient_overlap.txt").write_text("".join(overlap_lines), encoding="utf-8")
    skipped_lines = [f"{split}\t{sop}\n" for split in JSON_NAMES for sop in skipped[split]]
    (out_dir / "skipped_missing_report.txt").write_text("".join(skipped_lines), encoding="utf-8")

    print(f"Reports rows: {len(reports)} (duplicate SOPInstanceUID rows ignored: {duplicate_rows})")
    for split in JSON_NAMES:
        print(
            f"{split}: json {len(split_sops[split])}, written {written[split]['studies']}, "
            f"positives {written[split]['positives']}, missing report {len(skipped[split])}, "
            f"blank mask {len(blank[split])}"
        )
    print(f"Patient indexes in more than one split: {len(overlap)}")
    print(f"Studies listed in more than one JSON file: {len(repeated_sops)}")
    print(f"Wrote lists under {out_dir}")
    return {
        "written": written,
        "overlap": len(overlap),
        "skipped": {split: len(items) for split, items in skipped.items()},
        "repeated_sops": len(repeated_sops),
    }


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    build(root)


if __name__ == "__main__":
    main()
