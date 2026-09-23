#!/usr/bin/env python3
"""Build the CANDID-PTX split files this repo's loaders open.

The raw release looks like:

    CANDID-PTX/
      dataset/                  DICOM images
      Pneumothorax_reports.csv  SOPInstanceUID + EncodedPixels (RLE, or -1)
      chest_tube.csv            unused by the pneumothorax heads
      Rib_fracture_mask_rle.csv unused by the pneumothorax heads

This writes the files config/dataset_locations_jliang12.yml already points at:

    data_files_splits/candid_ptx/CANDIDPTX_cls_{train,val,test}.txt
    data_files_splits/candid_ptx/{train,test}.txt
    data_files_splits/candid_ptx/CANDID_PTX_{train_1,valid_1,test_1}.json
    png/                        RGB PNGs for the localization loader

Classification lines are "<relative-dicom> <0|1>".
Segmentation lines are "<relative-dicom>,<rle or -1>".
Localization JSON is COCO boxes around each pneumothorax region.
Patients are kept in one split (70% train, 10% val, 20% test).

    ./scripts-apptainer/prepare_candidptx_splits.sh /scratch/$USER/CANDID-PTX

Run it where numpy, pydicom, and Pillow are installed. On SOL that is inside
the container:

    cd /scratch/$USER/Foundation_X_Plus-CUDA-Container
    ./cuda-apptainer.sh exec python3 scripts-apptainer/prepare_candidptx_splits.py \\
        --root /scratch/$USER/CANDID-PTX
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

csv.field_size_limit(sys.maxsize)

ID_KEYS = (
    "sopinstanceuid",
    "imageid",
    "imageuid",
    "filename",
    "filepath",
    "dicomid",
)
RLE_KEYS = ("encodedpixels", "encodedpixel", "rle", "maskrle", "pixels")
PATIENT_KEYS = ("patientid", "patientindex", "subjectid", "patient")
IMAGE_SUFFIXES = {".dcm", ".dicom", ".ima"}


def _key(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _find_column(fieldnames, candidates, label):
    normalized = {_key(name): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise SystemExit(
        "Pneumothorax_reports.csv has no %s column. Headers: %s"
        % (label, ", ".join(fieldnames))
    )


def rle_to_mask(rle, rows, cols):
    """Match Candid_PTX_PXSDataset.rle2mask, which is called as (rle, *image.shape)."""
    mask = np.zeros(rows * cols, dtype=np.uint8)
    if rle == "-1":
        return mask.reshape(rows, cols)
    array = np.asarray([int(x) for x in rle.split()], dtype=np.int64)
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for start, length in zip(starts, lengths):
        current_position += int(start)
        end = current_position + int(length)
        if current_position < 0 or end > mask.size:
            raise ValueError("RLE runs outside a %dx%d image" % (rows, cols))
        mask[current_position:end] = 255
        current_position = end
    return mask.reshape(rows, cols).T


def mask_to_rle(mask, rows, cols):
    """Inverse of rle_to_mask. Empty masks become '-1'."""
    flat = np.asarray(mask).T.reshape(-1)
    if flat.shape[0] != rows * cols:
        raise ValueError("mask shape %s does not match %dx%d" % (mask.shape, rows, cols))
    parts = []
    pos = 0
    size = flat.shape[0]
    while pos < size:
        zeros = 0
        while pos < size and flat[pos] == 0:
            zeros += 1
            pos += 1
        if pos >= size:
            break
        length = 0
        while pos < size and flat[pos] != 0:
            length += 1
            pos += 1
        parts.append(str(zeros))
        parts.append(str(length))
    if not parts:
        return "-1"
    return " ".join(parts)


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    width = int(xs.max()) - x0 + 1
    height = int(ys.max()) - y0 + 1
    return [x0, y0, width, height]


def _self_test():
    rows, cols = 8, 8
    rle = "3 2 4 1"
    mask = rle_to_mask(rle, rows, cols)
    encoded = mask_to_rle(mask, rows, cols)
    if not np.array_equal(rle_to_mask(encoded, rows, cols), mask):
        raise SystemExit("RLE round-trip failed: %s -> %s" % (rle, encoded))
    merged = mask_to_rle(
        np.maximum(rle_to_mask("0 1", rows, cols), rle_to_mask("10 2", rows, cols)),
        rows,
        cols,
    )
    if mask_to_bbox(rle_to_mask("0 1", rows, cols)) is None:
        raise SystemExit("bbox missing for a one-pixel mask")
    if "-1" != mask_to_rle(np.zeros((cols, rows), dtype=np.uint8), rows, cols):
        raise SystemExit("empty mask did not encode as -1")
    print("RLE self-test ok (%s)" % merged)


def _is_image_file(path):
    if path.name.startswith("."):
        return False
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return True
    return path.suffix == ""


def _index_dicoms(dataset_dir):
    files = [path for path in dataset_dir.rglob("*") if path.is_file() and _is_image_file(path)]
    suffixed = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    if suffixed:
        files = suffixed
    by_stem = {}
    for path in files:
        by_stem.setdefault(path.stem.lower(), path)
    return files, by_stem


def _read_header(path):
    import pydicom

    dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    sop = str(getattr(dataset, "SOPInstanceUID", "") or "").strip()
    patient = str(getattr(dataset, "PatientID", "") or "").strip()
    rows = int(getattr(dataset, "Rows", 0) or 0)
    cols = int(getattr(dataset, "Columns", 0) or 0)
    return sop, patient, rows, cols


def _patient_from_name(path):
    token = path.stem.split("_")[0].strip()
    return token or path.stem


def _positive_rle(value):
    text = (value or "").strip()
    if text in {"", "-1", "nan", "none", "null"}:
        return None
    return text


def _load_reports(csv_path):
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("Pneumothorax_reports.csv has no header row")
        id_column = _find_column(reader.fieldnames, ID_KEYS, "image id")
        rle_column = _find_column(reader.fieldnames, RLE_KEYS, "mask")
        patient_column = None
        normalized = {_key(name): name for name in reader.fieldnames}
        for candidate in PATIENT_KEYS:
            if candidate in normalized:
                patient_column = normalized[candidate]
                break
        grouped = defaultdict(list)
        patients = {}
        for row in reader:
            image_id = (row.get(id_column) or "").strip()
            if not image_id:
                continue
            grouped[image_id].append(row.get(rle_column) or "")
            if patient_column:
                patient = (row.get(patient_column) or "").strip()
                if patient:
                    patients[image_id] = patient
    return grouped, patients, id_column, rle_column, patient_column


def _match_images(dataset_dir, grouped, csv_patients):
    files, by_stem = _index_dicoms(dataset_dir)
    if not files:
        raise SystemExit("No DICOM files under %s" % dataset_dir)

    print("Reading DICOM headers for %d files" % len(files))
    header_by_path = {}
    sop_to_path = {}
    for index, path in enumerate(files, start=1):
        if index % 1000 == 0 or index == len(files):
            print("  headers %d/%d" % (index, len(files)))
        try:
            sop, patient, rows, cols = _read_header(path)
        except Exception as exc:
            print("  skip unreadable %s (%s)" % (path, exc))
            continue
        header_by_path[path] = (patient, rows, cols)
        if sop:
            sop_to_path.setdefault(sop.lower(), path)

    records = []
    unmatched = []
    for image_id, rle_values in grouped.items():
        key = image_id.lower()
        path = by_stem.get(key) or sop_to_path.get(key)
        if path is None:
            stem = Path(image_id).stem.lower()
            path = by_stem.get(stem) or sop_to_path.get(stem)
        if path is None or path not in header_by_path:
            unmatched.append(image_id)
            continue
        patient, rows, cols = header_by_path[path]
        patient = csv_patients.get(image_id) or patient or _patient_from_name(path)
        if rows <= 0 or cols <= 0:
            rows, cols = 1024, 1024
        positive = []
        for value in rle_values:
            rle = _positive_rle(value)
            if rle is not None:
                positive.append(rle)
        records.append(
            {
                "id": image_id,
                "path": path,
                "rel": path.relative_to(dataset_dir).as_posix(),
                "patient": patient,
                "rows": rows,
                "cols": cols,
                "rles": positive,
                "label": 1 if positive else 0,
            }
        )
    return records, unmatched


def _assign_splits(records, seed, train_ratio, val_ratio):
    patients = sorted({record["patient"] for record in records})
    if len(patients) < 3:
        raise SystemExit("Need at least 3 patients to make train, val, and test splits")
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_ratio)))
    n_test = max(1, int(round(len(patients) * (1.0 - train_ratio - val_ratio))))
    if n_val + n_test >= len(patients):
        n_test = 1
        n_val = 1
    train_ids = set(patients[: len(patients) - n_val - n_test])
    val_ids = set(patients[len(train_ids) : len(train_ids) + n_val])
    test_ids = set(patients[len(train_ids) + n_val :])
    if (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
        or len(train_ids) + len(val_ids) + len(test_ids) != len(patients)
    ):
        raise SystemExit("patient split did not partition the patients")
    for record in records:
        if record["patient"] in train_ids:
            record["split"] = "train"
        elif record["patient"] in val_ids:
            record["split"] = "val"
        else:
            record["split"] = "test"
    return train_ids, val_ids, test_ids


def _segmentation_rle(record):
    if not record["rles"]:
        return "-1"
    if len(record["rles"]) == 1:
        return record["rles"][0]
    merged = None
    for rle in record["rles"]:
        mask = rle_to_mask(rle, record["rows"], record["cols"])
        merged = mask if merged is None else np.maximum(merged, mask)
    return mask_to_rle(merged, record["rows"], record["cols"])


def _write_lists(records, split_dir):
    buckets = {"train": [], "val": [], "test": []}
    for record in records:
        buckets[record["split"]].append(record)
    for name in buckets:
        buckets[name].sort(key=lambda record: record["rel"])

    def write_cls(split_name, filename):
        path = split_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for record in buckets[split_name]:
                handle.write("%s %d\n" % (record["rel"], record["label"]))
        return path

    def write_seg(split_name, filename):
        path = split_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for record in buckets[split_name]:
                handle.write("%s,%s\n" % (record["rel"], _segmentation_rle(record)))
        return path

    written = [
        write_cls("train", "CANDIDPTX_cls_train.txt"),
        write_cls("val", "CANDIDPTX_cls_val.txt"),
        write_cls("test", "CANDIDPTX_cls_test.txt"),
        write_seg("train", "train.txt"),
        write_seg("test", "test.txt"),
    ]
    return buckets, written


def _annotations_for(record):
    boxes = []
    for rle in record["rles"]:
        bbox = mask_to_bbox(rle_to_mask(rle, record["rows"], record["cols"]))
        if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
            boxes.append(bbox)
    return boxes


def _write_coco(buckets, split_dir, png_name_for):
    names = {
        "train": "CANDID_PTX_train_1.json",
        "val": "CANDID_PTX_valid_1.json",
        "test": "CANDID_PTX_test_1.json",
    }
    written = []
    for split_name, filename in names.items():
        images = []
        annotations = []
        ann_id = 1
        for image_id, record in enumerate(buckets[split_name], start=1):
            images.append(
                {
                    "id": image_id,
                    "file_name": png_name_for(record),
                    "width": record["cols"],
                    "height": record["rows"],
                }
            )
            for bbox in _annotations_for(record):
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "pneumothorax"}],
        }
        path = split_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        written.append(path)
    return written


def _png_name(record):
    rel = Path(record["rel"])
    return rel.with_suffix(".png").as_posix()


def _dicom_to_uint8(dataset):
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1) or 1)
    intercept = float(getattr(dataset, "RescaleIntercept", 0) or 0)
    pixels = pixels * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = pixels.max() - pixels
    low = float(pixels.min())
    high = float(pixels.max())
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    return ((pixels - low) * (255.0 / (high - low))).astype(np.uint8)


def _write_pngs(records, png_root, overwrite):
    import pydicom
    from PIL import Image

    written = 0
    skipped = 0
    for index, record in enumerate(records, start=1):
        destination = png_root / _png_name(record)
        if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset = pydicom.dcmread(str(record["path"]), force=True)
        image = Image.fromarray(_dicom_to_uint8(dataset)).convert("RGB")
        image.save(destination)
        written += 1
        if index % 500 == 0 or index == len(records):
            print("  png %d/%d" % (index, len(records)))
    return written, skipped


def _write_summary(split_dir, records, unmatched_path):
    lines = []
    for split_name in ("train", "val", "test"):
        chosen = [record for record in records if record["split"] == split_name]
        patients = {record["patient"] for record in chosen}
        positives = sum(record["label"] for record in chosen)
        lines.append(
            "%s images=%d patients=%d positive=%d negative=%d"
            % (split_name, len(chosen), len(patients), positives, len(chosen) - positives)
        )
    overlap = set()
    seen = {}
    for record in records:
        previous = seen.get(record["patient"])
        if previous is not None and previous != record["split"]:
            overlap.add(record["patient"])
        seen[record["patient"]] = record["split"]
    lines.append("patient_overlap=%d" % len(overlap))
    lines.append("unmatched_list=%s" % unmatched_path)
    text = "\n".join(lines) + "\n"
    (split_dir / "split_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/scratch/hflechsi/CANDID-PTX")
    parser.add_argument("--csv", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=0, help="Keep only the first N matched images")
    parser.add_argument("--skip-png", action="store_true")
    parser.add_argument("--overwrite-png", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    root = Path(args.root)
    csv_path = Path(args.csv) if args.csv else root / "Pneumothorax_reports.csv"
    dataset_dir = Path(args.dataset) if args.dataset else root / "dataset"
    split_dir = root / "data_files_splits" / "candid_ptx"
    png_root = root / "png"
    if not csv_path.is_file():
        raise SystemExit("Missing %s" % csv_path)
    if not dataset_dir.is_dir():
        raise SystemExit("Missing %s" % dataset_dir)

    grouped, csv_patients, id_column, rle_column, patient_column = _load_reports(csv_path)
    print(
        "CSV rows grouped into %d images (id=%s, mask=%s, patient=%s)"
        % (len(grouped), id_column, rle_column, patient_column or "DICOM PatientID")
    )
    records, unmatched = _match_images(dataset_dir, grouped, csv_patients)
    unmatched_path = split_dir / "unmatched_ids.txt"
    split_dir.mkdir(parents=True, exist_ok=True)
    unmatched_path.write_text("\n".join(unmatched) + ("\n" if unmatched else ""), encoding="utf-8")
    if unmatched:
        raise SystemExit(
            "%d CSV image ids did not match a DICOM. See %s" % (len(unmatched), unmatched_path)
        )
    records.sort(key=lambda record: record["rel"])
    if args.limit:
        records = records[: args.limit]
        print("Limited to %d images" % len(records))

    _assign_splits(records, args.seed, args.train_ratio, args.val_ratio)
    buckets, list_paths = _write_lists(records, split_dir)
    json_paths = _write_coco(buckets, split_dir, _png_name)
    if args.skip_png:
        print("Skipped PNG conversion")
    else:
        png_root.mkdir(parents=True, exist_ok=True)
        written, skipped = _write_pngs(records, png_root, args.overwrite_png)
        print("PNG written=%d skipped=%d under %s" % (written, skipped, png_root))
    for path in list_paths + json_paths:
        print("wrote %s" % path)
    _write_summary(split_dir, records, unmatched_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
