"""`data/dataset`의 유효 glyph 셀을 기준으로 재현 가능한 per-font
cell-holdout split manifest를 생성한다.

실행:
    uv run python scripts/generate-split.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.font_dataset import (
    HANGUL_TABLE,
    NUM_CHO,
    NUM_JONG,
    NUM_JUNG,
    decompose_hangul_syllable,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR.parent / "data" / "dataset"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "data" / "splits" / "cell-holdout-seed.json"
HASH_KEY_FORMAT = "font-classifier-cell-holdout|<seed>|<font_id>|<char_index>"
RATIO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FontSplit:
    train: list[int]
    val: list[int]
    test: list[int]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic per-font cell-holdout split manifest.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--min-val-per-font", type=int, default=1)
    parser.add_argument("--min-test-per-font", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_index(dataset_dir: Path) -> tuple[list[dict], Path]:
    index_path = dataset_dir / "index.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"{index_path}: expected a JSON list")

    ids = sorted(entry["id"] for entry in entries)
    if ids != list(range(len(entries))):
        raise ValueError(
            f"{index_path}: 'id' must form a contiguous 0..N-1 range "
            f"(N={len(entries)}), got e.g. {ids[:5]}"
        )

    for entry in entries:
        if "dir" not in entry:
            raise ValueError(f"{index_path}: each entry must contain 'dir'")

    entries.sort(key=lambda entry: entry["id"])
    return entries, index_path


def _scan_font_char_indices(dataset_dir: Path, entry: dict) -> list[int]:
    font_dir = dataset_dir / entry["dir"]
    if not font_dir.is_dir():
        raise ValueError(f"{font_dir}: font directory does not exist")

    char_indices: list[int] = []
    for path in font_dir.glob("*.png"):
        stem = path.stem
        if len(stem) != 4 or not stem.isdigit():
            continue
        char_index = int(stem)
        if not (0 <= char_index < len(HANGUL_TABLE)):
            raise ValueError(
                f"{path}: char_index must satisfy 0 <= idx < {len(HANGUL_TABLE)}"
            )
        char_indices.append(char_index)

    char_indices.sort()
    if not char_indices:
        raise ValueError(f"{font_dir}: no valid glyph PNG files found")
    return char_indices


def _cell_hash(seed: int, font_id: int, char_index: int) -> bytes:
    key = f"font-classifier-cell-holdout|{seed}|{font_id}|{char_index}"
    return hashlib.sha256(key.encode("utf-8")).digest()


def _compute_split_counts(
    valid_count: int,
    val_ratio: float,
    test_ratio: float,
    min_val_per_font: int,
    min_test_per_font: int,
) -> tuple[int, int, int]:
    val_count = round(valid_count * val_ratio)
    test_count = round(valid_count * test_ratio)

    if valid_count >= 20:
        val_count = max(val_count, min_val_per_font)
        test_count = max(test_count, min_test_per_font)

    train_count = valid_count - val_count - test_count
    if train_count >= 1:
        return train_count, val_count, test_count

    deficit = 1 - train_count
    val_floor = min_val_per_font if valid_count >= 20 else 0
    test_floor = min_test_per_font if valid_count >= 20 else 0

    while deficit > 0 and (val_count > 0 or test_count > 0):
        val_slack = val_count - val_floor
        test_slack = test_count - test_floor

        if val_slack > test_slack and val_count > 0:
            val_count -= 1
        elif test_slack > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        deficit -= 1

    train_count = valid_count - val_count - test_count
    if train_count < 1:
        raise ValueError(
            f"Unable to reserve at least one train cell from {valid_count} valid cell(s)"
        )
    return train_count, val_count, test_count


def _assign_font_split(
    font_id: int,
    char_indices: list[int],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    min_val_per_font: int,
    min_test_per_font: int,
) -> FontSplit:
    train_count, val_count, test_count = _compute_split_counts(
        len(char_indices),
        val_ratio,
        test_ratio,
        min_val_per_font,
        min_test_per_font,
    )

    hashed = sorted(
        char_indices,
        key=lambda char_index: (_cell_hash(seed, font_id, char_index), char_index),
    )
    test_chars = sorted(hashed[:test_count])
    val_chars = sorted(hashed[test_count:test_count + val_count])
    train_chars = sorted(hashed[test_count + val_count:])

    split = FontSplit(train=train_chars, val=val_chars, test=test_chars)
    _validate_font_split(char_indices, split, font_id)

    if len(split.train) != train_count:
        raise ValueError(f"Font {font_id}: unexpected train count mismatch")
    return split


def _validate_font_split(char_indices: list[int], split: FontSplit, font_id: int) -> None:
    train_set = set(split.train)
    val_set = set(split.val)
    test_set = set(split.test)
    full_set = set(char_indices)

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError(f"Font {font_id}: train/val/test overlap detected")

    if train_set | val_set | test_set != full_set:
        raise ValueError(f"Font {font_id}: split union does not match valid char set")


def _mean(values: list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _label_distribution(font_splits: list[dict]) -> dict[str, dict[str, list[int]]]:
    distribution = {
        split_name: {
            "cho": [0] * NUM_CHO,
            "jung": [0] * NUM_JUNG,
            "jong": [0] * NUM_JONG,
        }
        for split_name in ("train", "val", "test")
    }

    for font in font_splits:
        for split_name in ("train", "val", "test"):
            for char_index in font["splits"][split_name]:
                cho, jung, jong = decompose_hangul_syllable(HANGUL_TABLE[char_index])
                distribution[split_name]["cho"][cho] += 1
                distribution[split_name]["jung"][jung] += 1
                distribution[split_name]["jong"][jong] += 1
    return distribution


def _build_report(fonts: list[dict], totals: dict[str, int]) -> dict:
    total_cells_per_font = [font["counts"]["total"] for font in fonts]
    empty_split_fonts = []
    for font in fonts:
        empty = [name for name in ("train", "val", "test") if font["counts"][name] == 0]
        if empty:
            empty_split_fonts.append({
                "font_id": font["font_id"],
                "empty_splits": empty,
            })

    overlap_counts = {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    for font in fonts:
        train_set = set(font["splits"]["train"])
        val_set = set(font["splits"]["val"])
        test_set = set(font["splits"]["test"])
        overlap_counts["train_val"] += len(train_set & val_set)
        overlap_counts["train_test"] += len(train_set & test_set)
        overlap_counts["val_test"] += len(val_set & test_set)

    split_stats = {}
    for split_name in ("train", "val", "test"):
        counts = [font["counts"][split_name] for font in fonts]
        split_stats[split_name] = {
            "min": min(counts),
            "max": max(counts),
            "mean": _mean(counts),
        }

    return {
        "min_cells_per_font": min(total_cells_per_font),
        "max_cells_per_font": max(total_cells_per_font),
        "empty_split_fonts": empty_split_fonts,
        "split_count_stats_per_font": split_stats,
        "split_overlap_counts": overlap_counts,
        "label_distribution": _label_distribution(fonts),
        "num_fonts": len(fonts),
        "num_valid_cells": sum(total_cells_per_font),
        "totals": totals,
    }


def _print_report(fonts: list[dict], totals: dict[str, int], report: dict, output: Path) -> None:
    total_cells = report["num_valid_cells"]
    print(f"Fonts: {report['num_fonts']}")
    print(f"Valid cells: {total_cells}")
    for split_name in ("train", "val", "test"):
        count = totals[split_name]
        ratio = count / total_cells if total_cells else 0.0
        stats = report["split_count_stats_per_font"][split_name]
        print(
            f"{split_name}: {count} cell(s) ({ratio:.4%}) | "
            f"per-font min={stats['min']} max={stats['max']} mean={stats['mean']:.2f}"
        )
    if report["empty_split_fonts"]:
        print(f"Fonts with empty split(s): {len(report['empty_split_fonts'])}")
        for item in report["empty_split_fonts"][:20]:
            joined = ",".join(item["empty_splits"])
            print(f"  - font_id={item['font_id']}: {joined}")
        if len(report["empty_split_fonts"]) > 20:
            print("  - ...")
    else:
        print("Fonts with empty split(s): 0")

    overlaps = report["split_overlap_counts"]
    print(
        "Split overlaps: "
        f"train/val={overlaps['train_val']} "
        f"train/test={overlaps['train_test']} "
        f"val/test={overlaps['val_test']}"
    )
    print(f"Manifest: {output}")


def main() -> None:
    args = parse_args()

    if args.min_val_per_font < 0 or args.min_test_per_font < 0:
        raise SystemExit("--min-val-per-font and --min-test-per-font must be >= 0")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > RATIO_TOLERANCE:
        raise SystemExit(
            f"train/val/test ratios must sum to 1.0 within {RATIO_TOLERANCE}, got {ratio_sum}"
        )

    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {args.output} (pass --overwrite to replace it)"
        )

    entries, index_path = _load_index(args.dataset_dir)
    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()

    fonts: list[dict] = []
    totals = {"train": 0, "val": 0, "test": 0}

    for entry in entries:
        font_id = entry["id"]
        char_indices = _scan_font_char_indices(args.dataset_dir, entry)
        split = _assign_font_split(
            font_id,
            char_indices,
            args.seed,
            args.val_ratio,
            args.test_ratio,
            args.min_val_per_font,
            args.min_test_per_font,
        )

        font_record = {
            "font_id": font_id,
            "dir": entry["dir"],
            "counts": {
                "total": split.total,
                "train": len(split.train),
                "val": len(split.val),
                "test": len(split.test),
            },
            "splits": {
                "train": split.train,
                "val": split.val,
                "test": split.test,
            },
        }
        if "font_name" in entry:
            font_record["name"] = entry["font_name"]

        fonts.append(font_record)
        totals["train"] += len(split.train)
        totals["val"] += len(split.val)
        totals["test"] += len(split.test)

    report = _build_report(fonts, totals)
    if any(report["split_overlap_counts"].values()):
        raise SystemExit("split overlap count must be zero")

    manifest = {
        "schema_version": 1,
        "kind": "font-classifier-cell-holdout",
        "dataset_dir": str(args.dataset_dir),
        "created_by": "scripts/generate-split.py",
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "hash": {
            "algorithm": "sha256",
            "key_format": HASH_KEY_FORMAT,
        },
        "dataset_fingerprint": {
            "index_sha256": index_sha256,
            "num_fonts": len(fonts),
            "num_valid_cells": report["num_valid_cells"],
        },
        "totals": totals,
        "fonts": fonts,
        "report": report,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_report(fonts, totals, report, args.output)


if __name__ == "__main__":
    main()
