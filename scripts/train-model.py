"""`HangulFontRecognitionModel`(font_classifier/model.py)의 Phase 1 baseline을
학습하는 스크립트. `docs/model-design.md` 4.5절 로드맵의 Phase 1(재구성
없이 인코더 + 초중종성 헤더 + 폰트 헤더만 학습)만 다룬다 - 재구성
(모드 A/B, 4.2절)과 대조학습(4.4절)은 이번 범위가 아니다(자세한 근거는
docs/train-model.md 참고).

기본 권장 경로는 cell-holdout split manifest를 읽어 train split으로 학습하고,
val split을 augmentation 없이 평가하는 것이다. 기본 split manifest가 없으면
기존처럼 전체 데이터셋으로만 학습하되 validation/early stopping은 비활성화된다.

실행:
    uv run python scripts/train-model.py
    (Windows에서 --num-workers > 0을 쓰려면 반드시 이 스크립트처럼
    `if __name__ == "__main__":` 아래에서 실행해야 한다 - PyTorch의
    spawn 기반 multiprocessing 요구사항이다.)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.dataset_loader import (
    DATASET_DIR, DEFAULT_PRESCAN_WORKERS, FontGlyphDataset,
)
from font_classifier.model import HangulFontRecognitionModel

CHECKPOINT_DIR = DATASET_DIR.parent / "checkpoints"
DEFAULT_SPLIT_MANIFEST = DATASET_DIR.parent / "splits" / "cell-holdout-seed.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Phase 1 baseline (jamo + font classification heads only).")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--resume", type=Path, default=None,
                         help="이어서 학습할 체크포인트 파일 경로")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST,
                         help="cell-holdout split manifest 경로 (기본: data/splits/cell-holdout-seed.json)")
    parser.add_argument("--train-split", default="train",
                         help="학습에 사용할 manifest split 이름")
    parser.add_argument("--val-split", default="val",
                         help="validation에 사용할 manifest split 이름")
    parser.add_argument("--validate-every", type=int, default=1,
                         help="이 epoch 수마다 validation을 실행한다")
    parser.add_argument("--early-stop", action="store_true",
                         help="validation 기준 조기 정지를 켠다")
    parser.add_argument("--early-stop-min-epochs", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=1)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0015)
    parser.add_argument("--early-stop-loss-delta", type=float, default=0.003)
    parser.add_argument("--best-checkpoint-name", default="best.pt",
                         help="최고 validation 지점 저장 파일명")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-jamo", type=float, default=1.0)
    parser.add_argument("--lambda-font", type=float, default=1.0)
    parser.add_argument("--style-dim", type=int, default=512,
                         help="HangulFontRecognitionModel의 style_dim")
    parser.add_argument("--style-hidden-dim", type=int, default=1024,
                         help="HangulFontRecognitionModel의 style_hidden_dim")
    parser.add_argument("--prescan-workers", type=int, default=DEFAULT_PRESCAN_WORKERS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None,
                         help="기본값: cuda가 있으면 cuda, 없으면 cpu")
    parser.add_argument("--no-amp", action="store_true",
                         help="bfloat16 자동 혼합 정밀도를 끈다(기본은 cuda에서 켜짐)")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                         help="이 epoch 수마다 번호가 붙은 체크포인트를 남긴다")
    parser.add_argument("--log-every", type=int, default=50,
                         help="이 스텝 수마다 진행 상황을 출력하고 metrics.jsonl에 기록한다")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_lambda(warmup_steps: int, total_steps: int):
    """선형 warmup 후 코사인 감쇠(model-design.md 4.6절 "AdamW + cosine
    decay")."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return lr_lambda


class RunningAverage:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def value(self) -> float:
        return self.total / max(1, self.count)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def _split_manifest_was_provided() -> bool:
    for arg in sys.argv[1:]:
        if arg == "--split-manifest" or arg.startswith("--split-manifest="):
            return True
    return False


def _effective_split_manifest_path(path: Path | None) -> str | None:
    return str(path.resolve()) if path is not None else None


def _serialize_args(args: argparse.Namespace, split_manifest: Path | None) -> dict:
    payload = vars(args).copy()
    payload["dataset_dir"] = str(args.dataset_dir)
    payload["checkpoint_dir"] = str(args.checkpoint_dir)
    payload["resume"] = str(args.resume) if args.resume else None
    payload["split_manifest"] = _effective_split_manifest_path(split_manifest)
    return payload


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    *,
    split_manifest: Path | None,
    best_metric: dict | None,
    stale_validations: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": _serialize_args(args, split_manifest),
        "early_stop_state": {
            "stale_validations": stale_validations,
        },
    }
    if best_metric is not None:
        payload["best_metric"] = best_metric
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def load_split_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "font-classifier-cell-holdout":
        raise ValueError(f"{path}: unsupported split manifest kind {manifest.get('kind')!r}")
    fonts = manifest.get("fonts")
    if not isinstance(fonts, list) or not fonts:
        raise ValueError(f"{path}: split manifest must contain a non-empty 'fonts' list")
    return manifest


def build_flat_index_lookup(dataset: FontGlyphDataset) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for flat_index, cell in enumerate(dataset.valid_cells()):
        if cell in lookup:
            raise ValueError(f"Duplicate valid cell detected in dataset: {cell}")
        lookup[cell] = flat_index
    return lookup


def indices_from_manifest(dataset: FontGlyphDataset, manifest: dict, split: str) -> list[int]:
    lookup = build_flat_index_lookup(dataset)
    indices: list[int] = []
    seen_cells: set[tuple[int, int]] = set()

    if len(manifest["fonts"]) != dataset.num_font_classes:
        raise ValueError(
            "split manifest font count does not match dataset font count: "
            f"{len(manifest['fonts'])} vs {dataset.num_font_classes}"
        )

    for font in manifest["fonts"]:
        font_id = font["font_id"]
        splits = font.get("splits", {})
        if split not in splits:
            raise ValueError(f"split manifest font_id={font_id} missing split {split!r}")
        for char_index in splits[split]:
            cell = (font_id, char_index)
            if cell in seen_cells:
                raise ValueError(f"Duplicate cell in split manifest {split!r}: {cell}")
            try:
                indices.append(lookup[cell])
            except KeyError as exc:
                raise ValueError(
                    f"split manifest cell {cell} does not exist in dataset"
                ) from exc
            seen_cells.add(cell)
    return indices


def resolve_split_manifest(args: argparse.Namespace) -> Path | None:
    path = args.split_manifest
    if path is None:
        return None
    if path.exists():
        return path
    if _split_manifest_was_provided():
        raise SystemExit(f"Split manifest does not exist: {path}")
    print(
        f"[WARNING] Default split manifest not found at {path} - "
        "validation/early stopping disabled and full-dataset training will be used."
    )
    return None


def evaluate(model, loader: DataLoader, device: torch.device, use_amp: bool,
             args: argparse.Namespace) -> dict[str, float]:
    metric_names = [
        "val_loss", "val_loss_jamo", "val_loss_font", "val_cho_acc", "val_jung_acc",
        "val_jong_acc", "val_syllable_acc", "val_font_acc", "val_font_top5_acc",
    ]
    running = {name: RunningAverage() for name in metric_names}
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            cho_labels = batch["cho_label"].to(device, non_blocking=True)
            jung_labels = batch["jung_label"].to(device, non_blocking=True)
            jong_labels = batch["jong_label"].to(device, non_blocking=True)
            font_labels = batch["font_label"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model.encode(images)
                loss_cho = F.cross_entropy(out.cho_logits, cho_labels)
                loss_jung = F.cross_entropy(out.jung_logits, jung_labels)
                loss_jong = F.cross_entropy(out.jong_logits, jong_labels)
                loss_jamo = loss_cho + loss_jung + loss_jong
                loss_font = F.cross_entropy(out.font_logits, font_labels)
                loss = args.lambda_jamo * loss_jamo + args.lambda_font * loss_font

            cho_correct = out.cho_logits.argmax(-1) == cho_labels
            jung_correct = out.jung_logits.argmax(-1) == jung_labels
            jong_correct = out.jong_logits.argmax(-1) == jong_labels
            syllable_correct = cho_correct & jung_correct & jong_correct
            font_correct = out.font_logits.argmax(-1) == font_labels
            top5_k = min(5, out.font_logits.shape[-1])
            font_top5 = out.font_logits.topk(top5_k, dim=-1).indices
            font_top5_correct = (font_top5 == font_labels.unsqueeze(1)).any(dim=1)

            batch_size = images.shape[0]
            running["val_loss"].update(loss.item(), batch_size)
            running["val_loss_jamo"].update(loss_jamo.item(), batch_size)
            running["val_loss_font"].update(loss_font.item(), batch_size)
            running["val_cho_acc"].update(cho_correct.float().mean().item(), batch_size)
            running["val_jung_acc"].update(jung_correct.float().mean().item(), batch_size)
            running["val_jong_acc"].update(jong_correct.float().mean().item(), batch_size)
            running["val_syllable_acc"].update(syllable_correct.float().mean().item(), batch_size)
            running["val_font_acc"].update(font_correct.float().mean().item(), batch_size)
            running["val_font_top5_acc"].update(font_top5_correct.float().mean().item(), batch_size)

    if was_training:
        model.train()

    return {name: avg.value for name, avg in running.items()}


def is_better_validation(record: dict, best_metric: dict | None,
                         args: argparse.Namespace) -> bool:
    if best_metric is None:
        return True

    best_acc = float(best_metric["value"])
    current_acc = float(record["val_font_acc"])
    if current_acc >= best_acc + args.early_stop_min_delta:
        return True

    best_loss_font = float(best_metric.get("val_loss_font", float("inf")))
    same_accuracy = abs(current_acc - best_acc) < args.early_stop_min_delta
    better_loss = record["val_loss_font"] <= best_loss_font - args.early_stop_loss_delta
    return same_accuracy and better_loss


def make_best_metric(record: dict, epoch: int, global_step: int,
                     split_manifest: Path) -> dict:
    return {
        "name": "val_font_acc",
        "value": record["val_font_acc"],
        "epoch": epoch,
        "global_step": global_step,
        "val_loss_font": record["val_loss_font"],
        "split_manifest": str(split_manifest),
    }


def validate_resume_configuration(checkpoint: dict, split_manifest: Path | None) -> None:
    saved_args = checkpoint.get("args", {})
    saved_manifest = saved_args.get("split_manifest")
    current_manifest = _effective_split_manifest_path(split_manifest)

    if saved_manifest != current_manifest:
        raise SystemExit(
            "Resume split manifest mismatch: "
            f"checkpoint={saved_manifest!r}, current={current_manifest!r}"
        )


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    if args.validate_every < 1:
        raise SystemExit("--validate-every must be >= 1")
    if args.early_stop_min_epochs < 1:
        raise SystemExit("--early-stop-min-epochs must be >= 1")
    if args.early_stop_patience < 1:
        raise SystemExit("--early-stop-patience must be >= 1")

    split_manifest_path = resolve_split_manifest(args)
    if args.early_stop and split_manifest_path is None:
        raise SystemExit("--early-stop requires an active split manifest")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    manifest = None
    train_dataset = None
    val_loader = None

    print(f"Loading dataset from {args.dataset_dir} ...")
    if split_manifest_path is not None:
        manifest = load_split_manifest(split_manifest_path)
        train_base = FontGlyphDataset(
            args.dataset_dir, augment=True, prescan_workers=args.prescan_workers,
        )
        val_base = FontGlyphDataset(
            args.dataset_dir, augment=False, prescan_workers=args.prescan_workers,
        )
        if train_base.num_font_classes != val_base.num_font_classes:
            raise SystemExit("train/val dataset font class count mismatch")

        train_indices = indices_from_manifest(train_base, manifest, args.train_split)
        val_indices = indices_from_manifest(val_base, manifest, args.val_split)
        if not train_indices:
            raise SystemExit(f"split {args.train_split!r} contains no training samples")
        if not val_indices:
            raise SystemExit(f"split {args.val_split!r} contains no validation samples")

        train_dataset = Subset(train_base, train_indices)
        val_dataset = Subset(val_base, val_indices)
        print(
            f"Using split manifest {split_manifest_path} | "
            f"train={len(train_dataset)} sample(s) val={len(val_dataset)} sample(s)"
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        num_font_classes = train_base.num_font_classes
    else:
        print("[WARNING] Validation is disabled - training on the full dataset only.")
        train_dataset = FontGlyphDataset(
            args.dataset_dir, augment=True, prescan_workers=args.prescan_workers,
        )
        num_font_classes = train_dataset.num_font_classes

    print(f"{len(train_dataset)} training sample(s) across {num_font_classes} font(s)")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if len(train_loader) == 0:
        raise SystemExit("Training loader is empty - check batch size and split manifest")

    model = HangulFontRecognitionModel(
        num_font_classes,
        style_dim=args.style_dim,
        style_hidden_dim=args.style_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, total_steps))

    start_epoch = 0
    global_step = 0
    best_metric: dict | None = None
    stale_validations = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        validate_resume_configuration(checkpoint, split_manifest_path)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_metric = checkpoint.get("best_metric")
        stale_validations = checkpoint.get("early_stop_state", {}).get("stale_validations", 0)
        print(f"Resumed from {args.resume}: starting at epoch {start_epoch}")

    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Mixed precision (bf16): {'on' if use_amp else 'off'}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = (args.checkpoint_dir / "metrics.jsonl").open("a", encoding="utf-8")

    metric_names = ["loss", "loss_jamo", "loss_font", "cho_acc", "jung_acc",
                     "jong_acc", "syllable_acc", "font_acc", "font_top5_acc"]
    running = {name: RunningAverage() for name in metric_names}
    log_window_start = time.time()
    samples_since_log = 0

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                cho_labels = batch["cho_label"].to(device, non_blocking=True)
                jung_labels = batch["jung_label"].to(device, non_blocking=True)
                jong_labels = batch["jong_label"].to(device, non_blocking=True)
                font_labels = batch["font_label"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = model.encode(images)
                    loss_cho = F.cross_entropy(out.cho_logits, cho_labels)
                    loss_jung = F.cross_entropy(out.jung_logits, jung_labels)
                    loss_jong = F.cross_entropy(out.jong_logits, jong_labels)
                    loss_jamo = loss_cho + loss_jung + loss_jong
                    loss_font = F.cross_entropy(out.font_logits, font_labels)
                    loss = args.lambda_jamo * loss_jamo + args.lambda_font * loss_font

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    cho_correct = out.cho_logits.argmax(-1) == cho_labels
                    jung_correct = out.jung_logits.argmax(-1) == jung_labels
                    jong_correct = out.jong_logits.argmax(-1) == jong_labels
                    syllable_correct = cho_correct & jung_correct & jong_correct
                    font_correct = out.font_logits.argmax(-1) == font_labels
                    top5_k = min(5, out.font_logits.shape[-1])
                    font_top5 = out.font_logits.topk(top5_k, dim=-1).indices
                    font_top5_correct = (font_top5 == font_labels.unsqueeze(1)).any(dim=1)

                batch_size = images.shape[0]
                running["loss"].update(loss.item(), batch_size)
                running["loss_jamo"].update(loss_jamo.item(), batch_size)
                running["loss_font"].update(loss_font.item(), batch_size)
                running["cho_acc"].update(cho_correct.float().mean().item(), batch_size)
                running["jung_acc"].update(jung_correct.float().mean().item(), batch_size)
                running["jong_acc"].update(jong_correct.float().mean().item(), batch_size)
                running["syllable_acc"].update(syllable_correct.float().mean().item(), batch_size)
                running["font_acc"].update(font_correct.float().mean().item(), batch_size)
                running["font_top5_acc"].update(font_top5_correct.float().mean().item(), batch_size)
                samples_since_log += batch_size

                global_step += 1
                if global_step % args.log_every == 0:
                    elapsed = max(1e-6, time.time() - log_window_start)
                    samples_per_sec = samples_since_log / elapsed
                    record = {
                        "type": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "lr": scheduler.get_last_lr()[0],
                        "samples_per_sec": samples_per_sec,
                        **{name: avg.value for name, avg in running.items()},
                    }
                    print(
                        f"epoch {epoch} step {global_step} "
                        f"loss={record['loss']:.4f} (jamo={record['loss_jamo']:.4f} "
                        f"font={record['loss_font']:.4f}) "
                        f"acc[cho={record['cho_acc']:.3f} jung={record['jung_acc']:.3f} "
                        f"jong={record['jong_acc']:.3f} syllable={record['syllable_acc']:.3f} "
                        f"font={record['font_acc']:.3f} font_top5={record['font_top5_acc']:.3f}] "
                        f"lr={record['lr']:.2e} "
                        f"({samples_per_sec:.0f} samples/s)"
                    )
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    for avg in running.values():
                        avg.reset()
                    log_window_start = time.time()
                    samples_since_log = 0

            should_validate = (
                val_loader is not None and
                ((epoch + 1) % args.validate_every == 0 or epoch == args.epochs - 1)
            )
            if should_validate:
                val_metrics = evaluate(model, val_loader, device, use_amp, args)
                val_record = {
                    "type": "val",
                    "epoch": epoch,
                    "step": global_step,
                    "split_manifest": str(split_manifest_path),
                    "num_samples": len(val_loader.dataset),
                    **val_metrics,
                }
                print(
                    f"validation epoch {epoch} step {global_step} "
                    f"loss={val_record['val_loss']:.4f} "
                    f"(jamo={val_record['val_loss_jamo']:.4f} "
                    f"font={val_record['val_loss_font']:.4f}) "
                    f"acc[cho={val_record['val_cho_acc']:.3f} "
                    f"jung={val_record['val_jung_acc']:.3f} "
                    f"jong={val_record['val_jong_acc']:.3f} "
                    f"syllable={val_record['val_syllable_acc']:.3f} "
                    f"font={val_record['val_font_acc']:.3f} "
                    f"font_top5={val_record['val_font_top5_acc']:.3f}]"
                )
                metrics_file.write(json.dumps(val_record) + "\n")
                metrics_file.flush()

                if is_better_validation(val_record, best_metric, args):
                    best_metric = make_best_metric(val_record, epoch, global_step, split_manifest_path)
                    stale_validations = 0
                    best_path = args.checkpoint_dir / args.best_checkpoint_name
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        args,
                        split_manifest=split_manifest_path,
                        best_metric=best_metric,
                        stale_validations=stale_validations,
                    )
                    print(f"Saved best checkpoint: {best_path}")
                else:
                    stale_validations += 1
                    print(
                        f"Validation did not improve (stale={stale_validations}/"
                        f"{args.early_stop_patience})."
                    )

            if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
                epoch_path = args.checkpoint_dir / f"checkpoint-epoch-{epoch:04d}.pt"
                save_checkpoint(
                    epoch_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    args,
                    split_manifest=split_manifest_path,
                    best_metric=best_metric,
                    stale_validations=stale_validations,
                )
                print(f"Saved checkpoint: {epoch_path}")

            save_checkpoint(
                args.checkpoint_dir / "latest.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                args,
                split_manifest=split_manifest_path,
                best_metric=best_metric,
                stale_validations=stale_validations,
            )

            if (
                args.early_stop and
                val_loader is not None and
                should_validate and
                epoch + 1 >= args.early_stop_min_epochs and
                stale_validations >= args.early_stop_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}: no validation improvement for "
                    f"{stale_validations} validation run(s)."
                )
                break

        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted - saving checkpoint before exit...")
        save_checkpoint(
            args.checkpoint_dir / "interrupted.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            args,
            split_manifest=split_manifest_path,
            best_metric=best_metric,
            stale_validations=stale_validations,
        )
    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()
