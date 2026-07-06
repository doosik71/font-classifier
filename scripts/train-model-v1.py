"""`FontRecognitionModel`(font_classifier/model.py)의 Phase 1 baseline을
학습하는 스크립트. `docs/model-design.md` 4.5절 로드맵의 Phase 1(재구성
없이 인코더 + 초중종성 헤더 + 폰트 헤더만 학습)만 다룬다 - 재구성
(모드 A/B, 4.2절)과 대조학습(4.4절)은 이번 범위가 아니다(자세한 근거는
docs/train-model.md 1절 참고). `model.encode()`만 호출해 디코더 연산
자체를 건너뛴다.

학습/검증 분할도 이번 범위에 없다 - `data/dataset`의 유효한 (폰트, 글자)
전체로 학습하고, 로그에 나오는 정확도는 모두 학습 데이터 기준이다(held-out
지표가 아니다). 자세한 근거는 docs/train-model.md 참고.

실행:
    uv run python scripts/train-model-v1.py
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.batch_sampler import (
    DEFAULT_CHARS_PER_FONT, DEFAULT_FONTS_PER_BATCH, FontGroupBatchSampler,
)
from font_classifier.dataset_loader import (
    DATASET_DIR, DEFAULT_MAX_CACHE_BYTES, DEFAULT_PRESCAN_WORKERS, FontGlyphDataset,
)
from font_classifier.model import FontRecognitionModel

# v1(baseline) 결과는 v2와 비교할 수 있도록 checkpoints/v1 하위 폴더에 따로
# 저장한다(scripts/train-model-v2.py는 checkpoints/v2에 저장한다).
CHECKPOINT_DIR = DATASET_DIR.parent / "checkpoints" / "v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Phase 1 baseline (jamo + font classification heads only).")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--resume", type=Path, default=None,
                         help="이어서 학습할 체크포인트 파일 경로")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fonts-per-batch", "-K", type=int, default=DEFAULT_FONTS_PER_BATCH)
    parser.add_argument("--chars-per-font", "-M", type=int, default=DEFAULT_CHARS_PER_FONT)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-jamo", type=float, default=1.0)
    parser.add_argument("--lambda-font", type=float, default=1.0)

    parser.add_argument("--max-cache-bytes", type=int, default=DEFAULT_MAX_CACHE_BYTES)
    parser.add_argument("--prescan-workers", type=int, default=DEFAULT_PRESCAN_WORKERS)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--device", default=None,
                         help="기본값: cuda가 있으면 cuda, 없으면 cpu")
    parser.add_argument("--no-amp", action="store_true",
                         help="bfloat16 자동 혼합 정밀도를 끈다(기본은 cuda에서 켜짐)")

    parser.add_argument("--checkpoint-every", type=int, default=1,
                         help="이 에폭 수마다 번호가 붙은 체크포인트를 남긴다")
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
    decay"). `total_steps`는 `FontGroupBatchSampler`의 근사 길이
    (docs/batch-sampler.md 2.4절)로 추정한 값이라 정확하지 않을 수 있지만,
    코사인 곡선의 모양을 정하는 용도로는 충분하다."""

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


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                     global_step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args) | {"dataset_dir": str(args.dataset_dir),
                               "checkpoint_dir": str(args.checkpoint_dir),
                               "resume": str(args.resume) if args.resume else None},
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["epoch"], checkpoint["global_step"]


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    print(f"Loading dataset from {args.dataset_dir} ...")
    dataset = FontGlyphDataset(
        args.dataset_dir, augment=True,
        max_cache_bytes=args.max_cache_bytes, prescan_workers=args.prescan_workers,
    )
    print(f"{len(dataset)} valid sample(s) across {dataset.num_font_classes} font(s)")

    sampler = FontGroupBatchSampler(
        dataset, fonts_per_batch=args.fonts_per_batch, chars_per_font=args.chars_per_font,
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = FontRecognitionModel(dataset.num_font_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(sampler)  # 근사치 (docs/batch-sampler.md 2.4절)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, total_steps))

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1
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
            for batch in loader:
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
                    # 폰트 top-5: 정답 폰트가 상위 5개 logit 안에 들면 정답으로 센다.
                    # 폰트 수가 5보다 적을 수 있는 합성 데이터셋을 위해 k를 제한한다.
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
                        "epoch": epoch, "step": global_step,
                        "lr": scheduler.get_last_lr()[0], "samples_per_sec": samples_per_sec,
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

            if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
                epoch_path = args.checkpoint_dir / f"checkpoint-epoch-{epoch:04d}.pt"
                save_checkpoint(epoch_path, model, optimizer, scheduler, epoch, global_step, args)
                save_checkpoint(args.checkpoint_dir / "latest.pt", model, optimizer, scheduler,
                                 epoch, global_step, args)
                print(f"Saved checkpoint: {epoch_path}")

        print("Done.")

    except KeyboardInterrupt:
        print("\nInterrupted - saving checkpoint before exit...")
        save_checkpoint(args.checkpoint_dir / "interrupted.pt", model, optimizer, scheduler,
                         epoch, global_step, args)
    finally:
        metrics_file.close()


if __name__ == "__main__":
    main()
