"""`FontRecognitionModel`(font_classifier/model.py)을 학습하되, 폰트 인식
헤더에는 `docs/research-paper.md`가 제안한 **동적 후보 선택 기반 Top-k
Relaxed Negative Learning**을 적용하는 학습 스크립트(v2).

배경: 기존 baseline(scripts/train-model-v1.py)은 초/중/종성(글자 구조)
헤더는 잘 수렴하는데 폰트 헤더는 잘 학습되지 않는다. 이름은 다르지만
시각적으로 매우 유사한 폰트가 데이터셋에 섞여 있어, 정답 폰트를 제외한
모든 폰트를 똑같이 억제하는 softmax cross entropy가 지나치게 경직된
지도 신호를 주기 때문으로 본다(research-paper.md 1~2절).

v1과의 차이는 **폰트 손실 하나뿐이다.** 인코더/헤더 구조, K x M 배치
샘플러, 초중종성 손실(cross entropy), AdamW + warmup·cosine 스케줄,
증강, 체크포인트 포맷은 v1과 동일하게 두어 두 방법을 공정하게 비교할
수 있게 한다(사용법/설계 근거는 docs/train-model-v2.md 참고). 바뀌는 부분:

- 폰트 헤더 logit에 softmax 대신 **sigmoid**를 적용해 클래스별 독립
  activation을 쓴다(research-paper.md 2.1절). 모델 구조는 그대로다 -
  `FontHead`의 `Linear` 출력을 손실 함수에서 다르게 해석할 뿐이다.
- 폰트 손실을 curriculum으로 구성한다(research-paper.md 2.4절):
      L_curr(t) = (1 - alpha_t) * L_warm + alpha_t * L_TRN
  warm-up 구간(t < warmup_epochs)에는 alpha_t = 0이라 모든 비정답
  클래스를 음성으로 두는 sigmoid binary 손실(L_warm)만 쓰고, 이후
  ramp_epochs 동안 Top-k Relaxed Negative Loss(L_TRN)의 비중을 선형으로
  키운다.
- L_TRN은 매 스텝 모델의 현재 sigmoid activation에서 정답을 제외한 상위
  k개 클래스를 ambiguous candidate로 골라 negative 손실에서 제외하고,
  정답을 제외한 전체 비정답 activation에 약한 sparsity 정규화를 건다
  (research-paper.md 2.3절).

학습 결과(체크포인트/metrics.jsonl)는 v1과 구분되도록
`data/checkpoints/v2`에 저장한다.

학습/검증 분할은 v1과 마찬가지로 이번에도 없다 - `data/dataset`의 유효한
(폰트, 글자) 전체로 학습하고, 로그의 정확도는 모두 학습 데이터 기준이다
(held-out 지표가 아니다). 자세한 근거는 docs/train-model-v2.md 2.2절 참고.

실행:
    uv run python scripts/train-model-v2.py
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

# v2(제안 방법) 결과는 v1(baseline)과 비교할 수 있도록 checkpoints/v2 하위
# 폴더에 따로 저장한다(scripts/train-model-v1.py는 checkpoints/v1에 저장한다).
CHECKPOINT_DIR = DATASET_DIR.parent / "checkpoints" / "v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train with Top-k Relaxed Negative Learning on the font head "
                    "(research-paper.md); jamo heads keep plain cross entropy.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--resume", type=Path, default=None,
                         help="이어서 학습할 체크포인트 파일 경로")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fonts-per-batch", "-K", type=int, default=DEFAULT_FONTS_PER_BATCH)
    parser.add_argument("--chars-per-font", "-M", type=int, default=DEFAULT_CHARS_PER_FONT)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500,
                         help="학습률(LR) 선형 warmup 스텝 수 - curriculum warm-up"
                              "(--warmup-epochs)과는 다른 개념이다")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-jamo", type=float, default=1.0)
    parser.add_argument("--lambda-font", type=float, default=1.0,
                         help="전체 손실에서 폰트 손실(L_curr) 항에 곱하는 가중치")

    # --- Top-k Relaxed Negative Learning 하이퍼파라미터 (research-paper.md 2.3~2.4절) ---
    parser.add_argument("--topk-k", type=int, default=5,
                         help="정답을 제외하고 negative 손실에서 빼줄 ambiguous "
                              "candidate 개수 k (research-paper.md 2.2절)")
    parser.add_argument("--lambda-neg", type=float, default=1.0,
                         help="relaxed negative loss 강도 lambda "
                              "(L_warm/L_TRN 공통, research-paper.md 2.3~2.4절)")
    parser.add_argument("--beta-sparse", type=float, default=1e-4,
                         help="sparsity 정규화 강도 beta. 정답을 제외한 약 C개 "
                              "클래스 activation의 '합'에 곱하므로 lambda보다 "
                              "훨씬 작게 둔다 (research-paper.md 2.3절)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                         help="curriculum warm-up epoch 수 T_warm. 이 전까지는 "
                              "alpha=0이라 L_warm(단일 라벨)만 쓴다")
    parser.add_argument("--ramp-epochs", type=int, default=10,
                         help="warm-up 이후 alpha를 0->1로 선형 증가시키는 epoch "
                              "수 T_ramp (research-paper.md 2.4절)")

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


def curriculum_alpha(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """research-paper.md 2.4절의 curriculum 계수 alpha_t.

        alpha_t = 0                                  (t < T_warm)
        alpha_t = min(1, (t - T_warm) / T_ramp)      (그 외)

    epoch을 t로 쓴다(논문은 epoch 또는 iteration 둘 다 허용한다). alpha=0
    이면 폰트 손실이 완전히 L_warm(단일 라벨)이고, alpha=1이면 완전히
    L_TRN이다."""

    if epoch < warmup_epochs:
        return 0.0
    return min(1.0, (epoch - warmup_epochs) / max(1, ramp_epochs))


def font_loss_topk_relaxed(
    font_logits: torch.Tensor, font_labels: torch.Tensor, alpha: float,
    k: int, lambda_neg: float, beta_sparse: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """폰트 헤더에 대한 curriculum 손실 L_curr(research-paper.md 2.3~2.4절)을
    배치 평균으로 계산한다.

        L_curr = (1 - alpha) * L_warm + alpha * L_TRN

    - L_pos  = -log sigma(z_y)                              (정답 클래스, 두 손실 공통)
    - L_warm = L_pos + lambda * mean_{c != y} -log(1 - sigma(z_c))
    - L_TRN  = L_pos
               + lambda * mean_{c in N_relaxed} -log(1 - sigma(z_c))
               + beta   * sum_{c != y} sigma(z_c)
      여기서 N_relaxed = (정답 y와 상위 k개 ambiguous candidate)를 제외한 클래스.

    반환값: (L_curr, L_warm, L_TRN) 모두 배치 평균 스칼라. L_warm/L_TRN은
    로그로 두 항의 움직임을 따로 보기 위해 함께 돌려준다(alpha=0이면 L_TRN은
    손실에 기여하지 않지만 참고용으로 계산해 둔다).

    수치 안정성을 위해 log sigma(z) = logsigmoid(z),
    log(1 - sigma(z)) = logsigmoid(-z)를 쓴다."""

    batch_size, num_classes = font_logits.shape
    rows = torch.arange(batch_size, device=font_logits.device)

    log_sig = F.logsigmoid(font_logits)         # log sigma(z)      (양성용)
    log_one_minus = F.logsigmoid(-font_logits)  # log(1 - sigma(z)) (음성용)

    # 정답 y를 제외한 비정답 마스크(1=비정답). 두 손실이 공유한다.
    non_target = torch.ones_like(font_logits)
    non_target[rows, font_labels] = 0.0

    # L_pos: 정답 클래스의 -log sigma(z_y)
    l_pos = -log_sig[rows, font_labels]

    # --- L_warm: 모든 비정답 클래스를 음성으로 취급(단일 라벨 warm-up) ---
    neg_sum_all = -(log_one_minus * non_target).sum(dim=1)
    l_neg_warm = neg_sum_all / max(1, num_classes - 1)
    l_warm = l_pos + lambda_neg * l_neg_warm

    # --- L_TRN: 상위 k개 ambiguous candidate를 negative에서 제외 ---
    # 후보 선택은 현재 activation 기준이며 그래디언트를 흘리지 않는다(집합 선택).
    with torch.no_grad():
        scores = torch.sigmoid(font_logits).clone()
        scores[rows, font_labels] = float("-inf")  # 정답은 후보에서 제외
        # k가 비정답 클래스 수보다 클 수 없도록 방어(작은 합성 데이터셋 대비)
        eff_k = min(k, max(0, num_classes - 1))
        ambiguous = torch.zeros_like(font_logits, dtype=torch.bool)
        if eff_k > 0:
            topk_idx = scores.topk(eff_k, dim=1).indices
            ambiguous.scatter_(1, topk_idx, True)

    # relaxed negative 마스크 = 비정답이면서 ambiguous candidate가 아닌 클래스
    relaxed = non_target * (~ambiguous)
    relaxed_count = relaxed.sum(dim=1).clamp(min=1.0)
    l_neg_relaxed = -(log_one_minus * relaxed).sum(dim=1) / relaxed_count

    # sparsity: 정답을 제외한 전체 비정답 activation의 합(ambiguous 포함)
    l_sparse = (torch.sigmoid(font_logits) * non_target).sum(dim=1)

    l_trn = l_pos + lambda_neg * l_neg_relaxed + beta_sparse * l_sparse

    l_curr = (1.0 - alpha) * l_warm + alpha * l_trn
    return l_curr.mean(), l_warm.mean(), l_trn.mean()


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
    print(f"Top-k Relaxed Negative Learning: k={args.topk_k} lambda={args.lambda_neg} "
          f"beta={args.beta_sparse} warmup_epochs={args.warmup_epochs} "
          f"ramp_epochs={args.ramp_epochs}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = (args.checkpoint_dir / "metrics.jsonl").open("a", encoding="utf-8")

    metric_names = ["loss", "loss_jamo", "loss_font", "loss_font_warm", "loss_font_trn",
                     "cho_acc", "jung_acc", "jong_acc", "syllable_acc", "font_acc",
                     "font_top5_acc"]
    running = {name: RunningAverage() for name in metric_names}
    log_window_start = time.time()
    samples_since_log = 0

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            alpha = curriculum_alpha(epoch, args.warmup_epochs, args.ramp_epochs)
            print(f"epoch {epoch}: curriculum alpha={alpha:.3f} "
                  f"({'warm-up only' if alpha == 0.0 else 'ramping/relaxed'})")
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
                    # 폰트 손실만 v1과 다르다: Top-k Relaxed Negative curriculum
                    # (research-paper.md 2.3~2.4절). 자소 손실은 그대로 CE.
                    loss_font, loss_font_warm, loss_font_trn = font_loss_topk_relaxed(
                        out.font_logits.float(), font_labels, alpha,
                        args.topk_k, args.lambda_neg, args.beta_sparse,
                    )
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
                    # 폰트 top-1은 sigmoid에서도 argmax(logit)과 동일하다.
                    font_correct = out.font_logits.argmax(-1) == font_labels
                    # 폰트 top-5: 정답 폰트가 상위 5개 안에 들면 정답으로 센다. 유사
                    # 폰트를 허용하는 이 방법에서는 top-1보다 이 지표가 학습 진행을
                    # 더 잘 드러낸다(research-paper.md 2.6절, docs/train-model-v2.md 2.5절).
                    top5_k = min(5, out.font_logits.shape[-1])
                    font_top5 = out.font_logits.topk(top5_k, dim=-1).indices
                    font_top5_correct = (font_top5 == font_labels.unsqueeze(1)).any(dim=1)

                batch_size = images.shape[0]
                running["loss"].update(loss.item(), batch_size)
                running["loss_jamo"].update(loss_jamo.item(), batch_size)
                running["loss_font"].update(loss_font.item(), batch_size)
                running["loss_font_warm"].update(loss_font_warm.item(), batch_size)
                running["loss_font_trn"].update(loss_font_trn.item(), batch_size)
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
                        "epoch": epoch, "step": global_step, "alpha": alpha,
                        "lr": scheduler.get_last_lr()[0], "samples_per_sec": samples_per_sec,
                        **{name: avg.value for name, avg in running.items()},
                    }
                    print(
                        f"epoch {epoch} step {global_step} "
                        f"loss={record['loss']:.4f} (jamo={record['loss_jamo']:.4f} "
                        f"font={record['loss_font']:.4f} "
                        f"[warm={record['loss_font_warm']:.4f} trn={record['loss_font_trn']:.4f}]) "
                        f"acc[cho={record['cho_acc']:.3f} jung={record['jung_acc']:.3f} "
                        f"jong={record['jong_acc']:.3f} syllable={record['syllable_acc']:.3f} "
                        f"font={record['font_acc']:.3f} font_top5={record['font_top5_acc']:.3f}] "
                        f"alpha={alpha:.2f} lr={record['lr']:.2e} "
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
