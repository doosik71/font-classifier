r"""폰트 헤더 A/B 벤치마크: 현재 `FontHead`(선형 분류기)와 cosine+scale
헤더 중 어느 쪽이 **font loss를 더 빨리 줄이는지**를 같은 조건에서 나란히
측정한다.

배경(docs 논의 요약): jamo에 비해 font loss가 느리게 떨어지는 원인은
가중치 불균형이나 임베딩 차원이 아니라 3,480개 세밀 클래스라는 본질적
난이도로 보인다. 그걸 직접 겨냥하는 후보가 "cosine 분류기 + 학습 가능한
scale"(CosFace 계열)이다. 이 스크립트는 그 가설을 실제 학습 곡선으로
검증하기 위한 것이다.

설계 원칙 - **기존 모듈은 전혀 수정하지 않는다**:
- Variant A: `font_classifier.model.HangulFontRecognitionModel` 그대로.
- Variant B: 위 모델을 이 스크립트에서 subclass해 `font_head`만
  `CosineScaleFontHead`로 교체한다. 그 헤더는 model.py의 `FontHead.mlp`
  (동일한 MLP 정의)를 **그대로 재사용**하고, 마지막 선형 분류기만
  `logit = s * normalize(feat)·normalize(W)`로 바꾼다. 즉 실험 변수는
  "마지막 분류기의 형태" 하나뿐이다.
- 데이터셋 로더(`FontGlyphDataset`)와 손실/지표 계산은 train-model.py와
  똑같이 쓴다.

공정성 보장:
- 두 모델을 같은 seed로 만들어 encoder/content/style_proj/hangul_head/
  decoder의 초기 가중치가 완전히 동일하게 시작한다(차이는 font_head뿐).
- 매 스텝 **같은 배치**를 두 모델에 함께 먹인다(데이터 스트림 동일, 헤더만
  다름).

주의: 두 모델을 동시에 올리므로 메모리/연산이 baseline 학습의 약 2배다.
빠른 비교가 목적이므로 `--steps`로 짧게(기본 2,000 스텝) 돌린다.

실행:
    uv run python scripts/cosine-scale-benchmark.py
    uv run python scripts/cosine-scale-benchmark.py --steps 3000 --batch-size 128
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
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.dataset_loader import (
    DATASET_DIR, DEFAULT_PRESCAN_WORKERS, FontGlyphDataset,
)
from font_classifier.model import (
    STYLE_DIM, FontHead, HangulFontRecognitionModel,
)

RESULTS_DIR = DATASET_DIR.parent / "results"
DEFAULT_OUT = RESULTS_DIR / "cosine-scale-benchmark.jsonl"


class CosineScaleFontHead(nn.Module):
    """Variant B의 폰트 헤더. model.py의 `FontHead`와 동일한 MLP를 그대로
    재사용하되(정의를 복제하지 않는다), 마지막 선형 분류기만 cosine + 학습
    가능한 scale로 바꾼다.

        feat  = normalize(mlp(style))     # 특징 L2 정규화
        W_i   = normalize(weight)         # 클래스 벡터 L2 정규화
        logit = s * (feat · W_i)          # s = exp(log_scale), 학습 가능

    scale은 로그공간 파라미터로 두어 항상 양수를 유지한다. cosine 분류기는
    logit 크기를 s 하나로 조절할 수 있어, 정규화된 특징 위에서도 3,000개가
    넘는 클래스를 날카롭게 분리하도록 softmax를 스케일링한다."""

    def __init__(self, num_font_classes: int, style_dim: int = STYLE_DIM,
                 hidden_dim: int = 1024, dropout: float = 0.1,
                 init_scale: float = 16.0) -> None:
        super().__init__()
        # model.py의 FontHead를 하나 만들어 그 MLP만 떼어 쓴다 - MLP 구조/정의가
        # baseline과 완전히 동일함을 보장하고, 정의를 두 곳에 복제하지 않는다.
        self.mlp = FontHead(num_font_classes, style_dim, hidden_dim, dropout).mlp
        self.weight = nn.Parameter(torch.empty(num_font_classes, style_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        # _init_weights(model.py)의 관례(선형층 bias=0)에 맞춰 MLP bias를 0으로.
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, style: Tensor) -> Tensor:
        feat = F.normalize(self.mlp(style), dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return self.log_scale.exp() * F.linear(feat, weight)


class CosineScaleModel(HangulFontRecognitionModel):
    """baseline 모델과 모든 것이 같고 `font_head`만 cosine+scale로 바꾼
    변형. 기존 모델을 수정하지 않고 subclass로만 교체한다."""

    def __init__(self, num_font_classes: int, style_hidden_dim: int = 1024,
                 init_scale: float = 16.0) -> None:
        super().__init__(num_font_classes, style_hidden_dim=style_hidden_dim)
        # super().__init__()이 끝난 뒤(공유 부분 초기화 완료) 헤더만 교체한다.
        self.font_head = CosineScaleFontHead(
            num_font_classes, style_dim=STYLE_DIM,
            hidden_dim=style_hidden_dim, init_scale=init_scale,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B benchmark: baseline FontHead vs cosine+scale head.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="스텝별 A/B 지표를 append할 jsonl 경로")
    parser.add_argument("--steps", type=int, default=2000,
                        help="비교할 학습 스텝 수(짧게 유지)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-jamo", type=float, default=1.0)
    parser.add_argument("--lambda-font", type=float, default=1.0)
    parser.add_argument("--init-scale", type=float, default=16.0,
                        help="cosine 헤더의 초기 scale s")
    parser.add_argument("--style-hidden-dim", type=int, default=1024)
    parser.add_argument("--prescan-workers", type=int, default=DEFAULT_PRESCAN_WORKERS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None,
                        help="기본값: cuda가 있으면 cuda, 없으면 cpu")
    parser.add_argument("--no-amp", action="store_true",
                        help="bfloat16 AMP를 끈다(기본은 cuda에서 켜짐)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_lambda(warmup_steps: int):
    """선형 warmup 후 상수 lr. 짧은 벤치마크에서는 기울기 비교가 목적이라
    cosine decay 대신 상수를 쓴다(두 모델에 동일하게 적용)."""

    def lr_lambda(step: int) -> float:
        return min(1.0, step / max(1, warmup_steps))

    return lr_lambda


class Window:
    """log_every 구간의 합/개수를 모으는 소형 누적기(train-model.py의
    RunningAverage와 같은 역할)."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def value(self) -> float:
        return self.total / max(1, self.count)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def make_optimizer(model: nn.Module, args: argparse.Namespace):
    return torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)


def train_step(model, optimizer, scheduler, batch, device, use_amp,
               args) -> dict[str, float]:
    """한 배치로 한 스텝 학습하고 그 스텝의 지표를 돌려준다. 두 variant가
    같은 함수를 쓰므로 손실/최적화 절차가 완전히 동일하다."""

    images = batch["image"].to(device, non_blocking=True)
    cho = batch["cho_label"].to(device, non_blocking=True)
    jung = batch["jung_label"].to(device, non_blocking=True)
    jong = batch["jong_label"].to(device, non_blocking=True)
    font = batch["font_label"].to(device, non_blocking=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        out = model.encode(images)
        loss_jamo = (F.cross_entropy(out.cho_logits, cho)
                     + F.cross_entropy(out.jung_logits, jung)
                     + F.cross_entropy(out.jong_logits, jong))
        loss_font = F.cross_entropy(out.font_logits, font)
        loss = args.lambda_jamo * loss_jamo + args.lambda_font * loss_font

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        syllable = ((out.cho_logits.argmax(-1) == cho)
                    & (out.jung_logits.argmax(-1) == jung)
                    & (out.jong_logits.argmax(-1) == jong)).float().mean().item()
        font_acc = (out.font_logits.argmax(-1) == font).float().mean().item()
        top5_k = min(5, out.font_logits.shape[-1])
        top5 = out.font_logits.topk(top5_k, dim=-1).indices
        font_top5 = (top5 == font.unsqueeze(1)).any(dim=1).float().mean().item()

    return {
        "loss": loss.item(),
        "loss_jamo": loss_jamo.item(),
        "loss_font": loss_font.item(),
        "syllable_acc": syllable,
        "font_acc": font_acc,
        "font_top5_acc": font_top5,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device: {device} | AMP(bf16): {'on' if use_amp else 'off'}")

    print(f"Loading dataset from {args.dataset_dir} ...")
    dataset = FontGlyphDataset(args.dataset_dir, augment=True,
                               prescan_workers=args.prescan_workers)
    num_font_classes = dataset.num_font_classes
    print(f"{len(dataset)} sample(s) across {num_font_classes} font(s)")

    # 데이터 순서를 seed로 고정해 두 모델이 정확히 같은 배치 스트림을 본다.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, generator=loader_generator,
    )

    # 같은 seed로 두 모델을 만들어 공유 부분(encoder/content/style_proj/
    # hangul_head/decoder)의 초기 가중치를 동일하게 맞춘다. 차이는 font_head뿐.
    set_seed(args.seed)
    model_a = HangulFontRecognitionModel(
        num_font_classes, style_hidden_dim=args.style_hidden_dim).to(device)
    set_seed(args.seed)
    model_b = CosineScaleModel(
        num_font_classes, style_hidden_dim=args.style_hidden_dim,
        init_scale=args.init_scale).to(device)

    variants = {
        "A_baseline": {"model": model_a},
        "B_cosine_scale": {"model": model_b},
    }
    for name, v in variants.items():
        v["opt"] = make_optimizer(v["model"], args)
        v["sched"] = torch.optim.lr_scheduler.LambdaLR(
            v["opt"], build_lr_lambda(args.warmup_steps))
        v["window"] = {k: Window() for k in
                       ["loss", "loss_jamo", "loss_font", "syllable_acc",
                        "font_acc", "font_top5_acc"]}
        v["font_acc_reached"] = {}  # threshold -> step

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_file = args.out.open("a", encoding="utf-8")

    print(f"Benchmarking {args.steps} step(s), batch={args.batch_size}, "
          f"logging every {args.log_every} to {args.out}")
    window_start = time.time()
    samples_since_log = 0
    step = 0
    try:
        while step < args.steps:
            for batch in loader:
                if step >= args.steps:
                    break
                bsz = batch["image"].shape[0]
                for name, v in variants.items():
                    metrics = train_step(v["model"], v["opt"], v["sched"], batch,
                                          device, use_amp, args)
                    for key, window in v["window"].items():
                        window.update(metrics[key], bsz)
                    for thr in thresholds:
                        if thr not in v["font_acc_reached"] and metrics["font_acc"] >= thr:
                            v["font_acc_reached"][thr] = step
                samples_since_log += bsz
                step += 1

                if step % args.log_every == 0:
                    elapsed = max(1e-6, time.time() - window_start)
                    record = {"step": step, "samples_per_sec": samples_since_log / elapsed}
                    for name, v in variants.items():
                        for key, window in v["window"].items():
                            record[f"{name}.{key}"] = window.value
                            window.reset()
                    record["B_cosine_scale.scale"] = float(
                        model_b.font_head.log_scale.exp().item())
                    a_font = record["A_baseline.loss_font"]
                    b_font = record["B_cosine_scale.loss_font"]
                    print(
                        f"step {step:5d} | font_loss A={a_font:.4f} B={b_font:.4f} "
                        f"(B-A={b_font - a_font:+.4f}) | "
                        f"font_acc A={record['A_baseline.font_acc']:.3f} "
                        f"B={record['B_cosine_scale.font_acc']:.3f} | "
                        f"scale={record['B_cosine_scale.scale']:.2f} | "
                        f"({record['samples_per_sec']:.0f} samples/s)"
                    )
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                    window_start = time.time()
                    samples_since_log = 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        out_file.close()

    print("\n=== Summary: steps to reach training font top-1 accuracy ===")
    header = "threshold  " + "  ".join(f"{name:>16}" for name in variants)
    print(header)
    for thr in thresholds:
        cells = []
        for v in variants.values():
            reached = v["font_acc_reached"].get(thr)
            cells.append(f"{reached:>16}" if reached is not None else f"{'-':>16}")
        print(f"  {thr:>6.2f}   " + "  ".join(cells))
    print("\n(작을수록 좋음: 같은 스텝에서 더 낮은 font_loss / 더 빨리 도달한 쪽이 유리)")


if __name__ == "__main__":
    main()
