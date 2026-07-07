"""학습된 폰트 인식 체크포인트를 데이터셋 전체에 대해 평가하고, 한글/폰트
인식 성능과 인식 속도를 측정해 `data/results/eval.json`에 기록하는 스크립트.

`scripts/train-model.py`와 같은 `FontGlyphDataset`(폰트 중심 data/dataset)을
읽되, 학습과 달리 augmentation을 끄고(깨끗한 입력) 데이터 전체를 한 번씩
순회한다. 지표 정의는 train-model.py의 학습 로그와 맞춘다.

- 한글: 초/중/종성 개별 정확도와 음절 정확도(셋 다 맞음, argmax 기준),
  그리고 실제 디코딩 정확도 — 제한 디코딩(2,350자 표)과 개방 디코딩
  (11,172자)의 글자 정확도.
- 폰트: top-1 / top-5 / top-10 정확도.
- 속도: 한 글자를 인식(encode + 디코딩 + 폰트 top-k)하는 데 걸리는 시간
  (samples/s, ms/sample). 데이터 로딩·정답 대조 시간은 제외하고 순수 인식
  연산만 측정하며, 첫 배치는 워밍업으로 빼고 CUDA는 동기화해 측정한다.

`data/dataset`의 폰트 순서(index.json의 `id`)와 체크포인트의 폰트 클래스
순서가 일치해야 폰트 지표가 유의미하다(둘 다 `id - 1`을 라벨로 쓴다).
폰트 클래스 수가 데이터셋과 다르면 그 사실을 결과에 함께 남긴다.

실행:
    uv run python scripts/eval-model.py
    uv run python scripts/eval-model.py --checkpoint data/checkpoints/checkpoint-epoch-0003.pt
    uv run python scripts/eval-model.py --dataset-dir data/dataset --sample-percent 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from font_classifier.dataset_loader import (
    DATASET_DIR, DEFAULT_PRESCAN_WORKERS, FontGlyphDataset,
)
from font_classifier.font_dataset import HANGUL_TABLE
from font_classifier.model import (
    FontRecognitionModel, decode_open, decode_restricted,
)

CHECKPOINT_DIR = DATASET_DIR.parent / "checkpoints"
RESULTS_DIR = DATASET_DIR.parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a font-recognition checkpoint (hangul/font accuracy + speed).")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "latest.pt",
                         help="평가할 체크포인트 .pt 파일 (기본: data/checkpoints/latest.pt)")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR,
                         help="평가 대상 데이터셋 폴더 (기본: data/dataset)")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "eval.json",
                         help="결과 JSON 경로 (기본: data/results/eval.json)")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prescan-workers", type=int, default=DEFAULT_PRESCAN_WORKERS)
    parser.add_argument("--sample-percent", type=float, default=100.0,
                         help="평가에 사용할 데이터셋 비율(0 초과 100 이하, 기본: 100)")

    parser.add_argument("--device", default=None,
                         help="기본값: cuda가 있으면 cuda, 없으면 cpu")
    parser.add_argument("--no-amp", action="store_true",
                         help="bfloat16 자동 혼합 정밀도를 끈다(기본은 cuda에서 켜짐)")
    parser.add_argument("--log-every", type=int, default=1000,
                         help="이 배치 수마다 진행 상황을 출력한다")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> tuple[FontRecognitionModel, int]:
    """체크포인트에서 모델을 복원한다. 폰트 클래스 수는 폰트 분류기 가중치
    크기에서 직접 읽는다(train-model.py의 저장 형식과 같은 `model` 키)."""

    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model"]
    num_classes = state["font_head.classifier.weight"].shape[0]
    model = FontRecognitionModel(num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, num_classes


class Counter:
    """정답 개수/전체 개수를 세어 비율을 내는 작은 누적기."""

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, correct: int, total: int) -> None:
        self.correct += correct
        self.total += total

    @property
    def acc(self) -> float:
        return self.correct / max(1, self.total)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.sample_percent <= 100.0):
        raise SystemExit("--sample-percent는 0보다 크고 100 이하여야 합니다.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device: {device} | mixed precision (bf16): {'on' if use_amp else 'off'}")

    if not args.checkpoint.exists():
        raise SystemExit(f"체크포인트가 없습니다: {args.checkpoint}")
    print(f"Loading checkpoint {args.checkpoint} ...")
    model, num_classes = load_model(args.checkpoint, device)

    print(f"Loading dataset from {args.dataset_dir} ...")
    dataset = FontGlyphDataset(
        args.dataset_dir, augment=False, prescan_workers=args.prescan_workers)
    total_samples = len(dataset)
    print(f"{total_samples} valid sample(s) across {dataset.num_font_classes} font(s)")
    if num_classes != dataset.num_font_classes:
        print(f"[WARNING] 체크포인트의 폰트 클래스 수({num_classes})와 데이터셋의 "
              f"폰트 수({dataset.num_font_classes})가 다릅니다 - 폰트 지표가 "
              "왜곡될 수 있습니다(체크포인트와 데이터셋의 정합성 확인 필요).")

    eval_dataset = dataset
    if args.sample_percent < 100.0:
        sample_count = max(1, int(total_samples * (args.sample_percent / 100.0)))
        sampled_indices = torch.randperm(total_samples)[:sample_count].tolist()
        eval_dataset = Subset(dataset, sampled_indices)
        print(f"Randomly sampled {sample_count} / {total_samples} sample(s) "
              f"({args.sample_percent:.2f}%) for evaluation")
    else:
        sample_count = total_samples

    loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    cho_c, jung_c, jong_c = Counter(), Counter(), Counter()
    syllable_c = Counter()
    restricted_c, open_c = Counter(), Counter()
    font_top1, font_top5, font_top10 = Counter(), Counter(), Counter()

    infer_seconds = 0.0
    timed_samples = 0
    seen = 0
    total_target = sample_count
    wall_start = time.time()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        cho_labels = batch["cho_label"].to(device, non_blocking=True)
        jung_labels = batch["jung_label"].to(device, non_blocking=True)
        jong_labels = batch["jong_label"].to(device, non_blocking=True)
        font_labels = batch["font_label"].to(device, non_blocking=True)
        true_chars = [HANGUL_TABLE[i] for i in batch["char_index"].tolist()]
        batch_size = images.shape[0]

        # --- 인식(순수 연산) 구간: encode + 디코딩 + 폰트 top-k ---
        synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model.encode(images)
            restricted = decode_restricted(out.cho_logits, out.jung_logits, out.jong_logits)
            open_pred = decode_open(out.cho_logits, out.jung_logits, out.jong_logits)
            top10_k = min(10, num_classes)
            font_top = out.font_logits.topk(top10_k, dim=-1).indices
        synchronize()
        dt = time.perf_counter() - t0
        # 첫 배치는 워밍업(커널 컴파일/캐시 예열)이라 속도 측정에서 뺀다.
        if batch_idx > 0:
            infer_seconds += dt
            timed_samples += batch_size

        # --- 정확도 집계(측정 구간 밖) ---
        cho_ok = out.cho_logits.argmax(-1) == cho_labels
        jung_ok = out.jung_logits.argmax(-1) == jung_labels
        jong_ok = out.jong_logits.argmax(-1) == jong_labels
        cho_c.update(int(cho_ok.sum()), batch_size)
        jung_c.update(int(jung_ok.sum()), batch_size)
        jong_c.update(int(jong_ok.sum()), batch_size)
        syllable_c.update(int((cho_ok & jung_ok & jong_ok).sum()), batch_size)

        restricted_c.update(sum(p == t for p, t in zip(restricted, true_chars)), batch_size)
        open_c.update(sum(p == t for p, t in zip(open_pred, true_chars)), batch_size)

        labels_col = font_labels.unsqueeze(1)
        font_top1.update(int((font_top[:, :1] == labels_col).any(1).sum()), batch_size)
        font_top5.update(int((font_top[:, :min(5, top10_k)] == labels_col).any(1).sum()), batch_size)
        font_top10.update(int((font_top == labels_col).any(1).sum()), batch_size)

        seen += batch_size
        if (batch_idx + 1) % args.log_every == 0:
            print(f"  [{seen}/{total_target} ({min(100.0, 100.0 * seen / max(1, total_target)):.2f}%)] "
                  f"syllable={syllable_c.acc:.3f} restricted={restricted_c.acc:.3f} "
                  f"font_top1={font_top1.acc:.3f} font_top5={font_top5.acc:.3f}")


    wall_seconds = time.time() - wall_start
    samples_per_sec = timed_samples / infer_seconds if infer_seconds > 0 else 0.0

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "device": str(device),
        "amp_bf16": use_amp,
        "num_font_classes_checkpoint": num_classes,
        "num_font_classes_dataset": dataset.num_font_classes,
        "num_samples_evaluated": seen,
        "sample_percent": args.sample_percent,
        "hangul": {
            "cho_acc": cho_c.acc,
            "jung_acc": jung_c.acc,
            "jong_acc": jong_c.acc,
            "syllable_acc": syllable_c.acc,
            "restricted_char_acc": restricted_c.acc,
            "open_char_acc": open_c.acc,
        },
        "font": {
            "top1_acc": font_top1.acc,
            "top5_acc": font_top5.acc,
            "top10_acc": font_top10.acc,
        },
        "speed": {
            "batch_size": args.batch_size,
            "timed_samples": timed_samples,
            "inference_seconds": infer_seconds,
            "samples_per_second": samples_per_sec,
            "ms_per_sample": (1000.0 * infer_seconds / timed_samples) if timed_samples else 0.0,
            "note": "encode+디코딩+폰트 top-k 순수 연산 시간(첫 배치 워밍업 제외, "
                    "데이터 로딩/정답 대조 제외)",
        },
        "wall_seconds": wall_seconds,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Evaluation result ===")
    print(f"samples: {seen}")
    print(f"hangul : cho={cho_c.acc:.4f} jung={jung_c.acc:.4f} jong={jong_c.acc:.4f} "
          f"syllable={syllable_c.acc:.4f}")
    print(f"         restricted_char={restricted_c.acc:.4f} open_char={open_c.acc:.4f}")
    print(f"font   : top1={font_top1.acc:.4f} top5={font_top5.acc:.4f} top10={font_top10.acc:.4f}")
    print(f"speed  : {samples_per_sec:.1f} samples/s "
          f"({result['speed']['ms_per_sample']:.3f} ms/sample)")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
