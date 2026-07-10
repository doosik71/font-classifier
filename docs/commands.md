# 명령 모음

## 모델 학습

```bash
bin/train-model.sh --checkpoint-dir data/checkpoints/256-512 --style-dim 256 --style-hidden-dim 512 --device cuda:0 --early-stop
bin/train-model.sh --checkpoint-dir data/checkpoints/512-128 --style-dim 512 --style-hidden-dim 128 --device cuda:1 --early-stop
bin/train-model.sh --checkpoint-dir data/checkpoints/512-256 --style-dim 512 --style-hidden-dim 256 --device cuda:2 --early-stop
```

## 모델 평가

```bash
bin/eval-model.sh --style-dim 256 --style-hidden-dim 512 --device cuda:0 --checkpoint data/checkpoints/256-512/best.pt --output data/checkpoints/256-512/eval.json
bin/eval-model.sh --style-dim 512 --style-hidden-dim 128 --device cuda:1 --checkpoint data/checkpoints/512-128/best.pt --output data/checkpoints/512-128/eval.json
bin/eval-model.sh --style-dim 512 --style-hidden-dim 256 --device cuda:2 --checkpoint data/checkpoints/512-256/best.pt --output data/checkpoints/512-256/eval.json
```
