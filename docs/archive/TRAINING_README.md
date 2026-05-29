# Training README — 학습 실행 방법

## 스크립트 구조

```
scripts/train_seq_classifier.py      ← 셀 1개 학습 (model × scenario × T)
scripts/sweep_classifiers.py         ← 위 스크립트를 자동으로 반복 실행 (sweep 전체)
scripts/analyze_classifier_sweep.py  ← 결과 취합 → 그래프 저장
src/seq_classifier.py                ← 모델 정의 (RNN / GRU / Transformer)
```

---

## 1. 셀 1개 학습 — `train_seq_classifier.py`

가장 기본 단위. 모델 1개 + 시나리오 1개 + T 1개 조합을 학습한다.

```bash
python scripts/train_seq_classifier.py \
  --model gru \          # rnn / gru / transformer
  --scenario r2 \        # r1 / r2 / r3b
  --T 512                # 시퀀스 길이 (데이터셋 T_max=1000 이하면 자유)
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | 필수 | `rnn` / `gru` / `transformer` |
| `--scenario` | 필수 | `r1` / `r2` / `r3b` |
| `--T` | 필수 | 시퀀스 길이 (데이터셋 T_max=1000 이하) |
| `--epochs` | 30 | 최대 epoch 수 |
| `--patience` | 8 | val accuracy 개선 없으면 조기 종료 |
| `--lr` | 3e-3 | learning rate (AdamW) |
| `--batch-size` | 64 | 배치 크기 |
| `--device` | auto | `auto` / `cpu` / `mps` / `cuda` |
| `--seed` | 0 | 재현성 시드 |

### Transformer 전용 추가 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--d-model` | 64 | 임베딩 차원 |
| `--n-layers` | 1 | Transformer encoder 층 수 |
| `--n-heads` | 4 | attention head 수 |

### 출력 파일

결과는 `data/analysis/9_classifier/{scenario}/{model}_T{T}/` 에 저장된다.

| 파일 | 내용 |
|---|---|
| `model.pt` | val accuracy 최고 시점의 체크포인트 |
| `metrics.json` | epoch별 loss/acc + 테스트 정확도 + 클래스별 정확도 |
| `confusion.npy` | (24×24) 혼동 행렬 |
| `args.json` | 실행 시 사용한 파라미터 스냅샷 |

---

## 2. 전체 sweep — `sweep_classifiers.py`

model × scenario × T 조합 전부를 자동으로 순차 실행한다.

```bash
python scripts/sweep_classifiers.py \
  --models rnn gru transformer \
  --scenarios r1 r2 r3b \
  --T 128 512 \
  --epochs 50 \
  --patience 12 \
  --device auto
```

- `metrics.json`이 이미 있는 셀은 자동으로 **skip**
- `--force` 옵션을 붙이면 이미 완료된 셀도 **강제 재실행**
- 완료 후 `data/analysis/9_classifier/sweep_summary.csv` 자동 생성

---

## 3. 결과 시각화 — `analyze_classifier_sweep.py`

sweep 완료 후 그래프 5장을 생성한다.

```bash
python scripts/analyze_classifier_sweep.py \
  --scenarios r1 r2 r3b \
  --models rnn gru transformer \
  --T 128 512
```

### 생성되는 파일 (`data/analysis/9_classifier/`)

| 파일 | 내용 |
|---|---|
| `training_curves_loss.png` | epoch별 train/val loss 곡선 (scenarios × models 격자) |
| `training_curves_acc.png` | epoch별 train/val accuracy 곡선 (scenarios × models 격자) |
| `accuracy_vs_T.png` | T값에 따른 테스트 정확도 비교 |
| `per_class_bars.png` | 클래스별 정확도 막대그래프 |
| `confusion_grid.png` | 혼동 행렬 그리드 |

---

## 4. 모델 구조 (`src/seq_classifier.py`)

| 모델 | 파라미터 수 | 구조 |
|---|---|---|
| `VanillaRNN` | ~28K | 1-layer RNN, hidden=128 |
| `GRUClassifier` | ~78K | 1-layer GRU, hidden=128 |
| `TransformerClassifier` | ~233K | 2-layer Transformer, d_model=64, 4 heads |

입력 형태: `(batch, T, 8)` uint8 → `float32` 변환 후 linear projection → 각 모델 처리 → 24-class softmax

---

## 5. 학습 흐름 요약

```
데이터 로드: syndromes[:, :T, :]  (T는 자유롭게 선택)
    ↓
AdamW + CosineAnnealingLR 로 학습
    ↓
매 epoch마다 val accuracy 체크
    ↓
개선 없으면 patience 차감 → 0이 되면 early stop
    ↓
best val checkpoint 복원 → test set 평가
    ↓
metrics.json / confusion.npy / model.pt 저장
```

---

## 6. 주의사항 (Apple Silicon MPS)

Apple M 시리즈에서 실행할 때 알려진 이슈:

- `y.to(device, non_blocking=True)` 로 label을 MPS에 올리면 **비동기 전송 타이밍 문제**로 accuracy 계산이 완전히 틀릴 수 있음
- 현재 코드는 이 버그를 수정한 상태 — label(`y`)은 CPU에 유지, `y_dev = y.to(device)` (blocking) 는 loss 계산에만 사용, accuracy는 `logits.argmax().cpu() == y` 로 CPU에서 비교
- epoch당 소요 시간이 수백 초로 불규칙하게 튀는 현상이 있을 수 있음 (MPS 내부 스케줄링)

---

## 7. 실험 결과 요약 (18-cell sweep, epochs=50, patience=12)

| 시나리오 | 모델 | T | test accuracy | 비고 |
|---|---|---|---|---|
| R1 | RNN | 128/512 | ~4% | gradient vanishing, 랜덤 수준 |
| R1 | GRU | 128 | 41.8% | 학습 됨 |
| R1 | GRU | **512** | **73.4%** | R1 이론 한계 75% 근접 |
| R1 | Transformer | 128 | 8.7% | 학습 불안정 |
| R1 | Transformer | 512 | 51.2% | |
| R2 | RNN | 128/512 | 4~7% | gradient vanishing |
| R2 | GRU | 128 | 60.5% | |
| R2 | GRU | **512** | **95.3%** | **핵심 결과 — R1 한계 돌파** |
| R2 | Transformer | 128 | 6.7% | 학습 불안정 |
| R2 | Transformer | 512 | 4.6% | 실패 (LR 과다 추정) |
| R3b | RNN | 128/512 | ~4% | gradient vanishing |
| R3b | GRU | 128/512 | 진행 중 | |
| R3b | Transformer | 128/512 | 진행 중 | |

- **RNN**: 모든 시나리오에서 학습 실패 — 긴 시퀀스에서 gradient vanishing
- **GRU**: T가 클수록 성능 상승, R2+T512에서 95.3% 달성
- **Transformer**: 현재 설정(LR=3e-3)에서 불안정 — 추후 LR 조정 필요
