# Classifier Dataset — 데이터셋 설명

## 과제 정의 (Task #6)

> **syndrome sequence를 보고, 어떤 CNOT 게이트에 노이즈가 집중되어 있는지 맞추는 24-class 분류 문제**

- 회로에 CNOT 게이트가 24개 있음 (인덱스 0~23)
- 그 중 딱 1개가 "dominant noisy gate" (오류율 `p_high = 0.1`)
- 나머지 23개는 배경 노이즈 (`p_bg = 0.01`)
- 모델은 syndrome sequence만 보고 어떤 게이트가 noisy한지 맞춰야 함

---

## 파일 목록

| 파일명 | 내용 | 크기 |
|---|---|---|
| `r1_train.npz` | R1 학습 데이터 | ~31 MB |
| `r1_val.npz`   | R1 검증 데이터 | ~7.7 MB |
| `r1_test.npz`  | R1 테스트 데이터 | ~7.7 MB |
| `r1_meta.json` | R1 생성 파라미터 요약 | ~160 B |
| `r2_*`         | R2 시나리오 (동일 구조) | 동일 |
| `r3b_*`        | R3b 시나리오 (동일 구조) | 동일 |

---

## 배열 구조 (모든 시나리오 동일)

각 `.npz` 파일에는 3개의 key가 있다:

```
syndromes : (N, 1000, 8)  uint8   ← syndrome sequence
labels    : (N,)           int64   ← dominant CNOT 인덱스 (0~23)
meta      : ()             str     ← 생성 파라미터 JSON 문자열 (embedded)
```

| 차원 | 의미 |
|---|---|
| `N` | 샘플 수 — train=48,000 / val=12,000 / test=12,000 |
| `1000` | 시간축 — 최대 T=1000 라운드. 실제 학습 시 앞 T개만 잘라 사용 |
| `8` | 공간축 — 8개 syndrome 측정값 (각 비트: 0 또는 1) |

> 학습 시 `syndromes[:, :T, :]` 로 잘라서 사용하면 됨 (T는 자유롭게 선택)

클래스당 샘플 수는 균등하게 맞춰져 있음:
- train: 클래스당 2,000개 × 24클래스 = 48,000
- val: 클래스당 500개 × 24클래스 = 12,000
- test: 클래스당 500개 × 24클래스 = 12,000

---

## 시나리오별 차이

세 시나리오는 **같은 회로·같은 라벨**이지만, **적용하는 디코더가 다름**.
디코더가 syndrome에 feedback을 가하므로 시퀀스 패턴이 완전히 달라진다.

| 시나리오 | 파일 prefix | 디코더 | 설명 |
|---|---|---|---|
| **R1** | `r1_*` | `IdentityDecoder` | 디코더 없음. 순수 syndrome만 관찰. 인접 게이트 쌍이 구조적으로 구분 불가 (R1 ambiguity) |
| **R2** | `r2_*` | `PhenomDecoder` | Phenomenological(현상론적) 디코더 적용. 매 라운드 단순 다수결 기반 correction 수행 |
| **R3b** | `r3b_*` | `LookupDecoder` | 회로 수준 lookup table 디코더 적용. 가장 현실적인 QEC 시나리오 |

### 왜 시나리오가 중요한가?

- **R1**: 이론적 분류 한계가 75% — 인접 게이트 쌍 4그룹 `{6,7}`, `{14,15,17}`, `{18,19,20}`, `{22,23}` 이 syndrome 패턴상 구분 불가 (R1 ambiguity)
- **R2**: 디코더가 가하는 correction이 인접 게이트별로 달라져서 오히려 **구분 가능한 residual pattern** 생성 → GRU T=512에서 **95.3% 달성** (R1 한계 돌파)
- **R3b**: 회로 레벨 디코더 적용 → 구분에 필요한 시퀀스가 가장 길다 (분석상 최소 T≈2174 필요)

### R1 Ambiguity Groups (R1 이론 한계의 원인)

아래 그룹 내 CNOT들은 R1에서 syndrome 분포가 이론적으로 동일하여 구분 불가:

| 그룹 | CNOT 인덱스 |
|---|---|
| G1 | 6, 7 |
| G2 | 14, 15, 17 |
| G3 | 18, 19, 20 |
| G4 | 22, 23 |

---

## 생성 파라미터 (`*_meta.json` 내용)

```json
{
  "scenario": "r1",       // r1 / r2 / r3b
  "decoder": "identity",  // identity / phenom / lookup
  "p_bg": 0.01,           // 배경 CNOT 오류율
  "p_high": 0.1,          // dominant CNOT 오류율
  "T_max": 1000,          // 저장된 최대 라운드 수
  "n_train": 2000,        // 클래스당 train 샘플 수
  "n_val": 500,           // 클래스당 val 샘플 수
  "n_test": 500,          // 클래스당 test 샘플 수
  "seed": 0
}
```

---

## 데이터 불러오기 예시

```python
import numpy as np

data = np.load("r2_train.npz")
syndromes = data["syndromes"]   # (48000, 1000, 8) uint8
labels    = data["labels"]      # (48000,) int64

# 시퀀스 길이 T로 자르기 (예: T=512)
T = 512
X = syndromes[:, :T, :].astype(np.float32)   # (48000, 512, 8)
y = labels                                    # (48000,)
```

---

## 재생성 방법

데이터셋이 없으면 스크립트로 다시 만들 수 있다:

```bash
# R1, R3b
python scripts/generate_classifier_dataset.py --scenarios r1 r3b

# R2 (별도 스크립트)
python scripts/generate_r2_dataset.py
```

생성 시간 (Apple M 시리즈 기준): R1·R3b 각 약 10~20분, R2 약 30분

---

## 관련 분석 결과

학습 결과는 `data/analysis/9_classifier/` 에 저장되며, 분석 스크립트는 아래를 참고:

- **학습 스크립트**: `scripts/train_seq_classifier.py`
- **전체 sweep 실행**: `scripts/sweep_classifiers.py`
- **결과 시각화**: `scripts/analyze_classifier_sweep.py`
- **모델 정의**: `src/seq_classifier.py` (VanillaRNN / GRUClassifier / TransformerClassifier)
