# Correlated-Noise-Estimation

`d=3` rotated surface code에서 **특정 CNOT 게이트가 주된 noise source일 때, syndrome 측정 시퀀스만으로 해당 CNOT 위치를 식별**할 수 있는지 분석하고, 그 결과를 바탕으로 학습 모델을 설계하기 위한 연구 코드.

---

## 1. 연구 목표

### 1.1 큰 그림
- d=3 rotated surface code (17 qubits = 9 data + 8 ancilla)
- 반복 stabilizer measurement 회로
- **가정**: 24개 directed CNOT 중 하나가 fault source이고, 매 라운드 같은 위치에서 fault가 반복 발생 (temporally correlated)
- **목표**: syndrome bit sequence만 보고 어느 CNOT인지 추정 (24-class classification)

### 1.2 무시하는 nuisance variable
- fault가 일어난 round 시점
- fault의 Pauli type (XX, XY, ..., ZZ 중 어느 것인지)

### 1.3 동기
toy d=3에서는 lookup table로 풀 수 있지만, code distance가 커지면 lookup 비용↑. 한 번 학습된 모델로 추론하는 게 목적.

---

## 2. 현재까지의 분석 흐름 (이 README 작성 시점 기준)

| 단계 | 질문 | 결과 |
|---|---|---|
| (a) | XZ fault만 가정, n_rounds=1로 24 CNOT을 구별 가능? | 8 unique / 24, 11개 silent |
| (b) | n_rounds=2로 늘리면? | **24 unique (완전 식별)** — 임계점 |
| (c) | n_rounds=3 이상은? | 그대로. 추가 라운드는 같은 정보를 반복할 뿐 |
| (d) | 다른 Pauli pair는 어떤가? | XY/XZ/YX/YY/ZY = 24 unique, **ZX/YZ = 부분, XX = 17, ZZ = 11** |
| (e) | Multi-round XZ at [0,1]은 single-round과 구별? | round-0-visible CNOT은 구별 가능, silent는 동일 |
| (f) | **Pauli를 모르고 syndrome만 봤을 때 cross-Pauli ambiguity?** | **216 케이스 중 123 unique, 38 ambiguous, 단독 식별 0/24** |
| (g) | 라운드/모드를 바꾸면 (f)가 깨지나? | **불변** — n_rounds 2/3/4 동일, reset/no-reset 동일 |
| (h) | 사용자의 핵심 질문: 서로 다른 (CNOT, Pauli) 페어가 같은 syndrome? | **72쌍 (cross-CNOT cross-Pauli)** — 본질적 모호성 |

### 2.1 핵심 결론

> **단일 라운드 fault + mid-measure syndrome만 가지고, Pauli type을 모르는 상태에서 24-class CNOT 식별은 정보이론적으로 불가능.** 72쌍의 충돌이 라운드 수와 측정 모드와 무관하게 invariant.

### 2.2 정확한 명제 (사용자 코멘트 반영)
- "round 0 syndrome이 같다 → 영원히 같다" 는 **틀림** (round 1에서 풀릴 수 있음)
- 정확한 명제: **"round R까지 syndrome string 일치 = round R 종료 시점의 (data + ancilla) 상태 동일 → 이후 모든 라운드 동일"**

---

## 3. 디렉토리 구조

```text
.
├── README.md                           # 이 문서
├── requirements.txt                    # pennylane, numpy, matplotlib, stim
├── src/                                # forward simulator 모듈
├── scripts/                            # 데이터 생성 + 분석 스크립트
├── exercise/                           # 초기 prototyping 노트북
├── notebooks/                          # 디버그 노트북
└── data/
    ├── mid_measure_reset/              # forward-sim 데이터 (모드 1)
    ├── mid_measure_no_reset/           # forward-sim 데이터 (모드 2)
    ├── no_mid_measure_final_sample/    # forward-sim 데이터 (모드 3)
    ├── no_mid_measure_final_probs/     # forward-sim 데이터 (모드 4)
    └── analysis/                       # 분석 결과 (이 연구의 산물)
        ├── 1_per_pauli_degeneracy/
        ├── 2_pauli_sweep_summary/
        ├── 3_cross_pauli_conflict/
        ├── 4_cross_pauli_per_pauli_view/
        └── 5_cross_pauli_pair_collisions/
```

---

## 4. `src/` — Forward Simulator 모듈

| 파일 | 역할 |
|---|---|
| `config.py` | 기본 실험 설정 (n_rounds, shots, fault 위치, 4가지 측정 situation) |
| `surface_code_layout.py` | 데이터/ancilla 큐빗 번호, Z/X stabilizer 정의, `enumerate_stabilizer_cnots()` |
| `logical_state.py` | `|0_L⟩` 초기상태 준비 회로 |
| `fault_schedule.py` | `(control, target, round)` fault 스케줄 빌더, 라운드 부분집합 enumerator, occurrence vector 변환 |
| `stabilizer_circuit.py` | Z/X stabilizer 라운드, **방향 일치 조건의 CNOT 직후 Pauli 주입** |
| `simulator.py` | `make_repeated_stabilizer_qnode(...)` — 4가지 측정 모드 QNode 생성 |
| `postprocess.py` | 원시 출력 reshape, detection event 계산 (`syndrome[t+1] XOR syndrome[t]`), 비트 평탄화 |
| `dataset_generators.py` | dataset row 생성기 + 중복 패턴 탐색 helper |
| `io_utils.py` | CSV/JSON 저장 helper |

### Surface code layout (요약)
- Data qubits: `0..8`
- Z ancilla: `9, 10, 11, 12` (data → ancilla CNOT)
- X ancilla: `13, 14, 15, 16` (H → ancilla → data CNOT → H)
- 한 라운드당 24개 directed CNOT
- stabilizer 순서: `[Z0, Z1, Z2, Z3, X0, X1, X2, X3]`

### 4가지 측정 모드
1. **mid_measure + reset** — 표준 surface code cycle
2. **mid_measure + no reset** — ancilla 측정 후 collapse 상태 유지 (detection event에 가까운 시그니처)
3. **no mid-measure, final sample** — 마지막에 ancilla 1회 측정
4. **no mid-measure, final probs** — 마지막에 ancilla 8 wire의 256개 확률

---

## 5. `scripts/` — Forward 데이터 생성 스크립트

| 파일 | 역할 |
|---|---|
| `debug_run.py` | 빠른 점검 (1개 fault config 실행, 출력 확인) |
| `generate_fixed_cnot_patterns.py` | 한 CNOT 고정, 모든 라운드 occurrence 패턴 sweep (2^N 케이스) |
| `generate_all_cnot_single_fault.py` | 라운드 1개 고정, 24개 CNOT 위치 sweep |
| `generate_all_cnot_round_patterns.py` | 위 두 축의 곱집합 (24 × 2^N) |
| `inspect_dataset.py` | 생성된 CSV 점검 |

### CLI 공통 옵션
- `--n-rounds`, `--control`, `--target`, `--error-types`, `--mode`, `--output-root`

### `data/<mode>/` CSV 한 행 = 한 시뮬레이션
주요 컬럼: `sample_id, mode, n_rounds, control, target, error_rounds, occurrence_vector, error_types, syndrome_bits, detection_bits, final_bits, probabilities, fault_schedule_json, raw_output_json`

---

## 6. `scripts/` — 분석 스크립트 (이번 연구의 핵심)

각 스크립트는 출력을 `data/analysis/{번호}_{이름}/`에 자동 저장.

### 6.1 `analyze_round1_degeneracy.py` → `1_per_pauli_degeneracy/`
- **하나의 Pauli pair**에 대해 24 CNOT의 신드롬을 모으고 within-Pauli degeneracy class 시각화
- 가로축: 24 CNOT 실행순서 / 세로축: stabilizer ancilla × n_rounds (Z/X 분리, 라운드 분리)
- 상단 stripe: degeneracy class id / size
- CLI: `--n-rounds`, `--error-types XZ`, `--error-rounds 0` (또는 `0,1`), `--no-reset`

### 6.2 `sweep_pauli_pairs.py` → `2_pauli_sweep_summary/`
- 9개 Pauli pair × 24 CNOT을 모두 돌려서, **각 Pauli pair마다 unique syndromes 수, silent CNOT 수, max class size** 요약
- bar chart로 시각화
- CLI: `--n-rounds 2`, `--error-rounds 0`, `--no-reset`

### 6.3 `analyze_cross_pauli_degeneracy.py` → `3_cross_pauli_conflict/`
- 216 = 24 × 9 케이스 모두 모음
- 각 syndrome에 대해 **그 syndrome을 만드는 distinct CNOT 수** (= conflict size) 계산
- 24×9 행렬 시각화 (셀=conflict size; 1=식별 가능, ≥2=ambiguous)
- 충돌 그래프 출력 (어떤 CNOT이 어느 CNOT과 헷갈리는지)
- 가장 큰 ambiguous syndromes 상위 5개의 멤버 명세

### 6.4 `visualize_cross_pauli_per_pauli.py` → `4_cross_pauli_per_pauli_view/`
- 9개 Pauli pair 각각에 대해 `1_per_pauli_degeneracy/`와 같은 형식의 heatmap 생성
- **차이점**: 상단 stripe의 class id가 cross-Pauli **global id** (216 케이스 전체에서 계산)
- 다른 Pauli 그림에서 같은 global id가 보이면 → 그 두 (CNOT, Pauli)가 cross-Pauli 동일 syndrome
- 출력: 9개 PNG (`eXX.png` ~ `eZZ.png`) + `matches_by_global_id.csv` (global id로 정렬된 매칭표)
- 한 condition당 한 하위 폴더 (예: `nrounds2_r0_noreset/`)

### 6.5 `analyze_cross_pauli_cross_cnot_pairs.py` → `5_cross_pauli_pair_collisions/`
- syndrome을 공유하는 모든 unordered 페어를 enumerate한 뒤 세 카테고리로 분류:
  - `same_cnot_diff_pauli` — 같은 CNOT의 다른 Pauli (무해)
  - `diff_cnot_same_pauli` — within-Pauli degeneracy (Pauli 알면 해결)
  - **`diff_cnot_diff_pauli`** — **본질적 모호성** (사용자가 묻는 케이스)
- 24×24 CNOT-쌍 행렬 시각화 (셀=충돌 (α,β) 페어 수)
- CSV에는 `diff_cnot_diff_pauli` 페어만 저장

---

## 7. `data/analysis/` — 분석 결과 폴더 상세

### 7.1 `1_per_pauli_degeneracy/`
파일명 패턴: `nrounds{N}_e{Paulis}_r{rounds_str}_{mode}_(degeneracy.png | syndromes.csv)`

| 파일 | 설명 |
|---|---|
| `nrounds1_eXZ_r0_reset_*` | XZ fault at round 0, 1 라운드, reset → 8 unique / 24 |
| `nrounds2_eXZ_r0_reset_*` | XZ at [0], 2 라운드, reset → 24 unique |
| `nrounds2_eXZ_r0_noreset_*` | 위와 동일 condition, no-reset |
| `nrounds2_eXZ_r0-1_reset_*` | XZ at rounds [0,1], 2 라운드, reset |
| `nrounds2_eXZ_r0-1_noreset_*` | 위와 동일 condition, no-reset |

### 7.2 `2_pauli_sweep_summary/`
| 파일 | 설명 |
|---|---|
| `pauli_sweep_nrounds2_r0_reset.{png,csv}` | n=2에서 9개 Pauli pair 비교 (XX=17 unique, ZZ=11 unique 등) |

### 7.3 `3_cross_pauli_conflict/`
파일명: `cross_pauli_nrounds{N}_r{rounds}_{mode}.(png|csv)`

| 파일 | 설명 |
|---|---|
| `cross_pauli_nrounds2_r0_reset.*` | 216 케이스 통합. 24×9 heatmap; 셀=충돌 CNOT 수. **all-zero silent 8개 CNOT, 38 ambig** |
| `cross_pauli_nrounds2_r0_noreset.*` | 동일 결과 (모드 invariant) |
| `cross_pauli_nrounds3_r0_*` | n=3에서도 동일 결과 (라운드 invariant) |

### 7.4 `4_cross_pauli_per_pauli_view/`
| 하위 폴더 | 내용 |
|---|---|
| `nrounds2_r0_noreset/` | `eXX.png` ~ `eZZ.png` (9장) + `matches_by_global_id.csv` |

### 7.5 `5_cross_pauli_pair_collisions/`
파일명: `cross_pauli_pairs_nrounds{N}_r{rounds}_{mode}.(png|csv)`

| 파일 | 설명 |
|---|---|
| `cross_pauli_pairs_nrounds2_r0_noreset.*` | **72 cross-Pauli cross-CNOT 페어**. 24×24 행렬 |
| `cross_pauli_pairs_nrounds3_r0_noreset.*` | n=3에서도 같은 72쌍 (CSV diff: identical) |
| `cross_pauli_pairs_nrounds4_r0_noreset.*` | n=4에서도 같은 72쌍 |

CSV diff로 검증 완료:
- n_rounds 2 vs 3: identical
- n_rounds 2 vs 4: identical

---

## 8. 주요 발견 요약 (한눈에)

```
┌─────────────────────────────────────────────────────────────────┐
│ XZ fault only, n_rounds=1: 8 unique / 24 (11 silent)            │
│ XZ fault only, n_rounds=2: 24 unique (full identification)      │
│ XZ fault only, n_rounds=3+: same as n_rounds=2 (no new info)    │
│                                                                  │
│ Across 9 Pauli pairs (n_rounds=2):                              │
│   XY, XZ, YX, YY, ZY     → 24 unique                            │
│   YZ → 21,  ZX → 20      → mild ambiguity                       │
│   XX → 17 (4 silent),  ZZ → 11 (4 silent)  → worst              │
│                                                                  │
│ Cross-Pauli (216 cases, Pauli unknown):                         │
│   123 unique syndromes                                          │
│   38 ambiguous syndromes                                        │
│   0/24 CNOTs cleanly identifiable                               │
│   8-way silent class (XX silent 4 + ZZ silent 4)                │
│                                                                  │
│ Cross-Pauli cross-CNOT pair collisions:                         │
│   72 pairs (invariant under n_rounds ≥ 2 and reset mode)        │
│   Worst pair: CNOT00 ↔ CNOT01 (5 (α,β) combinations collide)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 다음 단계 (Roadmap)

72쌍의 본질적 충돌을 다루기 위한 후보 전략:

### 9.1 분석 확장 (코드 거의 그대로 재사용 가능)
- **(M1) Multi-round same-CNOT fault** — `error_rounds=[0,1,2,...]`로 라운드마다 fault 반복. 사용자의 원래 framing("특정 CNOT이 주로 fault")에 직접 부합. 정보 누적으로 72쌍이 깨지는지 확인.
- **(M2) Final data measurement 결합** — 기존에 만든 `no_mid_measure_final_*` 데이터의 9비트(또는 256 확률)를 syndrome에 붙임. 다른 관측가능량으로 충돌 깨기.
- **(M3) Fault round를 round 0이 아닌 다른 곳으로** — 마지막 라운드에서 fault가 나면 식별 가능성 떨어질 가능성 확인.

### 9.2 학습 모델 설계 (분석 결과 후 진입)
충돌 그래프를 어떻게 다룰지에 따라 모델 디자인이 갈림:

- **(L1) Soft classification** — 24-dim softmax 출력. ambiguous syndrome엔 확률이 여러 CNOT에 퍼지는 게 정답. Top-k accuracy, KL divergence 평가.
- **(L2) Coarse label** — 충돌 그래프의 connected component를 라벨로 묶기 (24-class → ~10 group). 해상도 trade-off로 풀 수 있는 문제로 변환.
- **(L3) Marginalized Bayesian** — Pauli pair에 대한 prior 가정 (예: depolarizing channel). P(CNOT | syndrome) 계산. 충돌 페어 내에서도 분포가 다르면 어느 정도 식별 가능.
- **(L4) Multi-task** — (CNOT × Pauli) 동시 추정. Pauli 분류 정확도와 CNOT 분류 정확도 trade-off 확인.

### 9.3 모델 아키텍처 후보
- Logistic regression (baseline, deterministic 매핑이 잘 풀리는지)
- MLP (2–3 hidden layers)
- GNN — surface code의 data/ancilla connectivity를 활용 (d≥5에서 가치 큼)
- Transformer — round 시퀀스를 자연스러운 입력으로

### 9.4 평가 디자인
- Random split (deterministic 데이터에선 의미 적음)
- **Pauli pair generalization** — XZ로 학습 → ZX/YY로 test
- Noise robustness — syndrome bit-flip 추가 후 정확도 측정
- 식별 가능성 상한 vs 모델 성능 비교

---

## 10. 설치 및 실행

### 10.1 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 10.2 Forward 데이터 생성 예시
```bash
# 한 CNOT 고정, 모든 라운드 패턴 sweep
python scripts/generate_fixed_cnot_patterns.py \
    --n-rounds 4 --control 0 --target 9 --error-types XZ --mode mid_reset

# 24 CNOT × 모든 라운드 패턴
python scripts/generate_all_cnot_round_patterns.py \
    --n-rounds 4 --error-types XZ --mode all
```

### 10.3 분석 스크립트 예시
```bash
# 1. 단일 Pauli, within-Pauli degeneracy
python scripts/analyze_round1_degeneracy.py \
    --n-rounds 2 --error-types XZ --error-rounds 0

# 2. 9 Pauli pair 비교
python scripts/sweep_pauli_pairs.py --n-rounds 2 --error-rounds 0

# 3. cross-Pauli conflict (216 케이스 통합)
python scripts/analyze_cross_pauli_degeneracy.py --n-rounds 2 --error-rounds 0

# 4. per-Pauli view with global ids
python scripts/visualize_cross_pauli_per_pauli.py --n-rounds 2 --error-rounds 0

# 5. cross-Pauli cross-CNOT pair collisions
python scripts/analyze_cross_pauli_cross_cnot_pairs.py --n-rounds 2 --error-rounds 0
```

기본 모드는 **no-reset** (실험적 cost 낮고 정보량 동일). reset 모드로 돌리려면 `--reset` 추가.

---

## 11. 핵심 가정 명시

이 연구의 모든 결론은 다음 가정에 의존:

1. **단일 fault source**: 한 시점에 fault가 발생하는 CNOT은 24개 중 정확히 하나
2. **결정론적 Pauli fault**: stochastic Pauli channel이 아니라, fault가 일어나면 특정 (X, Y, Z) 조합이 결정적으로 발생
3. **fault 직후 위치**: CNOT 게이트 직후에 Pauli error 주입 (`stabilizer_circuit.py:inject_faults_after_cnot`)
4. **CNOT 방향성**: `CNOT(c→t)`와 `CNOT(t→c)`는 다른 위치로 취급
5. **측정/reset 노이즈 없음**: ancilla 측정/reset 자체는 완벽

이 가정들이 완화되면 분석 결과(특히 72쌍 충돌)는 다시 평가 필요.

---

## 12. Long-sequence frame (브랜치 `long-sequence-analysis`)

1~11절은 **single-shot** frame을 분석합니다 — 한 번의 fault event, 한 syndrome string에서 (CNOT, Pauli) pair를 식별. 그 frame은 72쌍 충돌이라는 정보이론적 벽에 부딪혔고, 그 충돌은 라운드 수·측정 모드에 invariant. 브랜치 `long-sequence-analysis`는 **sequence-level 통계**로 이 벽을 우회하는 reframing입니다.

### 12.1 문제 재설정 (R1 시나리오)

| 항목 | Single-shot (main) | R1 (이 브랜치) |
|---|---|---|
| Fault 발생 | 지정 라운드에 결정론적 1회 | **매 라운드, 매 CNOT** 베르누이 추첨 |
| CNOT별 발생률 | n/a | 24개 모두 background `p_bg`, 지정된 1개에 **elevated `p_high`** |
| Event당 Pauli | 고정 pair | **15개 non-identity 2-qubit Pauli 균등 추첨** (II 제외) |
| 관측 | 단일 syndrome string | `(T, 8)` syndrome stream |
| 추정 대상 | (CNOT, Pauli) 공동 | **dominant CNOT만** (Pauli는 marginalize) |
| Decoder | n/a | 인터페이스 분리 — R1은 `IdentityDecoder` (no-op) |

기대: single-round 신드롬이 충돌해도, 서로 다른 dominant CNOT이 만드는 **T-라운드 시퀀스 분포** 는 구별 가능할 수 있고, 그러면 72쌍 ceiling을 깰 수 있음.

### 12.2 새 모듈

| 파일 | 역할 |
|---|---|
| `src/decoder.py` | `Decoder` ABC + `Correction` dataclass + `IdentityDecoder`. window decoder (R3b lookup decoder 등) 확장점 |
| `src/stochastic_faults.py` | `BackgroundElevatedSampler(p_bg, p_high, faulty_cnot_id, rng)` — 매 라운드 매 CNOT 베르누이 + 15-Pauli 균등 추첨 |
| `src/sequence_runner.py` | `run_sequence(T, sampler, decoder, ...)` — `IdentityDecoder`용 single-QNode fast path. 다른 decoder는 window state hand-off가 추가될 때까지 `NotImplementedError` |
| `src/sequence_dataset.py` | `generate_r1_dataset(...)` + `save_r1_dataset(...)` — N개 `(T, 8)` syndromes + seed + oracle fault log |

### 12.3 새 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/generate_r1_sequences.py` | R1 데이터셋 생성 CLI. 출력: `data/r1_sequences/{baseline\|cnotXX}/T{T}_pbg{p}_phigh{p}_seed{s}/{reset\|no_reset}/` |
| `scripts/sanity_check_r1_infra.py` | 회귀 테스트 — §12.4 참고 |

### 12.4 Sanity check (통과)

새 인프라가 main 브랜치 single-shot 결과를 통제된 케이스에서 비트 단위로 재현함.

- **Test B** — sampler dict 구조: 216 = 24 CNOT × 9 Pauli pair 전부에서, `BackgroundElevatedSampler(p_bg=0, p_high=1, faulty_cnot_id=k)`가 만드는 fault dict의 `(round, control, target, error_wires, error_types)`가 `make_fixed_cnot_fault_schedule` 출력과 일치 (추가 필드 `cnot_id`, `pauli_pair`는 허용).
- **Test C** — end-to-end syndrome 일치: 24개 CNOT 전부, Pauli=XZ가 T=2 시퀀스의 매 라운드에 발생하는 케이스에서, 결정론화한 sampler를 `run_sequence`에 태워 만든 `(T, 8)` syndromes가 main 브랜치 path (`make_fixed_cnot_fault_schedule` + `make_repeated_stabilizer_qnode`)와 비트 단위 일치. `reset` / `no_reset` 모드 모두 검증. 각 모드에서 **24/24 PASS**.

이걸로 `sampler → fault_schedule format → QNode → reshape` 사슬이 검증됨. Stochastic sampling 자체 (numpy 추첨 분포)는 별도 단위 테스트하지 않음 — Phase 2에서 분포가 이상하면 자동으로 노출.

### 12.5 캐퍼빌리티 요약

추가 코딩 없이 R1 인프라로 지금 할 수 있는 것:

- `(N, T, 8) uint8` 형태 데이터셋 + 결정론적 seed + 샘플별 oracle fault log
- 노브: `faulty_cnot_id ∈ {None, 0..23}`, `p_bg, p_high ∈ [0,1]`, `T ∈ ℤ⁺`, `reset` / `no_reset`
- 시나리오: R1 본 실험, baseline (elevated CNOT 없음), forced sampler로 single-shot 결과 재현

확장점: non-Identity decoder path (window state hand-off, `sequence_runner.py:72`), final data-qubit 측정 결합, 다른 Pauli pool, Stim 백엔드, code distance > 3.

### 12.6 다음 step — 디바이스 viability map

`p_bg`, `p_high`는 우리가 고르는 노브가 아니라 **물리 디바이스의 속성** (현재 초전도 2-qubit gate error 약 10⁻³ – 10⁻², 고장난 CNOT은 10⁻² – 10⁻¹). `T`는 coherence time이 상한인 실험 설계 자유도. 그래서 다음 step은 "파라미터를 고른다"가 아니라 **방법의 적용 범위를 측정**:

> `(p_bg, p_high, T)` 격자에서, 매 라운드의 syndrome 분포 (Pauli marginalized) 위의 Bayes-optimal classifier 정답률을 학습 없이 계산. 결과물은 *viability map*:
> 1. 어떤 디바이스 파라미터 regime에서 식별 task가 유효한가?
> 2. 목표 정답률을 위해 T는 얼마나 길게 관측해야 하나?
> 3. main 브랜치의 72-collision 쌍 중 시퀀스로 깨지는 것 vs 시퀀스에서도 살아남는 본질적 한계는?

이 ceiling은 이후 학습 모델의 상한이기도 하므로, 평가 benchmark 역할도 겸함.

---

## 13. R1 ceiling 결과 (Phase 2 — Task #3)

### 13.1 계산한 내용

각 후보 dominant CNOT k에 대해, 한 라운드의 detection event 분포

$$P_k(s) = \left(\bigast_{i \neq k} D_i^{(p_{bg})}\right) \ast D_k^{(p_{high})}, \quad s \in \mathbb{F}_2^8$$

를 24개 CNOT의 독립 contribution 분포의 XOR-convolution으로 계산. 각 $D_i^{(p)}$는 "확률 $1-p$로 syndrome 0, 확률 $p/|\text{Pauli pool}|$로 각 lookup syndrome". Reset 모드에서 detection event는 라운드별 i.i.d.이므로 T-라운드 Bayes-optimal classifier는 $\hat{k} = \arg\max_k \sum_t \log P_k(d_t)$. 정확도는 (p_bg, p_high, T) 격자 위에서 Monte Carlo로 측정.

서브태스크 스크립트:
- `src/ceiling.py` — lookup 빌더, XOR-convolution, Bayes classifier
- `scripts/compute_r1_ceiling.py` — (p_bg, p_high, T) sweep + 시각화
- `scripts/analyze_ceiling_groups.py` — 구조적 ambiguity 그룹 분석 + 15-Pauli vs 9-Pauli 비교

### 13.2 핵심 발견 — 4개의 구조적 ambiguity 그룹

두 dominant 가설 $k, k'$의 per-round 분포가 동일 ⟺ 두 CNOT의 **single-fault syndrome multiset** $\{\sigma_{k,\alpha}\}_\alpha$와 $\{\sigma_{k',\alpha}\}_\alpha$가 $\mathbb{F}_2^8$ 위 multiset으로서 동일. lookup으로 직접 확인:

| 그룹 | 멤버 | Stabilizer | Multiset (15-Pauli) |
|---|---|---|---|
| $G_1$ | CNOT 6, 7 | Z2 stab (data → ancilla) | $\{0^3, 4^4, 64^4, 68^4\}$ |
| $G_2$ | CNOT 14, 15, 17 | X1 stab 내부 (ancilla → data) | $\{0^7, 32^8\}$ |
| $G_3$ | CNOT 18, 19, 20 | X2 stab 내부 | $\{0^7, 64^8\}$ |
| $G_4$ | CNOT 22, 23 | X3 stab | $\{0^7, 128^8\}$ |

총 식별 가능 그룹 수 = 24 − (2+3+3+2) + 4 = **18**. Per-class accuracy 점근 한계 = $18/24 = 0.750$. Group-level 식별은 극한에서 1.0.

### 13.3 검증 — 큰 T에서 Monte Carlo

$(p_{bg}, p_{high}) = (0.01, 0.3)$ 에서:

| T | per-class acc | group acc |
|---|---|---|
| 10 | 0.42 | 0.56 |
| 100 | 0.747 | 0.996 |
| 300 | 0.749 | 1.000 |
| 1000 | 0.751 | 1.000 |

→ per-class는 18/24에서 정확히 saturate, group accuracy는 1.0에 수렴. Ambiguity 그룹 내부에서 Bayes-optimal classifier는 한 대표 멤버를 결정론적으로 예측 → 나머지 멤버는 0% accuracy.

### 13.4 Per-Pauli vs multiset — 정확한 주장 범위

같은 그룹의 두 CNOT이 **모든 Pauli에 대해 같은 syndrome을 내는 것은 아님**. CNOT 6 vs 7의 경우 15개 Pauli 중 7개는 같은 syndrome, 8개는 다른 syndrome. 두 CNOT이 구별 불가능한 이유는 단지 15-Pauli pool 위의 어떤 permutation $\pi$가 존재해서 $\sigma_{6,\alpha} = \sigma_{7,\pi(\alpha)}$이기 때문. **Pauli 균등 random 가정** 하에서 marginalize하면 분포가 동일해지지만, 만약 Pauli가 fault별로 알려진다면 (main 브랜치의 single-shot frame처럼) 두 CNOT은 구별 가능.

### 13.5 구조적 이유

라운드 순서 $[Z_0, Z_1, Z_2, Z_3, X_0, X_1, X_2, X_3]$에서 stabilizer $S$의 CNOT 직후 주입된 fault는 **(a) $S$보다 라운드 안에서 늦게 실행되는 stabilizer 중 (b) fault 위치의 data qubit을 공유하는 것**에만 영향.

4개 ambiguity 그룹의 멤버들은 모두 **data qubit의 "downstream stabilizer 멤버십"이 동일**한 CNOT들. 예: $X_2$ stab (qubits {3,4,6,7})에서 qubit 7만 후속 $X_3$ stab에 들어감. 따라서 CNOT 18 (target=3), 19 (target=4), 20 (target=6)은 같은 downstream footprint를 공유 → 같은 그룹. CNOT 21 (target=7)만 분리. $G_1, G_2, G_4$도 같은 논리.

→ ceiling은 **stabilizer 측정 스케줄 × code topology**에서 오는 결과이고 Pauli randomness 자체와는 별개. Pauli pool을 바꾸면 일부 그룹은 깰 수 있지만 X-stab 내부 그룹은 못 깸.

### 13.6 15-Pauli vs 9-Pauli 비교

9개 fully-two-qubit Pauli (X/Y/Z × X/Y/Z, single-leg 제외) 만 사용하면:

| Pauli pool | Ambiguity 그룹 | per-class ceiling |
|---|---|---|
| 15-Pauli (single-leg 포함) | $\{G_1, G_2, G_3, G_4\}$ | $18/24 = 0.750$ |
| 9-Pauli (single-leg 제외) | $\{G_2, G_3, G_4\}$ | $19/24 = 0.792$ |

$G_1$ (CNOT 6, 7)은 **pool 의존적**: 6개의 single-leg Pauli가 15-Pauli 모델에서 두 CNOT의 multiset을 "평형화"하는 역할. 빼면 multiset이 $\{0^2, 4^3, 64^1, 68^3\}$ vs $\{0^1, 4^2, 64^2, 68^4\}$로 갈라짐. X-stab 내부 그룹 $G_2, G_3, G_4$는 양쪽에서 살아남음 (구조적).

비교 plot: [data/analysis/6_sequence_ceiling/ceiling_compare_curves.png](data/analysis/6_sequence_ceiling/ceiling_compare_curves.png).

### 13.7 Decoding이 구조적 그룹을 깰 수 있는가?

**아마 깰 수 있을 것으로 추측** — window decoder (R3b 시나리오)를 도입하면 X-stab 내부 그룹도 깨질 가능성이 있음. 직관:

- Decoder 없으면 (R1): dominant CNOT은 *per-round detection event 분포*에만 영향. 두 CNOT의 분포가 동일하면 영원히 구별 불가능.
- Decoder 있으면: decoder는 관측한 syndrome $s$로부터 correction $C(s)$를 데이터 큐빗에 적용. Syndrome이 $k$와 $k'$에 대해 ambiguous할 때 decoder는 **같은** correction을 적용 — 하지만 **실제 데이터 큐빗 residual 오류는 다름** (실제 fault 위치 qubit이 다르므로). 이 residual은 **이후 라운드**의 syndrome에 구별 가능한 방식으로 전파됨.

구체적으로 $G_3 = \{18, 19, 20\}$의 경우: CNOT 18 fault는 data qubit 3에 residual, CNOT 19는 qubit 4에, CNOT 20은 qubit 6에. 단일 라운드 syndrome은 일치하지만, post-correction residual이 새 fault와 상호작용하면서 그 다음 라운드들의 Z-stab/X-stab 측정에서 통계적으로 구별되는 stream이 나옴.

→ R3b 확장이 단순한 "다른 시나리오"가 아니라 **과학적으로 중요한 후속 작업**. Task #7을 "future work"에서 "near-term experimental priority"로 격상할 만함.

### 13.8 실용적 함의

1. **목표를 18-group (9-Pauli이면 19-group) 식별로 재정의**. 이게 R1 문제의 정보이론적 해상도.
2. **Group accuracy가 의미 있는 지표**. Per-class accuracy는 0.75에서 천장이 막혀 있어 실제 분류기 품질과 구조적 ambiguity를 섞어버림.
3. **Viability region은 여전히 의미 있음**. (p_bg, p_high, T)에서 group accuracy는 1/18 (random)에서 1.0까지 변함. `(p_bg=1e-2, p_high=1e-1, T≈100)` 전이 구간이 학습 분류기 (Task #6) 평가의 sweet spot.
4. **세부 파라미터 결정 (Task #2)은 R3b ceiling 알기 전까지 보류**: 만약 R3b가 X-stab 그룹을 깬다면 (p_bg, p_high, T) 관심 영역이 달라짐.
