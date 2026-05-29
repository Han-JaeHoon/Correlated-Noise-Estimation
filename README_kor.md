# Correlated-Noise-Estimation

`d=3` rotated surface code에서 **특정 CNOT 게이트가 주된 noise source일 때, syndrome 측정 시퀀스만으로 해당 CNOT 위치를 식별**할 수 있는지 분석하고, 그 결과를 바탕으로 학습 모델을 설계하기 위한 연구 코드.

> **다른 컴퓨터/Claude 세션에서 이어받으시나요?** [`HANDOFF.md`](HANDOFF.md) 부터 읽어주세요 — 현재 브랜치·최신 결과·다음 추천 step을 30초에 파악하는 entry point입니다.

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

---

## 14. R3b ceiling 결과 (branch `decoder-add-analysis`)

§13.7 에서 열어둔 가설 — "single-round window decoder 를 추가하면 X-stab interior ambiguity 그룹이 깨질 것" — 에 대해 R3b 인프라를 end-to-end 로 구축하고 직접 검증했습니다. 한 줄 요약: **가설의 정성적 부분은 확인 (R3b 의 round-별 marginal 이 그룹 멤버 간 실제로 갈라짐), 단 정량적 주장은 falsified (그럼에도 marginal-Bayes 정확도가 R1 보다 *훨씬 낮음*).**

### 14.1 추가된 인프라

| 파일 | 역할 |
|---|---|
| `src/pauli_frame.py` | 17-큐빗 symplectic Pauli frame, in-place CNOT/H/reset update |
| `src/round_propagation.py` | `stabilizer_round` 와 게이트별로 동일한 symbolic propagator. post-round frame + 예측 ancilla outcome 둘 다 반환 |
| `src/decoder.py` `LookupDecoder` | window=1 decoder. 24×15 single-fault syndrome 의 역 lookup + residual 적용. tie-break = lex-min (k, α). multi-fault 라운드 → identity |
| `src/sequence_runner.py` window path | 매 라운드 1-round QNode 호출, `initial_error_list` 로 carried frame 주입. IdentityDecoder 의 경우 fast path 와 syndrome 완전 일치 (regression check) |
| `src/fast_simulator.py` `run_sequence_symbolic` | `run_sequence` 의 drop-in 대체. propagator 만 사용. ~100× faster, bit-identical (18/18 cases) |
| `src/ceiling_r3b.py` | MC marginal-Bayes ceiling estimator |
| `scripts/sanity_check_round_propagation.py` | PL vs symbolic single-round lookup (360/360), data-Pauli 보존 (27/27), 2-round end-to-end (360/360) |
| `scripts/sanity_check_fast_simulator.py` | PL vs symbolic full-sequence (18 cases) + benchmark |
| `scripts/sanity_check_r3b_runner.py` | fast vs window IdentityDecoder 동일성, LookupDecoder smoke, LookupDecoder ≠ IdentityDecoder |
| `scripts/compute_r3b_ceiling.py` | `(p_bg, p_high, T)` grid sweep, R1 (analytic) + R3b (MC) 둘 다 |
| `scripts/analyze_r3b_ceiling.py` | 그룹별 per-class accuracy, within-group spread, R3b marginal 들 사이 pairwise JS divergence |

R3b decoder 의 정확한 spec 은 [docs/r3b_design.md](docs/r3b_design.md).

### 14.2 측정 방법

각 candidate dominant CNOT k ∈ {0..23} 에 대해 `n_train` 개의 symbolic 시퀀스 (T 라운드) 를 `LookupDecoder` 와 함께 시뮬레이션. round 0 은 항상 frame=0 에서 시작하므로 decoder 와 무관 → 분석에서 제외. round 1+ 의 detection event 를 모아서 marginal P(d | k) 를 (Laplace smoothing 적용) histogram 으로 추정. Marginal Bayes 분류기

\[\hat{k} = \arg\max_k \sum_{t \ge 1} \log P(d_t | k)\]

를 독립 샘플 `n_test` 개에 적용. confusion matrix 로부터 per-class 정확도, group 정확도 (R1 ambiguity 그룹 정의 사용), marginal 들 사이 JS divergence 계산.

### 14.3 핵심 수치

8 cells, n_train=n_test=50–200, seed 0–1 (자료: `data/analysis/7_r3b_ceiling/`):

| (p_bg, p_high) | T | R1 per-class | R3b per-class | R1 group | R3b group |
|---|---|---|---|---|---|
| (0.001, 0.01) | 30  | 0.102 | 0.087 | 0.178 | 0.118 |
| (0.001, 0.01) | 100 | 0.190 | 0.087 | 0.255 | 0.133 |
| (0.001, 0.01) | 300 | 0.327 | 0.095 | 0.418 | 0.125 |
| (0.005, 0.05) | 30  | 0.225 | 0.113 | 0.290 | 0.162 |
| (0.005, 0.05) | 100 | 0.421 | 0.104 | 0.544 | 0.140 |
| (0.01,  0.1)  | 10  | 0.182 | 0.112 | 0.243 | 0.149 |
| (0.01,  0.1)  | 30  | 0.309 | 0.093 | 0.395 | 0.120 |
| (0.01,  0.1)  | 50  | 0.406 | 0.101 | 0.527 | 0.130 |

**R1 은 T 와 (p_bg, p_high) signal-to-background 비율에 따라 부드럽게 향상**, **R3b 는 T 나 p 에 관계없이 0.08–0.16 plateau**. 각 cell 의 가장 큰 T 에서 이미 R1 이 R3b 의 3–4×. 특히 (p_bg=0.001, p_high=0.01) 의 T 변화 (30 → 100 → 300) 에서 R1 은 0.10 → 0.19 → 0.33, R3b 는 0.087 → 0.087 → 0.095 로 plateau 가 두드러짐.

### 14.4 정성적 확인 — marginal 은 갈라짐

T=50, R3b 의 그룹 멤버들 사이 pairwise JS divergence (per-round marginal 기준):

| 그룹 | 평균 pairwise JS | 최댓값 pairwise JS |
|---|---|---|
| {6, 7} | 0.065 | 0.065 |
| {14, 15, 17} | 0.062 | 0.068 |
| {18, 19, 20} | 0.067 | 0.075 |
| {22, 23} | 0.066 | 0.066 |

R1 에선 이 값들이 정의상 정확히 0 (multiset 자체가 같으므로). R3b 는 marginal-level 의 equivalence 는 명백히 깨뜨림 — §13.7 이 예측한 메커니즘이 실제로 작동.

### 14.5 그런데 왜 R3b 가 더 나쁜가 — 잘못된 correction 의 노이즈 주입

Marginal 이 갈라졌음에도 R3b 분류 정확도는 R1 보다 훨씬 낮음. 원인은 lex-min single-fault decoder 자체의 작동 방식:

- 0 이 아닌 single-fault syndrome 하나가 **평균 9개의 (k, α) preimage** 를 가짐 (cross-CNOT syndrome collision 이 광범위).
- Decoder 는 그 syndrome 을 볼 때마다 lex-min (k, α) 를 골라 commit — true (k, α) 와는 거의 항상 다름.
- 잘못된 correction 이 데이터 큐빗 frame 에 거의 random 한 Pauli residual 을 누적.
- R1 이 "background noise 위의 느린 systematic bias" 로 천천히 모으던 dominant-CNOT signal 이 이 random 한 correction 들로 흩어짐.

R3b 의 within-group per-class 정확도 표준편차는 T=50 에서 모든 4개 그룹 ≤ 0.05 (균일하게 낮음). R1 에선 lex-min 매칭이 한 멤버를 특히 favor 해서 std 가 크지만, R3b 에선 그 favoring 효과가 깨지고 **모든 멤버가 골고루 낮음**.

### 14.6 §13.7 가설에 대한 판결

| 주장 | 결과 |
|---|---|
| "Decoder 가 모호한 두 history 에 같은 correction 을 적용하므로, 실제 residual 들이 후속 라운드의 syndrome 에 다르게 전파된다." | ✅ 확인 (JS > 0). |
| "그러므로 R3b 분류기가 ambiguity 그룹 멤버들을 구별할 수 있고 0.75 R1 ceiling 을 넘는다." | ❌ 이 decoder 한정으로 falsified. R3b 는 ≈ 0.10. |

차이는 작은 상수가 아니라 *order of magnitude*. 원인은 decoder 의 lex-min commit 정책 (window 크기나 propagator 정확성이 아니라).

### 14.7 X-stab interior 그룹을 실제로 깨려면

기대 효과 순서대로:

1. **Posterior-aware decoder**. lex-min 대신 `(p_bg, p_high)` prior 로 `(k, α)` posterior 를 계산해서 Bayes-averaged correction 또는 posterior sampling. Frame 에 가해지는 perturbation 의 평균 strength 가 낮아질 것.
2. **Multi-round window**. window > 1 이면 decoder 가 syndromes 여러 개를 보고 결정 → arbitrary tie-break 없이 single-fault hypothesis 가 disambiguate 될 수 있음.
3. **Full-sequence ML decoder**. Syndrome stream 자체를 관측으로 보고 dominant-CNOT 분류기를 직접 학습. Decoder choice 를 marginalize. R3 시나리오의 가장 강한 형태. 다음 실험 자연스러운 후보.
4. **Pauli pool 제한**. §13.6 에서 single-leg Pauli 6개를 빼면 (15→9) R1 의 G₁ = {6,7} 그룹이 깨지듯, R3b 의 marginal sharpening 에도 비슷한 효과 가능. 비용 작음.

### 14.8 프로젝트적 의미

- Task #7 ("R3b decoder") 는 window-1 lex-min decoder 에 대한 가설 검증으로 **완료**.
- 시퀀스 레벨 CNOT 식별의 headline ceiling 은 여전히 **R1 의 18/24 group 정확도** (또는 9-Pauli 일 경우 19/24).
- Task #5 (데이터셋), Task #6 (분류기) 둘 다 unblock. Classifier 평가 기준점은 여전히 R1 의 per-round marginal Bayes. R3b 는 reference 가 아니라 cautionary example.
- 구조적 그룹을 실제로 깨려면 §14.7 의 1~3 중 하나. 모두 이번에 만든 인프라로 trackable.


### 14.9 변형: HammingNearestDecoder (multi-fault → 가장 가까운 lookup)

기본 `LookupDecoder` 는 multi-fault round (detection event 가 24×15 single-fault lookup 에 없음) 에서 identity correction. p_bg = 0.01 이면 라운드 당 multi-fault 비율이 ≈ 0.24 라 이 branch 가 자주 발동. 이 multi-fault-identity 정책이 R3b 를 R1 아래로 끌어내리는 원인인지 분리하기 위해 `src/decoder.py:HammingNearestDecoder` 를 추가: lookup 에 없는 non-zero d 에 대해, lookup syndrome 중 Hamming 거리 가장 가까운 것의 correction 적용 (lex-min 으로 tie-break). d in lookup 또는 d == 0 에선 `LookupDecoder` 와 동일.

`(p_bg, p_high) = (0.01, 0.1)`, `n_train = n_test = 100`, `seed = 42`:

| T | R1 (analytic) | R3b LookupDecoder | R3b HammingNearestDecoder |
|---|---|---|---|
| 10 | 0.180 | 0.115 | 0.122 |
| 30 | 0.312 | 0.107 | 0.149 |
| 50 | 0.416 | 0.095 | 0.186 |

(per-class accuracy 기준; group accuracy 도 같은 순위, margin ≈ 1.3× 더 큼.)

T 별 trend 가 흥미로움:

1. **R1 은 T 에 monotonically 증가** (자연스러움 — log-likelihood marginal 추정 sample 증가).
2. **LookupDecoder 는 T 에 monotonically *감소*** (0.115 → 0.107 → 0.095). 긴 sequence 일수록 잘못된 correction 누적, §14.3 의 "plateau" 는 사실 천천히 *감소* 하는 것의 평균. multi-fault round identity fallback 은 정보를 다음 라운드로 carry 하지 않음.
3. **HammingNearestDecoder 는 T 에 monotonically *증가*** (0.122 → 0.149 → 0.186), R1 의 약 절반 기울기. identity fallback 을 nearest-Hamming guess 로 바꾸면 그동안 낭비되던 라운드가 부분적 신호 carrier 가 됨.

두 R3b 변형 모두 모든 T 에서 R1 보다 한참 낮음. §14.6 결론 유지 — 어떤 single-fault hypothesis decoder 도 R1 의 marginal-Bayes lower bound 에 지배됨. 단 **multi-fault fallback 정책이 lex-min tie-break 보다 더 영향력 있음** — identity → nearest-Hamming 으로 바꾸면 R3b 의 기울기 거의 두 배.

`scripts/compare_decoders.py` 가 한 (p_bg, p_high) cell 에서 T 별 곡선 생성. 출력: `data/analysis/7_r3b_ceiling/decoder_compare/decoder_compare_curves.png`.


---

## 15. R3b 하에서의 sequence-level 분리 가능성 (브랜치 `decoder-add-analysis`)

§14는 R3b의 marginal-Bayes 24-class 정확도가 T=300에서도 ~0.10 plateau로 끝났음. 그 측정은 L=1 per-round marginal만 쓰므로 *하한*에 해당. §15는 §14가 남긴 진짜 질문에 직접 답함:

> **R3b syndrome sequence 분포는 T가 커지면 dominant CNOT별로 unique 식별 가능한가, 아니면 일부 (k, k′) pair는 sequence-level에서 구조적으로 indistinguishable인가?**

이 검정은 classifier-free. 어떤 학습 알고리즘의 정확도가 아니라 *분포 자체*를 본다.

### 15.1 셋업

- 각 dominant CNOT `k ∈ {0..23}`에 대해 `N = 2000`개 R3b sequence, `T = 200`, `(p_bg, p_high) = (0.01, 0.1)`, reset, `LookupDecoder`. Symbolic simulator (`src/fast_simulator.py`); 총 ≈ 5분.
- Round 0 drop (frame이 비어 있어 decoder-agnostic).
- 코드: [scripts/test_seq_separability.py](scripts/test_seq_separability.py), [scripts/analyze_seq_separability_extra.py](scripts/analyze_seq_separability_extra.py), [scripts/analyze_d_scaling.py](scripts/analyze_d_scaling.py).
- 출력: [data/analysis/8_seq_separability/](data/analysis/8_seq_separability/).

### 15.2 측정 A — L-round marginal pairwise JS divergence

각 pair (k, k′)에 대해 L ∈ {1, 2} L-round marginal histogram의 Jensen–Shannon divergence를 sliding window로 계산. 각 k의 N 샘플을 절반으로 쪼개 JS(half₁ ‖ half₂)을 측정한 게 클래스별 sampling-noise floor (self-baseline); 이걸 빼면 signal-only 추정치.

| L | self-baseline (노이즈 floor) | within-group net mean | between-group net mean | within/between 비율 |
|---|---|---|---|---|
| 1 | 0.00177 | 0.00523 (range 0.00120–0.00851) | 0.01006 | ≈ 0.52 |
| 2 | 0.03003 | 0.01611 (range 0.00699–0.02201) | 0.03213 | ≈ 0.50 |

**해석.** Within-group JS @ L=1은 sampling 노이즈 floor의 ~3배 — 0보다 확실히 큼. Between-group JS는 within-group의 ~2배 — ambiguity-group pair는 분간이 절반 정도지만 **분리 신호는 실재**. L=2에선 절댓값이 L=1의 ~3배로 자람 → 더 긴 윈도우가 더 많은 구분 정보를 carry. within/between 비율은 ~0.5로 유지.

대표 plot: [data/analysis/8_seq_separability/js_curves_T200_N2000_main.png](data/analysis/8_seq_separability/js_curves_T200_N2000_main.png). 모든 within-group pair는 L=1, L=2 양쪽에서 between-group mean보다 *엄격히 작음*, 그러나 between-group min보다는 *큼*. (6, 7) pair가 가장 marginal.

### 15.3 측정 B — pair별 누적 log-likelihood ratio

각 R1 ambiguity-group pair (k₁, k₂)에 대해 true k = k₁ / true k = k₂에서 샘플한 sequence 각각에서 L=1 누적 log-LR

$$\Lambda_t = \sum_{s=1}^{t} \log \hat{P}(d_s \mid k_1) - \log \hat{P}(d_s \mid k_2)$$

를 계산. R1이라면 ambiguity-group 멤버에선 drift가 정확히 0이라 T-스케일링 분간 불가. R3b에선 각 pair마다 true k에 따른 positive drift가 보이고, 두 분포가 T 늘면서 점점 분리됨.

T = 200에서 각 pair의 Λ_T (mean ± std) 요약 (데이터: [cumulative_logLR_T200_N2000_main.csv](data/analysis/8_seq_separability/cumulative_logLR_T200_N2000_main.csv)):

| 그룹 | Pair (k₁, k₂) | E[Λ_T \| k₁] | E[Λ_T \| k₂] | Gap | Cohen's d |
|---|---|---|---|---|---|
| G1 | (6, 7) | +1.47 ± 4.80 | −1.51 ± 5.26 | 2.98 | 0.59 |
| G2 | (14, 15) | +6.50 ± 9.61 | −7.05 ± 11.52 | 13.55 | 1.28 |
| G2 | (14, 17) | +6.22 ± 8.65 | −6.59 ± 11.33 | 12.81 | 1.28 |
| G2 | (15, 17) | +7.23 ± 10.25 | −7.25 ± 11.47 | 14.48 | 1.33 |
| G3 | (18, 19) | +4.84 ± 9.14 | −5.30 ± 8.82 | 10.14 | 1.13 |
| G3 | (18, 20) | +4.01 ± 8.11 | −4.49 ± 8.52 | 8.50 | 1.02 |
| G3 | (19, 20) | +3.82 ± 7.23 | −3.54 ± 7.06 | 7.36 | 1.03 |
| G4 | (22, 23) | +4.02 ± 7.16 | −3.75 ± 6.87 | 7.78 | 1.11 |

Cohen's *d* = gap / pooled std. *d* ≈ 1이면 두 분포가 moderately 겹침, *d* ≥ 2면 거의 안 겹침. G2, G3, G4 (3/4 그룹)은 이미 T=200, L=1만으로 *d* > 1; G1 (6 vs 7)이 가장 어려워서 *d* ≈ 0.59.

Trajectory plot [cumulative_logLR_T200_N2000_main.png](data/analysis/8_seq_separability/cumulative_logLR_T200_N2000_main.png)에서 시각적으로도 동일: 파란 band (true k₁) 위로, 빨간 band (true k₂) 아래로 drift, 겹치는 영역이 t에 따라 줄어듦.

### 15.4 스케일링 — d(t) ∝ √t 및 명확한 분리에 필요한 T

각 pair의 trajectory에 d(t) = a · √t fit (t ≥ 30, OLS, intercept 없음). RMSE 0.04–0.15로 Gaussian 근사 노이즈 안에 들어옴 — **모든 pair에서 √t 스케일링이 통계 오차 안에 들어맞음**.

| Pair | d(T=200) | sqrt-fit a | d = 2 (≈ 95% 분리) 도달 필요 T |
|---|---|---|---|
| (6, 7) | 0.59 | 0.043 | **≈ 2,170** |
| (14, 15) | 1.28 | 0.102 | ≈ 385 |
| (14, 17) | 1.28 | 0.102 | ≈ 390 |
| (15, 17) | 1.33 | 0.100 | ≈ 400 |
| (18, 19) | 1.13 | 0.087 | ≈ 525 |
| (18, 20) | 1.02 | 0.078 | ≈ 650 |
| (19, 20) | 1.03 | 0.077 | ≈ 675 |
| (22, 23) | 1.11 | 0.082 | ≈ 590 |

Plot: [d_scaling_T200_N2000_main.png](data/analysis/8_seq_separability/d_scaling_T200_N2000_main.png) (측정 d(t) + fit), [d_projection_T200_N2000_main.png](data/analysis/8_seq_separability/d_projection_T200_N2000_main.png) (√t 외삽).

### 15.5 결론 — 직접 답변

> **Yes — R3b 하에서 syndrome sequence는 T가 커지면 dominant CNOT별로 unique 식별 가능. 모든 R1 ambiguity 그룹 내부에서도 마찬가지.**

구체적으로:
1. 모든 ambiguity-group pair에서 per-round L=1 분포 정보율이 양수 (within-group net JS > 0, sampling-noise floor 위).
2. 누적 log-LR Cohen's *d*가 T=200까지 √t로 자라며 saturation 없음.
3. √t fit 외삽 시 모든 ambiguity-group pair가 *d* = 2 (Gaussian 근사 하에서 ≈ 95% 분리, 즉 그 pair에 대한 2-class Bayes 정확도 ≥ 95%)에 유한 T에서 도달. 가장 어려운 pair (CNOT 6 vs 7)는 T ≈ 2,170 라운드 필요; 나머지는 380–700.
4. 이 모든 게 L=1 marginal만 써서 나온 결과. 더 긴 윈도우 decoder (또는 full-sequence likelihood)면 threshold가 더 낮아짐 — §14의 24-class marginal-Bayes ~0.10 plateau은 분포 자체의 한계가 아니라 *그 classifier의 한계*임이 밝혀짐.

### 15.6 §14 결과의 재해석

§14에서 R3b marginal-Bayes 24-class 정확도가 T=300에서도 ~0.10에 머문 건 *식별 가능성의 구조적 한계가 아님* — 분포 자체는 distinguishable. plateau은 24-way 동시 식별 시 finite-T 인공물: pair별 분리 정보율은 L=1에서 ~0.005–0.011 nat/round, 24-class에서 신뢰성 있게 분리하려면 log(24) ≈ 3.18 nat의 pairwise 정보가 필요. T = 300에서 worst pair는 300 × 0.005 ≈ 1.5 nat — 임계값 아래라서 plateau이 됨. T를 더 늘리거나, L을 늘리거나, non-marginal classifier를 쓰면 이 gap을 메울 수 있음.

따라서 후속 질문은 더 이상 "R3b가 indistinguishable인가?"가 아니라 "학습 classifier가 어느 T에서 asymptote에 도달하는가?" — 명확한 이론 목표가 정해진 Task #6 질문.


---

## 16. Task #6 — Sequence classifier (진행 중)

§15에서 R3b sequence 분포가 모든 (k, k′) pair에 대해 충분히 긴 T에서 distinguishable임을 확인. 헤드라인 미해결 질문은 **학습된 classifier가 실제로 per-class accuracy → 1을 달성하는가**. Task #6은 그 classifier를 만들고 세 reference에 대해 benchmark:

1. **Random baseline** = 1/24 ≈ 0.042
2. **R1 ceiling** (analytic, §13) = per-class 0.75, group 1.0 — R1 sequence에 대한 어떤 classifier도 못 넘는 한계
3. **R3b sequence-level ceiling** (§15) — T → ∞에서 per-class → 1, pairwise log-LR이 √t로 누적

Scenario 이름을 decoder 의미를 명시하는 방향으로 재정리:

| Scenario | Decoder | 의미 |
|---|---|---|
| **R1** | `IdentityDecoder` — 보정 없음 | 원래 §13 셋팅; data qubit Pauli frame 누적 |
| **R2** (다음 commit에서 추가) | `PhenomDecoder` — single-data-qubit Pauli hypothesis | 표준 surface code 식 phenomenological decoder: (data qubit q, Pauli P ∈ {X,Y,Z}) → 27 entries lookup |
| **R3b** | `LookupDecoder` — single-CNOT-fault hypothesis | §14에서 24 × 15 single-fault lookup으로 빌드한 circuit-level decoder |

(원래 R3b의 "b"는 변형 여지로 두었던 것; R2는 미사용이었는데 phenomenological 변형에 적합해 채용.)

### 16.1 빌드된 인프라

| 파일 | 역할 |
|---|---|
| [src/seq_classifier.py](src/seq_classifier.py) | `VanillaRNN` / `GRUClassifier` / `TransformerClassifier` (default 기준 ≈ 28 K / 78 K / 118 K 파라미터), `SyndromeEncoder` (`raw` 8-bit 선형 또는 `byte` 256-token embedding), `ModelConfig` + `build_model` registry |
| [scripts/build_classifier_dataset.py](scripts/build_classifier_dataset.py) | scenario × 24 class × T_max=1000 train/val/test 시퀀스 생성, 결정론 seed split |
| [scripts/train_seq_classifier.py](scripts/train_seq_classifier.py) | 단일 cell trainer (model × scenario × T): AdamW + CosineLR + val early stopping, MPS/CUDA/CPU 자동 선택, held-out test에서 per-class + group accuracy |
| [scripts/sweep_classifiers.py](scripts/sweep_classifiers.py) | (model × scenario × T) grid orchestrator; `sweep_summary.csv`로 통합 |
| [scripts/analyze_classifier_sweep.py](scripts/analyze_classifier_sweep.py) | Sweep-수준 plot: `accuracy_vs_T`, `per_class_bars`, `confusion_grid` |
| [scripts/visualize_classifier_dataset.py](scripts/visualize_classifier_dataset.py) | 학습 전 데이터 sanity check: 대표 sample heatmap, 클래스 평균 event rate, 24-class fingerprint summary |

PyTorch 2.12 + MPS 백엔드 `correst_env/`에 설치.

### 16.2 생성된 데이터셋

R1과 R3b 둘 다 `(p_bg, p_high) = (0.01, 0.1)`, reset 모드, symbolic Pauli-frame simulator (`src/fast_simulator.py`) 사용. 생성 시간 각 ≈ 36분.

| Scenario | Train | Val | Test | T_max | 총 sample | NPZ 크기 |
|---|---|---|---|---|---|---|
| R1 | 2000 × 24 | 500 × 24 | 500 × 24 | 1000 | 72,000 | 48 MB |
| R3b | 2000 × 24 | 500 × 24 | 500 × 24 | 1000 | 72,000 | 50 MB |

`data/classifier_dataset/` 아래 저장 (gitignore — seed로 재생성 가능).

### 16.3 데이터 preview (qualitative 관찰)

[scripts/visualize_classifier_dataset.py](scripts/visualize_classifier_dataset.py)가 preview 3 그림 + 요약 CSV를 [data/analysis/9_classifier/dataset_preview/](data/analysis/9_classifier/dataset_preview/) 아래 생성. 학습 전 핵심 관찰:

- **R1 syndrome은 saturated**: 평균 라운드당 3.96개 event (8 bit 중) — 약 50 % bit density. IdentityDecoder가 data qubit Pauli frame을 누적시키면, 수십 라운드 안에 stabilizer 측정이 거의 uniform random처럼 보임. k를 구분하는 신호는 그 50 % 노이즈 위의 *미세한 drift*.
- **R3b syndrome은 더 sparse**: 평균 3.17 event/round (≈ 40 %). LookupDecoder가 frame을 부분 복원해서 더 뚜렷한 class fingerprint 노출. 단 X-stab interior 그룹 (G3 = {18, 19, 20})은 여전히 50 % 근처.
- **클래스 fingerprint 구조**: round + sample 평균에서 모든 R3b 클래스가 stab별 rate signature 다름; ambiguity 그룹 ({6, 7}, {14, 15, 17}, {18, 19, 20}, {22, 23})의 row들은 *시각적으로 유사하지만 동일 아님* — §15가 정량화한 작은 차이.
- **Total event count는 분간 불가**: 24개 클래스의 mean total event는 두 시나리오에서 ~2% 안에 모임. 분간은 전적으로 *어떤 stab이 *언제* 점화하는가*에서.

### 16.4 현재 상태

인프라 준비 + R1/R3b 데이터셋 + preview qualitative 검증 완료. 다음:

1. `PhenomDecoder` 추가 + R2 데이터셋 생성
2. 첫 cell 학습: GRU on R3b, T=300 — §14 marginal-Bayes 0.10 plateau를 깰까? §15 projection ceiling에 얼마나 가까이?
3. (model × scenario × T) sweep

실험 진행에 따라 결과/plot/논의가 추가됨.
