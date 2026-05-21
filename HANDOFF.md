# HANDOFF — 다른 Claude 세션 / 사람이 이 프로젝트를 이어받을 때 읽는 문서

이 파일의 목적은 새 Claude 세션 (Claude Code 또는 claude.ai) 이나 사람이
이 프로젝트를 처음 열었을 때 **30초 안에 상황을 파악**하고 어디서부터
이어가야 할지 알 수 있게 하는 것입니다. 자세한 내용은 모두 [README.md](README.md)
에 있고, 이 문서는 그 README의 entry point + 현재 상태 요약입니다.

---

## 1. 프로젝트 한 줄 요약

`d=3` rotated surface code에서 **dominant faulty CNOT 위치를 syndrome
sequence만 보고 식별 가능한가**를 분석하는 연구. 현재 작업 브랜치는
**`long-sequence-analysis`** — main 브랜치의 single-shot frame을
sequence-level로 reframe한 **R1 시나리오**.

## 2. 현재 위치 (2026-05-21 갱신, 2회차)

활성 브랜치: **`decoder-add-analysis`** (R3b 분석 + sequence-level 분리 검증 완료, README §14+§15).
이전 작업 보존 브랜치: `long-sequence-analysis` (commit `701680f` 기준).

```
main ── 5692656 ─ 5074b1c ─ 891eca3 ─ 701680f  (long-sequence-analysis)
                                          │
                                          └─ … ─ 1501889 ─ NEW  (decoder-add-analysis)
                                                  ↑       ↑
                                                  §14    §15: sequence-level separability
                                                          (data/analysis/8_seq_separability/)
```

| 단계 | 상태 | 무엇이 됐나 |
|---|---|---|
| R1 인프라 (Phase 1) | ✅ | decoder ABC, BackgroundElevatedSampler, sequence runner |
| Sanity check (Phase 1.5) | ✅ | main single-shot 결과 비트 단위 재현 |
| Task #3 R1 ceiling (Phase 2) | ✅ | 4 구조적 ambiguity 그룹 발견 (§13) |
| Task #7 R3b 디코더 (Phase 6) | ✅ | §13.7 가설: directional TRUE / quantitative FALSE (§14) |
| **Task #7b sequence-level 분리** | ✅ | **R3b 분포가 모든 (k,k′) pair에서 distinguishable, Cohen's d ∝ √t scaling, T_required(d=2) ≈ 400–2200** (§15) |
| Task #5 본 데이터셋 (Phase 4) | ⏸ pending (unblocked) | |
| Task #6 분류기 (Phase 5) | ⏸ pending (unblocked) — 이제 theoretical target 명확 | |

이전에 있던 Task #4 (72-pair sequence 운명)는 Task #3에 흡수돼서 별도 진행
불필요. Task #2 (파라미터 픽)는 R3b 결과 봤으니 진행 가능.

## 3. 핵심 발견 1 — Task #3 ceiling

자세한 내용은 [README.md §13](README.md#13-r1-ceiling-result-phase-2--task-3).

R1 모델에서 dominant CNOT 식별의 **정보이론적 한계**가 18 그룹 / 24
CNOT으로 제한됨. 어떤 알고리즘도, 어떤 T로도 안 깨지는 **4개의 ambiguity
그룹**이 존재:

- `{6, 7}` — Z2 stab (data → ancilla), pool-dependent (9-Pauli 모델에선 깨짐)
- `{14, 15, 17}` — X1 stab 내부, **구조적**
- `{18, 19, 20}` — X2 stab 내부, **구조적**
- `{22, 23}` — X3 stab, **구조적**

**Ceilings (T → ∞)**:
- Per-class accuracy: 18/24 = **0.750** (15-Pauli) / 19/24 ≈ **0.792** (9-Pauli)
- Group accuracy: **1.000**

**구조적 이유**: 한 라운드 stabilizer 측정 순서 + data qubit의 "downstream
stabilizer 멤버십" 동일성. Pauli pool 변경으로는 못 깸. 자세한 설명은
README §13.5.

## 3.5 핵심 발견 2 — Task #7 R3b ceiling

자세한 내용은 [README.md §14](README.md#14-r3b-ceiling-result-branch-decoder-add-analysis).

§13.7 가설을 직접 검증:

| 주장 | 결과 |
|---|---|
| R3b 의 per-round marginal 이 그룹 멤버 간 다르다 (decoder 가 정보를 후속 라운드에 누설) | ✅ 확인 (pairwise JS 0.04–0.09, R1 에선 정확히 0) |
| 그러므로 R3b 가 ambiguity 그룹을 깨고 R1 ceiling 을 능가 | ❌ **거짓**. R3b 가 R1 보다 *나쁨* — R1 0.41 vs R3b 0.10 (T=50) |

원인: lex-min single-fault decoder 가 거의 매 라운드 잘못된 correction 을
적용 (각 non-zero syndrome 당 평균 9개 preimage). 잘못된 correction 의
누적 효과가 dominant-CNOT signal 을 random 노이즈로 흩뜨림.

후속 액션 후보 (README §14.7):
1. Posterior-aware decoder (Bayes-averaged correction)
2. Multi-round window
3. Full-sequence ML decoder
4. Restricted Pauli pool

## 3.6 핵심 발견 3 — Task #7b: sequence-level 분리 가능성 직접 검증

자세한 내용은 [README.md §15](README.md#15-sequence-level-separability-under-r3b-branch-decoder-add-analysis).

§14의 R3b plateau(~0.10)가 *분류기의 한계*인지 *분포 자체의 한계*인지
구분하기 위한 classifier-free 검정 (N=2000, T=200, L=1,2):

| 측정 | 결과 |
|---|---|
| Within-group L=1 marginal JS (self-baseline 빼고) | 0.005 — **양수, 분포 다름이 직접 확인** |
| Cohen's d at T=200 per pair | 0.59 (worst: 6,7) ~ 1.33 (best: 15,17) — 모두 양수 drift |
| d(t) scaling | **√t fit RMSE 0.04–0.15** — Stein's lemma 와 일치 |
| Projected T for d=2 (≈95% 분리) | 385–2,170 라운드. worst pair (6,7) ~2,170 |

→ **YES — R3b 하에서 충분히 긴 T에서 모든 24개 CNOT은 sequence-level에서 unique 식별 가능**.
§14의 0.10 plateau은 24-class 동시 식별의 finite-T 인공물이지 구조적 ceiling이 아님.

## 4. 다음 step 추천 순서

§15 결과로 우선순위 재정렬:

1. **Task #6 분류기 (Phase 5)** ★ — 이제 theoretical target이 명확:
   - **24-class accuracy → 1**이 도달 가능 (sequence-level ceiling = 1).
   - L=1 marginal-Bayes (= §14 plateau)는 *하한*. 더 똑똑한 classifier (multi-step likelihood, GRU/Transformer)가 더 잘 해야 함.
   - benchmark: T sweep으로 accuracy curve를 §15의 projected √t 외삽과 비교.
2. **Task #5 본 데이터셋 (Phase 4)** — Task #6 학습용. 24 faulty_cnot × T sweep × N.
   - §15의 T_required 값을 보고 T grid 선정: e.g. T ∈ {100, 300, 1000, 3000}로 worst pair까지 cover.
3. **(선택) L=2/3 marginal-Bayes 확장** — §15.6의 "더 큰 L"을 ablation 차원에서. L 늘릴 때 plateau가 얼마나 위로 올라가는지 측정하면 classifier에 필요한 model capacity의 directional 가이드.
4. **(선택) §14.7 의 후속 R3b decoder 변형** — posterior decoder, multi-window, restricted pool. 더 효율적인 sequence-level 분리.

## 5. 핵심 파일 맵

**코드 (Phase 1–2, `long-sequence-analysis` 시점부터):**
| 파일 | 역할 |
|---|---|
| `src/ceiling.py` | Task #3 core: XOR-convolution + Bayes classifier (R1 ceiling) |
| `src/decoder.py` | `Decoder` ABC + `IdentityDecoder` + `LookupDecoder` (R3b) |
| `src/sequence_runner.py` | fast path (Identity) + window-by-window path (R3b) |
| `src/stochastic_faults.py` | `BackgroundElevatedSampler` |
| `src/sequence_dataset.py` | R1 데이터셋 생성 헬퍼 |

**코드 (Phase 6 R3b 추가, `decoder-add-analysis`):**
| 파일 | 역할 |
|---|---|
| `src/pauli_frame.py` | 17-qubit symplectic frame, in-place gate update |
| `src/round_propagation.py` | stabilizer_round 의 symbolic propagator + single-fault residual lookup |
| `src/fast_simulator.py` | propagator-only sequence simulator (~100× faster than PennyLane runner, bit-identical) |
| `src/ceiling_r3b.py` | MC marginal-Bayes ceiling estimator |

**스크립트:**
| 파일 | 역할 |
|---|---|
| `scripts/sanity_check_r1_infra.py` | R1 회귀 테스트 (먼저 돌릴 것) |
| `scripts/sanity_check_round_propagation.py` | symbolic propagator 회귀 |
| `scripts/sanity_check_fast_simulator.py` | symbolic == PennyLane 회귀 + benchmark |
| `scripts/sanity_check_r3b_runner.py` | R3b end-to-end smoke |
| `scripts/compute_r1_ceiling.py` | (p_bg, p_high, T) R1 sweep CLI |
| `scripts/compute_r3b_ceiling.py` | (p_bg, p_high, T) R3b sweep CLI |
| `scripts/analyze_ceiling_groups.py` | R1 그룹 구조 + 15p vs 9p 비교 |
| `scripts/analyze_r3b_ceiling.py` | R3b 의 within-group spread + marginal JS |
| `scripts/generate_r1_sequences.py` | R1 데이터셋 생성 CLI |
| `scripts/test_seq_separability.py` | §15 main: 24-class × L 윈도우 JS 매트릭스 + heatmap (classifier-free) |
| `scripts/analyze_seq_separability_extra.py` | §15 보조: self-baseline JS + 누적 log-LR |
| `scripts/analyze_d_scaling.py` | §15 trajectory: Cohen's d(t) + √t fit + projection |

**산출물:**
| 경로 | 내용 |
|---|---|
| `data/analysis/6_sequence_ceiling/` | R1 ceiling: lookup, 그룹 json, sweep CSV, curves PNG |
| `data/analysis/7_r3b_ceiling/` | R3b ceiling: marginals npz, R1 vs R3b CSV, group-별 plot, within-group spread CSV, JS divergence CSV |
| `data/analysis/8_seq_separability/` | §15: pairwise JS NPZ + summary CSV + heatmaps, self-baseline + net JS, 누적 log-LR + Cohen's d trajectory + √t projection |
| `docs/r3b_design.md` | R3b decoder/runner 의 정식 spec |
| `NIGHT_LOG.md` | 2026-05-20 밤샘 작업의 시간순 진행 로그 (R3b 인프라 + 분석 통째) |

**문서:**
- [README.md](README.md) — 영어 메인 문서. §1~§11은 main 브랜치 (single-shot frame), §12는 R1 인프라, §13은 Task #3 ceiling 결과
- [README_kor.md](README_kor.md) — 한국어 버전, 동일 구조
- 이 파일 (`HANDOFF.md`)

## 6. 환경 + 운영 메모

- **Python 가상환경**: `source syndrome_env/bin/activate` (`syndrome_env/`은
  `.gitignore`에 있음, 새 환경에서는 `python -m venv syndrome_env` + `pip
  install -r requirements.txt` 로 재구축)
- **의존성**: PennyLane, numpy, matplotlib, stim
- **macOS 노이즈**: `.DS_Store`, `*.pyc`는 `.gitignore`에 있지만 과거에
  트래킹된 잔여물이 `git status`에 보일 수 있음 — **새 commit에 포함하지
  말 것**
- **R1 데이터셋 출력 경로** `data/r1_sequences/` 는 gitignore
- **기본 모드는 reset**. no-reset은 sanity check까지만 검증, ceiling은
  미분석 (reset 모드의 라운드 i.i.d. 가정이 깨지므로 별도 처리 필요)

## 7. 이어받자마자 돌려서 환경 검증

```bash
source syndrome_env/bin/activate

# R1 인프라 회귀 테스트 — ~30초
python scripts/sanity_check_r1_infra.py            # 끝줄: "OVERALL: PASS"

# R3b infra 회귀 — ~1분 (PennyLane vs symbolic)
python scripts/sanity_check_round_propagation.py   # 끝줄: "OVERALL: PASS"
python scripts/sanity_check_fast_simulator.py      # 끝줄: "OVERALL: PASS"
python scripts/sanity_check_r3b_runner.py          # 끝줄: "OVERALL: PASS"

# R1 ceiling 재현 — ~3분
python scripts/analyze_ceiling_groups.py
# 기대 출력에:
#   "15-Pauli ambiguity groups: [[6, 7], [14, 15, 17], [18, 19, 20], [22, 23]]"
#   "  asymptotic per-class ceiling = 0.7500"

# R3b ceiling 재현 (작은 grid) — ~1분
python scripts/compute_r3b_ceiling.py --p-bg 0.01 --p-high 0.1 --T 30
python scripts/analyze_r3b_ceiling.py
# 기대: T=30 cell 에서 R1 ≈ 0.31, R3b ≈ 0.09, JS divergence 0.04–0.09
```

모두 통과하면 R1 + R3b 인프라 정상 — 거기서부터 Task #6 (분류기) 진행 가능.

## 8. 대화 맥락은 어디 있나

이 프로젝트가 진행되는 동안의 Claude Code 대화 내용은 자동 sync되지
않음 (Claude Code는 로컬, claude.ai는 별개). **영구 기록은 다음 세 곳**:

1. `README.md` / `README_kor.md` — 모든 설계 결정과 결과의 narrative
2. Commit history (`git log --oneline`) — 변경 사항의 시간순 기록
3. `data/analysis/*/` — 분석 결과의 raw artifact

특정 "왜 이렇게 결정했나"가 궁금하면:
- 문제 reframing의 동기 → README §12.1
- ceiling 계산의 수학 → README §13.1
- multiset / 그룹 구조 → README §13.2–§13.5
- 9-Pauli 모델과 비교 → README §13.6
- decoder 가설 → README §13.7
- 디바이스 viability map framing → README §12.6

## 9. 새 Claude 세션 시작 시 프롬프트 예시

claude.ai 웹에서 시작할 때 첫 메시지에 붙여넣기 좋은 형태:

> 이 repo의 `long-sequence-analysis` 브랜치를 이어받으려고 합니다.
> `HANDOFF.md` 와 `README.md` §12~§13을 먼저 읽어주세요. 그 다음
> Task #7 (R3b decoder) 부터 진행할 예정입니다. 어떤 정보가 더 필요한지
> 알려주세요.

Claude Code에서는:

> long-sequence-analysis 브랜치에서 작업 이어가려고 해. HANDOFF.md
> 보고 다음 단계 plan 짜줘.
