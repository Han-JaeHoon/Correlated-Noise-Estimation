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

## 2. 현재 위치 (commit `891eca3` 기준, 2026-05-21)

브랜치: `long-sequence-analysis`

```
main ─── 5692656 (R1 인프라 스캐폴딩)
            │
            └─ 5074b1c (sanity check + README §12)
                  │
                  └─ 891eca3 (Task #3 R1 ceiling 결과 + README §13)  ← 여기
```

| 단계 | 상태 | 무엇이 됐나 |
|---|---|---|
| R1 인프라 (Phase 1) | ✅ | decoder ABC, BackgroundElevatedSampler, sequence runner |
| Sanity check (Phase 1.5) | ✅ | main 브랜치 single-shot 결과 비트 단위 재현 |
| **Task #3 R1 ceiling (Phase 2)** | ✅ | **4개 구조적 ambiguity 그룹 발견** |
| Task #5 본 데이터셋 (Phase 4) | ⏸ pending | |
| Task #6 분류기 (Phase 5) | ⏸ pending | |
| Task #7 R3b 디코더 (Phase 6) | ⏸ **우선순위 격상됨** | §13.7 가설 검증 |

이전에 있던 Task #4 (72-pair sequence 운명)는 Task #3에 흡수돼서 별도 진행
불필요. Task #2 (파라미터 픽)는 Task #7 결과 보고 결정 권장.

## 3. 가장 중요한 발견 — Task #3 ceiling

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

**열린 가설** (README §13.7): **Window decoder (R3b)를 추가하면 구조적
그룹도 깨질 수 있을 것으로 추측**. 동일 syndrome → 동일 correction → 다른
실제 residual → 후속 라운드에서 구별 가능한 흔적. **미검증, 다음 step의
주된 동기.**

## 4. 다음 step 추천 순서

1. **Task #7 부분 착수 (R3b decoder)** ★ — `src/decoder.py`에 lookup
   decoder 클래스 구현, `src/sequence_runner.py:72`의 `NotImplementedError`
   경로 채우기 (window state hand-off). R3b ceiling 계산해서 §13.7 가설
   직접 검증.
2. **Task #5 본 데이터셋** — 24 faulty_cnot × T sweep × N. group accuracy
   메트릭 기반.
3. **Task #6 분류기** — histogram MLE → MLP → GRU/Transformer, ceiling과
   gap 비교.

## 5. 핵심 파일 맵

**코드:**
| 파일 | 역할 |
|---|---|
| `src/ceiling.py` | Task #3 core: XOR-convolution + Bayes classifier |
| `src/decoder.py` | `Decoder` ABC, R3b 확장점 |
| `src/sequence_runner.py:72` | R3b 미구현 경로 (다음 task에서 채울 곳) |
| `src/stochastic_faults.py` | `BackgroundElevatedSampler` |
| `src/sequence_dataset.py` | R1 데이터셋 생성 헬퍼 |

**스크립트:**
| 파일 | 역할 |
|---|---|
| `scripts/sanity_check_r1_infra.py` | 회귀 테스트 (먼저 돌릴 것) |
| `scripts/compute_r1_ceiling.py` | (p_bg, p_high, T) sweep CLI |
| `scripts/analyze_ceiling_groups.py` | 그룹 구조 + 15p vs 9p 비교 |
| `scripts/generate_r1_sequences.py` | R1 데이터셋 생성 CLI |

**산출물:**
| 경로 | 내용 |
|---|---|
| `data/analysis/6_sequence_ceiling/lookup_reset.npy` | 24×15 single-fault syndrome lookup |
| `data/analysis/6_sequence_ceiling/ambiguity_groups.json` | 4개 그룹의 정식 정의 |
| `data/analysis/6_sequence_ceiling/ceiling_compare_curves.png` | 15p vs 9p 비교 plot |
| `data/analysis/6_sequence_ceiling/ceiling_compare_grid.csv` | sweep 결과 표 |

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
python scripts/sanity_check_r1_infra.py
# 기대 출력 끝줄: "OVERALL: PASS"

# Task #3 결과 재현 — ~3분
python scripts/analyze_ceiling_groups.py
# 기대 출력에 다음이 포함:
#   "15-Pauli ambiguity groups: [[6, 7], [14, 15, 17], [18, 19, 20], [22, 23]]"
#   "  asymptotic per-class ceiling = 0.7500"
#   " 9-Pauli ambiguity groups: [[14, 15, 17], [18, 19, 20], [22, 23]]"
#   "  asymptotic per-class ceiling = 0.7917"
```

두 스크립트 다 통과하면 코드 상태 정상 — 거기서부터 다음 task 진행 가능.

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
