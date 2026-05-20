# NIGHT_LOG — `decoder-add-analysis` 브랜치 밤샘 작업 진행 일지

이 문서는 사용자가 자는 동안 Claude 가 수행한 모든 작업의 **append-only
진행 로그** 입니다. 모든 단계 / 결정 / 발견을 시간순으로 누적합니다.

작업 종료 시 마지막 섹션 ("9. 최종 요약") 이 채워집니다.

---

## 1. 목표

README §13.7 의 가설 직접 검증:
> *"Window decoder (R3b) 가 X-stab interior 그룹 (`{14,15,17}`, `{18,19,20}`,
> `{22,23}`) 을 깨뜨릴 것이다 — 동일 syndrome → 동일 correction 이지만, 실제
> residual error 가 다르면 후속 라운드에서 구별 가능한 흔적이 남는다."*

Deliverables:
- (A) `LookupDecoder` 클래스 (`src/decoder.py` 확장)
- (B) `src/sequence_runner.py` 의 window-by-window 경로 (현재 `NotImplementedError`)
- (C) R3b ceiling 계산 (`src/ceiling.py` 확장 또는 신규)
- (D) 가설 검증 결과 (X-stab interior 그룹이 깨지는가?)
- (E) README §14 (영/한) + HANDOFF.md 업데이트

## 2. 시작 시각 / 환경

- **Start (UTC)**: 2026-05-20 16:52
- **Branch**: `decoder-add-analysis` (분기점: `701680f`)
- **모델**: Opus 4.7 (1M context)

## 3. 사용자가 위임한 default 결정사항 (자기 전 OK 받음)

| 항목 | 선택값 |
|---|---|
| Branch 정책 | `decoder-add-analysis` 에 누적 push, `long-sequence-analysis` 는 건드리지 않음 |
| Window size 출발점 | 1 라운드 (가장 단순, 가설에 직접) |
| R3b ceiling 방법 | analytic 가능 영역은 analytic, 그 외 Monte Carlo |
| T grid | Task #3 와 동일 (대략 [10, 50, 100, 500]) |
| Tie-break 정책 (decoder 가 동일 syndrome 후보 여러 개 만났을 때) | 가장 작은 CNOT id 선택 (deterministic) |
| PR 생성 | **하지 않음** (사용자 명시 요청 시에만) |
| Stop hook 알림 | 자동 push 로 거의 안 울릴 것이라 그대로 둠 |

## 4. Plan checklist

- [x] 1단계 — 컨텍스트 흡수: README §12–§13, `src/ceiling.py`, `src/decoder.py`, `src/sequence_runner.py`, `src/simulator.py`, `src/stabilizer_circuit.py` 정독 (`296e895`)
- [x] 2단계 — R3b design 노트 (`docs/r3b_design.md`)
- [x] 3a — `src/pauli_frame.py` symplectic representation (`2c242f1`)
- [x] 3b — `src/round_propagation.py` + sanity check (`7d33c84`)
- [x] 3c — `src/decoder.py` `LookupDecoder` (`b7d8117`)
- [x] 3d — `src/sequence_runner.py` window-by-window 경로 (`1761603`)
- [x] 3e — R3b runner smoke test (smoke + observation 기록)
- [x] 4a — `src/ceiling_r3b.py` MC marginal-Bayes ceiling estimator (`182a347`)
- [x] 4b — `src/fast_simulator.py` propagator-only sim (~100× speedup) + CLI (`26dbb15`)
- [x] 4c — 첫 R3b sweep + 분석 + plot (`52e29ae`)
- [x] 5/6a — README §14 영/한 + HANDOFF 업데이트 (`d76f39a`, 진행 중)
- [⏳] 더 큰 grid sweep (3 p값 × 3 T값, n=200) — 백그라운드
- [ ] 7단계 (옵션) — 작은 데이터셋 + histogram MLE baseline

## 5. 진행 로그 (append-only)

> 매 단계 시작/완료, 결정, 발견, blocker 를 timestamp 와 함께 누적.

### 5.0 (시작) — 2026-05-20 16:52 UTC
- NIGHT_LOG.md 생성. 첫 commit + push (`296e895`).

### 5.1 컨텍스트 흡수 — ~17:00 UTC
정독 파일: `HANDOFF.md`, `README.md §12–§13`, `src/ceiling.py`, `src/decoder.py`,
`src/sequence_runner.py`, `src/simulator.py`, `src/stabilizer_circuit.py`.

핵심 발견:
1. **회로 게이트 순서 명확** — Z stabs (4개, 각 Z stab 안에서 data→anc CNOT 순회)
   → X stabs (4개, 각 X stab 은 H + anc→data CNOTs + H) → ancilla measure(reset).
   순서: `[Z0, Z1, Z2, Z3, X0, X1, X2, X3]`. Fault 는 특정 CNOT 직후에 inject.
2. **회로에 Pauli frame 주입 가능** — `make_repeated_stabilizer_qnode` 가 이미
   `initial_error_list / initial_error_wires` 인자를 받음. 매 라운드 새 1-round QNode
   를 호출하면서 누적 frame 을 주입하는 패턴이 가능.
3. **R3b 구현 핵심 challenge** — round 사이에 데이터 큐빗 Pauli frame 을 우리가
   직접 추적해야 함 (PennyLane 의 statevector 는 매 QNode 호출에서 폐기됨).
   라운드 안에서 일어난 fault 가 *그 후의 CNOT 들을 통해* 다른 데이터 큐빗으로
   propagate 한 결과까지 손으로 계산 필요. 이건 결정론적, 한 번 구현하면 정확.
4. **R1 lookup 재사용 가능** — `src/ceiling.py:compute_single_round_lookup` 가
   이미 24×15 single-fault syndrome lookup 을 만듦. R3b decoder 가 거꾸로
   syndrome → (cnot, pauli, residual) 매핑에 그대로 활용 가능.

### 5.2 R3b design 노트 — ~17:10 UTC
`docs/r3b_design.md` 작성. 핵심 spec:
- Window size = 1 (가설에 가장 직접 + 단순)
- LookupDecoder: 24×15 single-fault syndrome 의 역 매핑. d → (k,α,residual) 후보.
- Tie-break: lexicographic min (k,α) — deterministic.
- Multi-fault round → identity correction (보수적, 후속 라운드에 정보 누적 위임).
- Pauli frame propagation: 17-qubit symplectic vector, 게이트 in-place 업데이트.
- Round-by-round runner: 매 라운드 1-round QNode, 누적 frame 을 `initial_error_list`
  로 주입. 결과: 새 frame = 이전 frame ⊕ (이 라운드 faults' residuals) ⊕ correction.

### 5.3 LookupDecoder + window-runner 구현 (3a–3e) — 17:10–17:55 UTC
- 3a (`src/pauli_frame.py`, `2c242f1`): 17-qubit symplectic frame + 게이트 update.
  단위 테스트 8개 모두 통과.
- 3b (`src/round_propagation.py`, `7d33c84`): 라운드 회로 게이트별 propagator
  + 24×15 single-fault residual lookup. **버그 발견 & 수정**: ancilla outcome
  은 z[anc] 가 아닌 **x[anc]** (Z basis 측정 = anti-commute with X 컴포넌트).
  수정 후 PL 회로와 360/360 bit-for-bit 일치.
- 3c (`src/decoder.py`, `b7d8117`): `LookupDecoder` 클래스. window_size=1,
  lex-min tie-break, multi-fault → identity.
- 3d (`src/sequence_runner.py`, `1761603`): window-by-window 경로.
  `force_window_path=True` 옵션으로 회귀 테스트. IdentityDecoder 의 fast
  vs window 경로가 11 random seeds 에서 syndromes 완전 일치.
- 3e (`scripts/sanity_check_r3b_runner.py`, `994629a`): D/E/F smoke 모두 PASS.

### 5.4 R3b ceiling 인프라 (4a) — ~17:55 UTC
- `src/ceiling_r3b.py` (`182a347`): MC marginal-Bayes ceiling estimator.
  R3b 가 round-iid 가 아니라 R1 의 analytic XOR-convolution 이 직접 통하지
  않음. 대신 per-round marginal P(d|k) 을 MC 로 추정하고 R1 의 marginal
  Bayes classifier 그대로 적용. 이는 *true ceiling 의 lower bound* 이지만
  "R3b marginal 이 ambiguity 그룹을 깨는가" 라는 가설을 직접 답함.
  `drop_first_round=True` 가 default — round 0 는 frame=0 이라 decoder-agnostic.

## 6. 결정 기록 (이유와 함께)

> default 가 아닌 의사결정이 일어날 때마다 여기 기록.

## 7. 미해결 질문 / blocker

> 사용자에게 묻지 않고 default 로 처리한 항목과 그 이유.

## 8. 핵심 발견 요약 (실시간 업데이트)

### 발견 1 (2026-05-20 18:30 UTC) — §13.7 가설 검증 완료

**Directional claim TRUE / Quantitative claim FALSE.**

- R3b 의 per-round marginal 이 ambiguity 그룹 멤버 간 분명히 다름
  (pairwise JS 0.04–0.09; R1 에선 정확히 0). 즉 §13.7 가설의 *메커니즘*
  — decoder 가 잘못된 correction 적용 → residual 가 후속 라운드 syndrome
  에 다른 흔적 — 은 실제로 작동.
- 그러나 결과 marginal-Bayes 정확도는 R1 보다 *훨씬 낮음*.
  - (p_bg=0.01, p_high=0.1, T=50): R1 0.41 / R3b 0.10
  - T 가 커져도 R3b 는 0.10 plateau (R1 은 향상)
- 원인: lex-min single-fault decoder 가 평균 매 라운드 잘못된 correction
  적용 (각 non-zero syndrome 에 평균 9개 (k,α) preimage). 잘못된 correction
  의 누적이 dominant-CNOT 신호를 random 노이즈로 변환.
- Within-group spread: R1 std 0.003–0.263 (lex-min favoring → 한 멤버만
  ~1.0 정확도), R3b std ≤ 0.05 (모든 멤버 균등하게 나쁨).
- 단 그룹 {14,15,17} 에서 R3b 평균 정확도 (0.167) 가 R1 (0.139) 보다 살짝
  높음 — 가설 방향의 *작은* signal. 다른 X-stab interior 그룹들은 R3b 가 더
  나쁨.

상세 narrative 와 후속 액션 제안은 README §14 (영/한).

## 9. 최종 요약 (밤샘 끝 시점에 작성)

> 사용자가 깼을 때 가장 먼저 읽을 곳.
