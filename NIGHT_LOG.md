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
- [ ] 4단계 — R3b ceiling 계산 (3단계의 의도된 5/6단계 합쳐서 진행)
- [ ] 5단계 — 가설 검증 분석 + plot
- [ ] 6단계 — README §14 + HANDOFF 업데이트
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

## 6. 결정 기록 (이유와 함께)

> default 가 아닌 의사결정이 일어날 때마다 여기 기록.

## 7. 미해결 질문 / blocker

> 사용자에게 묻지 않고 default 로 처리한 항목과 그 이유.

## 8. 핵심 발견 요약 (실시간 업데이트)

> 가설 검증 결과 등 큰 finding 이 나올 때마다 한 줄씩.

## 9. 최종 요약 (밤샘 끝 시점에 작성)

> 사용자가 깼을 때 가장 먼저 읽을 곳.
