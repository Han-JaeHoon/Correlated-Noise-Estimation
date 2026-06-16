# Phase 2 — Stim 기반 현실적 Surface Code (일반 d)

> English: [`PHASE2_PLAN.md`](PHASE2_PLAN.md)

이 브랜치(`1_realisticSurfaceCode`)는 시뮬레이션 백엔드를 PennyLane
(state-vector, d=3 하드코딩, 순차 stabilizer 스케줄)에서 **Stim**(Clifford
stabilizer 시뮬레이터)으로 이전하여 다음을 달성합니다:

1. surface code가 실제로 돌아가는 **constant-depth 병렬 stabilizer 스케줄**
   (라운드당 CNOT 4 timestep, 모든 ancilla 병렬) 사용 — Phase 0의 toy 순차
   스케줄 대체.
2. **일반 d로 확장** (d=3, 5, 7, …). state-vector로는 불가능 (d=5 → 49 큐빗 →
   9 PB 진폭).
3. Phase 0 분석(sequential 긴 시퀀스, spatial bag-of-shots)을 현실적 회로에서
   재실행.
4. spatial 분석을 **d·k 라운드** 윈도우로 확장 (k ∈ {1,2,3}, 무디코딩).

## 왜 Stim인가

우리 시스템 전체가 **Clifford + Pauli fault + 계산기저 측정**으로, 정확히
Stim의 (근사 아닌) 정확 영역입니다. Stim이 제공하는 것:
- 임의 d에 대해 correct-by-construction hook-error-aware 스케줄
- state-vector 대비 ~1000배 빠른 샘플링
- 내장 detection-event 샘플러 (우리 `(rounds, 8)` syndrome 포맷과 대응)
- detector error model(DEM) export → 향후 디코딩 / AlphaQubit류 작업

## Phase 0 vs Phase 1/2

| | Phase 0 (동결) | Phase 1 / Phase 2 (이 브랜치) |
|---|---|---|
| 브랜치 | `0_naiveSurfaceCode/{baseline,sequential,spatial,spatial-prebag}` | `1_realisticSurfaceCode` |
| 백엔드 | PennyLane state-vector | Stim |
| 스케줄 | 순차 per-stabilizer | constant-depth 4-step 병렬 |
| 거리 | d=3 하드코딩 | 일반 d |
| 상태 | 검증된 toy 결과 (fail / GRU 95.3% / LogReg 93.9%) | 진행 중 |

> 네이밍 참고: 원래 문서들은 naive 라인을 "Phase 1", 현실 라인을 "Phase 2"로
> 불렀습니다. 브랜치 재정리 이후 이는 각각 `0_naiveSurfaceCode/*`(동결) 와
> `1_realisticSurfaceCode`(활성) 네임스페이스이며, 아래의 "Phase 2"는 이
> 브랜치의 작업을 가리킵니다.

## 확인됨: Stim 스케줄 (d=3, rounds=1)

```
R   <전체 큐빗>                       # |0...0>로 reset
H   <X-ancilla>
CX  <tick 1>   ┐
CX  <tick 2>   │  4-step 병렬 스케줄
CX  <tick 3>   │  (모든 ancilla 동시에, 코너 하나씩)
CX  <tick 4>   ┘
H   <X-ancilla>
MR  <ancilla 전체>                    # measure + reset
```

d=3에서는 raw ancilla MR 레코드(`8·R`개)를 읽어 기존 **`(R, 8)` syndrome
포맷**으로 reshape — 다운스트림 파이프라인(`bag_classifier.py`, 학습/sweep/
confusion 스크립트)이 reshape만으로 붙습니다. syndrome 텐서에는 Stim의
`DETECTOR`/`OBSERVABLE` 레이어를 우회합니다. Stim의 detector 집합은 basis
비대칭(round-0 X-stab 랜덤; [`PHASE2_DESIGN_kor.md`](PHASE2_DESIGN_kor.md)
§2.3–2.4 참조)이기 때문입니다. detector/DEM 경로는 향후 디코딩용으로만 보존.

## 로드맵

- [x] **2-0** Stim 레이아웃 / CNOT 스케줄 / 측정 매핑 조사 — **완료**.
      d=3/5/7 레이아웃, 4-tick 병렬 스케줄, CNOT 방향, `(R, d²−1)` reshape,
      Phase 0 대비 round-0 projection 차이를 실측 검증.
      문서: [`PHASE2_DESIGN_kor.md`](PHASE2_DESIGN_kor.md).
- [x] **2-3** 일반 d surface code — **완료**.
      `src/backend_stim/surface_code.py`의 first-principles 구성
      (`RotatedSurfaceCode(d)`), 모든 d에서 constant 4-tick depth, Stim과
      bit-for-bit 검증(ancilla 집합 + 스케줄) + d=3/5/7 거리·logical error rate
      체크. [`SURFACE_CODE_IMPL_kor.md`](SURFACE_CODE_IMPL_kor.md) 참조;
      `python scripts/validate_surface_code.py` 실행.
- [x] **2-1** Fault injection — **완료** (결정론적). `build_circuit(injections=…)`
      이 단일-큐빗 데이터 Pauli(pre-round)와 2-큐빗 post-CNOT Pauli(hook error)를
      적용. stochastic `PAULI_CHANNEL_2` p_bg/p_high 샘플링은 추가 예정.
- [ ] **2-viz** 코드 작동 방식 웹 시각화 — **완료** (이번 마일스톤):
      `viz/surface_code.html`(자체 완결), 시뮬레이터와 데이터 일치.
- [ ] **2-2** Stim에서 single-fault enumeration 재현; Phase 0 387 atom과 **구조**
      비교 (**sanity check** — 구조는 다를 것으로 예상)
- [ ] **2-4** Stim에서 spatial bag 생성; `bag_classifier` 재실행
- [ ] **2-5** spatial을 d·k 라운드(k=1,2,3)로 확장, 무디코더; (N, k, d) 대비 정확도
- [ ] **2-6** sequential 파이프라인 Stim에서 재실행
