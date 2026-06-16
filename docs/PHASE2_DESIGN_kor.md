# Phase 2 — 설계 (Step 0: 조사 + 실측 근거)

> English: [`PHASE2_DESIGN.md`](PHASE2_DESIGN.md)

상태: **설계 확정, 백엔드 코드는 아직 없음.** 이 문서는 로드맵 **2-0**의 산출물로,
백엔드 코드를 쓰기 전에 (a) constant-depth surface-code 스케줄에 대한 선행연구와
(b) Stim이 생성하는 회로의 직접 조사에 근거해 현실적 회로 설계를 확정합니다.
동반 로드맵: [`PHASE2_PLAN_kor.md`](PHASE2_PLAN_kor.md).

---

## 1. 선행연구 — constant-depth 스케줄과 hook error

Phase 0의 naive 시뮬레이터는 stabilizer를 **하나씩 순차로** 측정해서, 게이트
depth가 stabilizer 수(∝ `d²`)에 비례했습니다. surface code는 이렇게 돌리지
않습니다. 표준은 **constant-depth** cycle입니다: 모든 ancilla가 d와 무관하게
동일한 **4 CNOT timestep**으로, 병렬로 자기 plaquette를 측정합니다.

CNOT 순서를 고정하는 두 가지 제약 (Tomita & Svore 2014; Fowler et al. 2012):

1. **데이터 큐빗 충돌 없음** — 한 timestep에서 한 데이터 큐빗은 최대 하나의
   CNOT 파트너만 될 수 있음.
2. **Hook error 방향** — ancilla에 cycle 중간에 생긴 단일 fault가 **두** 데이터
   큐빗으로 전파될 수 있음("hook" error). CNOT 순서는 이 상관 쌍을 코드의
   high-distance 축과 **나란히** 정렬시켜 유효 거리를 줄이지 않도록 해야 함.
   표준 해법은 X-stabilizer CNOT을 **N(Z) 모양**, Z-stabilizer CNOT을 전치된
   모양으로 스케줄해 X-hook과 Z-hook이 서로 직교(무해)하는 방향을 향하게 함.

우리는 이 스케줄을 **재유도하지 않습니다**. Stim의 `Circuit.generated(...)`가
이미 임의 d에 대해 올바른 hook-error-aware constant-depth 스케줄을 내놓으며,
이를 정본 레퍼런스로 삼아 그 위에 구축합니다 (Gidney 2021, *Stim*).

우리 문제에서의 메모: hook error는 보통 *최소화 대상의 부채*이지만, 여기서는
**신호의 일부**입니다 — 특정 CNOT 직후의 cycle-중간 fault는 구조화된 2-큐빗
잔차를 만들고, 그 syndrome 발자국이 바로 우리가 CNOT을 역추적하려는 대상입니다.
따라서 병렬 스케줄은 단지 "더 현실적"인 게 아니라 Phase 0 대비 fault→syndrome
사상을 바꾸므로, Phase 0의 atom table / ambiguity group은 가정이 아니라
**재계산** 대상입니다.

참고문헌:
- Tomita & Svore, "Low-distance surface codes under realistic quantum noise", PRA **90**, 062320 (2014). https://arxiv.org/abs/1404.3747
- Gidney, "Stim: a fast stabilizer circuit simulator", Quantum **5**, 497 (2021). https://arxiv.org/abs/2103.02202
- hook error / 스케줄 배경: Dennis–Kitaev–Landahl–Preskill (2002); Fowler et al., PRA **86**, 032324 (2012).

---

## 2. 실측 근거 — Stim이 실제로 생성하는 것

`stim==1.16.0`,
`Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=R)`에서
직접 측정:

### 2.1 라운드 구조 (d=3 검증)

```
R   <전체 큐빗>                       # |0...0>로 reset
H   <X-ancilla>
CX  <tick 1>   ┐  6쌍 (d=3)
CX  <tick 2>   │  constant-depth
CX  <tick 3>   │  4-step 병렬 스케줄
CX  <tick 4>   ┘
H   <X-ancilla>
MR  <ancilla 전체>                    # measure + reset
```

### 2.2 레이아웃은 예상대로 scaling (d = 3, 5, 7 측정)

| d | data (`d²`) | ancilla (`d²−1`) | X-anc | Z-anc | directed CX / round | ticks |
|---|---|---|---|---|---|---|
| 3 | 9  | 8  | 4  | 4  | **24** | 4 |
| 5 | 25 | 24 | 12 | 12 | **80** | 4 |
| 7 | 49 | 48 | 24 | 24 | **168** | 4 |

- **데이터 큐빗**은 odd-odd 좌표; **ancilla**는 face center, X/Z 체커보드,
  각 `(d²−1)/2`개.
- **CNOT 방향이 타입 의존**(확인): X-stabilizer는 **ancilla가 control**(H 이후);
  Z-stabilizer는 **데이터가 control**(ancilla가 target). post-CNOT 2-큐빗 Pauli가
  어디에 떨어지는지를 방향이 결정 — Phase 0의 "방향 일치 주입" 가정이 병렬
  스케줄에서 실현됨.
- d=3에서는 **24 directed CNOT/round — Phase 0의 24와 동일**, 그래서 d=3 클래스
  집합은 sanity check(2-2)에서 1:1 정렬됨 (단, *타이밍*은 다르므로 syndrome
  사상은 다름).

### 2.3 Phase 0와의 결정적 차이: round 0은 projection 라운드

Stim의 `memory_z`는 logical 코드워드 `|0_L>`가 아니라 **물리적** `|0…0>`에서
시작합니다. 따라서 **X-stabilizer는 round 0에서 랜덤**(상태가 아직 그것의 +1
고유상태가 아님); 첫 cycle이 랜덤 코드워드로 *projection*하고 round 1부터
stabilizer가 결정론적이 됩니다. Phase 0는 대신 `|0_L>`를 준비해 round 0부터
결정론적이었습니다.

**설계 귀결:** **round 0을 projection/reference 라운드**로 두고 round ≥ 1을
분석(또는 detection event = 연속 라운드 XOR)합니다. Phase 0가 §15/§17에서 이미
쓴 "drop round 0" 컨벤션과 동일해서 파이프라인 호환이 유지됩니다.

### 2.4 Syndrome 추출: Stim detector가 아니라 raw MR 레코드

`memory_z`에서 `num_measurements = 8·R + d²` (`8·R`개 ancilla MR 레코드 + `d²`개
최종 데이터 측정). 측정을 직접 샘플링해 앞쪽 `(d²−1)·R`개 레코드를 자르면
**`(R, d²−1)`**로 깔끔하게 reshape — Phase 0 raw syndrome 포맷. (검증: d=3, R=4
에서 `(shots, 4, 8)`.)

syndrome 텐서에는 Stim의 `DETECTOR`/`OBSERVABLE` 레이어를 **의도적으로
우회**합니다: Stim의 detector 집합은 basis 비대칭(d=3/R=4에서 32 detector,
경계 라운드 Z-only)인데, 우리 분석은 매 라운드 **균일한 `(R, d²−1)` raw ancilla
syndrome**을 원합니다. detector/DEM 경로는 향후 디코딩 작업용으로만 보존.

---

## 3. 설계 결정

### 3.1 백엔드 전략 — 재유도 말고 활용
`src/backend_stim/circuit.py`(현 `surface_code.py`)는 스케줄 백본을 우리 구성으로
직접 만들고 Stim과 검증하며, 구조화된 레이아웃(data/ancilla id, `(x,y)` 좌표,
X/Z 타입, **라운드별 CNOT enumeration** `[(tick, control, target), …]`)을 노출.
이 enumeration이 fault 주입 site의 인덱스 집합(d=3에서 24, …).

### 3.2 Fault 모델 — 새 스케줄 위에서 Phase 0 대응
`src/backend_stim/fault_inject.py`(현 `build_circuit(injections=…)`): §3.1
enumeration의 인덱스로 지정된 CNOT, 15개 비자명 `{I,X,Y,Z}²∖{II}` 2-큐빗 Pauli,
라운드 집합이 주어지면 해당 CNOT **직후** `(control, target)`에 Pauli를 적용한
수정 회로를 생성. stochastic 모드는 비자명 Pauli당 질량 `p/15`로
`PAULI_CHANNEL_2` 주입 — Phase 0의 "uniform 15-Pauli" `p_bg`/`p_high` 모델 재현.

### 3.3 측정 모드 — 둘 다 선택 가능, **데이터 기본값은 no-reset**
- **reset (MR)** — Stim 기본 cycle; ancilla measure 후 재초기화.
- **no-reset (M)** — `MR`을 `M`으로 교체; ancilla 상태가 라운드 간 carry
  (Phase 0 mode 2의 "차분" 시그니처).

둘 다 플래그로 노출. **데이터 생성 기본값은 no-reset** (프로젝트 결정); reset은
비교용으로 제공.

### 3.4 Syndrome 텐서 — `(R, d²−1)`, round 0은 reference
MR/M 레코드 샘플링 → `(R, d²−1)` reshape. raw와 detection-event
(`d_t = s_t ⊕ s_{t−1}`) 두 view 제공. 다운스트림은 Phase 0와 똑같이 reshape로 붙음.

### 3.5 Spatial `d·k` 라운드 윈도우 (로드맵 2-5)
spatial "shot" 하나가 **`d·k` 라운드**(k ∈ {1,2,3}, 무디코딩): d=3 → 3/6/9,
d=5 → 5/10/15. 정확도를 **`(N, k, d)`**로 측정 — "긴 shot 적게 vs 짧은 shot
많이" 정보 trade-off를 직접 답하고, mixture vocabulary가 `2^{(d²−1)·d·k}`로
커지는 양상 관찰.

---

## 4. 모듈 계획 (Step 1+)

```
src/backend_stim/
  circuit.py        # generated-schedule wrapper + layout + 라운드별 CNOT enumeration (2-3)
  fault_inject.py   # 선택 CNOT 직후 결정론/stochastic Pauli 주입 (2-1)
  syndrome.py       # 측정 샘플링 → (R, d²−1) raw + detection-event 텐서 (2-2)
```

(구현에서는 `surface_code.py` 하나에 통합되었고, fault injection은
`build_circuit(injections=…)`로 제공됨.) 다운스트림(`bag_classifier.py`,
sweep/confusion/sequential 스크립트)은 reshape 어댑터 뒤로 그대로 재사용.

## 5. Sanity check 계획 (로드맵 2-2)

1. Stim 스케줄 위에서 d=3 single-fault enumeration 구축 (24 CNOT × 15 Pauli =
   360 + idle-data atom), reset & no-reset.
2. **충돌 구조**를 Phase 0 387-atom table과 비교: silent class, CNOT별 Pauli
   uniqueness, ambiguity group. 기대: **구조 다름**(병렬 타이밍이 전파를 바꿈) —
   목표는 Phase 0 bit-for-bit 재현이 아니라 새 구조 특성화.
3. round 0이 projection 라운드로 동작하는지 교차확인 (§2.3).

## 6. 미결정 (작음, Step 1에서 결정 가능)

- 메모리 basis: `memory_z` vs `z`/`x` 둘 다 (X-fault/Z-fault 가시성). 기본
  `memory_z`; X 검출이 필요하면 `memory_x` 추가.
- 결정론적 enumeration의 fault 라운드: round 0(projection) 대신 round 1(첫
  결정론적 라운드)에 주입해 reference를 깨끗하게.
- 큰 d에서 "클래스 라벨"을 CNOT별(24/80/168)로 유지할지 coarsen할지 — d=3
  sanity check가 새 ambiguity 그래프를 보일 때까지 보류.
