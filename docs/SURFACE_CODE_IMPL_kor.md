# 일반 거리-d surface code — 구현 & 검증

> English: [`SURFACE_CODE_IMPL.md`](SURFACE_CODE_IMPL.md)

Phase 2 백엔드의 first-principles rotated surface code
(`src/backend_stim/surface_code.py`), 그 근거, 정확성의 증거, 그리고 그 위에
구축한 웹 시각화를 문서화합니다. 로드맵 **2-3**(일반-d 회로)과 **2-1**(fault
injection) 대부분을 충족합니다. [`PHASE2_PLAN_kor.md`](PHASE2_PLAN_kor.md)와
스케줄 조사 [`PHASE2_DESIGN_kor.md`](PHASE2_DESIGN_kor.md) 참조.

## 1. 목표

Phase 0의 toy d=3 PennyLane 시뮬레이터(stabilizer를 **순차** 측정 → 라운드당
depth가 stabilizer 수와 함께 증가)를 다음 구성으로 대체:

1. 실제 surface code가 돌리는 **constant-depth** 스케줄 — **라운드당 CNOT 4 tick,
   d와 무관**;
2. **일반 d로 확장** (3, 5, 7, …);
3. `stim.Circuit.generated`에 위임하지 않고 **명시적 기하 규칙으로 우리가 직접
   구성**한 뒤 Stim과 **검증**.

## 2. 구성 규칙 (first principles)

`2d × 2d` 격자의 정수 좌표.

**큐빗**
- **데이터**: odd-odd 점 `(2i+1, 2j+1)`, `i, j ∈ 0..d-1` → `d²`개.
- **Ancilla**: even-even face center `(x, y)`, `x, y ∈ {0,2,…,2d}`, 포함 조건:
  - 내부(`0<x<2d` 이고 `0<y<2d`): 항상; 또는
  - 위/아래 경계(`y ∈ {0, 2d}`): **X-type**일 때만; 또는
  - 좌/우 경계(`x ∈ {0, 2d}`): **Z-type**일 때만.
  코너는 제외. → 정확히 `d²−1` ancilla = `(d−1)²` 내부(weight-4) +
  `2(d−1)` 경계(weight-2).
- **타입**: `(x//2 + y//2)`이 홀수면 `X`, 짝수면 `Z`.

**Constant-depth hook-safe 스케줄.** 각 ancilla는 존재하는 corner 데이터 큐빗과
타입별로 동일한 고정 4-tick 순서로 결합:

| corner `(dx,dy)` | X-ancilla tick | Z-ancilla tick |
|---|---|---|
| `(+,+)` | 0 | 0 |
| `(−,+)` | 1 | 2 |
| `(+,−)` | 2 | 1 |
| `(−,−)` | 3 | 3 |

X와 Z가 **전치된** 순서를 써서 X-hook과 Z-hook이 직교(무해) 방향을 향함 —
표준 hook-error 회피 규칙 (Tomita & Svore 2014; Fowler et al. 2012). CNOT 방향은
타입 의존: **X-stabilizer → ancilla가 control**(H 이후); **Z-stabilizer →
데이터가 control**.

따라서 라운드당 depth는 **모든 d에서 4 CNOT tick**(+ H + MR 레이어). d와 함께
커지는 것은 *폭*(큐빗 수)뿐, depth가 아님.

## 3. 검증 증거

`scripts/validate_surface_code.py`가 **d = 3, 5, 7**에 대해 확인:

| 검사 | d=3 | d=5 | d=7 |
|---|---|---|---|
| ancilla 집합 + X/Z 타입 == Stim generated | ✅ | ✅ | ✅ |
| ancilla별 4-tick CNOT 스케줄 == Stim generated | ✅ | ✅ | ✅ |
| 라운드당 CNOT tick (constant depth) | 4 | 4 | 4 |
| detector error model 빌드 | ✅ | ✅ | ✅ |
| `shortest_graphlike_error()` 길이 == d | 3 | 5 | 7 |
| logical error rate vs Stim generated (p=1e-2, MWPM) | 0.018 / 0.021 | 0.025 / 0.025 | 0.025 / 0.026 |

레이아웃 scaling (= directed-CNOT fault site 수 = 클래스 수):

| d | data `d²` | ancilla `d²−1` | directed CNOTs / round |
|---|---|---|---|
| 3 | 9 | 8 | 24 |
| 5 | 25 | 24 | 80 |
| 7 | 49 | 48 | 168 |

구조 검사(1–2행)가 가장 강한 증거입니다: 우리가 독립적으로 구성한 ancilla
집합과 스케줄이 Stim의 hook-safe generator와 **bit-for-bit 동일**하며, 거리 /
logical-error-rate 검사가 조립된 회로가 진짜 거리-`d` 코드임을 확인합니다.

## 4. Fault injection (로드맵 2-1)

`build_circuit(..., injections=[...])`가 결정론적 Pauli fault 적용:
- `{"round": r, "pos": "pre", "coord": (x,y), "pauli": "X|Y|Z"}` — round `r`
  시작에서 단일-큐빗 데이터 오류;
- `{"round": r, "pos": "post_cx", "tick": t, "control": c, "target": tg,
  "pauli": "PP"}` — CNOT 직후 2-큐빗 Pauli (Phase 0 post-CNOT 컨벤션, 이제 병렬
  스케줄 위에서) → hook-error 연구 가능.

`cnot_enumeration()`이 라운드별 directed CNOT의 안정 정렬 리스트를 제공 — 이후
enumeration 작업(2-2)의 fault site 인덱스 집합.

## 5. Round 0은 projection 라운드

Stim(과 우리)의 memory-Z 회로는 물리적 `|0…0>`에서 시작하므로 X-stabilizer가
**round 0에서 랜덤**; 첫 cycle이 랜덤 코드워드로 projection하고 round 1부터
stabilizer가 결정론적. Phase 0는 대신 `|0_L>` 준비. 그래서 fault는 **round 1**
부터 주입하고 round 0을 reference로 둡니다 (Phase 0 §15/§17의 "drop round 0"
컨벤션).

## 6. 웹 시각화

`viz/surface_code.html` — **자체 완결 단일 파일**(서버/네트워크/CDN 불필요, 바로
열기). 빌드:

```bash
python scripts/build_viz_data.py      # 시뮬레이션 -> data/viz/surface_code_viz.json
python scripts/build_viz_html.py      # JSON 인라인 -> viz/surface_code.html
```

여기 표시되는 모든 syndrome은 **실제 회로 시뮬레이션**의 결과입니다: 각
(데이터 큐빗, Pauli)와 여러 cycle-중간 CNOT fault에 대해, 무노이즈 회로와
fault 주입 사본을 *동일 시드*로 샘플링해 raw ancilla 레코드를 XOR → fault가 매
라운드 어떤 stabilizer 측정을 뒤집는지 정확히 분리. (해석적 anticommutation
멤버십과 교차검증.)

기능: d(3/5) 선택; 데이터 큐빗 클릭 + Pauli 선택 → 점등 stabilizer; 라운드
슬라이더로 syndrome 지속(reset 모드) / detection event 확인; 4-tick CNOT
스케줄 stepper; cycle-중간 CNOT fault 선택 → hook error가 데이터 큐빗 2개로
퍼지는 모습.

![d=3: 중앙 데이터 큐빗의 Y 오류가 인접 stabilizer 4개를 모두 점등](../viz/preview_d3.png)

*d=3 — 중앙 데이터 큐빗의 `Y` 오류는 인접 stabilizer 4개(X 2 + Z 2)와 모두
anticommute해서 4개 다 점등; weight-2 경계 stabilizer는 안 건드림.*

![d=5: Z 오류가 인접 X-stabilizer 2개를 점등](../viz/preview_d5.png)

*d=5 — `Z` 오류는 인접 **X**-stabilizer 2개만 점등 (Z는 X-stabilizer와만
anticommute). 더 큰 거리에서도 회로 depth는 그대로인 채 같은 물리를 보여줌.*

## 7. 파일

| 경로 | 역할 |
|---|---|
| `src/backend_stim/surface_code.py` | `RotatedSurfaceCode(d)` — 레이아웃, 스케줄, `build_circuit`, fault injection |
| `scripts/validate_surface_code.py` | Stim 대조 구조 + 거리 + logical-error-rate 검증 |
| `scripts/build_viz_data.py` | fault→syndrome 응답 시뮬레이션 → JSON |
| `scripts/build_viz_html.py` | JSON 인라인 → 자체 완결 HTML |
| `viz/surface_code.html` | 시각화 (브라우저로 열기) |
| `viz/surface_code.template.html` | HTML/CSS/JS 소스 (데이터 placeholder) |
