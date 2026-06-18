# Correlated-Noise-Estimation — `1_realisticSurfaceCode` (Phase 2, 활성)

rotated surface code에서 **특정 CNOT 게이트가 주된 noise source일 때, syndrome 측정 시퀀스만으로 해당 CNOT 위치를 식별**할 수 있는지 분석하고, 그 결과를 바탕으로 학습 모델을 설계하기 위한 연구 코드.

(영문판: [`README.md`](README.md))

> **이 브랜치는 Phase 2 활성 브랜치입니다.** 백엔드를 **Stim**으로, 스케줄을
> **constant-depth 병렬**로, 거리를 **일반 `d`**로 이전한 뒤, Phase 0 분석을
> 현실적 회로에서 재실행하고 spatial 분석을 `d·k` 라운드 윈도우로 확장합니다.
>
> - **설계 + 로드맵**: [`docs/PHASE2_DESIGN.md`](docs/PHASE2_DESIGN.md) (Step 0 조사/근거), [`docs/PHASE2_PLAN.md`](docs/PHASE2_PLAN.md) (로드맵 2-0…2-6).
> - **Phase 0 (동결, 검증된 toy 결과)** — naive PennyLane `d=3` 라인은
>   `0_naiveSurfaceCode/*` 브랜치에 있고, phase 간 인덱스는 `main`에 있습니다.
>   이 브랜치에는 현실적 회로 작업만 남기고, 다운스트림에서 재사용하는 Phase 0
>   산출물 하나(387-atom 테이블, 2-2 비교용)만 보존합니다.

---

## Phase 2 진행 현황 (현실적 Stim 회로)

로드맵·상세는 [`docs/PHASE2_PLAN.md`](docs/PHASE2_PLAN.md). 한눈에 보기:

| 마일스톤 | 상태 |
|---|---|
| 2-0 Stim 스케줄 조사 + 설계 문서 | ✅ 완료 |
| 2-3 일반-d 표면부호 (상수 4-tick 깊이) | ✅ 완료, d=3/5/7 검증 |
| 2-1 결함 주입 (결정론적) | ✅ 완료 (확률적 p_bg/p_high 미구현) |
| 2-viz 자족형 웹 시각화 | ✅ 완료 |
| **2-2 단일결함 신드롬 enumeration** | 🔶 **enumeration 완료**(아래); Phase 0 구조 비교 미완 |
| 2-dec MWPM 디코더 (PyMatching) baseline | ✅ 완료 — [`docs/DECODING_SURVEY_kor.md`](docs/DECODING_SURVEY_kor.md) |
| 2-4 / 2-5 / 2-6 spatial bag · d·k 윈도우 · sequential 재실행 | ⬜ 미착수 |

### 2-2 · 단일결함 신드롬 enumeration (`scripts/build_stim_fault_enumeration.py`)

현실적 회로에서 **stabilizer 측정 1라운드**에 대해 모든 기본 결함을 나열하고 각
신드롬을 기록합니다. 물리적 `|0…0⟩` reset은 X-stabilizer의 고유상태가 **아니므로**,
단일 결함 라운드(round 1) 앞에 결함 없는 **projection 라운드**(round 0,
`|0…0⟩ → |0_L⟩`)를 둡니다. 신드롬은 round 1의 변화(detection-event)라 결함이 없으면
모두 0입니다. 이는 웹 viz가 이미 쓰던 projection convention과 동일합니다(`round 0이
projection 라운드`).

결함 종류:
- **data 단일 Pauli** — 각 data qubit에 X / Y / Z (라운드 시작 시);
- **CNOT 2-큐비트 Pauli** — 각 directed CNOT 직후 15종(II 제외). 같은 라운드의 이후
  CNOT들을 거쳐 전파 = hook error.

신드롬 추출은 검증된 차분 방식(무결함 vs 결함 주입을 같은 seed로 샘플 → 보조 측정 XOR)을
그대로 사용합니다.

| d | 보조 수 | case | 고유 신드롬 | silent(전부 0) |
|---|---|---|---|---|
| 3 | 8 | 387 (data 27 + CNOT 360) | 36 | 89 |
| 5 | 24 | 1275 (data 75 + CNOT 1200) | 168 | 261 |

거리별 출력은 `data/analysis/stim_fault_enumeration/d{3,5}/`:
`enumeration.csv`(case별 한 줄), `syndromes.npz`(신드롬 행렬 `N×(d²−1)` int8 + 메타),
`summary.txt`(개수 + silent 목록). `python scripts/build_stim_fault_enumeration.py`로
재생성(seed 고정 → 재현 가능).

참고용 Stim 회로 그림(timeline-svg)은 `viz/circuits/`에 있습니다 — 깨끗한 d=3/d=5
라운드와 D0-`Z` 결함 주입 d=3 회로.

> **2-2 남은 작업:** 이 테이블의 *구조*(충돌 / silent 클래스)를 Phase 0의 387-atom
> 테이블과 비교 — 병렬 스케줄이 fault→syndrome 맵을 바꾸므로 이 비교가 실제 sanity check.

---

## 1. 연구 목표

- **부호**: rotated surface code, 일반 거리 `d` (d=3 → 17 큐비트 = data 9 + 보조 8;
  d=5 → 49; …), constant-depth 병렬 stabilizer 스케줄.
- **가정**: directed CNOT 중 하나가 주된 결함원이며 같은 위치에서 반복 발생
  (temporally correlated).
- **목표**: syndrome 측정만으로 어느 CNOT이 결함인지 식별.
- **무시하는 nuisance**: 결함이 발생한 라운드, 결함의 Pauli 종류.
- **동기**: 작은 `d`에서는 룩업 테이블이 되지만 거리가 커지면 비용이 폭발 — 최종
  목표는 추론 시점에 질의하는 단일 학습 모델.

---

## 2. 디렉터리 구조

```text
.
├── README.md / README_kor.md
├── requirements.txt                       # numpy, stim (+ matplotlib)
├── docs/                                   # Phase 2 설계 + 로드맵 (en/kor)
│   ├── PHASE2_DESIGN*.md
│   ├── PHASE2_PLAN*.md
│   └── SURFACE_CODE_IMPL*.md
├── src/
│   ├── backend_stim/surface_code.py        # 일반-d Stim 표면부호
│   └── bag_classifier.py                   # bag-of-shots 분류기 (2-4 재사용용)
├── scripts/
│   ├── build_stim_fault_enumeration.py     # 2-2 단일결함 신드롬 테이블
│   ├── validate_surface_code.py            # 레이아웃 / 스케줄 / 거리 검증
│   ├── build_viz_data.py                   # 시뮬레이션 → viz JSON
│   └── build_viz_html.py                   # JSON → 자족형 HTML
├── viz/
│   ├── surface_code.html                   # 자족형 웹 시각화
│   ├── surface_code.template.html
│   ├── preview_d3.png / preview_d5.png
│   └── circuits/                           # 참고용 Stim 회로 그림 (svg)
└── data/
    ├── viz/surface_code_viz.json
    └── analysis/
        ├── stim_fault_enumeration/d{3,5}/  # 2-2 출력 (이 브랜치)
        └── fault_enumeration/              # Phase 0 387-atom 테이블 (2-2 비교용 보존)
```

---

## 3. `src/` 모듈

| 파일 | 역할 |
|---|---|
| `backend_stim/surface_code.py` | 일반-`d` rotated surface code 직접 구성(`RotatedSurfaceCode(d)`): 레이아웃, 상수 4-tick hook-safe 스케줄, 결정론적 결함 주입(`build_circuit(...)`: 라운드 시작 data Pauli + CNOT 직후 2-큐비트 Pauli). [`docs/SURFACE_CODE_IMPL.md`](docs/SURFACE_CODE_IMPL.md) 참조. |
| `bag_classifier.py` | bag-of-shots 분류기 모델(Empirical Bayes / LogReg / MLP / Deep Sets). 2-4에서 Stim bag으로 재실행하기 위해 보존. |

## 4. `scripts/`

| 파일 | 역할 |
|---|---|
| `build_stim_fault_enumeration.py` | 2-2 단일결함 신드롬 enumeration (위 Phase 2 진행 섹션 참조). |
| `validate_surface_code.py` | 레이아웃 / 4-tick 스케줄 / 거리 / 논리오류율을 Stim 생성기와 비교 검증(d=3/5/7). |
| `build_viz_data.py` | 결함 시뮬레이션 후 `data/viz/surface_code_viz.json` 생성. |
| `build_viz_html.py` | JSON을 자족형 `viz/surface_code.html`에 인라인. |

---

## 5. 설치 & 사용

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# 일반-d 표면부호 검증 (d=3/5/7)
python scripts/validate_surface_code.py

# 단일결함 신드롬 enumeration (d=3, d=5) → data/analysis/stim_fault_enumeration/
python scripts/build_stim_fault_enumeration.py

# 웹 시각화 재생성 후 viz/surface_code.html 열기
python scripts/build_viz_data.py && python scripts/build_viz_html.py
```

---

## 6. 가정

모든 결론은 다음 가정에 의존합니다:

1. **단일 결함원** — 한 번에 정확히 하나의 CNOT(또는 data 위치)만 결함.
2. **결정론적 Pauli 결함** — 결함 발생 시 고정 Pauli 적용(확률적 배경+상승 모델은 로드맵 2-1 / 2-4).
3. **CNOT 직후 위치** — 2-큐비트 Pauli를 CNOT 직후 주입(`build_circuit(injections=…, pos="post_cx")`)하여 라운드 잔여 게이트를 거쳐 전파(hook error).
4. **CNOT 방향성** — `CNOT(c→t)`와 `CNOT(t→c)`는 별개의 결함 사이트.
5. **측정 / 리셋 무잡음** — 보조 측정·리셋은 완전.
6. **round 0은 결함 없는 projection 라운드** — `|0_L⟩` 기준을 확립하고 결함은 round 1부터 주입(위 2-2 및 [`docs/PHASE2_DESIGN.md`](docs/PHASE2_DESIGN.md) 참조).

---

## Phase 0 참조 (다른 브랜치)

naive PennyLane `d=3` 라인 — 순차 per-stabilizer 스케줄, 216-케이스 단일 라운드 스윕,
**72 cross-Pauli cross-CNOT 충돌**, long-sequence 디코더(GRU 95.3 %), spatial
bag-of-shots 연구(LogReg 93.9 % at N=300) — 는
`0_naiveSurfaceCode/{baseline,sequential,spatial,spatial-prebag}` 브랜치에 보존돼
있습니다. phase 간 인덱스는 `main`에 있습니다.
