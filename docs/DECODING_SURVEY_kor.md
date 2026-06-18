# Surface code 디코딩 — 조사 + MWPM baseline 구현

> English: [`DECODING_SURVEY.md`](DECODING_SURVEY.md)

**목적.** correlated-noise 데이터를 생성하기 전에, 표준 surface code 디코딩
인프라를 마련합니다: 흔히 쓰는 디코더를 조사하고, 가장 대표적인 것(MWPM)을
구현해 기존 `RotatedSurfaceCode`와 호환시킵니다.

## 1. 디코딩 문제

노이즈가 있는 QEC cycle은 **detection event**(뒤집힌 detector)를 만듭니다.
디코더는 detection event → 보정으로 매핑하고, **logical observable**이 결국
뒤집혔는지로 채점합니다. Stim은 노이즈를 **detector error model(DEM)**로
패키징합니다: 각 기본 fault가 트리거하는 detector들을 잇는 hyperedge로 이루어진
오류 hypergraph. 아래 모든 디코더가 이 DEM(또는 그 그래프 형태)을 소비합니다.

## 2. 흔히 쓰는 디코더 (실무 표준)

| 디코더 | 아이디어 | 정확도 | 속도 | 사용처 |
|---|---|---|---|---|
| **MWPM** (matching) | detector를 ≤2개 뒤집는 오류 → 그래프 edge; 뒤집힌 detector들의 최소가중 매칭 | **높음 — 기준** | 중간 | surface code threshold 연구의 **표준 baseline** |
| **Union-Find** | 클러스터를 키우/합쳐 각자 보정이 되도록 | MWPM 근접 | **빠름**(≈ 준선형) | 큰 `d`, 실시간 디코딩 |
| **BP + OSD** | belief propagation + ordered-statistics 후처리 | 높음, non-graphlike 처리 | 느림 | 일반 / **상관** (qLDPC) 코드, MWPM의 그래프 가정이 깨질 때 |
| **신경망** (예: AlphaQubit) | 데이터로 학습 | 보고된 최고 | 비쌈 | 연구 프런티어; 학습 데이터 필요 |

**왜 MWPM이 대표인가.** surface code에서는 오류가 detector를 최대 2개 뒤집는
메커니즘으로 분해되어 DEM이 **graphlike**가 되고, 매칭이 거의 정확하고 잘
이해되어 있습니다. MWPM은 다른 모든 디코더가 비교 대상으로 삼는 정확도 기준
reference이고(Dennis–Kitaev–Landahl–Preskill 2002; Fowler et al. 2012),
**PyMatching 2**(Higgott & Gidney 2023)가 Stim DEM을 직접 소비하는 사실상의
빠른 구현입니다. 즉 MWPM은 표준이자 우리 Stim 기반 코드에 가장 마찰 없이 맞는
선택입니다.

## 3. 구현 (`src/backend_stim/decoding.py`)

`MatchingDecoder`는 PyMatching을 감싸며, `RotatedSurfaceCode.build_circuit`이
이미 내놓는 DEM만으로 구동됩니다 — 추가 모델링 없음:

```
circuit ──detector_error_model(decompose_errors=True)──▶ pymatching.Matching
        ──detection event 디코딩──▶ 예측 logical flip
```

```python
from src.backend_stim import RotatedSurfaceCode, MatchingDecoder, logical_error_rate

code = RotatedSurfaceCode(5)

# 간단한 circuit-level 노이즈 모델(게이트 + 측정)로 디코더 생성
dec = MatchingDecoder.from_code(code, after_cnot_depolarize=1e-3, measure_flip=1e-3)
ler = dec.logical_error_rate(shots=100_000)          # Monte-Carlo logical error rate

# 또는 한 줄
ler = logical_error_rate(code, p=1e-3, shots=100_000)

# 직접 만든 detection event 디코딩 (shots × num_detectors) -> (shots × num_observables)
pred = dec.decode_batch(detection_events)
```

핵심 성질: 디코더는 **노이즈 모델에 무관**합니다 — 회로가 컴파일되는 DEM이면
무엇이든 디코딩합니다. 노이즈 손잡이는 회로에 있고(`after_cnot_depolarize`,
`measure_flip`, 이후 correlated 채널), DEM이 graphlike로 유지되는 한 동일한
`MatchingDecoder`가 모두 처리합니다.

## 4. 검증 (`scripts/validate_decoder.py`)

| 검사 | 결과 |
|---|---|
| 무노이즈 → logical error 0 (d=3,5,7) | ✅ LER = 0 |
| 임계점 아래(p=1e-3): d 커질수록 LER 감소 | ✅ 3.5e-4 → 7.0e-5 → 1.0e-5 (d=3→5→7) |
| (p×d) sweep이 임계점을 사이에 둠 | ✅ 곡선이 p* ≈ 0.7–1% 근처에서 교차 |

`(p × d)` logical error rate (depolarizing CNOT + 측정 flip, rate p):

| p | d=3 | d=5 | d=7 |
|---|---|---|---|
| 0.003 | 0.003 | 0.001 | 0.000 |
| 0.006 | 0.011 | 0.010 | 0.006 |
| 0.010 | 0.028 | 0.036 | 0.037 |
| 0.015 | 0.058 | 0.098 | 0.132 |
| 0.020 | 0.096 | 0.177 | 0.260 |

~0.7% 아래에서는 큰 `d`가 유리(오류 억제), 위에서는 불리 — 교과서적 임계점
교차이며, 동시에 디코더가 우리 거리-`d` 코드에 올바로 묶였음을 확인합니다.

## 5. correlated-noise 작업과의 연결

- correlated-noise 조사([`CORRELATED_NOISE_SURVEY_kor.md`](CORRELATED_NOISE_SURVEY_kor.md))가
  설명하는 DEM/`pij` 그림은 이 디코더가 소비하는 **동일한 대상**입니다: 상관
  메커니즘 = 추가 DEM hyperedge. 따라서 `build_circuit`에 상관 채널을 더하면 이
  디코더가 그대로 돌아갑니다(PyMatching이 hyperedge를 그래프로 분해).
- 진짜 상관-인지 디코딩이 필요하면 **BP+OSD**가 자연스러운 업그레이드입니다
  (graphlike DEM을 가정하지 않음). MWPM은 baseline으로 남고, BP+OSD는 문서화된
  다음 옵션이며 여기서 구현하지는 않았습니다.
- 디코더는 인프라입니다: correlated-pair *분류* 과제는 syndrome에 직접
  작동하지만, 표준 디코더가 있으면 logical error rate 벤치마크와 분류기 분석이
  재사용할 DEM 도구를 확보합니다.

## 출처

- [Review on the decoding algorithms for surface codes (arXiv 2307.14989)](https://arxiv.org/html/2307.14989v4) — MWPM vs Union-Find vs BP+OSD vs 신경망.
- [PyMatching 2 (Higgott & Gidney)](https://pymatching.readthedocs.io/en/stable/) — `Matching.from_detector_error_model`, `decode_batch`.
- Dennis, Kitaev, Landahl, Preskill (2002); Fowler, Mariantoni, Martinis, Cleland, PRA 86, 032324 (2012) — surface code + MWPM.
