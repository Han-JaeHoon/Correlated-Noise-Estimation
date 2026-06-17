# 초전도 surface code 소자에서 고려해야 할 correlated error 조사 (노이즈 모델링용)

> English: [`CORRELATED_NOISE_SURVEY.md`](CORRELATED_NOISE_SURVEY.md)

**목적.** 이 프로젝트의 목표는 신드롬 데이터로부터 실제 소자의 **큐비트 간 correlation을
검출·분류**하는 것입니다. 그러려면 데이터를 뽑는 노이즈 모델이 임의가 아니라 물리적으로
근거 있어야 하므로, 초전도 surface code 하드웨어에서 실제로 발생하는 상관 오류를
조사했습니다. 범위: 초전도 큐비트(Google / IBM) 중심, 모델링에 직결되는 정보.

## 핵심 요약

- 실제 상관 오류는 약 5가지 메커니즘으로 나뉩니다(아래 표).
- **가장 지배적이고 가장 해로운** 상관 오류는 **2-큐비트 게이트(CZ/CNOT) *중* 발생하는
  stray `ZZ`** 로, **데이터 ↔ 보조(측정) 큐비트**를 결합합니다. 이는 기존에 쓰던
  "CNOT 직후 2-큐비트 Pauli 주입" 모델이 이미 포착하던 것과 본질적으로 같습니다.
- 상관 오류는 **신드롬 데이터에서 검출 가능**합니다: 실제 실험에서는 **detector–detector
  상관행렬 `pij`** 의 비대각 구조로 진단하고, **detector error model(DEM)의 hyperedge**
  로 표현합니다. 단일-detector 통계(detection probability)로는 상관이 안 보이며,
  쌍/시퀀스 수준 분석이 필요합니다 — 이것이 상관 분류기가 학습할 신호입니다.

## 메커니즘

| 메커니즘 | 결합 쌍 | Pauli / 채널 | 공간 · 시간 | 크기(수치) |
|---|---|---|---|---|
| **게이트 중 stray `ZZ`** (지배적) | **데이터 ↔ 보조**(게이트 쌍), 이웃 | 상관 `ZZ` + swap-like | 최근접, 게이트 시점(단일 라운드) | data–ancilla `p_ZZ ≈ 10⁻³`, data–data `≈ 10⁻⁴`(시뮬); Google: 누설+stray ≈ **17%**(측정) |
| **always-on 잔류 `ZZ`** | 데이터-데이터(NNN), 데이터-보조 | 상관 `Z`(dephasing), `p ≈ sin²(J·t)` | 상시, 지속 | tunable coupler로 **< 50 kHz**(이상적 < 10 kHz); IBM 예 ~10 kHz |
| **readout crosstalk** | **보조-보조**(공유/주파수다중 readout), 데이터-보조 | 상관 **측정 비트플립** + dephasing | 최근접, 측정 시점 | readout이 최대 국소 오류(SI1000에서 **5p**) |
| **누설(\|2⟩) 및 전파** | 데이터→측정 큐비트, 이웃 클러스터 | 비-Pauli, 다중큐비트 상관 | 국소 클러스터, 여러 라운드 지속 | Google ~17%에 포함; DQLR 끄면 d=5 Λ ~35% 하락 |
| **우주선 / QP 버스트** | 칩 광역(다수 동시) | 동시 `T1` 붕괴(에너지 완화) | ~30 큐비트~광역; 희귀하나 지속 | **시간당 ~1회**; 감쇠 시상수 **~400 µs** |

### 메커니즘별 메모

- **게이트 중 stray ZZ** 는 surface code에 가장 크고 가장 *해로운* 상관 오류입니다.
  데이터 큐비트를 그 측정 큐비트와 *보조가 회로 중간(중첩)에 있을 때* 결합하므로 오류가
  전파되어 **검출됩니다(silent 아님)**. 대표 시뮬 연구에서 게이트 기반 데이터-보조
  crosstalk이 임계값을 가장 크게 낮춥니다(0.74% → 0.63%).
- **always-on ZZ** 는 커플러를 공유하는 큐비트 간 상관 *dephasing(Z)* 을 만듭니다.
  상시 존재하며 교과서적 "ZZ crosstalk". tunable coupler로 수십 kHz까지 억제합니다.
- **readout crosstalk** 는 **보조-보조** 상관의 자연스러운 원인입니다. 주파수다중/공유
  공진기 readout 때문에 한 큐비트의 readout이 이웃 공진기를 off-resonant하게 구동하고,
  이웃 상태가 readout에 편향을 줍니다. 상관 측정 비트플립으로 모델링하며, 인접 공진기
  주파수 detuning으로 완화합니다.
- **누설** 은 비계산 상태(\|2⟩)로의 누출로 다중큐비트 상관 오류를 만들며 *단일-detector
  지표로는 보이지 않습니다*. Google은 데이터 큐비트 누설을 측정 큐비트로 스왑(DQLR).
  국소 오류가 줄수록 상대 비중이 커질 것으로 예상됩니다.
- **우주선 / QP 버스트** 는 희귀·치명적·공간확장 상관 이벤트(다수 큐비트 동시 에너지
  완화)입니다. 오류 바닥을 만들고, 시간 상관이 있어 표준 i.i.d. 노이즈의 범위 밖입니다.

### baseline(상관 없음) 참고 수치

Google 105큐비트 Willow surface-code 메모리: 평균 `T₁ ≈ 68 µs`, `T₂,CPMG ≈ 89 µs`,
라운드당 detection probability ≈ 8%, 논리오류/사이클 ≈ `1.4×10⁻³`(d=7, NN 디코더).
**SI1000** 회로 노이즈 모델 상대 강도: 측정 ≈ **5p**, 2-큐비트 게이트 ≈ **p**, 단일
큐비트/idle ≈ **p/10** (좋은 소자에서 `p` ~ `10⁻³`).

## 신드롬 데이터에서의 검출 가능성

- DEM은 **오류 hypergraph** 입니다: 각 확률적 메커니즘이 자기가 켜는 detector들을 잇는
  hyperedge. 상관 메커니즘 = 단일 결함으로는 안 이어질 detector들을 잇는 hyperedge.
- 실험에서는 **`pij` 방법**(detector 간 Pearson 상관행렬; 공간+시간 인덱스)으로 적합하며,
  비대각 구조가 상관된 detector 쌍과 "누락된 hyperedge" 신호를 드러냅니다.
- **이 프로젝트에 대한 함의:** "어느 큐비트 쌍이 상관됐나"가 곧 `pij` 분석이 뽑는
  정보입니다. 신드롬 bag/시퀀스로 학습한 상관 분류기는 사실상 `pij` 방식의 상관 검출기를
  학습하는 것입니다. 단일-detector 통계로는 상관이 안 보이므로 쌍/시퀀스 수준 특징이
  필요하며, 그래서 bag-of-shots·시퀀스 파이프라인이 올바른 도구입니다.

## 이 프로젝트 노이즈 모델에 대한 시사점

1. **데이터 ↔ 보조 쌍을 유지하라 — 가장 덜 중요한 게 아니라 가장 중요하다.**
   실제 지배적 상관 오류가 게이트 기반 데이터-보조 `ZZ`입니다. 앞서 이 쌍을 빼는 것을
   고려했지만(측정 시점에 Z-type 보조에 주입된 `Z`는 silent이므로), 현실의 상관은 게이트
   *중*의 `ZZ`라 전파되어 검출됩니다. → 거리≤2 인접집합(d=3에서 44쌍, 데이터-보조 √2
   24쌍 포함)은 타당합니다.
2. **기존 post-CNOT 주입은 꼼수가 아니라 올바른 메커니즘이었다.** "CNOT 직후 2-큐비트
   Pauli 주입" = 게이트 중 stray `ZZ`/swap. top-down 상관 모델도 게이트-인접·최근접 쌍을
   중심에 둬야 합니다.
3. **Pauli 구조:** 데이터엔 상관 `Z`(dephasing → X-stabilizer); 보조엔 데이터-보조 쌍은
   게이트 중 `ZZ`(전파·검출), 보조-보조 쌍은 상관 readout 비트플립.
4. **크기:** baseline 단일 큐비트 `p ≈ 10⁻³`(측정 ~`10⁻²`); 상관 세기 `c ≈ 10⁻³`
   (데이터-보조)~`10⁻⁴`(데이터-데이터) — 즉 `c`는 단일오류율의 ~0.1–1배. 이 범위를 스윕.
5. **"none" 클래스** = 표준 i.i.d. 회로 노이즈(모든 쌍 상관 0).
6. **이후 선택:** 누설과 우주선 버스트는 실재하나 모델링이 더 어렵습니다(비-Pauli /
   시간상관 / 모델 외). 이후 단계로 미룹니다.

## 출처

- [Quantum error correction below the surface code threshold — Google "Willow" (arXiv 2408.13687 / Nature s41586-024-08449-y)](https://arxiv.org/html/2408.13687v1) — 누설+stray `ZZ` ≈ 예산 17%, swap-like, 버스트 시간당~1회·~400 µs 감쇠·~30큐비트, `T₁`/`T₂`, detection probability.
- [Surface Code Error Correction with Crosstalk Noise (arXiv 2503.04642)](https://arxiv.org/html/2503.04642) — 데이터-보조 vs 데이터-데이터, always-on vs gate-based `ZZ`; `p_ZZ` 10⁻³/10⁻⁴; `J = 10 kHz`; Pauli-twirl `sin²(Jt)`; 임계값 0.74% → 0.63%.
- [Scalable Method for Eliminating Residual `ZZ` Interaction (arXiv 2111.13292)](https://arxiv.org/pdf/2111.13292) — 잔류 `ZZ` 억제(< 10–50 kHz).
- [Cosmic-ray-induced correlated errors in superconducting qubit array (arXiv 2402.04245 / Nature Commun. s41467-025-59778-z)](https://arxiv.org/abs/2402.04245) — muon/γ QP 버스트, 다수 큐비트 동시 완화.
- [Resolving catastrophic error bursts from cosmic rays (Nature Physics s41567-021-01432-8)](https://www.nature.com/articles/s41567-021-01432-8) — 칩 광역 상관 버스트.
- [Learning to Decode the Surface Code … (arXiv 2310.05900)](https://arxiv.org/pdf/2310.05900) 및 `pij` / DEM-hyperedge 방법론; **SI1000** 노이즈 모델(측정 가중 5p).

> 출처 신뢰도 메모: Google의 ~17% 상관 예산, 버스트 통계, `T₁`/`T₂`는 *측정값*입니다.
> `J = 10 kHz` 와 `p_ZZ` 10⁻³/10⁻⁴ 는 crosstalk 연구의 *대표 시뮬레이션 파라미터*(IBM
> 유사)로, 단일 실측 소자값이 아니라 자릿수 가이드로 사용하십시오.
