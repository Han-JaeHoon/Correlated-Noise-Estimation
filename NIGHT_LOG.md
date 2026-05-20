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

- **Start (UTC)**: 채워질 예정
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

- [ ] 1단계 — 컨텍스트 흡수: README §12–§13, `src/ceiling.py`, `src/decoder.py` 정독
- [ ] 2단계 — R3b design 노트 (`docs/r3b_design.md`)
- [ ] 3단계 — `LookupDecoder` 구현 + 단위 테스트
- [ ] 4단계 — `sequence_runner` window-by-window 경로 구현
- [ ] 5단계 — R3b sanity check (end-to-end)
- [ ] 6단계 — R3b ceiling 계산
- [ ] 7단계 — 가설 검증 분석 + plot
- [ ] 8단계 — README §14 + HANDOFF 업데이트
- [ ] 9단계 (옵션) — 작은 데이터셋 + histogram MLE baseline

## 5. 진행 로그 (append-only)

> 매 단계 시작/완료, 결정, 발견, blocker 를 timestamp 와 함께 누적.

### 5.0 (시작)
- NIGHT_LOG.md 생성. 첫 commit + push.

## 6. 결정 기록 (이유와 함께)

> default 가 아닌 의사결정이 일어날 때마다 여기 기록.

## 7. 미해결 질문 / blocker

> 사용자에게 묻지 않고 default 로 처리한 항목과 그 이유.

## 8. 핵심 발견 요약 (실시간 업데이트)

> 가설 검증 결과 등 큰 finding 이 나올 때마다 한 줄씩.

## 9. 최종 요약 (밤샘 끝 시점에 작성)

> 사용자가 깼을 때 가장 먼저 읽을 곳.
