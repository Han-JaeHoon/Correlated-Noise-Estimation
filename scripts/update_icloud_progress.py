"""
완료된 셀의 결과를 iCloud Drive에 progress.md + PNG 그래프로 기록.
셀이 하나 완료될 때마다 호출된다.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

ICLOUD_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs" \
             / "ETRI/연구/Correlated Noise Estimation by Syndrome Measurement" \
             / "[자동생성] training_progress"

CLASSIFIER_DIR = Path("data/analysis/9_classifier")

ALL_CELLS = [
    ("r1",  "rnn",         128), ("r1",  "rnn",         512),
    ("r1",  "gru",         128), ("r1",  "gru",         512),
    ("r1",  "transformer", 128), ("r1",  "transformer", 512),
    ("r2",  "rnn",         128), ("r2",  "rnn",         512),
    ("r2",  "gru",         128), ("r2",  "gru",         512),
    ("r2",  "transformer", 128), ("r2",  "transformer", 512),
    ("r3b", "rnn",         128), ("r3b", "rnn",         512),
    ("r3b", "gru",         128), ("r3b", "gru",         512),
    ("r3b", "transformer", 128), ("r3b", "transformer", 512),
]

STATUS_ICON = {"done": "✅", "running": "🔄", "pending": "⏳"}


def load_cell(sc, mo, T):
    p = CLASSIFIER_DIR / sc / f"{mo}_T{T}" / "metrics.json"
    if not p.exists():
        return None
    return json.load(open(p))


def write_progress():
    ICLOUD_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    done, pending = [], []
    for cell in ALL_CELLS:
        m = load_cell(*cell)
        if m:
            done.append((cell, m))
        else:
            pending.append(cell)

    lines = []
    lines.append("# Correlated Noise Estimation — 학습 진행 상황\n")
    lines.append(f"📍 **MacBook Pro (로컬)** | 마지막 업데이트: {now}\n")
    lines.append("> ⚠️ 이 파일은 자동으로 생성됩니다. 수정하지 마세요.\n")
    lines.append("---\n")
    lines.append(f"## 전체 진행률: {len(done)} / {len(ALL_CELLS)} 셀 완료\n")

    # Progress bar
    filled = int(len(done) / len(ALL_CELLS) * 20)
    bar = "█" * filled + "░" * (20 - filled)
    lines.append(f"`[{bar}]` {len(done)}/{len(ALL_CELLS)}\n")
    lines.append("---\n")

    # Completed cells table
    lines.append("## ✅ 완료된 셀\n")
    lines.append("| 시나리오 | 모델 | T | 테스트 정확도 | 그룹 정확도 | best epoch |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for (sc, mo, T), m in done:
        overall = m.get("test_overall_acc", 0)
        group   = m.get("test_group_acc", 0)
        best_ep = m.get("best_epoch", "-")
        # 헤드라인 셀 강조
        star = " ⭐" if overall >= 0.90 else ""
        lines.append(f"| {sc.upper()} | {mo} | {T} | **{overall:.1%}**{star} | {group:.1%} | ep {best_ep} |\n")

    if pending:
        lines.append("\n## ⏳ 대기 중인 셀\n")
        for sc, mo, T in pending:
            lines.append(f"- `{sc}/{mo}/T{T}`\n")
    else:
        lines.append("\n## 🎉 전체 sweep 완료!\n")

    # Key result callout
    lines.append("\n---\n")
    lines.append("## 핵심 결과 (현재까지)\n")
    lines.append("| 항목 | 결과 |\n")
    lines.append("|---|---|\n")
    lines.append("| R1 이론 한계 | 75% (인접 게이트 쌍 구조적 모호성) |\n")

    r1_gru_512 = load_cell("r1", "gru", 512)
    if r1_gru_512:
        v = r1_gru_512["test_overall_acc"]
        lines.append(f"| R1 / GRU / T=512 | {v:.1%} (한계 근접) |\n")

    r2_gru_512 = load_cell("r2", "gru", 512)
    if r2_gru_512:
        v = r2_gru_512["test_overall_acc"]
        lines.append(f"| **R2 / GRU / T=512** | **{v:.1%} ← 한계 돌파 🏆** |\n")

    out = ICLOUD_DIR / "progress.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"[iCloud] progress.md 업데이트 완료 ({now})")

    # PNG 그래프 복사
    for png in CLASSIFIER_DIR.glob("*.png"):
        shutil.copy2(png, ICLOUD_DIR / png.name)
        print(f"[iCloud] {png.name} 복사 완료")


if __name__ == "__main__":
    write_progress()
