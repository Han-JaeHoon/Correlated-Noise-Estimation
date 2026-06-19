"""Visualize the full decoding pipeline for a chosen distance d.

Given d and an arbitrary data-qubit Pauli error (specified or random), render a
3-panel figure:

  ① Surface code + the injected error
  ② Syndrome = detection events (which Z-stabilizers fired, in which round)
  ③ MWPM decoding = how the detectors were matched (inferred error chains) +
     the verdict (predicted vs actual logical flip)

Everything is real: the circuit is built by RotatedSurfaceCode, the syndrome is
simulated, and the matching is computed by PyMatching. Works for any d.

Usage:
  python scripts/visualize_decoding.py --d 3 --error 3,3,X@1
  python scripts/visualize_decoding.py --d 5 --random 3 --seed 1
  (--error coord pauli @round, repeatable as comma list; default: a demo error)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import pymatching

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend_stim.surface_code import RotatedSurfaceCode  # noqa: E402

DATA_C = "#cdd7e6"; ZC = "#43c7e0"; XC = "#ff6b6b"
LIT = "#ffd24a"; ERRC = "#b06bff"; MATCH = "#ff5d5d"; LOGI = "#7fe08a"


def draw_lattice(ax, code, title):
    span = 2 * code.d
    # logical Z string (row y=1) — the operator a logical error must cross
    ax.plot([0.4, span - 0.4], [1, 1], color=LOGI, lw=6, alpha=0.25, zorder=0)
    ax.text(span - 0.3, 1, " Z_L", color=LOGI, va="center", fontsize=8)
    # X stabilizers (faint — not decoded in memory_Z)
    for a in code.x_ancillas:
        ax.add_patch(Rectangle((a[0]-0.32, a[1]-0.32), 0.64, 0.64, angle=0,
                     facecolor=XC, alpha=0.12, edgecolor="none", zorder=1))
    # Z stabilizers (decoded)
    for a in code.z_ancillas:
        r = plt.matplotlib.patches.RegularPolygon((a[0], a[1]), 4, radius=0.42,
                     orientation=np.pi/4, facecolor=ZC, alpha=0.30,
                     edgecolor=ZC, lw=1, zorder=2)
        ax.add_patch(r)
    # data qubits
    for c in code.data_coords:
        ax.add_patch(plt.Circle(c, 0.20, facecolor=DATA_C, edgecolor="#33405a",
                     lw=0.8, zorder=4))
    ax.set_xlim(-1, span + 1); ax.set_ylim(-1, span + 1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11)


def nearest_x_boundary(code, sx):
    return 0 if sx < code.d else 2 * code.d


def zsupport(code, stab):
    return [dc for _, dc in code.schedule[stab]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--error", type=str, default=None,
                    help='e.g. "3,3,X@1" (x,y,Pauli@round); comma-separate multiple')
    ap.add_argument("--random", type=int, default=0, help="inject N random data Paulis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    code = RotatedSurfaceCode(args.d)
    R = args.d + 1
    rng = np.random.default_rng(args.seed)

    injections = []
    if args.error:
        for tok in args.error.split(";"):
            tok = tok.strip()
            body, _, rnd = tok.partition("@")
            x, y, p = body.split(",")
            injections.append({"round": int(rnd) if rnd else 1, "pos": "pre",
                               "coord": (int(x), int(y)), "pauli": p.strip().upper()})
    if args.random:
        for _ in range(args.random):
            c = code.data_coords[rng.integers(len(code.data_coords))]
            p = "XYZ"[rng.integers(3)]
            injections.append({"round": int(rng.integers(1, R)), "pos": "pre",
                               "coord": c, "pauli": p})
    if not injections:  # default demo
        injections = [{"round": 1, "pos": "pre", "coord": (3, 3), "pauli": "X"}]

    # simulate -> detection events (interpreted by the fault-free detectors)
    clean = code.build_circuit(rounds=R)
    errc = code.build_circuit(rounds=R, injections=injections)
    meas = errc.compile_sampler(seed=args.seed + 1).sample(1)
    det, obs = clean.compile_m2d_converter().convert(
        measurements=meas, separate_observables=True)
    det = det[0].astype(np.uint8)
    dcoord = {i: tuple(float(v) for v in c) for i, c in clean.get_detector_coordinates().items()}

    # MWPM
    noisy = code.build_circuit(rounds=R, after_cnot_depolarize=1e-3, measure_flip=1e-3)
    M = pymatching.Matching.from_detector_error_model(
        noisy.detector_error_model(decompose_errors=True))
    pred = M.decode(det)
    edges = M.decode_to_edges_array(det)

    success = bool(np.all(pred.astype(int) == obs[0].astype(int)))

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    draw_lattice(axes[0], code, "① Surface code + injected error")
    for f in injections:
        x, y = f["coord"]
        axes[0].add_patch(plt.Circle((x, y), 0.30, facecolor=ERRC,
                          edgecolor="white", lw=1.2, zorder=6))
        axes[0].text(x, y, f["pauli"], color="white", ha="center", va="center",
                     fontsize=9, zorder=7)
        axes[0].text(x, y + 0.5, f"r{f['round']}", color=ERRC, ha="center",
                     fontsize=7, zorder=7)

    draw_lattice(axes[1], code, "② Syndrome (detection events)")
    lit = [i for i in range(len(det)) if det[i]]
    for i in lit:
        x, y, t = dcoord[i]
        axes[1].add_patch(plt.matplotlib.patches.RegularPolygon(
            (x, y), 4, radius=0.46, orientation=np.pi/4, facecolor=LIT,
            edgecolor="#caa12a", lw=1.5, zorder=5))
        axes[1].text(x, y, f"r{int(t)}", ha="center", va="center", fontsize=7, zorder=6)
    axes[1].text(0.5, -0.04, f"{len(lit)} detector(s) fired",
                 transform=axes[1].transAxes, ha="center", fontsize=8, color="#555")

    draw_lattice(axes[2], code, "③ MWPM decoding (matched chains)")
    for i in lit:
        x, y, t = dcoord[i]
        axes[2].add_patch(plt.matplotlib.patches.RegularPolygon(
            (x, y), 4, radius=0.46, orientation=np.pi/4, facecolor=LIT,
            edgecolor="#caa12a", lw=1.2, zorder=5))
    for e in edges.tolist():
        a, b = e
        if b == -1 or a == -1:                       # boundary edge
            node = a if b == -1 else b
            x, y, t = dcoord[node]
            bx = nearest_x_boundary(code, x)
            axes[2].add_patch(FancyArrowPatch((x, y), (bx, y), color=MATCH,
                              lw=2.2, arrowstyle="-|>", mutation_scale=12, zorder=6))
        else:                                        # internal edge
            xa, ya, ta = dcoord[a]; xb, yb, tb = dcoord[b]
            style = "-" if abs(ta - tb) < 0.5 else "--"  # solid=space, dashed=time
            axes[2].plot([xa, xb], [ya, yb], style, color=MATCH, lw=2.4, zorder=6)
            # highlight shared correction qubit (space edge)
            if style == "-":
                sa = set(zsupport(code, (int(xa), int(ya))))
                sb = set(zsupport(code, (int(xb), int(yb))))
                for q in sa & sb:
                    axes[2].add_patch(plt.Circle(q, 0.26, facecolor="none",
                                      edgecolor=MATCH, lw=2.2, zorder=7))
    verdict = ("no logical error - decode OK" if success
               else "LOGICAL ERROR - decode failed")
    fig.text(0.5, 0.045,
             f"predicted logical flip={int(pred[0])}, actual={int(obs[0][0])}  ->  {verdict}",
             ha="center", fontsize=11, weight="bold",
             color=("#2a7" if success else "#c33"))

    fig.suptitle(f"Surface code decoding — d={args.d}, rounds={R}, "
                 f"MWPM (PyMatching)", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    out = args.out or f"data/viz/decode_d{args.d}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out, "| fired:", len(lit), "| matched edges:", edges.tolist(),
          "| success:", success)


if __name__ == "__main__":
    main()
