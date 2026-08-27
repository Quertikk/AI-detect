"""Renders the per-frame fake-probability timeline for a video analysis as a PNG."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_timeline_png(frame_results: list) -> bytes:
    timestamps = [r["timestamp_sec"] for r in frame_results]
    scores = [r["prob_fake"] for r in frame_results]

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.plot(timestamps, scores, marker="o", color="#d64541", linewidth=1.5)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(fake)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-frame fake probability")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()
