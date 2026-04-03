"""Generate seasonal_ar_report.pdf using matplotlib."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

OUT = "/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/seasonal_ar/seasonal_ar_report.pdf"

pages = []

# ── Page 1: Header + Data structure + Hourly ─────────────────────────────────
p1 = """SEASONAL AR STRUCTURE — GEMS TCO SPATIAL MEAN OZONE
Data: 2022–2025, April–September, lat [-3,2], lon [121,131], mm_cond=10
Method: OLS + AIC grid search (Hourly 116 specs · Daily 244 specs · Monthly 68 specs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Slots per day    : 8 consecutive hours (e.g. 08:00–15:00)
  Within-day gap   : ~1h
  Cross-day gap    : ~16–17h  ← day boundary is NOT a 1h gap!
  Wk               : Y[d, h-k]  same-day lag k slots prior
  Dk               : Y[d-k, h]  same slot k days ago (24k h gap)

  ► Cross-boundary lag (yesterday last → today first, ~17h) intentionally excluded.
    D_k uses same-slot across days only. Results justify this (D1–D14 all ΔAIC>3982).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOURLY AR   (n=5808, 116 specs tested)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Best model:  W1+W2+W3+W4+W5+W6   (k=7,  AIC=156.77,  ΔAIC_2nd=38.30)

  Rank  Model                    k     AIC        ΔAIC
  ───────────────────────────────────────────────────────
    1   W1+W2+W3+W4+W5+W6        7     156.77      0.00   ← BEST (clear winner)
    2   W1+W2+W3+W4+W5+W6+W7     8     195.06     38.30   ← W7 hurts
    3   W1+W2+W3+W4+W5            6    1772.88   1616.11
    4   W1+W2+W3+W4+D1            6    4139.34   3982.57
    …   (all D-lag models)        …     …        >3982

  Key findings:
  • 6 within-day lags optimal — ΔAIC gap of 38.3 → unambiguous winner
  • W7 (last within-day lag = slot 0 predicting slot 7) makes fit WORSE
    → Do NOT condition on W7; 6-lag window is sufficient
  • Cross-day lags D1–D14 completely uninformative (ΔAIC >3982)
    → Yesterday's same slot has negligible predictive value
  • Interpretation: strong within-day serial correlation over 6h window;
    no meaningful day-to-day carryover at hourly scale
"""
pages.append(p1)

# ── Page 2: Daily + Monthly + Summary ────────────────────────────────────────
p2 = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAILY AR   (n=726, 244 specs tested)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Best model:  L1+L8+L28   (k=4,  AIC=1223.10,  ΔAIC_2nd=1.26)

  Rank  Model              k     AIC        ΔAIC
  ──────────────────────────────────────────────────
    1   L1+L8+L28          4    1223.10      0.00
    2   L1+L7+L28+L19      5    1224.36      1.26
    3   L1+L7+L28+L15      5    1224.89      1.79
    4   L1+L7+L28+L17      5    1224.94      1.84
    5   L1+L7+L28+L18      5    1224.95      1.85
    6   L1+L7+L28+L8       5    1225.05      1.94
   17   L1+L7+L28          4    1226.43      3.33
   29   L1+L28             3    1227.25      4.15

  Key findings:
  • Top lags: L1 (1-day), L8 (~8-day/weakly), L28 (~monthly)
  • ΔAIC_2nd = 1.26 → AMBIGUOUS — many models within ΔAIC<4
  • L8 slightly beats L7 (strict weekly) → ~8-day periodicity, not exact 7
  • L28 consistently appears → ~monthly ozone cycle
  • Practical recommendation: L1+L7+L28 (ΔAIC=3.33) is simplest
    interpretable model with 1-day, weekly, and monthly components

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MONTHLY AR   (n=24, 68 specs tested)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Best model:  Baseline (intercept only)   (AIC=−1506.32,  ΔAIC_2nd=65.81)

  Rank  Model       k     AIC          ΔAIC
  ──────────────────────────────────────────
    1   Baseline    1   −1506.32        0.00   ← BEST (overwhelmingly)
    2   L1          2   −1440.51       65.81
    3   L2          2   −1377.03      129.29

  Year-over-year Pearson r:
    lag-6 (same month prev year) : r =  0.068  (n=18)
    lag-7                        : r = −0.578  (n=17, noisy)
    lag-12 (same month 2yr ago)  : r = −0.134  (n=12)

  Key findings:
  • NO monthly autocorrelation after year-month centering
  • Baseline wins by ΔAIC=65.8 — all AR lags make it worse
  • Year-over-year correlation ≈ 0
  • Caveat: only 24 data points (6 months × 4 years) — limited power

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scale    n_obs   Best model           AIC        ΔAIC_2nd  Decision
  ─────────────────────────────────────────────────────────────────────
  Hourly   5808   W1+W2+W3+W4+W5+W6    156.77     38.30     CLEAR
  Daily     726   L1+L8+L28            1223.10      1.26     AMBIGUOUS
  Monthly    24   Baseline            −1506.32     65.81     CLEAR

  Implications for spatiotemporal model:
  ┌────────────────────────────────────────────────────────────┐
  │ Hourly W7 (last within-day lag)  → Do NOT include         │
  │ Cross-day ~17h boundary lag       → Not needed            │
  │ Daily structure                   → L1 + ~weekly + ~L28   │
  │ Monthly / inter-annual            → No autocorrelation    │
  └────────────────────────────────────────────────────────────┘
"""
pages.append(p2)


def text_page(text, title=None):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    ax.text(0.02, 0.97, text, transform=ax.transAxes,
            fontsize=8.2, verticalalignment='top', fontfamily='monospace',
            wrap=False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


with PdfPages(OUT) as pdf:
    for p in pages:
        fig = text_page(p)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"Saved: {OUT}")
