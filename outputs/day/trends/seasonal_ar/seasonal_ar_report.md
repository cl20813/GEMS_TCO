# Seasonal AR Structure — GEMS TCO Spatial Mean Ozone
**Data**: 2022–2025, April–September, lat [-3,2], lon [121,131], mm_cond=10
**Scale**: Hourly (8 slots/day) · Daily · Monthly
**Method**: OLS + AIC grid search

---

## Data Structure Notes

| Feature | Detail |
|---------|--------|
| Slots per day | 8 consecutive hours (e.g. 08:00–15:00) |
| Within-day gap | ~1h |
| Cross-day gap | **~16–17h** (day boundary — NOT a 1h gap) |
| Wk notation | Y[d, h−k] — same-day lag k slots prior |
| Dk notation | Y[d−k, h] — same slot, k days ago (24k h gap) |

> **Cross-boundary lag (yesterday's last → today's first, ~17h) is intentionally excluded.**
> `Dk` only uses the same slot across days, not adjacent-slot across the day boundary.

---

## Hourly AR (n=5808, 116 specs tested)

### Best model: **W1+W2+W3+W4+W5+W6** (k=7, AIC=156.77)

| Rank | Model | k | AIC | ΔAIC |
|------|-------|---|-----|------|
| 1 | W1+W2+W3+W4+W5+W6 | 7 | 156.77 | 0.00 |
| 2 | W1+W2+W3+W4+W5+W6+W7 | 8 | 195.06 | **38.30** |
| 3 | W1+W2+W3+W4+W5 | 6 | 1772.88 | 1616.11 |
| … | W1+W2+W3+W4+D1 | 6 | 4139.34 | 3982.57 |

**Key findings:**
- **6 within-day lags optimal** — W7 adds ΔAIC=+38 (worse) → last slot (h=7) should NOT add W7
- **Cross-day lags (D1–D14) completely useless** — all ΔAIC > 3982
- ΔAIC_2nd = 38.3 → very clear winner, no ambiguity
- Interpretation: ozone within a day is strongly serially correlated over 6 hours; yesterday's same-slot has negligible predictive value

---

## Daily AR (n=726, 244 specs tested)

### Best model: **L1+L8+L28** (k=4, AIC=1223.10)

| Rank | Model | k | AIC | ΔAIC |
|------|-------|---|-----|------|
| 1 | L1+L8+L28 | 4 | 1223.10 | 0.00 |
| 2 | L1+L7+L28+L19 | 5 | 1224.36 | 1.26 |
| 3 | L1+L7+L28+L15 | 5 | 1224.89 | 1.79 |
| 4 | L1+L7+L28+L17 | 5 | 1224.94 | 1.84 |
| 5 | L1+L7+L28+L18 | 5 | 1224.95 | 1.85 |

**Key findings:**
- Top 3 lags: **L1** (1-day), **L8** (~weekly), **L28** (~monthly)
- ΔAIC_2nd = 1.26 → **ambiguous** — many 4–5 parameter models within ΔAIC<4
- L7 (exactly weekly) vs L8: L7+L28 alone gives ΔAIC=3.33, L8+L1 is slightly better → ~8-day cycle rather than strict 7
- L28 (~monthly) consistently appears in top models → monthly ozone cycle
- Practical choice: **L1+L7+L28** (ΔAIC=3.33) is the simplest interpretable model with weekly and monthly seasonality

---

## Monthly AR (n=24, 68 specs tested)

### Best model: **Baseline (intercept only)** (AIC=−1506.32)

| Rank | Model | k | AIC | ΔAIC |
|------|-------|---|-----|------|
| 1 | Baseline | 1 | −1506.32 | 0.00 |
| 2 | L1 | 2 | −1440.51 | **65.81** |
| 3 | L2 | 2 | −1377.03 | 129.29 |

**Year-over-year Pearson correlations (selected):**

| Lag | Label | r |
|-----|-------|---|
| 6 | same month prev year | 0.068 |
| 12 | same month 2yr ago | −0.134 |
| 7 | lag-7 | −0.578 ← noisy (n=17) |

**Key findings:**
- **No monthly autocorrelation** — baseline wins by ΔAIC=65.8
- Year-over-year correlation ≈ 0 (r=0.07 at lag-6)
- 24 data points (6 months × 4 years) — limited power; treat with caution
- Interpretation: inter-annual variability dominates; month-to-month carryover is negligible after centering by year-month

---

## Summary Table

| Scale | n_obs | Best model | Best AIC | ΔAIC 2nd |
|-------|-------|-----------|---------|---------|
| Hourly | 5808 | W1+W2+W3+W4+W5+W6 | 156.77 | **38.30** (clear) |
| Daily | 726 | L1+L8+L28 | 1223.10 | 1.26 (ambiguous) |
| Monthly | 24 | Baseline | −1506.32 | **65.81** (clear) |

---

## Implications for Spatiotemporal Model

| Question | Answer |
|----------|--------|
| Hourly within-day structure | Strong AR(6) — 6-slot window needed |
| Cross-day hourly predictors | Not useful (ΔAIC >3982) |
| W7 (last within-day lag) | **Do not include** — adds no information |
| Cross-boundary ~17h lag | **Not modeled, not needed** |
| Daily temporal structure | ~1-day + ~8-day + ~28-day lags |
| Monthly / inter-annual | No autocorrelation after year-month centering |
