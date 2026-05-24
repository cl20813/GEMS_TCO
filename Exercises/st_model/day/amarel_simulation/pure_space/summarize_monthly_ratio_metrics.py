"""Summarize monthly expected-periodogram fit ratios across smooth values.

Reads files produced by real_july_spline_smooth_spectral_052126.py:

  {output_root}/smooth_0p2/2024_07/monthly_average/202407_30day_mean_curves.csv

and reports log2(I / E[I]) by variant, resolution, and frequency band.
Negative values mean the model expected periodogram is above the empirical
periodogram, i.e. model overestimates power in that band.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def smooth_tag(smooth: float) -> str:
    s = f"{float(smooth):.6g}"
    return s.replace("-", "m").replace(".", "p")


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def input_path(root: Path, smooth: float, year: int, month: int) -> Path:
    return (
        root
        / f"smooth_{smooth_tag(smooth)}"
        / f"{year}_{month:02d}"
        / "monthly_average"
        / f"{year}{month:02d}_30day_mean_curves.csv"
    )


def ratio_frame(path: Path, smooth: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "variant",
        "resolution_label",
        "resolution_stride",
        "k_mid",
        "data_spectrum",
        "theory_spectrum_expected",
        "data_k_max",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} lacks columns: {missing}")

    out = df.copy()
    out["smooth"] = float(smooth)
    out["ratio"] = pd.to_numeric(out["data_spectrum"], errors="coerce") / pd.to_numeric(
        out["theory_spectrum_expected"], errors="coerce"
    )
    out["log2_ratio"] = np.log2(out["ratio"].where(out["ratio"] > 0))
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[(out["k_mid"] > 0) & out["ratio"].notna() & out["log2_ratio"].notna()]

    kmax = out["data_k_max"].where(out["data_k_max"].notna(), out.groupby(
        ["variant", "resolution_label"]
    )["k_mid"].transform("max"))
    frac = out["k_mid"] / kmax
    out["k_frac"] = frac
    out["band"] = pd.cut(
        frac,
        bins=[0.0, 0.25, 0.75, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["smooth", "variant", "resolution_label", "resolution_stride", "band"]
    rows = []
    for keys, sub in df.groupby(group_cols, observed=True):
        vals = sub["log2_ratio"].dropna().to_numpy()
        ratios = sub["ratio"].dropna().to_numpy()
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_bins": int(vals.size),
                "k_min": float(sub["k_mid"].min()),
                "k_max": float(sub["k_mid"].max()),
                "median_ratio": float(np.nanmedian(ratios)),
                "median_log2_ratio": float(np.nanmedian(vals)),
                "mean_log2_ratio": float(np.nanmean(vals)),
                "p25_log2_ratio": float(np.nanpercentile(vals, 25)),
                "p75_log2_ratio": float(np.nanpercentile(vals, 75)),
                "frac_model_overestimate": float(np.mean(vals < 0.0)),
                "frac_model_underestimate": float(np.mean(vals > 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def compare_smooths(summary: pd.DataFrame, baseline: float, target: float) -> pd.DataFrame:
    key_cols = ["variant", "resolution_label", "resolution_stride", "band"]
    cols = key_cols + ["smooth", "median_log2_ratio", "median_ratio", "frac_model_overestimate"]
    a = summary.loc[np.isclose(summary["smooth"], baseline), cols].rename(
        columns={
            "median_log2_ratio": f"median_log2_ratio_s{smooth_tag(baseline)}",
            "median_ratio": f"median_ratio_s{smooth_tag(baseline)}",
            "frac_model_overestimate": f"frac_model_overestimate_s{smooth_tag(baseline)}",
        }
    ).drop(columns=["smooth"])
    b = summary.loc[np.isclose(summary["smooth"], target), cols].rename(
        columns={
            "median_log2_ratio": f"median_log2_ratio_s{smooth_tag(target)}",
            "median_ratio": f"median_ratio_s{smooth_tag(target)}",
            "frac_model_overestimate": f"frac_model_overestimate_s{smooth_tag(target)}",
        }
    ).drop(columns=["smooth"])
    merged = a.merge(b, on=key_cols, how="inner")
    merged["delta_median_log2_ratio_target_minus_baseline"] = (
        merged[f"median_log2_ratio_s{smooth_tag(target)}"]
        - merged[f"median_log2_ratio_s{smooth_tag(baseline)}"]
    )
    merged["more_overestimate_in_target"] = (
        merged[f"median_log2_ratio_s{smooth_tag(target)}"]
        < merged[f"median_log2_ratio_s{smooth_tag(baseline)}"]
    )
    return merged.sort_values(key_cols).reset_index(drop=True)


def write_ratio_plot(df: pd.DataFrame, out_path: Path, smooths: list[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot; matplotlib import failed: {exc}")
        return

    labels_order = sorted(
        df["resolution_label"].dropna().astype(str).unique(),
        key=lambda x: int(x.lstrip("x")),
        reverse=True,
    )
    row_specs = [("nugget0", "nugget fixed 0"), ("nugget_free", "nugget free")]
    fig, axes = plt.subplots(
        len(row_specs),
        len(labels_order),
        figsize=(4.2 * len(labels_order), 3.0 * len(row_specs)),
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(row_specs), len(labels_order))
    colors = plt.cm.viridis(np.linspace(0.12, 0.86, len(smooths)))
    color_map = {float(s): colors[i] for i, s in enumerate(smooths)}
    for i, (variant, title) in enumerate(row_specs):
        for j, label in enumerate(labels_order):
            ax = axes[i, j]
            sub = df[(df["variant"] == variant) & (df["resolution_label"].astype(str) == label)]
            if sub.empty:
                ax.set_visible(False)
                continue
            for smooth in smooths:
                ss = sub[np.isclose(sub["smooth"], smooth)].sort_values("k_mid")
                if ss.empty:
                    continue
                ax.plot(
                    ss["k_mid"],
                    ss["log2_ratio"],
                    label=f"smooth={smooth:g}",
                    linewidth=1.8,
                    color=color_map[float(smooth)],
                )
            ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle=":")
            k_cut = sub["data_k_max"].dropna()
            if not k_cut.empty:
                ax.axvline(float(k_cut.iloc[0]), color="0.65", linewidth=0.9, linestyle=":")
            ax.set_title(f"{title}, {label}")
            ax.set_xlabel("radial frequency on full-grid scale")
            if j == 0:
                ax.set_ylabel("log2(I / E[I])")
            ax.grid(alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Monthly mean ratio diagnostics: negative = model overestimates power")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="/home/jl2815/tco/exercise_output/real_data/spline_smooth_spectral_052126",
        help="Root folder containing smooth_*/YYYY_MM/monthly_average outputs.",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--smooths", default="0.2,0.45")
    parser.add_argument("--compare", default="0.2,0.45", help="Two smooths: baseline,target.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    smooths = parse_float_list(args.smooths)
    out_dir = Path(args.out_dir) if args.out_dir else root / "smooth_comparison" / f"{args.year}_{args.month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for smooth in smooths:
        path = input_path(root, smooth, args.year, args.month)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Reading {path}")
        frames.append(ratio_frame(path, smooth))

    ratios = pd.concat(frames, ignore_index=True)
    summary = summarize(ratios)

    ratio_path = out_dir / f"{args.year}{args.month:02d}_smooth_ratio_points.csv"
    summary_path = out_dir / f"{args.year}{args.month:02d}_smooth_ratio_band_metrics.csv"
    ratios.to_csv(ratio_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {ratio_path}")
    print(f"Wrote {summary_path}")

    compare_vals = parse_float_list(args.compare)
    if len(compare_vals) == 2:
        cmp_df = compare_smooths(summary, compare_vals[0], compare_vals[1])
        cmp_path = out_dir / (
            f"{args.year}{args.month:02d}_smooth_{smooth_tag(compare_vals[0])}"
            f"_vs_{smooth_tag(compare_vals[1])}_band_delta.csv"
        )
        cmp_df.to_csv(cmp_path, index=False)
        print(f"Wrote {cmp_path}")

        high = cmp_df[cmp_df["band"].astype(str) == "high"].copy()
        if not high.empty:
            print("\nHigh-k median log2(I/E[I]) comparison")
            print(
                high[
                    [
                        "variant",
                        "resolution_label",
                        f"median_log2_ratio_s{smooth_tag(compare_vals[0])}",
                        f"median_log2_ratio_s{smooth_tag(compare_vals[1])}",
                        "delta_median_log2_ratio_target_minus_baseline",
                    ]
                ].to_string(index=False)
            )

    plot_path = out_dir / f"{args.year}{args.month:02d}_smooth_log2_ratio_by_resolution.png"
    write_ratio_plot(ratios, plot_path, smooths)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
