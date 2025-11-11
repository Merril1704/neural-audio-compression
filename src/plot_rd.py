"""
Plot rate-distortion curves from the CSV exported by ulaw_experiment.py.
Generates plots for SegSNR vs bitrate and STOI vs bitrate.

Usage (Windows cmd):
  python -m src.plot_rd --csv output\metrics.csv --outdir output\plots
  # Optionally choose which bitrate to plot: entropy or nominal
  python -m src.plot_rd --csv output\metrics.csv --outdir output\plots --bitrate entropy
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Plot RD curves from CSV")
    p.add_argument("--csv", type=str, required=True, help="Path to metrics CSV")
    p.add_argument("--outdir", type=str, default="output/plots", help="Directory to save plots")
    p.add_argument("--bitrate", type=str, choices=["entropy", "nominal"], default="entropy",
                   help="Which bitrate to use for x-axis")
    p.add_argument("--aggregate", action="store_true", help="Plot averaged curves over sources with error bars")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # Choose bitrate column
    if args.bitrate == "entropy":
        df["kbps"] = df["entropy_bps"] / 1000.0
        bname = "Entropy bitrate (kbps)"
    else:
        df["kbps"] = df["nominal_bps"] / 1000.0
        bname = "Nominal bitrate (kbps)"

    # Plot SegSNR
    plt.figure(figsize=(7, 5))
    if args.aggregate:
        # Average by method+bits; bitrate also averaged
        grp = df.groupby(["method", "bits"], as_index=False).agg(
            segsnr_db_mean=("segsnr_db", "mean"),
            segsnr_db_std=("segsnr_db", "std"),
            kbps_mean=("kbps", "mean"),
            kbps_std=("kbps", "std"),
        )
        for method, g in grp.groupby("method"):
            g_sorted = g.sort_values("kbps_mean")
            plt.errorbar(g_sorted["kbps_mean"], g_sorted["segsnr_db_mean"],
                         xerr=g_sorted["kbps_std"], yerr=g_sorted["segsnr_db_std"],
                         marker="o", capsize=3, label=method)
    else:
        for method, g in df.groupby("method"):
            g_sorted = g.sort_values("kbps")
            plt.plot(g_sorted["kbps"], g_sorted["segsnr_db"], marker="o", label=method)
    plt.xlabel(bname)
    plt.ylabel("Segmental SNR (dB)")
    plt.title("Rate–Distortion: SegSNR")
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend()
    seg_path = os.path.join(args.outdir, "rd_segsnr.png")
    plt.tight_layout()
    plt.savefig(seg_path, dpi=150)
    print("Saved:", os.path.abspath(seg_path))

    # Plot STOI if present
    if "stoi" in df.columns and df["stoi"].notna().any():
        plt.figure(figsize=(7, 5))
        dff = df[df["stoi"].notna()].copy()
        if args.aggregate:
            grp = dff.groupby(["method", "bits"], as_index=False).agg(
                stoi_mean=("stoi", "mean"), stoi_std=("stoi", "std"),
                kbps_mean=("kbps", "mean"), kbps_std=("kbps", "std"),
            )
            for method, g in grp.groupby("method"):
                g_sorted = g.sort_values("kbps_mean")
                plt.errorbar(g_sorted["kbps_mean"], g_sorted["stoi_mean"],
                             xerr=g_sorted["kbps_std"], yerr=g_sorted["stoi_std"],
                             marker="o", capsize=3, label=method)
        else:
            for method, g in dff.groupby("method"):
                g_sorted = g.sort_values("kbps")
                plt.plot(g_sorted["kbps"], g_sorted["stoi"], marker="o", label=method)
        plt.xlabel(bname)
        plt.ylabel("STOI")
        plt.ylim(0.0, 1.0)
        plt.title("Rate–Distortion: STOI")
        plt.grid(True, ls=":", alpha=0.6)
        plt.legend()
        stoi_path = os.path.join(args.outdir, "rd_stoi.png")
        plt.tight_layout()
        plt.savefig(stoi_path, dpi=150)
        print("Saved:", os.path.abspath(stoi_path))


if __name__ == "__main__":
    main()
