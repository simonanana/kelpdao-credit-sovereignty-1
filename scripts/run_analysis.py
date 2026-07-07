"""End-to-end replication pipeline.

Usage:
    python scripts/run_analysis.py [--no-cache]

Steps:
    1. Fetch (or load cached) TVL and APY data from DefiLlama.
    2. Build the protocol-day panel around the 2026-04-18 event.
    3. Table 3  — TVL event summary; Figure 1.
    4. Figure 2 — stablecoin migration.
    5. Table 4  — four DiD specifications (Aave vs. Spark; Aave vs. Morpho).
    6. Placebo tests and Figure 3 (rolling DiD scan).
    7. Table 5  — APY statistics; Figures 4-5.
    8. Table 6  — Levene variance tests.
    9. Table 7  — mechanism (Baron-Kenny) regressions; Figures 6-7.
   10. Figure 8 — Sky RWA collateral composition; whale-adjustment check.

All tables are written to results/tables/ and figures to results/figures/.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from defi_event_study import config, figures, models  # noqa: E402
from defi_event_study.data import build_panel, fetch_token_flows, load_all_sources  # noqa: E402


def main(use_cache: bool = True) -> None:
    print(f"Event date: {config.EVENT_TS.date()}")

    # ------------------------------------------------------------------ 1-2
    tvl_src, apy = load_all_sources(use_cache=use_cache)
    panel = build_panel(tvl_src)
    if panel.empty:
        raise SystemExit(
            "Panel is empty. Either the DefiLlama API is unreachable or no "
            "cached snapshots exist in data/raw/. See data/README.md."
        )
    print("\nPanel observations per protocol:")
    print(panel.groupby("protocol")["date"].count().to_string())

    # ------------------------------------------------------------------ 3
    rows = []
    for p in ["Aave_V3", "Morpho", "Spark"]:
        d = panel[panel["protocol"] == p]
        if d.empty:
            continue
        pre_mean = d.loc[d["post"] == 0, "tvlUsd"].mean() / 1e9
        i0 = (d["date"] - config.EVENT_TS).abs().idxmin()
        i7 = (d["date"] - (config.EVENT_TS + pd.Timedelta(days=7))).abs().idxmin()
        t0, t7 = d.loc[i0, "tvlUsd"] / 1e9, d.loc[i7, "tvlUsd"] / 1e9
        rows.append([p, round(pre_mean, 2), round(t0, 2), round(t7, 2), round((t7 / t0 - 1) * 100, 1)])
    table3 = pd.DataFrame(rows, columns=["Protocol", "PreMean_B", "T0_B", "Tplus7_B", "PctChange_T0_to_T7"])
    table3.to_csv(config.RESULTS_TABLES / "table3_tvl_event_summary.csv", index=False)
    print("\nTable 3 — TVL event summary")
    print(table3.to_string(index=False))
    figures.fig1_tvl_trajectories(panel, table3)

    # ------------------------------------------------------------------ 4
    aave_flows = fetch_token_flows("aave-v3", use_cache=use_cache)
    spark_flows = fetch_token_flows("spark", use_cache=use_cache)
    figures.fig2_stablecoin_migration(aave_flows, spark_flows)

    # ------------------------------------------------------------------ 5
    res_as = models.run_four_specs(panel, "Aave_V3", "Spark")
    res_am = models.run_four_specs(panel, "Aave_V3", "Morpho")
    rows = []
    for pair, res in [("Aave_vs_Spark", res_as), ("Aave_vs_Morpho", res_am)]:
        for r in res:
            if r is None:
                continue
            rows.append(
                [
                    pair, r["label"], r["N"], round(r["R2"], 4), round(r["DW"], 3),
                    round(r["did_coef"], 4), round(r["did_se"], 4), round(r["did_p"], 5),
                    models.sigstar(r["did_p"]),
                    np.nan if "FirstDiff" in r["label"] else round(models.pct_effect(r["did_coef"]), 1),
                ]
            )
    table4 = pd.DataFrame(
        rows, columns=["Pair", "Spec", "N", "R2", "DW", "DiD_coef", "HAC_SE", "p_value", "sig", "Pct_effect"]
    )
    table4.to_csv(config.RESULTS_TABLES / "table4_did_estimates.csv", index=False)
    print("\nTable 4 — DiD estimates")
    print(table4.to_string(index=False))
    print("\nNote: Specs A/B are upper-bound level estimates (low DW); "
          "Spec D is the preferred conservative causal specification.")

    # ------------------------------------------------------------------ 6
    plac = models.placebo_tests(tvl_src, build_panel, res_as)
    plac.to_csv(config.RESULTS_TABLES / "placebo_tests.csv", index=False)
    print("\nPlacebo tests")
    print(plac.to_string(index=False))

    scan = models.rolling_did_scan(tvl_src, build_panel)
    scan.to_csv(config.RESULTS_TABLES / "rolling_event_intensity.csv", index=False)
    figures.fig3_rolling_event_intensity(scan)

    # ------------------------------------------------------------------ 7
    rows = []
    for label, df in apy.items():
        if df.empty:
            continue
        pre, post = df[df["date"] < config.EVENT_TS], df[df["date"] >= config.EVENT_TS]
        rows.append(
            [
                label, "Aave" if "Aave" in label else "Spark", len(pre),
                pre["date"].min().date() if len(pre) else None,
                round(pre["apy"].mean(), 2) if len(pre) else np.nan,
                round(pre["apy"].std(), 3) if len(pre) else np.nan,
                len(post),
                round(post["apy"].mean(), 2) if len(post) else np.nan,
                round(post["apy"].std(), 3) if len(post) else np.nan,
                round(post["apy"].max(), 2) if len(post) else np.nan,
            ]
        )
    table5 = pd.DataFrame(
        rows,
        columns=["Pool", "Protocol", "N_pre", "Pre_start", "Pre_mean_APY", "Pre_sd_APY",
                 "N_post", "Post_mean_APY", "Post_sd_APY", "Post_max_APY"],
    )
    table5.to_csv(config.RESULTS_TABLES / "table5_apy_stats.csv", index=False)
    print("\nTable 5 — APY statistics")
    print(table5.to_string(index=False))
    figures.fig4_apy_series(apy)
    figures.fig5_rolling_apy_volatility(apy)

    # ------------------------------------------------------------------ 8
    table6 = models.levene_variance_tests(apy, config.PAIR_MAP)
    table6.to_csv(config.RESULTS_TABLES / "table6_variance_tests.csv", index=False)
    print("\nTable 6 — Levene variance tests")
    print("(Overlap_days = calendar days with simultaneous pre-event APY observations in both pools)")
    print(table6.to_string(index=False))

    # ------------------------------------------------------------------ 9
    mech = models.build_mechanism_panel(panel, apy)
    mech.to_csv(config.DATA_PROCESSED / "mechanism_panel.csv", index=False)
    table7, attenuation = models.mediation_analysis(mech)
    table7.to_csv(config.RESULTS_TABLES / "table7_mechanism_regressions.csv", index=False)
    print("\nTable 7 — Mechanism regressions")
    print(table7.to_string(index=False))
    print(f"DiD attenuation via rate channel: {attenuation}%")
    figures.fig6_rate_architecture()
    figures.fig7_apy_stress_scatter(mech)

    # ------------------------------------------------------------------ 10
    rwa = pd.DataFrame(
        [
            ["US T-Bills / Short-Term Bonds", 6.8],
            ["BlackRock BUIDL", 1.2],
            ["Private Credit", 1.5],
            ["Crypto Collateral", 2.1],
            ["Other RWA", 0.8],
        ],
        columns=["Asset_class", "Value_B"],
    )
    rwa.to_csv(config.RESULTS_TABLES / "rwa_collateral_breakdown.csv", index=False)
    figures.fig8_rwa_collateral(rwa)

    # Whale-adjusted robustness (Justin Sun deposit at T+2).
    sp = tvl_src["Spark"][tvl_src["Spark"]["date"] >= config.EVENT_TS].copy()
    if len(sp):
        sp["tvl_adj"] = sp["tvlUsd"]
        t2 = config.EVENT_TS + pd.Timedelta(days=2)
        sp.loc[sp["date"] >= t2, "tvl_adj"] -= config.WHALE_DEPOSIT_USD
        i0 = (sp["date"] - config.EVENT_TS).abs().idxmin()
        i7 = (sp["date"] - (config.EVENT_TS + pd.Timedelta(days=7))).abs().idxmin()
        t0, t7r, t7a = sp.loc[i0, "tvlUsd"], sp.loc[i7, "tvlUsd"], sp.loc[i7, "tvl_adj"]
        whale = pd.DataFrame(
            [
                {
                    "T0_raw_B": round(t0 / 1e9, 3),
                    "T7_raw_B": round(t7r / 1e9, 3),
                    "T7_adj_B": round(t7a / 1e9, 3),
                    "Raw_pct": round((t7r / t0 - 1) * 100, 1),
                    "Adj_pct": round((t7a / t0 - 1) * 100, 1),
                    "Whale_share_pct": round(config.WHALE_DEPOSIT_USD / (t7r - t0) * 100, 1)
                    if (t7r - t0) > 0 else np.nan,
                }
            ]
        )
        whale.to_csv(config.RESULTS_TABLES / "whale_adjustment_summary.csv", index=False)
        print("\nWhale adjustment summary:")
        print(whale.to_string(index=False))

    print("\nDone. Tables -> results/tables/, figures -> results/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download from the DefiLlama API instead of using data/raw/ snapshots.")
    args = parser.parse_args()
    main(use_cache=not args.no_cache)

