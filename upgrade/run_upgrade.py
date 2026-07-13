"""upgrade/run_upgrade.py — Reproduce all submission-ready results.

Runs on top of the existing `src/defi_event_study` package and produces the
FINAL_* tables and u_fig_* figures cited in the v11/v12 manuscript.

Usage
-----
    python upgrade/run_upgrade.py                    # use cached snapshot
    python upgrade/run_upgrade.py --no-cache         # refresh from APIs
    python upgrade/run_upgrade.py --skip-onchain     # skip archival eth_call

The on-chain utilization step requires a working public Ethereum archive
RPC; if unreachable, pass --skip-onchain and the paper's Figure 7 must be
generated separately (see upgrade/README.md).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

from defi_event_study import config as C  # noqa: E402
from defi_event_study.data import build_panel, fetch_token_flows, load_all_sources  # noqa: E402

from upgrade_analysis import (  # noqa: E402
    AAVE_STABLECOINS,
    DONOR_SLUGS_EXTENDED,
    EXTERNAL_CTRL,
    decompose_flow_vs_price,
    event_study_detrended,
    event_study_first_diff,
    external_control_estimates,
    fetch_onchain_reserve_history,
    laspeyres_tvl,
    persistence_across_windows,
    plot_utilization,
    randomization_inference,
    reconcile_supply_borrow_apy,
    regenerate_final_tables,
    rmspe_ratio_inference,
    ssr_truncation_robustness,
    stable_tvl,
    spec_D_generic,
    synthetic_control,
    synthetic_did,
    variance_tests_upgraded,
    whale_adjustment_check,
)
from defi_event_study.data import fetch_tvl  # noqa: E402


def main(use_cache: bool = True, skip_onchain: bool = False) -> None:
    print(f"Event date: {C.EVENT_TS.date()} | cache={use_cache}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    tvl, apy = load_all_sources(use_cache=use_cache)
    tokpan = {"Aave_V3": fetch_token_flows("aave-v3", use_cache=use_cache),
              "Spark":   fetch_token_flows("spark",   use_cache=use_cache)}

    # Extended donor pool
    donor_tvl: dict[str, pd.DataFrame] = {}
    for name, slug in DONOR_SLUGS_EXTENDED.items():
        d = fetch_tvl(slug, use_cache=use_cache)
        if (len(d) > 50 and
                d["date"].max() >= C.EVENT_TS + pd.Timedelta(days=C.POST_DAYS) and
                d[d["date"] < C.EVENT_TS]["tvlUsd"].tail(30).mean() > 20e6):
            donor_tvl[name] = d
    print(f"Donor pool usable: {len(donor_tvl)}")

    # ------------------------------------------------------------------
    # FINAL_ tables (§4)
    # ------------------------------------------------------------------
    tables = regenerate_final_tables(tvl, apy)
    print("\n== FINAL Table 4 ==")
    print(tables["table4"].to_string(index=False))

    # ------------------------------------------------------------------
    # §4.4  Price–flow decomposition
    # ------------------------------------------------------------------
    print("\n== §4.4 Decomposition ==")
    decomp = decompose_flow_vs_price(tokpan["Aave_V3"])
    print(f"Aave ΔTVL(T+7)= {decomp.dV/1e9:+.2f}B | flow "
          f"{decomp.flow_comp/1e9:+.2f}B ({100*decomp.flow_comp/decomp.dV if decomp.dV else np.nan:.0f}%) "
          f"| price {decomp.price_comp/1e9:+.2f}B")

    qty_src = {n: laspeyres_tvl(tokpan[n]) for n in ["Aave_V3", "Spark"]}
    stb_src = {n: stable_tvl(tokpan[n]) for n in ["Aave_V3", "Spark"]}
    for tag, src in [("quantity_index", qty_src), ("stablecoin_only", stb_src)]:
        pnl = build_panel(src, post_days=C.POST_DAYS)
        r = spec_D_generic(pnl, "Aave_V3", "Spark")
        if r:
            print(f"  SpecD [{tag}] δ={r['did_coef']:.4f} "
                  f"(p={r['did_p']:.5f}, pct={round((np.exp(r['did_coef'])-1)*100,1)}%)")

    # ------------------------------------------------------------------
    # §4.5  Design-robust inference
    # ------------------------------------------------------------------
    print("\n== §4.5 Design-robust inference ==")
    base = spec_D_generic(build_panel(tvl), "Aave_V3", "Spark")
    plc, p1, p2 = randomization_inference(
        {**tvl, **donor_tvl}, true_did=base["did_coef"])
    plc.to_csv(C.RESULTS_TABLES / "FINAL_randomization.csv", index=False)
    print(f"  Randomization: n={len(plc)}, one-sided p={p1:.3f}, two-sided p={p2:.3f}")

    donors = [n for n in donor_tvl]
    all_panel = build_panel({**tvl, **donor_tvl}, pre_days=C.PRE_DAYS + 40, post_days=60)
    synth, gap, att = synthetic_control(all_panel, "Aave_V3", donors)
    print(f"  Synthetic control ATT ≈ {att:.4f} log pts "
          f"(≈ {(np.exp(att)-1)*100:.1f}%)")
    rr, p_sc = rmspe_ratio_inference(all_panel, "Aave_V3", donors)
    print(f"  RMSPE rank of Aave = {int(rr.index.get_loc('Aave_V3'))+1}/{len(rr)} "
          f"→ permutation p = {p_sc:.3f}")
    att_sdid = synthetic_did(all_panel, "Aave_V3", donors)
    print(f"  Synthetic DiD ATT ≈ {att_sdid:.4f} log pts "
          f"(≈ {(np.exp(att_sdid)-1)*100:.1f}%)")

    # ------------------------------------------------------------------
    # §4.6  SUTVA bounds and external controls
    # ------------------------------------------------------------------
    print("\n== §4.6 SUTVA bounds and external controls ==")
    from upgrade_analysis import sutva_attribution_bounds
    tab_sutva = sutva_attribution_bounds(tvl)
    tab_sutva.to_csv(C.RESULTS_TABLES / "FINAL_sutva_bounds.csv", index=False)
    print(tab_sutva.to_string(index=False))
    tab_ext = external_control_estimates(
        {**tvl, **{k: donor_tvl[k] for k in EXTERNAL_CTRL if k in donor_tvl}})
    tab_ext.to_csv(C.RESULTS_TABLES / "FINAL_external_controls.csv", index=False)
    print(tab_ext.to_string(index=False))

    # ------------------------------------------------------------------
    # §4.7  Persistence, whale, SSR truncation
    # ------------------------------------------------------------------
    print("\n== §4.7 Persistence, whale, SSR truncation ==")
    tab_persist = persistence_across_windows(tvl, windows=(9, 30, 60))
    tab_persist.to_csv(C.RESULTS_TABLES / "FINAL_persistence.csv", index=False)
    print(tab_persist.to_string(index=False))
    print(f"  Whale check: {whale_adjustment_check(tvl)}")
    print(f"  SSR truncation: {ssr_truncation_robustness(tvl)}")

    # ------------------------------------------------------------------
    # §3.2  Event-study (correct versions)
    # ------------------------------------------------------------------
    print("\n== §3.2 Event-study ==")
    ext_panel = build_panel(tvl, post_days=60)
    es_det = event_study_detrended(ext_panel)
    es_fd = event_study_first_diff(ext_panel)
    for name, es in [("Detrended", es_det), ("FirstDiff", es_fd)]:
        pre = es[es["week"] < 0]
        n_sig = int((pre["coef"].abs() > 1.96 * pre["se"]).sum())
        print(f"  {name}: {n_sig}/{len(pre)} pre-event weeks significant at 5%")
    es_det.to_csv(C.RESULTS_TABLES / "FINAL_eventstudy_detrended.csv", index=False)
    es_fd.to_csv(C.RESULTS_TABLES / "FINAL_eventstudy_firstdiff.csv", index=False)

    # ------------------------------------------------------------------
    # §5  Variance tests
    # ------------------------------------------------------------------
    print("\n== §5 Variance tests (Brown–Forsythe, upgraded) ==")
    pair_map = [
        ("Aave V3 USDC Ethereum", "Spark Savings USDC ETH", "USDC"),
        ("Aave V3 DAI Ethereum",  "Spark USDS Ethereum",    "DAI/USDS"),
    ]
    tab_var = variance_tests_upgraded(apy, pair_map)
    tab_var.to_csv(C.RESULTS_TABLES / "FINAL_variance_tests.csv", index=False)
    print(tab_var.to_string(index=False))

    # ------------------------------------------------------------------
    # §5.2  On-chain utilization
    # ------------------------------------------------------------------
    if skip_onchain:
        print("\n[skip-onchain] utilization step skipped")
    else:
        print("\n== §5.2 On-chain utilization (archival eth_call) ==")
        util = {}
        for label, addr in AAVE_STABLECOINS.items():
            d = fetch_onchain_reserve_history(label, addr, use_cache=use_cache)
            if not d.empty:
                util[label] = d
                print(f"  {label}: {len(d)} obs, "
                      f"U∈[{d['U'].min():.2f}, {d['U'].max():.2f}]")
        if util:
            plot_utilization(util,
                             save_to=C.RESULTS_FIGURES / "fig7_utilization.png")
            if "Aave V3 DAI Ethereum" in util:
                rec = reconcile_supply_borrow_apy(
                    util["Aave V3 DAI Ethereum"], apy["Aave V3 DAI Ethereum"])
                print(f"  reconciliation: {rec}")

    print("\nDone. FINAL_ files written to results/tables/ and results/figures/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore data/raw/ snapshots; refetch everything")
    parser.add_argument("--skip-onchain", action="store_true",
                        help="Skip archival eth_call step (public RPCs required)")
    args = parser.parse_args()
    main(use_cache=not args.no_cache, skip_onchain=args.skip_onchain)
